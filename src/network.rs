use std::{
    collections::{BTreeMap, HashMap, HashSet},
    future, io,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use tokio::{
    net::{TcpListener, TcpStream, tcp::OwnedReadHalf, tcp::OwnedWriteHalf},
    sync::{mpsc as async_mpsc, oneshot},
    task::JoinHandle,
    time::{MissedTickBehavior, interval, timeout},
};

use crate::discovery::{Discovery, DiscoveryEvent};
use crate::exchange::{Batch, EarlyMessages};
use crate::message;
use crate::tcp::{recv_message, send_message};

const HELLO_MAX_SIZE: usize = 1024;
const DATA_MAX_SIZE: usize = 64 * 1024 * 1024;
const SEND_QUEUE_SIZE: usize = 32;
const EVENT_QUEUE_SIZE: usize = 64;
const EARLY_MAX_MESSAGES: usize = 128;
const EARLY_MAX_BYTES: usize = 128 * 1024 * 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(2);
// Per-neighbor unacknowledged DATA is bounded (shared Bytes, no payload copy).
const MAX_UNACKED_MESSAGES: usize = 4;
const MAX_UNACKED_BYTES: usize = 128 * 1024 * 1024;

type Reply = mpsc::Sender<Result<(), String>>;
type ExchangeResult = Result<HashMap<String, Bytes>, String>;
type ExchangeReply = mpsc::Sender<ExchangeResult>;

enum Command {
    Connect {
        peer_id: String,
        address: String,
        reply: Reply,
    },
    Peers {
        reply: mpsc::Sender<Vec<String>>,
    },
    Exchange {
        exchange_id: u64,
        data: Bytes, // already encoded as exchange_id + payload
        reply: ExchangeReply,
    },
    CancelExchange {
        exchange_id: u64,
    },
}

struct Established {
    peer_id: String,
    stream: TcpStream,
    outgoing: bool,
    reply: Option<Reply>,
}

// A connection attempt always reports completion, so 'connecting' gets cleared
// even if TCP/HELLO fails. Incoming accepted connections have no attempt ID.
enum ConnectEvent {
    Established(Established),
    Failed {
        peer_id: String,
        reason: String,
        reply: Option<Reply>,
    },
}

// Connection setup is shared by explicit connect() and mDNS auto-connect.
fn start_connect(
    my_id: &str,
    peer_id: String,
    addresses: Vec<String>,
    events: async_mpsc::UnboundedSender<ConnectEvent>,
    reply: Option<Reply>,
) {
    let my_id = my_id.to_owned();
    tokio::spawn(async move {
        let mut last_error = "No candidate addresses".to_owned();
        let mut connected = None;

        for address in addresses {
            let attempt = timeout(CONNECT_TIMEOUT, async {
                let stream = TcpStream::connect(&address).await?;
                handshake(stream, &my_id).await
            })
            .await;

            match attempt {
                Ok(Ok((actual_id, stream))) if actual_id == peer_id => {
                    connected = Some(stream);
                    break;
                }
                Ok(Ok((actual_id, _))) => {
                    last_error = format!("Expected {peer_id}, got {actual_id} at {address}");
                }
                Ok(Err(err)) => last_error = format!("{address}: {err}"),
                Err(_) => last_error = format!("{address}: connection timed out"),
            }
        }

        let event = match connected {
            Some(stream) => ConnectEvent::Established(Established {
                peer_id,
                stream,
                outgoing: true,
                reply,
            }),
            None => ConnectEvent::Failed {
                peer_id,
                reason: last_error,
                reply,
            },
        };

        if let Err(err) = events.send(event) {
            // Ensure a blocking Python connect() does not wait unnecessarily
            // when the Tokio event loop has already shut down.
            match err.0 {
                ConnectEvent::Established(e) => {
                    if let Some(reply) = e.reply {
                        let _ = reply.send(Err("Network is stopping".into()));
                    }
                }
                ConnectEvent::Failed {
                    reply: Some(reply), ..
                } => {
                    let _ = reply.send(Err("Network is stopping".into()));
                }
                _ => {}
            }
        }
    });
}

enum PeerEvent {
    Ack {
        peer_id: String,
        session: u64,
        exchange_id: u64,
    },
    Data {
        peer_id: String,
        session: u64,
        exchange_id: u64,
        data: Bytes,
    },
    Closed {
        peer_id: String,
        session: u64,
        reason: String,
    },
}

struct PeerConnection {
    sender: async_mpsc::Sender<Bytes>,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
    session: u64,
}

struct PendingExchange {
    batch: Batch,
    reply: ExchangeReply,
}

/// Messages that this neighbor has not acknowledged, retained across TCP sessions.
/// All access happens on the network event loop, so no Mutex is needed.
#[derive(Default)]
struct UnackedMessages {
    frames: BTreeMap<u64, Bytes>,
    bytes: usize,
}

impl UnackedMessages {
    fn check_capacity(&self, exchange_id: u64, frame_size: usize) -> Result<(), String> {
        if self.frames.contains_key(&exchange_id) {
            return Err(format!("Exchange {exchange_id} is already awaiting ACK"));
        }
        if self.frames.len() >= MAX_UNACKED_MESSAGES
            || frame_size > MAX_UNACKED_BYTES.saturating_sub(self.bytes)
        {
            return Err("Unacknowledged message cache full".into());
        }
        Ok(())
    }

    // Caller must have passed check_capacity() on the same event-loop iteration.
    fn insert(&mut self, exchange_id: u64, frame: Bytes) {
        self.bytes += frame.len();
        self.frames.insert(exchange_id, frame);
    }

    fn acknowledge(&mut self, exchange_id: u64) {
        if let Some(frame) = self.frames.remove(&exchange_id) {
            self.bytes -= frame.len();
        }
    }
}

async fn handshake(mut stream: TcpStream, my_id: &str) -> io::Result<(String, TcpStream)> {
    stream.set_nodelay(true)?;
    let hello = format!("HELLO {my_id}");
    send_message(&mut stream, hello.as_bytes(), HELLO_MAX_SIZE).await?;

    let buffer = recv_message(&mut stream, HELLO_MAX_SIZE).await?;
    let text = std::str::from_utf8(&buffer)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in HELLO"))?;
    let peer_id = text
        .strip_prefix("HELLO ")
        .filter(|id| !id.is_empty() && !id.contains(char::is_whitespace))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid HELLO"))?;

    if peer_id == my_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Self connection",
        ));
    }
    Ok((peer_id.to_owned(), stream))
}

// Reader owns its half for its entire lifetime: no partially read frame is cancelled.
async fn read_peer(
    mut reader: OwnedReadHalf,
    peer_id: String,
    session: u64,
    events: async_mpsc::Sender<PeerEvent>,
) {
    loop {
        let received = async {
            let frame = recv_message(&mut reader, DATA_MAX_SIZE).await?;
            message::decode(frame)
        }
        .await;

        match received {
            Ok(message::Message::Data {
                exchange_id,
                payload,
            }) => {
                if events
                    .send(PeerEvent::Data {
                        peer_id: peer_id.clone(),
                        session,
                        exchange_id,
                        data: payload,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Ok(message::Message::Ack { exchange_id }) => {
                if events
                    .send(PeerEvent::Ack {
                        peer_id: peer_id.clone(),
                        session,
                        exchange_id,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(err) => {
                let _ = events
                    .send(PeerEvent::Closed {
                        peer_id,
                        session,
                        reason: err.to_string(),
                    })
                    .await;
                return;
            }
        }
    }
}

async fn write_peer(
    mut writer: OwnedWriteHalf,
    mut outbound: async_mpsc::Receiver<Bytes>,
    peer_id: String,
    session: u64,
    events: async_mpsc::Sender<PeerEvent>,
) {
    while let Some(data) = outbound.recv().await {
        if let Err(err) = send_message(&mut writer, &data, DATA_MAX_SIZE).await {
            let _ = events
                .send(PeerEvent::Closed {
                    peer_id,
                    session,
                    reason: err.to_string(),
                })
                .await;
            return;
        }
    }
}

fn register_peer(
    node_id: &str,
    expected: Option<&HashSet<String>>,
    peers: &mut HashMap<String, PeerConnection>,
    next_session: &mut u64,
    events: &async_mpsc::Sender<PeerEvent>,
    established: Established,
) {
    let Established {
        peer_id,
        stream,
        outgoing,
        reply,
    } = established;

    let result = if expected.is_some_and(|allowed| !allowed.contains(&peer_id)) {
        Err(format!("Peer {peer_id} is not in configured neighbors"))
    } else if outgoing != (node_id < peer_id.as_str()) {
        Err("Unexpected connection direction".to_owned())
    } else if peers.contains_key(&peer_id) {
        Err(format!("Duplicate connection: {peer_id}"))
    } else {
        *next_session += 1;
        let session = *next_session;
        let (reader, writer) = stream.into_split();
        let (sender, outbound) = async_mpsc::channel(SEND_QUEUE_SIZE);

        let reader_task = tokio::spawn(read_peer(reader, peer_id.clone(), session, events.clone()));
        let writer_task = tokio::spawn(write_peer(
            writer,
            outbound,
            peer_id.clone(),
            session,
            events.clone(),
        ));

        peers.insert(
            peer_id.clone(),
            PeerConnection {
                sender,
                reader: reader_task,
                writer: writer_task,
                session,
            },
        );
        println!("[{node_id}] Connected to {peer_id}");
        Ok(())
    };
    if let Some(reply) = reply {
        let _ = reply.send(result);
    }
}

fn fail_exchange(pending: &mut Option<PendingExchange>, reason: String) {
    if let Some(active) = pending.take() {
        let _ = active.reply.send(Err(reason));
    }
}

fn complete_exchange(pending: &mut Option<PendingExchange>) {
    if pending
        .as_ref()
        .is_some_and(|active| active.batch.is_complete())
    {
        let active = pending.take().expect("checked pending exchange");
        let results = active.batch.take().expect("checked complete batch");
        let _ = active.reply.send(Ok(results));
    }
}

fn disconnect_peer(
    node_id: &str,
    peers: &mut HashMap<String, PeerConnection>,
    peer_id: &str,
    reason: &str,
) {
    if let Some(peer) = peers.remove(peer_id) {
        peer.reader.abort();
        peer.writer.abort();
        // Keep pending Batch and early data: the connection can return.
        // The Python exchange deadline still limits how long we wait.
        println!("[{node_id}] Disconnected from {peer_id}: {reason}");
    }
}

async fn run_network(
    node_id: String,
    port: u16,
    neighbors: Option<HashSet<String>>,
    group_id: String,
    mut stop_rx: oneshot::Receiver<()>,
    mut command_rx: async_mpsc::UnboundedReceiver<Command>,
    ready_tx: mpsc::SyncSender<Result<(), String>>,
) {
    // Bind all interfaces so discovered LAN addresses can reach the listener.
    // Use only in trusted networks; this is not an authenticated service.
    let address = format!("0.0.0.0:{port}");

    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => listener,
        Err(err) => {
            let _ = ready_tx.send(Err(err.to_string()));
            return;
        }
    };

    let actual_port = match listener.local_addr() {
        Ok(addr) => addr.port(),
        Err(err) => {
            let _ = ready_tx.send(Err(err.to_string()));
            return;
        }
    };
    // None = manual mode (backwards compatible). Some(...) = fixed-topology mDNS.
    let mut discovery = if neighbors.is_some() {
        match Discovery::new(&node_id, actual_port, &group_id) {
            Ok(discovery) => Some(discovery),
            Err(err) => {
                let _ = ready_tx.send(Err(format!("Cannot start mDNS: {err}")));
                return;
            }
        }
    } else {
        None
    };

    println!("[{node_id}] Listening on 0.0.0.0:{actual_port}");
    let _ = ready_tx.send(Ok(()));

    let (established_tx, mut established_rx) = async_mpsc::unbounded_channel::<ConnectEvent>();
    // Bounded channel: read tasks apply backpressure rather than buffering endlessly.
    let (event_tx, mut event_rx) = async_mpsc::channel::<PeerEvent>(EVENT_QUEUE_SIZE);
    let mut peers: HashMap<String, PeerConnection> = HashMap::new();
    let mut next_session = 0u64;
    let mut connecting = HashSet::<String>::new();
    let mut connected_before = HashSet::<String>::new();
    // Remember known addresses for both manual connect() and mDNS discovery.
    let mut addresses = HashMap::<String, Vec<String>>::new();
    let mut retry = interval(CONNECT_RETRY_INTERVAL);
    retry.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut pending: Option<PendingExchange> = None;
    let mut early = EarlyMessages::new(EARLY_MAX_MESSAGES, EARLY_MAX_BYTES);
    let mut last_started: Option<u64> = None;
    // Retain unacknowledged DATA separately for each peer, even while disconnected.
    let mut unacked = HashMap::<String, UnackedMessages>::new();

    loop {
        tokio::select! {
            _ = &mut stop_rx => break,

            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let my_id = node_id.clone();
                        let events = established_tx.clone();
                        tokio::spawn(async move {
                            match timeout(HANDSHAKE_TIMEOUT, handshake(stream, &my_id)).await {
                                Ok(Ok((peer_id, stream))) => {
                                    let _ = events.send(ConnectEvent::Established(Established {
                                        peer_id, stream, outgoing: false, reply: None,
                                    }));
                                }
                                Ok(Err(err)) => eprintln!("[{my_id}] HELLO failed: {err}"),
                                Err(_) => eprintln!("[{my_id}] HELLO timed out"),
                            }
                        });
                    }
                    Err(err) => {
                        eprintln!("[{node_id}] Accept failed: {err}");
                        break;
                    }
                }
            }

            command = command_rx.recv() => {
                match command {
                    Some(Command::Connect { peer_id, address, reply }) => {
                        if neighbors.as_ref().is_some_and(|ids| !ids.contains(&peer_id)) {
                            let _ = reply.send(Err(format!("Peer {peer_id} is not a configured neighbor")));
                        } else if peers.contains_key(&peer_id) || !connecting.insert(peer_id.clone()) {
                            let _ = reply.send(Err(format!("Already connected or connecting to {peer_id}")));
                        } else {
                            addresses.insert(peer_id.clone(), vec![address.clone()]);
                            start_connect(&node_id, peer_id, vec![address], established_tx.clone(), Some(reply));
                        }
                    }
                    Some(Command::Peers { reply }) => {
                        let mut ids: Vec<_> = peers.keys().cloned().collect();
                        ids.sort();
                        let _ = reply.send(ids);
                    }
                    Some(Command::Exchange { exchange_id, data, reply }) => {
                        if pending.is_some() {
                            let _ = reply.send(Err("Another exchange is in progress".into()));
                            continue;
                        }
                        if last_started.is_some_and(|last| exchange_id <= last) {
                            let _ = reply.send(Err(format!(
                                "Exchange ID {exchange_id} must be greater than the last started ID"
                            )));
                            continue;
                        }

                        // With a fixed topology, never silently exchange with a subset.
                        // Manual mode still uses the currently connected peer snapshot.
                        if let Some(expected) = &neighbors {
                            if !expected.iter().all(|id| peers.contains_key(id)) {
                                let _ = reply.send(Err("Not all configured neighbors are connected".into()));
                                continue;
                            }
                        }

                        // A local exchange snapshots exactly the currently connected neighbors.
                        // IDs must increase, even if an exchange later fails or times out.
                        last_started = Some(exchange_id);
                        if exchange_id > 0 {
                            early.discard_through(exchange_id - 1);
                        }
                        let mut batch = Batch::new(exchange_id, peers.keys().cloned().collect());

                        // Reuse messages that arrived before this node entered the exchange.
                        let mut error = None;
                        for (peer_id, payload) in early.take(exchange_id) {
                            if let Err(err) = batch.insert(peer_id.clone(), payload) {
                                error = Some(format!("Cannot use cached message from {peer_id}: {err}"));
                                break;
                            }
                        }

                        // Check cache bounds and reserve one writer slot per peer before
                        // sending anything. Dropping permits on error releases all slots.
                        let mut permits = Vec::with_capacity(peers.len());
                        if error.is_none() {
                            for (peer_id, peer) in &peers {
                                if let Err(err) = unacked.entry(peer_id.clone())
                                    .or_default().check_capacity(exchange_id, data.len())
                                {
                                    error = Some(format!("Cannot retain message for {peer_id}: {err}"));
                                    break;
                                }
                                match peer.sender.try_reserve() {
                                    Ok(permit) => permits.push((peer_id.clone(), permit)),
                                    Err(err) => {
                                        error = Some(format!("Cannot queue message to {peer_id}: {err}"));
                                        break;
                                    }
                                }
                            }
                        }

                        if let Some(reason) = error {
                            let _ = reply.send(Err(reason));
                        } else {
                            // Cache before queuing: a queued write is NOT proof of delivery.
                            // Clone(Bytes) shares the allocation across all neighbors.
                            for (peer_id, permit) in permits {
                                unacked.entry(peer_id).or_default()
                                    .insert(exchange_id, data.clone());
                                permit.send(data.clone());
                            }
                            pending = Some(PendingExchange { batch, reply });
                            complete_exchange(&mut pending); // e.g. no neighbors / all early data
                        }
                    }
                    Some(Command::CancelExchange { exchange_id }) => {
                        if pending.as_ref().is_some_and(|p| p.batch.exchange_id() == exchange_id) {
                            fail_exchange(&mut pending, "Exchange cancelled after timeout".into());
                        }
                    }
                    None => break,
                }
            }

            established = established_rx.recv() => {
                match established {
                    Some(ConnectEvent::Established(established)) => {
                        let peer_id = established.peer_id.clone();
                        if established.outgoing {
                            connecting.remove(&peer_id);
                        }
                        let is_reconnect = connected_before.contains(&peer_id);
                        let old_count = peers.len();
                        register_peer(
                            &node_id, neighbors.as_ref(), &mut peers,
                            &mut next_session, &event_tx, established,
                        );
                        if peers.len() != old_count {
                            connected_before.insert(peer_id.clone());
                            if is_reconnect {
                                // Replay only DATA for which this particular peer's ACK
                                // has not arrived. BTreeMap preserves Exchange ID order.
                                let sender = &peers.get(&peer_id).expect("new peer").sender;
                                let result = unacked.get(&peer_id).map(|messages| {
                                    messages.frames.values().try_for_each(|frame| {
                                        sender.try_send(frame.clone()).map_err(|err| err.to_string())
                                    })
                                });
                                if let Some(Err(err)) = result {
                                    disconnect_peer(&node_id, &mut peers, &peer_id,
                                        &format!("Replay queue error: {err}"));
                                }
                            } else {
                                // Only an entirely new neighbor changes the topology.
                                fail_exchange(&mut pending, "Peer set changed during exchange".into());
                            }
                        }
                    }
                    Some(ConnectEvent::Failed { peer_id, reason, reply }) => {
                        connecting.remove(&peer_id);
                        if reply.is_some() && neighbors.is_none() {
                            // A failed explicit connect() is still an error to Python.
                            // Only an established manual connection becomes retryable.
                            addresses.remove(&peer_id);
                        }
                        if let Some(reply) = reply {
                            let _ = reply.send(Err(reason));
                        }
                        // For auto-connect, a later retry tick will try again.
                    }
                    None => break,
                }
            }

            // mDNS only tells us where a node is; it never changes our topology.
            discovered_event = async {
                match discovery.as_mut() {
                    Some(discovery) => discovery.next().await,
                    None => future::pending::<Option<DiscoveryEvent>>().await,
                }
            } => {
                match discovered_event {
                    Some(DiscoveryEvent::Found(peer)) => {
                        let peer_id = peer.node_id;
                        if neighbors.as_ref().is_some_and(|ids| ids.contains(&peer_id)) {
                            let candidates = peer.addresses.into_iter()
                                .map(|addr| addr.to_string())
                                .collect::<Vec<_>>();
                            addresses.insert(peer_id.clone(), candidates.clone());
                            if node_id.as_str() < peer_id.as_str()
                                && !peers.contains_key(&peer_id)
                                && connecting.insert(peer_id.clone())
                            {
                                start_connect(&node_id, peer_id, candidates, established_tx.clone(), None);
                            }
                        }
                    }
                    Some(DiscoveryEvent::Removed(peer_id)) => {
                        // mDNS removal is not proof of TCP failure.
                        // Keep any established connection, but stop using this address for retries.
                        addresses.remove(&peer_id);
                    }
                    None => {
                        eprintln!("[{node_id}] mDNS discovery stopped");
                        discovery = None; // avoid a busy loop on a closed receiver
                    }
                }
            }

            // If a service was discovered before its TCP listener was ready,
            // retry connection without waiting for another mDNS notification.
            _ = retry.tick(), if !addresses.is_empty() => {
                for (peer_id, candidates) in &addresses {
                    if node_id.as_str() < peer_id.as_str()
                        && !peers.contains_key(peer_id)
                        && connecting.insert(peer_id.clone())
                    {
                        start_connect(
                            &node_id, peer_id.clone(), candidates.clone(),
                            established_tx.clone(), None,
                        );
                    }
                }
            }

            event = event_rx.recv() => {
                match event {
                    Some(PeerEvent::Ack { peer_id, session, exchange_id }) => {
                        if peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            if let Some(messages) = unacked.get_mut(&peer_id) {
                                messages.acknowledge(exchange_id);
                            }
                        }
                    }
                    Some(PeerEvent::Data { peer_id, session, exchange_id, data }) => {
                        if !peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            continue; // event from a previous TCP session
                        }

                        let accepted = if pending.as_ref().is_some_and(|p| p.batch.exchange_id() == exchange_id) {
                            match pending.as_mut().expect("checked pending")
                                .batch.insert(peer_id.clone(), data)
                            {
                                Ok(_) => true, // a duplicate is still acknowledged
                                Err(err) => {
                                    fail_exchange(&mut pending, format!(
                                        "Invalid exchange message from {peer_id}: {err}"
                                    ));
                                    false
                                }
                            }
                        } else if last_started.is_some_and(|last| exchange_id <= last) {
                            // Already finished / cancelled locally: terminally discard.
                            // ACK so the peer can stop retrying an obsolete frame.
                            true
                        } else {
                            // Future exchange: ACK only after bounded cache accepts it.
                            match early.insert(exchange_id, peer_id.clone(), data) {
                                Ok(_) => true,
                                Err(err) => {
                                    disconnect_peer(&node_id, &mut peers, &peer_id, err);
                                    false
                                }
                            }
                        };

                        if accepted {
                            // The ACK is best-effort asynchronous: exchange() never waits for it.
                            // On queue failure, reconnect/replay rather than claim delivery.
                            if let Some(peer) = peers.get(&peer_id) {
                                if let Err(err) = peer.sender.try_send(message::encode_ack(exchange_id)) {
                                    disconnect_peer(&node_id, &mut peers, &peer_id,
                                        &format!("Cannot queue ACK: {err}"));
                                }
                            }
                            complete_exchange(&mut pending);
                        }
                    }
                    Some(PeerEvent::Closed { peer_id, session, reason }) => {
                        if peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            disconnect_peer(&node_id, &mut peers, &peer_id, &reason);
                        }
                    }
                    None => break,
                }
            }
        }
    }

    fail_exchange(&mut pending, "Network stopped".into());
    for (_, peer) in peers.drain() {
        peer.reader.abort();
        peer.writer.abort();
    }
    if let Some(discovery) = discovery {
        if let Err(err) = discovery.close() {
            eprintln!("[{node_id}] Failed to stop mDNS: {err}");
        }
    }
    println!("[{node_id}] Stopped");
}

#[pyclass]
pub struct CoreNetwork {
    node_id: String,
    port: u16,
    neighbors: Option<HashSet<String>>,
    group_id: String,
    commands: Option<async_mpsc::UnboundedSender<Command>>,
    stop_tx: Option<oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl CoreNetwork {
    #[new]
    #[pyo3(signature = (node_id, port=0, neighbors=None, group_id="default"))]
    pub fn new(
        node_id: String,
        port: u16,
        neighbors: Option<Vec<String>>,
        group_id: &str,
    ) -> PyResult<Self> {
        if node_id.is_empty() || node_id.contains(char::is_whitespace) {
            return Err(PyValueError::new_err(
                "node_id must be nonempty and contain no whitespace",
            ));
        }
        if group_id.is_empty() {
            return Err(PyValueError::new_err("group_id must not be empty"));
        }
        let neighbors = neighbors
            .map(|ids| -> PyResult<HashSet<String>> {
                let mut expected = HashSet::new();
                for id in ids {
                    if id.is_empty() || id.contains(char::is_whitespace) || id == node_id {
                        return Err(PyValueError::new_err("Invalid neighbor ID"));
                    }
                    if !expected.insert(id) {
                        return Err(PyValueError::new_err("Duplicate neighbor ID"));
                    }
                }
                Ok(expected)
            })
            .transpose()?;

        Ok(Self {
            node_id,
            port,
            neighbors,
            group_id: group_id.to_owned(),
            commands: None,
            stop_tx: None,
            worker: None,
        })
    }

    pub fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.worker.is_some() {
            return Err(PyRuntimeError::new_err("Node already started"));
        }
        let (stop_tx, stop_rx) = oneshot::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (command_tx, command_rx) = async_mpsc::unbounded_channel();

        let node_id = self.node_id.clone();
        let port = self.port;
        let neighbors = self.neighbors.clone();
        let group_id = self.group_id.clone();
        let worker = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match rt {
                Ok(rt) => rt.block_on(run_network(
                    node_id, port, neighbors, group_id, stop_rx, command_rx, ready_tx,
                )),
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                }
            }
        });

        match py.detach(move || ready_rx.recv()) {
            Ok(Ok(())) => {
                self.commands = Some(command_tx);
                self.stop_tx = Some(stop_tx);
                self.worker = Some(worker);
                Ok(())
            }
            other => {
                let _ = stop_tx.send(());
                let _ = py.detach(move || worker.join());
                Err(PyRuntimeError::new_err(format!(
                    "Failed to start node: {other:?}"
                )))
            }
        }
    }

    pub fn connect(&self, py: Python<'_>, peer_id: String, address: String) -> PyResult<()> {
        if self.node_id.as_str() >= peer_id.as_str() {
            return Err(PyValueError::new_err(
                "Only the lexicographically smaller node ID may initiate the connection",
            ));
        }
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let (reply, result_rx) = mpsc::channel();
        commands
            .send(Command::Connect {
                peer_id,
                address,
                reply,
            })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;
        let result = py
            .detach(move || result_rx.recv_timeout(Duration::from_secs(8)))
            .map_err(|err| PyRuntimeError::new_err(format!("Connect response failed: {err}")))?;
        result.map_err(PyRuntimeError::new_err)
    }

    pub fn peers(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let (reply, result_rx) = mpsc::channel();
        commands
            .send(Command::Peers { reply })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;
        py.detach(move || result_rx.recv_timeout(COMMAND_TIMEOUT))
            .map_err(|err| PyRuntimeError::new_err(format!("Peers query failed: {err}")))
    }

    /// Wait for all configured neighbors to finish HELLO; this does not change the topology.
    /// This method is intended for initialization, so a short polling loop is sufficient.
    #[pyo3(signature = (timeout=None))]
    pub fn wait_connected(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        let expected = self.neighbors.as_ref().ok_or_else(|| {
            PyValueError::new_err("wait_connected requires a configured neighbors list")
        })?;
        let deadline = match timeout {
            Some(seconds) => {
                if !seconds.is_finite() || seconds < 0.0 {
                    return Err(PyValueError::new_err("timeout must be finite and >= 0"));
                }
                let duration = Duration::try_from_secs_f64(seconds)
                    .map_err(|_| PyValueError::new_err("timeout is too large"))?;
                Some(
                    Instant::now()
                        .checked_add(duration)
                        .ok_or_else(|| PyValueError::new_err("timeout is too large"))?,
                )
            }
            None => None,
        };

        loop {
            let connected: HashSet<String> = self.peers(py)?.into_iter().collect();
            if expected.is_subset(&connected) {
                return Ok(());
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(PyTimeoutError::new_err(
                    "Not all neighbors connected before timeout",
                ));
            }
            py.detach(|| thread::sleep(Duration::from_millis(50)));
        }
    }

    /// Send one payload to all registered peers and block until they all reply
    /// with a message carrying the same exchange ID.
    #[pyo3(signature = (exchange_id, data, timeout=None))]
    pub fn exchange(
        &self,
        py: Python<'_>,
        exchange_id: u64,
        data: &Bound<'_, PyBytes>,
        timeout: Option<f64>,
    ) -> PyResult<HashMap<String, Py<PyBytes>>> {
        let deadline = match timeout {
            Some(seconds) => {
                if !seconds.is_finite() || seconds < 0.0 {
                    return Err(PyValueError::new_err("timeout must be finite and >= 0"));
                }
                Some(
                    Duration::try_from_secs_f64(seconds)
                        .map_err(|_| PyValueError::new_err("timeout is too large"))?,
                )
            }
            None => None,
        };
        if data.as_bytes().len() > DATA_MAX_SIZE - message::HEADER_SIZE {
            return Err(PyValueError::new_err("Message too large"));
        }
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let encoded = message::encode(exchange_id, data.as_bytes());
        let (reply, result_rx) = mpsc::channel();
        commands
            .send(Command::Exchange {
                exchange_id,
                data: encoded,
                reply,
            })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;

        let result = py.detach(move || match deadline {
            Some(duration) => result_rx.recv_timeout(duration),
            None => result_rx
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        });
        match result {
            Ok(Ok(messages)) => Ok(messages
                .into_iter()
                .map(|(peer_id, payload)| (peer_id, PyBytes::new(py, &payload).unbind()))
                .collect()),
            Ok(Err(err)) => Err(PyRuntimeError::new_err(err)),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Cancel the pending local batch; late messages will be treated as obsolete.
                let _ = commands.send(Command::CancelExchange { exchange_id });
                Err(PyTimeoutError::new_err(format!(
                    "Exchange {exchange_id} timed out"
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(PyRuntimeError::new_err(
                "Network worker stopped during exchange",
            )),
        }
    }

    pub fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.commands.take();
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(worker) = self.worker.take() {
            py.detach(move || worker.join())
                .map_err(|_| PyRuntimeError::new_err("Network worker panicked"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod ack_tests {
    use super::*;

    #[test]
    fn unacked_is_bounded_and_ack_releases_space() {
        let mut cache = UnackedMessages::default();
        let frame = Bytes::from_static(b"data");

        for exchange_id in 1..=MAX_UNACKED_MESSAGES as u64 {
            cache.check_capacity(exchange_id, frame.len()).unwrap();
            cache.insert(exchange_id, frame.clone());
        }
        assert!(cache.check_capacity(100, frame.len()).is_err());
        assert_eq!(cache.bytes, 4 * MAX_UNACKED_MESSAGES);

        cache.acknowledge(1);
        cache.acknowledge(1); // duplicate ACK is harmless
        assert_eq!(cache.bytes, 4 * (MAX_UNACKED_MESSAGES - 1));
        cache.check_capacity(100, frame.len()).unwrap();
        cache.insert(100, frame);
    }

    #[test]
    fn unacked_checks_total_bytes() {
        let mut cache = UnackedMessages::default();
        // Simulate a nearly-full cache without allocating a huge test buffer.
        cache.bytes = MAX_UNACKED_BYTES - 1;
        assert!(cache.check_capacity(11, 2).is_err());
        cache.check_capacity(11, 1).unwrap();
    }
}
