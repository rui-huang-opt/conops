use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{c_int, c_void},
    future, io, ptr,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use log::{info, warn};
use numpy::PyReadonlyArray1;
use pyo3::{
    exceptions::{PyBufferError, PyRuntimeError, PyTimeoutError, PyValueError},
    ffi,
    prelude::*,
    types::{PyBytes, PyDict, PyMemoryView, PyTuple},
};
use tokio::{
    net::{TcpListener, TcpStream, tcp::OwnedReadHalf, tcp::OwnedWriteHalf},
    sync::{mpsc as async_mpsc, oneshot},
    task::JoinHandle,
    time::{MissedTickBehavior, interval, timeout},
};

use crate::discovery::{Discovery, DiscoveryEvent};
use crate::exchange::{Batch, CompletedKeys, EarlyMessages};
use crate::message;
use crate::tcp::{recv_message, send_message};

const HELLO_MAX_SIZE: usize = 1024;
const DEFAULT_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const SEND_QUEUE_SIZE: usize = 32;
const EVENT_QUEUE_SIZE: usize = 64;
const EARLY_MAX_MESSAGES: usize = 128;
const COMPLETED_MAX_KEYS: usize = 128;
const DEFAULT_MAX_EARLY_BYTES: usize = 128 * 1024 * 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(2);
// Per-neighbor unacknowledged DATA is bounded (shared Bytes, no payload copy).
const MAX_UNACKED_MESSAGES: usize = 4;
const DEFAULT_MAX_UNACKED_BYTES: usize = 128 * 1024 * 1024;

// Experimental scheduler settings. Both knobs are read once at start(), so
// the original behavior can be reproduced without rebuilding the extension:
//   CONOPS_TOKIO_EVENT_INTERVAL=61 CONOPS_TOKIO_SPLIT_PEER_IO=1
// The optimized defaults favor prompt I/O polling and one task per peer.
const DEFAULT_TOKIO_EVENT_INTERVAL: u32 = 8;

#[derive(Clone, Copy)]
struct SchedulerConfig {
    event_interval: u32,
    fuse_peer_io: bool,
}

impl SchedulerConfig {
    fn from_env() -> Self {
        let event_interval = std::env::var("CONOPS_TOKIO_EVENT_INTERVAL")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_TOKIO_EVENT_INTERVAL);
        let fuse_peer_io = !matches!(
            std::env::var("CONOPS_TOKIO_SPLIT_PEER_IO").as_deref(),
            Ok("1")
        );
        Self {
            event_interval,
            fuse_peer_io,
        }
    }
}

/// Limits are per Network instance, not global process settings.
#[derive(Clone, Copy)]
struct NetworkConfig {
    max_message_size: usize,
    max_unacked_bytes: usize,
    max_early_bytes: usize,
}

impl NetworkConfig {
    fn validate(&self) -> PyResult<()> {
        // DATA frames have a 3-byte fixed header, at least one key byte,
        // and a 4-byte metadata length. TCP framing uses a 32-bit length.
        if self.max_message_size < message::HEADER_SIZE + 1 + 4
            || self.max_message_size > u32::MAX as usize
        {
            return Err(PyValueError::new_err(
                "max_message_size must be between 8 and 4294967295 bytes",
            ));
        }
        if self.max_unacked_bytes < self.max_message_size {
            return Err(PyValueError::new_err(
                "max_unacked_bytes must be >= max_message_size",
            ));
        }
        if self.max_early_bytes < self.max_message_size {
            return Err(PyValueError::new_err(
                "max_early_bytes must be >= max_message_size",
            ));
        }
        Ok(())
    }
}

type Reply = mpsc::Sender<Result<(), String>>;
type ExchangeResult = Result<HashMap<String, Bytes>, String>;
type ExchangeReply = mpsc::Sender<ExchangeResult>;

/// A read-only Python buffer backed by a Rust Bytes allocation.
/// Python memoryview and NumPy references keep this owner alive.
#[pyclass(frozen)]
struct ReceivedBuffer {
    data: Bytes,
}

#[pymethods]
impl ReceivedBuffer {
    // SAFETY: `data` is immutable and owned by this object.
    // The exported Py_buffer holds a strong reference to `slf`, so `data` cannot be freed until the consumer releases its view.
    // No reallocations are possible.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("Null buffer view"));
        }
        if flags & ffi::PyBUF_WRITABLE != 0 {
            return Err(PyBufferError::new_err("Buffer is read-only"));
        }

        let (data_ptr, data_len) = {
            let owner = slf.borrow();
            (owner.data.as_ptr(), owner.data.len() as isize)
        };

        unsafe {
            (*view).buf = data_ptr as *mut c_void;
            (*view).len = data_len;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if flags & ffi::PyBUF_FORMAT != 0 {
                c"B".as_ptr() as *mut _
            } else {
                ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if flags & ffi::PyBUF_ND != 0 {
                &mut (*view).len
            } else {
                ptr::null_mut()
            };
            (*view).strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                ptr::null_mut()
            };
            (*view).suboffsets = ptr::null_mut();
            (*view).internal = ptr::null_mut();
            // Move a strong Python reference into the Py_buffer.
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }
}

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
        key: Bytes,
        data: ExchangeData,
        reply: ExchangeReply,
    },
    CancelExchange {
        key: Bytes,
    },
}

// Broadcast shares one DATA buffer; neighborwise keeps one DATA buffer per peer.
enum ExchangeData {
    Broadcast(Bytes),
    Neighborwise(HashMap<String, Bytes>),
}

impl ExchangeData {
    fn for_peer(&self, peer_id: &str) -> Option<&Bytes> {
        match self {
            Self::Broadcast(frame) => Some(frame),
            Self::Neighborwise(frames) => frames.get(peer_id),
        }
    }
}

struct Established {
    peer_id: String,
    stream: TcpStream,
    outgoing: bool,
    reply: Option<Reply>,
}

// A connection attempt always reports completion, so 'connecting' gets cleared even if TCP/HELLO fails.
// Incoming accepted connections have no attempt ID.
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
            // Ensure a blocking Python connect() does not wait unnecessarily when the Tokio event loop has already shut down.
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
        key: Bytes,
    },
    Data {
        peer_id: String,
        session: u64,
        key: Bytes,
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
    tasks: PeerTasks,
    session: u64,
}

// Split is kept only as an A/B baseline. Both variants use the same protocol,
// queues and socket halves; they differ solely in Tokio task scheduling.
enum PeerTasks {
    Fused(JoinHandle<()>),
    Split {
        reader: JoinHandle<()>,
        writer: JoinHandle<()>,
    },
}

impl PeerTasks {
    fn abort(&self) {
        match self {
            Self::Fused(task) => task.abort(),
            Self::Split { reader, writer } => {
                reader.abort();
                writer.abort();
            }
        }
    }
}

struct PendingExchange {
    batch: Batch,
    reply: ExchangeReply,
}

/// Messages not yet ACKed by this neighbor. Preserve the send order on replay.
/// Only a few frames are retained; a VecDeque keeps this simpler than two maps.
#[derive(Default)]
struct UnackedMessages {
    frames: VecDeque<(Bytes, Bytes)>,
    bytes: usize,
}

impl UnackedMessages {
    fn check_capacity(
        &self,
        key: &Bytes,
        frame_size: usize,
        max_unacked_bytes: usize,
    ) -> Result<(), String> {
        if self.frames.iter().any(|(stored, _)| stored == key) {
            return Err("Exchange key is already awaiting ACK".into());
        }
        if self.frames.len() >= MAX_UNACKED_MESSAGES
            || frame_size > max_unacked_bytes.saturating_sub(self.bytes)
        {
            return Err("Unacknowledged message cache full".into());
        }
        Ok(())
    }

    // Caller must have passed check_capacity() on the same event-loop iteration.
    fn insert(&mut self, key: Bytes, frame: Bytes) {
        self.bytes += frame.len();
        self.frames.push_back((key, frame));
    }

    fn acknowledge(&mut self, key: &Bytes) {
        if let Some(index) = self.frames.iter().position(|(stored, _)| stored == key) {
            let (_, frame) = self.frames.remove(index).expect("checked index");
            self.bytes -= frame.len();
        }
    }
}

async fn handshake(mut stream: TcpStream, my_id: &str) -> io::Result<(String, TcpStream)> {
    stream.set_nodelay(true)?;
    // The DATA/ACK frame layout changed: reject peers speaking the old protocol.
    let hello = format!("HELLO2 {my_id}");
    send_message(&mut stream, hello.as_bytes(), HELLO_MAX_SIZE).await?;

    let buffer = recv_message(&mut stream, HELLO_MAX_SIZE).await?;
    let text = std::str::from_utf8(&buffer)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in HELLO"))?;
    let peer_id = text
        .strip_prefix("HELLO2 ")
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
    max_message_size: usize,
    events: async_mpsc::Sender<PeerEvent>,
) {
    loop {
        let received = async {
            let frame = recv_message(&mut reader, max_message_size).await?;
            message::decode(frame)
        }
        .await;

        match received {
            Ok(message::Message::Data { key, payload }) => {
                if events
                    .send(PeerEvent::Data {
                        peer_id: peer_id.clone(),
                        session,
                        key,
                        data: payload,
                    })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Ok(message::Message::Ack { key }) => {
                if events
                    .send(PeerEvent::Ack {
                        peer_id: peer_id.clone(),
                        session,
                        key,
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
    max_message_size: usize,
    events: async_mpsc::Sender<PeerEvent>,
) {
    while let Some(data) = outbound.recv().await {
        if let Err(err) = send_message(&mut writer, &data, max_message_size).await {
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

// The same pinned read future is polled for the entire connection lifetime:
// a select! iteration NEVER drops a partially read TCP frame. Only a terminal
// read/write error, shutdown, or disconnect cancels either direction.
async fn run_peer_io(
    reader: OwnedReadHalf,
    writer: OwnedWriteHalf,
    outbound: async_mpsc::Receiver<Bytes>,
    peer_id: String,
    session: u64,
    max_message_size: usize,
    events: async_mpsc::Sender<PeerEvent>,
) {
    let read = read_peer(
        reader,
        peer_id.clone(),
        session,
        max_message_size,
        events.clone(),
    );
    let write = write_peer(writer, outbound, peer_id, session, max_message_size, events);
    tokio::pin!(read, write);
    tokio::select! {
        _ = &mut read => {},
        _ = &mut write => {},
    }
}

fn register_peer(
    node_id: &str,
    expected: Option<&HashSet<String>>,
    peers: &mut HashMap<String, PeerConnection>,
    next_session: &mut u64,
    max_message_size: usize,
    events: &async_mpsc::Sender<PeerEvent>,
    fuse_peer_io: bool,
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

        let tasks = if fuse_peer_io {
            // One Tokio task is woken for either socket readiness or outbound
            // queue data instead of scheduling separate read/write tasks.
            PeerTasks::Fused(tokio::spawn(run_peer_io(
                reader,
                writer,
                outbound,
                peer_id.clone(),
                session,
                max_message_size,
                events.clone(),
            )))
        } else {
            // Baseline: exactly the original independent read/write tasks.
            let reader = tokio::spawn(read_peer(
                reader,
                peer_id.clone(),
                session,
                max_message_size,
                events.clone(),
            ));
            let writer = tokio::spawn(write_peer(
                writer,
                outbound,
                peer_id.clone(),
                session,
                max_message_size,
                events.clone(),
            ));
            PeerTasks::Split { reader, writer }
        };

        peers.insert(
            peer_id.clone(),
            PeerConnection {
                sender,
                tasks,
                session,
            },
        );
        info!("[{node_id}] Connected to {peer_id}");
        Ok(())
    };
    if let Some(reply) = reply {
        let _ = reply.send(result);
    }
}

fn fail_exchange(
    pending: &mut Option<PendingExchange>,
    completed: &mut CompletedKeys,
    reason: String,
) {
    if let Some(active) = pending.take() {
        completed.insert(active.batch.key().clone());
        let _ = active.reply.send(Err(reason));
    }
}

fn complete_exchange(pending: &mut Option<PendingExchange>, completed: &mut CompletedKeys) {
    if pending
        .as_ref()
        .is_some_and(|active| active.batch.is_complete())
    {
        let active = pending.take().expect("checked pending exchange");
        completed.insert(active.batch.key().clone());
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
        peer.tasks.abort();
        // Keep pending Batch and early data: the connection can return.
        // The Python exchange deadline still limits how long we wait.
        info!("[{node_id}] Disconnected from {peer_id}: {reason}");
    }
}

async fn run_network(
    node_id: String,
    port: u16,
    neighbors: Option<HashSet<String>>,
    group_id: String,
    peer_endpoints: Option<HashMap<String, String>>,
    config: NetworkConfig,
    fuse_peer_io: bool,
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
    // Use mDNS only for fixed-topology networks without static endpoints.
    // Explicit peer endpoints keep the fixed topology but skip multicast discovery.
    let mut discovery = if neighbors.is_some() && peer_endpoints.is_none() {
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

    info!("[{node_id}] Listening on 0.0.0.0:{actual_port}");
    let _ = ready_tx.send(Ok(()));

    let (established_tx, mut established_rx) = async_mpsc::unbounded_channel::<ConnectEvent>();
    // Bounded channel: read tasks apply backpressure rather than buffering endlessly.
    let (event_tx, mut event_rx) = async_mpsc::channel::<PeerEvent>(EVENT_QUEUE_SIZE);
    let mut peers: HashMap<String, PeerConnection> = HashMap::new();
    let mut next_session = 0u64;
    let mut connecting = HashSet::<String>::new();
    let mut connected_before = HashSet::<String>::new();
    // Static endpoints are known from startup, so the existing retry and replay paths work without mDNS.
    // Only the smaller node ID initiates connections.
    let mut addresses: HashMap<String, Vec<String>> = peer_endpoints
        .unwrap_or_default()
        .into_iter()
        .map(|(peer_id, address)| (peer_id, vec![address]))
        .collect();
    let mut retry = interval(CONNECT_RETRY_INTERVAL);
    retry.set_missed_tick_behavior(MissedTickBehavior::Skip);

    let mut pending: Option<PendingExchange> = None;
    let mut early = EarlyMessages::new(EARLY_MAX_MESSAGES, config.max_early_bytes);
    let mut completed = CompletedKeys::new(COMPLETED_MAX_KEYS);
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
                                Ok(Err(err)) => warn!("[{my_id}] HELLO failed: {err}"),
                                Err(_) => warn!("[{my_id}] HELLO timed out"),
                            }
                        });
                    }
                    Err(err) => {
                        warn!("[{node_id}] Accept failed: {err}");
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
                    Some(Command::Exchange { key, data, reply }) => {
                        if pending.is_some() {
                            let _ = reply.send(Err("Another exchange is in progress".into()));
                            continue;
                        }
                        if completed.contains(&key) {
                            let _ = reply.send(Err("Exchange key has already been used".into()));
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

                        // A neighborwise call must supply exactly one DATA frame per connected peer.
                        if let ExchangeData::Neighborwise(frames) = &data {
                            if frames.len() != peers.len()
                                || frames.keys().any(|peer_id| !peers.contains_key(peer_id))
                            {
                                let _ = reply.send(Err("Neighborwise states must match connected peers".into()));
                                continue;
                            }
                        }

                        let mut batch = Batch::new(key.clone(), peers.keys().cloned().collect());

                        // Reuse messages that arrived before this node entered the exchange.
                        let mut error = None;
                        for (peer_id, payload) in early.take(&key) {
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
                                let frame = data.for_peer(peer_id).expect("checked neighborwise peers");
                                if let Err(err) = unacked.entry(peer_id.clone())
                                    .or_default().check_capacity(
                                        &key, frame.len(), config.max_unacked_bytes,
                                    )
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
                            // Broadcast clones share one allocation; neighborwise frames differ.
                            for (peer_id, permit) in permits {
                                let frame = data.for_peer(&peer_id).expect("checked neighborwise peers");
                                unacked.entry(peer_id).or_default()
                                    .insert(key.clone(), frame.clone());
                                permit.send(frame.clone());
                            }
                            pending = Some(PendingExchange { batch, reply });
                            complete_exchange(&mut pending, &mut completed);
                        }
                    }
                    Some(Command::CancelExchange { key }) => {
                        if pending.as_ref().is_some_and(|p| p.batch.key() == &key) {
                            fail_exchange(&mut pending, &mut completed,
                                "Exchange cancelled after timeout".into());
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
                            &mut next_session, config.max_message_size, &event_tx,
                            fuse_peer_io, established,
                        );
                        if peers.len() != old_count {
                            connected_before.insert(peer_id.clone());
                            if is_reconnect {
                                // Replay only DATA for which this particular peer's ACK
                                // has not arrived. Replay follows the original send order.
                                let sender = &peers.get(&peer_id).expect("new peer").sender;
                                let result = unacked.get(&peer_id).map(|messages| {
                                    messages.frames.iter().try_for_each(|(_, frame)| {
                                        sender.try_send(frame.clone()).map_err(|err| err.to_string())
                                    })
                                });
                                if let Some(Err(err)) = result {
                                    disconnect_peer(&node_id, &mut peers, &peer_id,
                                        &format!("Replay queue error: {err}"));
                                }
                            } else {
                                // Only an entirely new neighbor changes the topology.
                                fail_exchange(&mut pending, &mut completed, "Peer set changed during exchange".into());
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
                        warn!("[{node_id}] mDNS discovery stopped");
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
                    Some(PeerEvent::Ack { peer_id, session, key }) => {
                        if peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            if let Some(messages) = unacked.get_mut(&peer_id) {
                                messages.acknowledge(&key);
                            }
                        }
                    }
                    Some(PeerEvent::Data { peer_id, session, key, data }) => {
                        if !peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            continue; // event from a previous TCP session
                        }

                        let accepted = if pending.as_ref().is_some_and(|p| p.batch.key() == &key) {
                            match pending.as_mut().expect("checked pending")
                                .batch.insert(peer_id.clone(), data)
                            {
                                Ok(_) => true, // a duplicate is still acknowledged
                                Err(err) => {
                                    fail_exchange(&mut pending, &mut completed, format!(
                                        "Invalid exchange message from {peer_id}: {err}"
                                    ));
                                    false
                                }
                            }
                        } else if completed.contains(&key) {
                            // Already completed / cancelled: discard and ACK the duplicate.
                            true
                        } else {
                            // Not yet started locally: ACK only after bounded cache accepts it.
                            match early.insert(key.clone(), peer_id.clone(), data) {
                                Ok(_) => true,
                                Err(err) => {
                                    disconnect_peer(&node_id, &mut peers, &peer_id, err);
                                    false
                                }
                            }
                        };

                        if accepted {
                            // ACK is asynchronous: exchange() does not wait for it.
                            if let Some(peer) = peers.get(&peer_id) {
                                if let Err(err) = peer.sender.try_send(message::encode_ack(&key)) {
                                    disconnect_peer(&node_id, &mut peers, &peer_id,
                                        &format!("Cannot queue ACK: {err}"));
                                }
                            }
                            complete_exchange(&mut pending, &mut completed);
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

    fail_exchange(&mut pending, &mut completed, "Network stopped".into());
    for (_, peer) in peers.drain() {
        peer.tasks.abort();
    }
    if let Some(discovery) = discovery {
        if let Err(err) = discovery.close() {
            warn!("[{node_id}] Failed to stop mDNS: {err}");
        }
    }
    info!("[{node_id}] Stopped");
}

#[pyclass]
pub struct CoreNetwork {
    node_id: String,
    port: u16,
    neighbors: Option<HashSet<String>>,
    group_id: String,
    peer_endpoints: Option<HashMap<String, String>>,
    config: NetworkConfig,
    commands: Option<async_mpsc::UnboundedSender<Command>>,
    stop_tx: Option<oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl CoreNetwork {
    /// Shared blocking exchange path; the frame is already encoded and owned by Rust.
    fn exchange_encoded(
        &self,
        py: Python<'_>,
        key: Bytes,
        data: ExchangeData,
        timeout: Option<f64>,
    ) -> PyResult<HashMap<String, Py<PyMemoryView>>> {
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
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let (reply, mut result_rx) = mpsc::channel();
        commands
            .send(Command::Exchange {
                key: key.clone(),
                data,
                reply,
            })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;

        // Periodically check Python signals so Ctrl+C can interrupt exchange().
        let end = deadline.map(|duration| Instant::now() + duration);

        let result = loop {
            let wait = match end {
                Some(end) => end
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(100)),
                None => Duration::from_millis(100),
            };

            let rx = &mut result_rx;

            match py.detach(move || rx.recv_timeout(wait)) {
                Ok(value) => break Ok(value),

                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    break Err(mpsc::RecvTimeoutError::Disconnected);
                }

                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Process Ctrl+C and other Python signals.
                    if let Err(err) = py.check_signals() {
                        let _ = commands.send(Command::CancelExchange { key: key.clone() });
                        return Err(err);
                    }

                    if end.is_some_and(|end| Instant::now() >= end) {
                        break Err(mpsc::RecvTimeoutError::Timeout);
                    }
                }
            }
        };
        match result {
            Ok(Ok(messages)) => {
                let mut received = HashMap::with_capacity(messages.len());
                for (peer_id, payload) in messages {
                    let owner = Py::new(py, ReceivedBuffer { data: payload })?;
                    let view = PyMemoryView::from(owner.bind(py).as_any())?;
                    received.insert(peer_id, view.unbind());
                }
                Ok(received)
            }
            Ok(Err(err)) => Err(PyRuntimeError::new_err(err)),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Cancel the pending local batch; late messages will be treated as obsolete.
                let _ = commands.send(Command::CancelExchange { key: key.clone() });
                Err(PyTimeoutError::new_err("Exchange timed out"))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(PyRuntimeError::new_err(
                "Network worker stopped during exchange",
            )),
        }
    }
}

#[pymethods]
impl CoreNetwork {
    #[new]
    #[pyo3(signature = (
        node_id,
        port=0,
        neighbors=None,
        group_id="default",
        peer_endpoints=None,
        max_message_size=DEFAULT_MAX_MESSAGE_SIZE,
        max_unacked_bytes=DEFAULT_MAX_UNACKED_BYTES,
        max_early_bytes=DEFAULT_MAX_EARLY_BYTES,
    ))]
    pub fn new(
        node_id: String,
        port: u16,
        neighbors: Option<Vec<String>>,
        group_id: &str,
        peer_endpoints: Option<HashMap<String, String>>,
        max_message_size: usize,
        max_unacked_bytes: usize,
        max_early_bytes: usize,
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

        if let Some(ref endpoints) = peer_endpoints {
            let expected = neighbors.as_ref().ok_or_else(|| {
                PyValueError::new_err("peer_endpoints requires configured neighbors")
            })?;
            if endpoints.len() != expected.len()
                || endpoints.keys().any(|id| !expected.contains(id))
                || endpoints.values().any(String::is_empty)
            {
                return Err(PyValueError::new_err(
                    "peer_endpoints must contain exactly the configured neighbors with nonempty addresses",
                ));
            }
        }

        let config = NetworkConfig {
            max_message_size,
            max_unacked_bytes,
            max_early_bytes,
        };
        config.validate()?;

        Ok(Self {
            node_id,
            port,
            neighbors,
            group_id: group_id.to_owned(),
            peer_endpoints,
            config,
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
        let peer_endpoints = self.peer_endpoints.clone();
        let config = self.config;
        let scheduling = SchedulerConfig::from_env();
        let worker = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .event_interval(scheduling.event_interval)
                .enable_all()
                .build();
            match rt {
                Ok(rt) => rt.block_on(run_network(
                    node_id,
                    port,
                    neighbors,
                    group_id,
                    peer_endpoints,
                    config,
                    scheduling.fuse_peer_io,
                    stop_rx,
                    command_rx,
                    ready_tx,
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
            py.check_signals()?;

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

    /// Encode metadata and a contiguous uint8 NumPy view in one Rust allocation.
    /// Python no longer constructs a large intermediate bytes object.
    #[pyo3(signature = (key, meta, payload, timeout=None))]
    pub fn exchange_array(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyBytes>,
        meta: &Bound<'_, PyBytes>,
        payload: PyReadonlyArray1<'_, u8>,
        timeout: Option<f64>,
    ) -> PyResult<HashMap<String, Py<PyMemoryView>>> {
        let key = key.as_bytes();
        let meta = meta.as_bytes();
        let payload = payload.as_slice().map_err(|_| {
            PyValueError::new_err("payload must be a contiguous 1-D uint8 NumPy array")
        })?;
        let size = message::HEADER_SIZE
            .checked_add(key.len())
            .and_then(|size| size.checked_add(4))
            .and_then(|size| size.checked_add(meta.len()))
            .and_then(|size| size.checked_add(payload.len()))
            .ok_or_else(|| PyValueError::new_err("Message too large"))?;
        if size > self.config.max_message_size {
            return Err(PyValueError::new_err("Message too large"));
        }
        let encoded = message::encode_state(key, meta, payload)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.exchange_encoded(
            py,
            Bytes::copy_from_slice(key),
            ExchangeData::Broadcast(encoded),
            timeout,
        )
    }

    /// Send a different NumPy state to each peer under the same exchange key.
    #[pyo3(signature = (key, states, timeout=None))]
    pub fn neighborwise_exchange_array(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyBytes>,
        states: &Bound<'_, PyDict>,
        timeout: Option<f64>,
    ) -> PyResult<HashMap<String, Py<PyMemoryView>>> {
        let key = key.as_bytes();
        if let Some(expected) = &self.neighbors {
            if states.len() != expected.len() {
                return Err(PyValueError::new_err(
                    "Neighborwise states must match configured neighbors",
                ));
            }
        }

        let mut frames = HashMap::with_capacity(states.len());
        for (peer_id, state) in states.iter() {
            let peer_id: String = peer_id.extract()?;
            if self
                .neighbors
                .as_ref()
                .is_some_and(|ids| !ids.contains(&peer_id))
            {
                return Err(PyValueError::new_err(format!(
                    "Unexpected neighbor: {peer_id}"
                )));
            }

            let pair = state.cast::<PyTuple>().map_err(|_| {
                PyValueError::new_err("Neighborwise values must be (metadata, payload) pairs")
            })?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "Neighborwise values must contain two items",
                ));
            }
            let meta_obj = pair.get_item(0)?;
            let meta = meta_obj.cast::<PyBytes>()?;
            let payload_obj = pair.get_item(1)?;
            let payload = payload_obj.extract::<PyReadonlyArray1<'_, u8>>()?;
            let payload = payload.as_slice().map_err(|_| {
                PyValueError::new_err("payload must be a contiguous 1-D uint8 NumPy array")
            })?;

            let size = message::HEADER_SIZE
                .checked_add(key.len())
                .and_then(|size| size.checked_add(4))
                .and_then(|size| size.checked_add(meta.as_bytes().len()))
                .and_then(|size| size.checked_add(payload.len()))
                .ok_or_else(|| PyValueError::new_err("Message too large"))?;
            if size > self.config.max_message_size {
                return Err(PyValueError::new_err("Message too large"));
            }

            let frame = message::encode_state(key, meta.as_bytes(), payload)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            frames.insert(peer_id, frame);
        }

        self.exchange_encoded(
            py,
            Bytes::copy_from_slice(key),
            ExchangeData::Neighborwise(frames),
            timeout,
        )
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
            let key = Bytes::copy_from_slice(&exchange_id.to_be_bytes());
            cache
                .check_capacity(&key, frame.len(), DEFAULT_MAX_UNACKED_BYTES)
                .unwrap();
            cache.insert(key, frame.clone());
        }
        assert!(
            cache
                .check_capacity(
                    &Bytes::from_static(b"100"),
                    frame.len(),
                    DEFAULT_MAX_UNACKED_BYTES
                )
                .is_err()
        );
        assert_eq!(cache.bytes, 4 * MAX_UNACKED_MESSAGES);

        cache.acknowledge(&Bytes::copy_from_slice(&1u64.to_be_bytes()));
        cache.acknowledge(&Bytes::copy_from_slice(&1u64.to_be_bytes())); // duplicate ACK is harmless
        assert_eq!(cache.bytes, 4 * (MAX_UNACKED_MESSAGES - 1));
        cache
            .check_capacity(
                &Bytes::from_static(b"100"),
                frame.len(),
                DEFAULT_MAX_UNACKED_BYTES,
            )
            .unwrap();
        cache.insert(Bytes::from_static(b"100"), frame);
    }

    #[test]
    fn unacked_replays_in_send_order() {
        let mut cache = UnackedMessages::default();
        cache.insert(Bytes::from_static(b"z"), Bytes::from_static(b"first"));
        cache.insert(Bytes::from_static(b"a"), Bytes::from_static(b"second"));
        assert_eq!(cache.frames[0].0.as_ref(), b"z");
        assert_eq!(cache.frames[1].0.as_ref(), b"a");
    }

    #[test]
    fn unacked_checks_total_bytes() {
        let mut cache = UnackedMessages::default();
        // Simulate a nearly-full cache without allocating a huge test buffer.
        cache.bytes = DEFAULT_MAX_UNACKED_BYTES - 1;
        assert!(
            cache
                .check_capacity(&Bytes::from_static(b"11"), 2, DEFAULT_MAX_UNACKED_BYTES)
                .is_err()
        );
        cache
            .check_capacity(&Bytes::from_static(b"11"), 1, DEFAULT_MAX_UNACKED_BYTES)
            .unwrap();
    }

    #[test]
    fn unacked_uses_instance_budget() {
        let mut cache = UnackedMessages::default();
        cache.insert(Bytes::from_static(b"old"), Bytes::from_static(b"abcd"));
        assert!(
            cache
                .check_capacity(&Bytes::from_static(b"new"), 3, 6)
                .is_err()
        );
        cache
            .check_capacity(&Bytes::from_static(b"new"), 3, 7)
            .unwrap();
    }

    #[test]
    fn configurable_limits_are_validated() {
        let good = NetworkConfig {
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            max_unacked_bytes: DEFAULT_MAX_UNACKED_BYTES,
            max_early_bytes: DEFAULT_MAX_EARLY_BYTES,
        };
        assert!(good.validate().is_ok());
        assert!(
            NetworkConfig {
                max_early_bytes: 1,
                ..good
            }
            .validate()
            .is_err()
        );
        assert!(
            NetworkConfig {
                max_unacked_bytes: 1,
                ..good
            }
            .validate()
            .is_err()
        );
        assert!(
            NetworkConfig {
                max_message_size: 7,
                ..good
            }
            .validate()
            .is_err()
        );
        #[cfg(target_pointer_width = "64")]
        assert!(
            NetworkConfig {
                max_message_size: u32::MAX as usize + 1,
                ..good
            }
            .validate()
            .is_err()
        );
    }
}

#[cfg(test)]
mod neighborwise_tests {
    use super::*;

    #[test]
    fn exchange_data_selects_per_peer_frames() {
        let broadcast = ExchangeData::Broadcast(Bytes::from_static(b"same"));
        assert_eq!(broadcast.for_peer("A").unwrap().as_ref(), b"same");
        assert_eq!(broadcast.for_peer("B").unwrap().as_ref(), b"same");

        let mut frames = HashMap::new();
        frames.insert("A".into(), Bytes::from_static(b"to A"));
        frames.insert("B".into(), Bytes::from_static(b"to B"));
        let neighborwise = ExchangeData::Neighborwise(frames);
        assert_eq!(neighborwise.for_peer("A").unwrap().as_ref(), b"to A");
        assert_eq!(neighborwise.for_peer("B").unwrap().as_ref(), b"to B");
        assert!(neighborwise.for_peer("C").is_none());
    }
}
