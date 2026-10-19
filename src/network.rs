use std::{
    collections::HashMap,
    io,
    sync::{Mutex, mpsc},
    thread,
    time::Duration,
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
    time::timeout,
};

use crate::tcp::{recv_message, send_message};

const HELLO_MAX_SIZE: usize = 1024;
const DATA_MAX_SIZE: usize = 64 * 1024 * 1024;
const SEND_QUEUE_SIZE: usize = 32;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);

type Reply = mpsc::Sender<Result<(), String>>;

enum Command {
    Connect {
        peer_id: String,
        address: String,
        reply: Reply,
    },
    Peers {
        reply: mpsc::Sender<Vec<String>>,
    },
    Send {
        peer_id: String,
        data: Bytes,
        reply: Reply,
    },
}

struct Established {
    peer_id: String,
    stream: TcpStream,
    outgoing: bool,
    reply: Option<Reply>,
}

struct IncomingData {
    peer_id: String,
    data: Bytes,
}

enum PeerEvent {
    Data {
        peer_id: String,
        session: u64,
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

async fn handshake(mut stream: TcpStream, my_id: &str) -> io::Result<(String, TcpStream)> {
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

// 读/写分别由长期运行的 Tokio Task 负责，避免读取半帧时被 select! 取消。
async fn read_peer(
    mut reader: OwnedReadHalf,
    peer_id: String,
    session: u64,
    events: async_mpsc::UnboundedSender<PeerEvent>,
) {
    loop {
        match recv_message(&mut reader, DATA_MAX_SIZE).await {
            Ok(data) => {
                if events
                    .send(PeerEvent::Data {
                        peer_id: peer_id.clone(),
                        session,
                        data,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Err(err) => {
                let _ = events.send(PeerEvent::Closed {
                    peer_id,
                    session,
                    reason: err.to_string(),
                });
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
    events: async_mpsc::UnboundedSender<PeerEvent>,
) {
    while let Some(data) = outbound.recv().await {
        if let Err(err) = send_message(&mut writer, &data, DATA_MAX_SIZE).await {
            let _ = events.send(PeerEvent::Closed {
                peer_id,
                session,
                reason: err.to_string(),
            });
            return;
        }
    }
}

fn register_peer(
    node_id: &str,
    peers: &mut HashMap<String, PeerConnection>,
    next_session: &mut u64,
    events: &async_mpsc::UnboundedSender<PeerEvent>,
    established: Established,
) {
    let Established {
        peer_id,
        stream,
        outgoing,
        reply,
    } = established;
    let result = if outgoing != (node_id < peer_id.as_str()) {
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

async fn run_network(
    node_id: String,
    port: u16,
    mut stop_rx: oneshot::Receiver<()>,
    mut command_rx: async_mpsc::UnboundedReceiver<Command>,
    ready_tx: mpsc::SyncSender<Result<(), String>>,
    incoming_tx: mpsc::Sender<IncomingData>,
) {
    let address = format!("127.0.0.1:{port}");
    let listener = match TcpListener::bind(&address).await {
        Ok(listener) => listener,
        Err(err) => {
            let _ = ready_tx.send(Err(err.to_string()));
            return;
        }
    };
    println!("[{node_id}] Listening on {address}");
    let _ = ready_tx.send(Ok(()));

    let (established_tx, mut established_rx) = async_mpsc::unbounded_channel::<Established>();
    let (event_tx, mut event_rx) = async_mpsc::unbounded_channel::<PeerEvent>();
    let mut peers: HashMap<String, PeerConnection> = HashMap::new();
    let mut next_session = 0u64;

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
                                    let _ = events.send(Established {
                                        peer_id, stream, outgoing: false, reply: None,
                                    });
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
                        if peers.contains_key(&peer_id) {
                            let _ = reply.send(Err(format!("Already connected to {peer_id}")));
                            continue;
                        }
                        let my_id = node_id.clone();
                        let events = established_tx.clone();
                        tokio::spawn(async move {
                            let result = timeout(CONNECT_TIMEOUT, async {
                                let stream = TcpStream::connect(&address).await?;
                                handshake(stream, &my_id).await
                            }).await;

                            match result {
                                Ok(Ok((actual_id, stream))) if actual_id == peer_id => {
                                    let established = Established {
                                        peer_id, stream, outgoing: true, reply: Some(reply),
                                    };
                                    if let Err(err) = events.send(established) {
                                        if let Some(reply) = err.0.reply {
                                            let _ = reply.send(Err("Node is stopping".into()));
                                        }
                                    }
                                }
                                Ok(Ok((actual_id, _))) => {
                                    let _ = reply.send(Err(format!(
                                        "Expected {peer_id}, got {actual_id}"
                                    )));
                                }
                                Ok(Err(err)) => { let _ = reply.send(Err(err.to_string())); }
                                Err(_) => { let _ = reply.send(Err("Connection timed out".into())); }
                            }
                        });
                    }
                    Some(Command::Peers { reply }) => {
                        let mut ids: Vec<_> = peers.keys().cloned().collect();
                        ids.sort();
                        let _ = reply.send(ids);
                    }
                    Some(Command::Send { peer_id, data, reply }) => {
                        let result = match peers.get(&peer_id) {
                            Some(peer) => peer.sender.try_send(data)
                                .map_err(|err| format!("Cannot queue message to {peer_id}: {err}")),
                            None => Err(format!("Peer {peer_id} is not connected")),
                        };
                        // 仅表示加入写队列，不代表对端应用已经收到。
                        let _ = reply.send(result);
                    }
                    None => break,
                }
            }

            established = established_rx.recv() => {
                if let Some(established) = established {
                    register_peer(
                        &node_id, &mut peers, &mut next_session, &event_tx, established,
                    );
                }
            }

            event = event_rx.recv() => {
                match event {
                    Some(PeerEvent::Data { peer_id, session, data }) => {
                        if peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            let _ = incoming_tx.send(IncomingData { peer_id, data });
                        }
                    }
                    Some(PeerEvent::Closed { peer_id, session, reason }) => {
                        if peers.get(&peer_id).is_some_and(|p| p.session == session) {
                            if let Some(peer) = peers.remove(&peer_id) {
                                peer.reader.abort();
                                peer.writer.abort();
                            }
                            println!("[{node_id}] Disconnected from {peer_id}: {reason}");
                        }
                    }
                    None => break,
                }
            }
        }
    }

    for (_, peer) in peers.drain() {
        peer.reader.abort();
        peer.writer.abort();
    }
    println!("[{node_id}] Stopped");
}

#[pyclass]
pub struct CoreNetwork {
    node_id: String,
    port: u16,
    commands: Option<async_mpsc::UnboundedSender<Command>>,
    incoming: Option<Mutex<mpsc::Receiver<IncomingData>>>,
    stop_tx: Option<oneshot::Sender<()>>,
    worker: Option<thread::JoinHandle<()>>,
}

#[pymethods]
impl CoreNetwork {
    #[new]
    pub fn new(node_id: String, port: u16) -> PyResult<Self> {
        if node_id.is_empty() || node_id.contains(char::is_whitespace) {
            return Err(PyValueError::new_err(
                "node_id must be nonempty and contain no whitespace",
            ));
        }
        if port == 0 {
            return Err(PyValueError::new_err("port must be between 1 and 65535"));
        }
        Ok(Self {
            node_id,
            port,
            commands: None,
            incoming: None,
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
        let (incoming_tx, incoming_rx) = mpsc::channel();

        let node_id = self.node_id.clone();
        let port = self.port;
        let worker = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();
            match rt {
                Ok(rt) => rt.block_on(run_network(
                    node_id,
                    port,
                    stop_rx,
                    command_rx,
                    ready_tx,
                    incoming_tx,
                )),
                Err(err) => {
                    let _ = ready_tx.send(Err(err.to_string()));
                }
            }
        });

        match py.detach(move || ready_rx.recv()) {
            Ok(Ok(())) => {
                self.commands = Some(command_tx);
                self.incoming = Some(Mutex::new(incoming_rx));
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

    pub fn send(&self, py: Python<'_>, peer_id: String, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        if data.as_bytes().len() > DATA_MAX_SIZE {
            return Err(PyValueError::new_err("Message too large"));
        }
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let data = Bytes::copy_from_slice(data.as_bytes());
        let (reply, result_rx) = mpsc::channel();
        commands
            .send(Command::Send {
                peer_id,
                data,
                reply,
            })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;
        let result = py
            .detach(move || result_rx.recv_timeout(COMMAND_TIMEOUT))
            .map_err(|err| PyRuntimeError::new_err(format!("Send response failed: {err}")))?;
        result.map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (timeout=5.0))]
    pub fn recv(&self, py: Python<'_>, timeout: f64) -> PyResult<(String, Py<PyBytes>)> {
        if !timeout.is_finite() || timeout < 0.0 {
            return Err(PyValueError::new_err("timeout must be finite and >= 0"));
        }
        let incoming = self
            .incoming
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let duration = Duration::from_secs_f64(timeout);
        let result = py.detach(move || {
            let rx = incoming.lock().expect("incoming queue mutex poisoned");
            rx.recv_timeout(duration)
        });
        match result {
            Ok(IncomingData { peer_id, data }) => Ok((peer_id, PyBytes::new(py, &data).unbind())),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(PyTimeoutError::new_err(
                "No message received before timeout",
            )),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(PyRuntimeError::new_err("Network receive queue closed"))
            }
        }
    }

    pub fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        self.commands.take();
        self.incoming.take();
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
