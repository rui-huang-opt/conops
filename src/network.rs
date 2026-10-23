use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{c_int, c_void},
    future, io, ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
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
    sync::{Notify, mpsc as async_mpsc, oneshot},
    task::JoinHandle,
    time::{MissedTickBehavior, interval, timeout},
};

use crate::discovery::{Discovery, DiscoveryEvent};
use crate::exchange::{Batch, CompletedKeys, EarlyMessages, PeerIndex, PeerLayout};
use crate::message;
use crate::tcp::{FrameReader, recv_message, send_message, send_message_parts};

const HELLO_MAX_SIZE: usize = 1024;
const DEFAULT_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
const SEND_QUEUE_SIZE: usize = 32;
const EVENT_QUEUE_SIZE: usize = 64;
// Drain only already-ready events, without delaying the first message.
const EVENT_BATCH_LIMIT: usize = 16;
// Defensive bound for deferred ACK keys; normal sequential peers need far fewer.
const MAX_DEFERRED_ACKS: usize = 128;
const EARLY_MAX_MESSAGES: usize = 128;
const COMPLETED_MAX_KEYS: usize = 128;
const DEFAULT_MAX_EARLY_BYTES: usize = 128 * 1024 * 1024;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const COMMAND_TIMEOUT: Duration = Duration::from_secs(2);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(2);
// Per-neighbor unacknowledged DATA is bounded (shared frames, no replay copy).
const MAX_UNACKED_MESSAGES: usize = 4;
const DEFAULT_MAX_UNACKED_BYTES: usize = 128 * 1024 * 1024;

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
    Drain {
        deadline: Instant,
        reply: Reply,
    },
}

// Owned frames keep the existing copy=True / broadcast representation.
// Borrowed frames keep their actual Python exporter alive through a private
// memoryview. Clones share one owner through Arc, including during replay.
#[derive(Clone)]
enum DataFrame {
    Owned(Bytes),
    Borrowed(Arc<BorrowedFrame>),
}

impl DataFrame {
    fn len(&self) -> usize {
        match self {
            Self::Owned(frame) => frame.len(),
            // Construction checked this sum against max_message_size / u32.
            Self::Borrowed(frame) => frame.head.len() + frame.payload_len,
        }
    }
}

impl From<Bytes> for DataFrame {
    fn from(frame: Bytes) -> Self {
        Self::Owned(frame)
    }
}

/// A small owned protocol head plus an exported, immutable-by-contract payload.
///
/// The private memoryview owns a buffer export of the *prepared* NumPy uint8
/// array, not merely the user's original state. It keeps temporary contiguous
/// arrays and their base owners alive. It is never exposed to Python callers,
/// so callers cannot release this memoryview while a writer still uses it.
///
/// This is an opt-in borrowing contract, NOT a lock: copy=False callers must
/// not mutate, resize, or invalidate the exported storage (including through
/// aliases / reused Transform buffers), even after exchange return or timeout.
/// In this first API there is no send-completion handle: use fresh arrays for
/// subsequent updates, or keep copy=True when reusing writable storage.
struct BorrowedFrame {
    head: Bytes,
    _owner: Py<PyMemoryView>,
    payload_ptr: *const u8,
    payload_len: usize,
}

// SAFETY: new() requires the exporter to remain immutable and at a stable
// address until the last Arc is dropped. _owner keeps the export alive, and
// worker threads only read its bytes; they never call Python APIs through it.
// Py<T> handles deferred decref when dropped on a detached worker thread.
// Python references alone do NOT provide immutability: this relies on the
// explicitly documented copy=False caller contract (as non-copy I/O does).
unsafe impl Send for BorrowedFrame {}
// SAFETY: the same ownership / immutability invariant permits concurrent reads.
unsafe impl Sync for BorrowedFrame {}

impl BorrowedFrame {
    /// # Safety
    /// `payload` must be the contiguous byte storage exported by `object`.
    /// Its contents and address must not change, even via an alias, until all
    /// queue, writer, and unacknowledged/replay references have been released.
    unsafe fn new(head: Bytes, object: &Bound<'_, PyAny>, payload: &[u8]) -> PyResult<Self> {
        let owner = PyMemoryView::from(object)?.unbind();
        Ok(Self {
            head,
            _owner: owner,
            payload_ptr: payload.as_ptr(),
            payload_len: payload.len(),
        })
    }

    fn payload(&self) -> &[u8] {
        if self.payload_len == 0 {
            return &[];
        }
        // SAFETY: new() established the buffer/export invariants above. The
        // returned slice is tied to &self; the writer holds its Arc across the
        // entire async send, including partial writes and backpressure waits.
        unsafe { std::slice::from_raw_parts(self.payload_ptr, self.payload_len) }
    }
}

// Both broadcast and neighborwise can borrow their prepared payloads;
// the same frame representation and wire protocol are used for both.
enum ExchangeData {
    Broadcast(DataFrame),
    Neighborwise(HashMap<String, DataFrame>),
}

impl ExchangeData {
    fn for_peer(&self, peer_id: &str) -> Option<&DataFrame> {
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
        peer: PeerIndex,
        session: u64,
        key: Bytes,
    },
    Data {
        peer: PeerIndex,
        session: u64,
        key: Bytes,
        data: Bytes,
    },
    Closed {
        peer: PeerIndex,
        session: u64,
        reason: String,
    },
    DrainFlushed {
        peer: PeerIndex,
        session: u64,
        id: u64,
    },
}

// DATA retains an owned snapshot or a borrowed exporter; ACKs are unchanged.
enum Outbound {
    Data(DataFrame),
    Ack(Bytes),
    // Local queue marker: never sent over TCP.
    DrainFlush(u64),
}

#[derive(Default)]
struct AckBacklog {
    deferred: VecDeque<Bytes>,
    capacity_waiting: Arc<AtomicBool>,
}

impl AckBacklog {
    fn queue(&mut self, sender: &async_mpsc::Sender<Outbound>, key: Bytes) -> Result<(), String> {
        // Coalesce duplicates only in the unsent ACK backlog, not DATA messages.
        if !self.deferred.iter().any(|old| old == &key) {
            if self.deferred.len() >= MAX_DEFERRED_ACKS {
                return Err("Deferred ACK key budget exceeded".into());
            }
            self.deferred.push_back(key);
        }
        self.flush(sender)
    }

    fn flush(&mut self, sender: &async_mpsc::Sender<Outbound>) -> Result<(), String> {
        while let Some(key) = self.deferred.pop_front() {
            match sender.try_send(Outbound::Ack(key)) {
                Ok(()) => {}
                Err(async_mpsc::error::TrySendError::Closed(_)) => {
                    return Err("Peer writer has stopped".into());
                }
                Err(async_mpsc::error::TrySendError::Full(frame)) => {
                    // Register the need for a wakeup, then retry once. This also
                    // closes the pop-before-registration race with a writer.
                    self.capacity_waiting.store(true, Ordering::Release);
                    match sender.try_send(frame) {
                        Ok(()) => {}
                        Err(async_mpsc::error::TrySendError::Closed(_)) => {
                            return Err("Peer writer has stopped".into());
                        }
                        Err(async_mpsc::error::TrySendError::Full(Outbound::Ack(key))) => {
                            self.deferred.push_front(key);
                            return Ok(()); // backpressure, NOT a broken TCP connection
                        }
                        Err(async_mpsc::error::TrySendError::Full(
                            Outbound::Data(_) | Outbound::DrainFlush(_),
                        )) => {
                            unreachable!("only ACKs are retried here");
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

struct PeerConnection {
    sender: async_mpsc::Sender<Outbound>,
    acknowledgments: AckBacklog,
    reader: JoinHandle<()>,
    writer: JoinHandle<()>,
    session: u64,
}

/// Stable names/indices plus the currently live sessions. The fixed layout is
/// shared by all exchanges; manual-mode layouts change only at connect/close.
struct PeerConnections {
    ids: Vec<String>,
    indices: HashMap<String, PeerIndex>,
    live: HashMap<PeerIndex, PeerConnection>,
    layout: Arc<PeerLayout>,
    fixed: bool,
}

impl PeerConnections {
    fn new(expected: Option<&HashSet<String>>) -> Self {
        let mut ids: Vec<String> = expected
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        ids.sort();
        let indices = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let layout = Arc::new(PeerLayout::new(
            ids.iter()
                .enumerate()
                .map(|(i, id)| (i, id.clone()))
                .collect(),
        ));
        Self {
            ids,
            indices,
            live: HashMap::new(),
            layout,
            fixed: expected.is_some(),
        }
    }

    fn index(&self, id: &str) -> Option<PeerIndex> {
        self.indices.get(id).copied()
    }

    fn intern(&mut self, id: &str) -> PeerIndex {
        if let Some(index) = self.index(id) {
            return index;
        }
        let index = self.ids.len();
        self.ids.push(id.to_owned());
        self.indices.insert(id.to_owned(), index);
        index
    }

    fn contains_id(&self, id: &str) -> bool {
        self.index(id)
            .is_some_and(|index| self.live.contains_key(&index))
    }

    fn refresh_manual_layout(&mut self) {
        if !self.fixed {
            self.layout = Arc::new(PeerLayout::new(
                self.live
                    .keys()
                    .map(|&i| (i, self.ids[i].clone()))
                    .collect(),
            ));
        }
    }
}

struct PendingExchange {
    batch: Batch,
    reply: ExchangeReply,
}

/// Graceful shutdown only: normal exchange latency and completion are unchanged.
struct DrainRequest {
    id: u64,
    deadline: Instant,
    reply: Reply,
    // None: wait for outbound DATA ACKs; Some: await per-writer markers.
    waiting: Option<HashMap<PeerIndex, u64>>,
}

/// Messages not yet ACKed by this neighbor. Preserve the send order on replay.
/// Only a few frames are retained; a VecDeque keeps this simpler than two maps.
#[derive(Default)]
struct UnackedMessages {
    frames: VecDeque<(Bytes, DataFrame)>,
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
    fn insert(&mut self, key: Bytes, frame: impl Into<DataFrame>) {
        let frame = frame.into();
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

// The buffered reader lives for the entire session. It is never recreated
// between frames and never cancelled by a per-exchange timeout.
async fn read_peer(
    reader: OwnedReadHalf,
    peer: PeerIndex,
    session: u64,
    max_message_size: usize,
    events: async_mpsc::Sender<PeerEvent>,
) {
    let mut reader = FrameReader::new(reader);
    loop {
        let received = async {
            let frame = reader.recv(max_message_size).await?;
            message::decode(frame)
        }
        .await;
        let event = match received {
            Ok(message::Message::Data { key, payload }) => PeerEvent::Data {
                peer,
                session,
                key,
                data: payload,
            },
            Ok(message::Message::Ack { key }) => PeerEvent::Ack { peer, session, key },
            Err(err) => {
                let _ = events
                    .send(PeerEvent::Closed {
                        peer,
                        session,
                        reason: err.to_string(),
                    })
                    .await;
                return;
            }
        };
        if events.send(event).await.is_err() {
            return;
        }
    }
}

async fn write_peer(
    mut writer: OwnedWriteHalf,
    mut outbound: async_mpsc::Receiver<Outbound>,
    peer: PeerIndex,
    session: u64,
    max_message_size: usize,
    events: async_mpsc::Sender<PeerEvent>,
    capacity_waiting: Arc<AtomicBool>,
    capacity_changed: Arc<Notify>,
) {
    // Small ACK storage is reused; owned DATA keeps its original send path.
    let mut ack_buffer = [0u8; message::ACK_BUFFER_SIZE];
    while let Some(frame) = outbound.recv().await {
        if capacity_waiting.load(Ordering::Acquire)
            && capacity_waiting.swap(false, Ordering::AcqRel)
        {
            capacity_changed.notify_one();
        }
        let result = match frame {
            Outbound::Data(DataFrame::Owned(data)) => {
                send_message(&mut writer, &data, max_message_size).await
            }
            Outbound::Data(DataFrame::Borrowed(frame)) => {
                // The writer holds its Arc during the entire asynchronous write;
                // ACK can only drop the cache's independent reference.
                send_message_parts(&mut writer, &frame.head, frame.payload(), max_message_size)
                    .await
            }
            Outbound::Ack(key) => match message::encode_ack_into(&key, &mut ack_buffer) {
                Ok(data) => send_message(&mut writer, data, max_message_size).await,
                Err(err) => Err(err),
            },
            Outbound::DrainFlush(id) => {
                // All frames before this marker have completed write_all().
                if events
                    .send(PeerEvent::DrainFlushed { peer, session, id })
                    .await
                    .is_err()
                {
                    return;
                }
                Ok(())
            }
        };
        if let Err(err) = result {
            let _ = events
                .send(PeerEvent::Closed {
                    peer,
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
    peers: &mut PeerConnections,
    next_session: &mut u64,
    max_message_size: usize,
    events: &async_mpsc::Sender<PeerEvent>,
    capacity_changed: &Arc<Notify>,
    established: Established,
) -> Option<PeerIndex> {
    let Established {
        peer_id,
        stream,
        outgoing,
        reply,
    } = established;
    let mut registered = None;
    let result = if expected.is_some_and(|allowed| !allowed.contains(&peer_id)) {
        Err(format!("Peer {peer_id} is not in configured neighbors"))
    } else if outgoing != (node_id < peer_id.as_str()) {
        Err("Unexpected connection direction".to_owned())
    } else if peers.contains_id(&peer_id) {
        Err(format!("Duplicate connection: {peer_id}"))
    } else {
        let index = peers.intern(&peer_id);
        *next_session += 1;
        let session = *next_session;
        let (reader, writer) = stream.into_split();
        let (sender, outbound) = async_mpsc::channel(SEND_QUEUE_SIZE);
        let acknowledgments = AckBacklog::default();
        let reader_task = tokio::spawn(read_peer(
            reader,
            index,
            session,
            max_message_size,
            events.clone(),
        ));
        let writer_task = tokio::spawn(write_peer(
            writer,
            outbound,
            index,
            session,
            max_message_size,
            events.clone(),
            acknowledgments.capacity_waiting.clone(),
            capacity_changed.clone(),
        ));
        peers.live.insert(
            index,
            PeerConnection {
                sender,
                acknowledgments,
                reader: reader_task,
                writer: writer_task,
                session,
            },
        );
        peers.refresh_manual_layout();
        registered = Some(index);
        info!("[{node_id}] Connected to {peer_id}");
        Ok(())
    };
    if let Some(reply) = reply {
        let _ = reply.send(result);
    }
    registered
}

fn fail_exchange(
    pending: &mut Option<PendingExchange>,
    completed: &mut CompletedKeys,
    recycled_slots: &mut Vec<Option<Bytes>>,
    reason: String,
) {
    if let Some(active) = pending.take() {
        completed.insert(active.batch.key().clone());
        *recycled_slots = active.batch.recycle();
        let _ = active.reply.send(Err(reason));
    }
}

fn complete_exchange(
    pending: &mut Option<PendingExchange>,
    completed: &mut CompletedKeys,
    recycled_slots: &mut Vec<Option<Bytes>>,
) {
    if pending
        .as_ref()
        .is_some_and(|active| active.batch.is_complete())
    {
        let active = pending.take().expect("checked pending exchange");
        completed.insert(active.batch.key().clone());
        let (results, slots) = active.batch.take().expect("checked complete batch");
        *recycled_slots = slots;
        let _ = active.reply.send(Ok(results));
    }
}

/// Nonblocking progress: it must never wait on an mpsc queue from the network loop.
fn progress_drain(
    drain: &mut DrainRequest,
    peers: &PeerConnections,
    unacked: &HashMap<PeerIndex, UnackedMessages>,
    capacity_changed: &Notify,
) -> Result<bool, String> {
    if let Some(waiting) = drain.waiting.as_ref() {
        // A writer from another TCP session cannot satisfy an old marker.
        for (&index, &session) in waiting {
            if !peers.live.get(&index).is_some_and(|p| p.session == session) {
                return Err(format!(
                    "Peer {} disconnected during drain",
                    peers.ids[index]
                ));
            }
        }
        return Ok(waiting.is_empty());
    }

    // An ACK proves the peer has accepted our DATA. Still keep readers alive
    // so that receiving and acknowledging the peer's DATA can proceed.
    if unacked.values().any(|pending| !pending.frames.is_empty())
        || peers
            .live
            .values()
            .any(|peer| !peer.acknowledgments.deferred.is_empty())
    {
        return Ok(false);
    }

    // Insert one marker behind every queued ACK. Reserve all permits before
    // inserting any markers, so queue pressure cannot create a partial phase.
    let mut permits = Vec::with_capacity(peers.live.len());
    for (&index, peer) in &peers.live {
        match peer.sender.try_reserve() {
            Ok(permit) => permits.push((index, peer.session, permit)),
            Err(async_mpsc::error::TrySendError::Closed(_)) => {
                return Err(format!(
                    "Peer {} writer stopped during drain",
                    peers.ids[index]
                ));
            }
            Err(async_mpsc::error::TrySendError::Full(_)) => {
                // The writer will wake the network loop on available capacity.
                // Retry once after registering to avoid a lost wakeup.
                peer.acknowledgments
                    .capacity_waiting
                    .store(true, Ordering::Release);
                if let Ok(permit) = peer.sender.try_reserve() {
                    drop(permit);
                    capacity_changed.notify_one();
                }
                return Ok(false);
            }
        }
    }
    let mut waiting = HashMap::with_capacity(permits.len());
    for (index, session, permit) in permits {
        waiting.insert(index, session);
        permit.send(Outbound::DrainFlush(drain.id));
    }
    let finished = waiting.is_empty();
    drain.waiting = Some(waiting);
    Ok(finished)
}

fn disconnect_peer(node_id: &str, peers: &mut PeerConnections, index: PeerIndex, reason: &str) {
    if let Some(peer) = peers.live.remove(&index) {
        peer.reader.abort();
        peer.writer.abort();
        peers.refresh_manual_layout();
        // Pending batch, early data, stable index and unacked DATA all survive.
        // Deferred ACKs may be regenerated if the peer replays its DATA.
        info!(
            "[{node_id}] Disconnected from {}: {reason}",
            peers.ids[index]
        );
    }
}

fn handle_peer_event(
    event: PeerEvent,
    node_id: &str,
    peers: &mut PeerConnections,
    unacked: &mut HashMap<PeerIndex, UnackedMessages>,
    early: &mut EarlyMessages,
    pending: &mut Option<PendingExchange>,
    completed: &mut CompletedKeys,
    recycled_slots: &mut Vec<Option<Bytes>>,
    draining: &mut Option<DrainRequest>,
) {
    match event {
        PeerEvent::Ack { peer, session, key } => {
            if peers.live.get(&peer).is_some_and(|p| p.session == session) {
                if let Some(messages) = unacked.get_mut(&peer) {
                    messages.acknowledge(&key);
                }
            }
        }
        PeerEvent::Data {
            peer,
            session,
            key,
            data,
        } => {
            if !peers.live.get(&peer).is_some_and(|p| p.session == session) {
                return;
            }
            let accepted = if pending.as_ref().is_some_and(|p| p.batch.key() == &key) {
                match pending
                    .as_mut()
                    .expect("checked pending")
                    .batch
                    .insert(peer, data)
                {
                    Ok(_) => true,
                    Err(err) => {
                        fail_exchange(
                            pending,
                            completed,
                            recycled_slots,
                            format!("Invalid exchange message from {}: {err}", peers.ids[peer]),
                        );
                        false
                    }
                }
            } else if completed.contains(&key) {
                true
            } else {
                match early.insert(key.clone(), peer, data) {
                    Ok(_) => true,
                    Err(err) => {
                        disconnect_peer(node_id, peers, peer, err);
                        false
                    }
                }
            };
            if accepted {
                if let Some(connection) = peers.live.get_mut(&peer) {
                    if let Err(err) = connection.acknowledgments.queue(&connection.sender, key) {
                        disconnect_peer(node_id, peers, peer, &err);
                    }
                }
                // Return as soon as this DATA completes the batch, even while
                // other ready events remain in the bounded drain below.
                complete_exchange(pending, completed, recycled_slots);
            }
        }
        PeerEvent::Closed {
            peer,
            session,
            reason,
        } => {
            if peers.live.get(&peer).is_some_and(|p| p.session == session) {
                disconnect_peer(node_id, peers, peer, &reason);
            }
        }
        PeerEvent::DrainFlushed { peer, session, id } => {
            if let Some(drain) = draining.as_mut() {
                if drain.id == id {
                    if let Some(waiting) = drain.waiting.as_mut() {
                        if waiting.get(&peer) == Some(&session) {
                            waiting.remove(&peer);
                        }
                    }
                }
            }
        }
    }
}

async fn run_network(
    node_id: String,
    port: u16,
    neighbors: Option<HashSet<String>>,
    group_id: String,
    peer_endpoints: Option<HashMap<String, String>>,
    config: NetworkConfig,
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
    let mut peers = PeerConnections::new(neighbors.as_ref());
    let capacity_changed = Arc::new(Notify::new());
    let mut recycled_slots = Vec::<Option<Bytes>>::new();
    let mut next_session = 0u64;
    let mut connecting = HashSet::<String>::new();
    let mut connected_before = HashSet::<PeerIndex>::new();
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
    let mut draining: Option<DrainRequest> = None;
    let mut quiescing = false;
    let mut drained = false;
    let mut next_drain_id = 0u64;
    let mut early = EarlyMessages::new(EARLY_MAX_MESSAGES, config.max_early_bytes);
    let mut completed = CompletedKeys::new(COMPLETED_MAX_KEYS);
    // Retain unacknowledged DATA separately for each peer, even while disconnected.
    let mut unacked = HashMap::<PeerIndex, UnackedMessages>::new();
    early.reserve_peers(peers.ids.len());

    loop {
        // Copy the deadline to avoid borrowing `draining` across select branches.
        let drain_deadline = draining.as_ref().map(|request| request.deadline);
        tokio::select! {
            _ = &mut stop_rx => break,

            _ = async {
                match drain_deadline {
                    Some(deadline) => {
                        tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
                    }
                    None => future::pending::<()>().await,
                }
            } => {
                if let Some(request) = draining.take() {
                    let _ = request.reply.send(Err("Drain timed out".into()));
                }
            },

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
                        if quiescing {
                            let _ = reply.send(Err("Network is draining".into()));
                            continue;
                        }
                        if neighbors.as_ref().is_some_and(|ids| !ids.contains(&peer_id)) {
                            let _ = reply.send(Err(format!("Peer {peer_id} is not a configured neighbor")));
                        } else if peers.contains_id(&peer_id) || !connecting.insert(peer_id.clone()) {
                            let _ = reply.send(Err(format!("Already connected or connecting to {peer_id}")));
                        } else {
                            addresses.insert(peer_id.clone(), vec![address.clone()]);
                            start_connect(&node_id, peer_id, vec![address], established_tx.clone(), Some(reply));
                        }
                    }
                    Some(Command::Peers { reply }) => {
                        let mut ids: Vec<_> = peers.live.keys().map(|&i| peers.ids[i].clone()).collect();
                        ids.sort();
                        let _ = reply.send(ids);
                    }
                    Some(Command::Exchange { key, data, reply }) => {
                        if quiescing {
                            let _ = reply.send(Err("Network is draining".into()));
                            continue;
                        }
                        if pending.is_some() {
                            let _ = reply.send(Err("Another exchange is in progress".into()));
                            continue;
                        }
                        if completed.contains(&key) {
                            let _ = reply.send(Err("Exchange key has already been used".into()));
                            continue;
                        }
                        if let Some(expected) = &neighbors {
                            if peers.live.len() != expected.len() {
                                let _ = reply.send(Err("Not all configured neighbors are connected".into()));
                                continue;
                            }
                        }
                        if let ExchangeData::Neighborwise(frames) = &data {
                            if frames.len() != peers.live.len()
                                || frames.keys().any(|id| !peers.contains_id(id))
                            {
                                let _ = reply.send(Err("Neighborwise states must match connected peers".into()));
                                continue;
                            }
                        }

                        let mut batch = Batch::new(
                            key.clone(), peers.layout.clone(), std::mem::take(&mut recycled_slots),
                        );
                        let mut error = early.check_batch(&batch).err().map(str::to_owned);
                        let mut permits = Vec::with_capacity(peers.live.len());
                        if error.is_none() {
                            for (&index, peer) in &peers.live {
                                let frame = data.for_peer(&peers.ids[index]).expect("checked peers");
                                if let Err(err) = unacked.entry(index).or_default().check_capacity(
                                    &key, frame.len(), config.max_unacked_bytes,
                                ) {
                                    error = Some(format!("Cannot retain message for {}: {err}", peers.ids[index]));
                                    break;
                                }
                                match peer.sender.try_reserve() {
                                    Ok(permit) => permits.push((index, permit)),
                                    Err(err) => {
                                        error = Some(format!("Cannot queue message to {}: {err}", peers.ids[index]));
                                        break;
                                    }
                                }
                            }
                        }
                        if let Some(reason) = error {
                            // Nothing was sent and acknowledged early DATA was
                            // NOT removed. Releasing permits restores capacity.
                            drop(permits);
                            recycled_slots = batch.recycle();
                            let _ = reply.send(Err(reason));
                        } else {
                            // All fallible preflight checks passed before any
                            // early DATA is consumed or any DATA is submitted.
                            early.fill_batch(&mut batch).expect("validated early batch");
                            for (index, permit) in permits {
                                let frame = data.for_peer(&peers.ids[index]).expect("checked peers");
                                unacked.entry(index).or_default().insert(key.clone(), frame.clone());
                                permit.send(Outbound::Data(frame.clone()));
                            }
                            pending = Some(PendingExchange { batch, reply });
                            complete_exchange(&mut pending, &mut completed, &mut recycled_slots);
                        }
                    }

                    Some(Command::CancelExchange { key }) => {
                        if pending.as_ref().is_some_and(|p| p.batch.key() == &key) {
                            fail_exchange(&mut pending, &mut completed, &mut recycled_slots,
                                "Exchange cancelled after timeout".into());
                        }
                    }
                    Some(Command::Drain { deadline, reply }) => {
                        if drained {
                            let _ = reply.send(Ok(()));
                        } else if draining.is_some() {
                            let _ = reply.send(Err("Another drain is already in progress".into()));
                        } else if pending.is_some() {
                            let _ = reply.send(Err("Cannot drain while an exchange is in progress".into()));
                        } else {
                            // Terminal operation: subsequent exchanges are not accepted,
                            // including if this drain later times out.
                            quiescing = true;
                            next_drain_id = next_drain_id.wrapping_add(1);
                            draining = Some(DrainRequest {
                                id: next_drain_id,
                                deadline,
                                reply,
                                waiting: None,
                            });
                        }
                    }
                    None => break,
                }
            }

            established = established_rx.recv() => {
                match established {
                    Some(ConnectEvent::Established(established)) => {
                        let peer_id = established.peer_id.clone();
                        if established.outgoing { connecting.remove(&peer_id); }
                        if let Some(index) = register_peer(
                            &node_id, neighbors.as_ref(), &mut peers, &mut next_session,
                            config.max_message_size, &event_tx, &capacity_changed, established,
                        ) {
                            early.reserve_peers(peers.ids.len());
                            if !connected_before.insert(index) {
                                let sender = &peers.live.get(&index).expect("new peer").sender;
                                let result = unacked.get(&index).map(|messages| {
                                    messages.frames.iter().try_for_each(|(_, frame)| {
                                        sender.try_send(Outbound::Data(frame.clone())).map_err(|err| err.to_string())
                                    })
                                });
                                if let Some(Err(err)) = result {
                                    disconnect_peer(&node_id, &mut peers, index, &format!("Replay queue error: {err}"));
                                }
                            } else {
                                fail_exchange(&mut pending, &mut completed, &mut recycled_slots,
                                    "Peer set changed during exchange".into());
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
                                && !peers.contains_id(&peer_id)
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
                        && !peers.contains_id(peer_id)
                        && connecting.insert(peer_id.clone())
                    {
                        start_connect(
                            &node_id, peer_id.clone(), candidates.clone(),
                            established_tx.clone(), None,
                        );
                    }
                }
            }

            _ = capacity_changed.notified() => {
                // Only writers with a blocked ACK producer trigger this branch.
                let mut failed = Vec::new();
                for (&index, peer) in peers.live.iter_mut() {
                    if let Err(reason) = peer.acknowledgments.flush(&peer.sender) {
                        failed.push((index, reason));
                    }
                }
                for (index, reason) in failed {
                    disconnect_peer(&node_id, &mut peers, index, &reason);
                }
            }

            event = event_rx.recv() => {
                let Some(event) = event else { break; };
                handle_peer_event(event, &node_id, &mut peers, &mut unacked,
                    &mut early, &mut pending, &mut completed, &mut recycled_slots, &mut draining);
                for _ in 1..EVENT_BATCH_LIMIT {
                    let Ok(event) = event_rx.try_recv() else { break; };
                    handle_peer_event(event, &node_id, &mut peers, &mut unacked,
                        &mut early, &mut pending, &mut completed, &mut recycled_slots, &mut draining);
                }
            }
        }

        if draining
            .as_ref()
            .is_some_and(|request| Instant::now() >= request.deadline)
        {
            if let Some(request) = draining.take() {
                let _ = request.reply.send(Err("Drain timed out".into()));
            }
        } else {
            let progress = draining
                .as_mut()
                .map(|request| progress_drain(request, &peers, &unacked, &capacity_changed));
            match progress {
                Some(Ok(true)) => {
                    drained = true;
                    let request = draining.take().expect("completed drain");
                    let _ = request.reply.send(Ok(()));
                }
                Some(Err(reason)) => {
                    let request = draining.take().expect("failed drain");
                    let _ = request.reply.send(Err(reason));
                }
                _ => {}
            }
        }
    }

    if let Some(request) = draining.take() {
        let _ = request
            .reply
            .send(Err("Network stopped during drain".into()));
    }
    fail_exchange(
        &mut pending,
        &mut completed,
        &mut recycled_slots,
        "Network stopped".into(),
    );
    for (_, peer) in peers.live.drain() {
        peer.reader.abort();
        peer.writer.abort();
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
    /// Shared blocking exchange path; frames own snapshots or retain their exporters.
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
        let end = deadline
            .map(|duration| {
                Instant::now()
                    .checked_add(duration)
                    .ok_or_else(|| PyValueError::new_err("timeout is too large"))
            })
            .transpose()?;
        let (reply, mut result_rx) = mpsc::channel();
        commands
            .send(Command::Exchange {
                key: key.clone(),
                data,
                reply,
            })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;

        // Periodically check Python signals so Ctrl+C can interrupt exchange().
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
        let worker = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
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

    /// Broadcast one NumPy payload to all neighbors. By default it is copied
    /// once into an owned DATA snapshot; opt-in copy=false borrows its storage.
    #[pyo3(signature = (key, meta, payload, timeout=None, *, copy=true))]
    pub fn exchange_array(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyBytes>,
        meta: &Bound<'_, PyBytes>,
        payload: PyReadonlyArray1<'_, u8>,
        timeout: Option<f64>,
        copy: bool,
    ) -> PyResult<HashMap<String, Py<PyMemoryView>>> {
        let key = key.as_bytes();
        let meta = meta.as_bytes();
        let payload_bytes = payload.as_slice().map_err(|_| {
            PyValueError::new_err("payload must be a contiguous 1-D uint8 NumPy array")
        })?;
        let size = message::HEADER_SIZE
            .checked_add(key.len())
            .and_then(|size| size.checked_add(4))
            .and_then(|size| size.checked_add(meta.len()))
            .and_then(|size| size.checked_add(payload_bytes.len()))
            .ok_or_else(|| PyValueError::new_err("Message too large"))?;
        if size > self.config.max_message_size {
            return Err(PyValueError::new_err("Message too large"));
        }
        let frame = if copy {
            // Keep the existing broadcast path: one shared Rust-owned snapshot.
            DataFrame::Owned(
                message::encode_state(key, meta, payload_bytes)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            // Borrow the *prepared* NumPy view, shared by all peer writers.
            let head = message::encode_state(key, meta, &[])
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            // SAFETY: copy=false callers must not change the exported storage
            // through any alias while background writers/replay may access it.
            let borrowed = unsafe { BorrowedFrame::new(head, payload.as_any(), payload_bytes)? };
            DataFrame::Borrowed(Arc::new(borrowed))
        };
        self.exchange_encoded(
            py,
            Bytes::copy_from_slice(key),
            ExchangeData::Broadcast(frame),
            timeout,
        )
    }

    /// Send a different NumPy state to each peer under the same exchange key.
    #[pyo3(signature = (key, states, timeout=None, *, copy=true))]
    pub fn neighborwise_exchange_array(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyBytes>,
        states: &Bound<'_, PyDict>,
        timeout: Option<f64>,
        copy: bool,
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

            let frame = if copy {
                // Default: preserve the original independent DATA snapshot.
                DataFrame::Owned(
                    message::encode_state(key, meta.as_bytes(), payload)
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                )
            } else {
                // encode_state(key, meta, &[]) creates the unchanged DATA head.
                // The TCP length prefix accounts for head + borrowed payload.
                let head = message::encode_state(key, meta.as_bytes(), &[])
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                // SAFETY: The exporter is this exact contiguous NumPy uint8 view.
                // copy=False callers promise no writes or storage invalidation
                // through any alias until the last retained reference is gone.
                let frame = unsafe { BorrowedFrame::new(head, &payload_obj, payload)? };
                DataFrame::Borrowed(Arc::new(frame))
            };
            frames.insert(peer_id, frame);
        }

        self.exchange_encoded(
            py,
            Bytes::copy_from_slice(key),
            ExchangeData::Neighborwise(frames),
            timeout,
        )
    }

    /// Wait for outbound DATA acknowledgments and locally queued ACK writes.
    /// This is a terminal operation; call close() afterwards even on timeout.
    #[pyo3(signature = (timeout=5.0))]
    pub fn drain(&self, py: Python<'_>, timeout: f64) -> PyResult<()> {
        if !timeout.is_finite() || timeout < 0.0 {
            return Err(PyValueError::new_err("timeout must be finite and >= 0"));
        }
        let duration = Duration::try_from_secs_f64(timeout)
            .map_err(|_| PyValueError::new_err("timeout is too large"))?;
        let deadline = Instant::now()
            .checked_add(duration)
            .ok_or_else(|| PyValueError::new_err("timeout is too large"))?;
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Node is not started"))?;
        let (reply, mut result_rx) = mpsc::channel();
        commands
            .send(Command::Drain { deadline, reply })
            .map_err(|_| PyRuntimeError::new_err("Network worker has stopped"))?;

        loop {
            py.check_signals()?;
            let rx = &mut result_rx;
            let result = py.detach(move || rx.recv_timeout(Duration::from_millis(100)));
            match result {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(reason)) if reason == "Drain timed out" => {
                    return Err(PyTimeoutError::new_err(reason));
                }
                Ok(Err(reason)) => return Err(PyRuntimeError::new_err(reason)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(PyRuntimeError::new_err(
                        "Network worker stopped during drain",
                    ));
                }
            }
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
        let broadcast = ExchangeData::Broadcast(Bytes::from_static(b"same").into());
        assert!(
            matches!(broadcast.for_peer("A"), Some(DataFrame::Owned(f)) if f.as_ref() == b"same")
        );
        assert!(
            matches!(broadcast.for_peer("B"), Some(DataFrame::Owned(f)) if f.as_ref() == b"same")
        );

        let mut frames = HashMap::new();
        frames.insert("A".into(), DataFrame::Owned(Bytes::from_static(b"to A")));
        frames.insert("B".into(), DataFrame::Owned(Bytes::from_static(b"to B")));
        let neighborwise = ExchangeData::Neighborwise(frames);
        assert!(
            matches!(neighborwise.for_peer("A"), Some(DataFrame::Owned(f)) if f.as_ref() == b"to A")
        );
        assert!(
            matches!(neighborwise.for_peer("B"), Some(DataFrame::Owned(f)) if f.as_ref() == b"to B")
        );
        assert!(neighborwise.for_peer("C").is_none());
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::*;

    #[tokio::test]
    async fn full_ack_queue_defers_without_disconnect_and_preserves_order() {
        let (tx, mut rx) = async_mpsc::channel(1);
        assert!(
            tx.try_send(Outbound::Data(Bytes::from_static(b"first").into()))
                .is_ok()
        );
        let mut backlog = AckBacklog::default();
        backlog.queue(&tx, Bytes::from_static(b"a")).unwrap();
        backlog.queue(&tx, Bytes::from_static(b"b")).unwrap();
        backlog.queue(&tx, Bytes::from_static(b"a")).unwrap();
        assert_eq!(backlog.deferred.len(), 2); // duplicate ACK is coalesced
        assert!(backlog.capacity_waiting.load(Ordering::Acquire));
        assert!(matches!(rx.recv().await, Some(Outbound::Data(_))));
        backlog.flush(&tx).unwrap();
        match rx.recv().await {
            Some(Outbound::Ack(k)) => assert_eq!(k.as_ref(), b"a"),
            _ => panic!("expected first ACK"),
        }
        backlog.flush(&tx).unwrap();
        match rx.recv().await {
            Some(Outbound::Ack(k)) => assert_eq!(k.as_ref(), b"b"),
            _ => panic!("expected second ACK"),
        }
        assert!(backlog.deferred.is_empty());
    }

    #[test]
    fn closed_writer_is_not_misreported_as_temporary_backpressure() {
        let (tx, rx) = async_mpsc::channel(1);
        drop(rx);
        assert!(
            AckBacklog::default()
                .queue(&tx, Bytes::from_static(b"k"))
                .is_err()
        );
    }

    #[test]
    fn deferred_ack_memory_is_bounded() {
        let (tx, _rx) = async_mpsc::channel(1);
        assert!(tx.try_send(Outbound::Data(Bytes::new().into())).is_ok());
        let mut b = AckBacklog::default();
        for i in 0..MAX_DEFERRED_ACKS {
            b.queue(&tx, Bytes::copy_from_slice(&(i as u64).to_be_bytes()))
                .unwrap();
        }
        assert!(b.queue(&tx, Bytes::from_static(b"overflow")).is_err());
        assert_eq!(b.deferred.len(), MAX_DEFERRED_ACKS);
    }

    #[test]
    fn fixed_peer_indices_and_layout_do_not_depend_on_hash_order() {
        let expected = HashSet::from(["z".to_owned(), "a".to_owned()]);
        let mut peers = PeerConnections::new(Some(&expected));
        assert_eq!(peers.index("a"), Some(0));
        assert_eq!(peers.index("z"), Some(1));
        let layout = peers.layout.clone();
        assert_eq!(peers.intern("z"), 1);
        peers.refresh_manual_layout();
        assert!(Arc::ptr_eq(&layout, &peers.layout));
        assert_eq!(peers.ids.len(), 2);
    }

    #[test]
    fn manual_mode_indices_are_never_recycled() {
        let mut peers = PeerConnections::new(None);
        let first = peers.intern("A");
        let second = peers.intern("B");
        peers.refresh_manual_layout(); // currently no live connections
        assert!(peers.layout.is_empty());
        assert_eq!(peers.intern("A"), first);
        assert_ne!(first, second);
    }

    async fn reply_value<T>(rx: mpsc::Receiver<T>) -> T {
        timeout(Duration::from_secs(10), async move {
            loop {
                match rx.try_recv() {
                    Ok(value) => return value,
                    Err(mpsc::TryRecvError::Empty) => {
                        tokio::time::sleep(Duration::from_millis(1)).await
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        panic!("test reply sender disconnected")
                    }
                }
            }
        })
        .await
        .expect("test response timed out")
    }

    fn unused_loopback_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    }

    /// Exercises the unchanged protocol on real loopback sockets, with the new
    /// indexed batches and a persistent buffered reader. Not a performance test.
    #[tokio::test]
    async fn loopback_broadcast_neighborwise_and_retained_payloads() {
        let port_a = unused_loopback_port();
        let mut port_b = unused_loopback_port();
        while port_b == port_a {
            port_b = unused_loopback_port();
        }
        let config = NetworkConfig {
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            max_unacked_bytes: DEFAULT_MAX_UNACKED_BYTES,
            max_early_bytes: DEFAULT_MAX_EARLY_BYTES,
        };
        let (cmd_a, recv_a) = async_mpsc::unbounded_channel();
        let (cmd_b, recv_b) = async_mpsc::unbounded_channel();
        let (stop_a, stopped_a) = oneshot::channel();
        let (stop_b, stopped_b) = oneshot::channel();
        let (ready_a, started_a) = mpsc::sync_channel(1);
        let (ready_b, started_b) = mpsc::sync_channel(1);
        let a = tokio::spawn(run_network(
            "A".into(),
            port_a,
            Some(HashSet::from(["B".into()])),
            "test".into(),
            Some(HashMap::from([("B".into(), format!("127.0.0.1:{port_b}"))])),
            config,
            stopped_a,
            recv_a,
            ready_a,
        ));
        let b = tokio::spawn(run_network(
            "B".into(),
            port_b,
            Some(HashSet::from(["A".into()])),
            "test".into(),
            Some(HashMap::from([("A".into(), format!("127.0.0.1:{port_a}"))])),
            config,
            stopped_b,
            recv_b,
            ready_b,
        ));
        reply_value(started_a).await.unwrap();
        reply_value(started_b).await.unwrap();
        timeout(Duration::from_secs(10), async {
            loop {
                let (reply, rx) = mpsc::channel();
                assert!(cmd_a.send(Command::Peers { reply }).is_ok());
                let a_ready = reply_value(rx).await == vec!["B".to_owned()];
                let (reply, rx) = mpsc::channel();
                assert!(cmd_b.send(Command::Peers { reply }).is_ok());
                let b_ready = reply_value(rx).await == vec!["A".to_owned()];
                if a_ready && b_ready {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("connection timed out");

        let mut retained = Vec::new();
        for (iteration, size) in [0, 5, 256, 8191, 8192, 128 * 1024, 5]
            .into_iter()
            .enumerate()
        {
            let key = Bytes::copy_from_slice(&(iteration as u64).to_be_bytes());
            let left = vec![0x31; size];
            let right = vec![0x52; size];
            let frame_a = message::encode_state(&key, b"", &left).unwrap();
            let frame_b = message::encode_state(&key, b"", &right).unwrap();
            let (reply_a, result_a) = mpsc::channel();
            let (reply_b, result_b) = mpsc::channel();
            assert!(
                cmd_a
                    .send(Command::Exchange {
                        key: key.clone(),
                        data: ExchangeData::Broadcast(frame_a.into()),
                        reply: reply_a,
                    })
                    .is_ok()
            );
            // Let A's data reach B before B submits the matching command.
            tokio::time::sleep(Duration::from_millis(2)).await;
            assert!(
                cmd_b
                    .send(Command::Exchange {
                        key,
                        data: ExchangeData::Neighborwise(HashMap::from([(
                            "A".into(),
                            frame_b.into()
                        )])),
                        reply: reply_b,
                    })
                    .is_ok()
            );
            let received_a = reply_value(result_a).await.unwrap();
            let received_b = reply_value(result_b).await.unwrap();
            assert_eq!(&received_a["B"][4..], right.as_slice());
            assert_eq!(&received_b["A"][4..], left.as_slice());
            retained.push((received_a["B"].clone(), right));
        }
        for (frame, expected) in retained {
            assert_eq!(&frame[4..], expected.as_slice());
        }
        let _ = stop_a.send(());
        let _ = stop_b.send(());
        a.await.unwrap();
        b.await.unwrap();
    }

    /// Regression: B's future DATA has already been ACKed, while A's unacked
    /// cache is full. A's failed preflight must leave that future DATA intact.
    #[tokio::test]
    async fn early_data_survives_unacked_capacity_failure_and_retry() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let b_address = listener.local_addr().unwrap().to_string();
        let config = NetworkConfig {
            max_message_size: DEFAULT_MAX_MESSAGE_SIZE,
            max_unacked_bytes: DEFAULT_MAX_UNACKED_BYTES,
            max_early_bytes: DEFAULT_MAX_EARLY_BYTES,
        };
        let (commands, command_rx) = async_mpsc::unbounded_channel();
        let (stop, stop_rx) = oneshot::channel();
        let (ready, ready_rx) = mpsc::sync_channel(1);
        let worker = tokio::spawn(run_network(
            "A".into(),
            0,
            Some(HashSet::from(["B".into()])),
            "early-test".into(),
            Some(HashMap::from([("B".into(), b_address)])),
            config,
            stop_rx,
            command_rx,
            ready,
        ));
        reply_value(ready_rx).await.unwrap();
        let (stream, _) = timeout(Duration::from_secs(10), listener.accept())
            .await
            .unwrap()
            .unwrap();
        let (_, stream) = handshake(stream, "B").await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = FrameReader::new(reader);
        timeout(Duration::from_secs(10), async {
            loop {
                let (reply, rx) = mpsc::channel();
                assert!(commands.send(Command::Peers { reply }).is_ok());
                if reply_value(rx).await == vec!["B".to_owned()] {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();

        for i in 0..MAX_UNACKED_MESSAGES {
            let key = Bytes::copy_from_slice(&(i as u64).to_be_bytes());
            let remote = message::encode_state(&key, b"", b"from B").unwrap();
            send_message(&mut writer, &remote, config.max_message_size)
                .await
                .unwrap();
            let local = message::encode_state(&key, b"", b"from A").unwrap();
            let (reply, rx) = mpsc::channel();
            assert!(
                commands
                    .send(Command::Exchange {
                        key: key.clone(),
                        data: ExchangeData::Broadcast(local.into()),
                        reply,
                    })
                    .is_ok()
            );
            assert_eq!(&reply_value(rx).await.unwrap()["B"][4..], b"from B");
            let mut data_seen = false;
            let mut ack_seen = false;
            while !(data_seen && ack_seen) {
                let frame = timeout(
                    Duration::from_secs(10),
                    reader.recv(config.max_message_size),
                )
                .await
                .unwrap()
                .unwrap();
                match message::decode(frame).unwrap() {
                    message::Message::Data { key: received, .. } => {
                        assert_eq!(received, key);
                        data_seen = true;
                    }
                    message::Message::Ack { key: received } => {
                        assert_eq!(received, key);
                        ack_seen = true;
                    }
                }
            }
            // Deliberately withhold B's ACK for A's DATA.
        }

        let key = Bytes::from_static(b"future");
        let remote = message::encode_state(&key, b"", b"must survive").unwrap();
        send_message(&mut writer, &remote, config.max_message_size)
            .await
            .unwrap();
        let ack = timeout(
            Duration::from_secs(10),
            reader.recv(config.max_message_size),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            message::decode(ack).unwrap(),
            message::Message::Ack { key: key.clone() }
        );
        let local = message::encode_state(&key, b"", b"local future").unwrap();
        let (reply, rx) = mpsc::channel();
        assert!(
            commands
                .send(Command::Exchange {
                    key: key.clone(),
                    data: ExchangeData::Broadcast(local.clone().into()),
                    reply,
                })
                .is_ok()
        );
        let reason = reply_value(rx).await.unwrap_err();
        assert!(reason.contains("Unacknowledged message cache full"));

        let release = message::encode_ack(&0u64.to_be_bytes());
        send_message(&mut writer, &release, config.max_message_size)
            .await
            .unwrap();
        let result = timeout(Duration::from_secs(10), async {
            loop {
                let (reply, rx) = mpsc::channel();
                assert!(
                    commands
                        .send(Command::Exchange {
                            key: key.clone(),
                            data: ExchangeData::Broadcast(local.clone().into()),
                            reply,
                        })
                        .is_ok()
                );
                match reply_value(rx).await {
                    Ok(result) => break result,
                    Err(err) if err.contains("Unacknowledged message cache full") => {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    Err(err) => panic!("unexpected exchange failure: {err}"),
                }
            }
        })
        .await
        .expect("early DATA was lost during failed preflight");
        assert_eq!(&result["B"][4..], b"must survive");
        let _ = stop.send(());
        worker.await.unwrap();
    }
}

#[cfg(test)]
mod borrowed_frame_tests {
    use super::*;

    #[test]
    fn borrowed_frame_is_shared_and_cache_counts_full_wire_body() {
        Python::initialize();
        Python::attach(|py| {
            // An immutable exporter makes this Rust-only safety test independent
            // of NumPy mutation rules. Production uses the prepared NumPy view.
            let exporter = PyBytes::new(py, b"shared-payload");
            let original_ptr = exporter.as_bytes().as_ptr();
            let key = Bytes::from_static(b"k");
            let head = message::encode_state(&key, b"meta", &[]).unwrap();
            let expected_len = head.len() + exporter.as_bytes().len();
            // SAFETY: immutable bytes, exact exporter, held by the memoryview.
            let borrowed = unsafe {
                BorrowedFrame::new(head, exporter.as_any(), exporter.as_bytes()).unwrap()
            };
            assert_eq!(borrowed.payload().as_ptr(), original_ptr);
            let frame = DataFrame::Borrowed(Arc::new(borrowed));
            let writer = frame.clone();
            let weak = match &writer {
                DataFrame::Borrowed(borrowed) => Arc::downgrade(borrowed),
                _ => unreachable!(),
            };
            let mut cache = UnackedMessages::default();
            cache
                .check_capacity(&key, frame.len(), expected_len)
                .unwrap();
            cache.insert(key.clone(), frame);
            assert_eq!(cache.bytes, expected_len);
            assert!(
                cache
                    .check_capacity(&Bytes::from_static(b"next"), 1, expected_len)
                    .is_err()
            );
            drop(exporter);
            cache.acknowledge(&key);
            assert_eq!(cache.bytes, 0);
            // ACK cannot release memory still used by a writer.
            assert!(weak.upgrade().is_some());
            match &writer {
                DataFrame::Borrowed(frame) => assert_eq!(frame.payload(), b"shared-payload"),
                _ => unreachable!(),
            }
            drop(writer);
            assert!(weak.upgrade().is_none());
        });
    }

    #[test]
    fn borrowed_and_owned_frames_have_identical_wire_content() {
        Python::initialize();
        Python::attach(|py| {
            for payload in [b"".as_slice(), b"content".as_slice()] {
                let exporter = PyBytes::new(py, payload);
                let key = b"key";
                let meta = b"metadata";
                let owned = message::encode_state(key, meta, payload).unwrap();
                let head = message::encode_state(key, meta, &[]).unwrap();
                // SAFETY: immutable exporter and its exact byte slice.
                let borrowed = unsafe {
                    BorrowedFrame::new(head, exporter.as_any(), exporter.as_bytes()).unwrap()
                };
                let mut assembled = borrowed.head.to_vec();
                assembled.extend_from_slice(borrowed.payload());
                assert_eq!(assembled.as_slice(), owned.as_ref());
            }
        });
    }
}
