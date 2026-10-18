import typing
import threading
import uuid
import logging

import zmq
import pyre
import msgspec
import numpy as np
import numpy.typing as npt

from .transform import Transform, Identity

logger = logging.getLogger(__name__)


class State(msgspec.Struct, tag="STATE", tag_field="type", frozen=True):
    round_id: int
    exchange_id: int
    meta: bytes
    payload: bytes


class Recover(msgspec.Struct, tag="RECOVER", tag_field="type", frozen=True):
    round_id: int
    exchange_id: int


class Replay(msgspec.Struct, tag="REPLAY", tag_field="type", frozen=True):
    round_id: int
    exchange_id: int
    meta: bytes
    payload: bytes


Message = State | Recover | Replay


class Exchange:
    """
    Manages exchange progress and message state across communication rounds.

    Tracks the current round and exchange identifiers, outgoing and incoming
    states, outstanding frontend receives, and recovery-related state needed
    to replay or resume interrupted exchanges.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # (round_id, exchange_id)
        self._progress: tuple[int, int] = (0, 0)
        self._active = False

        # Current-round states sent to each peer.
        # States received before the frontend is ready for them.
        # peer_id -> (round_id, exchange_id) -> state
        self._sent: dict[str, dict[tuple[int, int], State]] = {}
        self._received: dict[str, dict[tuple[int, int], State]] = {}

        # Latest state already forwarded to the frontend for each peer.
        # Outstanding frontend receive for each peer.
        # peer_id -> (round_id, exchange_id)
        self._delivered: dict[str, tuple[int, int]] = {}
        self._waiting: dict[str, tuple[int, int]] = {}

    @property
    def round_id(self) -> int:
        with self._lock:
            return self._progress[0]

    @property
    def exchange_id(self) -> int:
        with self._lock:
            return self._progress[1]

    @property
    def progress(self) -> tuple[int, int]:
        with self._lock:
            return self._progress

    @property
    def sent_peers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sent)

    @property
    def waiting_peers(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._waiting)

    def begin(self) -> tuple[int, int]:
        with self._lock:
            if self._active:
                raise RuntimeError("Exchange already active")

            self._active = True

            return self._progress

    def end(self) -> tuple[int, int]:
        with self._lock:
            if not self._active:
                raise RuntimeError("No active exchange")

            self._active = False
            self._progress = (self._progress[0], self._progress[1] + 1)

            return self._progress

    def reset(self) -> None:
        with self._lock:
            self._ensure_idle()

            self._progress = (0, 0)
            self._sent.clear()
            self._received.clear()
            self._delivered.clear()

    def advance_round(self) -> int:
        with self._lock:
            self._ensure_idle()

            self._progress = (self._progress[0] + 1, 0)

            self._sent.clear()
            self._delivered.clear()
            self._prune_received()

            return self._progress[0]

    def recover_round(self, round_id: int) -> bool:
        with self._lock:
            if round_id <= self._progress[0]:
                return False

            old_progress = self._progress
            exchange_id = old_progress[1]

            sent_states: dict[str, State] = {}

            for peer_id, messages in self._sent.items():
                msg = messages.get(old_progress)

                if msg is not None:
                    sent_states[peer_id] = msg

            # Preserve the frontend exchange phase.
            self._progress = (round_id, exchange_id)
            new_progress = self._progress

            self._sent.clear()

            for peer_id, msg in sent_states.items():
                recovered = State(
                    round_id=round_id,
                    exchange_id=exchange_id,
                    meta=msg.meta,
                    payload=msg.payload,
                )

                self._sent.setdefault(peer_id, {})[new_progress] = recovered

            for peer_id, progress in self._waiting.items():
                if progress == old_progress:
                    self._waiting[peer_id] = new_progress

            # Already delivered values of the current frontend exchange
            # remain delivered after the round promotion.
            for peer_id, progress in self._delivered.items():
                if progress == old_progress:
                    self._delivered[peer_id] = new_progress

            self._prune_received()

            return True

    def cache_sent(self, peer_id: str, msg: State) -> None:
        with self._lock:
            progress = (msg.round_id, msg.exchange_id)

            current = self._progress

            if progress != current:
                raise RuntimeError(f"Exchange mismatch: {progress} != {current}")

            self._sent.setdefault(peer_id, {})[progress] = msg

    def get_sent(self, peer_id: str, progress: tuple[int, int]) -> State | None:
        with self._lock:
            messages = self._sent.get(peer_id)

            if messages is None:
                return None

            return messages.get(progress)

    def cache_received(self, peer_id: str, msg: State) -> None:
        with self._lock:
            progress = (msg.round_id, msg.exchange_id)

            delivered = self._delivered.get(peer_id)
            if delivered is not None and progress <= delivered:
                return

            self._received.setdefault(peer_id, {})[progress] = msg

    def take_received(self, peer_id: str) -> State | None:
        with self._lock:
            progress = self._progress

            messages = self._received.get(peer_id)

            if messages is None:
                return None

            delivered = self._delivered.get(peer_id)
            if delivered is not None and progress <= delivered:
                messages.pop(progress, None)
                return None

            return messages.pop(progress, None)

    def begin_wait(self, peer_id: str) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("No active exchange")

            self._waiting[peer_id] = self._progress

    def is_waiting(self, peer_id: str, msg: State) -> bool:
        with self._lock:
            return self._waiting.get(peer_id) == (msg.round_id, msg.exchange_id)

    def finish_wait(self, peer_id: str, msg: State) -> None:
        with self._lock:
            progress = (msg.round_id, msg.exchange_id)

            if self._waiting.get(peer_id) != progress:
                return

            self._waiting.pop(peer_id, None)

            self._delivered[peer_id] = progress

    def clear_received(self, peer_id: str) -> None:
        with self._lock:
            self._received.pop(peer_id, None)

    def _prune_received(self) -> None:
        for peer_id in list(self._received):
            messages = self._received[peer_id]

            for progress in list(messages):
                if progress[0] < self._progress[0]:
                    messages.pop(progress, None)

            if not messages:
                self._received.pop(peer_id, None)

    def _ensure_idle(self) -> None:
        if self._active:
            raise RuntimeError("Exchange is active")

        if self._waiting:
            raise RuntimeError(f"Waiting peers: {tuple(self._waiting)}")


class NetworkBackend:
    def __init__(
        self,
        node_id: str,
        namespace: str,
        exchange: Exchange,
        node: pyre.Pyre,
        router: zmq.SyncSocket,
    ) -> None:
        self._node_id = node_id
        self._namespace = namespace
        self._exchange = exchange
        self._node = node
        self._router = router

        # Reverse index for Pyre peers:
        # logical node ID -> current Pyre UUID
        self._peer_uuids: dict[str, uuid.UUID] = {}

    def handle_pyre_event(self, event: list[bytes]) -> None:
        if not event:
            return

        try:
            event_type = event[0].decode("utf-8")
        except UnicodeDecodeError:
            return

        if event_type == "WHISPER":
            self._handle_whisper(event)

        elif event_type == "ENTER":
            self._handle_enter(event)

        elif event_type == "EXIT":
            self._handle_exit(event)

    def handle_frontend_message(self, frames: list[bytes]) -> None:
        if len(frames) != 3:
            return

        peer_bytes, meta, payload = frames

        try:
            peer_id = peer_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return

        round_id, exchange_id = self._exchange.progress

        msg = State(
            round_id=round_id,
            exchange_id=exchange_id,
            meta=meta,
            payload=payload,
        )

        self._exchange.cache_sent(peer_id, msg)

        self._exchange.begin_wait(peer_id)

        # A matching message may have arrived before the frontend
        # started waiting for this peer.
        received = self._exchange.take_received(peer_id)

        if received is not None:
            self._exchange.finish_wait(peer_id, received)
            self._forward_to_frontend(peer_id, received)

        self._send_message(peer_id, msg)

    def _handle_enter(self, event: list[bytes]) -> None:
        pyre_uuid = self._extract_uuid(event)

        if pyre_uuid is None:
            return

        peer_id = self._extract_node_id(event)

        if peer_id is None or peer_id == self._node_id:
            return

        namespace = self._extract_namespace(event)

        if namespace != self._namespace:
            return

        peer_uuid = self._peer_uuids.get(peer_id)

        restarted = peer_uuid is not None and peer_uuid != pyre_uuid

        self._peer_uuids[peer_id] = pyre_uuid

        if restarted:
            self._exchange.clear_received(peer_id)

        progress = self._exchange.progress
        msg = self._exchange.get_sent(peer_id, progress)

        if msg is not None:
            self._send_message_by_uuid(pyre_uuid, msg)

    def _handle_exit(self, event: list[bytes]) -> None:
        peer_id = self._extract_node_id(event)

        if peer_id is None:
            return

        pyre_uuid = self._extract_uuid(event)

        if pyre_uuid is None:
            return

        peer_uuid = self._peer_uuids.get(peer_id)

        if peer_uuid is None or peer_uuid != pyre_uuid:
            return

        self._peer_uuids.pop(peer_id, None)

    def _handle_whisper(self, event: list[bytes]) -> None:
        peer_id = self._extract_node_id(event)

        if peer_id is None:
            return

        msg = self._extract_message(event)

        if msg is None:
            return

        if isinstance(msg, State):
            self._handle_state(peer_id, msg)

        elif isinstance(msg, Recover):
            self._handle_recover(msg)

        elif isinstance(msg, Replay):
            self._handle_replay(peer_id, msg)

    def _handle_state(self, peer_id: str, msg: State) -> None:
        local = self._exchange.progress
        remote = (msg.round_id, msg.exchange_id)

        if remote == local:
            self._deliver_or_cache(peer_id, msg)
            return

        if remote < local:
            # Cross-round lag.
            if msg.round_id < local[0]:
                recover = Recover(round_id=local[0], exchange_id=local[1])
                self._send_message(peer_id, recover)
                progress = (local[0], remote[1])

            # Same-round lag.
            else:
                progress = remote

            cached = self._exchange.get_sent(peer_id, progress)

            if cached is not None:
                replay = Replay(
                    round_id=cached.round_id,
                    exchange_id=cached.exchange_id,
                    meta=cached.meta,
                    payload=cached.payload,
                )
                self._send_message(peer_id, replay)

            return

        # Peer is ahead.
        self._exchange.cache_received(peer_id, msg)

    def _handle_recover(self, recover: Recover) -> None:
        before = self._exchange.progress

        if recover.round_id <= before[0]:
            return

        changed = self._exchange.recover_round(recover.round_id)

        if changed:
            self._drain_received()
            self._resend_current()

    def _handle_replay(self, peer_id: str, replay: Replay) -> None:
        msg = State(
            round_id=replay.round_id,
            exchange_id=replay.exchange_id,
            meta=replay.meta,
            payload=replay.payload,
        )

        local = self._exchange.progress
        remote = (replay.round_id, replay.exchange_id)

        if remote == local:
            self._deliver_or_cache(peer_id, msg)
            return

        if remote > local:
            self._exchange.cache_received(peer_id, msg)

        # remote < local:
        # replay 已经过期，直接丢。

    def _drain_received(self) -> None:
        for peer_id in self._exchange.waiting_peers:
            msg = self._exchange.take_received(peer_id)

            if msg is None:
                continue

            self._exchange.finish_wait(peer_id, msg)
            self._forward_to_frontend(peer_id, msg)

    def _resend_current(self) -> None:
        progress = self._exchange.progress

        for peer_id in self._exchange.sent_peers:
            msg = self._exchange.get_sent(peer_id, progress)

            if msg is None:
                continue

            self._send_message(peer_id, msg)

    def _deliver_or_cache(self, peer_id: str, msg: State) -> None:
        if self._exchange.is_waiting(peer_id, msg):
            self._exchange.finish_wait(peer_id, msg)
            self._forward_to_frontend(peer_id, msg)
        else:
            self._exchange.cache_received(peer_id, msg)

    def _forward_to_frontend(self, peer_id: str, msg: State) -> None:
        self._router.send_multipart([peer_id.encode("utf-8"), msg.meta, msg.payload])

    def _send_message(self, peer_id: str, msg: Message) -> None:
        pyre_uuid = self._peer_uuids.get(peer_id)

        if pyre_uuid is None:
            return

        self._node.whisper(pyre_uuid, msgspec.msgpack.encode(msg))

    def _send_message_by_uuid(self, pyre_uuid: uuid.UUID, msg: Message) -> None:
        self._node.whisper(pyre_uuid, msgspec.msgpack.encode(msg))

    @staticmethod
    def _extract_uuid(event: list[bytes]) -> uuid.UUID | None:
        if len(event) < 2:
            return None

        value = event[1]

        if not isinstance(value, bytes) or len(value) != 16:
            return None

        return uuid.UUID(bytes=value)

    @staticmethod
    def _extract_node_id(event: list[bytes]) -> str | None:
        if len(event) < 3:
            return None

        value = event[2]

        if not isinstance(value, bytes):
            return None

        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _extract_namespace(event: list[bytes]) -> str | None:
        if len(event) < 4:
            return None

        candidate = event[3]

        if not isinstance(candidate, bytes):
            return None

        try:
            headers = msgspec.json.decode(candidate, type=dict[str, str])

            return headers.get("namespace")

        except msgspec.DecodeError:
            return None

    @staticmethod
    def _extract_message(event: list[bytes]) -> Message | None:
        if len(event) < 4:
            return None

        try:
            return msgspec.msgpack.decode(event[3], type=Message)
        except msgspec.DecodeError:
            return None


class Network:
    """
    Synchronous communication interface backed by an asynchronous network layer.

    Each network instance represents one node and communicates with its
    weighted neighbors through local ZeroMQ DEALER sockets and a background
    Pyre-based communication thread.

    Parameters
    ----------
    node_id : str
        Unique identifier of the local node.

    neighbors : dict[str, float]
        Mapping from neighbor node IDs to their associated edge weights.

    namespace : str, optional
        Namespace used to isolate independent network instances.
        Defaults to ``"default"``.

    transform : Transform | None, optional
        Transformation applied to exchanged states. If ``None``,
        :class:`Identity` is used.

    recv_timeout_ms : int, optional
        Polling timeout of the background network loop, in milliseconds.
        Defaults to ``200``.

    stop_timeout : float, optional
        Maximum time to wait for the background thread to stop, in seconds.
        Defaults to ``1.0``.

    context : zmq.SyncContext | None, optional
        ZeroMQ context used by the network. If ``None``, the process-wide
        shared context returned by :meth:`zmq.Context.instance` is used.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: dict[str, float],
        *,
        namespace: str = "default",
        transform: Transform | None = None,
        recv_timeout_ms: int = 200,
        stop_timeout: float = 1.0,
        context: zmq.SyncContext | None = None,
    ) -> None:
        self._node_id = node_id
        self._neighbors = neighbors
        self._namespace = namespace
        self._transform = transform if transform is not None else Identity()
        self._recv_timeout_ms = recv_timeout_ms
        self._stop_timeout = stop_timeout

        if context is None:
            self._context: zmq.SyncContext = zmq.Context.instance()
        else:
            self._context = context

        self._exchange = Exchange()

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._dealers: dict[str, zmq.SyncSocket] = {}

        backend_uuid = uuid.uuid4().hex
        self._endpoint = f"inproc://{self._namespace}:{self._node_id}:{backend_uuid}"

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def neighbors(self) -> typing.KeysView[str]:
        return self._neighbors.keys()

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    @property
    def round_id(self) -> int:
        return self._exchange.round_id

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"[{self._node_id}] Network already started")

        if self._dealers:
            raise RuntimeError(f"[{self._node_id}] Stale dealer sockets")

        self._running.set()
        thread = threading.Thread(target=self._run, daemon=True)
        self._thread = thread
        thread.start()

        try:
            for neighbor_id in self._neighbors:
                dealer = self._context.socket(zmq.DEALER)
                dealer.setsockopt(zmq.LINGER, 0)
                dealer.setsockopt(zmq.IDENTITY, neighbor_id.encode("utf-8"))
                dealer.connect(self._endpoint)

                self._dealers[neighbor_id] = dealer

        except Exception:
            self._running.clear()

            for dealer in self._dealers.values():
                dealer.close()
            self._dealers.clear()

            thread.join(self._stop_timeout)

            if not thread.is_alive():
                self._thread = None

            raise

    def stop(self) -> None:
        self._running.clear()

        for dealer in self._dealers.values():
            dealer.close()
        self._dealers.clear()

        thread = self._thread

        if thread is None:
            return

        thread.join(self._stop_timeout)

        if thread.is_alive():
            logger.warning("[%s] Network stop timed out", self._node_id)
            return

        self._thread = None

    def _reset(self) -> None:
        """
        Reset the local exchange state to its initial state.

        The caller must ensure that no exchange is active and that the current
        exchange state is no longer needed by any peer. This method performs no
        distributed synchronization.
        """
        self._exchange.reset()

    def next_round(self) -> int:
        """
        Advances the network to the next communication round.

        Returns:
            int: The new round identifier.
        """
        return self._exchange.advance_round()

    def neighborwise_exchange(
        self, state_map: dict[str, npt.NDArray]
    ) -> dict[str, npt.NDArray]:
        """
        Exchanges the given state map with all neighbor nodes.

        This method broadcasts the state map to all neighbors and then gathers their states.

        Args:
            state_map (dict[str, NDArray[np.float64]]): The state map to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state maps.
        """
        if state_map.keys() != self._neighbors.keys():
            missing = self._neighbors.keys() - state_map.keys()
            extra = state_map.keys() - self._neighbors.keys()
            raise ValueError(
                "State dictionary keys do not match neighbor names. "
                f"Missing: {missing}, Extra: {extra}."
            )

        self._exchange.begin()

        for j in self._neighbors:
            state = state_map[j]
            meta, payload = self._transform.encode(state)
            payload = np.ascontiguousarray(payload)
            dealer = self._dealers[j]
            dealer.send_multipart([meta, payload])

        neighbor_states: dict[str, npt.NDArray] = {}
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta, payload = dealer.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        self._exchange.end()

        return neighbor_states

    def exchange(self, state: npt.NDArray) -> dict[str, npt.NDArray]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        self._exchange.begin()

        meta_bytes, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j in self._neighbors:
            dealer = self._dealers[j]
            dealer.send_multipart([meta_bytes, payload])

        neighbor_states: dict[str, npt.NDArray] = {}
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta_bytes, payload = dealer.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta_bytes, payload)

        self._exchange.end()

        return neighbor_states

    def exchange_as_array(self, state: npt.NDArray) -> npt.NDArray:
        """
        Exchanges the given state with all neighbor nodes and returns their states as a stacked array.

        Note: Using this method will add an extra copy of the neighbor states in memory compared to the `exchange` method.
        This is because the states are first received as individual memory buffers and then stacked into a single array.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            NDArray[np.float64]: A 2D array where each row corresponds to a neighbor's received state array.
        """
        self._exchange.begin()

        meta_bytes, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j in self._neighbors:
            dealer = self._dealers[j]
            dealer.send_multipart([meta_bytes, payload])

        neighbor_states: list[npt.NDArray] = []
        for j in self._neighbors:
            dealer = self._dealers[j]
            meta_bytes, payload = dealer.recv_multipart()
            neighbor_states.append(self._transform.decode(meta_bytes, payload))

        self._exchange.end()

        return np.stack(neighbor_states, axis=0)

    def laplacian(self, state: npt.NDArray) -> npt.NDArray:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        neighbor_states = self.exchange(state)
        laplacian = state * len(neighbor_states)
        for neighbor_state in neighbor_states.values():
            laplacian -= neighbor_state

        return laplacian

    def weighted_mix(self, state: npt.NDArray) -> npt.NDArray:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        neighbor_states = self.exchange(state)
        nbrs = self._neighbors
        mix = (1.0 - sum(nbrs.values())) * state
        for j, neighbor_state in neighbor_states.items():
            mix += nbrs[j] * neighbor_state

        return mix

    def _run(self) -> None:
        """
        The main event loop for the network backend.
        """
        node = pyre.Pyre(self._node_id, ctx=self._context)
        node.set_header("namespace", self._namespace)
        pyre_socket = node.socket()

        router = self._context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.bind(self._endpoint)

        poller = zmq.Poller()
        poller.register(pyre_socket, zmq.POLLIN)
        poller.register(router, zmq.POLLIN)

        backend = NetworkBackend(
            node_id=self._node_id,
            namespace=self._namespace,
            exchange=self._exchange,
            node=node,
            router=router,
        )

        try:
            node.start()

            while self._running.is_set():
                events = dict(poller.poll(timeout=self._recv_timeout_ms))

                if pyre_socket in events:
                    event = node.recv()
                    backend.handle_pyre_event(event)

                if router in events:
                    frames = router.recv_multipart()
                    backend.handle_frontend_message(frames)

        except Exception:
            if self._running.is_set():
                logger.exception("[%s] Network event loop failed", self._node_id)

        finally:
            poller.unregister(pyre_socket)
            poller.unregister(router)

            try:
                node.stop()
            except Exception:
                logger.exception("[%s] Error stopping Pyre node", self._node_id)

            router.close()

            logger.info("[%s] Network thread stopped", self._node_id)
