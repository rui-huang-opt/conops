import struct
from collections.abc import KeysView, Mapping
from urllib.parse import urlsplit

import msgspec
import numpy as np
from numpy.typing import NDArray

from ._core import CoreNetwork
from .transform import Identity, Transform

_META_LENGTH = struct.Struct("!I")

ExchangeKey = int | str | tuple[int, str]


def _encode_exchange_key(key: ExchangeKey) -> bytes:
    """Encode a semantic key deterministically (never use Python hash())."""
    if type(key) not in (int, str) and not (
        type(key) is tuple
        and len(key) == 2
        and type(key[0]) is int
        and type(key[1]) is str
    ):
        raise TypeError("key must be int, str, or tuple[int, str]")

    encoded = msgspec.msgpack.encode(key)
    if len(encoded) > 256:
        raise ValueError("Encoded exchange key exceeds 256 bytes")

    return encoded


def _parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    """Return a TCP host:port address suitable for Tokio, plus its port."""
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        port = parsed.port
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid TCP endpoint: {endpoint!r}") from exc

    if (
        parsed.scheme != "tcp"
        or not host
        or port is None
        or port == 0
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"Invalid TCP endpoint: {endpoint!r}")

    if ":" in host:  # IPv6 literal; preserve brackets for Tokio's address parser.
        host = f"[{host}]"
    return f"{host}:{port}", port


class Network:
    """Synchronous neighbor exchanges of NumPy arrays over the Rust backend.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node within the group.

    neighbors : Mapping[str, float]
        Fixed neighbor IDs and their associated weights.

    group_id : str, optional
        Group identifier for mDNS discovery. Defaults to ``"default"``.

    port : int, optional
        Local TCP listening port. Defaults to ``0`` for automatic assignment.
        With ``peer_endpoints``, the local port is taken from this node's endpoint.

    transform : Transform | None, optional
        Transformation applied to exchanged states. Defaults to ``Identity``.

    peer_endpoints : Mapping[str, str] | None, optional
        TCP endpoints for this node and every neighbor, e.g.
        ``{"0": "tcp://node-0:5000", "1": "tcp://node-1:5000"}``.
        When provided, disables mDNS discovery and uses the configured
        addresses for initial connections and reconnection attempts.
        When ``None``, neighbors are discovered through mDNS.

    max_message_size : int, optional
        Maximum encoded DATA frame size in bytes. Defaults to 64 MiB.

    max_unacked_bytes : int, optional
        Maximum unacknowledged DATA retained per neighbor in bytes.
        Defaults to 128 MiB.

    max_early_bytes : int, optional
        Maximum future-exchange DATA buffered in total in bytes.
        Defaults to 128 MiB.

    Notes
    -----
    The cache byte limits must each be at least ``max_message_size``.
    Every logical exchange needs a unique ``key``, used consistently by
    neighboring nodes. Keys need not be ordered, but neighbors must execute
    compatible sequences of exchanges. Key types: ``int``, ``str``, or
    ``tuple[int, str]``. Encoded keys are limited to 256 bytes.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: Mapping[str, float],
        *,
        group_id: str = "default",
        port: int = 0,
        transform: Transform | None = None,
        peer_endpoints: Mapping[str, str] | None = None,
        max_message_size: int = 64 * 1024 * 1024,
        max_unacked_bytes: int = 128 * 1024 * 1024,
        max_early_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self._node_id = node_id
        self._neighbors = dict(neighbors)
        self._transform = transform if transform is not None else Identity()

        peer_endpoints_ = None
        if peer_endpoints is not None:
            port, peer_endpoints_ = self._resolve_peer_endpoints(peer_endpoints, port)

        self._core = CoreNetwork(
            node_id,
            port,
            neighbors=list(self._neighbors),
            group_id=group_id,
            peer_endpoints=peer_endpoints_,
            max_message_size=max_message_size,
            max_unacked_bytes=max_unacked_bytes,
            max_early_bytes=max_early_bytes,
        )

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def neighbors(self) -> KeysView[str]:
        return self._neighbors.keys()

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    def start(self) -> None:
        self._core.start()

    def wait_connected(self, timeout: float | None = None) -> None:
        """Wait for the configured, fixed set of neighbors to connect."""
        self._core.wait_connected(timeout=timeout)

    def connect(self, peer_id: str, address: str) -> None:
        """Manually connect to a neighbor (usually unnecessary in either mode)."""
        self._core.connect(peer_id, address)

    def peers(self) -> list[str]:
        """Return currently connected peer IDs."""
        return self._core.peers()

    def close(self, timeout: float = 5.0) -> None:
        try:
            if timeout != 0:
                self._core.drain(timeout=timeout)
        finally:
            self._core.close()

    def _resolve_peer_endpoints(
        self,
        peer_endpoints: Mapping[str, str],
        port: int,
    ) -> tuple[int, dict[str, str]]:
        expected = set(self._neighbors) | {self._node_id}
        provided = set(peer_endpoints)

        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            raise ValueError(
                "peer_endpoints must cover this node and all neighbors; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        parsed = {
            peer_id: _parse_tcp_endpoint(peer_endpoints[peer_id])
            for peer_id in expected
        }

        local_port = parsed[self._node_id][1]
        if port not in (0, local_port):
            raise ValueError(f"Port mismatch: {port} != {local_port}")

        addresses = {peer_id: parsed[peer_id][0] for peer_id in self._neighbors}

        return local_port, addresses

    def _prepare(self, state: NDArray) -> tuple[bytes, NDArray[np.uint8]]:
        meta, payload = self._transform.encode(state)
        if not isinstance(meta, bytes):
            raise TypeError("Transform.encode() must return metadata as bytes")
        if not isinstance(payload, np.ndarray):
            raise TypeError("Transform.encode() must return a NumPy array as payload")
        if payload.dtype.hasobject:
            raise TypeError("Cannot send NumPy arrays containing Python objects")
        if len(meta) > 0xFFFFFFFF:
            raise ValueError("Transform metadata is too large")

        # No Python bytes allocation: only make a copy when the input is strided.
        # Flatten first, so scalar and multidimensional arrays work too.
        contiguous = np.ascontiguousarray(payload)
        return meta, contiguous.reshape(-1).view(np.uint8)

    def _unpack(self, frame: bytes | memoryview) -> NDArray:
        view = memoryview(frame)
        if len(view) < _META_LENGTH.size:
            raise ValueError("Invalid state frame: missing metadata length")
        (meta_length,) = _META_LENGTH.unpack_from(view)
        offset = _META_LENGTH.size + meta_length
        if offset > len(view):
            raise ValueError("Invalid state frame: truncated metadata")

        meta = bytes(view[_META_LENGTH.size : offset])
        payload = view[offset:]
        return self._transform.decode(meta, payload)

    def exchange(
        self,
        state: NDArray,
        *,
        key: ExchangeKey,
        timeout: float | None = None,
    ) -> dict[str, NDArray]:
        """Exchange ``state`` with all fixed neighbors using a semantic key."""
        encoded_key = _encode_exchange_key(key)
        meta, payload = self._prepare(state)
        incoming = self._core.exchange_array(
            encoded_key,
            meta,
            payload,
            timeout=timeout,
        )
        return {peer_id: self._unpack(data) for peer_id, data in incoming.items()}

    def neighborwise_exchange(
        self,
        state_map: Mapping[str, NDArray],
        *,
        key: ExchangeKey,
        timeout: float | None = None,
    ) -> dict[str, NDArray]:
        """Exchange a different state with each neighbor under the same key."""
        expected = set(self._neighbors)
        provided = set(state_map)
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            raise ValueError(
                "state_map must cover exactly the configured neighbors; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        encoded_key = _encode_exchange_key(key)
        prepared = {
            peer_id: self._prepare(state_map[peer_id]) for peer_id in self._neighbors
        }
        incoming = self._core.neighborwise_exchange_array(
            key=encoded_key,
            states=prepared,
            timeout=timeout,
        )
        return {peer_id: self._unpack(data) for peer_id, data in incoming.items()}

    def laplacian(
        self,
        state: NDArray,
        *,
        key: ExchangeKey,
        timeout: float | None = None,
    ) -> NDArray:
        """Compute degree * state - sum(neighbor states)."""
        received = self.exchange(state, key=key, timeout=timeout)
        result = len(self._neighbors) * state
        for neighbor_state in received.values():
            result -= neighbor_state
        return result

    def weighted_mix(
        self,
        state: NDArray,
        *,
        key: ExchangeKey,
        timeout: float | None = None,
    ) -> NDArray:
        """Compute (1 - sum(weights)) * state + sum(weight * neighbor state)."""
        received = self.exchange(state, key=key, timeout=timeout)
        result = (1.0 - sum(self._neighbors.values())) * state
        for peer_id, neighbor_state in received.items():
            result += self._neighbors[peer_id] * neighbor_state
        return result
