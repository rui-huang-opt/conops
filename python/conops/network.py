"""
NumPy-facing ConOps network, backed by the Rust CoreNetwork.

The Python layer owns serialization and numerical operations; Rust owns
transport, neighbor discovery, exchange matching and connection recovery.
"""

import struct
import time
from typing import KeysView, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from ._core import CoreNetwork
from .transform import Identity, Transform

_META_LENGTH = struct.Struct("!I")


class Network:
    """Synchronous neighbor exchanges of NumPy arrays over the Rust backend.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node within the group.
    neighbors : Mapping[str, float]
        Fixed neighbor IDs and their weights, as used by ``weighted_mix``.
    port : int, optional
        Local TCP listening port. Defaults to 0, allowing the operating system
        to automatically assign an available port.
    group_id : str
        mDNS discovery group used to isolate independent experiments.
    transform : Transform | None
        Unchanged ConOps transformation protocol. Defaults to ``Identity``.

    Notes
    -----
    Every exchange operation requires an explicit, monotonically increasing
    ``exchange_id``. Network does not manage algorithm rounds or iteration IDs.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: Mapping[str, float],
        *,
        port: int = 0,
        group_id: str = "default",
        transform: Transform | None = None,
    ) -> None:
        self._node_id = node_id
        self._neighbors = dict(neighbors)
        self._transform = transform if transform is not None else Identity()
        self._core = CoreNetwork(
            node_id,
            port,
            neighbors=list(self._neighbors),
            group_id=group_id,
        )

        self._profile_pack = 0.0
        self._profile_core = 0.0
        self._profile_unpack = 0.0
        self._profile_count = 0

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
        """Manually connect to a neighbor (usually unnecessary with mDNS)."""
        self._core.connect(peer_id, address)

    def peers(self) -> list[str]:
        """Return currently connected peer IDs."""
        return self._core.peers()

    def close(self) -> None:
        self._core.close()

    def _pack(self, state: NDArray) -> bytes:
        meta, payload = self._transform.encode(state)
        if not isinstance(meta, bytes):
            raise TypeError("Transform.encode() must return metadata as bytes")
        if not isinstance(payload, np.ndarray):
            raise TypeError("Transform.encode() must return a NumPy array as payload")
        if payload.dtype.hasobject:
            raise TypeError("Cannot send NumPy arrays containing Python objects")
        if len(meta) > 0xFFFFFFFF:
            raise ValueError("Transform metadata is too large")

        payload = np.ascontiguousarray(payload)

        return b"".join((
            _META_LENGTH.pack(len(meta)),
            meta,
            payload.data.cast("B"),
        ))

    def _unpack(self, frame: bytes) -> NDArray:
        if len(frame) < _META_LENGTH.size:
            raise ValueError("Invalid state frame: missing metadata length")

        (meta_length,) = _META_LENGTH.unpack_from(frame)
        offset = _META_LENGTH.size + meta_length

        if offset > len(frame):
            raise ValueError("Invalid state frame: truncated metadata")

        meta = frame[_META_LENGTH.size:offset]
        payload = cast(bytes, memoryview(frame)[offset:])

        return self._transform.decode(meta, payload)

    def exchange(
        self,
        exchange_id: int,
        state: NDArray,
        *,
        timeout: float | None = None,
    ) -> dict[str, NDArray]:
        t0 = time.perf_counter()

        frame = self._pack(state)

        t1 = time.perf_counter()

        incoming = self._core.exchange(
            exchange_id, frame, timeout=timeout
        )

        t2 = time.perf_counter()

        result = {
            peer_id: self._unpack(data)
            for peer_id, data in incoming.items()
        }

        t3 = time.perf_counter()

        self._profile_pack += t1 - t0
        self._profile_core += t2 - t1
        self._profile_unpack += t3 - t2
        self._profile_count += 1

        return result

    def exchange_as_array(
        self,
        exchange_id: int,
        state: NDArray,
        *,
        timeout: float | None = None,
    ) -> NDArray:
        """Return neighbors' states stacked in the order of ``neighbors``."""
        received = self.exchange(exchange_id, state, timeout=timeout)
        if not self._neighbors:
            return np.empty((0, *state.shape), dtype=state.dtype)
        return np.stack([received[peer_id] for peer_id in self._neighbors], axis=0)

    def laplacian(
        self,
        exchange_id: int,
        state: NDArray,
        *,
        timeout: float | None = None,
    ) -> NDArray:
        """Compute degree * state - sum(neighbor states)."""
        received = self.exchange(exchange_id, state, timeout=timeout)
        result = len(self._neighbors) * state
        for neighbor_state in received.values():
            result = result - neighbor_state
        return result

    def weighted_mix(
        self,
        exchange_id: int,
        state: NDArray,
        *,
        timeout: float | None = None,
    ) -> NDArray:
        """Compute (1 - sum(weights)) * state + sum(weight * neighbor state)."""
        received = self.exchange(exchange_id, state, timeout=timeout)
        result = (1.0 - sum(self._neighbors.values())) * state
        for peer_id, neighbor_state in received.items():
            result = result + self._neighbors[peer_id] * neighbor_state
        return result
