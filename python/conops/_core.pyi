import numpy as np
from numpy.typing import NDArray

class CoreNetwork:
    def __init__(
        self,
        node_id: str,
        port: int = 0,
        neighbors: list[str] | None = None,
        group_id: str = "default",
        peer_endpoints: dict[str, str] | None = None,
        max_message_size: int = 64 * 1024 * 1024,
        max_unacked_bytes: int = 128 * 1024 * 1024,
        max_early_bytes: int = 128 * 1024 * 1024,
    ) -> None: ...
    def start(self) -> None: ...
    def connect(self, peer_id: str, address: str) -> None: ...
    def peers(self) -> list[str]: ...
    def wait_connected(self, timeout: float | None = None) -> None: ...
    def exchange_array(
        self,
        key: bytes,
        meta: bytes,
        payload: NDArray[np.uint8],
        timeout: float | None = None,
        *,
        copy: bool = True,
    ) -> dict[str, memoryview]: ...
    def neighborwise_exchange_array(
        self,
        key: bytes,
        states: dict[str, tuple[bytes, NDArray[np.uint8]]],
        timeout: float | None = None,
        *,
        copy: bool = True,
    ) -> dict[str, memoryview]: ...
    def drain(self, timeout: float = 5.0) -> None: ...
    def close(self) -> None: ...
