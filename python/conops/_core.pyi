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
        exchange_id: int,
        meta: bytes,
        payload: NDArray[np.uint8],
        timeout: float | None = None,
    ) -> dict[str, memoryview]: ...
    def close(self) -> None: ...
