"""
Test ConOps TCP reconnect and unacknowledged DATA replay.

Two worker processes communicate through a local TCP proxy. At exchange ID 3,
the proxy consumes (but does not forward) both DATA frames, closes the TCP
connection, and then permits the Rust backend to reconnect and replay them.

Run: python test_reconnect.py
No root privileges or Rust source changes are needed.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import socket
import struct
import threading
import time

import numpy as np

from conops import Network

N_EXCHANGES = 7
CUT_EXCHANGE = 3
EXCHANGE_TIMEOUT = 20.0


def available_ports() -> tuple[int, int]:
    """Pick two distinct local ports for the worker TCP listeners."""
    reservations = [socket.socket(), socket.socket()]
    try:
        for sock in reservations:
            sock.bind(("127.0.0.1", 0))
        return tuple(sock.getsockname()[1] for sock in reservations)
    finally:
        for sock in reservations:
            sock.close()


def recv_exact(sock: socket.socket, length: int) -> bytes:
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    while offset < length:
        count = sock.recv_into(view[offset:])
        if count == 0:
            raise EOFError("TCP connection closed")
        offset += count
    return bytes(data)


class FaultProxy:
    """TCP forwarding proxy that drops both DATA frames of one exchange."""

    def __init__(self, target_port: int, cut_exchange: int):
        self.target_port = target_port
        self.cut_exchange = cut_exchange
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.2)
        self.port = self.listener.getsockname()[1]

        self._stop = threading.Event()
        self._cut = threading.Event()
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._connections: list[tuple[socket.socket, socket.socket]] = []
        self.connection_count = 0
        self._accept_thread = threading.Thread(target=self._accept, daemon=True)

    def __enter__(self):
        self._accept_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self.listener.close()
        with self._lock:
            pairs = list(self._connections)
        for pair in pairs:
            self._close_pair(pair)
        self._accept_thread.join(timeout=2)

    @property
    def did_cut(self) -> bool:
        return self._cut.is_set()

    def _close_pair(self, pair: tuple[socket.socket, socket.socket]) -> None:
        for sock in pair:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return

            try:
                server = socket.create_connection(
                    ("127.0.0.1", self.target_port), timeout=5
                )
                client.settimeout(None)
                server.settimeout(None)
            except OSError as exc:
                print(f"[proxy] Cannot connect to node 2: {exc}", flush=True)
                client.close()
                continue

            pair = (client, server)
            with self._lock:
                self.connection_count += 1
                number = self.connection_count
                self._connections.append(pair)
            print(f"[proxy] TCP connection #{number} established", flush=True)

            threading.Thread(
                target=self._relay, args=(client, server, "1->2", pair), daemon=True
            ).start()
            threading.Thread(
                target=self._relay, args=(server, client, "2->1", pair), daemon=True
            ).start()

    def _relay(
        self,
        source: socket.socket,
        destination: socket.socket,
        direction: str,
        pair: tuple[socket.socket, socket.socket],
    ) -> None:
        try:
            while not self._stop.is_set():
                prefix = recv_exact(source, 4)
                size = struct.unpack("!I", prefix)[0]
                frame = recv_exact(source, size)

                # DATA frame = type: u8 (0), exchange_id: u64 (big endian), payload.
                target_data = (
                    not self._cut.is_set()
                    and len(frame) >= 9
                    and frame[0] == 0
                    and struct.unpack_from("!Q", frame, 1)[0] == self.cut_exchange
                )
                if target_data:
                    with self._lock:
                        self._seen.add(direction)
                        print(
                            f"[proxy] Intercepted exchange {self.cut_exchange} ({direction})",
                            flush=True,
                        )
                        if len(self._seen) == 2 and not self._cut.is_set():
                            print(
                                "[proxy] Dropping both DATA frames and disconnecting TCP",
                                flush=True,
                            )
                            self._cut.set()
                            self._close_pair(pair)
                    # The first direction must also drop its frame; neither is forwarded.
                    self._cut.wait(timeout=EXCHANGE_TIMEOUT)
                    return

                destination.sendall(prefix)
                destination.sendall(frame)
        except (OSError, EOFError):
            pass
        finally:
            self._close_pair(pair)


def worker(node_id: str, endpoints: dict[str, str]) -> tuple[str, list[float]]:
    peer_id = "2" if node_id == "1" else "1"
    network = Network(
        node_id=node_id,
        neighbors={peer_id: 1.0},
        group_id="reconnect-smoke-test",
        peer_endpoints=endpoints,
    )

    elapsed: list[float] = []
    try:
        network.start()
        network.wait_connected(timeout=20)

        for exchange_id in range(N_EXCHANGES):
            state = np.array([int(node_id), exchange_id], dtype=np.int64)
            start = time.perf_counter()
            received = network.exchange(
                exchange_id=exchange_id,
                state=state,
                timeout=EXCHANGE_TIMEOUT,
            )
            elapsed.append(time.perf_counter() - start)

            expected = np.array([int(peer_id), exchange_id], dtype=np.int64)
            np.testing.assert_array_equal(received[peer_id], expected)

        return node_id, elapsed
    finally:
        network.close()


def main() -> None:
    node1_port, node2_port = available_ports()
    addr1 = f"tcp://127.0.0.1:{node1_port}"
    addr2 = f"tcp://127.0.0.1:{node2_port}"

    with FaultProxy(node2_port, CUT_EXCHANGE) as proxy:
        # Only node 1 actively connects (smaller ID). Route that connection via proxy.
        # Node 2 listens on its actual port and responds through the same proxy.
        endpoints1 = {"1": addr1, "2": f"tcp://127.0.0.1:{proxy.port}"}
        endpoints2 = {"1": addr1, "2": addr2}

        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=get_context("spawn"),
        ) as executor:
            futures = [
                executor.submit(worker, "1", endpoints1),
                executor.submit(worker, "2", endpoints2),
            ]
            results = [future.result(timeout=90) for future in futures]

        print("\nExchange latencies (seconds):")
        for node_id, times in results:
            print(f"  Node {node_id}: {[round(t, 3) for t in times]}")

        if not proxy.did_cut or proxy.connection_count < 2:
            raise AssertionError(
                "The proxy did not complete the disconnect/reconnect test"
            )

        print(
            "\nPASS: TCP was interrupted; nodes reconnected and all exchanges matched."
        )
        print("The interrupted exchange succeeded without restarting either process.")


if __name__ == "__main__":
    main()
