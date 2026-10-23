"""Test neighbor-specific DATA replay after a TCP disconnection.

Run from the project root after installing the Rust extension:
    python tests/test_neighborwise_reconnect.py

Three nodes form a chain: 1 -- 2 -- 3. A local proxy sits ONLY on the
2 -- 3 connection. At key=(2, 'lambda'), it drops DATA in both directions,
closes that TCP session, and forwards the later reconnect/replay traffic.
No root access, mDNS, external services, or Rust source changes are required.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import socket
import struct
import threading

import msgspec
import numpy as np

from conops import Network

NODE_IDS = ("1", "2", "3")
NEIGHBORS = {"1": ("2",), "2": ("1", "3"), "3": ("2",)}
CUT_KEY = (2, "lambda")
N_EXCHANGES = 6
EXCHANGE_TIMEOUT = 25.0


def available_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def recv_exact(sock: socket.socket, count: int) -> bytes:
    result = bytearray(count)
    view = memoryview(result)
    offset = 0
    while offset < count:
        size = sock.recv_into(view[offset:])
        if size == 0:
            raise EOFError("TCP connection closed before frame was complete")
        offset += size
    return bytes(result)


class FaultProxy:
    """Drop one keyed DATA frame per direction on the 2 -- 3 connection."""

    def __init__(self, target_port: int, cut_key: tuple[int, str]):
        self.target_port = target_port
        self.cut_key = msgspec.msgpack.encode(cut_key)
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
        return self._cut.is_set() and self._seen == {"2->3", "3->2"}

    @staticmethod
    def _close_pair(pair: tuple[socket.socket, socket.socket]) -> None:
        for sock in pair:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _is_cut_data(self, frame: bytes) -> bool:
        # DATA: type (u8=0) | key length (u16 BE) | encoded key | payload.
        # HELLO and ACK frames must be forwarded without modification.
        if len(frame) < 3 or frame[0] != 0:
            return False
        key_length = struct.unpack_from("!H", frame, 1)[0]
        return (
            len(frame) >= 3 + key_length and frame[3 : 3 + key_length] == self.cut_key
        )

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
                print(f"[proxy] Connection to node 3 failed: {exc}", flush=True)
                client.close()
                continue

            pair = (client, server)
            with self._lock:
                self.connection_count += 1
                count = self.connection_count
                self._connections.append(pair)
            print(f"[proxy] TCP session #{count}", flush=True)

            threading.Thread(
                target=self._relay,
                args=(client, server, "2->3", pair),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._relay,
                args=(server, client, "3->2", pair),
                daemon=True,
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
                frame = recv_exact(source, struct.unpack("!I", prefix)[0])

                if not self._cut.is_set() and self._is_cut_data(frame):
                    with self._lock:
                        self._seen.add(direction)
                        print(
                            f"[proxy] Dropped {CUT_KEY!r} DATA ({direction})",
                            flush=True,
                        )
                        if len(self._seen) == 2 and not self._cut.is_set():
                            self._cut.set()
                            self._close_pair(pair)
                    # Keep the first direction blocked until both frames were dropped.
                    self._cut.wait(timeout=EXCHANGE_TIMEOUT)
                    return

                destination.sendall(prefix)
                destination.sendall(frame)
        except (OSError, EOFError):
            pass
        finally:
            self._close_pair(pair)


def make_data(sender: str, destination: str, step: int) -> np.ndarray:
    # The destination field makes an accidental broadcast detectable.
    return np.array([int(sender), int(destination), step], dtype=np.int64)


def worker(node_id: str, endpoints: dict[str, str]) -> str:
    neighbors = {peer: 0.5 for peer in NEIGHBORS[node_id]}
    # Static endpoints must cover only self + this node's actual neighbors.
    local_endpoints = {peer: endpoints[peer] for peer in (node_id, *neighbors)}
    network = Network(node_id, neighbors, peer_endpoints=local_endpoints)
    try:
        network.start()
        network.wait_connected(timeout=20)

        for step in range(N_EXCHANGES):
            to_neighbors = {peer: make_data(node_id, peer, step) for peer in neighbors}
            received = network.neighborwise_exchange(
                to_neighbors, key=(step, "lambda"), timeout=EXCHANGE_TIMEOUT
            )
            assert set(received) == set(neighbors)
            for peer, actual in received.items():
                np.testing.assert_array_equal(actual, make_data(peer, node_id, step))

        return node_id
    finally:
        network.close()


def test_neighborwise_reconnect() -> None:
    ports = available_ports(len(NODE_IDS))
    endpoints = {node: f"tcp://127.0.0.1:{port}" for node, port in zip(NODE_IDS, ports)}

    with FaultProxy(ports[2], CUT_KEY) as proxy:
        # Smaller node IDs initiate TCP connections. Node 2 initiates to 3,
        # so only its configured endpoint for node 3 goes through the proxy.
        node2_endpoints = dict(endpoints)
        node2_endpoints["3"] = f"tcp://127.0.0.1:{proxy.port}"

        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(NODE_IDS), mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    worker,
                    node,
                    node2_endpoints if node == "2" else endpoints,
                )
                for node in NODE_IDS
            ]
            results = [future.result(timeout=90) for future in futures]

        assert set(results) == set(NODE_IDS)
        assert proxy.did_cut, "The proxy did not drop DATA in both directions"
        assert proxy.connection_count >= 2, "The 2 -- 3 link did not reconnect"


if __name__ == "__main__":
    test_neighborwise_reconnect()
