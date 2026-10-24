"""
Smoke tests for ConOps shutdown on persistent loopback connections.

Run with: python tests/test_graceful_close.py
      or: python -m pytest -v tests/test_graceful_close.py
Requires the ConOps extension built with maturin develop --release.

The multi-process tests deliberately add no barrier or fixed sleep: each node
closes immediately after its final exchange. The scripted TCP peer separately
checks that Rust close() waits for a deliberately withheld DATA acknowledgment.
"""

import socket
import struct
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError
from multiprocessing import get_context

import numpy as np

from conops import LiteNetwork, Network

NODES = ["1", "2", "3"]
NEIGHBORS = {"1": {"2": 0.5}, "2": {"1": 0.5, "3": 0.5}, "3": {"2": 0.5}}
N_EXCHANGES = 8
TIMEOUT = 15.0


def available_ports(n: int) -> list[int]:
    """Find available localhost TCP ports (released before child bind)."""
    sockets = [socket.socket() for _ in range(n)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def run_node(backend: str, node_id: str, endpoints: dict[str, str]) -> bool:
    neighbors = NEIGHBORS[node_id]
    local_endpoints = {peer: endpoints[peer] for peer in (node_id, *neighbors)}

    if backend == "Rust":
        network = Network(
            node_id=node_id,
            neighbors=neighbors,
            peer_endpoints=local_endpoints,
        )
        started = False
        try:
            network.start()
            started = True
            network.wait_connected(timeout=TIMEOUT)

            for k in range(N_EXCHANGES):
                state = np.full(4096, 100 * int(node_id) + k, dtype=np.float64)
                received = network.exchange(state, key=k, timeout=TIMEOUT)
                assert set(received) == set(neighbors)
                for peer_id, value in received.items():
                    np.testing.assert_array_equal(
                        value, np.full_like(state, 100 * int(peer_id) + k)
                    )
            return True
        finally:
            # No inter-node synchronization before closing. This is the
            # regression case for closing while another node is still active.
            network.close(timeout=10.0 if started else 0.0)

    if backend == "Lite":
        network = LiteNetwork(
            node_id=node_id,
            neighbors=neighbors,
            peer_endpoints=local_endpoints,
        )
        try:
            for k in range(N_EXCHANGES):
                state = np.full(4096, 100 * int(node_id) + k, dtype=np.float64)
                received = network.exchange(state, key=k)
                assert set(received) == set(neighbors)
                for peer_id, value in received.items():
                    np.testing.assert_array_equal(
                        value, np.full_like(state, 100 * int(peer_id) + k)
                    )
            return True
        finally:
            # Lite has no application-level ACK guarantee; exercise its
            # normal best-effort closing path without assuming one.
            network.close()

    raise ValueError(f"Unknown backend: {backend}")


def run_processes(backend: str) -> None:
    endpoints = {
        node_id: f"tcp://127.0.0.1:{port}"
        for node_id, port in zip(NODES, available_ports(len(NODES)))
    }
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(NODES), mp_context=ctx) as executor:
        tasks = [executor.submit(run_node, backend, node_id, endpoints) for node_id in NODES]
        assert all(task.result(timeout=90) for task in tasks)


def test_rust_close_after_exchange() -> None:
    run_processes("Rust")


def test_lite_close_after_exchange() -> None:
    run_processes("Lite")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray(n)
    view = memoryview(data)
    offset = 0
    while offset < n:
        received = sock.recv_into(view[offset:])
        if received == 0:
            raise EOFError("TCP connection closed")
        offset += received
    return bytes(data)


def recv_frame(sock: socket.socket) -> bytes:
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if size > 1024 * 1024:
        raise AssertionError(f"Unexpected TCP frame size: {size}")
    return recv_exact(sock, size)


def send_frame(sock: socket.socket, frame: bytes) -> None:
    sock.sendall(struct.pack("!I", len(frame)) + frame)


def test_rust_close_waits_for_ack() -> None:
    """Closing must wait for the peer's DATA ACK, not just exchange return."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(TIMEOUT)
    port1 = available_ports(1)[0]
    port2 = listener.getsockname()[1]
    endpoints = {
        "1": f"tcp://127.0.0.1:{port1}",
        "2": f"tcp://127.0.0.1:{port2}",
    }

    got_local_ack = threading.Event()
    release_ack = threading.Event()
    finish_peer = threading.Event()
    close_started = threading.Event()

    def fake_peer() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(TIMEOUT)
                assert recv_frame(conn) == b"HELLO2 1"
                send_frame(conn, b"HELLO2 2")

                data = recv_frame(conn)
                assert len(data) >= 7 and data[0] == 0  # DATA frame
                key_size = struct.unpack_from("!H", data, 1)[0]
                key = data[3 : 3 + key_size]
                body_start = 3 + key_size
                meta_size = struct.unpack_from("!I", data, body_start)[0]
                end_of_meta = body_start + 4 + meta_size
                assert end_of_meta <= len(data)

                # Reply using the original metadata and a different float64.
                response = data[:end_of_meta] + np.array([222.0]).tobytes()
                send_frame(conn, response)

                ack = b"\x01" + struct.pack("!H", key_size) + key
                assert recv_frame(conn) == ack  # Our response was acknowledged.
                got_local_ack.set()

                # Deliberately hold back the ACK for the original DATA.
                if not release_ack.wait(TIMEOUT):
                    raise AssertionError("Test did not release the withheld ACK")
                send_frame(conn, ack)
                if not finish_peer.wait(TIMEOUT):
                    raise AssertionError("Test did not finish closing")
        finally:
            listener.close()

    network = Network(
        node_id="1",
        neighbors={"2": 1.0},
        peer_endpoints=endpoints,
    )
    close_future = None
    with ThreadPoolExecutor(max_workers=2) as executor:
        peer_future = executor.submit(fake_peer)
        try:
            network.start()
            network.wait_connected(timeout=TIMEOUT)
            received = network.exchange(
                np.array([111.0]), key="final", timeout=TIMEOUT
            )
            np.testing.assert_array_equal(received["2"], [222.0])
            assert got_local_ack.wait(TIMEOUT)

            def close_network() -> None:
                close_started.set()
                network.close(timeout=10.0)

            close_future = executor.submit(close_network)
            assert close_started.wait(TIMEOUT)
            # An assertion timeout is intentional fault injection, NOT a
            # synchronization barrier or an arbitrary benchmark sleep.
            try:
                close_future.result(timeout=0.2)
            except TimeoutError:
                pass
            else:
                raise AssertionError("Network.close() returned before the DATA ACK")

            release_ack.set()
            close_future.result(timeout=TIMEOUT)
        finally:
            release_ack.set()
            finish_peer.set()
            if close_future is None:
                network.close(timeout=0)
            peer_future.result(timeout=TIMEOUT)


def test_close_without_start() -> None:
    """The immediate close path must work before the worker is started."""
    network = Network(node_id="1", neighbors={})
    network.close(timeout=0)


if __name__ == "__main__":
    test_rust_close_after_exchange()
    test_lite_close_after_exchange()
    test_rust_close_waits_for_ack()
    test_close_without_start()
