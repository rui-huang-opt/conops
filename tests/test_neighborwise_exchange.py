"""
Integration test for per-neighbor DATA frames and ordinary exchanges.

Run after building the Rust extension:
    python tests/test_neighborwise_exchange.py
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import socket

import numpy as np

from conops import Network

NODE_IDS = ("1", "2", "3")


def available_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def worker(node_id: str, endpoints: dict[str, str]) -> str:
    neighbors = {peer_id: 0.25 for peer_id in NODE_IDS if peer_id != node_id}
    network = Network(node_id, neighbors, peer_endpoints=endpoints)
    try:
        network.start()
        network.wait_connected(timeout=20)

        for k in range(5):
            # Each neighbor receives a different array from this node.
            state_map = {
                peer_id: np.array([int(node_id), int(peer_id), k], dtype=np.float64)
                for peer_id in neighbors
            }
            incoming = network.neighborwise_exchange(
                state_map,
                key=(k, "lambda"),
                timeout=15,
            )
            assert set(incoming) == set(neighbors)
            for peer_id, state in incoming.items():
                expected = np.array([int(peer_id), int(node_id), k], dtype=np.float64)
                np.testing.assert_array_equal(state, expected)

            # The existing broadcast path still works between neighborwise calls.
            incoming = network.exchange(
                np.array([int(node_id), k], dtype=np.int64),
                key=(k, "broadcast"),
                timeout=15,
            )
            for peer_id, state in incoming.items():
                np.testing.assert_array_equal(
                    actual=state,
                    desired=np.array([int(peer_id), k], dtype=np.int64),
                )

        # Incorrect neighbor maps must fail before any network message is sent.
        try:
            network.neighborwise_exchange({}, key="invalid")
        except ValueError:
            pass
        else:
            raise AssertionError("Missing neighbor states were not rejected")

        return node_id
    finally:
        network.close()


def test_neighborwise_exchange() -> None:
    ports = available_ports(len(NODE_IDS))
    endpoints = {
        node_id: f"tcp://127.0.0.1:{port}" for node_id, port in zip(NODE_IDS, ports)
    }

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(NODE_IDS), mp_context=ctx) as executor:
        futures = [
            executor.submit(worker, node_id, endpoints)
            for node_id in NODE_IDS
        ]
        results = [future.result(timeout=90) for future in futures]

    assert set(results) == set(NODE_IDS)


if __name__ == "__main__":
    test_neighborwise_exchange()
