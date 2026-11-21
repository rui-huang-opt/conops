"""
Three-process smoke test for automatic exchange IDs and early DATA.

Run with: python tests/test_auto_exchange_ids.py
Requires ConOps built with maturin develop --release.
"""

import socket
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from conops import Network

N_STEPS = 4
NEIGHBORS = {"1": {"2": 0.5}, "2": {"1": 0.5, "3": 0.5}, "3": {"2": 0.5}}


def run_node(node_id: str, endpoints: dict[str, str]) -> bool:
    neighbors = NEIGHBORS[node_id]
    local_endpoints = {peer: endpoints[peer] for peer in (node_id, *neighbors)}
    network = Network(
        node_id=node_id,
        neighbors=neighbors,
        peer_endpoints=local_endpoints,
    )
    try:
        network.start()
        network.wait_connected(timeout=15)

        for step in range(N_STEPS):
            # Hold one end of the chain back so the middle node buffers
            # a future message from its faster neighbor.
            if node_id == "3" and step == 0:
                time.sleep(0.1)

            state = np.full(128, int(node_id) * 100 + step, dtype=np.float64)

            # Both methods must share the same internal exchange ID counter.
            if step % 2 == 0:
                incoming = network.exchange(state, timeout=15)
            else:
                states = {peer: state for peer in neighbors}
                incoming = network.neighborwise_exchange(states, timeout=15)

            assert set(incoming) == set(neighbors)
            for peer_id, value in incoming.items():
                expected = np.full_like(state, int(peer_id) * 100 + step)
                np.testing.assert_array_equal(value, expected)

        return True
    finally:
        network.close()


def find_ports(n: int) -> list[int]:
    """Find available localhost TCP ports (released before child bind)."""
    sockets = [socket.socket() for _ in range(n)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def test_auto_exchange_ids() -> None:
    endpoints = {
        node_id: f"tcp://127.0.0.1:{port}"
        for node_id, port in zip(NEIGHBORS, find_ports(len(NEIGHBORS)))
    }

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(NEIGHBORS), mp_context=ctx) as executor:
        tasks = [executor.submit(run_node, node_id, endpoints) for node_id in NEIGHBORS]
        assert all(task.result(timeout=60) for task in tasks)


if __name__ == "__main__":
    test_auto_exchange_ids()
