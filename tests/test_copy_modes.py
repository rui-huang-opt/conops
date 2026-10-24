"""Three-process smoke test for Rust exchange(copy=True/False).

Run with: python tests/test_copy_modes.py
      or: python -m pytest -v tests/test_copy_modes.py
Requires the ConOps extension built with maturin develop --release.

Covers broadcast, neighborwise, laplacian, and weighted_mix. Sent buffers are
never modified in place; copy=False requires that the actual payload storage
(including any aliases or transform buffers) stays unchanged while in use.
"""

import gc
import socket
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from conops import Network

NODES = ["1", "2", "3"]
NEIGHBORS = {
    node_id: {peer_id: 0.25 for peer_id in NODES if peer_id != node_id}
    for node_id in NODES
}
TIMEOUT = 15.0
N_STATE = 16_384


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


def run_node(node_id: str, endpoints: dict[str, str]) -> bool:
    neighbors = NEIGHBORS[node_id]
    network = Network(
        node_id=node_id,
        neighbors=neighbors,
        peer_endpoints=endpoints,
    )
    started = False
    try:
        network.start()
        started = True
        network.wait_connected(timeout=TIMEOUT)

        for copy in (True, False):
            for step in range(3):
                # Broadcast: the same payload goes to two neighbors.
                state = np.full(N_STATE, 100 * int(node_id) + step, dtype=np.float64)
                received = network.exchange(
                    state,
                    key=f"broadcast-{copy}-{step}",
                    timeout=TIMEOUT,
                    copy=copy,
                )
                assert set(received) == set(neighbors)
                for peer_id, value in received.items():
                    np.testing.assert_array_equal(
                        value, np.full_like(state, 100 * int(peer_id) + step)
                    )

                # Replace the name with a NEW array; do not mutate the
                # original storage, which may still be retained for replay.
                state = state + 1.0

                # Neighborwise: each neighbor receives a distinct payload.
                dtype = np.float64 if step % 2 == 0 else np.int64
                state_map = {
                    peer_id: np.full(
                        N_STATE, 1000 * int(node_id) + 100 * int(peer_id) + step,
                        dtype=dtype,
                    )
                    for peer_id in neighbors
                }
                received = network.neighborwise_exchange(
                    state_map,
                    key=f"neighborwise-{copy}-{step}",
                    timeout=TIMEOUT,
                    copy=copy,
                )
                assert set(received) == set(neighbors)
                for peer_id, value in received.items():
                    expected = np.full(
                        N_STATE,
                        1000 * int(peer_id) + 100 * int(node_id) + step,
                        dtype=dtype,
                    )
                    np.testing.assert_array_equal(value, expected)

                # These convenience methods must forward the copy option.
                x = np.full(32, float(node_id), dtype=np.float64)
                laplacian = network.laplacian(
                    x,
                    key=f"laplacian-{copy}-{step}",
                    timeout=TIMEOUT,
                    copy=copy,
                )
                expected_laplacian = len(neighbors) * int(node_id) - sum(
                    int(peer_id) for peer_id in neighbors
                )
                np.testing.assert_array_equal(laplacian, expected_laplacian)

                mixed = network.weighted_mix(
                    x,
                    key=f"weighted-{copy}-{step}",
                    timeout=TIMEOUT,
                    copy=copy,
                )
                expected_mix = (1.0 - sum(neighbors.values())) * int(node_id)
                expected_mix += sum(
                    weight * int(peer_id) for peer_id, weight in neighbors.items()
                )
                np.testing.assert_array_equal(mixed, expected_mix)

                del state_map, state
                gc.collect()

        return True
    finally:
        network.close(timeout=10.0 if started else 0.0)


def test_copy_modes() -> None:
    endpoints = {
        node_id: f"tcp://127.0.0.1:{port}"
        for node_id, port in zip(NODES, available_ports(len(NODES)))
    }
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(NODES), mp_context=ctx) as executor:
        tasks = [executor.submit(run_node, node_id, endpoints) for node_id in NODES]
        assert all(task.result(timeout=90) for task in tasks)


if __name__ == "__main__":
    test_copy_modes()
