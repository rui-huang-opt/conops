"""Boundary tests for ConOps 0.5.1 neighborwise_exchange().

Run from the project root, after installing the Rust extension:
    python tests/test_network_edge_cases.py

Uses static loopback endpoints; does not require multicast/mDNS or pytest.
"""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Protocol
import socket

import numpy as np

from conops import Network


class BarrierLike(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...


NODE_IDS = ("1", "2", "3")
EXCHANGE_TIMEOUT = 15.0


def available_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def expect_error(exc_type: type[Exception], operation, label: str) -> None:
    try:
        operation()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__}: {label}")


def test_local_validation() -> None:
    """Bad inputs should fail before sending anything; no TCP startup required."""
    network = Network("1", neighbors={"2": 0.5, "3": 0.5})
    state = np.array([1.0], dtype=np.float64)
    valid = {"2": state, "3": state}

    try:
        for state_map, label in (
            ({}, "missing both neighbors"),
            ({"2": state}, "missing one neighbor"),
            ({**valid, "4": state}, "unexpected neighbor"),
        ):
            expect_error(
                ValueError,
                lambda state_map=state_map: network.neighborwise_exchange(
                    state_map, key="invalid-map"
                ),
                label,
            )

        for key in (None, True, 1.5, [1, "x"], ("x", 1), (1, 2)):
            expect_error(
                TypeError,
                lambda key=key: network.neighborwise_exchange(valid, key=key),  # type: ignore
                f"invalid key {key!r}",
            )

        expect_error(
            ValueError,
            lambda: network.neighborwise_exchange(valid, key="k" * 300),
            "oversized encoded key",
        )
        expect_error(
            TypeError,
            lambda: network.neighborwise_exchange(
                {"2": np.array([object()], dtype=object), "3": state},
                key="object-array",
            ),
            "object dtype",
        )
    finally:
        network.close()

    # A small frame limit must be checked on every neighbor before queuing DATA.
    limited = Network("1", neighbors={"2": 0.5}, max_message_size=128)
    try:
        expect_error(
            ValueError,
            lambda: limited.neighborwise_exchange(
                {"2": np.arange(100, dtype=np.float64)}, key="too-large"
            ),
            "DATA larger than max_message_size",
        )
    finally:
        limited.close()


def make_state(sender: str, destination: str, case: int) -> np.ndarray:
    """Use deterministic arrays; some inputs are deliberately non-contiguous."""
    marker = 10 * int(sender) + int(destination)

    if case == 0:
        return np.array(marker + 0.25, dtype=np.float64)  # 0-D scalar
    if case == 1:
        return np.empty((0, 3), dtype=np.float32)  # Empty 2-D array
    if case == 2:
        return np.arange(12, dtype=np.int16).reshape(3, 4) + marker
    if case == 3:
        base = np.arange(24, dtype=np.float64).reshape(4, 6) + marker
        return base[:, ::2]  # Strided two-dimensional view
    if case == 4:
        dtype = np.float32 if int(destination) % 2 else np.int64
        return np.array([marker, marker + 1, marker + 2], dtype=dtype)
    if case == 5:
        return (np.arange(16, dtype=np.int64) + marker)[::-1]  # Negative stride

    raise ValueError(f"Unknown test case: {case}")


def worker(node_id: str, endpoints: dict[str, str], barrier: BarrierLike) -> str:
    neighbors = {other: 0.25 for other in NODE_IDS if other != node_id}
    network = Network(node_id, neighbors, peer_endpoints=endpoints)
    try:
        network.start()
        network.wait_connected(timeout=20)

        for case in range(6):
            state_map = {peer: make_state(node_id, peer, case) for peer in neighbors}
            received = network.neighborwise_exchange(
                state_map, key=(case, "edge-case"), timeout=EXCHANGE_TIMEOUT
            )
            assert set(received) == set(neighbors)
            for peer, actual in received.items():
                expected = make_state(peer, node_id, case)
                assert actual.shape == expected.shape, (node_id, peer, case, "shape")
                assert actual.dtype == expected.dtype, (node_id, peer, case, "dtype")
                np.testing.assert_array_equal(actual, expected)

        # The same logical key cannot be reused immediately after completion.
        expect_error(
            RuntimeError,
            lambda: network.neighborwise_exchange(
                {peer: make_state(node_id, peer, 0) for peer in neighbors},
                key=(5, "edge-case"),
                timeout=EXCHANGE_TIMEOUT,
            ),
            "reusing a recently completed key",
        )

        # Keep sockets alive until every node finishes its last exchange.
        barrier.wait(timeout=30)
        return node_id
    finally:
        network.close()


def test_array_roundtrips() -> None:
    ports = available_ports(len(NODE_IDS))
    endpoints = {node: f"tcp://127.0.0.1:{port}" for node, port in zip(NODE_IDS, ports)}
    ctx = get_context("spawn")
    with ctx.Manager() as manager:
        barrier = manager.Barrier(len(NODE_IDS))
        with ProcessPoolExecutor(max_workers=len(NODE_IDS), mp_context=ctx) as pool:
            futures = [
                pool.submit(worker, node_id, endpoints, barrier) for node_id in NODE_IDS
            ]
            results = [future.result(timeout=90) for future in futures]
    assert set(results) == set(NODE_IDS)


if __name__ == "__main__":
    test_local_validation()
    test_array_roundtrips()
