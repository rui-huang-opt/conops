"""
Benchmark neighborwise_exchange() on a five-node ring (Rust vs Lite).

Each node sends distinct arrays to its neighbors. Arrays are prepared before
measurement, so timed loops emphasize exchange rather than array creation.

Examples:
    python tests/benchmark_neighborwise.py --n_iter 10000 --n_state 5
    python tests/benchmark_neighborwise.py --n_iter 100 --n_state 1000000
    python tests/benchmark_neighborwise.py --n_state 1000000 --repeats 1 --debug
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median
from typing import Callable, Protocol

import numpy as np

from conops import Graph, LiteNetwork, Network


class BarrierLike(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...


NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

EXCHANGE_TIMEOUT = 20.0  # Rust only; Lite has no exchange timeout.

graph = Graph(NODES, EDGES)


def benchmark_rust(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    debug: bool,
    barrier: BarrierLike,
) -> float:
    if debug:
        print(f"[Rust {node_id}] Initializing", flush=True)

    network = Network(node_id=node_id, neighbors=neighbors, group_id=group_id)

    try:
        network.start()
        network.wait_connected(timeout=30)

        if debug:
            print(f"[Rust {node_id}] Connected", flush=True)

        rng = np.random.default_rng(int(node_id))

        x_map = {
            peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors
        }
        y_map = {
            peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors
        }

        for peer in network.neighbors:
            x_map[peer][0] = 100 * int(node_id) + int(peer)
            y_map[peer][0] = 10000 + 100 * int(node_id) + int(peer)

        # Preflight correctness checks.
        for state_map, key, marker_offset in (
            (x_map, "check-x", 0),
            (y_map, "check-y", 10000),
        ):
            if debug:
                print(f"[Rust {node_id}] Preflight {key}", flush=True)

            received = network.neighborwise_exchange(
                state_map,
                key=key,
                timeout=EXCHANGE_TIMEOUT,
            )

            if set(received) != set(state_map):
                raise AssertionError(f"Unexpected received peers on node {node_id}")

            for peer, state in received.items():
                expected = marker_offset + 100 * int(peer) + int(node_id)

                if state.shape != (n_state,) or state[0] != expected:
                    raise AssertionError(
                        f"Incorrect data from {peer} to {node_id}: "
                        f"got shape={state.shape}, first={state[0]}"
                    )

        # Warmup.
        for k in range(n_warmup):
            if debug:
                print(f"[Rust {node_id}] Warmup {k}", flush=True)

            network.neighborwise_exchange(
                x_map,
                key=f"warmup-{k}",
                timeout=EXCHANGE_TIMEOUT,
            )

        if debug:
            print(f"[Rust {node_id}] Timed loop starts", flush=True)

        before = time.perf_counter()

        for k in range(n_iter - 1):
            if debug:
                print(f"[Rust {node_id}] k={k} x", flush=True)

            network.neighborwise_exchange(
                x_map,
                key=(k, "x"),
                timeout=EXCHANGE_TIMEOUT,
            )

            if debug:
                print(f"[Rust {node_id}] k={k} y", flush=True)

            network.neighborwise_exchange(
                y_map,
                key=(k, "y"),
                timeout=EXCHANGE_TIMEOUT,
            )

        elapsed = time.perf_counter() - before

        # All nodes finish before any node closes its connection.
        # The barrier is excluded from the measured time.
        if debug:
            print(
                f"[Rust {node_id}] Finished, waiting at barrier",
                flush=True,
            )

        barrier.wait(timeout=30)

        return elapsed

    finally:
        network.close()

        if debug:
            print(f"[Rust {node_id}] Closed", flush=True)


def benchmark_lite(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    debug: bool,
    barrier: BarrierLike,
) -> float:
    if debug:
        print(f"[Lite {node_id}] Initializing", flush=True)

    network = LiteNetwork(
        node_id=node_id,
        neighbors=neighbors,
        group_id=group_id,
    )

    try:
        if debug:
            print(f"[Lite {node_id}] Connected", flush=True)

        rng = np.random.default_rng(int(node_id))

        x_map = {
            peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors
        }
        y_map = {
            peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors
        }

        for peer in network.neighbors:
            x_map[peer][0] = 100 * int(node_id) + int(peer)
            y_map[peer][0] = 10000 + 100 * int(node_id) + int(peer)

        # Preflight correctness checks.
        for state_map, key, marker_offset in (
            (x_map, "check-x", 0),
            (y_map, "check-y", 10000),
        ):
            if debug:
                print(f"[Lite {node_id}] Preflight {key}", flush=True)

            received = network.neighborwise_exchange(
                state_map,
                key=key,
            )

            if set(received) != set(state_map):
                raise AssertionError(f"Unexpected received peers on node {node_id}")

            for peer, state in received.items():
                expected = marker_offset + 100 * int(peer) + int(node_id)

                if state.shape != (n_state,) or state[0] != expected:
                    raise AssertionError(
                        f"Incorrect data from {peer} to {node_id}: "
                        f"got shape={state.shape}, first={state[0]}"
                    )

        # Warmup.
        for k in range(n_warmup):
            if debug:
                print(f"[Lite {node_id}] Warmup {k}", flush=True)

            network.neighborwise_exchange(
                x_map,
                key=f"warmup-{k}",
            )

        if debug:
            print(f"[Lite {node_id}] Timed loop starts", flush=True)

        before = time.perf_counter()

        for k in range(n_iter - 1):
            if debug:
                print(f"[Lite {node_id}] k={k} x", flush=True)

            network.neighborwise_exchange(
                x_map,
                key=(k, "x"),
            )

            if debug:
                print(f"[Lite {node_id}] k={k} y", flush=True)

            network.neighborwise_exchange(
                y_map,
                key=(k, "y"),
            )

        elapsed = time.perf_counter() - before

        # Keep connections alive until every node has finished.
        if debug:
            print(
                f"[Lite {node_id}] Finished, waiting at barrier",
                flush=True,
            )

        barrier.wait(timeout=30)

        return elapsed

    finally:
        network.close()

        if debug:
            print(f"[Lite {node_id}] Closed", flush=True)


def run_benchmark(
    worker: Callable[
        [str, dict[str, float], str, int, int, int, bool, BarrierLike],
        float,
    ],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    debug: bool,
) -> list[float]:
    ctx = get_context("spawn")

    with ctx.Manager() as manager:
        barrier = manager.Barrier(len(NODES))

        with ProcessPoolExecutor(
            max_workers=len(NODES),
            mp_context=ctx,
        ) as executor:
            tasks = [
                executor.submit(
                    worker,
                    node_id,
                    graph[node_id],
                    group_id,
                    n_iter,
                    n_state,
                    n_warmup,
                    debug,
                    barrier,
                )
                for node_id in NODES
            ]

            return [task.result() for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1_000_000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print worker progress (affects timing)",
    )

    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error(
            "Require n_iter >= 2, n_state >= 1, " "n_warmup >= 0, repeats >= 1"
        )

    results: dict[str, list[float]] = {
        "Rust": [],
        "Lite": [],
    }

    for repeat in range(args.repeats):
        benchmarks = (
            [("Rust", benchmark_rust), ("Lite", benchmark_lite)]
            if repeat % 2 == 0
            else [("Lite", benchmark_lite), ("Rust", benchmark_rust)]
        )

        for name, worker in benchmarks:
            group_id = f"neighborwise-bench-" f"{os.getpid()}-{repeat}-{name.lower()}"

            times = run_benchmark(
                worker,
                group_id,
                args.n_iter,
                args.n_state,
                args.n_warmup,
                args.debug,
            )

            elapsed = max(times)
            results[name].append(elapsed)

            print(
                f"Run {repeat + 1:02d} | {name:4s} | "
                f"max={elapsed:.3f}s | "
                f"nodes={[round(t, 3) for t in times]}",
                flush=True,
            )

    print("\nSummary (median of maximum node times)")

    for name, times in results.items():
        print(
            f"{name:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)"
        )


if __name__ == "__main__":
    main()
