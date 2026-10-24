"""Benchmark neighborwise_exchange() on a persistent five-node localhost ring.

Each subprocess creates Rust and Lite connections once for all repeats.
State arrays, correctness checks and warmup are outside the timed intervals.
The --copy / --no-copy option applies only to Rust; Lite is unchanged.
No process barrier is used.

Examples
--------
python benchmark_neighborwise.py --n_state 5 --n_iter 10000
python benchmark_neighborwise.py --n_state 2000000 --no-copy
python benchmark_neighborwise.py --n_state 2000000 --copy
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median

import numpy as np

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]
EXCHANGE_TIMEOUT = 20.0  # Rust only; Lite does not have an exchange timeout.
GRAPH = Graph(NODES, EDGES)


def make_state_maps(
    node_id: str,
    neighbor_ids: list[str],
    n_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Each backend and repeat receives identically initialized per-peer arrays.
    rng = np.random.default_rng(int(node_id))
    x_map = {peer: rng.uniform(-100.0, 100.0, n_state) for peer in neighbor_ids}
    y_map = {peer: rng.uniform(-100.0, 100.0, n_state) for peer in neighbor_ids}
    for peer in neighbor_ids:
        x_map[peer][0] = 100 * int(node_id) + int(peer)
        y_map[peer][0] = 10000 + 100 * int(node_id) + int(peer)
    return x_map, y_map


def check_received(
    received: dict[str, np.ndarray],
    state_map: dict[str, np.ndarray],
    node_id: str,
    n_state: int,
    marker_offset: int,
) -> None:
    if set(received) != set(state_map):
        raise AssertionError(f"Unexpected received peers on node {node_id}")
    for peer, state in received.items():
        expected = marker_offset + 100 * int(peer) + int(node_id)
        if state.shape != (n_state,) or state[0] != expected:
            raise AssertionError(
                f"Incorrect data from {peer} to node {node_id}: "
                f"got shape={state.shape}, first={state[0]}"
            )


def measure_rust(
    network: Network,
    node_id: str,
    repeat: int,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    copy: bool,
) -> float:
    x_map, y_map = make_state_maps(node_id, list(network.neighbors), n_state)

    for state_map, key, offset in (
        (x_map, f"check-{repeat}-x", 0),
        (y_map, f"check-{repeat}-y", 10000),
    ):
        received = network.neighborwise_exchange(
            state_map,
            key=key,
            timeout=EXCHANGE_TIMEOUT,
            copy=copy,
        )
        check_received(received, state_map, node_id, n_state, offset)

    for k in range(n_warmup):
        network.neighborwise_exchange(
            x_map,
            key=f"warmup-{repeat}-{k}",
            timeout=EXCHANGE_TIMEOUT,
            copy=copy,
        )

    before = time.perf_counter()
    for k in range(n_iter - 1):
        network.neighborwise_exchange(
            x_map,
            key=(k, f"{repeat}-x"),
            timeout=EXCHANGE_TIMEOUT,
            copy=copy,
        )

        network.neighborwise_exchange(
            y_map,
            key=(k, f"{repeat}-y"),
            timeout=EXCHANGE_TIMEOUT,
            copy=copy,
        )

    return time.perf_counter() - before


def measure_lite(
    network: LiteNetwork,
    node_id: str,
    repeat: int,
    n_iter: int,
    n_state: int,
    n_warmup: int,
) -> float:
    x_map, y_map = make_state_maps(node_id, list(network.neighbors), n_state)

    for state_map, key, offset in (
        (x_map, f"check-{repeat}-x", 0),
        (y_map, f"check-{repeat}-y", 10000),
    ):
        received = network.neighborwise_exchange(state_map, key=key)
        check_received(received, state_map, node_id, n_state, offset)

    for k in range(n_warmup):
        network.neighborwise_exchange(x_map, key=f"warmup-{repeat}-{k}")

    before = time.perf_counter()
    for k in range(n_iter - 1):
        network.neighborwise_exchange(x_map, key=(k, f"{repeat}-x"))
        network.neighborwise_exchange(y_map, key=(k, f"{repeat}-y"))
    return time.perf_counter() - before


def benchmark_node(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    repeats: int,
    copy: bool,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    rust: Network | None = None
    lite: LiteNetwork | None = None
    rust_started = False

    try:
        rust = Network(
            node_id=node_id,
            neighbors=neighbors,
            group_id=f"{group_id}-rust",
        )
        rust.start()
        rust_started = True
        rust.wait_connected(timeout=30)

        lite = LiteNetwork(
            node_id=node_id,
            neighbors=neighbors,
            group_id=f"{group_id}-lite",
        )
        print(f"Node {node_id} | Rust and Lite connected", flush=True)

        for repeat in range(repeats):
            if repeat % 2 == 0:
                rust_elapsed = measure_rust(
                    network=rust,
                    node_id=node_id,
                    repeat=repeat,
                    n_iter=n_iter,
                    n_state=n_state,
                    n_warmup=n_warmup,
                    copy=copy,
                )
                results["Rust"].append(rust_elapsed)

                lite_elapsed = measure_lite(
                    network=lite,
                    node_id=node_id,
                    repeat=repeat,
                    n_iter=n_iter,
                    n_state=n_state,
                    n_warmup=n_warmup,
                )
                results["Lite"].append(lite_elapsed)

            else:
                lite_elapsed = measure_lite(
                    network=lite,
                    node_id=node_id,
                    repeat=repeat,
                    n_iter=n_iter,
                    n_state=n_state,
                    n_warmup=n_warmup,
                )
                results["Lite"].append(lite_elapsed)

                rust_elapsed = measure_rust(
                    network=rust,
                    node_id=node_id,
                    repeat=repeat,
                    n_iter=n_iter,
                    n_state=n_state,
                    n_warmup=n_warmup,
                    copy=copy,
                )
                results["Rust"].append(rust_elapsed)

        return results

    finally:
        # Connections remain active throughout all repeats; no barrier needed.
        try:
            if lite is not None:
                lite.close(timeout=10.0)
        finally:
            if rust is not None:
                rust.close(timeout=10.0 if rust_started else 0.0)


def run_benchmark(
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    repeats: int,
    copy: bool,
) -> list[dict[str, list[float]]]:
    with ProcessPoolExecutor(
        max_workers=len(NODES),
        mp_context=get_context("spawn"),
    ) as executor:
        tasks = [
            executor.submit(
                benchmark_node,
                node_id,
                GRAPH[node_id],
                group_id,
                n_iter,
                n_state,
                n_warmup,
                repeats,
                copy,
            )
            for node_id in NODES
        ]
        return [task.result() for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent five-node localhost neighborwise benchmark"
    )
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1_000_000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--group_id", default=None)
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rust only: copy payload into independent send snapshots (default: true). "
        "Use --no-copy to borrow payloads; Lite is unchanged.",
    )

    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_iter >= 2, n_state >= 1, n_warmup >= 0, repeats >= 1")

    group_id = args.group_id or f"neighborwise-bench-{os.getpid()}"
    print(
        f"Five-node ring | persistent connections | Rust copy={args.copy} | "
        f"group={group_id}",
        flush=True,
    )
    node_results = run_benchmark(
        group_id=group_id,
        n_iter=args.n_iter,
        n_state=args.n_state,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
        copy=args.copy,
    )

    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    print("\nPer-repeat maximum node times")
    for repeat in range(args.repeats):
        order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")
        for kind in order:
            times = [node[kind][repeat] for node in node_results]
            elapsed = max(times)
            results[kind].append(elapsed)
            print(
                f"Run {repeat + 1:02d} | {kind:4s} | "
                f"max={elapsed:.3f}s | "
                f"nodes={[round(t, 3) for t in times]}",
                flush=True,
            )

    print(f"\nSummary (median of maximum node times; Rust copy={args.copy})")
    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
