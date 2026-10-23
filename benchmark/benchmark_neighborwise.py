"""
Benchmark neighborwise_exchange() on a persistent five-node localhost ring.

Each node starts Rust and Lite once and reuses both connections for all repeats.
Arrays, correctness checks, and warmup are prepared outside each timed interval.
No process barrier is used; connections remain alive through all repeats.

Examples:
    python benchmark_neighborwise_persistent.py --n_iter 10000 --n_state 5
    python benchmark_neighborwise_persistent.py --n_iter 100 --n_state 1000000
    python benchmark_neighborwise_persistent.py --n_state 2000000 --repeats 5
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


def measure_round(
    network: Network | LiteNetwork,
    kind: str,
    node_id: str,
    repeat: int,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    debug: bool,
) -> float:
    # Prepare different arrays for each neighbor, as in the original benchmark.
    # Both backends and every repeat begin with the same deterministic data.
    rng = np.random.default_rng(int(node_id))
    x_map = {peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors}
    y_map = {peer: rng.uniform(-100.0, 100.0, n_state) for peer in network.neighbors}
    for peer in network.neighbors:
        x_map[peer][0] = 100 * int(node_id) + int(peer)
        y_map[peer][0] = 10000 + 100 * int(node_id) + int(peer)

    # The two backends accept different optional arguments.
    exchange = network.neighborwise_exchange
    options = {"timeout": EXCHANGE_TIMEOUT} if kind == "Rust" else {}

    # Check correctness before each timed run. Keys are unique across repeats.
    for state_map, key, offset in (
        (x_map, f"check-{repeat}-x", 0),
        (y_map, f"check-{repeat}-y", 10000),
    ):
        if debug:
            print(f"[{kind} {node_id}] Preflight {key}", flush=True)

        received = exchange(state_map, key=key, **options)

        if set(received) != set(state_map):
            raise AssertionError(f"Unexpected received peers on node {node_id}")

        for peer, state in received.items():
            expected = offset + 100 * int(peer) + int(node_id)
            if state.shape != (n_state,) or state[0] != expected:
                raise AssertionError(
                    f"Incorrect data from {peer} to {node_id}: "
                    f"got shape={state.shape}, first={state[0]}"
                )

    for k in range(n_warmup):
        if debug:
            print(f"[{kind} {node_id}] Warmup {repeat}-{k}", flush=True)
        exchange(x_map, key=f"warmup-{repeat}-{k}", **options)

    if debug:
        print(f"[{kind} {node_id}] Timed loop starts (repeat {repeat + 1})", flush=True)

    before = time.perf_counter()
    for k in range(n_iter - 1):
        if debug:
            print(f"[{kind} {node_id}] repeat={repeat} k={k} x", flush=True)
        exchange(x_map, key=(k, f"{repeat}-x"), **options)

        if debug:
            print(f"[{kind} {node_id}] repeat={repeat} k={k} y", flush=True)
        exchange(y_map, key=(k, f"{repeat}-y"), **options)

    return time.perf_counter() - before


def benchmark_node(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    repeats: int,
    debug: bool,
) -> dict[str, list[float]]:
    """Run all repeats on the same two live connections in one subprocess."""
    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    rust: Network | None = None
    lite: LiteNetwork | None = None
    rust_started = False

    try:
        if debug:
            print(f"[Rust {node_id}] Initializing", flush=True)

        rust = Network(
            node_id=node_id,
            neighbors=neighbors,
            group_id=f"{group_id}-rust",
        )
        rust.start()
        rust_started = True
        rust.wait_connected(timeout=30)

        if debug:
            print(f"[Lite {node_id}] Initializing", flush=True)

        lite = LiteNetwork(
            node_id=node_id,
            neighbors=neighbors,
            group_id=f"{group_id}-lite",
        )

        print(f"Node {node_id} | Rust and Lite connected", flush=True)

        networks: dict[str, Network | LiteNetwork] = {"Rust": rust, "Lite": lite}
        for repeat in range(repeats):
            order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")
            for kind in order:
                elapsed = measure_round(
                    network=networks[kind],
                    kind=kind,
                    node_id=node_id,
                    repeat=repeat,
                    n_iter=n_iter,
                    n_state=n_state,
                    n_warmup=n_warmup,
                    debug=debug,
                )
                results[kind].append(elapsed)

        return results

    finally:
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
    debug: bool,
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
                debug,
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
        "--debug",
        action="store_true",
        help="Print worker progress (affects timing)",
    )
    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_iter >= 2, n_state >= 1, n_warmup >= 0, repeats >= 1")

    group_id = args.group_id or f"neighborwise-bench-{os.getpid()}"

    print(f"Five-node ring | persistent connections | group={group_id}", flush=True)

    node_results = run_benchmark(
        group_id=group_id,
        n_iter=args.n_iter,
        n_state=args.n_state,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
        debug=args.debug,
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

    print("\nSummary (median of maximum node times)")

    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
