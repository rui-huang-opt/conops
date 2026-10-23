"""
Benchmark ConOps on a persistent five-node localhost ring (Rust vs Lite).

Five subprocesses are started once. Each subprocess establishes its Rust and
Lite connections once, then reuses them for all repeats. Initialization and
warmup are excluded from the measurements; no barriers are introduced.

Examples
--------
python benchmark_local_persistent.py --n_state 5 --n_iter 1000
python benchmark_local_persistent.py --n_state 2000000 --n_iter 100 --repeats 5
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]
GRAPH = Graph(NODES, EDGES)


def measure_round(
    network: Network | LiteNetwork,
    node_id: str,
    repeat: int,
    n_state: int,
    n_iter: int,
    n_warmup: int,
) -> float:
    npr.seed(int(node_id))
    x = npr.uniform(-100.0, 100.0, n_state)
    y = npr.uniform(-100.0, 100.0, n_state)

    for k in range(n_warmup):
        network.exchange(x, key=f"warmup-{repeat}-{k}")

    before = time.perf_counter()
    for k in range(n_iter - 1):
        x = x - 0.45 * network.laplacian(x, key=(k, f"{repeat}-x"))
        y = y - 0.20 * network.laplacian(y, key=(k, f"{repeat}-y"))
    return time.perf_counter() - before


def benchmark_node(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    repeats: int,
) -> dict[str, list[float]]:
    """One subprocess per node; both network objects survive all repeats."""
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

        networks: dict[str, Network | LiteNetwork] = {"Rust": rust, "Lite": lite}
        for repeat in range(repeats):
            order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")
            for kind in order:
                elapsed = measure_round(
                    network=networks[kind],
                    node_id=node_id,
                    repeat=repeat,
                    n_state=n_state,
                    n_iter=n_iter,
                    n_warmup=n_warmup,
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
) -> list[dict[str, list[float]]]:
    # The process pool and network connections are created only once.
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
            )
            for node_id in NODES
        ]
        return [task.result() for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent five-node localhost ConOps benchmark"
    )
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1_000_000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--group_id", default=None)
    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_iter >= 2, n_state >= 1, n_warmup >= 0, repeats >= 1")

    # Generate once in the parent, then share the same group ID with all nodes.
    group_id = args.group_id or f"bench-{os.getpid()}"
    print(f"Five-node ring | persistent connections | group={group_id}", flush=True)

    node_results = run_benchmark(
        group_id=group_id,
        n_iter=args.n_iter,
        n_state=args.n_state,
        n_warmup=args.n_warmup,
        repeats=args.repeats,
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
