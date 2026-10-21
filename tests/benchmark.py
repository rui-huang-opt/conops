import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import median
from typing import Callable

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)


def benchmark_rust(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
) -> float:
    network = Network(node_id=node_id, neighbors=neighbors, group_id=group_id)

    try:
        network.start()
        network.wait_connected(timeout=30)

        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for k in range(n_warmup):
            network.exchange(x, key=f"warmup-{k}")

        before = time.perf_counter()

        for k in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x, key=(k, "x"))
            y = y - 0.2 * network.laplacian(y, key=(k, "y"))

        return time.perf_counter() - before

    finally:
        network.close()


def benchmark_lite(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
) -> float:
    network = LiteNetwork(node_id=node_id, neighbors=neighbors, group_id=group_id)

    try:
        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for _ in range(n_warmup):
            network.exchange(x)

        before = time.perf_counter()

        for _ in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x)
            y = y - 0.2 * network.laplacian(y)

        return time.perf_counter() - before

    finally:
        network.close()


def run_benchmark(
    worker: Callable[[str, dict[str, float], str, int, int, int], float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
) -> list[float]:
    with ProcessPoolExecutor(
        max_workers=len(NODES),
        mp_context=get_context("spawn"),
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
    args = parser.parse_args()

    results: dict[str, list[float]] = {"Rust": [], "Lite": []}

    for repeat in range(args.repeats):
        benchmarks = (
            [("Rust", benchmark_rust), ("Lite", benchmark_lite)]
            if repeat % 2 == 0
            else [("Lite", benchmark_lite), ("Rust", benchmark_rust)]
        )

        for name, worker in benchmarks:
            group_id = f"bench-{os.getpid()}-{repeat}-{name.lower()}"

            times = run_benchmark(
                worker,
                group_id,
                args.n_iter,
                args.n_state,
                args.n_warmup,
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
