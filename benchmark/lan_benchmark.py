"""Five-machine LAN benchmark for ConOps Network vs LiteNetwork.

Run once on each machine, changing only --node_id. Keep group ID and benchmark
parameters identical across machines. No fixed IPs or extra barriers are used.

Use --profile to measure Rust Network.exchange() Python-side timing:
  prepare: semantic key encoding and array preparation
  core:    synchronous PyO3 call, including communication and network waiting
  unpack: construction/decoding of received arrays
Profiling is opt-in because timers themselves slightly affect the benchmark.
"""

import argparse
import time
from statistics import median
from typing import Callable

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)

# (prepare_ns, core_ns, unpack_ns, number_of_exchange_calls)
Profile = tuple[int, int, int, int]
BenchmarkResult = tuple[float, Profile | None]


def benchmark_rust(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    profile: bool,
) -> BenchmarkResult:
    network = Network(node_id=node_id, neighbors=neighbors, group_id=group_id)

    try:
        network.start()
        network.wait_connected(timeout=120)

        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for k in range(n_warmup):
            network.exchange(x, key=f"warmup-{k}")

        if profile:
            # The timer only covers the 2*(n_iter-1) timed exchange calls.
            network._profile = [0, 0, 0, 0]

        before = time.perf_counter()
        for k in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x, key=(k, "x"))
            y = y - 0.2 * network.laplacian(y, key=(k, "y"))

        elapsed = time.perf_counter() - before
        profile_result = tuple(network._profile) if profile else None
        return elapsed, profile_result

    finally:
        network.close()


def benchmark_lite(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    profile: bool,
) -> BenchmarkResult:
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

        return time.perf_counter() - before, None

    finally:
        network.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ConOps multi-machine LAN benchmark")
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1_000_000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Time Rust Python prepare/core/unpack stages after warmup",
    )
    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_iter >= 2, n_state >= 1, n_warmup >= 0, repeats >= 1")

    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    neighbors = graph[args.node_id]

    for repeat in range(args.repeats):
        benchmarks: list[tuple[str, Callable[..., BenchmarkResult]]] = (
            [("Rust", benchmark_rust), ("Lite", benchmark_lite)]
            if repeat % 2 == 0
            else [("Lite", benchmark_lite), ("Rust", benchmark_rust)]
        )

        for name, worker in benchmarks:
            # Same ID on all machines: never include local PID or hostname.
            group_id = f"{args.group_id}-{repeat}-{name.lower()}"
            print(
                f"Node {args.node_id} | Run {repeat + 1:02d} | {name}: connecting...",
                flush=True,
            )

            elapsed, stats = worker(
                args.node_id,
                neighbors,
                group_id,
                args.n_iter,
                args.n_state,
                args.n_warmup,
                args.profile,
            )
            results[name].append(elapsed)

            print(
                f"Node {args.node_id} | Run {repeat + 1:02d} | "
                f"{name:4s} | elapsed={elapsed:.3f}s",
                flush=True,
            )

            if stats is not None:
                prepare_ns, core_ns, unpack_ns, count = stats
                if count:
                    print(
                        f"  Rust Python profile ({count} exchanges, average us/call): "
                        f"prepare={prepare_ns / count / 1_000:.2f}, "
                        f"core={core_ns / count / 1_000:.2f}, "
                        f"unpack={unpack_ns / count / 1_000:.2f}; "
                        f"prepare+unpack={(prepare_ns + unpack_ns) / count / 1_000:.2f}",
                        flush=True,
                    )

    print(f"\nNode {args.node_id} summary (median of local elapsed times)")
    for name, times in results.items():
        print(
            f"{name:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
