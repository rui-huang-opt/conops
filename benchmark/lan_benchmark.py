"""
Five-machine LAN benchmark for ConOps Network vs LiteNetwork.

Run once on each machine, changing only --node_id. Use the same --group_id,
benchmark parameters, and topology on every machine. No fixed IP or barrier.

Profiling options:
  --rust-profile  Rust internal profiling via CONOPS_RUST_PROFILE=1.
                  Requires instrumented Rust src/network.rs; includes warmup.
  --lite-profile  Lite encode/send/receive/decode profiling.
                  Requires instrumented conops/lite.py; excludes warmup.

Profiling affects runtime; run without these flags for final comparisons.
"""

import argparse
import os
import time
from statistics import median
from typing import Callable

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)

# LiteProfileStats is obtained via LiteNetwork.profile_stats().
BenchmarkResult = tuple[float, object | None]


def benchmark_rust(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    lite_profile: bool,
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

        before = time.perf_counter()
        for k in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x, key=(k, "x"))
            y = y - 0.2 * network.laplacian(y, key=(k, "y"))

        return time.perf_counter() - before, None
    finally:
        # Rust internal profiling prints on close() when enabled.
        network.close()


def benchmark_lite(
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_iter: int,
    n_state: int,
    n_warmup: int,
    lite_profile: bool,
) -> BenchmarkResult:
    network = LiteNetwork(node_id=node_id, neighbors=neighbors, group_id=group_id)

    try:
        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for _ in range(n_warmup):
            network.exchange(x)

        if lite_profile:
            network.start_profile()  # Reset after warmup.

        before = time.perf_counter()
        for _ in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x)
            y = y - 0.2 * network.laplacian(y)
        elapsed = time.perf_counter() - before

        stats = network.stop_profile() if lite_profile else None
        if stats is not None:
            expected_count = 2 * (n_iter - 1)
            if stats.count != expected_count:
                raise RuntimeError(
                    f"Lite profile recorded {stats.count} exchanges; "
                    f"expected {expected_count}."
                )

        return elapsed, stats
    finally:
        network.close()


def print_lite_profile(stats: object) -> None:
    # Only called after checking the instrumented LiteNetwork was installed.
    count = stats.count
    us = lambda ns: ns / count / 1_000
    print(f"  Lite profile ({count} exchanges, average us/call):", flush=True)
    print(f"    encode/prepare         {us(stats.encode_ns):9.2f} us", flush=True)
    print(f"    send_multipart         {us(stats.send_ns):9.2f} us", flush=True)
    print(f"    recv_multipart waiting {us(stats.recv_wait_ns):9.2f} us", flush=True)
    print(f"    decode                 {us(stats.decode_ns):9.2f} us", flush=True)
    print(f"    exchange total         {us(stats.total_ns):9.2f} us", flush=True)
    for peer in sorted(stats.peer_recv_wait_ns):
        print(
            f"    peer {peer}: recv wait {us(stats.peer_recv_wait_ns[peer]):.2f} us, "
            f"recv completed at {us(stats.peer_recv_complete_ns[peer]):.2f} us "
            f"since exchange start",
            flush=True,
        )
    print(
        "    Note: send times measure ZeroMQ submission, not delivery; "
        "receives are sequential and may be already queued.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ConOps multi-machine LAN benchmark")
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1_000_000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--rust-profile",
        action="store_true",
        help="Enable Rust internal profiling (requires profiled src/network.rs)",
    )
    parser.add_argument(
        "--lite-profile",
        action="store_true",
        help="Enable Lite encode/send/receive/decode profiling (requires profiled lite.py)",
    )
    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_iter >= 2, n_state >= 1, n_warmup >= 0, repeats >= 1")

    # Fail before connecting, not mid-run where other nodes could wait forever.
    if args.lite_profile and not all(
        callable(getattr(LiteNetwork, name, None))
        for name in ("start_profile", "stop_profile")
    ):
        parser.error("--lite-profile requires the instrumented conops/lite.py")

    if args.rust_profile:
        os.environ["CONOPS_RUST_PROFILE"] = "1"

    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    neighbors = graph[args.node_id]

    for repeat in range(args.repeats):
        benchmarks: list[tuple[str, Callable[..., BenchmarkResult]]] = (
            [("Rust", benchmark_rust), ("Lite", benchmark_lite)]
            if repeat % 2 == 0
            else [("Lite", benchmark_lite), ("Rust", benchmark_rust)]
        )

        for name, worker in benchmarks:
            group_id = f"{args.group_id}-{repeat}-{name.lower()}"
            print(
                f"Node {args.node_id} | Run {repeat + 1:02d} | {name}: connecting...",
                flush=True,
            )

            elapsed, profile_stats = worker(
                args.node_id,
                neighbors,
                group_id,
                args.n_iter,
                args.n_state,
                args.n_warmup,
                args.lite_profile,
            )
            results[name].append(elapsed)

            print(
                f"Node {args.node_id} | Run {repeat + 1:02d} | "
                f"{name:4s} | elapsed={elapsed:.3f}s",
                flush=True,
            )

            if name == "Lite" and profile_stats is not None:
                print_lite_profile(profile_stats)

    print(f"\nNode {args.node_id} summary (median of local elapsed times)")
    for name, times in results.items():
        print(
            f"{name:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
