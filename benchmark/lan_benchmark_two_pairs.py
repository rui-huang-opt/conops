"""LAN benchmark for ConOps Network vs LiteNetwork.

Run one process per node in the selected topology. Use the same --topology,
--group_id, --backend and benchmark parameters for both ends of each pair.
No fixed IP or extra barriers are used.

Topologies:
  ring    1-2-3-4-5-1
  local   1-2
  remote  1-5
  pair_a  1-5
  pair_b  2-6

Backend:
  --backend both  Alternate Rust/Lite order (default)
  --backend rust  Run only Rust
  --backend lite  Run only Lite

Profiling:
  --rust-profile  Rust internal profiling.
  --lite-profile  Lite internal profiling.

Disable profiling for final performance comparisons.
"""

import argparse
import os
import time
from statistics import median
from typing import Callable

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

TOPOLOGIES = {
    "ring": (
        ["1", "2", "3", "4", "5"],
        [
            ("1", "2"),
            ("2", "3"),
            ("3", "4"),
            ("4", "5"),
            ("5", "1"),
        ],
    ),
    "local": (["1", "2"], [("1", "2")]),
    "remote": (["1", "5"], [("1", "5")]),
    "pair_a": (["1", "5"], [("1", "5")]),
    "pair_b": (["2", "6"], [("2", "6")]),
}

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
    network = Network(
        node_id=node_id,
        neighbors=neighbors,
        group_id=group_id,
    )

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

        elapsed = time.perf_counter() - before
        return elapsed, None

    finally:
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
    network = LiteNetwork(
        node_id=node_id,
        neighbors=neighbors,
        group_id=group_id,
    )

    try:
        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for _ in range(n_warmup):
            network.exchange(x)

        if lite_profile:
            network.start_profile()

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
                    f"Lite profile recorded {stats.count} "
                    f"exchanges; expected {expected_count}."
                )

        return elapsed, stats

    finally:
        network.close()


def print_lite_profile(stats: object) -> None:
    count = stats.count

    def us(ns):
        return ns / count / 1_000

    print(
        f"  Lite profile ({count} exchanges, " f"average us/call):",
        flush=True,
    )

    print(
        f"    encode/prepare         " f"{us(stats.encode_ns):9.2f} us",
        flush=True,
    )
    print(
        f"    send_multipart         " f"{us(stats.send_ns):9.2f} us",
        flush=True,
    )
    print(
        f"    recv_multipart waiting " f"{us(stats.recv_wait_ns):9.2f} us",
        flush=True,
    )
    print(
        f"    decode                 " f"{us(stats.decode_ns):9.2f} us",
        flush=True,
    )
    print(
        f"    exchange total         " f"{us(stats.total_ns):9.2f} us",
        flush=True,
    )

    for peer in sorted(stats.peer_recv_wait_ns):
        print(
            f"    peer {peer}: recv wait "
            f"{us(stats.peer_recv_wait_ns[peer]):.2f} us, "
            f"recv completed at "
            f"{us(stats.peer_recv_complete_ns[peer]):.2f} us "
            f"since exchange start",
            flush=True,
        )

    print(
        "    Note: send times measure ZeroMQ submission, "
        "not delivery; receives are sequential.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ConOps LAN benchmark")

    parser.add_argument(
        "--node_id",
        required=True,
    )
    parser.add_argument(
        "--topology",
        choices=TOPOLOGIES,
        default="ring",
    )
    parser.add_argument(
        "--backend",
        choices=("rust", "lite", "both"),
        default="both",
        help="Run Rust only, Lite only, or both",
    )
    parser.add_argument(
        "--group_id",
        default="lan-benchmark",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n_state",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rust-profile",
        action="store_true",
    )
    parser.add_argument(
        "--lite-profile",
        action="store_true",
    )

    args = parser.parse_args()

    if args.n_iter < 2 or args.n_state < 1 or args.n_warmup < 0 or args.repeats < 1:
        parser.error(
            "Require n_iter >= 2, n_state >= 1, " "n_warmup >= 0, repeats >= 1"
        )

    nodes, edges = TOPOLOGIES[args.topology]

    if args.node_id not in nodes:
        parser.error(
            f"Node {args.node_id!r} is not in topology "
            f"{args.topology!r}; choose from {nodes}"
        )

    if args.lite_profile and args.backend != "rust":
        if not all(
            callable(getattr(LiteNetwork, name, None))
            for name in ("start_profile", "stop_profile")
        ):
            parser.error("--lite-profile requires the instrumented " "conops/lite.py")

    if args.rust_profile:
        os.environ["CONOPS_RUST_PROFILE"] = "1"

    neighbors = Graph(nodes, edges)[args.node_id]

    results: dict[str, list[float]] = {
        "Rust": [],
        "Lite": [],
    }

    print(
        f"Node {args.node_id} | "
        f"topology={args.topology} | "
        f"backend={args.backend} | "
        f"neighbors={list(neighbors)} | "
        f"group={args.group_id}",
        flush=True,
    )

    for repeat in range(args.repeats):

        if args.backend == "rust":
            benchmarks: list[tuple[str, Callable[..., BenchmarkResult]]] = [
                ("Rust", benchmark_rust),
            ]

        elif args.backend == "lite":
            benchmarks = [
                ("Lite", benchmark_lite),
            ]

        else:
            benchmarks = (
                [
                    ("Rust", benchmark_rust),
                    ("Lite", benchmark_lite),
                ]
                if repeat % 2 == 0
                else [
                    ("Lite", benchmark_lite),
                    ("Rust", benchmark_rust),
                ]
            )

        for name, worker in benchmarks:
            group_id = f"{args.group_id}-" f"{repeat}-{name.lower()}"

            print(
                f"Node {args.node_id} | "
                f"Run {repeat + 1:02d} | "
                f"{name}: connecting...",
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
                f"Node {args.node_id} | "
                f"Run {repeat + 1:02d} | "
                f"{name:4s} | "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )

            if name == "Lite" and profile_stats is not None:
                print_lite_profile(profile_stats)

    print(f"\nNode {args.node_id} summary " f"(median of local elapsed times)")

    for name, times in results.items():
        if not times:
            continue

        print(
            f"{name:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, "
            f"max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
