"""Benchmark ConOps exchange() on a persistent six-node LAN topology.

Run once per node with the same options and a different --node_id.
Rust and Lite connect once and reuse their connections across repeats.
The --copy / --no-copy option applies only to Rust.
"""

import os
import argparse
import time
from statistics import median

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5", "6"]
EDGES = [
    ("1", "2"),
    ("2", "3"),
    ("3", "4"),
    ("4", "5"),
    ("5", "6"),
    ("6", "1"),
    ("1", "4"),
    ("2", "5"),
    ("3", "6"),
]


def measure_rust(
    network: Network,
    node_id: str,
    repeat: int,
    n_state: int,
    n_iter: int,
    n_warmup: int,
    copy: bool,
) -> float:
    npr.seed(int(node_id))
    x = npr.uniform(-100.0, 100.0, n_state)
    y = npr.uniform(-100.0, 100.0, n_state)

    for k in range(n_warmup):
        network.exchange(x, key=f"warmup-{repeat}-{k}", copy=copy)

    before = time.perf_counter()
    for k in range(n_iter - 1):
        x = x - 0.45 * network.laplacian(x, key=(k, f"{repeat}-x"), copy=copy)
        y = y - 0.20 * network.laplacian(y, key=(k, f"{repeat}-y"), copy=copy)
    return time.perf_counter() - before


def measure_lite(
    network: LiteNetwork,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent six-node LAN exchange benchmark"
    )
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_state", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rust only: use a send snapshot; --no-copy borrows the NumPy payload.",
    )
    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_state >= 1, n_iter >= 2, n_warmup >= 0, repeats >= 1")

    neighbors = Graph(NODES, EDGES)[args.node_id]
    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    rust: Network | None = None
    lite: LiteNetwork | None = None
    rust_started = False

    print(
        f"Node {args.node_id} | six-node graph | neighbors={list(neighbors)} | "
        f"Rust copy={args.copy} | group={args.group_id}",
        flush=True,
    )

    try:
        print(f"Node {args.node_id} | Rust: connecting once...", flush=True)
        rust = Network(
            node_id=args.node_id,
            neighbors=neighbors,
            group_id=f"{args.group_id}-rust",
        )
        rust.start()
        rust_started = True
        rust.wait_connected(timeout=120)

        print(f"Node {args.node_id} | Lite: connecting once...", flush=True)
        lite = LiteNetwork(
            node_id=args.node_id,
            neighbors=neighbors,
            group_id=f"{args.group_id}-lite",
        )
        print(f"Node {args.node_id} | both backends connected", flush=True)

        for repeat in range(args.repeats):
            if repeat % 2 == 0:
                rust_elapsed = measure_rust(
                    network=rust,
                    node_id=args.node_id,
                    repeat=repeat,
                    n_state=args.n_state,
                    n_iter=args.n_iter,
                    n_warmup=args.n_warmup,
                    copy=args.copy,
                )
                results["Rust"].append(rust_elapsed)
                print(
                    f"Node {args.node_id} | Run {repeat + 1:02d} | "
                    f"Rust | elapsed={rust_elapsed:.3f}s",
                    flush=True,
                )

                lite_elapsed = measure_lite(
                    network=lite,
                    node_id=args.node_id,
                    repeat=repeat,
                    n_state=args.n_state,
                    n_iter=args.n_iter,
                    n_warmup=args.n_warmup,
                )
                results["Lite"].append(lite_elapsed)
                print(
                    f"Node {args.node_id} | Run {repeat + 1:02d} | "
                    f"Lite | elapsed={lite_elapsed:.3f}s",
                    flush=True,
                )
            else:
                lite_elapsed = measure_lite(
                    network=lite,
                    node_id=args.node_id,
                    repeat=repeat,
                    n_state=args.n_state,
                    n_iter=args.n_iter,
                    n_warmup=args.n_warmup,
                )
                results["Lite"].append(lite_elapsed)
                print(
                    f"Node {args.node_id} | Run {repeat + 1:02d} | "
                    f"Lite | elapsed={lite_elapsed:.3f}s",
                    flush=True,
                )

                rust_elapsed = measure_rust(
                    network=rust,
                    node_id=args.node_id,
                    repeat=repeat,
                    n_state=args.n_state,
                    n_iter=args.n_iter,
                    n_warmup=args.n_warmup,
                    copy=args.copy,
                )
                results["Rust"].append(rust_elapsed)
                print(
                    f"Node {args.node_id} | Run {repeat + 1:02d} | "
                    f"Rust | elapsed={rust_elapsed:.3f}s",
                    flush=True,
                )
    finally:
        try:
            if lite is not None:
                lite.close(timeout=10.0)
        finally:
            if rust is not None:
                rust.close(timeout=10.0 if rust_started else 0.0)

    print(f"\nNode {args.node_id} summary (median of local elapsed times)")
    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
