import argparse
import time
from statistics import median
from typing import cast

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


def benchmark(
    kind,
    node_id,
    neighbors,
    group_id,
    n_state,
    n_iter,
    n_warmup,
):
    if kind == "Rust":
        network = Network(
            node_id=node_id,
            neighbors=neighbors,
            group_id=group_id,
        )
    else:
        network = LiteNetwork(
            node_id=node_id,
            neighbors=neighbors,
            group_id=group_id,
        )

    try:
        if kind == "Rust":
            network = cast(Network, network)
            network.start()
            network.wait_connected(timeout=120)

        npr.seed(int(node_id))
        x = npr.uniform(-100.0, 100.0, n_state)
        y = npr.uniform(-100.0, 100.0, n_state)

        for k in range(n_warmup):
            if kind == "Rust":
                network.exchange(x, key=f"warmup-{k}")
            else:
                network.exchange(x, key=f"warmup-{k}")

        start = time.perf_counter()

        for k in range(n_iter - 1):
            if kind == "Rust":
                x = x - 0.45 * network.laplacian(x, key=(k, "x"))
                y = y - 0.20 * network.laplacian(y, key=(k, "y"))
            else:
                x = x - 0.45 * network.laplacian(x, key=(k, "x"))
                y = y - 0.20 * network.laplacian(y, key=(k, "y"))

        return time.perf_counter() - start

    finally:
        network.close()


def main():
    parser = argparse.ArgumentParser(description="ConOps five-node ring LAN benchmark")
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_state", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)

    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Invalid benchmark parameters")

    neighbors = Graph(NODES, EDGES)[args.node_id]
    results = {"Rust": [], "Lite": []}

    print(
        f"Node {args.node_id} | ring | "
        f"neighbors={list(neighbors)} | "
        f"group={args.group_id}",
        flush=True,
    )

    for repeat in range(args.repeats):
        order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")

        for kind in order:
            group_id = f"{args.group_id}-{repeat}-{kind.lower()}"

            print(
                f"Node {args.node_id} | "
                f"Run {repeat + 1:02d} | "
                f"{kind}: connecting...",
                flush=True,
            )

            elapsed = benchmark(
                kind,
                args.node_id,
                neighbors,
                group_id,
                args.n_state,
                args.n_iter,
                args.n_warmup,
            )
            results[kind].append(elapsed)

            print(
                f"Node {args.node_id} | "
                f"Run {repeat + 1:02d} | "
                f"{kind:4s} | elapsed={elapsed:.3f}s",
                flush=True,
            )

    print(f"\nNode {args.node_id} summary " f"(median of local elapsed times)")

    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, "
            f"max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
