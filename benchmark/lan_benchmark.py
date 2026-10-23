"""
Benchmark communication on a persistent six-node LAN topology.

Run this script once on each of the six nodes, with a different --node_id
and identical benchmark options. Each backend connects just once. Repeats
reset the local arrays and measure the same existing connections; startup,
initialization, and warmup are excluded from each timed interval.

Examples
--------
python benchmark_lan_persistent.py --node_id 1 --n_state 5 --n_iter 1000
python benchmark_lan_persistent.py --node_id 2 --n_state 2000000 --n_iter 100 --repeats 5
"""

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


def measure_round(
    network: Network | LiteNetwork,
    node_id: str,
    repeat: int,
    n_state: int,
    n_iter: int,
    n_warmup: int,
) -> float:
    # Match the original benchmark's NumPy values. Every repeat and backend
    # starts from the same local x/y states; none of this is timed.
    npr.seed(int(node_id))
    x = npr.uniform(-100.0, 100.0, n_state)
    y = npr.uniform(-100.0, 100.0, n_state)

    # Same message size as the timed exchanges. Keys must remain unique across
    # repeats because the Rust backend does not allow reusing exchange keys.
    for k in range(n_warmup):
        network.exchange(x, key=f"warmup-{repeat}-{k}")

    start = time.perf_counter()
    for k in range(n_iter - 1):
        x = x - 0.45 * network.laplacian(x, key=(k, f"{repeat}-x"))
        y = y - 0.20 * network.laplacian(y, key=(k, f"{repeat}-y"))

    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent-connection, six-node LAN communication benchmark"
    )
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_state", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_state >= 1, n_iter >= 2, n_warmup >= 0, repeats >= 1")

    neighbors = Graph(NODES, EDGES)[args.node_id]
    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    networks: dict[str, Network | LiteNetwork] = {}

    print(
        f"Node {args.node_id} | six-node graph | "
        f"neighbors={list(neighbors)} | group={args.group_id}",
        flush=True,
    )

    try:
        # Both backends remain connected until ALL repeats have finished.
        # Each backend has its own discovery group so they cannot interfere.
        print(f"Node {args.node_id} | Rust: connecting once...", flush=True)
        rust = Network(
            node_id=args.node_id,
            neighbors=neighbors,
            group_id=f"{args.group_id}-rust",
        )
        networks["Rust"] = rust
        rust.start()
        rust.wait_connected(timeout=120)

        print(f"Node {args.node_id} | Lite: connecting once...", flush=True)
        lite = LiteNetwork(
            node_id=args.node_id,
            neighbors=neighbors,
            group_id=f"{args.group_id}-lite",
        )
        networks["Lite"] = lite

        print(f"Node {args.node_id} | both backends connected", flush=True)

        for repeat in range(args.repeats):
            order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")
            for kind in order:
                elapsed = measure_round(
                    networks[kind],
                    args.node_id,
                    repeat,
                    args.n_state,
                    args.n_iter,
                    args.n_warmup,
                )
                results[kind].append(elapsed)
                print(
                    f"Node {args.node_id} | Run {repeat + 1:02d} | "
                    f"{kind:4s} | elapsed={elapsed:.3f}s",
                    flush=True,
                )

    finally:
        # Keep both network connections alive throughout the measurements,
        # including periods when one of the backends is idle.
        for network in reversed(list(networks.values())):
            network.close()

    print(f"\nNode {args.node_id} summary (median of local elapsed times)")
    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
