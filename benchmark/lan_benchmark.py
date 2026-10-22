import argparse
import time
from statistics import median
from typing import cast

import numpy.random as npr

from conops import Graph, LiteNetwork, Network

NODES = ["1", "2", "3", "4", "5"]
EDGES = [
    ("1", "2"),
    ("2", "3"),
    ("3", "4"),
    ("4", "5"),
    ("5", "1"),
]

# NUC: 1, 2, 3
# HP EliteBook: 4, 5
PEER_ENDPOINTS = {
    "1": "tcp://192.168.55.51:50051",
    "2": "tcp://192.168.55.51:50052",
    "3": "tcp://192.168.55.51:50053",
    "4": "tcp://192.168.55.63:50051",
    "5": "tcp://192.168.55.63:50052",
}


def endpoints_for_run(
    repeat: int,
    kind: str,
) -> dict[str, str]:
    # Each backend/repeat uses a distinct set of ports.
    offset = (2 * repeat + (0 if kind == "Rust" else 1)) * 10

    endpoints = {}
    for node_id, endpoint in PEER_ENDPOINTS.items():
        host, port = endpoint.rsplit(":", 1)
        endpoints[node_id] = f"{host}:{int(port) + offset}"

    return endpoints


def benchmark(
    kind: str,
    node_id: str,
    neighbors: dict[str, float],
    group_id: str,
    n_state: int,
    n_iter: int,
    n_warmup: int,
    peer_endpoints: dict[str, str] | None,
) -> float:
    if peer_endpoints is not None:
        peer_endpoints = {
            peer_id: peer_endpoints[peer_id] for peer_id in (node_id, *neighbors)
        }

    if kind == "Rust":
        network = Network(
            node_id=node_id,
            neighbors=neighbors,
            group_id=group_id,
            peer_endpoints=peer_endpoints,
        )
    else:
        network = LiteNetwork(
            node_id=node_id,
            neighbors=neighbors,
            group_id=group_id,
            peer_endpoints=peer_endpoints,
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
            network.exchange(x, key=f"warmup-{k}")

        start = time.perf_counter()

        for k in range(n_iter - 1):
            x = x - 0.45 * network.laplacian(x, key=(k, "x"))
            y = y - 0.20 * network.laplacian(y, key=(k, "y"))

        return time.perf_counter() - start

    finally:
        network.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ConOps five-node ring LAN benchmark")

    parser.add_argument(
        "--node_id",
        required=True,
        choices=NODES,
    )
    parser.add_argument(
        "--group_id",
        default="lan-benchmark",
    )
    parser.add_argument(
        "--n_state",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1000,
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
        "--discovery",
        choices=("static", "mdns"),
        default="static",
        help="static: fixed IP; mdns: automatic discovery",
    )

    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Invalid benchmark parameters")

    neighbors = Graph(NODES, EDGES)[args.node_id]

    results = {
        "Rust": [],
        "Lite": [],
    }

    print(
        f"Node {args.node_id} | ring | "
        f"discovery={args.discovery} | "
        f"neighbors={list(neighbors)} | "
        f"group={args.group_id}",
        flush=True,
    )

    for repeat in range(args.repeats):
        order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")

        for kind in order:
            group_id = f"{args.group_id}-{repeat}-{kind.lower()}"

            peer_endpoints = (
                endpoints_for_run(repeat, kind) if args.discovery == "static" else None
            )

            print(
                f"Node {args.node_id} | "
                f"Run {repeat + 1:02d} | "
                f"{kind}: connecting..."
                + (
                    f" | endpoint={peer_endpoints[args.node_id]}"
                    if peer_endpoints
                    else ""
                ),
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
                peer_endpoints,
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
