"""
Benchmark ConOps exchange() on a persistent six-node LAN topology.

Run once per node with the same options and a different --node_id.
Rust and Lite connect once and reuse their connections across repeats.
The --copy / --no-copy option applies only to Rust.
"""

import argparse
import time
from statistics import median

import numpy.random as npr
import zmq

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


class LanBarrier:
    def __init__(
        self,
        node_id: str,
        host: str,
        port: int = 47000,
        parties: int = 6,
        timeout: float = 120.0,
    ):
        self.leader = node_id == "1"
        self.parties = parties
        self.round = 0

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER if self.leader else zmq.REQ)

        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))

        if self.leader:
            self.socket.bind(f"tcp://0.0.0.0:{port}")
        else:
            self.socket.setsockopt(zmq.IDENTITY, node_id.encode())
            self.socket.connect(f"tcp://{host}:{port}")

    def wait(self) -> None:
        token = self.round.to_bytes(8, "big")

        if self.leader:
            peers = []
            seen = set()

            for _ in range(self.parties - 1):
                identity, delimiter, message = self.socket.recv_multipart()

                if delimiter != b"" or message != token:
                    raise RuntimeError("Barrier round mismatch")
                if identity in seen:
                    raise RuntimeError("Duplicate barrier participant")

                seen.add(identity)
                peers.append(identity)

            for identity in peers:
                self.socket.send_multipart([identity, b"", token])

        else:
            self.socket.send(token)

            if self.socket.recv() != token:
                raise RuntimeError("Barrier round mismatch")

        self.round += 1

    def close(self) -> None:
        self.socket.close(0)
        self.context.term()


def measure_rust(
    barrier: LanBarrier,
    network: Network,
    node_id: str,
    n_state: int,
    n_iter: int,
    n_warmup: int,
    copy: bool,
) -> float:
    npr.seed(int(node_id))
    x = npr.uniform(-100.0, 100.0, n_state)
    y = npr.uniform(-100.0, 100.0, n_state)

    for k in range(n_warmup):
        network.exchange(x, copy=copy)

    barrier.wait()

    before = time.perf_counter()
    for k in range(n_iter - 1):
        x = x - 0.25 * network.laplacian(x, copy=copy)
        y = y - 0.20 * network.laplacian(y, copy=copy)

    elapsed = time.perf_counter() - before

    barrier.wait()

    return elapsed


def measure_lite(
    barrier: LanBarrier,
    network: LiteNetwork,
    node_id: str,
    n_state: int,
    n_iter: int,
    n_warmup: int,
) -> float:
    npr.seed(int(node_id))
    x = npr.uniform(-100.0, 100.0, n_state)
    y = npr.uniform(-100.0, 100.0, n_state)

    for k in range(n_warmup):
        network.exchange(x)

    barrier.wait()

    before = time.perf_counter()
    for k in range(n_iter - 1):
        x = x - 0.25 * network.laplacian(x)
        y = y - 0.20 * network.laplacian(y)

    elapsed = time.perf_counter() - before

    barrier.wait()

    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent six-node LAN exchange benchmark"
    )
    parser.add_argument("--node_id", required=True, choices=NODES)
    parser.add_argument("--group_id", default="lan-benchmark")
    parser.add_argument("--n_state", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rust only: use a send snapshot; --no-copy borrows the NumPy payload.",
    )
    parser.add_argument("--barrier-host", required=True)
    parser.add_argument("--barrier-port", type=int, default=47000)
    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n_state >= 1, n_iter >= 2, n_warmup >= 0, repeats >= 1")

    neighbors = Graph(NODES, EDGES)[args.node_id]
    results: dict[str, list[float]] = {"Rust": [], "Lite": []}
    rust: Network | None = None
    lite: LiteNetwork | None = None
    rust_started = False

    barrier = LanBarrier(
        node_id=args.node_id,
        host=args.barrier_host,
        port=args.barrier_port,
        parties=len(NODES),
    )

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
                    barrier=barrier,
                    network=rust,
                    node_id=args.node_id,
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
                    barrier=barrier,
                    network=lite,
                    node_id=args.node_id,
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
                    barrier=barrier,
                    network=lite,
                    node_id=args.node_id,
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
                    barrier=barrier,
                    network=rust,
                    node_id=args.node_id,
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
            try:
                if rust is not None:
                    rust.close(timeout=10.0 if rust_started else 0.0)
            finally:
                if barrier is not None:
                    barrier.close()

    print(f"\nNode {args.node_id} summary (median of local elapsed times)")
    for kind, times in results.items():
        print(
            f"{kind:4s}: {median(times):.3f}s "
            f"(min={min(times):.3f}s, max={max(times):.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
