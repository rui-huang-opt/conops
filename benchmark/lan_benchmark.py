"""
Benchmark ConOps communication performance across two LAN machines.

This benchmark compares Network (Rust) and LiteNetwork (ZeroMQ).

Ray is required for distributed process management. To run the benchmark,
install Ray on both machines and configure a Ray cluster before execution.

Run on the head node:

    python benchmark/ray_lan_benchmark.py
    python benchmark/ray_lan_benchmark.py --static
"""

import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import argparse
import time
from statistics import median
from typing import Literal
from types import SimpleNamespace

import numpy as np
import ray
from ray.actor import ActorProxy
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
NODE_IPS = {
    "1": "192.168.1.101",
    "2": "192.168.1.101",
    "3": "192.168.1.101",
    "4": "192.168.1.111",
    "5": "192.168.1.111",
    "6": "192.168.1.111",
}
NODE_PORTS = {node: 48000 + int(node) for node in NODES}


class BenchmarkNode:
    """One persistent process with one Rust and one Lite connection."""

    def __init__(self, node_id: str, config: dict):
        self.node_id = node_id
        self.args = SimpleNamespace(**config)
        self.neighbors = Graph(NODES, EDGES)[node_id]
        self.rust: Network | None = None
        self.lite: LiteNetwork | None = None
        self.rust_started = False
        self.prepared_kind: str | None = None

    @ray.method
    def connect(self) -> None:
        args = self.args
        peer_endpoints = (
            {
                peer_id: f"tcp://{NODE_IPS[peer_id]}:{NODE_PORTS[peer_id]}"
                for peer_id in [self.node_id, *self.neighbors]
            }
            if args.static
            else None
        )
        print(
            f"Node {self.node_id} | host={NODE_IPS[self.node_id]} | "
            f"neighbors={list(self.neighbors)} | "
            f"Rust copy={args.copy} | static={args.static}",
            flush=True,
        )

        self.rust = Network(
            node_id=self.node_id,
            neighbors=self.neighbors,
            group_id=f"{args.group_id}-rust",
            port=NODE_PORTS[self.node_id] if args.static else 0,
            peer_endpoints=peer_endpoints,
        )
        self.rust.start()
        self.rust_started = True
        self.rust.wait_connected(timeout=120)

        self.lite = LiteNetwork(
            node_id=self.node_id,
            neighbors=self.neighbors,
            group_id=f"{args.group_id}-lite",
        )
        print(f"Node {self.node_id} | both backends connected", flush=True)

    @ray.method
    def warmup(self, kind: str) -> None:
        rng = np.random.default_rng(int(self.node_id))
        x = rng.uniform(-100.0, 100.0, self.args.n_state)
        self.prepared_kind = None

        if kind == "Rust":
            assert self.rust is not None
            for _ in range(self.args.n_warmup):
                self.rust.exchange(x, copy=self.args.copy)
        elif kind == "Lite":
            assert self.lite is not None
            for _ in range(self.args.n_warmup):
                self.lite.exchange(x)
        else:
            raise ValueError(f"Unknown backend: {kind}")

        self.prepared_kind = kind

    @ray.method
    def measure(self, kind: str) -> float:
        if kind != self.prepared_kind:
            raise RuntimeError(f"Node {self.node_id} was not warmed up for {kind}")

        rng = np.random.default_rng(int(self.node_id))
        x = rng.uniform(-100.0, 100.0, self.args.n_state)
        y = rng.uniform(-100.0, 100.0, self.args.n_state)

        start = time.perf_counter()
        if kind == "Rust":
            assert self.rust is not None
            for _ in range(self.args.n_iter - 1):
                x = x - 0.25 * self.rust.laplacian(x, copy=self.args.copy)
                y = y - 0.20 * self.rust.laplacian(y, copy=self.args.copy)
        else:
            assert self.lite is not None
            for _ in range(self.args.n_iter - 1):
                x = x - 0.25 * self.lite.laplacian(x)
                y = y - 0.20 * self.lite.laplacian(y)
        elapsed = time.perf_counter() - start

        self.prepared_kind = None
        print(f"Node {self.node_id} | {kind} | elapsed={elapsed:.3f}s", flush=True)
        return elapsed

    @ray.method
    def close(self) -> None:
        try:
            if self.lite is not None:
                self.lite.close(timeout=10.0)
                self.lite = None
        finally:
            if self.rust is not None:
                self.rust.close(timeout=10.0 if self.rust_started else 0.0)
                self.rust = None


RemoteBenchmarkNode = ray.remote(BenchmarkNode)


def call_all(
    actors: dict[str, ActorProxy[BenchmarkNode]],
    method: Literal["connect", "warmup", "measure"],
    kind: str = "",
    timeout: float | None = None,
) -> dict:
    if method == "connect":
        refs = {n: a.connect.remote() for n, a in actors.items()}
    elif method == "warmup":
        refs = {n: a.warmup.remote(kind) for n, a in actors.items()}
    else:
        refs = {n: a.measure.remote(kind) for n, a in actors.items()}

    values = ray.get(list(refs.values()), timeout=timeout)
    return dict(zip(refs.keys(), values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", default="lan-benchmark")
    parser.add_argument("--n-state", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--copy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Disable Rust UDP scouting",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.n_state < 1 or args.n_iter < 2 or args.n_warmup < 0 or args.repeats < 1:
        parser.error("Require n-state >= 1, n-iter >= 2, n-warmup >= 0, repeats >= 1")
    if args.timeout <= 0:
        parser.error("Timeout must be positive")

    ray.init(address="auto", log_to_driver=True)
    actors: dict[str, ActorProxy[BenchmarkNode]] = {}

    try:
        alive_nodes = {
            node["NodeManagerAddress"]: node["NodeID"]
            for node in ray.nodes()
            if node["Alive"]
        }

        missing = sorted(set(NODE_IPS.values()) - set(alive_nodes))
        if missing:
            raise RuntimeError(
                f"Missing Ray nodes: {missing}; available: {sorted(alive_nodes)}"
            )

        print(f"Ray cluster: {sorted(alive_nodes)}", flush=True)
        print(
            f"Benchmark: n_state={args.n_state}, n_iter={args.n_iter}, "
            f"n_warmup={args.n_warmup}, repeats={args.repeats}, "
            f"copy={args.copy}, static={args.static}",
            flush=True,
        )

        # Create six persistent actors on the specified machines.
        for node_id in NODES:
            ip = NODE_IPS[node_id]
            actors[node_id] = RemoteBenchmarkNode.options(
                num_cpus=0.1,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=alive_nodes[ip],
                    soft=False,
                ),
            ).remote(node_id, vars(args))

        # Establish all connections concurrently.
        call_all(actors, "connect", timeout=args.timeout)

        results = {node_id: {"Rust": [], "Lite": []} for node_id in NODES}

        for repeat in range(args.repeats):
            order = ("Rust", "Lite") if repeat % 2 == 0 else ("Lite", "Rust")

            for kind in order:
                call_all(actors, "warmup", kind, timeout=args.timeout)

                elapsed = call_all(actors, "measure", kind, timeout=args.timeout)

                for node_id in NODES:
                    results[node_id][kind].append(elapsed[node_id])

                print(
                    f"Run {repeat + 1:02d} | {kind:4s} | "
                    f"slowest={max(elapsed.values()):.3f}s | "
                    + ", ".join(f"node {n}={elapsed[n]:.3f}s" for n in NODES),
                    flush=True,
                )

        print("\n=== Per-node median elapsed time ===")
        for node_id in NODES:
            print(
                f"Node {node_id}: "
                + " | ".join(
                    f"{kind}={median(results[node_id][kind]):.3f}s"
                    for kind in ("Rust", "Lite")
                ),
                flush=True,
            )

        print("\n=== Six-node benchmark (slowest local time per round) ===")
        for kind in ("Rust", "Lite"):
            slowest_each_repeat = [
                max(results[n][kind][repeat] for n in NODES)
                for repeat in range(args.repeats)
            ]

            print(
                f"{kind:4s}: median={median(slowest_each_repeat):.3f}s "
                f"(min={min(slowest_each_repeat):.3f}s, "
                f"max={max(slowest_each_repeat):.3f}s)",
                flush=True,
            )

    finally:
        if actors:
            try:
                ray.get(
                    [actor.close.remote() for actor in actors.values()],
                    timeout=10,
                )
            except Exception:
                pass

            for actor in actors.values():
                try:
                    ray.kill(actor)
                except Exception:
                    pass

        ray.shutdown()


if __name__ == "__main__":
    main()
