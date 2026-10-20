import argparse
import logging
import time

import numpy.random as npr

from conops.network import Network
from conops.graph import Graph

logging.basicConfig(level=logging.INFO)

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)


class Args(argparse.Namespace):
    node_id: str
    n_iter: int
    n_state: int
    interval: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id", type=str, choices=NODES)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1000000)
    parser.add_argument("--interval", type=float, default=0.1)

    args = parser.parse_args(namespace=Args())

    network = Network(
        node_id=args.node_id,
        neighbors=graph[args.node_id],
        group_id="consensus-test",
    )

    try:
        network.start()
        network.wait_connected()

        print(
            f"[{args.node_id}] Connected: {network.peers()}",
            flush=True,
        )

        npr.seed(int(args.node_id))
        x = npr.uniform(-100.0, 100.0, args.n_state)
        y = npr.uniform(-100.0, 100.0, args.n_state)

        n_warmup = 10
        for exchange_id in range(n_warmup):
            network.exchange(exchange_id, x)

        before = time.perf_counter()

        for k in range(args.n_iter - 1):
            x = x - 0.45 * network.laplacian(
                exchange_id=n_warmup + 2 * k,
                state=x,
            )

            y = y - 0.2 * network.laplacian(
                exchange_id=n_warmup + 2 * k + 1,
                state=y,
            )

        elapsed = time.perf_counter() - before

        print(
            f"[{args.node_id}] - Final: x = {x}, y = {y} (elapsed {elapsed:.3f}s)",
            flush=True,
        )

    finally:
        network.close()


if __name__ == "__main__":
    main()
