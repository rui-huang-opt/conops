import argparse
import time
import logging

import numpy.random as npr

from conops import Graph, FastNetwork

logging.basicConfig(level=logging.INFO)

NODES = ["1", "2", "3", "4", "5"]
EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(NODES, EDGES)


class Args(argparse.Namespace):
    node_id: str
    n_iter: int
    n_state: int
    interval: float = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id", type=str, choices=NODES)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_state", type=int, default=1000000)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args(namespace=Args())

    network = FastNetwork(node_id=args.node_id, neighbors=graph[args.node_id])

    try:
        npr.seed(int(args.node_id))  # Ensure reproducibility for each node
        x = npr.uniform(-100.0, 100.0, args.n_state)
        y = npr.uniform(-100.0, 100.0, args.n_state)

        k = 0

        while k < args.n_iter - 1:
            x = x - network.laplacian(x) * 0.45
            y = y - network.laplacian(y) * 0.2

            k = k + 1

            time.sleep(args.interval)

        print(f"[{args.node_id}] - Round {k}: x = {x}, y = {y}", flush=True)

    finally:
        network.close()


if __name__ == "__main__":
    main()
