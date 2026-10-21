import logging

logging.basicConfig(level=logging.INFO)

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.typing as npt

from conops import Graph, Network

N_STATE = 3

NODES = ["1", "2", "3", "4", "5"]
N_NODES = len(NODES)

RING_EDGES = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]
STAR_EDGES = [("1", "2"), ("1", "3"), ("1", "4"), ("1", "5")]

ring = Graph(NODES, RING_EDGES)
star = Graph(NODES, STAR_EDGES)


def consensus(
    idx: str,
    neighbors: dict[str, float],
    group_id: str,
    n_state: int,
    alpha: float,
    n_iter: int = 50,
) -> npt.NDArray[np.float64]:
    network = Network(node_id=idx, neighbors=neighbors, group_id=group_id)

    x = np.zeros((n_iter, n_state))
    npr.seed(int(idx))
    x[0] = npr.uniform(-100.0, 100.0, n_state)

    try:
        network.start()
        network.wait_connected(timeout=30)

        for k in range(n_iter - 1):
            x[k + 1] = x[k] - alpha * network.laplacian(x[k], key=k)

    finally:
        network.close()

    return x


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_NODES) as executor:
        ring_tasks = [
            executor.submit(consensus, idx, ring[idx], "ring", N_STATE, 0.35)
            for idx in NODES
        ]
        ring_states = [task.result() for task in ring_tasks]

        star_tasks = [
            executor.submit(consensus, idx, star[idx], "star", N_STATE, 0.35)
            for idx in NODES
        ]
        star_states = [task.result() for task in star_tasks]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    _, ax1 = plt.subplots()

    for i in range(N_NODES):
        states = ring_states[i]
        for j in range(N_STATE):
            (line,) = ax1.plot(states[:, j], color=colors[i])
            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("State Value")
    ax1.grid()
    ax1.legend()
    ax1.set_title(f"Ring Graph Consensus (Nodes={N_NODES}, State Dim={N_STATE})")

    _, ax2 = plt.subplots()

    for i in range(N_NODES):
        states = star_states[i]
        for j in range(N_STATE):
            (line,) = ax2.plot(states[:, j], color=colors[i])
            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax2.set_title(f"Star Graph Consensus (Nodes={N_NODES}, State Dim={N_STATE})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("State Value")
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    main()
