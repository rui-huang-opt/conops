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
    n_state: int,
    alpha1: float,
    alpha2: float,
    n_iter: int = 50,
) -> tuple[npt.NDArray, npt.NDArray]:
    network = Network(node_id=idx, neighbors=neighbors)

    x = np.zeros((n_iter, n_state))
    y = np.zeros((n_iter, n_state))

    npr.seed(int(idx))
    x[0] = npr.uniform(-100.0, 100.0, n_state)
    y[0] = npr.uniform(-100.0, 100.0, n_state)

    try:
        network.start()
        network.wait_connected(timeout=30)

        for k in range(n_iter - 1):
            x[k + 1] = x[k] - alpha1 * network.laplacian(x[k], key=(k, "x"))
            y[k + 1] = y[k] - alpha2 * network.laplacian(y[k], key=(k, "y"))

    finally:
        network.close()

    return x, y


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_NODES) as executor:
        tasks = [
            executor.submit(
                consensus,
                str(i + 1),
                ring[str(i + 1)],
                N_STATE,
                0.35,
                0.1,
            )
            for i in range(N_NODES)
        ]
        results = [task.result() for task in tasks]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    _, ax1 = plt.subplots()

    for i in range(N_NODES):
        x = results[i][0]
        for j in range(N_STATE):
            (line,) = ax1.plot(x[:, j], color=colors[i])
            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("x Value")
    ax1.grid()
    ax1.legend()

    _, ax2 = plt.subplots()

    for i in range(N_NODES):
        y = results[i][1]
        for j in range(N_STATE):
            (line,) = ax2.plot(y[:, j], color=colors[i])
            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("y Value")
    ax2.legend()
    ax2.grid()

    plt.show()


if __name__ == "__main__":
    main()
