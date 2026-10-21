from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import numpy.typing as npt

from conops import Graph, Network
from conops.transform import UniformQuantize

N_STATE = 3

L = np.array(
    [
        [2, -1, 0, 0, -1],
        [-1, 2, -1, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 0, -1, 2, -1],
        [-1, 0, 0, -1, 2],
    ]
)
N_NODES = L.shape[0]
W = np.eye(N_NODES) - L * 0.45

graph = Graph.from_mixing_matrix(W)


def consensus(
    idx: str,
    neighbors: dict[str, float],
    n_state: int,
    n_iter: int = 50,
) -> npt.NDArray[np.float64]:
    x = np.zeros((n_iter, n_state))
    npr.seed(int(idx))
    x[0] = npr.uniform(-100.0, 100.0, n_state)

    network = Network(node_id=idx, neighbors=neighbors, transform=UniformQuantize())

    try:
        network.start()
        network.wait_connected(timeout=30)

        for k in range(n_iter - 1):
            x[k + 1] = x[k] - 0.45 * network.laplacian(exchange_id=k, state=x[k])
    finally:
        network.close()

    return x


def main() -> None:
    with ProcessPoolExecutor(max_workers=N_NODES) as executor:
        tasks = [
            executor.submit(consensus, str(i + 1), graph[str(i + 1)], N_STATE)
            for i in range(N_NODES)
        ]
        node_states = [task.result() for task in tasks]

    _, ax = plt.subplots()

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i in range(N_NODES):
        states = node_states[i]
        for j in range(N_STATE):
            (line,) = ax.plot(states[:, j], color=colors[i])
            if j == 0:
                line.set_label(f"Node {i + 1}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("State Value")
    ax.set_title(f"Quantized Gossip Algorithm States for {N_NODES} Nodes")
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
