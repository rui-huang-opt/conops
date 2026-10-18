from logging import getLogger
from typing import KeysView
from collections import deque

import numpy as np
import numpy.typing as npt

logger = getLogger(__name__)


def is_symmetric_doubly_stochastic(matrix: npt.NDArray[np.float64]) -> bool:
    """
    Check if a matrix is symmetric doubly stochastic.

    Parameters
    ----------
    matrix : NDArray[float64]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is symmetric doubly stochastic, False otherwise.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T):
        return False
    if np.any(matrix < 0):
        return False

    return np.allclose(np.sum(matrix, axis=0), 1.0)


class Graph:
    """
    A lightweight helper class for constructing node-wise inputs for `Network`.

    In decentralized optimization, communication topologies are often specified
    globally by a set of nodes and edges or by a weighted mixing matrix `W`.
    This class provides a simple graph representation for such topology
    descriptions.

    After a `Graph` is initialized, `graph[i]` returns the local information
    needed to initialize the `Network` of node `i`, namely its neighbors
    and the corresponding communication weights.

    This class is only a convenience utility and is not part of the core
    optimization logic.

    Parameters
    ----------
    nodes : list[str] | None, optional
        An iterable of node names to initialize the graph. If None, the graph starts with no nodes.

    edges : list[tuple[str, str]] | None, optional
        An iterable of edges represented as tuples of node names (u, v). If None, the graph starts with no edges.

    Attributes
    ----------
    nodes : KeysView[str]
        A view of the node names in the graph.

    edges : list[tuple[str, str]]
        A list of edges in the graph represented as tuples of node names (u, v).

    num_nodes : int
        The total number of nodes in the graph.

    adj : dict[str, dict[str, float]]
        The adjacency representation of the graph, where each key is a node name,
        and the value is a dictionary mapping neighboring node names to their weights.

    Notes
    -----
    - The `from_mixing_matrix` method provides a convenient way to create a graph from a mixing matrix,
      ensuring the matrix is symmetric and double-stochastic.
    """

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        nodes_ = nodes or []
        edges_ = edges or []
        self._adj: dict[str, dict[str, float]] = {u: {} for u in nodes_}

        for u, v in edges_:
            if u not in self._adj:
                self._adj[u] = {}
            if v not in self._adj:
                self._adj[v] = {}

            self._adj[u][v] = 0.0
            self._adj[v][u] = 0.0

        for n_i in self._adj.values():
            deg = len(n_i)
            if deg > 0:
                weight = 1.0 / deg
                for j in n_i:
                    n_i[j] = weight

    @classmethod
    def from_mixing_matrix(
        cls,
        mixing_matrix: npt.NDArray[np.float64],
        nodes: list[str] | None = None,
    ) -> "Graph":
        """
        Create a Graph instance from a mixing matrix.
        The mixing matrix must be symmetric doubly stochastic.

        Parameters
        ----------
        mixing_matrix : NDArray[float64]
            A square matrix representing the mixing coefficients between nodes.

        nodes : list[str], optional
            A list of self defined node names. If not provided,
            nodes will be named sequentially as "1", "2", ..., "n".
        """
        if not is_symmetric_doubly_stochastic(mixing_matrix):
            err_msg = "The mixing matrix must be symmetric doubly stochastic."
            logger.error(err_msg)
            raise ValueError(err_msg)

        n = len(mixing_matrix)

        if nodes is None:
            nodes = [str(i + 1) for i in range(n)]
        elif len(nodes) != n:
            err_msg = "The length of nodes must match the size of the mixing matrix."
            logger.error(err_msg)
            raise ValueError(err_msg)

        adj: dict[str, dict[str, float]] = {u: {} for u in nodes}

        nz_rows, nz_cols = np.nonzero(mixing_matrix)
        mask = nz_rows < nz_cols

        for i, j in zip(nz_rows[mask], nz_cols[mask]):
            weight = mixing_matrix[i, j].item()
            u = nodes[i]
            v = nodes[j]
            adj[u][v] = weight
            adj[v][u] = weight

        graph = cls()
        graph._adj = adj

        return graph

    def __getitem__(self, key: str) -> dict[str, float]:
        """
        Get the neighbors and their weights for a given node.
        """
        return self._adj[key]

    @property
    def nodes(self) -> KeysView[str]:
        return self._adj.keys()

    @property
    def edges(self) -> list[tuple[str, str]]:
        edges = []
        for u in self._adj:
            for v in self._adj[u]:
                if (v, u) not in edges:
                    edges.append((u, v))
        return edges

    @property
    def num_nodes(self) -> int:
        return len(self._adj)

    @property
    def adj(self) -> dict[str, dict[str, float]]:
        return self._adj

    def is_connected(self) -> bool:
        """
        Check if the graph is connected.

        Returns
        -------
        bool
            True if the graph is connected, False otherwise.
        """
        if not self._adj:
            return True

        start = next(iter(self._adj))
        visited = {start}

        q = deque([start])

        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)

        return len(visited) == len(self._adj)
