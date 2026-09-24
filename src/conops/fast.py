import uuid
from json import loads
from logging import getLogger
from typing import KeysView
from urllib.parse import urlsplit

import pyre
import zmq
import numpy as np
from numpy.typing import NDArray

from .transform import Transform, Identity

logger = getLogger(__name__)


class FastNetwork:
    """
    Lightweight synchronous communication interface for fast state exchange.

    Unlike :class:`Network`, which uses a background communication thread
    and provides round-aware recovery, replay, and out-of-order message
    handling, ``FastNetwork`` implements only the synchronous communication
    path. Pyre is used for neighbor discovery during initialization, while
    exchanged states are transmitted directly through ZeroMQ ROUTER/DEALER
    sockets.

    This reduces protocol overhead at the cost of omitting the recovery
    semantics provided by :class:`Network`. ``FastNetwork`` is therefore
    intended for synchronized and performance-sensitive experiments where
    application-level recovery is not required.

    Parameters
    ----------
    node_id : str
        Unique identifier of the local node.

    neighbors : dict[str, float]
        Mapping from neighbor node IDs to their associated edge weights.

    namespace : str, optional
        Namespace used to isolate independent network instances.
        Defaults to ``"default"``.

    transform : Transform | None, optional
        Transformation applied to exchanged states. If ``None``,
        :class:`Identity` is used.

    context : zmq.SyncContext | None, optional
        ZeroMQ context used by the network. If ``None``, the process-wide
        shared context returned by :meth:`zmq.Context.instance` is used.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: dict[str, float],
        *,
        namespace: str = "default",
        transform: Transform | None = None,
        context: zmq.SyncContext | None = None,
    ) -> None:
        self._node_id = node_id
        self._namespace = namespace
        self._transform = transform if transform is not None else Identity()

        if context is None:
            self._context: zmq.SyncContext = zmq.Context.instance()
        else:
            self._context = context

        self._neighbors = neighbors.copy()
        self._weight = 1.0 - sum(neighbors.values())

        self._neighbor_id_bytes = [j.encode("utf-8") for j in self._neighbors]

        self._out_socket = self._context.socket(zmq.ROUTER)
        self._out_socket.setsockopt(zmq.LINGER, 0)

        identity = self._node_id.encode("utf-8")

        self._dealers: dict[str, zmq.SyncSocket] = {}
        for peer_id in self._neighbors:
            dealer = self._context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.IDENTITY, identity)

            self._dealers[peer_id] = dealer

        self._pyre_node = pyre.Pyre(self._node_id, ctx=self._context)

        try:
            port = self._out_socket.bind_to_random_port("tcp://*")

            self._pyre_node.set_header("port", str(port))
            self._pyre_node.join(self._namespace)
            self._pyre_node.start()

            endpoints = self._discover_neighbors()
            self._connect_to_neighbors(endpoints)

        except Exception:
            self.close()
            raise

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def degree(self) -> int:
        return len(self._neighbors)

    @property
    def neighbors(self) -> KeysView[str]:
        return self._neighbors.keys()

    @property
    def weights(self) -> dict[str, float]:
        return self._neighbors

    def close(self) -> None:
        pyre_node = self._pyre_node
        self._pyre_node = None

        if pyre_node is not None:
            try:
                pyre_node.stop()
            except Exception:
                logger.exception("[%s] Failed to stop Pyre node.", self._node_id)

        self._out_socket.close(linger=0)

        for dealer in self._dealers.values():
            dealer.close(linger=0)

        self._dealers.clear()

        logger.info("[%s] Node closed.", self._node_id)

    def _create_sockets(self) -> int:
        self._out_socket = self._context.socket(zmq.ROUTER)
        self._out_socket.setsockopt(zmq.LINGER, 0)

        data_port = self._out_socket.bind_to_random_port("tcp://*")

        identity = self._node_id.encode("utf-8")

        for peer_id in self._neighbors:
            dealer = self._context.socket(zmq.DEALER)
            dealer.setsockopt(zmq.LINGER, 0)
            dealer.setsockopt(zmq.IDENTITY, identity)

            self._dealers[peer_id] = dealer

        return data_port

    def _start_pyre(self, data_port: int) -> None:
        pyre_node = pyre.Pyre(self._node_id, ctx=self._context)

        self._pyre_node = pyre_node

        pyre_node.set_header("port", str(data_port))
        pyre_node.join(self._namespace)
        pyre_node.start()

    def _discover_neighbors(self) -> dict[str, str]:
        pyre_node = self._pyre_node

        if pyre_node is None:
            raise RuntimeError(f"[{self._node_id}] Pyre is not started")

        endpoints: dict[str, str] = {}
        pending: dict[bytes, tuple[str, int]] = {}

        while len(endpoints) < self.degree:
            msg = pyre_node.recv()

            event = msg[0].decode("utf-8")
            peer_uuid_bytes = msg[1]

            if event == "ENTER":
                peer_id = msg[2].decode("utf-8")
                headers: dict[str, str] = loads(msg[3].decode("utf-8"))

                port = headers.get("port")
                if port is None:
                    continue

                pending[peer_uuid_bytes] = (peer_id, int(port))
                continue

            if event != "JOIN":
                continue

            namespace = msg[3].decode("utf-8")
            if namespace != self._namespace:
                continue

            info = pending.pop(peer_uuid_bytes, None)
            if info is None:
                continue

            peer_id, port = info

            if peer_id not in self._neighbors:
                continue

            peer_uuid = uuid.UUID(bytes=peer_uuid_bytes)
            pyre_endpoint = pyre_node.peer_address(peer_uuid)

            host = urlsplit(pyre_endpoint).hostname
            if host is None:
                continue

            endpoint = f"tcp://{host}:{port}"
            endpoints[peer_id] = endpoint

        return endpoints

    def _connect_to_neighbors(self, endpoints: dict[str, str]) -> None:
        for peer_id, dealer in self._dealers.items():
            dealer.connect(endpoints[peer_id])

        # Initial handshake so each remote ROUTER learns our identity.
        for dealer in self._dealers.values():
            dealer.send(b"")

        connected: set[str] = set()

        while len(connected) < self.degree:
            client_id, _ = self._out_socket.recv_multipart()
            peer_id = client_id.decode("utf-8")

            if peer_id in self._neighbors:
                connected.add(peer_id)

        logger.info("[%s] Connected to all %d neighbors.", self._node_id, self.degree)

    def neighborwise_exchange(
        self, state_map: dict[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """
        Exchanges the given state map with all neighbor nodes.

        This method broadcasts the state map to all neighbors and then gathers their states.

        Args:
            state_map (dict[str, NDArray[np.float64]]): The state map to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state maps.
        """
        if self._out_socket is None:
            raise RuntimeError("FastNetwork is not bound")

        if state_map.keys() != self._neighbors.keys():
            missing = self._neighbors.keys() - state_map.keys()
            extra = state_map.keys() - self._neighbors.keys()
            raise ValueError(
                f"[{self._node_id}] State dictionary keys do not match neighbor names. "
                f"Missing: {missing}, Extra: {extra}."
            )

        for j in self._neighbors:
            state = state_map[j]
            meta, payload = self._transform.encode(state)
            payload = np.ascontiguousarray(payload)
            msgs = [j.encode(), meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: dict[str, NDArray[np.float64]] = {}
        for j, dealer in self._dealers.items():
            meta, payload = dealer.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        return neighbor_states

    def exchange(self, state: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """
        Exchanges the given state with all neighbor nodes.

        This method broadcasts the state to all neighbors and then gathers their states.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            dict[str, NDArray[np.float64]]: A dictionary mapping neighbor names to their received state arrays.
        """
        if self._out_socket is None:
            raise RuntimeError("FastNetwork is not bound")

        meta, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j_bytes in self._neighbor_id_bytes:
            msgs = [j_bytes, meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: dict[str, NDArray[np.float64]] = {}
        for j, dealer in self._dealers.items():
            meta, payload = dealer.recv_multipart()
            neighbor_states[j] = self._transform.decode(meta, payload)

        return neighbor_states

    def exchange_as_array(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Exchanges the given state with all neighbor nodes and returns their states as a stacked array.

        Note: Using this method will add an extra copy of the neighbor states in memory compared to the `exchange` method.
        This is because the states are first received as individual memory buffers and then stacked into a single array.

        Args:
            state (NDArray[np.float64]): The state array to exchange with neighbors.

        Returns:
            NDArray[np.float64]: A 2D array where each row corresponds to a neighbor's received state array.
        """
        if self._out_socket is None:
            raise RuntimeError("FastNetwork is not bound")

        meta, payload = self._transform.encode(state)
        payload = np.ascontiguousarray(payload)
        for j_bytes in self._neighbor_id_bytes:
            msgs = [j_bytes, meta, payload]
            self._out_socket.send_multipart(msgs, copy=False)

        neighbor_states: list[NDArray[np.float64]] = []
        for j in self._neighbors:
            meta, payload = self._dealers[j].recv_multipart()
            neighbor_states.append(self._transform.decode(meta, payload))

        return np.stack(neighbor_states, axis=0)

    def laplacian(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Laplacian of the given state vector based on the states of neighboring nodes.

        The Laplacian is calculated as:

            laplacian = state * number_of_neighbors - sum_of_neighbor_states

        Args:
            state (NDArray[float64]): The state vector of the current node.

        Returns:
            NDArray[float64]: The Laplacian vector representing the difference between the current state and the average state of its neighbors.
        """
        neighbor_states = self.exchange(state)
        laplacian = state * len(neighbor_states)
        for nbr_state in neighbor_states.values():
            laplacian -= nbr_state

        return laplacian

    def weighted_mix(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Performs the weighted mixing operation for distributed optimization using the weight matrix W.

        For a given node i, the mixed state is computed as the i-th row of Wx, where x is the stacked state vector of all nodes.
        If x_i is multi-dimensional, the operation is applied element-wise.
        Specifically:

            mixed_state = W_ii * state + sum_j(W_ij * neighbor_state_j)

        where W_ii is self._weight and W_ij are the weights in self._neighbor_weights.

        Args:
            state (NDArray[np.float64]): The current state vector of node i.

        Returns:
            NDArray[float64]: The mixed state vector corresponding to the i-th row of Wx.
        """
        neighbor_states = self.exchange(state)
        nbrs = self._neighbors
        mixed_state = state * self._weight
        for j, nbr_state in neighbor_states.items():
            weight = nbrs[j]
            mixed_state += weight * nbr_state

        return mixed_state
