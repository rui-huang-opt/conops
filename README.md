# ConOps (Consensus Operations)

**ConOps** is a Python package providing graph abstractions and synchronous neighbor-to-neighbor communication primitives for distributed consensus and optimization algorithms.

It offers two communication backends:

- **`Network`** — a Rust-backed TCP implementation with neighbor discovery, exchange matching, and basic connection recovery.
- **`LiteNetwork`** — a lightweight ZeroMQ implementation for synchronous experiments that do not require application-level message recovery.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/rui-huang-opt/conops.git
```

Or install in editable mode for development:

```bash
git clone https://github.com/rui-huang-opt/conops.git
cd conops
pip install -e .
```

**Note:** These commands build the Rust extension from source, so a Rust toolchain is required. For an optimized local build during development, use `maturin develop --release` in an activated virtual environment.

## Undirected Graphs

An undirected graph is defined by

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}),
$$

where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of unordered edges. An edge $(u,v)$ represents a communication link between nodes $u$ and $v$.

### Ring topology

For a five-node ring,

$$
\mathcal{V}=\{1,2,3,4,5\},\qquad
\mathcal{E}=\{(1,2),(2,3),(3,4),(4,5),(5,1)\}.
$$

Create the topology using `Graph`:

```python
from conops import Graph

nodes = ["1", "2", "3", "4", "5"]
edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(nodes, edges)

# Each node only needs its own neighbors and edge weights.
print(graph["1"])
```

You can visualize the topology with NetworkX and Matplotlib:

```python
import matplotlib.pyplot as plt
import networkx as nx

ring = nx.Graph()
ring.add_nodes_from(nodes)
ring.add_edges_from(edges)

nx.draw(ring, with_labels=True)
plt.show()
```

![Ring topology](docs/images/ring_topology.png)

## Deploying a Network

Each node needs only its own `node_id` and a dictionary mapping neighbor IDs to edge weights. Nodes run independently, without a central communication server.

The Rust-backed `Network` supports two connection modes:

- **Automatic discovery (default):** mDNS discovers configured neighbors on a local network.
- **Manual configuration:** `peer_endpoints` supplies TCP addresses directly, which is useful on servers and in Kubernetes environments where multicast discovery may not be available.

For example, a node may be initialized as follows:

```python
from conops import Network

network = Network(
    node_id="1",
    neighbors={"2": 0.45, "5": 0.45},
    group_id="consensus-demo",
)

network.start()
network.wait_connected(timeout=30)
```

Once connected, the node can exchange NumPy arrays and apply operators such as `laplacian()` and `weighted_mix()`. **Each exchange needs an explicitly specified, monotonically increasing `exchange_id`**, and neighboring nodes must use the same ID for the same logical exchange.

When configured manually, `peer_endpoints` maps the local node and its neighbors to endpoints, for example:

```python
peer_endpoints = {
    "1": "tcp://node-1:5001",
    "2": "tcp://node-2:5002",
    "5": "tcp://node-5:5005",
}

network = Network(
    node_id="1",
    neighbors={"2": 0.45, "5": 0.45},
    peer_endpoints=peer_endpoints,
)
```

Here, `node-1`, `node-2`, and `node-5` are example hostnames; they must resolve to reachable machines or containers.

## Laplacian Consensus Example

For a connected undirected graph, a standard Laplacian consensus iteration is

$$
x(k+1) = x(k) - \alpha Lx(k),
$$

where $L$ is the graph Laplacian and $\alpha>0$ is a suitable step size. Each node only communicates with its neighbors. Under the usual step-size conditions, the node states converge to the average of their initial values.

Run the following code in a separate process or machine for **each** node, changing `node_id` and its corresponding `neighbors` dictionary:

```python
import numpy as np

from conops import Network

node_id = "1"
neighbors = {"2": 0.45, "5": 0.45}

network = Network(node_id=node_id, neighbors=neighbors)

rng = np.random.default_rng(int(node_id))
x = rng.uniform(-100.0, 100.0, 3)
alpha = 0.45

print(f"Node {node_id} initial state: {x}")

try:
    network.start()
    network.wait_connected(timeout=30)

    for k in range(50):
        x = x - alpha * network.laplacian(
            exchange_id=k,
            state=x,
        )
finally:
    network.close()

print(f"Node {node_id} final state: {x}")
```

For a complete single-machine example that launches all nodes as separate processes, see the `examples` directory.

![Laplacian consensus](docs/images/consensus.png)

## Lightweight Communication

`LiteNetwork` offers the same basic numerical operators through a simpler ZeroMQ communication path, without Rust `Network`'s exchange-ID matching and replay features. Its communication calls do not require an `exchange_id`:

```python
from conops import LiteNetwork

network = LiteNetwork(node_id=node_id, neighbors=neighbors)
try:
    x = x - alpha * network.laplacian(x)
finally:
    network.close()
```

Choose `Network` for explicit exchange matching and basic recovery, or `LiteNetwork` for simple, tightly synchronized experiments.
