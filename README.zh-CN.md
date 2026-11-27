# ConOps (Consensus Operations)

[English](README.md) | [简体中文](README.zh-CN.md)

**ConOps** 是一个 Python 软件包，为分布式一致性与优化算法提供图结构抽象，以及同步的邻居间通信原语。

它提供两种通信实现：

- **`Network`** — 基于 Rust 和 TCP 的实现，支持邻居发现、自动生成交换 ID、基于确认消息（ACK）的消息跟踪，以及连接恢复。

- **`LiteNetwork`** — 原有的 Python/ZeroMQ 实现，为了兼容旧代码并复现早期研究结果而保留。它采用更简单的通信机制，不提供显式的消息匹配或恢复机制。

两种实现都提供相近的接口，可用于交换 NumPy 数组和执行常见的一致性操作。

## 安装

### 预编译 Wheels

预编译的 wheel 文件可以从 [GitHub Releases](https://github.com/rui-huang-opt/conops/releases) 获取。

下载与你的 Python 版本和平台兼容的 wheel 文件后，运行：

```bash
pip install ./conops-*.whl
```

安装预编译 wheel 不需要 Rust 工具链。目前提供 Linux x86-64 平台上 Python 3.10、3.11 和 3.12 对应的 wheel。

### 从源代码安装

也可以直接从 GitHub 安装：

```bash
pip install git+https://github.com/rui-huang-opt/conops.git
```

或者克隆仓库，以可编辑模式安装，方便开发：

```bash
git clone https://github.com/rui-huang-opt/conops.git
cd conops
pip install -e .
```

**注意：** 从源代码安装时需要编译 Rust 扩展，但 `pip` 在某些环境下可以自动安装所需的 Rust 工具链，因此不一定需要提前手动安装 Rust。对于本地开发，也可以在安装了 Maturin 和 Rust 工具链的环境中使用 `maturin develop --release` 构建优化版本。

## 无向图

无向图定义为

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E}),
$$

其中，$\mathcal{V}$ 是节点集合，$\mathcal{E}$ 是无序边的集合。一条边 $(u,v)$ 表示节点 $u$ 与节点 $v$ 之间存在通信连接。

### 环形拓扑

以一个包含五个节点的环形图为例：

$$
\mathcal{V}=\{1,2,3,4,5\},\qquad
\mathcal{E}=\{(1,2),(2,3),(3,4),(4,5),(5,1)\}.
$$

使用 `Graph` 创建拓扑：

```python
from conops import Graph

nodes = ["1", "2", "3", "4", "5"]
edges = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "1")]

graph = Graph(nodes, edges)

# Each node only needs its own neighbors and edge weights.
print(graph["1"])
```

![环形拓扑](docs/images/ring_topology.png)

## 部署网络

每个节点只需要知道自己的 `node_id`，以及一个将邻居 ID 映射到边权重的字典。节点独立运行，不需要中心通信服务器。

基于 Rust 的 `Network` 支持两种连接模式：

- **自动发现（默认）：** 使用 UDP Scouting，通过组播 SCOUT 请求和单播 HELLO 响应发现已配置的邻居。

- **手动配置：** 使用 `peer_endpoints` 直接提供 TCP 地址，适用于组播发现可能不可用的服务器或 Kubernetes 等环境。

例如，可以按下面的方式初始化节点：

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

### 指定组播网卡

如果机器有多个网络接口，可以通过 `multicast_interface` 指定用于 UDP 组播发现的网卡在本机上的 IPv4 地址：

```python
network = Network(
    node_id="1",
    neighbors={"2": 0.45, "5": 0.45},
    multicast_interface="10.42.0.1",
)
```

这里填写的是**当前节点**对应网卡的 IPv4 地址，而不是 `wlan1` 这样的接口名称。如果不指定，则由操作系统选择组播接口。该参数影响 UDP 组播发现，不会改变 TCP 监听接口；提供 `peer_endpoints` 时，它不会生效。

连接建立后，节点可以交换 NumPy 数组，并使用 `laplacian()`、`weighted_mix()` 等算子。

基于 Rust 的 `Network` 会自动分配内部递增的交换 ID，无需也不支持显式传入 `key`：

```python
network.exchange(x)
network.laplacian(x)
network.laplacian(y)
```

每次成功执行 `exchange()`、`neighborwise_exchange()`、`laplacian()` 或 `weighted_mix()` 都会推进内部交换 ID。所有参与通信的邻居必须按相同的逻辑顺序执行相匹配的通信操作。如果某次交换失败，当前交换会话会被标记为无效；开始新会话前应关闭并重新启动网络。

在手动配置模式下，`peer_endpoints` 需要包含当前节点及其所有邻居的端点，例如：

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

这里的 `node-1`、`node-2` 和 `node-5` 只是示例主机名；它们必须能解析到可访问的机器或容器。

### 数据复制行为

基于 Rust 的 `Network` 在 `exchange()`、`neighborwise_exchange()`、`laplacian()` 和 `weighted_mix()` 中提供可选的 `copy` 参数。

默认的 `copy=True` 会为待发送数据创建一份快照，使通信后端能够在异步发送和可能的重传过程中保留数据，而不受原数组后续修改的影响。

设置 `copy=False` 可以避免这次额外的载荷复制，在交换较大的 NumPy 数组时可能减少内存开销。

```python
network.exchange(x, copy=False)
network.laplacian(x, copy=False)
```

当 `copy=False` 时，通信后端会在发送或重传仍可能需要数据时保留底层载荷的引用。因此，这段时间内载荷必须保持不变，甚至在交换函数返回之后也可能仍需要满足这一要求。

这通常与迭代优化算法中常见的非原地更新兼容：

```python
x = x - alpha * grad
```

上述更新会创建新数组，不会直接修改之前发送的数据。但这并不是普遍的安全保证：原地操作、通过共享视图修改数据，或者自定义 Transform 重用缓冲区，都可能改变仍在传输的数据。

```python
# Potentially unsafe with copy=False
x -= alpha * grad
```

用户需要自行确保，在通信后端仍可能使用载荷时，其底层数据不会被修改。

无法保证这一点时，建议使用默认的 `copy=True`。`copy=False` 虽然可能减少数据复制，但不一定提升整体通信性能。

### 优雅关闭

基于 Rust 的 `Network` 支持优雅关闭，以降低不同节点在不同时间结束时丢失待处理消息的风险。

默认情况下，`close()` 会尝试等待尚未确认的外发 DATA 消息获得确认，以及本地排队的 ACK 消息完成写入，然后再停止通信后端。

```python
network.close(timeout=5.0)
```

`timeout` 限制等待排空（drain）的时间。如果未能在规定时间内完成，可能会抛出异常，但通信后端仍然会关闭。

如果需要立即停止而不等待待处理消息，可以使用：

```python
network.close(timeout=0)
```

优雅关闭适用于正常终止场景；对于进程崩溃、永久性连接故障或其他不可恢复的通信错误，它不保证消息一定送达。

## Laplacian 一致性示例

对于连通的无向图，一个标准的 Laplacian 一致性迭代为

$$
x(k+1) = x(k) - \alpha Lx(k),
$$

其中 $L$ 为图 Laplacian 矩阵，$\alpha>0$ 为合适的步长。每个节点只与邻居通信。在通常的步长条件下，节点状态收敛到初始状态的平均值。

对于**每个**节点，在独立进程或机器上运行以下代码，并按节点修改 `node_id` 及对应的 `neighbors` 字典：

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
        x = x - alpha * network.laplacian(x)

finally:
    network.close()

print(f"Node {node_id} final state: {x}")
```

如需在单台机器上启动多个独立进程运行全部节点，可以参考 `examples` 目录中的完整示例。

![Laplacian 一致性](docs/images/consensus.png)

## 轻量级通信

`LiteNetwork` 是 ConOps 原有的 Python/ZeroMQ 网络实现，之前也叫 `Network`。引入基于 Rust 的实现后，原实现更名为 `LiteNetwork`。

它使用 Pyre 进行初始邻居发现，并使用 ZeroMQ ROUTER/DEALER 套接字直接通信。也可以显式提供 `peer_endpoints`，跳过 Pyre 发现。

与基于 Rust 的 `Network` 不同，`LiteNetwork` 不维护交换 ID、ACK 缓存或消息重放机制。

`LiteNetwork` 保留了较简单的同步通信设计，适用于所有节点按相同通信顺序运行、且无需额外恢复机制的实验。

```python
from conops import LiteNetwork

network = LiteNetwork(node_id=node_id, neighbors=neighbors)

try:
    for k in range(50):
        x = x - alpha * network.laplacian(x)

finally:
    network.close()
```

### 旧版行为与限制

`LiteNetwork` 保留原有的 ZeroMQ 实现，以兼容旧代码并支持复现此前发表的研究结果。

外发数据使用 `copy=False`，通常适用于迭代优化中的非原地更新。不过，ZeroMQ 仍有可能在交换函数返回后访问底层缓冲区，因此原地修改或缓冲区重用仍可能影响传输中的数据。

与基于 Rust 的 `Network` 不同，`LiteNetwork` 不提供应用层 ACK 或消息重放，其关闭流程也不能保证消息一定送达。

对于需要更可靠通信机制的新应用，推荐使用基于 Rust 的 `Network`；`LiteNetwork` 则继续用于轻量级实验和早期研究结果复现。
