import socket
import struct

from conops._core import CoreNode


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """从 TCP 字节流中准确读取指定长度的数据。"""
    buffer = bytearray()

    while len(buffer) < size:
        chunk = sock.recv(size - len(buffer))

        if not chunk:
            raise ConnectionError("Connection closed unexpectedly")

        buffer.extend(chunk)

    return bytes(buffer)


def main():
    # 1. Python 创建 Rust 对象
    node = CoreNode("A", 5000)

    try:
        # 2. 启动 Rust 后台线程
        node.start()
        print("[Python] Rust node started")

        # 3. Python 连接 Rust TCP Server
        with socket.create_connection(("127.0.0.1", 5000), timeout=5) as sock:
            message = b"Hello from Python!"

            # 4. 发送长度前缀 + 数据
            sock.sendall(struct.pack("!I", len(message)) + message)

            print("[Python] Sent:", message)

            # 5. 接收 Rust 返回的数据
            length = struct.unpack("!I", recv_exact(sock, 4))[0]

            reply = recv_exact(sock, length)

            print("[Python] Received:", reply)

            assert reply == message

    finally:
        # 6. 关闭 Rust 后台线程
        node.close()
        print("[Python] Rust node stopped")


if __name__ == "__main__":
    main()
