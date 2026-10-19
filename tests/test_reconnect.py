"""Test in-flight Exchange recovery without killing either node.

Run after `maturin develop`:
    python tests/test_rust_reconnect.py

A -> proxy -> B. The proxy drops A's first Exchange 20 DATA message and
closes both sockets. A must reconnect and replay the lost DATA frame.
"""

import select
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from conops._core import CoreNetwork

A_PORT = 54000
B_PORT = 54001
PROXY_PORT = 54002


class DropOnceProxy:
    """Small TCP forwarding proxy that can drop one client->server packet."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.arm = threading.Event()
        self.dropped = threading.Event()
        self.sessions = 0
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", PROXY_PORT))
        self.listener.listen()
        self.listener.settimeout(0.2)
        self.worker = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.worker.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                server = socket.create_connection(("127.0.0.1", B_PORT), timeout=3)
                server.settimeout(None)
            except OSError:
                client.close()
                continue

            self.sessions += 1
            with client, server:
                sockets = (client, server)
                while not self.stop.is_set():
                    try:
                        readable, _, _ = select.select(sockets, (), (), 0.1)
                    except (OSError, ValueError):
                        break
                    broken = False
                    for src in readable:
                        try:
                            chunk = src.recv(65536)
                        except OSError:
                            broken = True
                            break
                        if not chunk:
                            broken = True
                            break
                        if src is client and self.arm.is_set():
                            # Do NOT forward any of A's DATA; cut both TCP sockets.
                            self.arm.clear()
                            self.dropped.set()
                            print(
                                "[Proxy] Dropped A's DATA and cut the TCP connection",
                                flush=True,
                            )
                            broken = True
                            break
                        dst = server if src is client else client
                        try:
                            dst.sendall(chunk)
                        except OSError:
                            broken = True
                            break
                    if broken:
                        break

    def close(self) -> None:
        self.stop.set()
        self.listener.close()
        self.worker.join(timeout=3)


def wait_until(predicate, seconds: float, message: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(message)


def main() -> None:
    proxy = DropOnceProxy()
    a = CoreNetwork("A", A_PORT)
    b = CoreNetwork("B", B_PORT)

    try:
        b.start()
        a.start()
        proxy.start()
        a.connect("B", f"127.0.0.1:{PROXY_PORT}")
        wait_until(
            lambda: a.peers() == ["B"] and b.peers() == ["A"],
            5,
            "Initial connection failed",
        )
        print("[Test] Initial connection ready", flush=True)

        proxy.arm.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            waiting_a = pool.submit(a.exchange, 20, b"A20", timeout=20.0)
            assert proxy.dropped.wait(5), "Proxy did not intercept DATA(20)"
            wait_until(
                lambda: proxy.sessions >= 2
                and a.peers() == ["B"]
                and b.peers() == ["A"],
                15,
                "TCP did not reconnect",
            )
            print("[Test] TCP reconnected; B enters Exchange 20", flush=True)

            # A's original DATA(20) never reached B on the first TCP session.
            # B can receive it only if Rust replayed DATA(20) after reconnect.
            result_b = b.exchange(20, b"B20", timeout=10.0)
            result_a = waiting_a.result(timeout=10)
            assert result_b == {"A": b"A20"}, result_b
            assert result_a == {"B": b"B20"}, result_a
            print(
                "PASS: Exchange 20 recovered after lost DATA and TCP reconnect",
                flush=True,
            )

            next_a = pool.submit(a.exchange, 21, b"A21", timeout=10.0)
            next_b = pool.submit(b.exchange, 21, b"B21", timeout=10.0)
            assert next_a.result(timeout=12) == {"B": b"B21"}
            assert next_b.result(timeout=12) == {"A": b"A21"}
            print("PASS: Exchange 21 also completed", flush=True)
    finally:
        a.close()
        b.close()
        proxy.close()


if __name__ == "__main__":
    main()
