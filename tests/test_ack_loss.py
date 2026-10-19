"""ACK-loss/reconnect test for ConOps, using only the Python standard library.

Run after ``maturin develop``:
    python tests/test_rust_ack_loss.py

A connects to B through a TCP frame-aware proxy.  The proxy drops the first
ACK(20) sent from B to A, then breaks that TCP session.  After reconnect, A
must replay DATA(20); B must deduplicate it and send ACK(20) again.  Exchange
20 and the next exchange must both complete.
"""

import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from conops._core import CoreNetwork

A_PORT = 54100
B_PORT = 54101
PROXY_PORT = 54102


def recv_exact(sock: socket.socket, count: int, stop: threading.Event) -> bytes:
    data = bytearray()
    while len(data) < count:
        if stop.is_set():
            raise ConnectionAbortedError("Proxy shutting down")
        try:
            chunk = sock.recv(count - len(data))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError("TCP peer closed")
        data.extend(chunk)
    return bytes(data)


class DropFirstAckProxy:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.dropped_ack = threading.Event()
        self.replayed_data = threading.Event()
        self.sessions = 0
        self._session_lock = threading.Lock()
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
            except OSError:
                client.close()
                continue

            with self._session_lock:
                self.sessions += 1
                session_id = self.sessions

            with client, server:
                client.settimeout(0.2)
                server.settimeout(0.2)
                end = threading.Event()

                def forward(
                    src: socket.socket, dst: socket.socket, server_to_client: bool
                ) -> None:
                    try:
                        while not self.stop.is_set() and not end.is_set():
                            header = recv_exact(src, 4, end)
                            length = struct.unpack("!I", header)[0]
                            assert length <= 64 * 1024 * 1024
                            frame = recv_exact(src, length, end)

                            # Only inspect data frames, never HELLO strings.
                            if not frame.startswith(b"HELLO ") and len(frame) >= 9:
                                kind = frame[0]
                                exchange_id = int.from_bytes(frame[1:9], "big")
                                if (
                                    session_id == 1
                                    and server_to_client
                                    and kind == 1
                                    and exchange_id == 20
                                    and not self.dropped_ack.is_set()
                                ):
                                    print(
                                        "[Proxy] Drop B->A ACK(20), close both sockets",
                                        flush=True,
                                    )
                                    self.dropped_ack.set()
                                    end.set()
                                    return
                                if (
                                    session_id >= 2
                                    and not server_to_client
                                    and kind == 0
                                    and exchange_id == 20
                                ):
                                    print(
                                        "[Proxy] Observed replayed A->B DATA(20)",
                                        flush=True,
                                    )
                                    self.replayed_data.set()

                            dst.sendall(header + frame)
                    except (EOFError, ConnectionError, OSError):
                        pass
                    finally:
                        end.set()
                        for sock in (client, server):
                            try:
                                sock.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass

                directions = [
                    threading.Thread(
                        target=forward, args=(client, server, False), daemon=True
                    ),
                    threading.Thread(
                        target=forward, args=(server, client, True), daemon=True
                    ),
                ]
                for t in directions:
                    t.start()
                for t in directions:
                    t.join()

    def close(self) -> None:
        self.stop.set()
        self.listener.close()
        self.worker.join(timeout=3)


def wait_for(predicate, timeout: float, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(message)


def main() -> None:
    proxy = DropFirstAckProxy()
    a = CoreNetwork("A", A_PORT)
    b = CoreNetwork("B", B_PORT)
    try:
        b.start()
        a.start()
        proxy.start()
        a.connect("B", f"127.0.0.1:{PROXY_PORT}")
        wait_for(
            lambda: a.peers() == ["B"] and b.peers() == ["A"],
            5,
            "Initial connection was not registered",
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            waiting_a = pool.submit(a.exchange, 20, b"A20", timeout=20.0)
            assert proxy.dropped_ack.wait(5), "ACK(20) was not sent by B"
            wait_for(
                lambda: proxy.sessions >= 2
                and a.peers() == ["B"]
                and b.peers() == ["A"],
                15,
                "Automatic reconnect failed",
            )
            assert proxy.replayed_data.wait(5), "A did not replay DATA(20)"

            # A sent DATA 20 before B entered the exchange. B kept it in
            # EarlyMessages, even though the TCP connection was interrupted.
            assert b.exchange(20, b"B20", timeout=10.0) == {"A": b"A20"}
            assert waiting_a.result(timeout=12) == {"B": b"B20"}
            print(
                "PASS: lost ACK, duplicate DATA, reconnect and Exchange 20", flush=True
            )

            next_a = pool.submit(a.exchange, 21, b"A21", timeout=10.0)
            next_b = pool.submit(b.exchange, 21, b"B21", timeout=10.0)
            assert next_a.result(timeout=12) == {"B": b"B21"}
            assert next_b.result(timeout=12) == {"A": b"A21"}
            print("PASS: Exchange 21", flush=True)
    finally:
        a.close()
        b.close()
        proxy.close()


if __name__ == "__main__":
    main()
