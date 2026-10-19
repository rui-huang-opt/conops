"""Run after `maturin develop`: python examples/test_rust_messages.py"""

import time
from conops._core import CoreNetwork


def wait_for_peer(node: CoreNetwork, peer: str, seconds: float = 3.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if peer in node.peers():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Peer {peer} was not registered in time")


def main() -> None:
    a = CoreNetwork("A", 5000)
    b = CoreNetwork("B", 5001)

    try:
        a.start()
        b.start()

        a.connect("B", "127.0.0.1:5001")

        wait_for_peer(a, "B")
        wait_for_peer(b, "A")

        a.send("B", b"Hello from A")
        b.send("A", b"Hello from B")

        assert b.recv(timeout=5) == ("A", b"Hello from A")
        assert a.recv(timeout=5) == ("B", b"Hello from B")

        print("PASS: two-way Rust TCP messages via Python")

    finally:
        b.close()
        a.close()


if __name__ == "__main__":
    main()
