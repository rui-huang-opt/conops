"""Cross-process disconnect test for conops._core.CoreNetwork.

Run from the project root after `maturin develop`:

    python tests/test_rust_disconnect.py

This script starts node A in the parent Python process and node B in a
separate child process. B deliberately never calls exchange(). After A starts
exchange 20 and is blocked waiting for B, the parent forcibly kills B. A must
fail promptly with a peer-disconnected error, not wait for the exchange timeout.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

from conops._core import CoreNetwork

HOST = "127.0.0.1"
PORT_A = 5000
PORT_B = 5001
EXCHANGE_ID = 20


def run_peer_b() -> None:
    """Child process: accept A's connection, but do not enter exchange 20."""
    node = CoreNetwork("B", PORT_B)
    try:
        node.start()
        print("[B] Ready; deliberately NOT calling exchange()", flush=True)
        while True:
            time.sleep(1)
    finally:
        node.close()


def connect_with_retry(
    node: CoreNetwork, child: subprocess.Popen, seconds: float = 10
) -> None:
    """Handle the brief interval before child B binds its listening port."""
    deadline = time.monotonic() + seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Node B exited unexpectedly: code={child.returncode}")
        try:
            node.connect("B", f"{HOST}:{PORT_B}")
            return
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Could not connect to B: {last_error}")


def run_test() -> None:
    a = CoreNetwork("A", PORT_A)
    child: subprocess.Popen | None = None
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        a.start()
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--peer-b"],
            stdin=subprocess.DEVNULL,
        )
        connect_with_retry(a, child)
        assert a.peers() == ["B"], f"Unexpected A peers: {a.peers()}"
        print("[A] Connected to B", flush=True)

        # The call must run on a separate Python thread: exchange() blocks.
        future = executor.submit(a.exchange, EXCHANGE_ID, b"A20", timeout=15.0)
        time.sleep(0.5)
        if future.done():
            try:
                early_result = future.result()
            except Exception as exc:
                raise AssertionError("Exchange failed before B was killed") from exc
            raise AssertionError(f"Exchange returned too early: {early_result!r}")
        if child.poll() is not None:
            raise AssertionError("Node B exited before the deliberate kill")

        print("[A] Exchange 20 is waiting; forcibly killing B...", flush=True)
        start = time.monotonic()
        child.kill()  # SIGKILL on Unix: no graceful node.close() in B
        child.wait(timeout=5)

        try:
            received = future.result(timeout=5.0)
        except RuntimeError as exc:
            elapsed = time.monotonic() - start
            if "Peer B disconnected" not in str(exc):
                raise AssertionError(
                    f"Expected peer-disconnect error, got: {exc}"
                ) from exc
            print(f"[A] Expected disconnect error ({elapsed:.3f}s): {exc}", flush=True)
            print("PASS: exchange failed promptly after B was killed", flush=True)
        except FutureTimeout as exc:
            raise AssertionError(
                "Exchange did not report the disconnect within 5 seconds"
            ) from exc
        else:
            raise AssertionError(f"Exchange unexpectedly succeeded: {received!r}")
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        a.close()
        executor.shutdown(wait=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-b", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.peer_b:
        run_peer_b()
    else:
        run_test()


if __name__ == "__main__":
    main()
