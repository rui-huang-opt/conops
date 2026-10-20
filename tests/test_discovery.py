"""Run in three terminals: python tests/test_rust_auto_discovery.py A (or B/C).

Uses mDNS for auto-discovery, no manual connect() or hard-coded remote IPs.
Run after `maturin develop` in the same trusted LAN.
"""

import argparse
import time

from conops._core import CoreNetwork

PORTS = {"A": 5000, "B": 5001, "C": 5002}
NEIGHBORS = {
    "A": ["B"],
    "B": ["A", "C"],
    "C": ["B"],
}
GROUP_ID = "conops-autodiscovery-test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node_id", choices=PORTS)
    parser.add_argument("--group", default=GROUP_ID)
    args = parser.parse_args()

    node_id = args.node_id
    neighbors = NEIGHBORS[node_id]
    network = CoreNetwork(
        node_id,
        PORTS[node_id],
        neighbors=neighbors,
        group_id=args.group,
    )

    try:
        network.start()
        print(f"[{node_id}] Waiting for mDNS neighbors {neighbors}...", flush=True)
        network.wait_connected(timeout=30.0)
        assert set(network.peers()) == set(neighbors)
        print(f"[{node_id}] Connected: {network.peers()}", flush=True)

        if node_id == "B":
            # Other nodes can enter exchange 10 before B does.
            time.sleep(2)

        for exchange_id in (10, 11):
            payload = f"{node_id}:{exchange_id}".encode()
            received = network.exchange(exchange_id, payload, timeout=20.0)
            expected = {peer: f"{peer}:{exchange_id}".encode() for peer in neighbors}
            assert (
                received == expected
            ), f"{node_id} exchange {exchange_id}: {received!r} != {expected!r}"
            print(f"[{node_id}] exchange {exchange_id}: {received}", flush=True)

        print(f"[{node_id}] PASS: discovered chain Exchange", flush=True)
        input("Press Enter to close this node (after ALL nodes PASS)...")
    finally:
        network.close()


if __name__ == "__main__":
    main()
