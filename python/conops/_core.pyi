class CoreNode:
    """Rust-backed TCP network node."""

    def __init__(self, node_id: str, port: int) -> None:
        """Create a network node.

        Args:
            node_id: Logical identifier of the node.
            port: Local TCP listening port.
        """
        ...

    def start(self) -> None:
        """Start the background Rust networking runtime.

        Blocks until the TCP listener is initialized.
        Raises RuntimeError if startup fails.
        """
        ...

    def close(self) -> None:
        """Stop the background runtime and join its worker thread."""
        ...
