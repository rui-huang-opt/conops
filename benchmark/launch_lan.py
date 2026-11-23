"""
Launch a six-node ConOps LAN benchmark across this PC and one SSH host.

Place this file in conops/benchmark/ and rename it to launch_lan.py.

    python launch_lan.py --remote rui@192.168.1.111 --dry-run
    python launch_lan.py --remote rui@192.168.1.111

Requires: pip install asyncssh (on the local machine only).
Defaults: nodes 1,2,3 local; nodes 4,5,6 on the remote computer.
"""

import argparse
import asyncio
import os
import shlex
import signal
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import asyncssh

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARK = BENCHMARK_DIR / "lan_benchmark.py"


def benchmark_args(node: str, args: argparse.Namespace) -> list[str]:
    return [
        "--node_id",
        node,
        "--n_state",
        str(args.n_state),
        "--n_iter",
        str(args.n_iter),
        "--barrier-host",
        args.barrier_host,
        "--barrier-port",
        str(args.barrier_port),
    ]


def remote_root(project: str) -> str:
    # Expand the remote HOME on the server, keeping the remaining path quoted.
    return '"$HOME"/' + shlex.quote(project)


def remote_command(node: str, args: argparse.Namespace) -> str:
    root = remote_root(args.remote_project)
    return (
        f"cd {root}/benchmark && "
        f"exec {root}/.venv/bin/python -u lan_benchmark.py "
        + shlex.join(benchmark_args(node, args))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        required=True,
        help="SSH target, e.g. rui@192.168.1.111",
    )
    parser.add_argument("--barrier-host", required=True)
    parser.add_argument("--barrier-port", type=int, default=47000)
    parser.add_argument("--local-nodes", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--remote-nodes", nargs="+", default=["4", "5", "6"])
    parser.add_argument(
        "--remote-project",
        default="code/conops",
        help="Remote project directory relative to remote $HOME",
    )
    parser.add_argument("--n-state", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Max runtime in seconds",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ids = args.local_nodes + args.remote_nodes
    if not ids or len(ids) != len(set(ids)):
        parser.error("Node IDs must be nonempty and unique")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (
        not args.remote_project
        or args.remote_project.startswith("/")
        or ".." in Path(args.remote_project).parts
    ):
        parser.error("--remote-project must be relative to remote $HOME")
    if not BENCHMARK.is_file():
        parser.error(f"Cannot find {BENCHMARK}; place this script in benchmark/")
    return args


async def stop_local(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def stop_remote(process: asyncssh.SSHClientProcess) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        try:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=2)
        except (OSError, asyncio.TimeoutError):
            process.close()


async def supervise(
    local: dict[str, asyncio.subprocess.Process],
    remote: dict[str, asyncssh.SSHClientProcess],
    timeout: float,
    log_dir: Path,
) -> int:
    async def remote_exit(process: asyncssh.SSHClientProcess) -> int:
        result = await process.wait()
        return result.returncode if result.returncode is not None else -1

    waiting = {
        asyncio.create_task(process.wait()): node for node, process in local.items()
    }
    waiting.update(
        {
            asyncio.create_task(remote_exit(process)): node
            for node, process in remote.items()
        }
    )
    deadline = asyncio.get_running_loop().time() + timeout

    try:
        while waiting:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait(
                waiting,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                print(f"Timed out after {timeout:g}s; see {log_dir}", file=sys.stderr)
                return 1

            for task in done:
                node = waiting.pop(task)
                try:
                    code = task.result()
                except Exception as exc:
                    print(f"Node {node} failed: {exc}; see {log_dir}", file=sys.stderr)
                    return 1
                if code != 0:
                    print(
                        f"Node {node} exited with {code}; see {log_dir}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"Node {node} completed", flush=True)

        print(f"\nAll {len(local) + len(remote)} nodes completed. Logs: {log_dir}")
        return 0
    finally:
        for task in waiting:
            task.cancel()
        if waiting:
            await asyncio.gather(*waiting, return_exceptions=True)


async def run(args: argparse.Namespace) -> int:
    username, separator, hostname = args.remote.rpartition("@")
    if not separator:
        hostname, username = args.remote, None

    for node in args.local_nodes:
        cmd = [sys.executable, "-u", str(BENCHMARK), *benchmark_args(node, args)]
        print(f"Node {node} (local): {shlex.join(cmd)}")
    for node in args.remote_nodes:
        print(f"Node {node} ({args.remote}, AsyncSSH): {remote_command(node, args)}")
    if args.dry_run:
        return 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BENCHMARK_DIR / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLogs: {log_dir}\n", flush=True)

    local: dict[str, asyncio.subprocess.Process] = {}
    remote: dict[str, asyncssh.SSHClientProcess] = {}

    try:
        # Authenticate and check the remote environment BEFORE starting local nodes.
        async with asyncssh.connect(
            hostname,
            username=username,
            connect_timeout=10,
        ) as conn:
            root = remote_root(args.remote_project)
            check_cmd = (
                f"cd {root}/benchmark && "
                f"{root}/.venv/bin/python -c {shlex.quote('import conops')}"
            )
            await conn.run(check_cmd, check=True, timeout=20)
            print("Remote Python environment OK", flush=True)

            with ExitStack() as stack:
                try:
                    for node in args.local_nodes:
                        log = stack.enter_context(
                            (log_dir / f"node_{node}.log").open("w", encoding="utf-8")
                        )
                        cmd = [
                            sys.executable,
                            "-u",
                            str(BENCHMARK),
                            *benchmark_args(node, args),
                        ]
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            cwd=BENCHMARK_DIR,
                            stdout=log,
                            stderr=asyncio.subprocess.STDOUT,
                            start_new_session=True,
                        )
                        local[node] = process
                        print(
                            f"Started node {node} locally (PID {process.pid})",
                            flush=True,
                        )

                    for node in args.remote_nodes:
                        log = stack.enter_context(
                            (log_dir / f"node_{node}.log").open("w", encoding="utf-8")
                        )
                        process = await conn.create_process(
                            remote_command(node, args),
                            stdin=asyncssh.DEVNULL,
                            stdout=log,
                            stderr=asyncssh.STDOUT,
                        )
                        remote[node] = process
                        print(f"Started node {node} via AsyncSSH", flush=True)

                    return await supervise(local, remote, args.timeout, log_dir)
                finally:
                    # Stop processes while the SSH connection is still active.
                    await asyncio.gather(
                        *(stop_local(p) for p in local.values()),
                        *(stop_remote(p) for p in remote.values()),
                        return_exceptions=True,
                    )
    except (OSError, asyncssh.Error, asyncio.TimeoutError) as exc:
        print(f"Launch failed: {exc}; see {log_dir}", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
