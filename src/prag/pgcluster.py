"""Manage a private Postgres cluster: initdb, start, stop.

The cluster listens on 127.0.0.1 only and has no unix socket. Local connections
use trust authentication, which is acceptable for a single-user benchmark box
and not for anything shared; see README "Limits".
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

from . import config

SERVER_SETTINGS = {
    "listen_addresses": "'127.0.0.1'",
    "unix_socket_directories": "''",
    "shared_buffers": "'1GB'",
    "work_mem": "'32MB'",
    "maintenance_work_mem": "'1GB'",
    "max_connections": "100",
    "effective_cache_size": "'4GB'",
    "random_page_cost": "1.1",
    "jit": "off",
    "max_parallel_maintenance_workers": "3",
}


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, capture_output=True, text=True, **kw)


def initdb(data_dir: Path, settings: dict[str, str] | None = None) -> None:
    if (data_dir / "PG_VERSION").exists():
        return
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(config.PG_BIN / "initdb"),
            "-D", str(data_dir),
            "-U", "postgres",
            "--auth=trust",
            "--encoding=UTF8",
            "--locale=C",
        ]
    )
    conf = data_dir / "postgresql.conf"
    lines = [f"{k} = {v}" for k, v in (settings or SERVER_SETTINGS).items()]
    with conf.open("a") as fh:
        fh.write("\n# permissioned-rag settings\n" + "\n".join(lines) + "\n")


def is_running(data_dir: Path) -> bool:
    res = subprocess.run(
        [str(config.PG_BIN / "pg_ctl"), "status", "-D", str(data_dir)],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def start(data_dir: Path, port: int, log_file: Path) -> None:
    if is_running(data_dir):
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(config.PG_BIN / "pg_ctl"),
            "start",
            "-D", str(data_dir),
            "-l", str(log_file),
            "-o", f"-p {port}",
            "-w",
            "-t", "60",
        ]
    )
    wait_for_port("127.0.0.1", port)


def stop(data_dir: Path) -> None:
    if not is_running(data_dir):
        return
    _run([str(config.PG_BIN / "pg_ctl"), "stop", "-D", str(data_dir), "-m", "fast", "-w"])


def wait_for_port(host: str, port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"Postgres did not open {host}:{port}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
