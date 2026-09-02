# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One training run per run directory: an advisory ``flock`` on ``<out_dir>/.train.lock`` held for the duration of
``train()``. Two runs sharing an ``out_dir`` would otherwise write the same checkpoints, the same ``train.log`` and
the same ``run_config.json``, and could resume off each other's checkpoints. The lock file records who holds it
(pid, host, since) so the error message can say so; the OS releases the lock when the holder dies, so a leftover
lock file is never stale and never has to be removed by hand — a lock left behind by a killed run is simply taken
again. Sequential runs (a fresh run, then a resume) are unaffected: the lock is released on every way out of the
context manager, normal return and exception alike.

Mirrors ``data_preparation/lib/build/lock.py`` (one build per dataset directory); the two locks are independent —
a training run with ``auto_prepare`` holds both, this one on ``out_dir`` and the build one on ``dataset_dir``.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RUN_LOCK_NAME = ".train.lock"


class RunDirectoryLocked(RuntimeError):
    """Another training run is already using the same run directory."""


@contextmanager
def run_directory_lock(run_directory: Path) -> Iterator[None]:
    """Hold ``run_directory/.train.lock`` exclusively for the block; :class:`RunDirectoryLocked` (naming the holder)
    if another process — or another lock of this one — already has it."""
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / RUN_LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunDirectoryLocked(
                f"another training run is already using {run_directory} ({_holder(path)}); wait for it to finish, "
                "or give this run its own out_dir"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, _holder_record().encode())
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _holder_record() -> str:
    return json.dumps(
        {"pid": os.getpid(), "host": socket.gethostname(), "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    )


def _holder(path: Path) -> str:
    try:
        record = json.loads(path.read_text() or "{}")
        return f"pid {record.get('pid')} on {record.get('host')} since {record.get('since')}"
    except (OSError, ValueError):
        return "holder unknown"
