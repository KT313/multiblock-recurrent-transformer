# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One run per program at a time: an advisory ``flock`` on a lock file held for the duration of the run —
``<dataset_dir>/.build.lock`` for a build (``prepare.py prepare`` and ``train.py``'s auto-prepare, or two training runs
sharing ``dataset_dir``, would otherwise interleave directory removals, shard writes and manifest saves) and
``<out_dir>/.train.lock`` for a training run. The lock file records who holds it (program, pid, host, since) so the
error message can say so; the OS releases the lock when the holder dies, so a lock file is never stale and never has
to be removed by hand — a run that lost its terminal and continues headless holds it until it finishes. ``status`` /
``--dry_run`` do not take it."""

from __future__ import annotations

import fcntl
import json
import os
import socket
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from data_preparation.lib.log import get_logger

log = get_logger(__name__)

BUILD_LOCK_NAME = ".build.lock"
TRAIN_LOCK_NAME = ".train.lock"


@dataclass(frozen=True)
class Holder:
    """Who holds a lock, as recorded in the lock file."""

    program: str
    pid: int
    host: str
    since: str  # ISO 8601, UTC

    def started(self) -> str:
        """``since`` in the local time zone (``2026-09-03 07:45:22 JST``)."""
        return datetime.fromisoformat(self.since).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class RunLocked(RuntimeError):
    """Another process holds the lock: a run of the same program (or of the build a training run needs) is going."""

    def __init__(self, path: Path, program: str, holder: Holder | None) -> None:
        self.path = path
        self.holder = holder
        if holder is None:
            message = f"{program} expects one run at a time on this system; another run holds {path} (holder unknown)"
        else:
            message = (
                f"{holder.program} expects one run at a time on this system; one is already running (started "
                f"{holder.started()}, pid {holder.pid} on {holder.host}). Wait for it to finish, or stop it with: "
                f"kill -INT {holder.pid}"
            )
        super().__init__(message)


@contextmanager
def run_lock(path: Path, program: str) -> Iterator[None]:
    """Hold ``path`` exclusively for the block as ``program``; :class:`RunLocked` (naming the holder) if it is taken."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunLocked(path, program, _read_holder(path)) from None
        os.ftruncate(fd, 0)
        os.write(fd, _holder_record(program).encode())
        log.debug("holding %s", path)
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def build_lock(root: Path) -> AbstractContextManager[None]:
    """The build lock of the dataset directory ``root`` (``root/.build.lock``)."""
    return run_lock(root / BUILD_LOCK_NAME, "data preparation")


def _holder_record(program: str) -> str:
    return json.dumps(
        {"program": program, "pid": os.getpid(), "host": socket.gethostname(), "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    )


def _read_holder(path: Path) -> Holder | None:
    try:
        record = json.loads(path.read_text() or "{}")
        return Holder(str(record["program"]), int(record["pid"]), str(record["host"]), str(record["since"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
