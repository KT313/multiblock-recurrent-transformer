# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
One run per program at a time: an advisory flock on a lock file held for the whole run.

<dataset_dir>/.build.lock guards preparation and the entire training reader lifetime; <out_dir>/.train.lock guards a training run.
The file records who holds it (program, pid, host, since) for the error message. The OS releases ownership once every owning descriptor closes (a forked child can retain one), so an unused
lock file is never stale; a run that lost its terminal and continues headless holds it until it
finishes. status and --dry_run do not take it.
"""

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
    """
    Who holds a lock, as recorded in the lock file.
    """

    program: str
    pid: int
    host: str
    since: str  # ISO 8601, UTC

    def started(self) -> str:
        """
        since in the local time zone (2026-09-03 07:45:22 JST).
        """

        return datetime.fromisoformat(self.since).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class RunLocked(RuntimeError):
    """
    Another process holds the lock: a run of the same program, or of the build a training run needs, is going.
    """

    def __init__(self, path: Path, program: str, holder: Holder | None) -> None:
        self.path = path
        self.holder = holder
        if holder is None:
            message = f"{program} expects one run at a time on this system; another run holds {path} (holder unknown)"
        else:
            message = (
                f"{holder.program} expects one run at a time on this system; one is already running (started "
                f"{holder.started()}, pid {holder.pid} on {holder.host}; protected root {path.parent}). "
                "Wait for it to finish, or stop it with: "
                f"kill -INT {holder.pid}"
            )
        super().__init__(message)


@contextmanager
def run_lock(path: Path, program: str) -> Iterator[None]:
    """
    Hold path exclusively for the block as program; :class:`RunLocked` (naming the holder) if it is taken.
    """

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


class DatasetLease:
    """Opaque, process-local ownership, issued only by :func:`dataset_lock`. Never send to workers."""

    __slots__ = ()

    def __init__(self) -> None:
        raise TypeError("DatasetLease is issued by dataset_lock; it cannot be constructed directly")


# Identity membership prevents an ordinary constructed/copied token bypassing the lock. PID rejects fork copies.
_ACTIVE_LEASES: dict[DatasetLease, tuple[Path, int]] = {}


def validate_dataset_lease(lease: DatasetLease, root: Path) -> None:
    """Reject expired, foreign-root, forged or fork-inherited ownership before any dataset access."""

    expected = (root.resolve(), os.getpid())
    if type(lease) is not DatasetLease or _ACTIVE_LEASES.get(lease) != expected:
        raise ValueError(f"invalid dataset lease for {expected[0]}: ownership must be active in this process")


@contextmanager
def dataset_lock(
    root: Path, program: str = "data preparation", *, lease: DatasetLease | None = None
) -> Iterator[DatasetLease]:
    """Hold canonical root/.build.lock exclusively, or borrow a validated lease without releasing it.

    The inode is never replaced or removed. Cooperating training and all preparation selections use this same
    lock; status/dry-run remain best-effort read-only observations. Advisory locking cannot restrain external
    writers. A fork inherits the descriptor, but cannot borrow the parent's lease.
    """

    root = root.resolve()
    if lease is not None:
        validate_dataset_lease(lease, root)
        yield lease
        return
    with run_lock(root / BUILD_LOCK_NAME, program):
        owned = object.__new__(DatasetLease)
        _ACTIVE_LEASES[owned] = (root, os.getpid())
        try:
            yield owned
        finally:
            del _ACTIVE_LEASES[owned]


def build_lock(root: Path) -> AbstractContextManager[DatasetLease]:
    """Compatibility entry point for the common exclusive dataset-operation lock."""

    return dataset_lock(root)


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
