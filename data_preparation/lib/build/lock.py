# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Advisory reader/writer locks held for the whole operation.

<dataset_dir>/.build.lock permits multiple read-only training runs or one exclusive preparer/auto-prepare run.
<out_dir>/<run_name>/.train.lock exclusively guards a training run and records its holder for diagnostics.
Dataset locks do not record holders. The OS releases ownership once every owning descriptor closes (a forked child can retain one), so an unused
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
            message = f"{program} cannot acquire {path}: another operation holds a conflicting lock (holder unknown)"
        else:
            message = (
                f"{holder.program} expects one run at a time on this system; one is already running (started "
                f"{holder.started()}, pid {holder.pid} on {holder.host}; protected root {path.parent}). "
                "Wait for it to finish, or stop it with: "
                f"kill -INT {holder.pid}"
            )
        super().__init__(message)


@contextmanager
def run_lock(
    path: Path, program: str, *, shared: bool = False, record_holder: bool = True,
) -> Iterator[None]:
    """
    Hold path without waiting; shared readers coexist, exclusive holders exclude everyone else.
    Only exclusive holders may write metadata; dataset callers disable metadata entirely.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    write_holder = record_holder and not shared
    try:
        try:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder(path) if record_holder else None
            raise RunLocked(path, program, holder) from None
        if write_holder:
            os.ftruncate(fd, 0)
            os.write(fd, _holder_record(program).encode())
        log.debug("holding %s (%s)", path, "shared" if shared else "exclusive")
        try:
            yield
        finally:
            if write_holder:
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
_ACTIVE_LEASES: dict[DatasetLease, tuple[Path, int, bool]] = {}  # root, PID, shared/read-only


def validate_dataset_lease(lease: DatasetLease, root: Path, *, shared: bool = False) -> None:
    """Reject expired, foreign-root, forged or fork-inherited ownership before any dataset access."""

    expected = (root.resolve(), os.getpid())
    record = _ACTIVE_LEASES.get(lease)
    if type(lease) is not DatasetLease or record is None or record[:2] != expected:
        raise ValueError(f"invalid dataset lease for {expected[0]}: ownership must be active in this process")
    if record[2] and not shared:
        raise ValueError("a shared dataset lease cannot authorize preparation")


@contextmanager
def dataset_lock(
    root: Path, program: str = "data preparation", *, lease: DatasetLease | None = None, shared: bool = False,
) -> Iterator[DatasetLease]:
    """Hold canonical root/.build.lock (exclusive by default), or borrow a sufficient active lease.

    The inode is never replaced or removed. Cooperating training and all preparation selections use this same
    lock; status/dry-run remain best-effort read-only observations. Advisory locking cannot restrain external
    writers. A fork inherits the descriptor, but cannot borrow the parent's lease.
    """

    root = root.resolve()
    if lease is not None:
        validate_dataset_lease(lease, root, shared=shared)
        yield lease
        return
    with run_lock(root / BUILD_LOCK_NAME, program, shared=shared, record_holder=False):
        owned = object.__new__(DatasetLease)
        _ACTIVE_LEASES[owned] = (root, os.getpid(), shared)
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
