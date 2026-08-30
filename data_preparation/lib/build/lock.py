# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One build per dataset directory: an advisory ``flock`` on ``<dataset_dir>/.build.lock`` held for the duration of
``build``. ``prepare.py build`` and ``train.py``'s auto-prepare (or two training runs sharing ``dataset_dir``) would
otherwise interleave directory removals, shard writes and manifest saves. The lock file records who holds it so the
error message can say so; ``status`` / ``--dry_run`` do not take it."""

from __future__ import annotations

import fcntl
import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from data_preparation.lib.log import get_logger

log = get_logger(__name__)

BUILD_LOCK_NAME = ".build.lock"


class BuildLocked(RuntimeError):
    """Another process is building in the same dataset directory."""


@contextmanager
def build_lock(root: Path) -> Iterator[None]:
    """Hold ``root/.build.lock`` exclusively for the block; :class:`BuildLocked` (naming the holder) if it is taken."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / BUILD_LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BuildLocked(f"another build holds {path} ({_holder(path)}); wait for it to finish or remove a stale lock file") from None
        os.ftruncate(fd, 0)
        os.write(fd, _holder_record().encode())
        log.debug("holding %s", path)
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _holder_record() -> str:
    return json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "since": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _holder(path: Path) -> str:
    try:
        record = json.loads(path.read_text() or "{}")
        return f"pid {record.get('pid')} on {record.get('host')} since {record.get('since')}"
    except (OSError, ValueError):
        return "holder unknown"


__all__ = ["BUILD_LOCK_NAME", "BuildLocked", "build_lock"]
