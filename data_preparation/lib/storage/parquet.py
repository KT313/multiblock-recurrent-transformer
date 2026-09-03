# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Parquet shard files: the data-NNNNN.parquet naming, listing, atomic publishing and the incremental writer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.lib.log import get_logger
from data_preparation.lib.storage.atomic import write_atomically

log = get_logger(__name__)
SHARD_PATTERN = re.compile(r"^data-(\d{5,})\.parquet$")
SHARD_COMPRESSION: Literal["zstd"] = "zstd"  # every shard written from now on; older snappy shards stay readable


def list_parquet_files(directory: Path) -> list[Path]:
    """
    Sorted *.parquet files inside directory (empty if the directory does not exist).
    """

    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.parquet"))


def shard_index(path: Path) -> int | None:
    """
    Index n of a data-{n:05d}.parquet shard name, or None for other files.
    """

    match = SHARD_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def shard_name(index: int) -> str:
    """
    File name of shard index: data-{index:05d}.parquet.
    """

    return f"data-{index:05d}.parquet"


class ShardWriter:
    """
    Writes dict rows as out_dir/data-NNNNN.parquet shards of shard_size rows. add(row) buffers rows and
    publishes every full shard as soon as it is written (atomically, then on_shard(path) so the caller records it
    in a manifest). Numbering starts at start_shard (append mode); stale shards >= start_shard in out_dir
    are removed on enter. An exception leaves the published shards in place and discards only the buffered partial
    shard: raw downloads are append-only, and losing a whole increment to a network error would throw away hours of
    transfer. Several writers (one per output directory) can be fed from one input stream.
    """

    def __init__(self, out_dir: Path, shard_size: int, *, start_shard: int = 0, on_shard: Callable[[Path], None]) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if start_shard < 0:
            raise ValueError(f"start_shard must be >= 0, got {start_shard}")
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.start_shard = start_shard
        self.shards_written = 0
        self._on_shard = on_shard
        self._buffer: list[dict[str, Any]] = []

    def __enter__(self) -> ShardWriter:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._remove_stale_shards()
        for leftover in self.out_dir.glob("*.parquet.tmp"):
            leftover.unlink()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self._buffer = []
            return
        if self._buffer:
            self.write_shard(pa.Table.from_pylist(self._buffer))
            self._buffer = []

    def add(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self.write_shard(pa.Table.from_pylist(self._buffer))
            self._buffer = []

    def write_shard(self, table: pa.Table) -> None:
        """
        Publish table as the next shard (the caller sizes it), then the on_shard callback.
        """

        path = publish_shard(table, self.out_dir / shard_name(self.start_shard + self.shards_written))
        self.shards_written += 1
        self._on_shard(path)

    def _remove_stale_shards(self) -> None:
        for existing in list_parquet_files(self.out_dir):
            index = shard_index(existing)
            if index is not None and index >= self.start_shard:
                log.info("removing stale shard %s", existing)
                existing.unlink()


def publish_shard(table: pa.Table, path: Path) -> Path:
    """
    Write table to path atomically (:func:`write_atomically`) and return path.
    """

    with write_atomically(path) as tmp:
        pq.write_table(table, tmp, compression=SHARD_COMPRESSION)
    return path
