# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Helpers shared by the data preparation stages: HF cache setup, hashing, token estimate, parquet shard I/O."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.lib.log import get_logger

log = get_logger(__name__)
SHARD_PATTERN = re.compile(r"^data-(\d{5,})\.parquet$")
_WHITESPACE = re.compile(r"\s+")


def configure_hf_cache(cache_dir: Path | None) -> None:
    """Point every HuggingFace cache at ``cache_dir``; must run before ``datasets``/``transformers`` are imported."""
    if cache_dir is None:
        return
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        os.environ[var] = str(cache_dir)
    log.info("using HuggingFace cache %s", cache_dir)


# --- text hashing / token estimate ----------------------------------------------------------------------------------


def md5_hex(text: str) -> str:
    """MD5 hex digest of ``text`` (used for exact deduplication)."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def normalized_text(text: str) -> str:
    """Lower-cased text with runs of whitespace collapsed to one space and stripped (exact-dedup key)."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def normalized_hash(text: str) -> str:
    """MD5 of :func:`normalized_text` — the key of the normalized exact deduplication pass."""
    return md5_hex(normalized_text(text))


def text_hash64(text: str, normalize: bool = True) -> int:
    """The exact-dedup key of ``text`` as a signed 64-bit integer (the first 64 bits of :func:`normalized_hash` with
    ``normalize``, else of :func:`md5_hex`); stored as the int64 ``hash`` column of processed shards."""
    digest = normalized_hash(text) if normalize else md5_hex(text)
    return int.from_bytes(bytes.fromhex(digest[:16]), "big", signed=True)


def estimate_tokens(text: str) -> int:
    """Cheap token count estimate (characters / 4)."""
    return len(text) // 4


# --- shard files -----------------------------------------------------------------------------------------------------


def list_parquet_files(directory: Path) -> list[Path]:
    """Sorted ``*.parquet`` files inside ``directory`` (empty if the directory does not exist)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.parquet"))


def shard_index(path: Path) -> int | None:
    """Index ``n`` of a ``data-{n:05d}.parquet`` shard name, or None for other files."""
    match = SHARD_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def shard_name(index: int) -> str:
    """File name of shard ``index``: ``data-{index:05d}.parquet``."""
    return f"data-{index:05d}.parquet"


def write_parquet_shards(
    batches: Iterable[pa.RecordBatch | pa.Table],
    out_dir: Path,
    shard_size: int,
    *,
    start_shard: int = 0,
) -> int:
    """Write a stream of record batches as ``out_dir/data-NNNNN.parquet`` shards of ``shard_size`` rows each.

    Atomic with respect to ``out_dir``: shards are written into the sibling ``<out_dir>.tmp/`` (recreated fresh) and
    moved into ``out_dir`` only after the whole stream has been consumed; if the iterator raises, ``out_dir`` is
    unchanged and the temp dir is removed. Numbering starts at ``start_shard`` (append mode); pre-existing shards in
    ``out_dir`` with index >= ``start_shard`` are deleted (stale-shard cleanup), lower ones are kept. Batches are
    re-chunked so every shard except the last has exactly ``shard_size`` rows. Returns the number of shards written.
    """
    with ShardWriter(out_dir, shard_size, start_shard=start_shard) as writer:
        for shard_table in _rechunk(batches, shard_size):
            writer.write_shard(shard_table)
    return writer.written


class ShardWriter:
    """Incremental version of :func:`write_parquet_shards` (same atomicity, numbering and stale-shard cleanup):
    ``add(row)`` buffers dict rows and writes a shard every ``shard_size`` rows, the last partial shard and the
    move into ``out_dir`` happen when the ``with`` block exits normally; an exception leaves ``out_dir`` unchanged.
    Several writers (one per output directory) can be fed from one input stream."""

    def __init__(self, out_dir: Path, shard_size: int, *, start_shard: int = 0) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if start_shard < 0:
            raise ValueError(f"start_shard must be >= 0, got {start_shard}")
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.start_shard = start_shard
        self.written = 0
        self._buffer: list[dict[str, Any]] = []
        self._tmp_dir = out_dir.with_name(out_dir.name + ".tmp")

    def __enter__(self) -> ShardWriter:
        if self._tmp_dir.exists():
            log.warning("removing leftover temp dir %s", self._tmp_dir)
            shutil.rmtree(self._tmp_dir)
        self._tmp_dir.mkdir(parents=True)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            return
        if self._buffer:
            self.write_shard(pa.Table.from_pylist(self._buffer))
            self._buffer = []
        self._publish()

    def add(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self.write_shard(pa.Table.from_pylist(self._buffer))
            self._buffer = []

    def write_shard(self, table: pa.Table) -> None:
        """Write ``table`` as the next shard into the temp dir (the caller sizes it)."""
        pq.write_table(table, self._tmp_dir / shard_name(self.start_shard + self.written))
        self.written += 1

    def _publish(self) -> None:
        """Drop the shards the new ones replace, then move the new ones into place."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for existing in list_parquet_files(self.out_dir):
            index = shard_index(existing)
            if index is not None and index >= self.start_shard:
                log.info("removing stale shard %s", existing)
                existing.unlink()
        for shard in sorted(self._tmp_dir.glob("data-*.parquet")):
            shard.replace(self.out_dir / shard.name)
        shutil.rmtree(self._tmp_dir)
        log.info("wrote %d shard(s) to %s (starting at %d)", self.written, self.out_dir, self.start_shard)


def _rechunk(batches: Iterable[pa.RecordBatch | pa.Table], shard_size: int) -> Iterator[pa.Table]:
    """Regroup a stream of batches/tables into tables of exactly ``shard_size`` rows (the last one may be shorter).

    Empty batches are skipped. Rows keep their order; a batch that straddles a shard boundary is sliced.
    """
    pending: list[pa.RecordBatch] = []  # batches not yet emitted, in order
    pending_rows = 0
    for item in batches:
        for batch in item.to_batches() if isinstance(item, pa.Table) else [item]:
            if len(batch) == 0:
                continue
            pending.append(batch)
            pending_rows += len(batch)
            while pending_rows >= shard_size:
                yield pa.Table.from_batches(_take_rows(pending, shard_size))
                pending_rows -= shard_size
    if pending_rows > 0:
        yield pa.Table.from_batches(_take_rows(pending, pending_rows))


def _take_rows(pending: list[pa.RecordBatch], rows: int) -> list[pa.RecordBatch]:
    """Remove the first ``rows`` rows from the front of ``pending`` (slicing the last batch if needed)."""
    taken: list[pa.RecordBatch] = []
    remaining = rows
    while remaining > 0:
        batch = pending.pop(0)
        if len(batch) <= remaining:
            taken.append(batch)
            remaining -= len(batch)
        else:
            taken.append(batch.slice(0, remaining))
            pending.insert(0, batch.slice(remaining))
            remaining = 0
    return taken


def write_dict_rows(rows: Iterable[dict[str, Any]], out_dir: Path, shard_size: int, *, start_shard: int = 0) -> int:
    """Write an iterable of dict rows to parquet shards (see :func:`write_parquet_shards`)."""

    def batches() -> Iterator[pa.RecordBatch]:
        buffer: list[dict[str, Any]] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= shard_size:
                yield pa.RecordBatch.from_pylist(buffer)
                buffer = []
        if buffer:
            yield pa.RecordBatch.from_pylist(buffer)

    return write_parquet_shards(batches(), out_dir, shard_size, start_shard=start_shard)
