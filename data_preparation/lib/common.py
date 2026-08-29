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



def md5_hex(text: str) -> str:
    """MD5 hex digest of ``text`` (used for exact deduplication)."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def normalized_text(text: str) -> str:
    """Lower-cased text with runs of whitespace collapsed to one space and stripped (exact-dedup key)."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def normalized_hash(text: str) -> str:
    """MD5 of :func:`normalized_text` — the key of the normalized exact deduplication pass."""
    return md5_hex(normalized_text(text))


def estimate_tokens(text: str) -> int:
    """Cheap token count estimate (characters / 4)."""
    return len(text) // 4


def list_parquet_files(directory: Path) -> list[Path]:
    """Sorted ``*.parquet`` files inside ``directory`` (empty if the directory does not exist)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.parquet"))


def shard_index(path: Path) -> int | None:
    """Index ``n`` of a ``data-{n:05d}.parquet`` shard name, or None for other files."""
    match = SHARD_PATTERN.match(path.name)
    return int(match.group(1)) if match else None



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
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    if start_shard < 0:
        raise ValueError(f"start_shard must be >= 0, got {start_shard}")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        log.warning("removing leftover temp dir %s", tmp_dir)
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    shard_num = start_shard

    def flush(rows: int) -> None:
        nonlocal pending, pending_rows, shard_num
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
        pending_rows -= rows
        pq.write_table(pa.Table.from_batches(taken), tmp_dir / f"data-{shard_num:05d}.parquet")
        shard_num += 1

    try:
        for item in batches:
            for batch in item.to_batches() if isinstance(item, pa.Table) else [item]:
                if len(batch) == 0:
                    continue
                pending.append(batch)
                pending_rows += len(batch)
                while pending_rows >= shard_size:
                    flush(shard_size)
        if pending_rows > 0:
            flush(pending_rows)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list_parquet_files(out_dir):
        index = shard_index(stale)
        if index is not None and index >= start_shard:
            log.info("removing stale shard %s", stale)
            stale.unlink()
    for shard in sorted(tmp_dir.glob("data-*.parquet")):
        shard.replace(out_dir / shard.name)
    shutil.rmtree(tmp_dir)
    written = shard_num - start_shard
    log.info("wrote %d shard(s) to %s (starting at %d)", written, out_dir, start_shard)
    return written



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
