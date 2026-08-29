# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Helpers shared by the data preparation CLIs: argument defaults, HF cache setup, parquet shard I/O."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # `datasets` is imported lazily at runtime (after the HF cache is configured)
    from datasets import Dataset

RANDOM_SEED = 42
DEFAULT_DATASET_DIR = Path("dataset")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--dataset_dir`` and ``--cache_dir`` to a CLI parser."""
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Root directory for all prepared data (default: dataset/ relative to the current directory)",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (default: the HF default, usually ~/.cache/huggingface)",
    )


def configure_hf_cache(cache_dir: Path | None) -> None:
    """Point every HuggingFace cache at ``cache_dir``; must run before ``datasets``/``transformers`` are imported."""
    if cache_dir is None:
        return
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        os.environ[var] = str(cache_dir)
    print(f"Using HuggingFace cache: {cache_dir}")


def print_header(title: str, width: int = 80) -> None:
    """Print a titled separator line."""
    print("=" * width)
    print(title)
    print("=" * width)


def md5_hex(text: str) -> str:
    """MD5 hex digest of ``text`` (used for exact deduplication)."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap token count estimate (characters / 4)."""
    return len(text) // 4


def list_parquet_files(directory: Path, prefix: str) -> list[Path]:
    """Sorted ``<prefix>-*.parquet`` files inside ``directory`` (empty if the directory does not exist)."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{prefix}-*.parquet"))


def select_dataset_dirs(root: Path, patterns: list[str] | None) -> list[Path]:
    """Sub-directories of ``root`` matching any of ``patterns`` (glob), or all sub-directories when ``patterns`` is None."""
    if not root.is_dir():
        return []
    if not patterns:
        return sorted(d for d in root.iterdir() if d.is_dir())
    selected: list[Path] = []
    for pattern in patterns:
        selected.extend(d for d in sorted(root.glob(pattern)) if d.is_dir() and d not in selected)
    return selected


def write_parquet_shards(
    batches: Iterable[pa.RecordBatch | pa.Table],
    out_dir: Path,
    shard_size: int,
    prefix: str = "data",
) -> int:
    """Write a stream of record batches to ``out_dir/<prefix>-NNNNN.parquet`` files of ``shard_size`` rows each.

    Returns the number of shards written. Batches are re-chunked so every shard except the last has exactly
    ``shard_size`` rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    shard_num = 0

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
        pq.write_table(pa.Table.from_batches(taken), out_dir / f"{prefix}-{shard_num:05d}.parquet")
        shard_num += 1

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
    return shard_num


def iter_dataset_tables(dataset: "Dataset", batch_size: int = 10_000) -> Iterator[pa.Table]:
    """Yield a ``datasets.Dataset`` as arrow tables of ``batch_size`` rows, honouring select/filter index mappings."""
    arrow_view = dataset.with_format("arrow")
    for start in range(0, len(dataset), batch_size):
        # with_format("arrow") makes slicing return a Table; `datasets` declares dict | list, hence the cast
        yield cast(pa.Table, arrow_view[start : start + batch_size])


def write_dict_rows(rows: Iterable[dict[str, Any]], out_dir: Path, shard_size: int, prefix: str = "data") -> int:
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

    return write_parquet_shards(batches(), out_dir, shard_size, prefix)
