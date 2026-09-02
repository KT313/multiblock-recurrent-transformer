# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Streaming parquet datasets. Rows are yielded as dicts; tokenization happens in the collate function."""

import logging
import random
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

Row = dict[str, Any]
T = TypeVar("T")

DEFAULT_DATA_SIGNATURE: dict[str, Any] = {"keys": ["text"], "format_fn": "pass_text"}
PARQUET_READ_BATCH_ROWS = 1024


class ParquetTextDataset(IterableDataset[Row]):
    """One ``hfds`` directory of parquet files, streamed in sorted file order without shuffling.

    Each row is a dict with the ``data_signature["keys"]`` columns plus ``data_signature`` and ``data_id``
    (the spec prefix, used for batch-composition logging). ``skip_rows`` / ``max_rows`` restrict the dataset to the
    row range ``[skip_rows, skip_rows + max_rows)`` of the directory (rows counted in sorted file order); the range
    is clipped to the rows that exist. One pass of ``__iter__`` is one epoch over the range. Rows of the range are
    dealt round-robin across ``world * num_workers`` shards: shard ``rank * num_workers + worker_id`` takes every
    ``num_shards``-th row of the range, so all shards together yield every row of the range exactly once.

    ``set_resume_offset`` starts every following epoch that many rows into the range — how a resume skips the rows
    the interrupted run already trained on. `training.data.loader.RunDataloaders` sets it right before the first
    epoch after a resume and back to 0 before every later one (a permanent offset would hide the rows before it
    forever).
    """

    def __init__(
        self,
        data_dir: str | Path,
        prefix: str,
        data_signature: dict[str, Any] | None = None,
        shard: tuple[int, int] = (0, 1),
        skip_rows: int = 0,
        max_rows: int | None = None,
    ) -> None:
        if skip_rows < 0 or (max_rows is not None and max_rows < 0):
            raise ValueError(f"{prefix}: skip_rows and max_rows must be non-negative, got {skip_rows}, {max_rows}")
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        self.data_signature = data_signature or DEFAULT_DATA_SIGNATURE
        self.rank, self.world = shard
        self.files = sorted(self.data_dir.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files in {self.data_dir}")
        columns = set(pq.ParquetFile(self.files[0]).schema_arrow.names)  # metadata only, no row is read
        missing = [k for k in self.data_signature["keys"] if k not in columns]
        if missing:
            raise ValueError(
                f"{prefix}: parquet files in {self.data_dir} lack the column(s) {missing} required by "
                f"data_signature {self.data_signature}; found {sorted(columns)}"
            )
        # per-file row counts from the parquet footers (metadata only), read once per instance
        self.file_rows = [pq.ParquetFile(f).metadata.num_rows for f in self.files]
        self.total_rows = sum(self.file_rows)
        self.start = min(skip_rows, self.total_rows)  # first row of the range (directory index)
        self.stop = self.total_rows if max_rows is None else min(self.total_rows, self.start + max_rows)
        self.num_rows = self.stop - self.start
        self.resume_offset = 0  # rows of the range the next epoch skips; see `set_resume_offset`

    def __len__(self) -> int:
        """Rows in the range over all shards (one epoch)."""
        return self.num_rows

    def set_resume_offset(self, rows: int) -> None:
        """Skip the first `rows` rows of the range in every following epoch (taken modulo the range, so more
        consumed rows than the range holds wrap around to where the last epoch stood)."""
        if rows < 0:
            raise ValueError(f"{self.prefix}: resume offset must be non-negative, got {rows}")
        self.resume_offset = rows % self.num_rows if self.num_rows else 0

    def _shard(self) -> tuple[int, int]:
        """(shard_id, num_shards) for the calling process/worker; the single place sharding is decided."""
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        return self.rank * num_workers + worker_id, self.world * num_workers

    def _range_batches(self, keys: list[str]) -> Iterator[tuple[int, pa.RecordBatch]]:
        """``(range_idx, batch)`` pairs covering exactly rows ``[start, stop)``; ``range_idx`` is the position of
        the batch's first row within the range. Files and row groups outside the range are never opened/read."""
        file_start = 0
        for file, n_rows in zip(self.files, self.file_rows):
            file_stop = file_start + n_rows
            if file_stop <= self.start or n_rows == 0:  # entirely before the range (or empty): skip via the footer
                file_start = file_stop
                continue
            if file_start >= self.stop:
                return
            parquet = pq.ParquetFile(file)
            row_groups: list[int] = []
            group_start = file_start
            first_group_start = file_start
            for group in range(parquet.num_row_groups):
                group_stop = group_start + parquet.metadata.row_group(group).num_rows
                if group_stop > self.start and group_start < self.stop:
                    if not row_groups:
                        first_group_start = group_start
                    row_groups.append(group)
                group_start = group_stop
            global_idx = first_group_start
            for batch in parquet.iter_batches(batch_size=PARQUET_READ_BATCH_ROWS, columns=keys, row_groups=row_groups):
                batch_stop = global_idx + batch.num_rows
                lo, hi = max(global_idx, self.start), min(batch_stop, self.stop)
                if lo < hi:
                    yield lo - self.start, batch.slice(lo - global_idx, hi - lo)
                global_idx = batch_stop
                if global_idx >= self.stop:
                    return
            file_start = file_stop

    def __iter__(self) -> Iterator[Row]:
        shard_id, num_shards = self._shard()
        offset = self.resume_offset
        keys: list[str] = list(self.data_signature["keys"])
        logger.info(
            f"{self.prefix}: shard {shard_id}/{num_shards} over rows [{self.start + offset}, {self.stop}) "
            f"({self.num_rows - offset} of {self.num_rows} rows) in {self.data_dir}"
        )
        for range_idx, batch in self._range_batches(keys):
            if range_idx + batch.num_rows <= offset:  # entirely before the resume offset: never decoded
                continue
            record: Row
            for record in batch.to_pylist():
                if range_idx >= offset and range_idx % num_shards == shard_id:
                    record["data_signature"] = self.data_signature
                    record["data_id"] = self.prefix
                    yield record
                range_idx += 1


class WeightedMixtureDataset(IterableDataset[T], Generic[T]):
    """Draws each row from one of several datasets with fixed probabilities; exhausted datasets restart."""

    def __init__(self, datasets: Sequence[Iterable[T]], weights: Sequence[float], seed: int) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("Need one weight per dataset.")
        total = float(sum(weights))
        self.datasets = datasets
        self.weights = [w / total for w in weights]
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        rng = random.Random(self.seed)
        iterators = [iter(ds) for ds in self.datasets]
        indices = range(len(self.datasets))
        while True:
            (idx,) = rng.choices(indices, weights=self.weights, k=1)
            try:
                yield next(iterators[idx])
            except StopIteration:
                logger.info(f"Dataset '{getattr(self.datasets[idx], 'prefix', idx)}' exhausted, restarting.")
                iterators[idx] = iter(self.datasets[idx])
                yield next(iterators[idx])
