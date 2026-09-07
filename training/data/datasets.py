# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Streaming parquet datasets. Rows are yielded as dicts; tokenization happens in the collate function.
"""

import logging
import random
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from data_preparation.lib.storage.parquet import SHARD_PATTERN

logger = logging.getLogger(__name__)

Row = dict[str, Any]
T = TypeVar("T")

DEFAULT_DATA_SIGNATURE: dict[str, Any] = {"keys": ["text"], "format_fn": "pass_text"}
PARQUET_READ_BATCH_ROWS = 1024


class ParquetTextDataset(IterableDataset[Row]):
    """
    One directory of build shards (data-NNNNN.parquet), streamed in sorted file order without shuffling.

    Each row is a dict with the data_signature["keys"] columns plus data_signature and data_id.
    skip_rows / max_rows restrict the dataset to the row range [skip_rows, skip_rows + max_rows), clipped
    to the rows that exist. One __iter__ is one epoch over the range, dealt round-robin over
    world * num_workers shards. set_resume_offset starts the next epoch that many rows into the range;
    `RunDataloaders` sets it before the first epoch after a resume and back to 0 before every later one.
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
        self.rank, self.world_size = shard
        # the build's shards only (`SHARD_PATTERN`, data-NNNNN.parquet), the same set `dataset_resolver` counts the
        # rows of: a stray parquet file would shift every row index behind the validation split
        self.files = sorted(path for path in self.data_dir.glob("*.parquet") if SHARD_PATTERN.match(path.name))
        if not self.files:
            raise FileNotFoundError(f"No data-NNNNN.parquet shard in {self.data_dir}")
        columns = set(pq.ParquetFile(self.files[0]).schema_arrow.names)  # metadata only, no row is read
        missing = [key for key in self.data_signature["keys"] if key not in columns]
        if missing:
            raise ValueError(
                f"{prefix}: parquet files in {self.data_dir} lack the column(s) {missing} required by "
                f"data_signature {self.data_signature}; found {sorted(columns)}"
            )
        # per-file row counts from the parquet footers (metadata only), read once per instance
        self.file_rows = [pq.ParquetFile(file).metadata.num_rows for file in self.files]
        self.total_rows = sum(self.file_rows)
        self.start = min(skip_rows, self.total_rows)  # first row of the range (directory index)
        self.stop = self.total_rows if max_rows is None else min(self.total_rows, self.start + max_rows)
        self.num_rows = self.stop - self.start
        self.resume_offset = 0  # rows of the range the next epoch skips; see `set_resume_offset`

    def __len__(self) -> int:
        """
        Rows in the range over all shards (one epoch).
        """

        return self.num_rows

    def set_resume_offset(self, rows: int) -> None:
        """
        Skip the first `rows` rows of the range in every following epoch (taken modulo the range, so more
        consumed rows than the range holds wrap around to where the last epoch stood).
        """

        if rows < 0:
            raise ValueError(f"{self.prefix}: resume offset must be non-negative, got {rows}")
        self.resume_offset = rows % self.num_rows if self.num_rows else 0

    def _shard(self) -> tuple[int, int]:
        """
        (shard_id, num_shards) for the calling process/worker; the single place sharding is decided.
        """

        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        return self.rank * num_workers + worker_id, self.world_size * num_workers

    def _range_batches(self, keys: list[str]) -> Iterator[tuple[int, pa.RecordBatch]]:
        """
        (range_idx, batch) pairs covering exactly rows [start, stop); range_idx is the position of
        the batch's first row within the range. Files and row groups outside the range are never opened/read.
        """

        file_start = 0
        for file, rows_in_file in zip(self.files, self.file_rows):
            file_stop = file_start + rows_in_file
            if file_stop <= self.start or rows_in_file == 0:  # before the range (or empty): skipped via the footer
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
            batch_start = first_group_start  # directory index of the batch's first row
            for batch in parquet.iter_batches(batch_size=PARQUET_READ_BATCH_ROWS, columns=keys, row_groups=row_groups):
                batch_stop = batch_start + batch.num_rows
                clip_start, clip_stop = max(batch_start, self.start), min(batch_stop, self.stop)
                if clip_start < clip_stop:
                    yield clip_start - self.start, batch.slice(clip_start - batch_start, clip_stop - clip_start)
                batch_start = batch_stop
                if batch_start >= self.stop:
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
    """
    Draws each row from one of several datasets with fixed probabilities (a seeded draw per row) until every
    member is read once; an exhausted member leaves the draw and the others renormalise. The validation loaders of
    a stage with several validation sources read this.
    """

    def __init__(self, datasets: Sequence[Iterable[T]], weights: Sequence[float], seed: int) -> None:
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("Need one weight per dataset.")
        total = float(sum(weights))
        self.datasets = datasets
        self.weights = [weight / total for weight in weights]
        self.seed = seed

    def __iter__(self) -> Iterator[T]:
        rng = random.Random(self.seed)
        iterators = [iter(dataset) for dataset in self.datasets]
        remaining = list(range(len(self.datasets)))  # members with rows left, in construction order
        while remaining:
            (member,) = rng.choices(remaining, weights=[self.weights[index] for index in remaining], k=1)
            try:
                yield next(iterators[member])
            except StopIteration:
                remaining.remove(member)
