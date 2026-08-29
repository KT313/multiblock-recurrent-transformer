# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Streaming parquet datasets. Rows are yielded as dicts; tokenization happens in the collate function."""

import logging
import random
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

Row = dict[str, Any]
T = TypeVar("T")

DEFAULT_DATA_SIGNATURE: dict[str, Any] = {"keys": ["text"], "format_fn": "pass_text"}
PARQUET_READ_BATCH_ROWS = 1024


class ParquetTextDataset(IterableDataset[Row]):
    """One ``hfds`` directory of parquet files, streamed in file order without shuffling.

    Each row is a dict with the ``data_signature["keys"]`` columns plus ``data_signature`` and ``data_id``
    (the spec prefix, used for batch-composition logging). Rows are dealt round-robin across
    ``world * num_workers`` shards: shard ``rank * num_workers + worker_id`` takes every ``num_shards``-th row.
    """

    def __init__(
        self,
        data_dir: str | Path,
        prefix: str,
        data_signature: dict[str, Any] | None = None,
        shard: tuple[int, int] = (0, 1),
    ) -> None:
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
        self.num_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in self.files)

    def __len__(self) -> int:
        return self.num_rows

    def _shard(self) -> tuple[int, int]:
        """(shard_id, num_shards) for the calling process/worker; the single place sharding is decided."""
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        return self.rank * num_workers + worker_id, self.world * num_workers

    def __iter__(self) -> Iterator[Row]:
        shard_id, num_shards = self._shard()
        keys: list[str] = list(self.data_signature["keys"])
        logger.info(f"{self.prefix}: shard {shard_id}/{num_shards} over {self.num_rows} rows in {self.data_dir}")
        global_idx = 0
        for file in self.files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=PARQUET_READ_BATCH_ROWS, columns=keys):
                record: Row
                for record in batch.to_pylist():
                    if global_idx % num_shards == shard_id:
                        record["data_signature"] = self.data_signature
                        record["data_id"] = self.prefix
                        yield record
                    global_idx += 1


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
