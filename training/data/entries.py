# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Framework-neutral descriptions of resolved training sources and validation stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from data_preparation import DatasetConfig


@dataclass
class DataEntry:
    """
    One parquet dataset directory with the row range its loader reads: a run-wide train source
    (`ResolvedDataset.train_sources`, prefix = the source name) or a member of a stage's validation mixture
    (`ResolvedStage.val_data`, prefix = `<stage>-<source>`).
    """

    prefix: str  # unique name within its list, used for logging (train: the data_id of `data_composition/...`)
    data_dir: str  # directory with *.parquet files
    weight: float = 1.0  # sampling weight relative to the other entries of the same stage
    data_signature: Optional[dict[str, Any]] = None  # {"keys": [...], "format_fn": "..."}; default: text column
    skip_rows: int = 0  # rows of the directory to skip from the start (shard order data-00000, data-00001, ...)
    max_rows: Optional[int] = None  # at most this many rows after the skip; None = up to the last row


@dataclass
class ResolvedStage:
    """
    One training stage: token budget, base LR, transition length, sampling weights over the run-wide train sources
    and its validation entries resolved on disk. The stage structure changes the WEIGHTS only.
    """

    name: str
    tokens: int
    base_lr: float
    transition_pct: float
    train_weights: dict[str, float]  # source name -> sampling weight (the dataset config's `stage.train`, sum 1)
    val_data: list[DataEntry]


@dataclass
class ResolvedDataset:
    config: DatasetConfig
    config_hash: str
    tokenizer_dir: str
    stages: list[ResolvedStage]
    train_sources: list[DataEntry]  # one per source any stage trains on, in config order; read by ONE loader all run
    validation_rows: dict[str, int]  # per source: rows [0, n) of processed/<source> are validation, the rest training
    source_rows: dict[str, int]  # per source: the rows of processed/<source> (what a checkpoint stores and a resume verifies)
    rows_on_disk: dict[str, int]  # per processed directory (`DataEntry.data_dir`): its rows, counted once at setup
    dataset_build_id: str | None = None  # None only for legacy/manual callers, never a resolved managed dataset
