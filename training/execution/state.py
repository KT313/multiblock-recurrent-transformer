# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Objects shared by the training orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend
from training.checkpoint import CheckpointMetadata
from training.data.dataset_resolver import ResolvedDataset
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps import TrainingProgress


@dataclass(frozen=True)
class RunState:
    """
    The run once it is set up, handed to `restore_checkpoint_if_resuming`, `save_run_checkpoint` and
    `export_if_requested` as one argument. Every member keeps its identity for the whole run; only `progress` moves.
    """

    settings: Settings
    run_directory: Path
    backend: Backend
    model: Module
    optimizer: Optimizer
    dataset: ResolvedDataset
    stage_manager: StageManager
    progress: TrainingProgress



@dataclass(frozen=True)
class ResumePoint:
    """
    Where a resumed run continues from: the checkpoint it was restored from and the data-stream state stored in
    it (None in a checkpoint written before the stream existed), for `BatchStream.load_state_dict`.
    """

    checkpoint: Path
    data_stream: dict[str, Any] | None
    metadata: CheckpointMetadata
