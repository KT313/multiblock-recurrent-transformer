# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Turn a run's `dataset_config` into what the training loop consumes: verified (or auto-prepared) data
directories per stage, the tokenizer path and the stage token budgets for `StageManager`.

Framework-neutral apart from `data_preparation.lib.*`; no torch. The only cross-over between the run config and
the dataset config happens here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from data_preparation.lib.build import build, plan as compute_plan, status
from data_preparation.dataset_config import DatasetConfig, StageConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard
from training.settings import DataEntry, Settings
from training.stage_manager import TrainingStage

log = get_logger(__name__)

INSTRUCT_DATA_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}
CHECKPOINT_HASH_KEY = "dataset_config_hash"


class _MainRankBarrier(Protocol):
    """The two `training.backend.Backend` members the resolver needs (kept as a Protocol to stay torch-free)."""

    is_main: bool

    def barrier(self) -> None: ...


@dataclass
class ResolvedStage:
    """One training stage with its data directories resolved on disk."""

    name: str
    tokens: int
    base_lr: float
    transition_pct: float
    train_data: list[DataEntry]
    val_data: list[DataEntry]


@dataclass
class ResolvedDataset:
    config: DatasetConfig
    config_hash: str
    tokenizer_dir: str
    stages: list[ResolvedStage]

    def stage_manager_stages(self) -> list[TrainingStage]:
        """The stages in the plain-dict form `training.stage_manager.StageManager` expects."""
        return [
            TrainingStage(
                name=stage.name,
                tokens=stage.tokens,
                base_lr=stage.base_lr,
                transition_pct=stage.transition_pct,
                train_data=[vars(entry) for entry in stage.train_data],
                val_data=[vars(entry) for entry in stage.val_data],
            )
            for stage in self.stages
        ]


def build_command(dataset_config: str, dataset_dir: str) -> str:
    """The CLI command that materialises the dataset config (quoted in error messages)."""
    return f"python data_preparation/prepare.py build --dataset_config {dataset_config} --dataset_dir {dataset_dir}"


# --- stage keys -> data entries -------------------------------------------------------------------------------------


def _data_entry(cfg: DatasetConfig, layout: DatasetLayout, stage_name: str, key: str, weight: float) -> DataEntry:
    """The `DataEntry` for one stage key (`<source>`, `<source>/validation`, `<mixture>`, `<mixture>/train` or
    `<mixture>/validation`)."""
    base, _, split = key.partition("/")
    prefix = f"{stage_name}-{key.replace('/', '-')}"

    if base in cfg.instruct_mixtures:
        mixture_dir = layout.instruct_mixture_dir(cfg.name, base, split or "train")
        return DataEntry(
            prefix=prefix, data_dir=str(mixture_dir), weight=weight, data_signature=dict(INSTRUCT_DATA_SIGNATURE)
        )

    kind = cfg.sources[base].kind
    if kind == "pretrain" and split == "validation":
        source_dir = layout.validation_dir(base)  # the source's own held-out split (`validation_tokens`)
    elif kind == "pretrain":
        source_dir = layout.source_dir(base, "processed")
    elif kind == "validation":
        source_dir = layout.validation_dir(base)
    else:  # unreachable: DatasetConfig rejects instruct sources outside instruct mixtures
        raise ValueError(f"stage {stage_name}: source {base!r} of kind {kind!r} cannot be used directly")
    return DataEntry(prefix=prefix, data_dir=str(source_dir), weight=weight)


def _entries(cfg: DatasetConfig, layout: DatasetLayout, stage_name: str, keys: dict[str, float]) -> list[DataEntry]:
    entries = [_data_entry(cfg, layout, stage_name, key, weight) for key, weight in keys.items()]
    prefixes = [entry.prefix for entry in entries]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"stage {stage_name}: duplicate data entry prefixes in {prefixes}")
    return entries


def resolve_entries(
    cfg: DatasetConfig, layout: DatasetLayout, stage: StageConfig
) -> tuple[list[DataEntry], list[DataEntry]]:
    """`(train_data, val_data)` of one dataset-config stage.

    A pretrain source maps to `sources/<name>/processed`, a validation source to `sources/<name>/validation` (both read
    the `text` column); `<mixture>` / `<mixture>/train` / `<mixture>/validation` map to the per-config mixture
    split with the instruction/input/output signature. Prefixes are `<stage>-<key>` and unique per stage.
    """
    return _entries(cfg, layout, stage.name, stage.train), _entries(cfg, layout, stage.name, stage.val)


# --- run config <-> dataset config ----------------------------------------------------------------------------------


def validate_settings(settings: Settings, cfg: DatasetConfig) -> None:
    """The two cross-checks between run config and dataset config."""
    if len(settings.stage_base_lrs) != len(cfg.stages):
        raise ValueError(
            f"stage_base_lrs has {len(settings.stage_base_lrs)} entries but dataset config "
            f"{settings.dataset_config!r} ({cfg.name}) has {len(cfg.stages)} stages "
            f"{[s.name for s in cfg.stages]}; give one base LR per stage, in order"
        )
    if settings.block_size > cfg.max_seq_length:
        raise ValueError(
            f"block_size {settings.block_size} exceeds max_seq_length {cfg.max_seq_length} of dataset config "
            f"{settings.dataset_config!r}; documents were token-counted with that cap"
        )


def _ensure_prepared(
    settings: Settings, cfg: DatasetConfig, layout: DatasetLayout, backend: Optional[_MainRankBarrier]
) -> None:
    """Verify the dataset on disk; build what is missing when `auto_prepare` allows it, else raise."""
    plan = status(cfg, layout)  # logs the status table
    if plan.complete:
        return
    missing = "\n  ".join(plan.missing())
    if not settings.auto_prepare:
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is not prepared under {settings.dataset_dir!r} "
            f"(auto_prepare is off). Missing:\n  {missing}\nRun:\n  "
            + build_command(settings.dataset_config, settings.dataset_dir)
        )

    log.info("dataset %s is incomplete, preparing missing data (%d item(s))", cfg.name, len(plan.missing()))
    if backend is None or backend.is_main:
        with Dashboard() as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=layout.root / BUILD_LOG_NAME):
            build(
                cfg,
                layout,
                num_workers=settings.prepare_num_workers,
                max_parallel_downloads=settings.prepare_max_parallel_downloads,
                hf_token=os.environ.get("HF_TOKEN"),
            )
    if backend is not None:
        backend.barrier()

    plan = compute_plan(cfg, layout)  # `build` already logged the final status table; re-verify silently
    if not plan.complete:
        still_missing = "\n  ".join(plan.missing())
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is still incomplete after preparing:\n  {still_missing}"
        )


def resolve_dataset(settings: Settings, backend: Optional[_MainRankBarrier] = None) -> ResolvedDataset:
    """Load, verify and (with `auto_prepare`) build the dataset of a run; see the module docstring.

    The build runs on the main rank only (`backend is None or backend.is_main`), followed by `backend.barrier()`.
    Raises `RuntimeError` when data is missing and cannot / must not be prepared here.
    """
    configure_logging(logging.INFO)
    cfg = load_dataset_config(settings.dataset_config)
    validate_settings(settings, cfg)
    layout = DatasetLayout(Path(settings.dataset_dir))
    _ensure_prepared(settings, cfg, layout, backend)

    stages: list[ResolvedStage] = []
    for stage, base_lr in zip(cfg.stages, settings.stage_base_lrs):
        train_data, val_data = resolve_entries(cfg, layout, stage)
        stages.append(
            ResolvedStage(
                name=stage.name,
                tokens=stage.tokens,
                base_lr=base_lr,
                transition_pct=stage.transition_pct,
                train_data=train_data,
                val_data=val_data,
            )
        )
    return ResolvedDataset(
        config=cfg,
        config_hash=cfg.config_hash(),
        tokenizer_dir=str(layout.tokenizer_dir(cfg.tokenizer.name)),
        stages=stages,
    )


def check_checkpoint_dataset_hash(extra: dict[str, Any], expected_hash: str, allow_change: bool) -> None:
    """Compare a checkpoint's stored dataset-config hash with the current one.

    A checkpoint without the key (older format) only logs a warning; a mismatch raises unless `allow_change`.
    """
    stored = extra.get(CHECKPOINT_HASH_KEY)
    if stored is None:
        log.warning("checkpoint carries no %s (older format); cannot verify the dataset config", CHECKPOINT_HASH_KEY)
        return
    if stored == expected_hash:
        return
    message = f"checkpoint was written with dataset config hash {stored}, the current dataset config hashes to {expected_hash}"
    if allow_change:
        log.warning("%s; continuing because allow_dataset_change is set", message)
        return
    raise RuntimeError(f"{message}. Set allow_dataset_change: true to resume anyway.")


__all__ = [
    "CHECKPOINT_HASH_KEY",
    "INSTRUCT_DATA_SIGNATURE",
    "ResolvedDataset",
    "ResolvedStage",
    "build_command",
    "check_checkpoint_dataset_hash",
    "resolve_dataset",
    "resolve_entries",
    "validate_settings",
]
