# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Turn a run's `dataset_config` into what the training loop consumes: verified (or auto-prepared) data
directories per stage with their row ranges, the tokenizer path and the stage token budgets for `StageManager`.

The validation split is decided here, once per source and run, never by the data pipeline: a source used only for
training is read whole, a source used only for validation is read whole as validation, and a source used for both
holds out its first `ceil(validation_fraction_of(source) × rows)` processed rows (`validation_rows`) for validation
and trains on the rest. Rows are counted from the parquet footers of `processed/<source>` and cross-checked against
the manifest; the same source gets the same split in every stage. The chosen `validation_rows` per source travel
with every checkpoint next to `dataset_config_hash` and are verified on resume.

Framework-neutral apart from `data_preparation.*` (manifests, parquet footers); no torch. The only cross-over
between the run config and the dataset config happens here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from data_preparation.lib.build import prepare, status, summarize_dataset_state
from data_preparation.dataset_config import DatasetConfig, StageConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.log import ROOT_LOGGER_NAME, configure_logging, get_logger
from data_preparation.lib.storage.manifest import MANIFEST_NAME, Manifest, shard_rows
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard
from training.settings import Settings
from training.stage_manager import TrainingStage

log = get_logger(__name__)

INSTRUCT_DATA_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}
CHECKPOINT_HASH_KEY = "dataset_config_hash"
CHECKPOINT_VALIDATION_ROWS_KEY = "dataset_validation_rows"  # {source: validation_rows} as chosen by `resolve_splits`

Part = Literal["train", "val"]


@dataclass
class DataEntry:
    """One parquet dataset directory inside a stage mixture, with the row range the loader reads from it."""

    prefix: str  # unique name within its stage, used for logging
    data_dir: str  # directory with *.parquet files
    weight: float = 1.0  # sampling weight relative to the other entries of the same stage
    data_signature: Optional[dict[str, Any]] = None  # {"keys": [...], "format_fn": "..."}; default: text column
    skip_rows: int = 0  # rows of the directory to skip from the start (shard order data-00000, data-00001, ...)
    max_rows: Optional[int] = None  # at most this many rows after the skip; None = up to the last row


class _MainRankBarrier(Protocol):
    """The two `training.backend.Backend` members the resolver needs (kept as a Protocol to stay torch-free)."""

    is_main: bool

    def barrier(self) -> None: ...


@dataclass
class ResolvedStage:
    """One training stage with its data directories (and row ranges) resolved on disk."""

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
    validation_rows: dict[str, int]  # per source: rows [0, n) of processed/<source> are validation, the rest training

    def training_stages(self) -> list[TrainingStage]:
        """The stages as `training.stage_manager.StageManager` takes them: name, token budget, base LR and transition
        length per stage (the data entries stay here; the manager never reads them)."""
        return [
            TrainingStage(name=stage.name, tokens=stage.tokens, base_lr=stage.base_lr, transition_pct=stage.transition_pct)
            for stage in self.stages
        ]


def build_command(dataset_config: str, dataset_dir: str) -> str:
    """The CLI command that materialises the dataset config (quoted in error messages)."""
    return f"python data_preparation/prepare.py prepare --dataset_config {dataset_config} --dataset_dir {dataset_dir}"


# --- the validation split ---------------------------------------------------------------------------------------------


def validation_rows_of(cfg: DatasetConfig, source_name: str, total_rows: int) -> int:
    """How many of the first rows of `processed/<source>` (`total_rows` in all) are validation rows.

    Every row of a source used only for validation, none of a source used only for training (its
    `validation_fraction_of` is 0), `ceil(validation_fraction_of(source) × total_rows)` of a source used for both.
    The fraction is multiplied as the decimal written in the YAML (`Fraction(str(...))`): a float product such as
    `0.07 × 100 = 7.000000000000001` would otherwise round up one row too many.
    """
    if not cfg.used_in_train(source_name):
        return total_rows
    return ceil(Fraction(str(cfg.validation_fraction_of(source_name))) * total_rows)


def _rows_on_disk(directory: Path, what: str) -> int:
    """Rows of the `data-*.parquet` shards of `directory` from their parquet footers (no manifest involved); the
    directory must exist and hold at least one shard. `what` names the item checked in the errors."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{what}: processed folder {directory} does not exist")
    shards = sorted(directory.glob("data-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"{what}: processed folder {directory} holds no data-*.parquet shard")
    return sum(shard_rows(shard) for shard in shards)


def processed_rows(directory: Path, what: str) -> int:
    """Rows of a processed folder: counted from the parquet footers and cross-checked against its manifest (a
    listed shard with the wrong count or an unlisted shard on disk is an error naming the folder)."""
    rows_on_disk = _rows_on_disk(directory, what)
    manifest = Manifest.load(directory)
    if manifest is None:
        raise RuntimeError(f"{what}: processed folder {directory} has no {MANIFEST_NAME}")
    if manifest.rows() != rows_on_disk:
        raise RuntimeError(
            f"{what}: {MANIFEST_NAME} of {directory} lists {manifest.rows()} rows in {len(manifest.shards)} shard(s) but "
            f"the data-*.parquet shard(s) on disk hold {rows_on_disk}; rebuild the folder"
        )
    return rows_on_disk


def _stage_keys(cfg: DatasetConfig, source_name: str) -> str:
    """`pretrain_a.train, pretrain_a.val, ...`: where a source is used (for error messages)."""
    keys = [f"{stage.name}.train" for stage in cfg.stages if source_name in stage.train]
    keys += [f"{stage.name}.val" for stage in cfg.stages if source_name in stage.val]
    return ", ".join(keys)


def resolve_splits(cfg: DatasetConfig, layout: DatasetLayout) -> dict[str, int]:
    """`{source: validation_rows}` for every source of the config, from the rows in `processed/<source>` on disk
    (footers cross-checked against the manifest). Decided once per run; every stage uses the same split."""
    splits: dict[str, int] = {}
    for name in cfg.sources:
        directory = layout.processed_dir(name)
        total = processed_rows(directory, f"source {name!r} (stage keys {_stage_keys(cfg, name)})")
        validation = validation_rows_of(cfg, name, total)
        if validation == 0:
            detail = "all training"
        elif validation == total:
            detail = "all validation"
        else:
            detail = f"rows [0, {validation}) validation ({cfg.validation_fraction_of(name):.1%}), [{validation}, {total}) training"
        log.info("source %s: %d processed rows, %s", name, total, detail)
        splits[name] = validation
    return splits


# --- stage keys -> data entries -------------------------------------------------------------------------------------


def _data_entry(
    cfg: DatasetConfig, layout: DatasetLayout, stage_name: str, key: str, weight: float, part: Part, validation_rows: int
) -> DataEntry:
    """The `DataEntry` for one stage key (a source name): its `processed/<source>` folder, read through the text
    column (pretrain) or the instruction/input/output signature (instruct), restricted to the validation rows
    `[0, validation_rows)` (part `val`) or the training rows from `validation_rows` on (part `train`)."""
    prefix = f"{stage_name}-{key}"
    signature = dict(INSTRUCT_DATA_SIGNATURE) if cfg.sources[key].kind == "instruct" else None
    skip_rows, max_rows = (0, validation_rows) if part == "val" else (validation_rows, None)
    return DataEntry(
        prefix=prefix,
        data_dir=str(layout.processed_dir(key)),
        weight=weight,
        data_signature=signature,
        skip_rows=skip_rows,
        max_rows=max_rows,
    )


def _entries(
    cfg: DatasetConfig,
    layout: DatasetLayout,
    stage_name: str,
    keys: dict[str, float],
    part: Part,
    validation_rows: Mapping[str, int],
) -> list[DataEntry]:
    entries = [_data_entry(cfg, layout, stage_name, key, weight, part, validation_rows[key]) for key, weight in keys.items()]
    prefixes = [entry.prefix for entry in entries]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"stage {stage_name}: duplicate data entry prefixes in {prefixes}")
    return entries


def resolve_entries(
    cfg: DatasetConfig, layout: DatasetLayout, stage: StageConfig, validation_rows: Mapping[str, int]
) -> tuple[list[DataEntry], list[DataEntry]]:
    """`(train_data, val_data)` of one dataset-config stage.

    Every stage key is a source name and maps to `processed/<name>`; pretrain sources read the `text` column,
    instruct sources use the instruction/input/output signature. `validation_rows[name]` (see `resolve_splits`)
    gives the row range: validation entries read rows `[0, validation_rows)`, training entries the rows from
    `validation_rows` to the end. Prefixes are `<stage>-<key>` and unique per stage. Pure path arithmetic, no I/O.
    """
    return (
        _entries(cfg, layout, stage.name, stage.train, "train", validation_rows),
        _entries(cfg, layout, stage.name, stage.val, "val", validation_rows),
    )


def check_entries_on_disk(stages: list[ResolvedStage]) -> None:
    """Direct filesystem check of every data entry, independent of manifests and of the planner: the directory
    exists, holds at least one `data-*.parquet` shard, and the entry's row range (clipped to the rows on disk as
    `ParquetTextDataset` clips it) contains at least one row. Errors name the stage, the part, the entry
    (`<stage>-<source>`) and the folder."""
    rows_cache: dict[str, int] = {}
    for stage in stages:
        for part, entries in (("train", stage.train_data), ("val", stage.val_data)):
            for entry in entries:
                what = f"stage {stage.name!r} {part} entry {entry.prefix!r}"
                if entry.data_dir not in rows_cache:
                    rows_cache[entry.data_dir] = _rows_on_disk(Path(entry.data_dir), what)
                total = rows_cache[entry.data_dir]
                start = min(entry.skip_rows, total)
                stop = total if entry.max_rows is None else min(total, start + entry.max_rows)
                if stop <= start:
                    end = "end" if entry.max_rows is None else str(entry.skip_rows + entry.max_rows)
                    raise RuntimeError(
                        f"{what}: row range [{entry.skip_rows}, {end}) of {entry.data_dir} is empty ({total} rows on "
                        f"disk); the {'validation' if part == 'val' else 'training'} part of the split has no row"
                    )


# --- run config <-> dataset config ----------------------------------------------------------------------------------


def validate_settings(settings: Settings, cfg: DatasetConfig) -> None:
    """The two cross-checks between run config and dataset config: one base LR per stage, and the same
    `block_size` (the planner counted sequences with the dataset config's; `block_size <= max_seq_length` follows
    from the dataset-config schema)."""
    if len(settings.stage_base_lrs) != len(cfg.stages):
        raise ValueError(
            f"stage_base_lrs has {len(settings.stage_base_lrs)} entries but dataset config "
            f"{settings.dataset_config!r} ({cfg.name}) has {len(cfg.stages)} stages "
            f"{[s.name for s in cfg.stages]}; give one base LR per stage, in order"
        )
    if settings.block_size != cfg.block_size:
        raise ValueError(
            f"block_size {settings.block_size} of the run config does not match block_size {cfg.block_size} of dataset "
            f"config {settings.dataset_config!r}; the planner sized the data in sequences of the dataset config's "
            "block_size, so the two must be equal"
        )


def _ensure_prepared(
    settings: Settings, cfg: DatasetConfig, layout: DatasetLayout, backend: Optional[_MainRankBarrier]
) -> None:
    """Verify the dataset on disk; prepare what is missing when `auto_prepare` allows it, else raise.

    Auto-prepare never deletes raw data and never prompts: it runs `prepare` with `assume_yes=False` and a `confirm`
    that always declines, so a stale or outdated raw folder fails the run with the same message `prepare.py` prints
    (rerun it with `--yes` to confirm the deletion).
    """
    report = status(settings.dataset_config, settings.dataset_dir)  # logs the status table
    if report.complete:
        return
    missing = ", ".join(report.missing())
    if not settings.auto_prepare:
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is not prepared under {settings.dataset_dir!r} "
            f"(auto_prepare is off). Missing: {missing}\n{report.describe()}\nRun:\n  "
            + build_command(settings.dataset_config, settings.dataset_dir)
        )

    log.info("dataset %s is incomplete, preparing missing data (%s)", cfg.name, missing)
    if backend is None or backend.is_main:
        with Dashboard() as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=layout.root / BUILD_LOG_NAME):
            prepare(
                settings.dataset_config,
                settings.dataset_dir,
                num_workers=settings.prepare_num_workers,
                max_parallel_downloads=settings.prepare_max_parallel_downloads,
                assume_yes=False,
                confirm=lambda _message: False,  # never delete raw from a training run
                hf_token=os.environ.get("HF_TOKEN"),
            )
    if backend is not None:
        backend.barrier()

    report = summarize_dataset_state(cfg, layout)  # `prepare` already logged the final status table; re-verify silently
    if not report.complete:
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is still incomplete after preparing: "
            f"{', '.join(report.missing())}\n{report.describe()}"
        )


def resolve_dataset(settings: Settings, backend: Optional[_MainRankBarrier] = None) -> ResolvedDataset:
    """Load, verify and (with `auto_prepare`) build the dataset of a run, then decide the validation split; see the
    module docstring.

    The build runs on the main rank only (`backend is None or backend.is_main`), followed by `backend.barrier()`.
    Raises `RuntimeError` when data is missing and cannot / must not be prepared here, or when a folder on disk
    does not match its manifest or leaves a used part of the split empty.
    """
    configure_logging(logging.INFO)
    cfg = load_dataset_config(settings.dataset_config)
    validate_settings(settings, cfg)
    layout = DatasetLayout(Path(settings.dataset_dir))
    _ensure_prepared(settings, cfg, layout, backend)
    validation_rows = resolve_splits(cfg, layout)

    stages: list[ResolvedStage] = []
    for stage, base_lr in zip(cfg.stages, settings.stage_base_lrs):
        train_data, val_data = resolve_entries(cfg, layout, stage, validation_rows)
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
    check_entries_on_disk(stages)
    return ResolvedDataset(
        config=cfg,
        config_hash=cfg.config_hash(),
        tokenizer_dir=str(layout.tokenizer_dir(cfg.tokenizer.name)),
        stages=stages,
        validation_rows=validation_rows,
    )


# --- resume checks ---------------------------------------------------------------------------------------------------


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


def check_checkpoint_validation_rows(extra: dict[str, Any], expected: Mapping[str, int], allow_change: bool) -> None:
    """Compare a checkpoint's stored validation split (`{source: validation_rows}`) with the freshly resolved one.

    The split can only differ when the data on disk changed (a source grew or shrank, a source was added or
    removed); a resumed run would then validate on rows it has trained on, or vice versa. A checkpoint without the
    key (older format) only logs a warning; a mismatch raises unless `allow_change`, naming every source and both
    numbers.
    """
    stored: Any = extra.get(CHECKPOINT_VALIDATION_ROWS_KEY)
    if stored is None:
        log.warning(
            "checkpoint carries no %s (older format); cannot verify the validation split", CHECKPOINT_VALIDATION_ROWS_KEY
        )
        return
    mismatches = [
        f"{name!r}: checkpoint {stored.get(name, 'absent')}, now {expected.get(name, 'absent')}"
        for name in sorted(set(stored) | set(expected))
        if stored.get(name) != expected.get(name)
    ]
    if not mismatches:
        return
    message = "the validation split differs from the checkpoint's (validation rows per source: " + "; ".join(mismatches) + ")"
    if allow_change:
        log.warning("%s; continuing because allow_dataset_change is set", message)
        return
    raise RuntimeError(f"{message}. Set allow_dataset_change: true to resume anyway.")


__all__ = [
    "CHECKPOINT_HASH_KEY",
    "CHECKPOINT_VALIDATION_ROWS_KEY",
    "INSTRUCT_DATA_SIGNATURE",
    "DataEntry",
    "ResolvedDataset",
    "ResolvedStage",
    "build_command",
    "check_checkpoint_dataset_hash",
    "check_checkpoint_validation_rows",
    "check_entries_on_disk",
    "processed_rows",
    "resolve_dataset",
    "resolve_entries",
    "resolve_splits",
    "validate_settings",
    "validation_rows_of",
]
