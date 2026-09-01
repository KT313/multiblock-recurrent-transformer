# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Turn a run's `dataset_config` into what the training loop consumes: verified (or auto-prepared) data
directories per stage with their row ranges, the tokenizer path and the stage token budgets for `StageManager`.

The validation split is decided here, once per source and run, never by the data pipeline: a source used only for
training is read whole, a source used only for validation is read whole as validation, and a source used for both
holds out its first `ceil(validation_fraction_of(source) × rows)` processed rows (`validation_rows`) for validation
and trains on the rest. Rows are counted from the parquet footers of `processed/<source>` and cross-checked against
the manifest; the same source gets the same split in every stage. The chosen `validation_rows` per source travel
with every checkpoint next to `dataset_config_hash` and are verified on resume (`check_dataset_unchanged`).
The split is also checked against what the loaders and evaluation need: a stage whose validation loader cannot fill
one micro-batch (`check_validation_batches`) and an entry with fewer rows than its loader has worker shards
(`check_entry_shards`) fail here, at setup, instead of at the first evaluation step or mid-training in a worker.

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
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

from data_preparation.lib.abort import StopCheck
from data_preparation.lib.build import ConfirmationRequired, prepare, status, summarize_dataset_state
from data_preparation.dataset_config import DatasetConfig, StageConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.log import ROOT_LOGGER_NAME, get_logger
from data_preparation.lib.storage.manifest import MANIFEST_NAME, Manifest, shard_rows
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard
from training.settings import Settings
from training.stage_manager import TrainingStage

if TYPE_CHECKING:  # annotation only: this module stays torch-free, `training.checkpoint` imports torch
    from training.checkpoint import CheckpointMetadata

log = get_logger(__name__)

INSTRUCT_DATA_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}

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
    """The `training.backend.Backend` members the resolver needs (kept as a Protocol to stay torch-free):
    `is_main` / `barrier` for the build, `world_size` for the validation-batch check."""

    is_main: bool
    world_size: int

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


def validation_rows_of(dataset_config: DatasetConfig, source_name: str, total_rows: int) -> int:
    """How many of the first rows of `processed/<source>` (`total_rows` in all) are validation rows.

    Every row of a source used only for validation, none of a source used only for training (its
    `validation_fraction_of` is 0), `ceil(validation_fraction_of(source) × total_rows)` of a source used for both.
    The fraction is multiplied as the decimal written in the YAML (`Fraction(str(...))`): a float product such as
    `0.07 × 100 = 7.000000000000001` would otherwise round up one row too many.
    """
    if not dataset_config.used_in_train(source_name):
        return total_rows
    return ceil(Fraction(str(dataset_config.validation_fraction_of(source_name))) * total_rows)


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


def _stage_keys(dataset_config: DatasetConfig, source_name: str) -> str:
    """`pretrain_a.train, pretrain_a.val, ...`: where a source is used (for error messages)."""
    keys = [f"{stage.name}.train" for stage in dataset_config.stages if source_name in stage.train]
    keys += [f"{stage.name}.val" for stage in dataset_config.stages if source_name in stage.val]
    return ", ".join(keys)


def resolve_splits(dataset_config: DatasetConfig, layout: DatasetLayout) -> dict[str, int]:
    """`{source: validation_rows}` for every source of the config, from the rows in `processed/<source>` on disk
    (footers cross-checked against the manifest). Decided once per run; every stage uses the same split."""
    splits: dict[str, int] = {}
    for name in dataset_config.sources:
        directory = layout.processed_dir(name)
        total = processed_rows(directory, f"source {name!r} (stage keys {_stage_keys(dataset_config, name)})")
        validation = validation_rows_of(dataset_config, name, total)
        if validation == 0:
            detail = "all training"
        elif validation == total:
            detail = "all validation"
        else:
            detail = f"rows [0, {validation}) validation ({dataset_config.validation_fraction_of(name):.1%}), [{validation}, {total}) training"
        log.info("source %s: %d processed rows, %s", name, total, detail)
        splits[name] = validation
    return splits


# --- stage keys -> data entries -------------------------------------------------------------------------------------


def _data_entry(
    dataset_config: DatasetConfig, layout: DatasetLayout, stage_name: str, key: str, weight: float, part: Part, validation_rows: int
) -> DataEntry:
    """The `DataEntry` for one stage key (a source name): its `processed/<source>` folder, read through the text
    column (pretrain) or the instruction/input/output signature (instruct), restricted to the validation rows
    `[0, validation_rows)` (part `val`) or the training rows from `validation_rows` on (part `train`)."""
    prefix = f"{stage_name}-{key}"
    signature = dict(INSTRUCT_DATA_SIGNATURE) if dataset_config.sources[key].kind == "instruct" else None
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
    dataset_config: DatasetConfig,
    layout: DatasetLayout,
    stage_name: str,
    keys: dict[str, float],
    part: Part,
    validation_rows: Mapping[str, int],
) -> list[DataEntry]:
    entries = [_data_entry(dataset_config, layout, stage_name, key, weight, part, validation_rows[key]) for key, weight in keys.items()]
    prefixes = [entry.prefix for entry in entries]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"stage {stage_name}: duplicate data entry prefixes in {prefixes}")
    return entries


def resolve_entries(
    dataset_config: DatasetConfig, layout: DatasetLayout, stage: StageConfig, validation_rows: Mapping[str, int]
) -> tuple[list[DataEntry], list[DataEntry]]:
    """`(train_data, val_data)` of one dataset-config stage.

    Every stage key is a source name and maps to `processed/<name>`; pretrain sources read the `text` column,
    instruct sources use the instruction/input/output signature. `validation_rows[name]` (see `resolve_splits`)
    gives the row range: validation entries read rows `[0, validation_rows)`, training entries the rows from
    `validation_rows` to the end. Prefixes are `<stage>-<key>` and unique per stage. Pure path arithmetic, no I/O.
    """
    return (
        _entries(dataset_config, layout, stage.name, stage.train, "train", validation_rows),
        _entries(dataset_config, layout, stage.name, stage.val, "val", validation_rows),
    )


def entry_rows_in_range(entry: DataEntry, total_rows: int) -> int:
    """Rows a loader actually reads from `entry`: its range `[skip_rows, skip_rows + max_rows)` clipped to the
    `total_rows` on disk, exactly as `ParquetTextDataset` clips it."""
    start = min(entry.skip_rows, total_rows)
    stop = total_rows if entry.max_rows is None else min(total_rows, start + entry.max_rows)
    return stop - start


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
                if entry_rows_in_range(entry, total) <= 0:
                    end = "end" if entry.max_rows is None else str(entry.skip_rows + entry.max_rows)
                    raise RuntimeError(
                        f"{what}: row range [{entry.skip_rows}, {end}) of {entry.data_dir} is empty ({total} rows on "
                        f"disk); the {'validation' if part == 'val' else 'training'} part of the split has no row"
                    )


def loader_shards(num_workers: int, world_size: int) -> int:
    """Shards a loader's datasets deal their rows over: one per dataloader worker and rank
    (`ParquetTextDataset._shard`). `num_workers=0` loads in the calling process, which is ONE shard per rank."""
    return world_size * max(num_workers, 1)


def check_entry_shards(stages: list[ResolvedStage], dataloader_num_workers: int, world_size: int = 1) -> None:
    """Fail at setup when a data entry has fewer rows than the loader has shards.

    `ParquetTextDataset` deals the rows of an entry's range round-robin over `world_size × num_workers` shards, so a
    shard is empty as soon as the range holds fewer rows than there are shards. An empty member of a
    `WeightedMixtureDataset` is fatal mid-run: the mixture restarts a member that runs out and immediately gets a
    second `StopIteration` from the empty shard, which Python turns into a `RuntimeError` inside the worker and kills
    the training run. Checked for the train entries (`dataloader_num_workers`) and the validation entries (built with
    `num_workers=0`, so one shard per rank) alike; the error names the entry, its folder, its row count and the
    worker count.
    """
    rows_cache: dict[str, int] = {}
    for stage in stages:
        for part, entries, num_workers in (
            ("train", stage.train_data, dataloader_num_workers),
            ("val", stage.val_data, 0),
        ):
            shards = loader_shards(num_workers, world_size)
            if shards <= 1:  # a single in-process shard reads the whole range; nothing to split
                continue
            for entry in entries:
                what = f"stage {stage.name!r} {part} entry {entry.prefix!r}"
                if entry.data_dir not in rows_cache:
                    rows_cache[entry.data_dir] = _rows_on_disk(Path(entry.data_dir), what)
                rows = entry_rows_in_range(entry, rows_cache[entry.data_dir])
                if rows >= shards:
                    continue
                ranks = f" × world size {world_size}" if world_size > 1 else ""
                fix = (
                    f"Lower dataloader_num_workers to at most {rows}, or give the source more rows"
                    if num_workers > 0
                    else "Give the source more rows, or lower the world size"
                )
                raise ValueError(
                    f"{what}: {entry.data_dir} gives it {rows} row(s) after the validation split, but its loader "
                    f"deals the rows round-robin over {shards} shards ({num_workers} dataloader worker(s){ranks}), "
                    f"so {shards - rows} shard(s) would be empty and the run would fail during training. {fix}"
                )


def validation_batches_available(entries: list[DataEntry], micro_batch_size: int, world_size: int) -> int | None:
    """How many micro-batches one evaluation can draw from a stage's validation loader; `None` = unbounded.

    What `training.data.loader.build_dataloader` builds decides this, and the two cases differ fundamentally:

    * ONE entry: the loader reads that single `ParquetTextDataset` directly. One `__iter__` is one epoch over the
      entry's row range and then stops, so the loader is FINITE — `ceil(rows / micro_batch_size)` batches (the last
      one short; `drop_last` is off). This is the case that can come up short of `eval_iters`.
    * SEVERAL entries: the loader reads a `WeightedMixtureDataset`, which restarts every member that runs out and
      therefore never raises `StopIteration`. Such a loader always delivers `eval_iters` batches (with repeated
      rows once the smallest member has wrapped around), so there is nothing to check — hence `None`.

    Rows are dealt round-robin over `world_size × num_workers` shards (`ParquetTextDataset`); validation loaders run
    with `num_workers=0`, so each rank reads every `world_size`-th row and the smallest shard holds
    `rows // world_size` of them. The row range is clipped to the rows on disk exactly as the dataset clips it.
    """
    if len(entries) != 1:
        return None
    entry = entries[0]
    total = _rows_on_disk(Path(entry.data_dir), f"validation entry {entry.prefix!r}")
    rows_per_rank = entry_rows_in_range(entry, total) // world_size
    return -(-rows_per_rank // micro_batch_size)  # ceil, in integers


def check_validation_batches(
    stages: list[ResolvedStage], micro_batch_size: int, eval_iters: int, world_size: int = 1
) -> None:
    """Fail (or warn) at setup time about a validation split that cannot feed `training.evaluation.evaluate`.

    A stage whose validation loader delivers no batch at all is a hard error naming the stage, its validation
    entries and both numbers — `evaluate` would otherwise raise in the middle of the run, at the first evaluation
    step. Fewer than `eval_iters` batches is only a warning: the loader still hands out a last, short batch, and
    `evaluate` averages the batches it actually receives, so the reported loss stays correct — it is just measured
    on less data than the config asks for. Loaders that mix several sources are unbounded (see
    `validation_batches_available`) and are never reported.
    """
    for stage in stages:
        available = validation_batches_available(stage.val_data, micro_batch_size, world_size)
        if available is None:
            continue
        entries = ", ".join(entry.prefix for entry in stage.val_data)
        rank = f" per rank (world size {world_size})" if world_size > 1 else ""
        if available == 0:
            raise RuntimeError(
                f"stage {stage.name!r}: its validation data ({entries}) yields 0 micro-batches of {micro_batch_size} "
                f"rows{rank} but eval_iters is {eval_iters}, so evaluation would have nothing to score. Raise "
                "validation_fraction for the source in the dataset config, give the stage a larger validation "
                "source, or lower micro_batch_size"
            )
        if available < eval_iters:
            log.warning(
                "stage %s: its validation data (%s) yields %d micro-batch(es) of %d rows%s, fewer than eval_iters "
                "(%d); every evaluation of this stage averages the %d batch(es) it gets",
                stage.name,
                entries,
                available,
                micro_batch_size,
                rank,
                eval_iters,
                available,
            )


# --- run config <-> dataset config ----------------------------------------------------------------------------------


def validate_settings(settings: Settings, dataset_config: DatasetConfig) -> None:
    """The two cross-checks between run config and dataset config: one base LR per stage, and the same
    `block_size` (the planner counted sequences with the dataset config's; `block_size <= max_seq_length` follows
    from the dataset-config schema)."""
    if len(settings.stage_base_lrs) != len(dataset_config.stages):
        raise ValueError(
            f"stage_base_lrs has {len(settings.stage_base_lrs)} entries but dataset config "
            f"{settings.dataset_config!r} ({dataset_config.name}) has {len(dataset_config.stages)} stages "
            f"{[s.name for s in dataset_config.stages]}; give one base LR per stage, in order"
        )
    if settings.block_size != dataset_config.block_size:
        raise ValueError(
            f"block_size {settings.block_size} of the run config does not match block_size {dataset_config.block_size} of dataset "
            f"config {settings.dataset_config!r}; the planner sized the data in sequences of the dataset config's "
            "block_size, so the two must be equal"
        )


def _ensure_prepared(
    settings: Settings,
    dataset_config: DatasetConfig,
    layout: DatasetLayout,
    backend: Optional[_MainRankBarrier],
    should_stop: StopCheck | None = None,
) -> None:
    """Verify the dataset on disk; prepare what is missing when `auto_prepare` allows it, else raise.

    Auto-prepare never deletes or truncates raw data and never prompts: it runs `prepare` with `assume_yes=False`
    and a `confirm` that always declines, so a stale or outdated raw folder — or a broken one whose truncation
    would drop healthy shards — fails the run with the list of folders `prepare.py` would ask about and the
    `prepare.py prepare ... --yes` command that confirms the repair. `should_stop` is the run's stop request (the CLI's Ctrl-C):
    the build polls it between shards and raises `BuildAborted` with everything published so far kept.
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

    log.info("dataset %s is incomplete, preparing missing data (%s)", dataset_config.name, missing)
    if backend is None or backend.is_main:
        with Dashboard() as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=layout.root / BUILD_LOG_NAME):
            try:
                prepare(
                    settings.dataset_config,
                    settings.dataset_dir,
                    num_workers=settings.prepare_num_workers,
                    pass_workers=settings.prepare_pass_workers,
                    max_parallel_downloads=settings.prepare_max_parallel_downloads,
                    assume_yes=False,
                    confirm=lambda _message: False,  # never delete or truncate raw from a training run
                    hf_token=os.environ.get("HF_TOKEN"),
                    should_stop=should_stop,
                )
            except ConfirmationRequired as err:
                raise RuntimeError(
                    f"{err.message.rstrip()}\nauto-prepare never deletes or truncates raw data; to confirm the repair run:\n  "
                    + build_command(settings.dataset_config, settings.dataset_dir)
                    + " --yes"
                ) from err
    if backend is not None:
        backend.barrier()

    report = summarize_dataset_state(dataset_config, layout)  # `prepare` already logged the final status table; re-verify silently
    if not report.complete:
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is still incomplete after preparing: "
            f"{', '.join(report.missing())}\n{report.describe()}"
        )


def resolve_dataset(
    settings: Settings, backend: Optional[_MainRankBarrier] = None, *, should_stop: StopCheck | None = None
) -> ResolvedDataset:
    """Load, verify and (with `auto_prepare`) build the dataset of a run, then decide the validation split; see the
    module docstring.

    The build runs on the main rank only (`backend is None or backend.is_main`), followed by `backend.barrier()`,
    and polls `should_stop` between shards (`BuildAborted` when it says stop; None: never).
    Raises `RuntimeError` when data is missing and cannot / must not be prepared here, when a folder on disk does
    not match its manifest or leaves a used part of the split empty, or when a stage's validation loader cannot
    fill one evaluation micro-batch (`check_validation_batches`, which warns about a split shorter than
    `eval_iters` batches); `ValueError` when an entry has fewer rows than its loader has worker shards
    (`check_entry_shards`).
    """
    dataset_config = load_dataset_config(settings.dataset_config)
    validate_settings(settings, dataset_config)
    layout = DatasetLayout(Path(settings.dataset_dir))
    _ensure_prepared(settings, dataset_config, layout, backend, should_stop)
    validation_rows = resolve_splits(dataset_config, layout)

    stages: list[ResolvedStage] = []
    for stage, base_lr in zip(dataset_config.stages, settings.stage_base_lrs):
        train_data, val_data = resolve_entries(dataset_config, layout, stage, validation_rows)
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
    world_size = 1 if backend is None else backend.world_size
    check_entries_on_disk(stages)
    check_entry_shards(stages, settings.dataloader_num_workers, world_size)
    check_validation_batches(stages, settings.micro_batch_size, settings.eval_iters, world_size)
    return ResolvedDataset(
        config=dataset_config,
        config_hash=dataset_config.config_hash(),
        tokenizer_dir=str(layout.tokenizer_dir(dataset_config.tokenizer.name)),
        stages=stages,
        validation_rows=validation_rows,
    )


# --- resume checks ---------------------------------------------------------------------------------------------------


def check_dataset_unchanged(metadata: CheckpointMetadata, dataset: ResolvedDataset, allow_change: bool) -> None:
    """Verify that a checkpoint was written against the dataset the run now resolves to.

    Two things are compared: the dataset-config hash (sources, stages, budgets, tokenizer, processing) and the
    validation split (`{source: validation_rows}`, which only differs when the data on disk changed — a source
    grew or shrank, was added or removed; a resumed run would then validate on rows it has trained on, or vice
    versa). A mismatch raises `RuntimeError` naming every difference (for the split: every source and both numbers)
    unless `allow_change` (`allow_dataset_change` in the run config), which only logs a warning.
    """
    problems: list[str] = []
    if metadata.dataset_config_hash != dataset.config_hash:
        problems.append(
            f"checkpoint was written with dataset config hash {metadata.dataset_config_hash}, the current dataset "
            f"config hashes to {dataset.config_hash}"
        )
    stored, expected = metadata.validation_rows, dataset.validation_rows
    mismatches = [
        f"{name!r}: checkpoint {stored.get(name, 'absent')}, now {expected.get(name, 'absent')}"
        for name in sorted(set(stored) | set(expected))
        if stored.get(name) != expected.get(name)
    ]
    if mismatches:
        problems.append(
            "the validation split differs from the checkpoint's (validation rows per source: "
            + "; ".join(mismatches)
            + ")"
        )
    if not problems:
        return
    message = "; ".join(problems)
    if allow_change:
        log.warning("%s; continuing because allow_dataset_change is set", message)
        return
    raise RuntimeError(f"{message}. Set allow_dataset_change: true to resume anyway.")


__all__ = [
    "INSTRUCT_DATA_SIGNATURE",
    "DataEntry",
    "ResolvedDataset",
    "ResolvedStage",
    "build_command",
    "check_dataset_unchanged",
    "check_entries_on_disk",
    "check_entry_shards",
    "check_validation_batches",
    "entry_rows_in_range",
    "loader_shards",
    "processed_rows",
    "resolve_dataset",
    "resolve_entries",
    "resolve_splits",
    "validate_settings",
    "validation_batches_available",
    "validation_rows_of",
]
