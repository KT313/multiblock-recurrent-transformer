# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Turn a run's `dataset_config` into what the training loop consumes: one verified (or auto-prepared) data
directory per TRAIN SOURCE with its row range, the per-stage validation entries and weights, the tokenizer path
and the stage token budgets.

The validation split is decided here, once per source and run: a source used only for training is read whole, one
used only for validation is read whole as validation, one used for both holds out its first
`ceil(validation_fraction × rows)` processed rows. Rows are counted once per source from the parquet footers and
cross-checked against the manifest. The chosen `validation_rows` and the row count per source (`source_rows`)
travel with every checkpoint and are verified on resume (`check_dataset_unchanged`): the stream resumes by row
offset, so a source re-prepared in between would continue from other rows. Every entry is checked at setup
(`check_entries`, `check_validation_batches`) instead of mid-run.

Framework-neutral apart from `data_preparation.*`; no torch. The only cross-over between the run config and the
dataset config happens here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

from data_preparation.lib.abort import StopCheck
from data_preparation.lib.build.planner import summarize_dataset_state
from data_preparation.lib.build.repair import ConfirmationRequired
from data_preparation.lib.build.runner import prepare, status
from data_preparation.dataset_config import DatasetConfig, StageConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.log import ROOT_LOGGER_NAME, get_logger
from data_preparation.lib.storage.manifest import MANIFEST_NAME, Manifest, shard_rows
from data_preparation.lib.storage.parquet import SHARD_PATTERN
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, DataDashboard
from training.settings import Settings

if TYPE_CHECKING:  # annotation only: this module stays torch-free, `training.checkpoint` imports torch
    from training.checkpoint import CheckpointMetadata

log = get_logger(__name__)

INSTRUCT_DATA_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}

Part = Literal["train", "val"]


TRAIN_LOADER_NUM_WORKERS = 1  # every per-source train loader runs one worker process; fixed, not a setting (its worker
# batch size and prefetch depth are `training.data.loader.TRAIN_LOADER_BATCH_ROWS` / `TRAIN_LOADER_PREFETCH_FACTOR`)


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


class _BuildBackend(Protocol):
    """
    The `training.backend.Backend` members the resolver needs (kept as a Protocol to stay torch-free):
    `is_main` / `barrier` for the build, `world_size` for the validation-batch check.
    """

    is_main: bool
    world_size: int

    def barrier(self) -> None: ...


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


def build_command(dataset_config: str, dataset_dir: str) -> str:
    """
    The CLI command that materialises the dataset config (quoted in error messages).
    """

    return f"python data_preparation/prepare.py prepare --dataset_config {dataset_config} --dataset_dir {dataset_dir}"


# --- the validation split ---------------------------------------------------------------------------------------------


def validation_rows_of(dataset_config: DatasetConfig, source_name: str, total_rows: int) -> int:
    """
    How many of the first rows of `processed/<source>` (`total_rows` in all) are validation rows.

    Every row of a source used only for validation, none of a source used only for training (its
    `validation_fraction_of` is 0), `ceil(validation_fraction_of(source) × total_rows)` of a source used for both.
    The fraction is multiplied as the decimal written in the YAML (`Fraction(str(...))`): a float product such as
    `0.07 × 100 = 7.000000000000001` would otherwise round up one row too many.
    """

    if not dataset_config.used_in_train(source_name):
        return total_rows
    return ceil(Fraction(str(dataset_config.validation_fraction_of(source_name))) * total_rows)


def _rows_on_disk(directory: Path, what: str) -> int:
    """
    Rows of the `data-*.parquet` shards of `directory` from their parquet footers (no manifest involved); the
    directory must exist and hold at least one shard. `what` names the item checked in the errors.
    """

    if not directory.is_dir():
        raise FileNotFoundError(f"{what}: processed folder {directory} does not exist")
    shards = sorted(path for path in directory.glob("*.parquet") if SHARD_PATTERN.match(path.name))  # what `ParquetTextDataset` reads
    if not shards:
        raise FileNotFoundError(f"{what}: processed folder {directory} holds no data-*.parquet shard")
    return sum(shard_rows(shard) for shard in shards)


def processed_rows(directory: Path, what: str) -> int:
    """
    Rows of a processed folder: counted from the parquet footers and cross-checked against its manifest (a
    listed shard with the wrong count or an unlisted shard on disk is an error naming the folder).
    """

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
    """
    `pretrain_a.train, pretrain_a.val, ...`: where a source is used (for error messages).
    """

    keys = [f"{stage.name}.train" for stage in dataset_config.stages if source_name in stage.train]
    keys += [f"{stage.name}.val" for stage in dataset_config.stages if source_name in stage.val]
    return ", ".join(keys)


def processed_row_counts(dataset_config: DatasetConfig, layout: DatasetLayout) -> dict[str, int]:
    """
    `{processed directory: rows}` for every source of the config (`processed_rows`: footers cross-checked
    against the manifest), counted once per run; the split and every entry check read this mapping.
    """

    return {
        str(layout.processed_dir(name)): processed_rows(
            layout.processed_dir(name), f"source {name!r} (stage keys {_stage_keys(dataset_config, name)})"
        )
        for name in dataset_config.sources
    }


def resolve_splits(dataset_config: DatasetConfig, layout: DatasetLayout, rows_on_disk: Mapping[str, int]) -> dict[str, int]:
    """
    `{source: validation_rows}` for every source of the config, from its rows in `rows_on_disk`
    (`processed_row_counts`). Decided once per run; every stage uses the same split.
    """

    splits: dict[str, int] = {}
    for name in dataset_config.sources:
        total = rows_on_disk[str(layout.processed_dir(name))]
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


# --- sources and stage keys -> data entries ---------------------------------------------------------------------------


def _data_entry(
    dataset_config: DatasetConfig, layout: DatasetLayout, key: str, prefix: str, weight: float, part: Part, validation_rows: int
) -> DataEntry:
    """
    The `DataEntry` for one source name `key`: its `processed/<source>` folder, read through the text column
    (pretrain) or the instruction/input/output signature (instruct), restricted to the validation rows
    `[0, validation_rows)` (part `val`) or the training rows from `validation_rows` on (part `train`).
    """

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


def resolve_train_sources(
    dataset_config: DatasetConfig, layout: DatasetLayout, validation_rows: Mapping[str, int]
) -> list[DataEntry]:
    """
    One `DataEntry` per source any stage trains on, in dataset-config order (the order the stream draws over).

    Each entry is read by ONE loader for the whole run. The prefix is the plain source name, the key of the stream's
    consumed-row counters and of `data_composition/...` logging. The range starts after the validation holdout.
    Pure path arithmetic, no I/O.
    """

    return [
        _data_entry(dataset_config, layout, name, name, 1.0, "train", validation_rows[name])
        for name in dataset_config.sources
        if dataset_config.used_in_train(name)
    ]


def resolve_val_entries(
    dataset_config: DatasetConfig, layout: DatasetLayout, stage: StageConfig, validation_rows: Mapping[str, int]
) -> list[DataEntry]:
    """
    The `val_data` of one dataset-config stage: every `stage.val` key with its weight, reading the held-out
    validation rows `[0, validation_rows[name])` of `processed/<name>`. Prefixes are `<stage>-<key>` and unique
    per stage. Pure path arithmetic, no I/O.
    """

    entries = [
        _data_entry(dataset_config, layout, key, f"{stage.name}-{key}", weight, "val", validation_rows[key])
        for key, weight in stage.val.items()
    ]
    prefixes = [entry.prefix for entry in entries]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"stage {stage.name}: duplicate data entry prefixes in {prefixes}")
    return entries


def entry_rows_in_range(entry: DataEntry, total_rows: int) -> int:
    """
    Rows a loader actually reads from `entry`: its range `[skip_rows, skip_rows + max_rows)` clipped to the
    `total_rows` on disk, exactly as `ParquetTextDataset` clips it.
    """

    start = min(entry.skip_rows, total_rows)
    stop = total_rows if entry.max_rows is None else min(total_rows, start + entry.max_rows)
    return stop - start


def _labeled_entries(train_sources: list[DataEntry], stages: list[ResolvedStage]) -> Iterator[tuple[str, Part, DataEntry]]:
    """
    Every data entry of a run with its error label: the run-wide train sources, then each stage's validation
    entries.
    """

    for entry in train_sources:
        yield f"train source {entry.prefix!r}", "train", entry
    for stage in stages:
        for entry in stage.val_data:
            yield f"stage {stage.name!r} val entry {entry.prefix!r}", "val", entry


def check_entries(
    train_sources: list[DataEntry], stages: list[ResolvedStage], rows_on_disk: Mapping[str, int], world_size: int = 1
) -> None:
    """
    One pass over every data entry of the run (the train sources, then each stage's validation entries) against
    the rows on disk (`processed_row_counts`): its range holds at least one row (`check_entry_rows`) and at least
    one row per loader shard (`check_entry_shards`). Errors name the entry (a train source, or a stage's
    `<stage>-<source>` validation entry) and the folder.
    """

    for what, part, entry in _labeled_entries(train_sources, stages):
        total = rows_on_disk[entry.data_dir]
        check_entry_rows(what, part, entry, total)
        # the train sources are read by the main rank alone (one shard whatever the world size); the validation
        # entries are dealt over the ranks
        check_entry_shards(what, part, entry, total, 1 if part == "train" else world_size)


def check_entry_rows(what: str, part: Part, entry: DataEntry, total: int) -> None:
    """
    The entry's row range (clipped to the `total` rows on disk as `ParquetTextDataset` clips it) holds at least
    one row. That guarantees rows ON DISK, not usable samples: a row the collate drops for lack of a supervised
    label is read and yields nothing. `RunDataloaders.next_train_batch` checks the samples per epoch and refuses
    to restart a train source whose full epoch produced none, which is what keeps the restart from spinning.
    """

    if entry_rows_in_range(entry, total) <= 0:
        end = "end" if entry.max_rows is None else str(entry.skip_rows + entry.max_rows)
        raise RuntimeError(
            f"{what}: row range [{entry.skip_rows}, {end}) of {entry.data_dir} is empty ({total} rows on "
            f"disk); the {'validation' if part == 'val' else 'training'} part of the split has no row"
        )


def loader_shards(num_workers: int, world_size: int) -> int:
    """
    Shards a loader's datasets deal their rows over: one per dataloader worker and rank
    (`ParquetTextDataset._shard`). `num_workers=0` loads in the calling process, which is ONE shard per rank. The
    train loaders live on the main rank only, so `check_entries` passes them a world size of 1.
    """

    return world_size * max(num_workers, 1)


def check_entry_shards(what: str, part: Part, entry: DataEntry, total: int, world_size: int) -> None:
    """
    Fail at setup when a data entry has fewer rows than its loader has shards.

    `ParquetTextDataset` deals rows round-robin over `world_size × num_workers` shards; an empty shard kills a train
    loader's restart mid-run and makes ranks score different validation data. Train loaders run one worker per
    source on the main rank (one shard: the caller passes world size 1) and validation loaders in-process on every
    rank (`world_size` shards): only a larger world can starve a validation entry.
    """

    num_workers = TRAIN_LOADER_NUM_WORKERS if part == "train" else 0
    shards = loader_shards(num_workers, world_size)
    if shards <= 1:  # a single shard reads the whole range; nothing to split
        return
    rows = entry_rows_in_range(entry, total)
    if rows >= shards:
        return
    raise ValueError(
        f"{what}: {entry.data_dir} gives it {rows} row(s) after the validation split, but its loader "
        f"deals the rows round-robin over {shards} shards ({num_workers} dataloader worker(s) × world size "
        f"{world_size}), so {shards - rows} shard(s) would be empty and the run would fail during training. "
        "Give the source more rows, or lower the world size"
    )


def validation_batches_available(
    entries: list[DataEntry], rows_on_disk: Mapping[str, int], validation_batch_size: int, world_size: int
) -> int:
    """
    How many batches one evaluation can draw from a stage's validation loader.

    One `__iter__` of the loader is one pass over every entry's row range, then it stops: `ceil(rows /
    validation_batch_size)` batches, the last one short. Validation loaders run with `num_workers=0`, so each rank
    reads every `world_size`-th row and the smallest shard holds `rows // world_size` of them.
    """

    rows_per_rank = sum(entry_rows_in_range(entry, rows_on_disk[entry.data_dir]) // world_size for entry in entries)
    return -(-rows_per_rank // validation_batch_size)  # ceil, in integers


def check_validation_batches(
    stages: list[ResolvedStage],
    rows_on_disk: Mapping[str, int],
    validation_batch_size: int,
    eval_iters: int,
    world_size: int = 1,
) -> None:
    """
    Fail (or warn) at setup about a validation split that cannot feed `training.evaluation.evaluate`.

    `eval_iters` is the count PER RANK (`Settings.eval_iters_per_rank`), like the batches available per rank. No
    batch at all is an error naming the stage and its entries; fewer than `eval_iters` batches is a warning, since
    `evaluate` averages the batches it gets.
    """

    for stage in stages:
        available = validation_batches_available(stage.val_data, rows_on_disk, validation_batch_size, world_size)
        entries = ", ".join(entry.prefix for entry in stage.val_data)
        per_rank = f" per rank (world size {world_size})" if world_size > 1 else ""
        if available == 0:
            raise RuntimeError(
                f"stage {stage.name!r}: its validation data ({entries}) yields 0 batches of {validation_batch_size} "
                f"rows{per_rank} but eval_iters is {eval_iters}, so evaluation would have nothing to score. Raise "
                "validation_fraction for the source in the dataset config, give the stage a larger validation "
                "source, or lower validation_batch_size"
            )
        if available < eval_iters:
            log.warning(
                "stage %s: its validation data (%s) yields %d batch(es) of %d rows%s, fewer than eval_iters "
                "(%d); every evaluation of this stage averages the %d batch(es) it gets",
                stage.name,
                entries,
                available,
                validation_batch_size,
                per_rank,
                eval_iters,
                available,
            )


# --- run config <-> dataset config ----------------------------------------------------------------------------------


def validate_settings(settings: Settings, dataset_config: DatasetConfig) -> None:
    """
    The cross-checks between run config and dataset config: one base LR per stage, and a run no longer than the
    rows were cut at (`training/run.py::check_sequence_lengths` checks all three lengths once the model config is
    known; this one runs before any data is touched).
    """

    if settings.training_max_sequence_length > dataset_config.dataset_max_sequence_length:
        raise ValueError(
            f"training_max_sequence_length ({settings.training_max_sequence_length}) exceeds dataset_max_sequence_length "
            f"({dataset_config.dataset_max_sequence_length}) of {Path(settings.dataset_config).as_posix()}: the rows are cut shorter than the run trains"
        )
    if len(settings.stage_base_lrs) != len(dataset_config.stages):
        raise ValueError(
            f"stage_base_lrs has {len(settings.stage_base_lrs)} entries but dataset config "
            f"{settings.dataset_config!r} has {len(dataset_config.stages)} stages "
            f"{[stage.name for stage in dataset_config.stages]}; give one base LR per stage, in order"
        )


def _ensure_prepared(
    settings: Settings,
    dataset_config: DatasetConfig,
    layout: DatasetLayout,
    backend: Optional[_BuildBackend],
    should_stop: StopCheck | None = None,
) -> None:
    """
    Verify the dataset on disk; prepare what is missing when `auto_prepare` allows it, else raise.

    Auto-prepare never confirms a repair and never prompts: `prepare` runs with a `confirm` that always declines, so
    a folder needing a repair fails the run with the `prepare.py prepare ... --yes` command that confirms it.
    `should_stop` is polled between shards (`BuildAborted`, everything published so far kept).
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

    log.info("dataset %s is incomplete, preparing missing data (%s)", settings.dataset_config, missing)
    if backend is None or backend.is_main:
        with DataDashboard() as dashboard, dashboard.attach(logging.getLogger(ROOT_LOGGER_NAME), log_file=layout.root / BUILD_LOG_NAME):
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
            except ConfirmationRequired as error:
                raise RuntimeError(
                    f"{error.message.rstrip()}\nauto-prepare never confirms a repair; to confirm it run:\n  "
                    + build_command(settings.dataset_config, settings.dataset_dir)
                    + " --yes"
                ) from error
    if backend is not None:
        backend.barrier()

    report = summarize_dataset_state(dataset_config, layout)  # `prepare` already logged the final status table; re-verify silently
    if not report.complete:
        raise RuntimeError(
            f"dataset config {settings.dataset_config!r} is still incomplete after preparing: "
            f"{', '.join(report.missing())}\n{report.describe()}"
        )


def resolve_dataset(
    settings: Settings, backend: Optional[_BuildBackend] = None, *, should_stop: StopCheck | None = None
) -> ResolvedDataset:
    """
    Load, verify and (with `auto_prepare`) build the dataset of a run, then decide the validation split.

    The build runs on the main rank only, followed by `backend.barrier()`. Raises `RuntimeError` for missing data
    that cannot be prepared here, a folder that does not match its manifest, an empty part of the split or a
    validation loader that cannot fill one micro-batch; `ValueError` for an entry with fewer rows than loader shards.
    """

    dataset_config = load_dataset_config(settings.dataset_config)
    validate_settings(settings, dataset_config)
    layout = DatasetLayout(Path(settings.dataset_dir))
    _ensure_prepared(settings, dataset_config, layout, backend, should_stop)
    rows_on_disk = processed_row_counts(dataset_config, layout)
    validation_rows = resolve_splits(dataset_config, layout, rows_on_disk)

    train_sources = resolve_train_sources(dataset_config, layout, validation_rows)
    stages: list[ResolvedStage] = []
    for stage, base_lr in zip(dataset_config.stages, settings.stage_base_lrs):
        stages.append(
            ResolvedStage(
                name=stage.name,
                tokens=stage.tokens,
                base_lr=base_lr,
                transition_pct=stage.transition_pct,
                train_weights=dict(stage.train),
                val_data=resolve_val_entries(dataset_config, layout, stage, validation_rows),
            )
        )
    world_size = 1 if backend is None else backend.world_size
    check_entries(train_sources, stages, rows_on_disk, world_size)
    check_validation_batches(
        stages, rows_on_disk, settings.validation_batch_size, settings.eval_iters_per_rank(world_size), world_size
    )
    return ResolvedDataset(
        config=dataset_config,
        config_hash=dataset_config.config_hash(),
        tokenizer_dir=str(layout.tokenizer_dir(dataset_config.tokenizer.name)),
        stages=stages,
        train_sources=train_sources,
        validation_rows=validation_rows,
        source_rows={name: rows_on_disk[str(layout.processed_dir(name))] for name in dataset_config.sources},
        rows_on_disk=rows_on_disk,
    )


# --- resume checks ---------------------------------------------------------------------------------------------------


def check_dataset_unchanged(metadata: CheckpointMetadata, dataset: ResolvedDataset, allow_change: bool) -> None:
    """
    Verify that a checkpoint was written against the dataset the run now resolves to.

    Compared: the dataset-config hash, the rows per source (`{source: rows}`: the stream resumes by row offset
    into each source, so a source re-prepared to another row count would continue from other rows) and the
    validation split (`{source: validation_rows}`: a resumed run would otherwise validate on rows it trained on).
    A mismatch raises `RuntimeError` naming every difference, unless `allow_change` (`allow_dataset_change`), which
    only warns.
    """

    problems: list[str] = []
    if metadata.dataset_config_hash != dataset.config_hash:
        problems.append(
            f"checkpoint was written with dataset config hash {metadata.dataset_config_hash}, the current dataset "
            f"config hashes to {dataset.config_hash}"
        )
    for what, stored, expected in (
        ("the rows per source differ from the checkpoint's (processed rows per source: ", metadata.source_rows, dataset.source_rows),
        ("the validation split differs from the checkpoint's (validation rows per source: ", metadata.validation_rows, dataset.validation_rows),
    ):
        mismatches = [
            f"{name!r}: checkpoint {stored.get(name, 'absent')}, now {expected.get(name, 'absent')}"
            for name in sorted(set(stored) | set(expected))
            if stored.get(name) != expected.get(name)
        ]
        if mismatches:
            problems.append(what + "; ".join(mismatches) + ")")
    if not problems:
        return
    message = "; ".join(problems)
    if allow_change:
        log.warning("%s; continuing because allow_dataset_change is set", message)
        return
    raise RuntimeError(f"{message}. Set allow_dataset_change: true to resume anyway.")
