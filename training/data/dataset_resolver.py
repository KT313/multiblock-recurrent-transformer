# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Turn a run's `dataset_config` into what the training loop consumes: one verified (or auto-prepared) data
directory per TRAIN SOURCE with its row range (read once, continuously, for the whole run), the per-stage
validation entries and sampling weights, the tokenizer path and the stage token budgets for `StageManager`.

The validation split is decided here, once per source and run, never by the data pipeline: a source used only for
training is read whole, a source used only for validation is read whole as validation, and a source used for both
holds out its first `ceil(validation_fraction_of(source) × rows)` processed rows (`validation_rows`) for validation
and trains on the rest. Rows are counted ONCE per source from the parquet footers of `processed/<source>` and
cross-checked against the manifest (`processed_row_counts`, kept as `ResolvedDataset.rows_on_disk`); the same
source gets the same split in every stage. The chosen `validation_rows` per source travel with every checkpoint
next to `dataset_config_hash` and are verified on resume (`check_dataset_unchanged`).
The split is also checked against what the loaders and evaluation need: a stage whose validation loader cannot fill
one micro-batch (`check_validation_batches`), a train source or validation entry whose range is empty
(`check_entry_rows` — what makes the stream's restart-on-exhaustion safe) and an entry with fewer rows than its
loader has worker shards (`check_entry_shards`) fail here, at setup (`check_entries`, one pass over every entry),
instead of at the first evaluation step or mid-training in a worker.

Framework-neutral apart from `data_preparation.*` (manifests, parquet footers); no torch. The only cross-over
between the run config and the dataset config happens here.
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
from data_preparation.lib.ui.dashboard import BUILD_LOG_NAME, Dashboard
from training.settings import Settings

if TYPE_CHECKING:  # annotation only: this module stays torch-free, `training.checkpoint` imports torch
    from training.checkpoint import CheckpointMetadata

log = get_logger(__name__)

INSTRUCT_DATA_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}

Part = Literal["train", "val"]


TRAIN_LOADER_NUM_WORKERS = 1  # every per-source train loader runs one worker process; fixed, not a setting (the
# old per-stage-mixture loaders had a worker-count knob; per-source readers make it dead)


@dataclass
class DataEntry:
    """One parquet dataset directory with the row range its loader reads: a run-wide train source
    (`ResolvedDataset.train_sources`, prefix = the source name) or a member of a stage's validation mixture
    (`ResolvedStage.val_data`, prefix = `<stage>-<source>`)."""

    prefix: str  # unique name within its list, used for logging (train: the data_id of `data_composition/...`)
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
    """One training stage: its token budget, base LR and transition length (what `training.stage_manager` turns into
    step boundaries), its sampling weights over the run-wide train sources (`ResolvedDataset.train_sources`) and its
    validation entries resolved on disk. The stage structure changes the WEIGHTS only — the train readers themselves
    run once per source for the whole run."""

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
    train_sources: list[DataEntry]  # one entry per source any stage trains on, in dataset-config order; each is
    # read by ONE loader for the whole run (rows validation_rows -> end), so stages sharing a source never re-read
    validation_rows: dict[str, int]  # per source: rows [0, n) of processed/<source> are validation, the rest training
    rows_on_disk: dict[str, int]  # per processed directory (`DataEntry.data_dir`): its rows, counted once at setup


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


def processed_row_counts(dataset_config: DatasetConfig, layout: DatasetLayout) -> dict[str, int]:
    """`{processed directory: rows}` for every source of the config (`processed_rows`: footers cross-checked
    against the manifest), counted once per run; the split and every entry check read this mapping."""
    return {
        str(layout.processed_dir(name)): processed_rows(
            layout.processed_dir(name), f"source {name!r} (stage keys {_stage_keys(dataset_config, name)})"
        )
        for name in dataset_config.sources
    }


def resolve_splits(dataset_config: DatasetConfig, layout: DatasetLayout, rows_on_disk: Mapping[str, int]) -> dict[str, int]:
    """`{source: validation_rows}` for every source of the config, from its rows in `rows_on_disk`
    (`processed_row_counts`). Decided once per run; every stage uses the same split."""
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
    """The `DataEntry` for one source name `key`: its `processed/<source>` folder, read through the text column
    (pretrain) or the instruction/input/output signature (instruct), restricted to the validation rows
    `[0, validation_rows)` (part `val`) or the training rows from `validation_rows` on (part `train`)."""
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
    """One `DataEntry` per source any stage trains on, in dataset-config order — the deterministic order the
    stream draws over and the loaders are built in.

    Each entry is read by ONE loader, continuously, for the whole run (the stage structure only changes sampling
    weights), so consecutive stages sharing a source never re-read its rows. The prefix is the plain source name:
    the key of the stream's consumed-row counters (`data_stream` in a checkpoint) and of `data_composition/...`
    logging. The range starts after the validation holdout (`validation_rows[name]`, see `resolve_splits`) and
    runs to the end of the folder. Pure path arithmetic, no I/O.
    """
    return [
        _data_entry(dataset_config, layout, name, name, 1.0, "train", validation_rows[name])
        for name in dataset_config.sources
        if dataset_config.used_in_train(name)
    ]


def resolve_val_entries(
    dataset_config: DatasetConfig, layout: DatasetLayout, stage: StageConfig, validation_rows: Mapping[str, int]
) -> list[DataEntry]:
    """The `val_data` of one dataset-config stage: every `stage.val` key with its weight, reading the held-out
    validation rows `[0, validation_rows[name])` of `processed/<name>`. Prefixes are `<stage>-<key>` and unique
    per stage. Pure path arithmetic, no I/O."""
    entries = [
        _data_entry(dataset_config, layout, key, f"{stage.name}-{key}", weight, "val", validation_rows[key])
        for key, weight in stage.val.items()
    ]
    prefixes = [entry.prefix for entry in entries]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError(f"stage {stage.name}: duplicate data entry prefixes in {prefixes}")
    return entries


def entry_rows_in_range(entry: DataEntry, total_rows: int) -> int:
    """Rows a loader actually reads from `entry`: its range `[skip_rows, skip_rows + max_rows)` clipped to the
    `total_rows` on disk, exactly as `ParquetTextDataset` clips it."""
    start = min(entry.skip_rows, total_rows)
    stop = total_rows if entry.max_rows is None else min(total_rows, start + entry.max_rows)
    return stop - start


def _labeled_entries(train_sources: list[DataEntry], stages: list[ResolvedStage]) -> Iterator[tuple[str, Part, DataEntry]]:
    """Every data entry of a run with its error label: the run-wide train sources, then each stage's validation
    entries."""
    for entry in train_sources:
        yield f"train source {entry.prefix!r}", "train", entry
    for stage in stages:
        for entry in stage.val_data:
            yield f"stage {stage.name!r} val entry {entry.prefix!r}", "val", entry


def check_entries(
    train_sources: list[DataEntry], stages: list[ResolvedStage], rows_on_disk: Mapping[str, int], world_size: int = 1
) -> None:
    """One pass over every data entry of the run (the train sources, then each stage's validation entries) against
    the rows on disk (`processed_row_counts`): its range holds at least one row (`check_entry_rows`) and at least
    one row per loader shard (`check_entry_shards`). Errors name the entry (a train source, or a stage's
    `<stage>-<source>` validation entry) and the folder."""
    for what, part, entry in _labeled_entries(train_sources, stages):
        total = rows_on_disk[entry.data_dir]
        check_entry_rows(what, part, entry, total)
        check_entry_shards(what, part, entry, total, world_size)


def check_entry_rows(what: str, part: Part, entry: DataEntry, total: int) -> None:
    """The entry's row range (clipped to the `total` rows on disk as `ParquetTextDataset` clips it) contains at
    least one row.

    The at-least-one-row guarantee for the train sources is what makes the stream's restart-on-exhaustion safe:
    a source that runs dry mid-run is restarted (`RunDataloaders.next_train_batch`), which would spin forever on
    an empty range — impossible after this check."""
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


def check_entry_shards(what: str, part: Part, entry: DataEntry, total: int, world_size: int) -> None:
    """Fail at setup when a data entry has fewer rows than its loader has shards.

    `ParquetTextDataset` deals the rows of an entry's range round-robin over `world_size × num_workers` shards, so
    a shard is empty as soon as the range holds fewer rows than there are shards — fatal mid-run for a train
    source: the restart of its exhausted loader immediately gets a second `StopIteration` from the empty shard,
    which Python turns into a `RuntimeError` and kills the training run; for a validation entry the rank with the
    empty shard would score different data than the others. Train loaders run one worker per source
    (`TRAIN_LOADER_NUM_WORKERS`) and validation loaders in-process, so both have `world_size` shards:
    with one device this reduces to the at-least-one-row guarantee `check_entry_rows` already gives, and only a
    larger world can starve a shard. The error names the entry, its folder, its row count and the shard count.
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
    entries: list[DataEntry], rows_on_disk: Mapping[str, int], micro_batch_size: int, world_size: int
) -> int:
    """How many micro-batches one evaluation can draw from a stage's validation loader.

    One `__iter__` of the loader `training.data.loader.build_dataloader` builds is one pass over every entry's row
    range — a single `ParquetTextDataset`, or the `WeightedMixtureDataset` of several, which yields every member's
    rows once — and then stops: `ceil(rows / micro_batch_size)` batches, the last one short (`drop_last` is off).
    Rows are dealt round-robin over `world_size × num_workers` shards (`ParquetTextDataset`); validation loaders run
    with `num_workers=0`, so each rank reads every `world_size`-th row of every entry and the smallest shard holds
    `rows // world_size` of them. The row range is clipped to the rows on disk (`rows_on_disk`, keyed by directory)
    exactly as the dataset clips it.
    """
    rows_per_rank = sum(entry_rows_in_range(entry, rows_on_disk[entry.data_dir]) // world_size for entry in entries)
    return -(-rows_per_rank // micro_batch_size)  # ceil, in integers


def check_validation_batches(
    stages: list[ResolvedStage],
    rows_on_disk: Mapping[str, int],
    micro_batch_size: int,
    eval_iters: int,
    world_size: int = 1,
) -> None:
    """Fail (or warn) at setup time about a validation split that cannot feed `training.evaluation.evaluate`.

    A stage whose validation loader delivers no batch at all is a hard error naming the stage, its validation
    entries and both numbers — `evaluate` would otherwise raise in the middle of the run, at the first evaluation
    step. Fewer than `eval_iters` batches is only a warning: the loader still hands out a last, short batch, and
    `evaluate` averages the batches it actually receives, so the reported loss stays correct — it is just measured
    on less data than the config asks for.
    """
    for stage in stages:
        available = validation_batches_available(stage.val_data, rows_on_disk, micro_batch_size, world_size)
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
    check_validation_batches(stages, rows_on_disk, settings.micro_batch_size, settings.eval_iters, world_size)
    return ResolvedDataset(
        config=dataset_config,
        config_hash=dataset_config.config_hash(),
        tokenizer_dir=str(layout.tokenizer_dir(dataset_config.tokenizer.name)),
        stages=stages,
        train_sources=train_sources,
        validation_rows=validation_rows,
        rows_on_disk=rows_on_disk,
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
