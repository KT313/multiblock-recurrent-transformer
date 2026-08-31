# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Planner: what a dataset config needs on disk versus what the manifests say is there, counted in **sequences**.

The trainer draws *rows* from a source with the stage weight and pads or truncates every row to ``block_size``, so a
stage consumes ``stage.tokens × weight ÷ block_size`` rows of a source — its :meth:`DatasetConfig.sequence_budget`.
That is the planner's unit: :func:`rows_needed` turns it into a download target, :func:`plan_downloads` into the
rows still missing per source, :func:`every_source_satisfies_its_budget` / :func:`summarize_dataset_state` decide
whether the processed folders serve the budget (the status table). There is no tokens-per-row estimate anywhere
in this arithmetic any more: a source whose rows are shorter than ``block_size`` is no longer over-downloaded, and
the realised **token** mix of a stage is ``weight × mean_tokens_per_row ÷ block_size``-weighted (the README says
so; ``describe.py`` prints an estimate from ``describe_tokens_per_row`` for the token table only).

Everything here reads manifests only (no parquet footers): broken shards are the repair step's business
(``lib/build/repair.py``) and the training resolver checks the folders on disk independently. Pure functions of
``(config, layout)``; ``lib/build/runner.py`` executes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil

from data_preparation.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.layout import DatasetLayout, processed_columns
from data_preparation.lib.stages.download import current_raw_manifest, raw_manifest_state
from data_preparation.lib.storage.manifest import Manifest

_MARGIN = Fraction(str(SAFETY_MARGIN))  # exact arithmetic: 50 × 1.2 is 60, not 60.000000000000007

# --- rows -----------------------------------------------------------------------------------------------------------


def rows_needed(config: DatasetConfig, name: str) -> int:
    """Raw rows to download for source ``name``.

    A source used for training (and maybe validation): ``ceil(sequence_budget × SAFETY_MARGIN ÷ (1 −
    validation_fraction_of(name)))`` — the margin covers what the length filter and the dedup drop, the division
    keeps the *training* part at the sequence budget after the training resolver holds ``validation_fraction`` of
    the processed rows out. A source used only for validation: its ``rows``. No tokens-per-row estimate is involved
    (see the module docstring): the trainer draws rows, so rows are what is counted.
    """
    source = config.sources[name]
    if not config.used_in_train(name):
        return int(source.rows or 0)
    held_out = Fraction(str(config.validation_fraction_of(name)))
    return ceil(config.sequence_budget(name) * _MARGIN / (1 - held_out))


def rows_sufficient(config: DatasetConfig, name: str) -> int:
    """Processed rows at which a source serves its budget: ``rows_needed ÷ SAFETY_MARGIN`` (= the sequence budget
    over the training share of the rows, or the ``rows`` of a validation-only source, less the download margin)."""
    return ceil(rows_needed(config, name) / _MARGIN)


def training_rows_after_split(config: DatasetConfig, name: str, processed_rows: int) -> int:
    """Rows of ``processed/<name>`` the trainer trains on: all but the first ``ceil(validation_fraction × rows)``
    (the training resolver's split; the fraction is multiplied as the decimal written in the YAML)."""
    held_out = Fraction(str(config.validation_fraction_of(name)))
    return processed_rows - ceil(held_out * processed_rows)


# --- manifests -------------------------------------------------------------------------------------------------------


def tokenizer_is_prepared(config: DatasetConfig, layout: DatasetLayout) -> bool:
    """Whether ``tokenizers/<name>`` carries the current tokenizer manifest and the tokenizer files."""
    directory = layout.tokenizer_dir(config.tokenizer.name)
    manifest = Manifest.load(directory)
    if manifest is None:
        return False
    return manifest.is_current(config.tokenizer_hash()) and (directory / "tokenizer_config.json").is_file()


def raw_is_exhausted(config: DatasetConfig, name: str, raw: Manifest) -> bool:
    """Whether the loader of ``name`` has nothing more to give: the raw manifest says exhausted — unless it was
    exhausted by a ``check_limit`` that has since grown or been removed (``download`` reads on then)."""
    if not raw.extra.get("exhausted"):
        return False
    reached = raw.extra.get("check_limit")
    if reached is None:
        return True
    limit = config.sources[name].check_limit
    return limit is not None and limit <= int(reached)


def current_processed_manifest(config: DatasetConfig, name: str, layout: DatasetLayout) -> Manifest | None:
    """The processed manifest of ``name`` when it carries the current ``processed_hash`` and the current columns."""
    manifest = Manifest.load(layout.processed_dir(name))
    if manifest is None or manifest.stage != "processed" or not manifest.is_current(config.processed_hash(name)):
        return None
    if manifest.extra.get("columns") != list(processed_columns(config.sources[name].kind)):
        return None
    return manifest


def processed_covers_raw(processed: Manifest, raw: Manifest) -> bool:
    """Whether every raw shard has been built into ``processed`` (``extra["input_shards"]`` lists them all)."""
    return processed.extra.get("input_shards") == [[shard.name, shard.rows] for shard in raw.shards]


def build_is_pending(config: DatasetConfig, name: str, layout: DatasetLayout) -> bool:
    """Whether ``name`` has raw shards its processed folder does not cover yet (or no current processed manifest);
    False without a current raw manifest — there is nothing to build from."""
    raw = current_raw_manifest(config, name, layout)
    if raw is None:
        return False
    processed = current_processed_manifest(config, name, layout)
    return processed is None or not processed_covers_raw(processed, raw)


def sources_with_pending_raw_shards(config: DatasetConfig, layout: DatasetLayout, sources: Iterable[str] | None = None) -> list[str]:
    """The sources (all, or ``sources``) whose build has raw shards left to process, in config order."""
    return [name for name in _selected(config, sources) if build_is_pending(config, name, layout)]


# --- the download plan -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDownload:
    """Rows of one source: on disk (raw manifest), needed, and the difference still to fetch."""

    name: str
    rows_present: int  # rows in the raw manifest (0 when missing)
    rows_needed: int  # :func:`rows_needed`
    rows_to_fetch: int  # max(0, needed − present); 0 when the loader is exhausted
    reason: str  # "raw missing" | "rows N < M" | "exhausted" | "enough rows"


@dataclass
class DownloadPlan:
    """One :class:`SourceDownload` per planned source."""

    sources: list[SourceDownload] = field(default_factory=list)

    def to_fetch(self) -> list[SourceDownload]:
        """The sources with rows to fetch."""
        return [source for source in self.sources if source.rows_to_fetch > 0]

    def total_rows_to_fetch(self) -> int:
        return sum(source.rows_to_fetch for source in self.sources)

    def summary(self) -> str:
        """One line: ``"3 source(s) short, downloading 12,000 rows (a 4,000, b 8,000, c 0)"`` or ``"nothing to
        download"``."""
        short = self.to_fetch()
        if not short:
            return "nothing to download"
        per_source = ", ".join(f"{source.name} {source.rows_to_fetch:,}" for source in short)
        return f"{len(short)} source(s) short, downloading {self.total_rows_to_fetch():,} rows ({per_source})"

    def describe(self) -> str:
        """A fixed-width table: source, rows present, rows needed, rows to fetch, reason."""
        header = ("source", "present", "needed", "fetch", "reason")
        rows = [(s.name, f"{s.rows_present:,}", f"{s.rows_needed:,}", f"{s.rows_to_fetch:,}", s.reason) for s in self.sources]
        return format_table(header, rows)


def plan_downloads(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> DownloadPlan:
    """Rows still missing per source (all, or ``sources``) against the raw manifests.

    A raw folder that is stale or outdated is planned as "nothing to fetch" with the state as its reason: the
    repair step deletes it (after confirmation) before any download runs, and a dry run shows the state instead of
    failing — the download never appends to a folder whose rows the current config would not have produced.
    """
    plan = DownloadPlan()
    for name in _selected(config, sources):
        plan.sources.append(_plan_source_download(config, name, layout))
    return plan


def _plan_source_download(config: DatasetConfig, name: str, layout: DatasetLayout) -> SourceDownload:
    state = raw_manifest_state(config, name, layout)
    needed = rows_needed(config, name)
    if state not in ("missing", "current"):
        return SourceDownload(name, 0, needed, 0, f"raw {state}: the repair step deletes it after confirmation")
    raw = current_raw_manifest(config, name, layout)
    if raw is None:
        return SourceDownload(name, 0, needed, needed, "raw missing")
    present = raw.rows()
    if raw_is_exhausted(config, name, raw):
        return SourceDownload(name, present, needed, 0, "exhausted")
    if present >= needed:
        return SourceDownload(name, present, needed, 0, "enough rows")
    return SourceDownload(name, present, needed, needed - present, f"rows {present:,} < {needed:,}")


# --- satisfaction and the status table -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceState:
    """Where one source stands: satisfied when its processed folder serves the budget (see :func:`source_state`)."""

    name: str
    kind: str
    rows_needed: int
    raw_rows: int
    processed_rows: int
    exhausted: bool  # the loader ran dry before `rows_needed` (complete with a warning when built)
    satisfied: bool
    reason: str  # "ok", or what is missing
    epochs: float | None = None  # sequence budget ÷ training rows after the split; only when satisfied and trained on

    def state(self) -> str:
        if not self.satisfied:
            return "incomplete"
        return "exhausted" if self.exhausted else "complete"


@dataclass
class DatasetReport:
    """The status of a whole dataset config: one :class:`SourceState` per source plus the tokenizer."""

    sources: list[SourceState] = field(default_factory=list)
    tokenizer_complete: bool = False
    needs_repair: list[str] = field(default_factory=list)  # sources the repair step would touch (`status` only; `prepare` repaired first)

    @property
    def complete(self) -> bool:
        return self.tokenizer_complete and not self.needs_repair and all(source.satisfied for source in self.sources)

    def missing(self) -> list[str]:
        """Names of the items that are not satisfied or need a repair: the sources, plus ``"tokenizer"`` when it
        is missing (empty iff :attr:`complete`)."""
        names = [source.name for source in self.sources if not source.satisfied or source.name in self.needs_repair]
        if not self.tokenizer_complete:
            names.append("tokenizer")
        return names

    def unsatisfied(self) -> list[SourceState]:
        """The sources that do not serve their budget yet (the runner names them after its rounds)."""
        return [source for source in self.sources if not source.satisfied]

    def table(self) -> str:
        """A fixed-width table: source, kind, rows needed, raw rows, processed rows, epochs, state, reason."""
        header = ("source", "kind", "needed", "raw", "processed", "epochs", "state", "reason")
        rows = [
            (
                s.name, s.kind, f"{s.rows_needed:,}", f"{s.raw_rows:,}", f"{s.processed_rows:,}",
                "-" if s.epochs is None else f"{s.epochs:.2f}", "needs repair" if s.name in self.needs_repair else s.state(), s.reason,
            )
            for s in self.sources
        ]  # fmt: skip
        rows.append(("tokenizer", "tokenizer", "", "", "", "", "complete" if self.tokenizer_complete else "incomplete", ""))
        return format_table(header, rows)

    def describe(self) -> str:
        """The table plus the overall verdict line (``dataset complete`` / ``dataset INCOMPLETE``)."""
        return self.table() + "\n" + f"dataset {'complete' if self.complete else 'INCOMPLETE'}"


def source_state(config: DatasetConfig, name: str, layout: DatasetLayout) -> SourceState:
    """Satisfied = the processed manifest is current, covers every raw shard and holds at least
    :func:`rows_sufficient` rows (so the training part after the split reaches the sequence budget) — or the raw
    folder is exhausted and every raw shard is built. A stale / outdated raw folder is reported (the repair step
    deletes it after confirmation), never counted."""
    kind = config.sources[name].kind
    needed = rows_needed(config, name)
    state = raw_manifest_state(config, name, layout)
    if state not in ("missing", "current"):
        return SourceState(name, kind, needed, 0, 0, False, False, f"raw {state}: the repair step deletes it after confirmation")
    raw = current_raw_manifest(config, name, layout)
    if raw is None:
        return SourceState(name, kind, needed, 0, 0, False, False, "raw missing")
    exhausted = raw_is_exhausted(config, name, raw)
    processed = current_processed_manifest(config, name, layout)
    if processed is None:
        why = "processed missing" if Manifest.load(layout.processed_dir(name)) is None else "processed stale"
        return SourceState(name, kind, needed, raw.rows(), 0, exhausted, False, why)
    processed_rows = processed.rows()
    if not processed_covers_raw(processed, raw):
        return SourceState(name, kind, needed, raw.rows(), processed_rows, exhausted, False, "processed behind raw")
    sufficient = rows_sufficient(config, name)
    if processed_rows >= sufficient:
        reason = "ok"
    elif exhausted:
        reason = f"exhausted at {processed_rows:,} of {sufficient:,} rows"
    else:
        return SourceState(name, kind, needed, raw.rows(), processed_rows, exhausted, False, f"processed rows {processed_rows:,} < {sufficient:,}")
    return SourceState(name, kind, needed, raw.rows(), processed_rows, exhausted, True, reason, _epochs(config, name, processed_rows))


def _epochs(config: DatasetConfig, name: str, processed_rows: int) -> float | None:
    """How often the trainer cycles the training rows of a satisfied source to serve its sequence budget (the
    largest single-stage demand); None for a source it does not train on or without training rows."""
    budget = config.sequence_budget(name)
    training_rows = training_rows_after_split(config, name, processed_rows)
    if budget <= 0 or training_rows <= 0:
        return None
    return budget / training_rows


def every_source_satisfies_its_budget(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> bool:
    """Whether every source (all, or ``sources``) is satisfied (:func:`source_state`)."""
    return all(source_state(config, name, layout).satisfied for name in _selected(config, sources))


def summarize_dataset_state(config: DatasetConfig, layout: DatasetLayout, *, needs_repair: Iterable[str] = ()) -> DatasetReport:
    """The :class:`DatasetReport` of every source of ``config`` under ``layout`` plus the tokenizer.
    ``needs_repair`` names the sources a repair dry run would touch (``status``): they count as incomplete."""
    return DatasetReport(
        sources=[source_state(config, name, layout) for name in config.sources],
        tokenizer_complete=tokenizer_is_prepared(config, layout),
        needs_repair=sorted(set(needs_repair)),
    )


# --- helpers ---------------------------------------------------------------------------------------------------------


def _selected(config: DatasetConfig, sources: Iterable[str] | None) -> list[str]:
    """``sources`` in config order (every source when None); unknown names are an error."""
    if sources is None:
        return list(config.sources)
    wanted = set(sources)
    unknown = wanted - set(config.sources)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}")
    return [name for name in config.sources if name in wanted]


def format_table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """Left-aligned columns, two spaces apart, each as wide as its widest cell (header included)."""
    widths = [max(len(header[column]), *(len(row[column]) for row in rows)) for column in range(len(header))]
    lines = [_table_line(header, widths)]
    lines.extend(_table_line(row, widths) for row in rows)
    return "\n".join(lines)


def _table_line(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()


__all__ = [
    "DatasetReport",
    "DownloadPlan",
    "SourceDownload",
    "SourceState",
    "build_is_pending",
    "current_processed_manifest",
    "every_source_satisfies_its_budget",
    "format_table",
    "plan_downloads",
    "processed_covers_raw",
    "raw_is_exhausted",
    "rows_needed",
    "rows_sufficient",
    "source_state",
    "sources_with_pending_raw_shards",
    "summarize_dataset_state",
    "tokenizer_is_prepared",
    "training_rows_after_split",
]
