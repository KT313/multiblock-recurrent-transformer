# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Planner: what a dataset config needs on disk versus what the manifests say is there, counted in **sequences**.

The trainer draws *rows* from one continuous stream per source with the stage weight (linearly interpolated across
a transition window) and pads or truncates every row to ``block_size``, so the run consumes the integral of the
source's weight schedule over the stage token budgets, ÷ ``block_size`` — its :meth:`DatasetConfig.sequence_budget`
(each stage adds ``(tokens − transition tokens) × weight`` plus the trapezoid ``transition tokens × (weight +
next stage's weight) / 2``; the 1.2 safety margin comes on top).
That is the planner's unit: :func:`rows_needed` turns it into a download target and :func:`rows_sufficient` into the
processed rows that serve it. There is no tokens-per-row estimate anywhere in this arithmetic: a source whose rows
are shorter than ``block_size`` is no longer over-downloaded, and the realised **token** mix of a stage is
``weight × mean_tokens_per_row ÷ block_size``-weighted (the README says so; ``describe.py`` prints an estimate from
``describe_tokens_per_row`` for the token table only).

One :class:`SourceLedger` per source answers **both** questions the pipeline asks — "what is still to download?"
(:meth:`SourceLedger.rows_to_fetch`) and "is this source done?" (:meth:`SourceLedger.satisfaction`) — from one read
of the config and the manifests. :func:`plan_downloads`, :func:`every_source_satisfies_its_budget`,
``runner.another_round_can_fetch_more`` and :func:`summarize_dataset_state` all read that object, so they cannot
disagree. They used to: the plan looked only at *raw* rows and satisfaction only at *processed* rows, so a source
whose dedup or length filter dropped more than the safety margin was short forever with nothing planned (the round
loop gave up and ``prepare`` failed with no way forward), and a source that yielded *zero* processed rows but was
exhausted counted as complete — a wrong ``fields`` / ``converter`` / ``filter`` / ``language`` reported as success.
Now the ledger sizes a **top-up** from the observed yield (processed ÷ raw) and an exhausted source with no rows is
a failure, as "a failed source is a failed build" says it must be — and so is an exhausted source whose few rows all
go to the training-time validation holdout (:func:`training_rows_after_split`): training would only fail at startup
with a confusing empty-range error, so preparation's "satisfied" mirrors the split training will make.

Everything here reads manifests only (no parquet footers): a processed folder's health is the shared verdict of
``lib/build/assessment.py`` in its manifest-only mode (``check_files=False``), so broken or stray shard files are
the repair step's business (``lib/build/repair.py``, the same verdict with the files checked) and the training
resolver checks the folders on disk independently. Pure functions of ``(config, layout)``;
``lib/build/runner.py`` executes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Literal

from data_preparation.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.layout import DatasetLayout, processed_columns
from data_preparation.lib.build.assessment import ProcessedProblem, assess_processed_folder
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.download import RawManifestState, current_raw_manifest, raw_manifest_state
from data_preparation.lib.storage.manifest import shard_list, Manifest
from data_preparation.lib.storage.raw_folder import check_limit_reached, is_exhausted, rejected_rows

log = get_logger(__name__)


# --- rows -----------------------------------------------------------------------------------------------------------


def rows_needed(config: DatasetConfig, name: str) -> int:
    """Raw rows to download for source ``name``: :meth:`DatasetConfig.rows_needed` — the formula lives in the
    schema, where the shuffled-build cap reads the same number at config load, so the two cannot drift. In short:
    the sequence budget times ``SAFETY_MARGIN`` over the training share left by the validation split, or the
    ``rows`` of a validation-only source. No tokens-per-row estimate is involved (see the module docstring): the
    trainer draws rows, so rows are what is counted."""
    return config.rows_needed(name)


def rows_sufficient(config: DatasetConfig, name: str) -> int:
    """Processed rows at which a source serves its budget: ``rows_needed ÷ SAFETY_MARGIN`` (= the sequence budget
    over the training share of the rows, or the ``rows`` of a validation-only source, less the download margin)."""
    return ceil(rows_needed(config, name) / SAFETY_MARGIN)


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
    if not is_exhausted(raw):
        return False
    reached = check_limit_reached(raw)
    if reached is None:
        return True
    limit = config.sources[name].check_limit
    return limit is not None and limit <= reached


def load_processed_manifest(directory: Path) -> Manifest | None:
    """The processed manifest in ``directory``, or None when it is absent **or unreadable**.

    ``Manifest.load`` raises next to shards, which is the right caution for a raw folder (expensive, never guessed)
    but not for a derived one: the repair step deletes an unreadable *processed* folder without asking and the build
    writes it again, so the planner must report that folder as not built instead of crashing ``status`` /
    ``prepare --dry_run`` (and training's auto-prepare, which runs status first) on the very state repair heals.
    Raw manifests keep raising.
    """
    try:
        return Manifest.load(directory)
    except RuntimeError:
        return None


def current_processed_manifest(config: DatasetConfig, name: str, layout: DatasetLayout) -> Manifest | None:
    """The processed manifest of ``name`` when it carries the current ``processed_hash`` and the current columns."""
    manifest = load_processed_manifest(layout.processed_dir(name))
    if manifest is None or manifest.stage != "processed" or not manifest.is_current(config.processed_hash(name)):
        return None
    if manifest.extra.get("columns") != list(processed_columns(config.sources[name].kind)):
        return None
    return manifest


def processed_covers_raw(processed: Manifest, raw: Manifest) -> bool:
    """Whether every raw shard has been built into ``processed`` (``extra["input_shards"]`` lists them all)."""
    return processed.extra.get("input_shards") == shard_list(raw.shards)


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
    """Rows of one source: on disk (raw manifest), needed, and how many still to fetch
    (:meth:`SourceLedger.rows_to_fetch`)."""

    name: str
    rows_present: int  # rows in the raw manifest (0 when missing)
    rows_needed: int  # :func:`rows_needed`
    rows_to_fetch: int  # 0 when the loader is exhausted, when raw is stale, or when the budget is served
    reason: str  # "raw missing" | "rows N < M" | "top-up: ..." | "exhausted" | "enough rows"

    @property
    def rows_target(self) -> int:
        """What :func:`~data_preparation.lib.stages.download.download` is asked for — a **target**, not an increment:
        the rows already on disk plus the ones missing. Equal to :attr:`rows_needed` on a first pass, larger for a
        top-up round that scales the shortfall by the observed yield."""
        return self.rows_present + self.rows_to_fetch


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
    """Rows still missing per source (all, or ``sources``), from each source's :class:`SourceLedger`.

    A raw folder that is stale or outdated is planned as "nothing to fetch" with the state as its reason: the
    repair step deletes it (after confirmation) before any download runs, and a dry run shows the state instead of
    failing — the download never appends to a folder whose rows the current config would not have produced.
    """
    return DownloadPlan([ledger.download() for ledger in read_ledgers(config, layout, sources=sources)])


# --- satisfaction and the status table -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceState:
    """Where one source stands: satisfied when its processed folder serves the budget
    (:meth:`SourceLedger.satisfaction`)."""

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


class Satisfaction(Enum):
    """Whether a source serves its budget, and why not when it does not — one case per outcome, named, so that the
    plan, the round loop and the status table all give the same answer."""

    OK = "ok"  # the processed folder holds at least `rows_sufficient` rows
    EXHAUSTED_SMALL = "exhausted_small"  # the loader ran dry with fewer rows, but at least one survives the training-time validation holdout: served, with a warning
    EXHAUSTED_EMPTY = "exhausted_empty"  # the loader ran dry and NOT ONE row survived the build: a config mistake
    EXHAUSTED_ALL_VALIDATION = "exhausted_all_validation"  # the loader ran dry and the training-time validation holdout takes every processed row: training would start empty
    RAW_BROKEN = "raw_broken"  # the raw folder is stale / outdated: the repair step deletes it after confirmation
    RAW_MISSING = "raw_missing"  # nothing downloaded yet
    NOT_BUILT = "not_built"  # the processed folder is missing, stale, or behind the raw shards
    SHORT_BUT_FETCHABLE = "short_but_fetchable"  # too few processed rows and the loader still has more to give

    @property
    def satisfied(self) -> bool:
        """A source that ran dry is served only when at least one row is left for *training* after the holdout the
        training resolver will take (:func:`training_rows_after_split`) — the sampler cycles what is there. One that
        ran dry with nothing is not: its rows were all rejected, which is a wrong ``fields`` / ``converter`` /
        ``filter`` / ``language``; nor is one whose few rows all go to the validation holdout — training would fail
        at startup with an empty range. A failed source is a failed build, never a silently smaller dataset."""
        return self in (Satisfaction.OK, Satisfaction.EXHAUSTED_SMALL)


ProcessedState = Literal["missing", "unreadable", "stale", "behind_raw", "built"]

_PROCESSED_PROBLEM: dict[str, str] = {
    "missing": "processed missing",
    "unreadable": "processed manifest unreadable: the repair step deletes the folder and builds it again",
    "stale": "processed stale",
    "behind_raw": "processed behind raw",
}


@dataclass(frozen=True)
class SourceLedger:
    """One source as ``prepare`` sees it: the budget from the config, everything else from the manifests, read once.

    The two questions the pipeline asks are answered from this one object — :meth:`rows_to_fetch` ("what is still to
    download?") and :meth:`satisfaction` ("is this source done?") — so they cannot contradict each other. See the
    module docstring for the two bugs that lived in the gap between them.
    """

    name: str
    kind: str
    rows_needed: int  # raw rows to download (:func:`rows_needed`)
    rows_sufficient: int  # processed rows that serve the budget (:func:`rows_sufficient`)
    sequence_budget: int  # sequences the whole run draws (weight-schedule integral; 0 when the source is not trained on)
    raw_state: RawManifestState  # "missing" | "current" | "stale" | "outdated"
    raw_rows: int  # rows in the raw manifest (0 unless the folder is current)
    exhausted: bool  # the loader has nothing more to give (:func:`raw_is_exhausted`)
    skipped_malformed: int  # source rows the converter rejected (raw manifest)
    dropped_too_long: int  # source rows over the token cap (raw manifest)
    processed_state: ProcessedState
    processed_rows: int  # rows in the processed manifest (0 unless it is current)
    training_rows: int  # processed rows left after the training-time validation split

    # --- what to download ------------------------------------------------------------------------------------------

    def download(self) -> SourceDownload:
        """This source's entry in the download plan."""
        rows, reason = self._to_fetch()
        return SourceDownload(self.name, self.raw_rows, self.rows_needed, rows, reason)

    def rows_to_fetch(self) -> int:
        """Raw rows to add: the difference to :attr:`rows_needed` while raw is short; **a top-up sized by the
        observed yield** once raw is long enough but the build dropped more than the safety margin covers; 0 when the
        loader is dry, the raw folder is the repair step's business, or the budget is served."""
        return self._to_fetch()[0]

    def _to_fetch(self) -> tuple[int, str]:
        if self.raw_state not in ("missing", "current"):
            return 0, f"raw {self.raw_state}: the repair step deletes it after confirmation"
        if self.raw_state == "missing":
            return self.rows_needed, "raw missing"
        if self.exhausted:
            return 0, "exhausted"
        if self.raw_rows < self.rows_needed:
            return self.rows_needed - self.raw_rows, f"rows {self.raw_rows:,} < {self.rows_needed:,}"
        if self.processed_state != "built" or self.processed_rows >= self.rows_sufficient:
            return 0, "enough rows"  # nothing to top up: either not built yet (build first), or the budget is served
        top_up = self._top_up_rows()
        if top_up <= 0:
            return 0, f"no row of {self.raw_rows:,} raw rows survives the build"  # more raw would be dropped too
        return top_up, f"top-up: {self.processed_rows:,} of {self.rows_sufficient:,} rows survived {self.raw_rows:,} raw"

    def _top_up_rows(self) -> int:
        """Raw rows to add when the build dropped more than the ``SAFETY_MARGIN`` covers: the shortfall in processed
        rows divided by the yield this source actually showed (``processed ÷ raw``) and multiplied by the same margin
        the first download uses — the yield is one measurement, and overshooting costs a few rows while
        undershooting costs another round. 0 when there is no yield to extrapolate from.

        **Capped at** :attr:`rows_needed`, the full requirement of the budget: a pathological yield (0.08 % of the
        rows surviving, say) extrapolates to billions of rows and would ask the loader for a download nobody wants.
        This is a runtime measurement going wrong mid-download, not a config mistake, so the round is capped with a
        warning and the next round measures the yield again on more data.
        """
        if self.raw_rows <= 0 or self.processed_rows <= 0:
            return 0
        observed_yield = Fraction(self.processed_rows, self.raw_rows)
        wanted = ceil((self.rows_sufficient - self.processed_rows) * SAFETY_MARGIN / observed_yield)
        if wanted <= self.rows_needed:
            return wanted
        log.warning(
            "%s: only %.3f%% of %s raw rows survived the build; a top-up of %s rows would serve the budget — "
            "capping this round at the full requirement of %s rows",
            self.name, 100 * float(observed_yield), f"{self.raw_rows:,}", f"{wanted:,}", f"{self.rows_needed:,}",
        )
        return self.rows_needed

    # --- is it done ------------------------------------------------------------------------------------------------

    def satisfaction(self) -> Satisfaction:
        """Satisfied = the processed manifest is current, covers every raw shard and holds at least
        :attr:`rows_sufficient` rows (so the training part after the split reaches the sequence budget) — or the
        loader is dry with at least one row left for training after the validation holdout
        (:attr:`training_rows`; a source used only in validation holds nothing out, so any row serves it). A stale /
        outdated raw folder is reported (the repair step deletes it after confirmation), never counted."""
        if self.raw_state not in ("missing", "current"):
            return Satisfaction.RAW_BROKEN
        if self.raw_state == "missing":
            return Satisfaction.RAW_MISSING
        if self.processed_state != "built":
            return Satisfaction.NOT_BUILT
        if self.processed_rows >= self.rows_sufficient:
            return Satisfaction.OK
        if self.exhausted:
            if self.processed_rows == 0:
                return Satisfaction.EXHAUSTED_EMPTY
            if self.training_rows < 1:
                return Satisfaction.EXHAUSTED_ALL_VALIDATION
            return Satisfaction.EXHAUSTED_SMALL
        return Satisfaction.SHORT_BUT_FETCHABLE

    def reason(self) -> str:
        """One line for the status table: ``"ok"``, or what is missing."""
        case = self.satisfaction()
        if case is Satisfaction.RAW_BROKEN:
            return f"raw {self.raw_state}: the repair step deletes it after confirmation"
        if case is Satisfaction.RAW_MISSING:
            return "raw missing"
        if case is Satisfaction.NOT_BUILT:
            return _PROCESSED_PROBLEM[self.processed_state]
        if case is Satisfaction.OK:
            return "ok"
        if case is Satisfaction.EXHAUSTED_SMALL:
            return f"exhausted at {self.processed_rows:,} of {self.rows_sufficient:,} rows"
        if case is Satisfaction.EXHAUSTED_EMPTY:
            return (
                f"exhausted and NOT ONE of {self.raw_rows:,} raw rows survived the build "
                f"({self.skipped_malformed:,} malformed, {self.dropped_too_long:,} too long) — "
                "check the source's fields / converter / filter / language"
            )
        if case is Satisfaction.EXHAUSTED_ALL_VALIDATION:
            held_out = self.processed_rows - self.training_rows
            return (
                f"exhausted, and {self.processed_rows:,} processed rows − {held_out:,} validation holdout leaves "
                "0 training rows — lower the source's validation_fraction or give it more rows"
            )
        return f"processed rows {self.processed_rows:,} < {self.rows_sufficient:,}"

    def epochs(self) -> float | None:
        """How often the trainer cycles this source's training rows to serve its sequence budget (the run's total
        demand over all stages); None while it is not satisfied, for a source it does not train on, or without rows."""
        if not self.satisfaction().satisfied or self.sequence_budget <= 0 or self.training_rows <= 0:
            return None
        return self.sequence_budget / self.training_rows

    def state(self) -> SourceState:
        """This source's row of the status table."""
        case = self.satisfaction()
        return SourceState(
            name=self.name, kind=self.kind, rows_needed=self.rows_needed, raw_rows=self.raw_rows,
            processed_rows=self.processed_rows, exhausted=self.exhausted, satisfied=case.satisfied,
            reason=self.reason(), epochs=self.epochs(),
        )


def source_ledger(config: DatasetConfig, name: str, layout: DatasetLayout) -> SourceLedger:
    """Read one source's ledger: the budget from ``config``, the rest from the raw and processed manifests. A raw
    folder that is not current contributes nothing (its rows are about to be deleted or were never downloaded), so
    its processed folder is not counted either."""
    raw = current_raw_manifest(config, name, layout)
    skipped, dropped = (0, 0) if raw is None else rejected_rows(raw)
    processed: tuple[ProcessedState, int] = ("missing", 0) if raw is None else _processed_state(config, name, layout, raw)
    processed_state, processed_rows = processed
    return SourceLedger(
        name=name,
        kind=config.sources[name].kind,
        rows_needed=rows_needed(config, name),
        rows_sufficient=rows_sufficient(config, name),
        sequence_budget=config.sequence_budget(name),
        raw_state=raw_manifest_state(config, name, layout),
        raw_rows=0 if raw is None else raw.rows(),
        exhausted=raw is not None and raw_is_exhausted(config, name, raw),
        skipped_malformed=skipped,
        dropped_too_long=dropped,
        processed_state=processed_state,
        processed_rows=processed_rows,
        training_rows=training_rows_after_split(config, name, processed_rows),
    )


# the shared verdict's manifest-level problems in the planner's vocabulary; the file-level problems (broken /
# stray shards) never appear in manifest-only mode — they are the repair step's business
_ASSESSED_STATE: dict[ProcessedProblem, ProcessedState] = {
    "absent": "missing",
    "no_manifest": "missing",
    "unreadable_manifest": "unreadable",
    "stale": "stale",
}


def _processed_state(config: DatasetConfig, name: str, layout: DatasetLayout, raw: Manifest) -> tuple[ProcessedState, int]:
    """The state of ``processed/<name>`` against the current raw shards, and the rows it holds (0 unless current).

    The folder's health is the shared verdict (:func:`~data_preparation.lib.build.assessment.assess_processed_folder`
    with ``check_files=False``: the planner reads manifests only, see the module docstring). On top of it the
    planner keeps its own knowledge: a manifest from another stage or with other columns is as stale as a wrong
    hash, and a current folder that does not cover every raw shard yet is ``behind_raw`` — a pending build, not a
    repair. An unreadable manifest is the repair step's deletion (no confirmation, processed data is derived),
    reported here instead of raised so ``status`` / ``prepare --dry_run`` describe the very state repair heals.
    """
    directory = layout.processed_dir(name)
    assessment = assess_processed_folder(config, name, directory, shard_list(raw.shards), check_files=False)
    state = _ASSESSED_STATE.get(assessment.problem)
    if state is not None:
        if assessment.problem == "unreadable_manifest":
            log.warning("%s: unreadable manifest in %s; the repair step deletes the folder and builds it again", name, directory)
        return state, 0
    manifest = assessment.manifest
    if manifest is None:  # unreachable: every problem without a readable manifest is mapped above
        return "missing", 0
    if manifest.stage != "processed" or manifest.extra.get("columns") != list(processed_columns(config.sources[name].kind)):
        return "stale", 0
    if not processed_covers_raw(manifest, raw):
        return "behind_raw", manifest.rows()
    return "built", manifest.rows()


def read_ledgers(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> list[SourceLedger]:
    """The ledger of every source (all, or ``sources``), in config order."""
    return [source_ledger(config, name, layout) for name in _selected(config, sources)]


def every_source_satisfies_its_budget(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> bool:
    """Whether every source (all, or ``sources``) is satisfied (:meth:`SourceLedger.satisfaction`)."""
    return all(ledger.satisfaction().satisfied for ledger in read_ledgers(config, layout, sources=sources))


def summarize_dataset_state(config: DatasetConfig, layout: DatasetLayout, *, needs_repair: Iterable[str] = ()) -> DatasetReport:
    """The :class:`DatasetReport` of every source of ``config`` under ``layout`` plus the tokenizer.
    ``needs_repair`` names the sources a repair dry run would touch (``status``): they count as incomplete."""
    return DatasetReport(
        sources=[ledger.state() for ledger in read_ledgers(config, layout)],
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
