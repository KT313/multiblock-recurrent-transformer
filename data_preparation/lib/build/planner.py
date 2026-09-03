# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Planner: what a dataset config needs on disk versus what the manifests say is there, counted in sequences.

The trainer draws rows from one continuous stream per source with the stage weight and pads or truncates every row
to block_size, so the run consumes the integral of the source's weight schedule over the stage token budgets
divided by block_size: its :meth:`DatasetConfig.sequence_budget` (the 1.2 safety margin comes on top). That is
the planner's unit: :meth:`DatasetConfig.rows_needed` turns it into a download target,
:meth:`DatasetConfig.rows_sufficient` into the processed rows that serve it. No tokens-per-row estimate enters this
arithmetic (describe.py prints one for its token table only).

One :class:`SourceLedger` per source answers both questions the pipeline asks, "what is still to download?"
(:attr:`SourceLedger.rows_to_fetch`) and "is this source done?" (:meth:`SourceLedger.satisfaction`), from one read
of the config and the manifests, so the plan and the satisfaction check cannot disagree. The ledger sizes a top-up
from the observed yield (processed / raw). An exhausted source with no rows, or whose few rows all go to the
training-time validation holdout (:func:`training_rows_after_split`), is a failure: a failed source is a failed
build, never a silently smaller dataset.

Everything here reads manifests only (no parquet footers): a processed folder's health is the shared verdict of
lib/build/assessment.py with check_files=False; broken or stray shard files are the repair step's business.
Pure functions of (config, layout); lib/build/runner.py executes them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from math import ceil

from data_preparation.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.assessment import ProcessedAssessment, ProcessedProblem, assess_processed_folder
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.download import RawManifestState, inspect_raw
from data_preparation.lib.storage.manifest import shard_list, Manifest

log = get_logger(__name__)


# --- rows ---------------------------------------------------------------------------------------------------------


def training_rows_after_split(config: DatasetConfig, name: str, processed_rows: int) -> int:
    """
    Rows of processed/<name> the trainer trains on: all but the first ceil(validation_fraction × rows)
    (the training resolver's split; the fraction is multiplied as the decimal written in the YAML).
    """

    held_out = Fraction(str(config.validation_fraction_of(name)))
    return processed_rows - ceil(held_out * processed_rows)


# --- manifests -------------------------------------------------------------------------------------------------------


def tokenizer_is_prepared(config: DatasetConfig, layout: DatasetLayout) -> bool:
    """
    Whether tokenizers/<name> carries the current tokenizer manifest and the tokenizer files.
    """

    directory = layout.tokenizer_dir(config.tokenizer.name)
    manifest = Manifest.load(directory)
    if manifest is None:
        return False
    return manifest.is_current(config.tokenizer_hash()) and (directory / "tokenizer_config.json").is_file()


def raw_is_exhausted(config: DatasetConfig, name: str, raw: Manifest) -> bool:
    """
    Whether the loader of name has nothing more to give: the raw manifest says exhausted, unless it was
    exhausted by a check_limit that has since grown or been removed (download reads on then).
    """

    if not raw.exhausted:
        return False
    reached = raw.check_limit_reached
    if reached is None:
        return True
    limit = config.sources[name].check_limit
    return limit is not None and limit <= reached


def assess_processed(config: DatasetConfig, name: str, layout: DatasetLayout, raw: Manifest) -> ProcessedAssessment:
    """
    The shared verdict on processed/<name> against the current raw shards, manifests only (the planner and the
    build read no parquet footers; broken or stray shard files are the repair step's business).
    """

    return assess_processed_folder(config, name, layout.processed_dir(name), shard_list(raw.shards), check_files=False)


def build_is_pending(config: DatasetConfig, name: str, layout: DatasetLayout) -> bool:
    """
    Whether name has raw shards its processed folder does not cover yet (or no healthy processed folder);
    False without a current raw manifest: there is nothing to build from.
    """

    raw = inspect_raw(config, name, layout).current_manifest
    return raw is not None and assess_processed(config, name, layout, raw).problem != "none"


def sources_with_pending_raw_shards(config: DatasetConfig, layout: DatasetLayout, sources: Iterable[str] | None = None) -> list[str]:
    """
    The sources (all, or sources) whose build has raw shards left to process, in config order.
    """

    return [name for name in selected_sources(config, sources) if build_is_pending(config, name, layout)]


# --- the download plan -----------------------------------------------------------------------------------------------


@dataclass
class DownloadPlan:
    """
    The :class:`SourceLedger` of every planned source; what each one has to fetch is :attr:`SourceLedger.rows_to_fetch`.
    """

    sources: list[SourceLedger] = field(default_factory=list)

    def to_fetch(self) -> list[SourceLedger]:
        """
        The sources with rows to fetch.
        """

        return [source for source in self.sources if source.rows_to_fetch[0] > 0]

    def total_rows_to_fetch(self) -> int:
        return sum(source.rows_to_fetch[0] for source in self.sources)

    def summary(self) -> str:
        """
        One line: "3 source(s) short, downloading 12,000 rows (a 4,000, b 8,000, c 0)" or "nothing to
        download".
        """

        short = self.to_fetch()
        if not short:
            return "nothing to download"
        per_source = ", ".join(f"{source.name} {source.rows_to_fetch[0]:,}" for source in short)
        return f"{len(short)} source(s) short, downloading {self.total_rows_to_fetch():,} rows ({per_source})"

    def describe(self) -> str:
        """
        A fixed-width table: source, rows present, rows needed, rows to fetch, reason.
        """

        header = ("source", "present", "needed", "fetch", "reason")
        rows = [
            (source.name, f"{source.raw_rows:,}", f"{source.rows_needed:,}", f"{source.rows_to_fetch[0]:,}", source.rows_to_fetch[1])
            for source in self.sources
        ]
        return format_table(header, rows)


def plan_downloads(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> DownloadPlan:
    """
    Rows still missing per source (all, or sources), from each source's :class:`SourceLedger`.

    A raw folder that is stale or outdated is planned as "nothing to fetch" with the state as its reason: the
    repair step deletes it (after confirmation) before any download runs, and a dry run shows the state instead of
    failing. The download never appends to a folder whose rows the current config would not have produced.
    """

    return DownloadPlan(read_ledgers(config, layout, sources=sources))


# --- satisfaction and the status table -----------------------------------------------------------------------------


@dataclass
class DatasetReport:
    """
    The status of a whole dataset config: one :class:`SourceLedger` per source plus the tokenizer.
    """

    sources: list[SourceLedger] = field(default_factory=list)
    tokenizer_complete: bool = False
    needs_repair: list[str] = field(default_factory=list)  # sources the repair step would touch (`status` only; `prepare` repaired first)

    @property
    def complete(self) -> bool:
        return self.tokenizer_complete and not self.needs_repair and all(source.satisfaction()[0] for source in self.sources)

    def missing(self) -> list[str]:
        """
        Names of the items that are not satisfied or need a repair: the sources, plus "tokenizer" when it
        is missing (empty iff :attr:`complete`).
        """

        names = [source.name for source in self.sources if not source.satisfaction()[0] or source.name in self.needs_repair]
        if not self.tokenizer_complete:
            names.append("tokenizer")
        return names

    def unsatisfied(self) -> list[SourceLedger]:
        """
        The sources that do not serve their budget yet (the runner names them after its rounds).
        """

        return [source for source in self.sources if not source.satisfaction()[0]]

    def table(self) -> str:
        """
        A fixed-width table: source, kind, rows needed, raw rows, processed rows, epochs, state, reason.
        """

        header = ("source", "kind", "needed", "raw", "processed", "epochs", "state", "reason")
        rows = []
        for source in self.sources:
            epochs = source.epochs()
            state = "needs repair" if source.name in self.needs_repair else source.state()
            rows.append((
                source.name, source.kind, f"{source.rows_needed:,}", f"{source.raw_rows:,}", f"{source.processed_rows:,}",
                "-" if epochs is None else f"{epochs:.2f}", state, source.satisfaction()[1],
            ))  # fmt: skip
        rows.append(("tokenizer", "tokenizer", "", "", "", "", "complete" if self.tokenizer_complete else "incomplete", ""))
        return format_table(header, rows)

    def describe(self) -> str:
        """
        The table plus the overall verdict line (dataset complete / dataset INCOMPLETE).
        """

        return self.table() + "\n" + f"dataset {'complete' if self.complete else 'INCOMPLETE'}"


@dataclass(frozen=True)
class SourceLedger:
    """
    One source as prepare sees it: the budget from the config, everything else from the manifests, read once.

    The two questions the pipeline asks, :attr:`rows_to_fetch` ("what is still to download?") and
    :meth:`satisfaction` ("is this source done?"), are answered from this one object, so they cannot contradict
    each other (see the module docstring).
    """

    name: str
    kind: str
    rows_needed: int  # raw rows to download (:meth:`DatasetConfig.rows_needed`)
    rows_sufficient: int  # processed rows that serve the budget (:meth:`DatasetConfig.rows_sufficient`)
    sequence_budget: int  # sequences the whole run draws (weight-schedule integral; 0 when the source is not trained on)
    raw_state: RawManifestState  # "missing" | "current" | "stale" | "outdated"
    raw_reason: str  # the state's reason line (:func:`inspect_raw`): what the plan and the status table say about it
    raw_rows: int  # rows in the raw manifest (0 unless the folder is current)
    exhausted: bool  # the loader has nothing more to give (:func:`raw_is_exhausted`)
    skipped_malformed: int  # source rows the converter rejected (raw manifest)
    dropped_too_long: int  # source rows over the token cap (raw manifest)
    processed_problem: ProcessedProblem  # the shared verdict on the processed folder (manifests only): "none" = built
    processed_reason: str  # its reason line
    processed_rows: int  # rows in the processed manifest (0 unless it is built or behind raw)
    training_rows: int  # processed rows left after the training-time validation split

    # --- what to download ------------------------------------------------------------------------------------------

    @property
    def built(self) -> bool:
        """
        The processed folder is current, has the expected columns and covers every raw shard.
        """

        return self.processed_problem == "none"

    @property
    def rows_target(self) -> int:
        """
        What :func:`~data_preparation.lib.stages.download.download` is asked for: a target, not an increment.
        The rows already on disk plus the ones missing; equal to :attr:`rows_needed` on a first pass, larger for a
        top-up round that scales the shortfall by the observed yield.
        """

        return self.raw_rows + self.rows_to_fetch[0]

    @cached_property
    def rows_to_fetch(self) -> tuple[int, str]:
        """
        (rows, reason): raw rows to add. The difference to :attr:`rows_needed` while raw is short; a top-up
        sized by the observed yield once raw is long enough but the build dropped more than the safety margin
        covers; 0 when the loader is dry, the raw folder is the repair step's business, or the budget is served.
        Computed once per ledger (the pathological-yield warning is logged once).
        """

        if self.raw_state not in ("missing", "current"):
            return 0, f"raw {self.raw_reason}; the repair step deletes it after confirmation"
        if self.raw_state == "missing":
            return self.rows_needed, f"raw {self.raw_reason}"
        if self.exhausted:
            return 0, "exhausted"
        if self.raw_rows < self.rows_needed:
            return self.rows_needed - self.raw_rows, f"rows {self.raw_rows:,} < {self.rows_needed:,}"
        if not self.built or self.processed_rows >= self.rows_sufficient:
            return 0, "enough rows"  # nothing to top up: either not built yet (build first), or the budget is served
        top_up = self._top_up_rows()
        if top_up <= 0:
            return 0, f"no row of {self.raw_rows:,} raw rows survives the build"  # more raw would be dropped too
        return top_up, f"top-up: {self.processed_rows:,} of {self.rows_sufficient:,} rows survived {self.raw_rows:,} raw"

    def _top_up_rows(self) -> int:
        """
        Raw rows to add when the build dropped more than the SAFETY_MARGIN covers: the shortfall in processed
        rows divided by the yield this source showed (processed / raw), times the same margin the first download
        uses. 0 when there is no yield to extrapolate from.

        Capped at :attr:`rows_needed`: a pathological yield (0.08 % surviving, say) extrapolates to billions of
        rows. The round is capped with a warning and the next round measures the yield again on more data.
        """

        if self.raw_rows <= 0 or self.processed_rows <= 0:
            return 0
        observed_yield = Fraction(self.processed_rows, self.raw_rows)
        wanted = ceil((self.rows_sufficient - self.processed_rows) * SAFETY_MARGIN / observed_yield)
        if wanted <= self.rows_needed:
            return wanted
        log.warning(
            "%s: only %.3f%% of %s raw rows survived the build; a top-up of %s rows would serve the budget; "
            "capping this round at the full requirement of %s rows",
            self.name, 100 * float(observed_yield), f"{self.raw_rows:,}", f"{wanted:,}", f"{self.rows_needed:,}",
        )
        return self.rows_needed

    # --- is it done ------------------------------------------------------------------------------------------------

    def satisfaction(self) -> tuple[bool, str]:
        """
        (satisfied, reason). Satisfied: the processed folder is current, covers every raw shard and holds at
        least :attr:`rows_sufficient` rows, or the loader is dry with at least one row left for training after the
        validation holdout (:attr:`training_rows`; the sampler cycles what is there). A source that ran dry with
        nothing is not satisfied (its rows were all rejected: a wrong fields / converter / filter /
        language), nor is one whose few rows all go to the validation holdout: a failed source is a failed
        build, never a silently smaller dataset. A stale / outdated raw folder is reported, never counted. The
        reason is the status table's last column: "ok", or what is missing.
        """

        if self.raw_state not in ("missing", "current"):
            return False, f"raw {self.raw_reason}; the repair step deletes it after confirmation"
        if self.raw_state == "missing":
            return False, f"raw {self.raw_reason}"
        if not self.built:
            return False, f"processed {self.processed_reason}"
        if self.processed_rows >= self.rows_sufficient:
            return True, "ok"
        if self.exhausted:
            if self.processed_rows == 0:
                return False, (
                    f"exhausted and NOT ONE of {self.raw_rows:,} raw rows survived the build "
                    f"({self.skipped_malformed:,} malformed, {self.dropped_too_long:,} too long); "
                    "check the source's fields / converter / filter / language"
                )
            if self.training_rows < 1:
                held_out = self.processed_rows - self.training_rows
                return False, (
                    f"exhausted, and {self.processed_rows:,} processed rows − {held_out:,} validation holdout leaves "
                    "0 training rows; lower the source's validation_fraction or give it more rows"
                )
            return True, f"exhausted at {self.processed_rows:,} of {self.rows_sufficient:,} rows"
        return False, f"processed rows {self.processed_rows:,} < {self.rows_sufficient:,}"

    def state(self) -> str:
        """
        The status table's state column: incomplete, exhausted (satisfied by a dry loader) or complete.
        """

        if not self.satisfaction()[0]:
            return "incomplete"
        return "exhausted" if self.exhausted else "complete"

    def epochs(self) -> float | None:
        """
        How often the trainer cycles this source's training rows to serve its sequence budget (the run's total
        demand over all stages); None while it is not satisfied, for a source it does not train on, or without rows.
        """

        if not self.satisfaction()[0] or self.sequence_budget <= 0 or self.training_rows <= 0:
            return None
        return self.sequence_budget / self.training_rows


def source_ledger(config: DatasetConfig, name: str, layout: DatasetLayout) -> SourceLedger:
    """
    Read one source's ledger: the budget from config, the rest from the raw and processed manifests. A raw
    folder that is not current contributes nothing (its rows are about to be deleted or were never downloaded), so
    its processed folder is not counted either. An unreadable processed manifest is the repair step's deletion (no
    confirmation, processed data is derived), reported instead of raised so status / prepare --dry_run
    describe the very state repair heals.
    """

    raw_inspection = inspect_raw(config, name, layout)
    raw = raw_inspection.current_manifest
    processed = ProcessedAssessment("absent", "missing", None) if raw is None else assess_processed(config, name, layout, raw)
    if processed.problem == "unreadable_manifest":
        log.warning("%s: unreadable manifest in %s; the repair step deletes the folder and builds it again", name, layout.processed_dir(name))
    processed_rows = processed.manifest.rows() if processed.manifest is not None and processed.problem in ("none", "behind_raw") else 0
    return SourceLedger(
        name=name,
        kind=config.sources[name].kind,
        rows_needed=config.rows_needed(name),
        rows_sufficient=config.rows_sufficient(name),
        sequence_budget=config.sequence_budget(name),
        raw_state=raw_inspection.state,
        raw_reason=raw_inspection.reason,
        raw_rows=0 if raw is None else raw.rows(),
        exhausted=raw is not None and raw_is_exhausted(config, name, raw),
        skipped_malformed=0 if raw is None else raw.skipped_malformed,
        dropped_too_long=0 if raw is None else raw.dropped_too_long,
        processed_problem=processed.problem,
        processed_reason=processed.reason,
        processed_rows=processed_rows,
        training_rows=training_rows_after_split(config, name, processed_rows),
    )


def read_ledgers(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> list[SourceLedger]:
    """
    The ledger of every source (all, or sources), in config order.
    """

    return [source_ledger(config, name, layout) for name in selected_sources(config, sources)]


def every_source_satisfies_its_budget(config: DatasetConfig, layout: DatasetLayout, *, sources: Iterable[str] | None = None) -> bool:
    """
    Whether every source (all, or sources) is satisfied (:meth:`SourceLedger.satisfaction`).
    """

    return all(ledger.satisfaction()[0] for ledger in read_ledgers(config, layout, sources=sources))


def summarize_dataset_state(config: DatasetConfig, layout: DatasetLayout, *, needs_repair: Iterable[str] = ()) -> DatasetReport:
    """
    The :class:`DatasetReport` of every source of config under layout plus the tokenizer.
    needs_repair names the sources a repair dry run would touch (status): they count as incomplete.
    """

    return DatasetReport(
        sources=read_ledgers(config, layout),
        tokenizer_complete=tokenizer_is_prepared(config, layout),
        needs_repair=sorted(set(needs_repair)),
    )


# --- helpers ---------------------------------------------------------------------------------------------------------


def selected_sources(config: DatasetConfig, sources: Iterable[str] | None) -> list[str]:
    """
    sources in config order, each once (every source when None); unknown names are an error.
    """

    if sources is None:
        return list(config.sources)
    wanted = set(sources)
    unknown = wanted - set(config.sources)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}")
    return [name for name in config.sources if name in wanted]


def format_table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    """
    Left-aligned columns, two spaces apart, each as wide as its widest cell (header included).
    """

    widths = [max(len(header[column]), *(len(row[column]) for row in rows)) for column in range(len(header))]
    lines = [_table_line(header, widths)]
    lines.extend(_table_line(row, widths) for row in rows)
    return "\n".join(lines)


def _table_line(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
