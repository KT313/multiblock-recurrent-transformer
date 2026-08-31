# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Budget planner: what a dataset config needs on disk versus what the manifests say is there (pure arithmetic).

Every source (both kinds) has a ``raw`` and a ``processed`` folder and is planned the same way. A source used for
training is sized by its sequence budget (``DatasetConfig.sequence_budget``, the largest per-stage demand — stages
share the folders, so max, not sum); a source used only for validation by its ``rows``. ``plan`` only reads
manifests and parquet footers (``verify_shards``); ``lib/build/runner.py`` executes a plan, ``prepare.py status``
prints it.

Interim budget arithmetic (task 8 replaces it with a sequences formula): the token budget of a trained source is
``sequence_budget × block_size`` and it is turned into rows with the measured tokens per raw row of the processed
manifest when it is current, else the source's ``describe_tokens_per_row``, times ``SAFETY_MARGIN``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from data_preparation.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.layout import DatasetLayout, processed_columns
from data_preparation.lib.storage.manifest import Manifest, verify_shards

SOURCE_STAGES: tuple[str, ...] = ("raw", "processed")  # the folders of every source, in build order


def rows_for_budget(budget_tokens: float, tokens_per_row: float, margin: float = SAFETY_MARGIN) -> int:
    """Rows to fetch for ``budget_tokens`` at ``tokens_per_row`` (× safety ``margin``), at least 1."""
    return max(1, ceil(budget_tokens / max(tokens_per_row, 1e-9) * margin))


# --- plan dataclasses --------------------------------------------------------------------------------------------------


@dataclass
class SourcePlan:
    """State of one source. A source used only for validation has ``budget_tokens`` 0 and ``rows_needed`` = its
    ``rows``."""

    name: str
    kind: str
    budget_tokens: int
    tokens_per_row: float  # measured from the manifests if present, else the config's describe estimate (task 8 drops it)
    rows_needed: int
    rows_present: int
    rows_to_fetch: int
    tokens_present: int
    manifest_current: bool  # both folder manifests of the source exist and carry the current hashes
    exhausted: bool  # the loader ran dry before the budget was reached (complete with a warning)
    complete: bool
    reason: str  # "ok", or what is missing

    @property
    def epochs(self) -> float | None:
        """How often the rows on disk are cycled by the training sampler to serve ``budget_tokens`` (the largest
        single-stage demand): ``budget ÷ tokens``; < 1 means only part of the data is seen. None without tokens or
        budget (validation-only sources)."""
        return _epochs(self.budget_tokens, self.tokens_present)


@dataclass
class Plan:
    sources: list[SourcePlan] = field(default_factory=list)
    tokenizer_complete: bool = False
    complete: bool = False

    def missing(self) -> list[str]:
        """Human-readable one-liners, one per incomplete item (empty iff ``complete``)."""
        lines: list[str] = []
        if not self.tokenizer_complete:
            lines.append("tokenizer: missing or stale")
        lines.extend(f"source {s.name}: {s.reason}" for s in self.sources if not s.complete)
        return lines

    def summary(self) -> str:
        """A fixed-width table of every source plus the tokenizer and overall state (``epochs``: how often the
        training sampler cycles the rows on disk to serve the budget, see :attr:`SourcePlan.epochs`)."""
        header = ("item", "kind", "budget", "tokens", "rows", "needed", "fetch", "tok/row", "epochs", "state", "reason")
        rows: list[tuple[str, ...]] = [_source_summary_row(source) for source in self.sources]
        rows.append(_tokenizer_summary_row(self.tokenizer_complete))
        table = _format_table(header, rows)
        return table + "\n" + f"dataset {'complete' if self.complete else 'INCOMPLETE'}"


# --- summary table formatting ------------------------------------------------------------------------------------------


def _source_summary_row(source: SourcePlan) -> tuple[str, ...]:
    return (
        source.name,
        source.kind,
        _fmt(source.budget_tokens),
        _fmt(source.tokens_present),
        _fmt(source.rows_present),
        _fmt(source.rows_needed),
        "-" if source.complete else _fmt(source.rows_to_fetch),
        f"{source.tokens_per_row:.1f}",
        _fmt_epochs(source.epochs if source.complete else None),  # meaningless before the source is built
        _source_state(source),
        source.reason,
    )


def _source_state(source: SourcePlan) -> str:
    if not source.complete:
        return "incomplete"
    if source.exhausted:
        return "exhausted"
    return "complete"


def _tokenizer_summary_row(tokenizer_complete: bool) -> tuple[str, ...]:
    state = "complete" if tokenizer_complete else "incomplete"
    return ("tokenizer", "tokenizer", "", "", "", "", "", "", "", state, "")


def _format_table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Left-aligned columns, two spaces apart, each as wide as its widest cell (header included)."""
    widths = [max(len(header[column]), *(len(row[column]) for row in rows)) for column in range(len(header))]
    lines = [_format_table_line(header, widths)]
    for row in rows:
        lines.append(_format_table_line(row, widths))
    return "\n".join(lines)


def _format_table_line(cells: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))


def _fmt(n: int) -> str:
    return f"{n:,}"


def _epochs(budget_tokens: int, tokens_present: int) -> float | None:
    if tokens_present <= 0 or budget_tokens <= 0:
        return None
    return budget_tokens / tokens_present


def _fmt_epochs(epochs: float | None) -> str:
    return "-" if epochs is None else f"{epochs:.2f}"


# --- planning ----------------------------------------------------------------------------------------------------------


def plan(cfg: DatasetConfig, layout: DatasetLayout) -> Plan:
    """Compare ``cfg`` with the manifests under ``layout`` (the config rejects sources no stage uses)."""
    tokenizer_complete = _tokenizer_complete(cfg, layout)
    result = Plan(tokenizer_complete=tokenizer_complete)
    for name in cfg.sources:
        result.sources.append(_plan_source(cfg, name, layout, tokenizer_complete))
    result.complete = tokenizer_complete and all(source.complete for source in result.sources)
    return result


def _tokenizer_complete(cfg: DatasetConfig, layout: DatasetLayout) -> bool:
    out = layout.tokenizer_dir(cfg.tokenizer.name)
    manifest = Manifest.load(out)
    if manifest is None:
        return False
    return manifest.is_current(cfg.tokenizer_hash()) and (out / "tokenizer_config.json").is_file()


def _current(directory: Path, source_hash: str, stage: str) -> tuple[Manifest | None, str | None]:
    """(manifest, problem): the manifest if it is current and its shards verify, else None and why."""
    manifest = Manifest.load(directory)
    if manifest is None:
        return None, f"{stage}: manifest missing"
    if manifest.stage != stage or not manifest.is_current(source_hash):
        return None, f"{stage}: manifest stale"
    problems = verify_shards(directory, manifest)
    if problems:
        return None, f"{stage}: {problems[0]}"
    return manifest, None


def stage_dir(layout: DatasetLayout, name: str, stage: str) -> Path:
    """The folder of ``stage`` (``raw`` / ``processed``) of source ``name``."""
    return layout.raw_dir(name) if stage == "raw" else layout.processed_dir(name)


def stage_hash(cfg: DatasetConfig, name: str, stage: str) -> str:
    """The manifest key of ``stage`` of source ``name``."""
    return cfg.raw_hash(name) if stage == "raw" else cfg.processed_hash(name)


def stage_problems(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> dict[str, str]:
    """``{stage: problem}`` for the source folders whose manifest is present but stale or unverifiable (the build
    removes or repairs those folders before rerunning the step)."""
    problems: dict[str, str] = {}
    for stage in SOURCE_STAGES:
        directory = stage_dir(layout, name, stage)
        if Manifest.load(directory) is None:
            continue  # nothing present is not a problem, only stale or broken folders are
        _, problem = _current(directory, stage_hash(cfg, name, stage), stage)
        if problem is not None:
            problems[stage] = problem
    return problems


# --- one source --------------------------------------------------------------------------------------------------------


def budget_tokens_of(cfg: DatasetConfig, name: str) -> int:
    """Interim token budget of a source: ``sequence_budget × block_size`` (0 for a source used only for validation).
    task 8: the planner counts sequences and this disappears."""
    return cfg.sequence_budget(name) * cfg.block_size


def _plan_source(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> SourcePlan:
    source = cfg.sources[name]
    trained = cfg.used_in_train(name)
    budget = budget_tokens_of(cfg, name)
    manifests, problem = _current_stage_manifests(cfg, name, layout)
    raw = manifests.get("raw")
    processed = manifests.get("processed")

    exhausted = raw is not None and bool(raw.extra.get("exhausted"))
    rows_present = raw.rows() if raw is not None else 0
    tokens_present = (processed.tokens() or 0) if processed is not None else 0

    tokens_per_row = float(min(source.describe_tokens_per_row, cfg.max_seq_length))  # pretrain counts are capped there
    measured = _measured_tokens_per_row(raw, processed)
    if measured is not None:
        tokens_per_row = measured
    # task 8: rows_needed = ceil(sequence_budget × SAFETY_MARGIN ÷ (1 − validation_fraction)); no tokens per row
    rows_needed = rows_for_budget(budget, tokens_per_row) if trained else int(source.rows or 0)
    rows_to_fetch = max(0, rows_needed - rows_present)

    if problem is None and raw is not None and processed is not None:
        problem = _pipeline_problem(source.kind, raw, processed, budget, rows_needed, trained, exhausted)
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"

    if problem is not None:
        reason = problem
    elif trained and tokens_present < budget:
        reason = f"exhausted at {tokens_present} of {budget} tokens"
    elif not trained and rows_present < rows_needed:
        reason = f"exhausted at {rows_present} of {rows_needed} rows"
    else:
        reason = "ok"

    return SourcePlan(
        name=name,
        kind=source.kind,
        budget_tokens=budget,
        tokens_per_row=tokens_per_row,
        rows_needed=rows_needed,
        rows_present=rows_present,
        rows_to_fetch=rows_to_fetch,
        tokens_present=tokens_present,
        manifest_current=len(manifests) == len(SOURCE_STAGES),
        exhausted=exhausted,
        complete=problem is None,
        reason=reason,
    )


def _current_stage_manifests(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> tuple[dict[str, Manifest], str | None]:
    """The current, verified manifests of the source's folders (``raw`` / ``processed``) plus the problem of the
    first folder that has none (None if both are fine)."""
    manifests: dict[str, Manifest] = {}
    first_problem: str | None = None
    for stage in SOURCE_STAGES:
        manifest, problem = _current(stage_dir(layout, name, stage), stage_hash(cfg, name, stage), stage)
        if manifest is not None:
            manifests[stage] = manifest
        elif first_problem is None:
            first_problem = problem
    return manifests, first_problem


def _pipeline_problem(
    kind: str, raw: Manifest, processed: Manifest, budget: int, rows_needed: int, trained: bool, exhausted: bool
) -> str | None:
    """Why the folders of a source are not finished even though both manifests are current, or None."""
    if processed.extra.get("columns") != list(processed_columns(kind)):
        return "processed: predates the current columns"  # the build rebuilds it from the raw shards, no download
    if processed.extra.get("input_shards") != [[shard.name, shard.rows] for shard in raw.shards]:
        return "processed: behind raw"
    if exhausted:
        return None
    if trained:
        tokens = processed.tokens() or 0
        if tokens < budget:
            return f"tokens {tokens} < budget {budget}"
    elif raw.rows() < rows_needed:
        return f"rows {raw.rows()} < {rows_needed}"
    return None


def _measured_tokens_per_row(raw: Manifest | None, processed: Manifest | None) -> float | None:
    """Processed tokens per **raw** row over the raw shards the processed manifest covers (this includes what the
    filters and the dedup drop); before anything is processed, the raw manifest's own token counts per raw row
    (available right after the download); None without usable counts."""
    if raw is None:
        return None
    if processed is not None:
        tokens = processed.tokens() or 0
        covered = len(processed.extra.get("input_shards", []))
        raw_rows = sum(shard.rows for shard in raw.shards[:covered])
        if tokens > 0 and raw_rows > 0:
            return tokens / raw_rows
    raw_tokens = raw.tokens()
    if raw_tokens is not None and raw_tokens > 0 and raw.rows() > 0:
        return raw_tokens / raw.rows()
    return None


__all__ = ["SOURCE_STAGES", "Plan", "SourcePlan", "budget_tokens_of", "plan", "rows_for_budget", "stage_dir", "stage_hash", "stage_problems"]
