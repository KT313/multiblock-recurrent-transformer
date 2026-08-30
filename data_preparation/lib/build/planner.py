# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Budget planner: what a dataset config needs on disk versus what the manifests say is there (pure arithmetic).

Per pretrain source the needed tokens are the **largest** per-stage demand (``DatasetConfig.source_budget_tokens``;
stages share the files, so max, not sum), turned into rows with the measured tokens/row of the processed manifest
when it is current, else the config's ``tokens_per_row_estimate``, times ``SAFETY_MARGIN``. ``plan`` only reads
manifests and parquet footers (``verify_shards``); ``lib/build/runner.py`` executes a plan, ``prepare.py status``
prints it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from data_preparation.lib.schema.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, PROCESSED_COLUMNS, SOURCE_STAGES, DatasetLayout
from data_preparation.lib.storage.manifest import Manifest, verify_shards


def rows_for_budget(budget_tokens: float, tokens_per_row: float, margin: float = SAFETY_MARGIN) -> int:
    """Rows to fetch for ``budget_tokens`` at ``tokens_per_row`` (× safety ``margin``), at least 1."""
    return max(1, ceil(budget_tokens / max(tokens_per_row, 1e-9) * margin))


# --- plan dataclasses --------------------------------------------------------------------------------------------------


@dataclass
class SourcePlan:
    """State of one pretrain or validation source. ``rows_needed``/``rows_to_fetch`` of a validation are its ``rows``."""

    name: str
    kind: str
    budget_tokens: int
    tokens_per_row: float  # measured from the processed manifest if current, else the config estimate
    rows_needed: int
    rows_present: int
    rows_to_fetch: int
    tokens_present: int
    manifest_current: bool  # every stage manifest of the source exists and carries the current source hash
    exhausted: bool  # the loader ran dry before the budget was reached (complete with a warning)
    complete: bool
    reason: str  # "ok", or what is missing

    @property
    def epochs(self) -> float | None:
        """How often the rows on disk are cycled by the training sampler to serve ``budget_tokens`` (the largest
        single-stage demand): ``budget ÷ tokens``; < 1 means only part of the data is seen. None without tokens."""
        return _epochs(self.budget_tokens, self.tokens_present)


@dataclass
class InstructMixturePlan:
    name: str
    budget_tokens: int
    present: bool  # both split manifests exist
    current: bool  # ... and carry the current mixture hash
    short_sources: list[str]  # sources with fewer raw rows than the mixture needed and not exhausted
    tokens_present: int  # tokens of the built train split (0 unless current)
    complete: bool
    reason: str

    @property
    def epochs(self) -> float | None:
        """See :attr:`SourcePlan.epochs` (over the train split)."""
        return _epochs(self.budget_tokens, self.tokens_present)


@dataclass
class Plan:
    sources: list[SourcePlan] = field(default_factory=list)
    validations: list[SourcePlan] = field(default_factory=list)
    instruct_mixtures: list[InstructMixturePlan] = field(default_factory=list)
    tokenizer_complete: bool = False
    complete: bool = False

    def missing(self) -> list[str]:
        """Human-readable one-liners, one per incomplete item (empty iff ``complete``)."""
        lines: list[str] = []
        if not self.tokenizer_complete:
            lines.append("tokenizer: missing or stale")
        lines.extend(f"source {s.name}: {s.reason}" for s in self.sources if not s.complete)
        lines.extend(f"validation {s.name}: {s.reason}" for s in self.validations if not s.complete)
        lines.extend(f"instruct_mixture {m.name}: {m.reason}" for m in self.instruct_mixtures if not m.complete)
        return lines

    def summary(self) -> str:
        """A fixed-width table of every planned item plus the tokenizer and overall state (``epochs``: how often
        the training sampler cycles the rows on disk to serve the budget, see :attr:`SourcePlan.epochs`)."""
        header = ("item", "kind", "budget", "tokens", "rows", "needed", "fetch", "tok/row", "epochs", "state", "reason")
        rows: list[tuple[str, ...]] = []
        for source in self.sources + self.validations:
            rows.append(_source_summary_row(source))
        for mixture in self.instruct_mixtures:
            rows.append(_instruct_mixture_summary_row(mixture))
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
        _fmt_epochs(source.epochs),
        _source_state(source),
        source.reason,
    )


def _source_state(source: SourcePlan) -> str:
    if not source.complete:
        return "incomplete"
    if source.exhausted:
        return "exhausted"
    return "complete"


def _instruct_mixture_summary_row(mixture: InstructMixturePlan) -> tuple[str, ...]:
    return (
        mixture.name,
        "instruct_mixture",
        _fmt(mixture.budget_tokens),
        _fmt(mixture.tokens_present),
        "",  # rows
        "",  # needed
        "",  # fetch
        "",  # tok/row
        _fmt_epochs(mixture.epochs),
        "complete" if mixture.complete else "incomplete",
        mixture.reason,
    )


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
    """Compare ``cfg`` with the manifests under ``layout``; sources no stage or mixture uses are not part of it."""
    tokenizer_complete = _tokenizer_complete(cfg, layout)
    result = Plan(tokenizer_complete=tokenizer_complete)

    for name in cfg.sources_of_kind("pretrain"):
        if cfg.source_budget_tokens(name) > 0:
            result.sources.append(_plan_pretrain(cfg, name, layout, tokenizer_complete))

    used_validations = {key for stage in cfg.stages for key in stage.val}
    for name in cfg.sources_of_kind("validation"):
        if name in used_validations:
            result.validations.append(_plan_validation(cfg, name, layout, tokenizer_complete))

    for name in cfg.instruct_mixtures:
        if cfg.instruct_mixture_budget_tokens(name) > 0:
            result.instruct_mixtures.append(_plan_instruct_mixture(cfg, name, layout, tokenizer_complete))

    result.complete = (
        tokenizer_complete
        and all(source.complete for source in result.sources)
        and all(validation.complete for validation in result.validations)
        and all(mixture.complete for mixture in result.instruct_mixtures)
    )
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


def _stage_dir(layout: DatasetLayout, name: str, stage: str) -> Path:
    if stage == "validation":
        return layout.validation_dir(name)
    return layout.source_dir(name, stage)


def stage_problems(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> dict[str, str]:
    """``{stage: problem}`` for the source stage directories whose manifest is present but stale or unverifiable
    (the build removes those directories before rerunning the stage)."""
    problems: dict[str, str] = {}
    stages = ("validation",) if cfg.sources[name].kind == "validation" else SOURCE_STAGES
    for stage in stages:
        directory = _stage_dir(layout, name, stage)
        if Manifest.load(directory) is None:
            continue  # nothing present is not a problem, only stale or broken directories are
        _, problem = _current(directory, cfg.stage_hash(name, stage), stage)
        if problem is not None:
            problems[stage] = problem
    return problems


# --- pretrain sources --------------------------------------------------------------------------------------------------


def _plan_pretrain(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> SourcePlan:
    source = cfg.sources[name]
    budget = cfg.source_budget_tokens(name)
    manifests, problem = _current_stage_manifests(cfg, name, layout)
    raw = manifests.get("raw")
    processed = manifests.get("processed")

    exhausted = raw is not None and bool(raw.extra.get("exhausted"))
    rows_present = raw.rows() if raw is not None else 0
    tokens_present = (processed.tokens() or 0) if processed is not None else 0

    tokens_per_row = float(min(source.tokens_per_row_estimate, cfg.max_seq_length))  # counts are capped there
    measured = _measured_tokens_per_row(raw, processed)
    if measured is not None:
        tokens_per_row = measured
    rows_needed = rows_for_budget(budget, tokens_per_row)
    rows_to_fetch = max(0, rows_needed - rows_present)

    if problem is None and raw is not None and processed is not None:
        problem = _pipeline_problem(raw, processed, budget, exhausted)
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"

    if problem is not None:
        reason = problem
    elif tokens_present >= budget:
        reason = "ok"
    else:
        reason = f"exhausted at {tokens_present} of {budget} tokens"

    return SourcePlan(
        name=name,
        kind="pretrain",
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
    """The current, verified manifests of the source's stages (``raw``/``processed``) plus the problem
    of the first stage that has none (None if every stage is fine)."""
    manifests: dict[str, Manifest] = {}
    first_problem: str | None = None
    for stage in SOURCE_STAGES:
        manifest, stage_problem = _current(layout.source_dir(name, stage), cfg.stage_hash(name, stage), stage)
        if manifest is not None:
            manifests[stage] = manifest
        elif first_problem is None:
            first_problem = stage_problem
    return manifests, first_problem


def _pipeline_problem(raw: Manifest, processed: Manifest, budget: int, exhausted: bool) -> str | None:
    """Why the stages of a source are not finished even though every manifest is current, or None."""
    if processed.extra.get("columns") != list(PROCESSED_COLUMNS):
        return "processed: predates the hash column"  # `process` rebuilds it from the raw shards, no download
    processed_inputs = processed.extra.get("input_shards")
    if processed_inputs != [[shard.name, shard.rows] for shard in raw.shards]:
        return "processed: behind raw"
    tokens = processed.tokens() or 0
    if tokens < budget and not exhausted:
        return f"tokens {tokens} < budget {budget}"
    return None


def _measured_tokens_per_row(raw: Manifest | None, processed: Manifest | None) -> float | None:
    """Processed tokens per **raw** row over the raw shards the processed manifest covers (this includes what the
    length filter and dedup drop); before anything is processed, the raw manifest's own token counts per raw row
    (available right after the download); None without usable counts."""
    if raw is None:
        return None
    if processed is not None:
        tokens = processed.tokens()
        covered = len(processed.extra.get("input_shards", []))
        raw_rows = sum(shard.rows for shard in raw.shards[:covered])
        if tokens is not None and tokens > 0 and raw_rows > 0:
            return tokens / raw_rows
    raw_tokens = raw.tokens()
    if raw_tokens is not None and raw_tokens > 0 and raw.rows() > 0:
        return raw_tokens / raw.rows()
    return None


# --- validation sources ------------------------------------------------------------------------------------------------


def _plan_validation(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> SourcePlan:
    source = cfg.sources[name]
    wanted = int(source.rows or 0)
    manifest, problem = _current(layout.validation_dir(name), cfg.validation_hash(name), "validation")

    rows_present = manifest.rows() if manifest is not None else 0
    tokens_present = (manifest.tokens() or 0) if manifest is not None else 0
    exhausted = manifest is not None and rows_present < wanted

    if problem is None and manifest is not None and manifest.extra.get("requested_rows") != wanted:
        problem = f"validation: has {rows_present} rows, config wants {wanted}"
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"

    if rows_present > 0:
        tokens_per_row = tokens_present / rows_present
    else:
        tokens_per_row = float(min(source.tokens_per_row_estimate, cfg.max_seq_length))

    if problem is not None:
        reason = problem
    elif exhausted:
        reason = f"exhausted at {rows_present} of {wanted} rows"
    else:
        reason = "ok"

    return SourcePlan(
        name=name,
        kind="validation",
        budget_tokens=0,
        tokens_per_row=tokens_per_row,
        rows_needed=wanted,
        rows_present=rows_present,
        rows_to_fetch=0 if manifest is not None else wanted,
        tokens_present=tokens_present,
        manifest_current=manifest is not None,
        exhausted=exhausted,
        complete=problem is None,
        reason=reason,
    )


# --- instruct mixtures -------------------------------------------------------------------------------------------------


def _plan_instruct_mixture(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> InstructMixturePlan:
    budget = cfg.instruct_mixture_budget_tokens(name)
    split_dirs = {split: layout.instruct_mixture_dir(cfg.name, name, split) for split in INSTRUCT_MIXTURE_SPLITS}
    present = all(Manifest.load(directory) is not None for directory in split_dirs.values())

    manifests, problem = _current_split_manifests(cfg, name, split_dirs)
    current = len(manifests) == len(INSTRUCT_MIXTURE_SPLITS)

    short: list[str] = []
    tokens_present = 0
    if current:
        train = manifests["train"]
        tokens_present = train.tokens() or 0
        short, sources_problem = _check_mixture_sources(cfg, name, layout, train, budget)
        if problem is None:
            problem = sources_problem
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"

    return InstructMixturePlan(
        name=name,
        budget_tokens=budget,
        present=present,
        current=current,
        short_sources=short,
        tokens_present=tokens_present,
        complete=problem is None,
        reason=problem or "ok",
    )


def _current_split_manifests(cfg: DatasetConfig, name: str, split_dirs: dict[str, Path]) -> tuple[dict[str, Manifest], str | None]:
    """The current, verified manifests of the mixture's splits plus the problem of the first split that has none."""
    mixture_hash = cfg.instruct_mixture_hash(name)
    manifests: dict[str, Manifest] = {}
    first_problem: str | None = None
    for split, directory in split_dirs.items():
        manifest, split_problem = _current(directory, mixture_hash, "instruct_mixture")
        if manifest is not None:
            manifests[split] = manifest
        elif first_problem is None:
            first_problem = f"{split} {split_problem}"
    return manifests, first_problem


def _check_mixture_sources(
    cfg: DatasetConfig, name: str, layout: DatasetLayout, train: Manifest, budget: int
) -> tuple[list[str], str | None]:
    """Compare a built mixture (its ``train`` manifest) with the raw shards of its sources and the budget.

    Returns the sources that were short when the mixture was built and are not exhausted (a new fetch could help),
    and the first problem found, or None.
    """
    problem: str | None = None
    short: list[str] = []
    raw_shards: dict[str, list[list[object]]] = {}
    built_from_short_sources = train.extra.get("short_sources", {})
    for src in cfg.instruct_mixtures[name].sources:
        raw, raw_problem = _current(layout.source_dir(src, "raw"), cfg.raw_hash(src), "raw")
        if raw is None:
            if problem is None:
                problem = f"source {src} {raw_problem}"
            continue
        raw_shards[src] = [[shard.name, shard.rows] for shard in raw.shards]
        if src in built_from_short_sources and not raw.extra.get("exhausted"):
            short.append(src)

    if problem is None and train.extra.get("input_shards") != raw_shards:
        problem = "sources changed since the mixture was built"
    if problem is None and train.extra.get("budget_tokens") != budget:
        problem = f"built for {train.extra.get('budget_tokens')} tokens, budget is {budget}"
    if problem is None and short:
        problem = f"short sources {short}"
    return short, problem


__all__ = ["InstructMixturePlan", "Plan", "SourcePlan", "plan", "rows_for_budget", "stage_problems"]
