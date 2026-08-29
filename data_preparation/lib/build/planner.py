# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Budget planner: what a dataset config needs on disk versus what the manifests say is there (pure arithmetic).

Per pretrain source the needed tokens are the **largest** per-stage demand (``DatasetConfig.source_budget_tokens``;
stages share the files, so max, not sum), turned into rows with the measured tokens/row of the processed manifest
when it is current, else the config's ``tokens_per_row_estimate``, times ``SAFETY_MARGIN``. ``plan`` only reads
manifests and parquet footers (``verify_shards``); ``lib/build.py`` executes a plan, ``prepare.py status`` prints it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from data_preparation.lib.schema.dataset_config import SAFETY_MARGIN, DatasetConfig
from data_preparation.lib.schema.layout import MIXTURE_SPLITS, SOURCE_STAGES, DatasetLayout
from data_preparation.lib.storage.manifest import Manifest, verify_shards


def rows_for_budget(budget_tokens: float, tokens_per_row: float, margin: float = SAFETY_MARGIN) -> int:
    """Rows to fetch for ``budget_tokens`` at ``tokens_per_row`` (× safety ``margin``), at least 1."""
    return max(1, ceil(budget_tokens / max(tokens_per_row, 1e-9) * margin))


@dataclass
class SourcePlan:
    """State of one pretrain or holdout source. ``rows_needed``/``rows_to_fetch`` of a holdout are its ``rows``."""

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


@dataclass
class MixturePlan:
    name: str
    budget_tokens: int
    present: bool  # both split manifests exist
    current: bool  # ... and carry the current mixture hash
    short_sources: list[str]  # sources with fewer raw rows than the mixture needed and not exhausted
    complete: bool
    reason: str


@dataclass
class Plan:
    sources: list[SourcePlan] = field(default_factory=list)
    holdouts: list[SourcePlan] = field(default_factory=list)
    mixtures: list[MixturePlan] = field(default_factory=list)
    tokenizer_complete: bool = False
    complete: bool = False

    def missing(self) -> list[str]:
        """Human-readable one-liners, one per incomplete item (empty iff ``complete``)."""
        lines: list[str] = []
        if not self.tokenizer_complete:
            lines.append("tokenizer: missing or stale")
        lines.extend(f"source {s.name}: {s.reason}" for s in self.sources if not s.complete)
        lines.extend(f"holdout {s.name}: {s.reason}" for s in self.holdouts if not s.complete)
        lines.extend(f"mixture {m.name}: {m.reason}" for m in self.mixtures if not m.complete)
        return lines

    def summary(self) -> str:
        """A fixed-width table of every planned item plus the tokenizer and overall state."""
        header = ("item", "kind", "budget", "tokens", "rows", "needed", "fetch", "tok/row", "state", "reason")
        rows: list[tuple[str, ...]] = []
        for s in self.sources + self.holdouts:
            state = "complete" if s.complete else "incomplete"
            if s.complete and s.exhausted:
                state = "exhausted"
            rows.append(
                (
                    s.name,
                    s.kind,
                    _fmt(s.budget_tokens),
                    _fmt(s.tokens_present),
                    _fmt(s.rows_present),
                    _fmt(s.rows_needed),
                    _fmt(s.rows_to_fetch),
                    f"{s.tokens_per_row:.1f}",
                    state,
                    s.reason,
                )
            )
        for m in self.mixtures:
            rows.append((m.name, "mixture", _fmt(m.budget_tokens), "", "", "", "", "", "complete" if m.complete else "incomplete", m.reason))
        rows.append(("tokenizer", "tokenizer", "", "", "", "", "", "", "complete" if self.tokenizer_complete else "incomplete", ""))
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
        lines.extend("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
        lines.append(f"dataset {'complete' if self.complete else 'INCOMPLETE'}")
        return "\n".join(lines)


def _fmt(n: int) -> str:
    return f"{n:,}"


# --- planning ----------------------------------------------------------------------------------------------------------


def plan(cfg: DatasetConfig, layout: DatasetLayout) -> Plan:
    """Compare ``cfg`` with the manifests under ``layout``; sources no stage or mixture uses are not part of it."""
    tokenizer_complete = _tokenizer_complete(cfg, layout)
    result = Plan(tokenizer_complete=tokenizer_complete)
    for name in cfg.sources_of_kind("pretrain"):
        if cfg.source_budget_tokens(name) > 0:
            result.sources.append(_plan_pretrain(cfg, name, layout, tokenizer_complete))
    used_holdouts = {key for stage in cfg.stages for key in stage.val}
    for name in cfg.sources_of_kind("holdout"):
        if name in used_holdouts:
            result.holdouts.append(_plan_holdout(cfg, name, layout, tokenizer_complete))
    for name in cfg.mixtures:
        if cfg.mixture_budget_tokens(name) > 0:
            result.mixtures.append(_plan_mixture(cfg, name, layout, tokenizer_complete))
    result.complete = (
        tokenizer_complete
        and all(s.complete for s in result.sources)
        and all(h.complete for h in result.holdouts)
        and all(m.complete for m in result.mixtures)
    )
    return result


def _tokenizer_complete(cfg: DatasetConfig, layout: DatasetLayout) -> bool:
    out = layout.tokenizer_dir(cfg.tokenizer.name)
    manifest = Manifest.load(out)
    return manifest is not None and manifest.is_current(cfg.tokenizer_hash()) and (out / "tokenizer_config.json").is_file()


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


def stage_problems(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> dict[str, str]:
    """``{stage: problem}`` for the source stage directories whose manifest is present but stale or unverifiable
    (the build removes those directories before rerunning the stage)."""
    problems: dict[str, str] = {}
    source_hash = cfg.source_hash(name)
    stages = ("holdout",) if cfg.sources[name].kind == "holdout" else SOURCE_STAGES
    for stage in stages:
        directory = layout.holdout_dir(name) if stage == "holdout" else layout.source_dir(name, stage)
        manifest = Manifest.load(directory)
        if manifest is None:
            continue
        _, problem = _current(directory, source_hash, stage)
        if problem is not None:
            problems[stage] = problem
    return problems


def _plan_pretrain(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> SourcePlan:
    source = cfg.sources[name]
    source_hash = cfg.source_hash(name)
    budget = cfg.source_budget_tokens(name)
    manifests: dict[str, Manifest] = {}
    problem: str | None = None
    for stage in SOURCE_STAGES:
        manifest, stage_problem = _current(layout.source_dir(name, stage), source_hash, stage)
        if manifest is None:
            problem = problem or stage_problem
            continue
        manifests[stage] = manifest
    raw, filtered, processed = manifests.get("raw"), manifests.get("filtered"), manifests.get("processed")
    manifest_current = len(manifests) == len(SOURCE_STAGES)
    exhausted = bool(raw is not None and raw.extra.get("exhausted"))
    rows_present = raw.rows() if raw is not None else 0
    tokens_present = (processed.tokens() or 0) if processed is not None else 0

    tokens_per_row = float(source.tokens_per_row_estimate)
    if raw is not None and processed is not None and not source.repeat_to_budget:
        measured = _measured_tokens_per_row(raw, processed)
        if measured is not None:
            tokens_per_row = measured
    rows_needed = rows_for_budget(budget, tokens_per_row)
    if source.repeat_to_budget:
        rows_to_fetch = 0 if raw is not None and raw.rows_fetched > 0 else rows_needed
    else:
        rows_to_fetch = max(0, rows_needed - rows_present)

    if problem is None and raw is not None and filtered is not None and processed is not None:
        if len(filtered.extra.get("input_shards", [])) != len(raw.shards):
            problem = "filtered: behind raw"
        elif processed.extra.get("input_shards") != [[s.name, s.rows] for s in filtered.shards]:
            problem = "processed: behind filtered"
        elif source.repeat_to_budget and processed.extra.get("target_tokens") != budget:
            problem = f"processed: repeated to {processed.extra.get('target_tokens')} tokens, budget is {budget}"
        elif tokens_present < budget and not exhausted:
            problem = f"tokens {tokens_present} < budget {budget}"
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"
    complete = problem is None
    reason = problem or ("ok" if tokens_present >= budget else f"exhausted at {tokens_present} of {budget} tokens")
    return SourcePlan(
        name=name,
        kind="pretrain",
        budget_tokens=budget,
        tokens_per_row=tokens_per_row,
        rows_needed=rows_needed,
        rows_present=rows_present,
        rows_to_fetch=rows_to_fetch,
        tokens_present=tokens_present,
        manifest_current=manifest_current,
        exhausted=exhausted,
        complete=complete,
        reason=reason,
    )


def _measured_tokens_per_row(raw: Manifest, processed: Manifest) -> float | None:
    """Processed tokens per **raw** row over the raw shards the processed manifest covers (filtered shard N mirrors
    raw shard N), or None without usable counts."""
    tokens = processed.tokens()
    covered = len(processed.extra.get("input_shards", []))
    raw_rows = sum(s.rows for s in raw.shards[:covered])
    if tokens is None or tokens <= 0 or raw_rows <= 0:
        return None
    return tokens / raw_rows


def _plan_holdout(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> SourcePlan:
    source = cfg.sources[name]
    wanted = int(source.rows or 0)
    manifest, problem = _current(layout.holdout_dir(name), cfg.source_hash(name), "holdout")
    rows_present = manifest.rows() if manifest is not None else 0
    tokens_present = (manifest.tokens() or 0) if manifest is not None else 0
    exhausted = manifest is not None and rows_present < wanted
    if problem is None and manifest is not None and manifest.extra.get("requested_rows") != wanted:
        problem = f"holdout: has {rows_present} rows, config wants {wanted}"
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"
    tokens_per_row = tokens_present / rows_present if rows_present > 0 else float(source.tokens_per_row_estimate)
    reason = problem or ("ok" if not exhausted else f"exhausted at {rows_present} of {wanted} rows")
    return SourcePlan(
        name=name,
        kind="holdout",
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


def _plan_mixture(cfg: DatasetConfig, name: str, layout: DatasetLayout, tokenizer_complete: bool) -> MixturePlan:
    mixture = cfg.mixtures[name]
    mixture_hash = cfg.mixture_hash(name)
    budget = cfg.mixture_budget_tokens(name)
    dirs = {split: layout.mixture_dir(cfg.name, name, split) for split in MIXTURE_SPLITS}
    loaded = {split: Manifest.load(path) for split, path in dirs.items()}
    present = all(m is not None for m in loaded.values())
    problem: str | None = None
    manifests: dict[str, Manifest] = {}
    for split, path in dirs.items():
        manifest, split_problem = _current(path, mixture_hash, "mixture")
        if manifest is None:
            problem = problem or f"{split} {split_problem}"
        else:
            manifests[split] = manifest
    current = len(manifests) == len(MIXTURE_SPLITS)
    short: list[str] = []
    if current:
        train = manifests["train"]
        raw_shards: dict[str, list[list[object]]] = {}
        for src in mixture.sources:
            raw, raw_problem = _current(layout.source_dir(src, "raw"), cfg.source_hash(src), "raw")
            if raw is None:
                problem = problem or f"source {src} {raw_problem}"
                continue
            raw_shards[src] = [[s.name, s.rows] for s in raw.shards]
            if src in train.extra.get("short_sources", {}) and not raw.extra.get("exhausted"):
                short.append(src)
        if problem is None and train.extra.get("input_shards") != raw_shards:
            problem = "sources changed since the mixture was built"
        if problem is None and train.extra.get("budget_tokens") != budget:
            problem = f"built for {train.extra.get('budget_tokens')} tokens, budget is {budget}"
        if problem is None and short:
            problem = f"short sources {short}"
    if problem is None and not tokenizer_complete:
        problem = "tokenizer missing"
    return MixturePlan(
        name=name,
        budget_tokens=budget,
        present=present,
        current=current,
        short_sources=short,
        complete=problem is None,
        reason=problem or "ok",
    )


__all__ = ["MixturePlan", "Plan", "SourcePlan", "plan", "rows_for_budget", "stage_problems"]
