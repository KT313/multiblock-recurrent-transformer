# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build`` / ``status``: execute a :func:`planner.plan` (tokenizer -> pretrain sources -> validation sources -> instruct mixtures)
and report it.

Per pretrain source the estimate -> measured refinement loop runs download -> length_filter -> process until the
processed tokens reach the budget or the loader is exhausted (at most ``max_rounds`` rounds, then an error). A
mixture is rebuilt until none of its sources is short. Stage directories whose manifest is stale or whose shards
do not verify are removed before their stage reruns. Every stage failure propagates after logging which source
failed; nothing is swallowed.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

from data_preparation.lib.schema.dataset_config import DatasetConfig
from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import progress
from data_preparation.lib.build.planner import InstructMixturePlan, Plan, SourcePlan, plan, rows_for_budget, stage_problems
from data_preparation.lib.stages import build_instruct_mixture, download, validation, length_filter, prepare_tokenizer, process

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "filter", "process", "validation", "instruct_mixtures")
DEFAULT_MAX_ROUNDS = 5


def status(cfg: DatasetConfig, layout: DatasetLayout) -> Plan:
    """The plan for ``cfg`` under ``layout``, logged as a summary (warnings for exhausted sources)."""
    result = plan(cfg, layout)
    _log_plan(result)
    return result


def build(
    cfg: DatasetConfig,
    layout: DatasetLayout,
    *,
    sources: list[str] | None = None,
    steps: set[str] | None = None,
    num_workers: int = 1,
    hf_token: str | None = None,
    dry_run: bool = False,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Plan:
    """Materialise what the plan says is missing; returns the re-computed plan.

    ``steps`` (subset of :data:`STEPS`) limits which stages run, ``sources`` limits to the named sources / mixtures.
    ``dry_run`` logs the plan and returns it without writing anything.
    """
    active = set(STEPS) if steps is None else set(steps)
    if not active <= set(STEPS):
        raise ValueError(f"unknown steps {sorted(active - set(STEPS))}; expected a subset of {STEPS}")
    if sources is not None:
        unknown = set(sources) - set(cfg.sources) - set(cfg.instruct_mixtures)
        if unknown:
            raise ValueError(f"unknown sources/instruct mixtures {sorted(unknown)}")
    selected = None if sources is None else set(sources)

    current = plan(cfg, layout)
    if dry_run:
        log.info("dry run, plan for %s under %s:\n%s", cfg.name, layout.root, current.summary())
        return current
    log.info("building %s under %s (%d item(s) missing)", cfg.name, layout.root, len(current.missing()))

    items: list[tuple[str, str, Callable[[], object]]] = []
    if "tokenizer" in active and not current.tokenizer_complete:
        items.append(("tokenizer", cfg.tokenizer.name, lambda: prepare_tokenizer(cfg, layout)))
    for source_plan in current.sources:
        if selected is not None and source_plan.name not in selected:
            continue
        if source_plan.complete:
            log.info("%s: complete, skipping", source_plan.name)
            continue
        items.append(("source", source_plan.name, partial(_build_pretrain_source, cfg, source_plan, layout, active, num_workers, hf_token, max_rounds)))
    if "validation" in active:
        for validation_plan in current.validations:
            if selected is not None and validation_plan.name not in selected:
                continue
            if validation_plan.complete:
                log.info("%s: complete, skipping", validation_plan.name)
                continue
            items.append(("validation", validation_plan.name, partial(_build_validation, cfg, validation_plan, layout)))
    if "instruct_mixtures" in active:
        for instruct_mixture_plan in current.instruct_mixtures:
            if selected is not None and instruct_mixture_plan.name not in selected:
                continue
            if instruct_mixture_plan.complete:
                log.info("%s: complete, skipping", instruct_mixture_plan.name)
                continue
            items.append(("instruct_mixture", instruct_mixture_plan.name, partial(_build_instruct_mixture, cfg, instruct_mixture_plan, layout, hf_token, max_rounds)))
    with progress(total=len(items), desc="sources", unit="item") as bar:
        for position, (what, name, action) in enumerate(items, start=1):
            bar.set_description(f"sources {position}/{len(items)}: {name}")
            _run(what, name, action)
            bar.update(1)

    final = plan(cfg, layout)
    _log_plan(final)
    return final


# --- helpers ---------------------------------------------------------------------------------------------------------


def _run(what: str, name: str, action: Callable[[], object]) -> None:
    try:
        action()
    except Exception:
        log.error("%s %s failed", what, name)
        raise


def _log_plan(result: Plan) -> None:
    for item in result.sources + result.validations:
        if item.complete and item.exhausted:
            if item.reason == "ok":
                log.warning(
                    "%s: source exhausted at %d rows (%d tokens, budget %d)",
                    item.name, item.rows_present, item.tokens_present, item.budget_tokens,
                )
            else:
                log.warning("%s: %s", item.name, item.reason)
    log.info("dataset status:\n%s", result.summary())


def _remove_broken_stages(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> None:
    """Delete stage directories whose manifest is stale or whose shards do not verify, so the stage rebuilds."""
    for stage, problem in stage_problems(cfg, name, layout).items():
        directory = layout.validation_dir(name) if stage == "validation" else layout.source_dir(name, stage)
        log.warning("%s: %s; removing %s", name, problem, directory)
        shutil.rmtree(directory)


def _build_pretrain_source(
    cfg: DatasetConfig,
    source_plan: SourcePlan,
    layout: DatasetLayout,
    active: set[str],
    num_workers: int,
    hf_token: str | None,
    max_rounds: int,
) -> None:
    name, budget = source_plan.name, source_plan.budget_tokens
    _remove_broken_stages(cfg, name, layout)
    tokens_per_row = source_plan.tokens_per_row
    for round_index in range(max_rounds):
        rows_needed = rows_for_budget(budget, tokens_per_row)
        if "download" in active:
            raw = download(cfg, name, layout, rows_needed=rows_needed, hf_token=hf_token)
        else:
            raw = None
        if "filter" in active:
            length_filter(cfg, name, layout)
        if "process" not in active:
            return
        processed = process(cfg, name, layout, num_workers=num_workers)
        tokens = processed.tokens() or 0
        exhausted = raw is not None and bool(raw.extra.get("exhausted"))
        if tokens >= budget or exhausted or raw is None:
            if tokens < budget:
                log.warning("%s: %s at %d tokens, budget is %d", name, "exhausted" if exhausted else "no download step", tokens, budget)
            return
        tokens_per_row = max(tokens / max(raw.rows(), 1), 1.0)  # floor: never more than `budget` rows per round
        log.info("%s: round %d: %d of %d tokens after processing, refining to %.1f tokens/row", name, round_index + 1, tokens, budget, tokens_per_row)
    raise RuntimeError(f"{name}: token budget {budget} not reached after {max_rounds} rounds")


def _build_validation(cfg: DatasetConfig, validation_plan: SourcePlan, layout: DatasetLayout) -> None:
    name = validation_plan.name
    _remove_broken_stages(cfg, name, layout)  # `rows` is part of the source hash, so a stale manifest covers row changes
    validation(cfg, name, layout)


def _build_instruct_mixture(cfg: DatasetConfig, instruct_mixture_plan: InstructMixturePlan, layout: DatasetLayout, hf_token: str | None, max_rounds: int) -> None:
    instruct_mixture_name, budget = instruct_mixture_plan.name, instruct_mixture_plan.budget_tokens
    mixture = cfg.instruct_mixtures[instruct_mixture_name]
    for src in mixture.sources:
        _remove_broken_stages(cfg, src, layout)
    for split in INSTRUCT_MIXTURE_SPLITS:
        directory = layout.instruct_mixture_dir(cfg.name, instruct_mixture_name, split)
        if directory.exists() and not instruct_mixture_plan.current:
            _remove_dir(directory)
    tokens_per_row = {src: float(cfg.sources[src].tokens_per_row_estimate) for src in mixture.sources}
    short: dict[str, object] = {}
    for round_index in range(max_rounds):
        exhausted = set()
        for src, share in mixture.sources.items():
            raw = download(cfg, src, layout, rows_needed=rows_for_budget(budget * share, tokens_per_row[src]), hf_token=hf_token)
            if raw.extra.get("exhausted"):
                exhausted.add(src)
        result = build_instruct_mixture(cfg, instruct_mixture_name, layout, budget_tokens=budget)
        short = {src: info for src, info in result["train"].extra["short_sources"].items() if src not in exhausted}
        if not short:
            for src in exhausted & set(result["train"].extra["short_sources"]):
                log.warning("%s: source %s exhausted before its share of the budget", instruct_mixture_name, src)
            return
        for src in short:
            tokens_per_row[src] = max(result["train"].extra["tokens_per_row"][src], 1.0)
        log.info("%s: round %d: short sources %s, refining tokens/row", instruct_mixture_name, round_index + 1, sorted(short))
    raise RuntimeError(f"{instruct_mixture_name}: sources {sorted(short)} still short after {max_rounds} rounds")


def _remove_dir(directory: Path) -> None:
    log.warning("removing stale %s", directory)
    shutil.rmtree(directory)


__all__ = ["DEFAULT_MAX_ROUNDS", "STEPS", "build", "status"]
