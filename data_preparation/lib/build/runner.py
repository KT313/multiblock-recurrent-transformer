# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build`` / ``status``: execute a :func:`planner.plan` (tokenizer -> pretrain sources -> validation sources -> instruct mixtures)
and report it.

Per pretrain source the estimate -> measured refinement loop runs download -> process until the
processed tokens reach the budget or the loader is exhausted (at most ``max_rounds`` rounds, then an error);
``github_code`` sources of one repo download together in a single pass over its files (one work item per repo). A
mixture is rebuilt until none of its sources is short. Stage directories whose manifest is stale or whose shards
do not verify are removed before their stage reruns.

The tokenizer is prepared first (downloads count tokens with it); every other work item then runs in a thread of a
bounded pool so that downloads (network) and processing (CPU) of *different* items overlap: at most
``max_parallel_downloads`` items download and at most ``num_workers`` items process at any time (:class:`_Slots`).
Each item is still its own download -> process sequence writing only its own directories, so the files on disk do
not depend on the interleaving. Every stage failure propagates after logging which item failed; the other items
stop at their next shard (``_Slots.should_stop`` is handed to the stages) and nothing is swallowed; the same
happens on Ctrl-C, with everything published so far kept on disk.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from data_preparation.lib.abort import BuildAborted, check_stop
from data_preparation.lib.schema.dataset_config import DatasetConfig
from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.ui.dashboard import progress
from data_preparation.lib.build.planner import InstructMixturePlan, Plan, SourcePlan, plan, rows_for_budget, stage_problems
from data_preparation.lib.sources import github_code_repo_key
from data_preparation.lib.stages import (
    build_instruct_mixture,
    download,
    download_github_code_group,
    prepare_tokenizer,
    process,
    truncate_raw_to_good_prefix,
    validation,
)
from data_preparation.lib.storage.manifest import Manifest

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "process", "validation", "instruct_mixtures")
DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_PARALLEL_DOWNLOADS = 2


@dataclass
class _Slots:
    """The bounded concurrency of one ``build``: a download slot and a processing slot are held for the duration
    of a stage; ``stop(reason)`` makes every item stop — at its next stage boundary, and inside a running stage at
    its next shard (the stages take ``should_stop``)."""

    download: threading.BoundedSemaphore
    process: threading.BoundedSemaphore
    abort: threading.Event
    reason: str = "another item failed"

    @classmethod
    def create(cls, max_parallel_downloads: int, num_workers: int) -> _Slots:
        if max_parallel_downloads < 1 or num_workers < 1:
            raise ValueError(f"max_parallel_downloads and num_workers must be >= 1, got {max_parallel_downloads} and {num_workers}")
        return cls(threading.BoundedSemaphore(max_parallel_downloads), threading.BoundedSemaphore(num_workers), threading.Event())

    def stop(self, reason: str) -> None:
        """Request every item to stop (the first reason wins)."""
        if not self.abort.is_set():
            self.reason = reason
            self.abort.set()

    def should_stop(self) -> bool:
        return self.abort.is_set()

    @contextmanager
    def downloading(self) -> Iterator[None]:
        with self._held(self.download):
            yield

    @contextmanager
    def processing(self) -> Iterator[None]:
        with self._held(self.process):
            yield

    @contextmanager
    def _held(self, slot: threading.BoundedSemaphore) -> Iterator[None]:
        check_stop(self.should_stop)
        with slot:
            check_stop(self.should_stop)
            yield


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
    max_parallel_downloads: int = DEFAULT_MAX_PARALLEL_DOWNLOADS,
) -> Plan:
    """Materialise what the plan says is missing; returns the re-computed plan.

    ``steps`` (subset of :data:`STEPS`) limits which stages run, ``sources`` limits to the named sources / mixtures.
    ``dry_run`` logs the plan and returns it without writing anything. ``max_parallel_downloads`` items download
    and ``num_workers`` items process concurrently (see the module docstring).
    """
    active_steps = set(STEPS) if steps is None else set(steps)
    _check_steps(active_steps)
    _check_sources(cfg, sources)
    selected = None if sources is None else set(sources)

    current = plan(cfg, layout)
    if dry_run:
        log.info("dry run, plan for %s under %s:\n%s", cfg.name, layout.root, current.summary(), extra={"keep": True})
        return current
    log.info("building %s under %s (%d item(s) missing)", cfg.name, layout.root, len(current.missing()))

    _remove_orphaned_filtered_dirs(layout)
    slots = _Slots.create(max_parallel_downloads, num_workers)
    if "tokenizer" in active_steps and not current.tokenizer_complete:
        _run(_WorkItem("tokenizer", cfg.tokenizer.name, lambda: prepare_tokenizer(cfg, layout)), slots)
    items = _work_items(cfg, layout, current, active_steps, selected, num_workers, hf_token, max_rounds, slots)
    _run_all(items, slots, max_workers=max_parallel_downloads + num_workers)

    final = plan(cfg, layout)
    _log_plan(final)
    return final


# --- argument checks and work list ------------------------------------------------------------------------------------


def _check_steps(active_steps: set[str]) -> None:
    unknown = active_steps - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; expected a subset of {STEPS}")


def _check_sources(cfg: DatasetConfig, sources: list[str] | None) -> None:
    if sources is None:
        return
    unknown = set(sources) - set(cfg.sources) - set(cfg.instruct_mixtures)
    if unknown:
        raise ValueError(f"unknown sources/instruct mixtures {sorted(unknown)}")


@dataclass
class _WorkItem:
    what: str  # "tokenizer" | "source" | "validation" | "instruct_mixture" (for the log line on failure)
    name: str
    action: Callable[[], object]


def _work_items(
    cfg: DatasetConfig,
    layout: DatasetLayout,
    current: Plan,
    active_steps: set[str],
    selected: set[str] | None,
    num_workers: int,
    hf_token: str | None,
    max_rounds: int,
    slots: _Slots,
) -> list[_WorkItem]:
    """Every source / validation / instruct mixture that is incomplete, selected and has an active step, in plan
    order (the tokenizer is not an item: ``build`` prepares it before anything else)."""
    items: list[_WorkItem] = []

    wanted_sources = [p for p in current.sources if _wanted(p.name, p.complete, selected)]
    for group in _github_code_groups(cfg, wanted_sources):
        action = partial(_build_github_code_group, cfg, group, layout, active_steps, num_workers, hf_token, max_rounds, slots)
        items.append(_WorkItem("github_code group", ", ".join(p.name for p in group), action))
        wanted_sources = [p for p in wanted_sources if p not in group]
    for source_plan in wanted_sources:
        action = partial(_build_pretrain_source, cfg, source_plan, layout, active_steps, num_workers, hf_token, max_rounds, slots)
        items.append(_WorkItem("source", source_plan.name, action))

    if "validation" in active_steps:
        for validation_plan in current.validations:
            if _wanted(validation_plan.name, validation_plan.complete, selected):
                action = partial(_build_validation, cfg, validation_plan, layout, slots)
                items.append(_WorkItem("validation", validation_plan.name, action))

    if "instruct_mixtures" in active_steps:
        for mixture_plan in current.instruct_mixtures:
            if _wanted(mixture_plan.name, mixture_plan.complete, selected):
                action = partial(_build_instruct_mixture, cfg, mixture_plan, layout, hf_token, max_rounds, slots)
                items.append(_WorkItem("instruct_mixture", mixture_plan.name, action))

    return items


def _remove_orphaned_filtered_dirs(layout: DatasetLayout) -> None:
    """Delete ``sources/*/filtered/`` directories left behind by builds from before the length filter moved into
    ``process`` (they were a third copy of the text; raw and processed are the only copies now)."""
    for directory in sorted((layout.root / "sources").glob("*/filtered")):
        if directory.is_dir():
            log.warning("removing orphaned %s (the filtered stage no longer exists)", directory)
            shutil.rmtree(directory)


def _github_code_groups(cfg: DatasetConfig, plans: list[SourcePlan]) -> list[list[SourcePlan]]:
    """The `github_code` sources among ``plans`` that share a repo (:func:`github_code_repo_key`), two or more per
    group, in plan order; a single source of a repo goes through the ordinary per-source path."""
    groups: dict[tuple[str | None, str | None, str], list[SourcePlan]] = {}
    for source_plan in plans:
        source = cfg.sources[source_plan.name]
        if source.loader == "github_code":
            groups.setdefault(github_code_repo_key(source), []).append(source_plan)
    return [group for group in groups.values() if len(group) >= 2]


def _wanted(name: str, complete: bool, selected: set[str] | None) -> bool:
    """Whether a planned item is to be built: selected (or nothing selected) and not already complete."""
    if selected is not None and name not in selected:
        return False
    if complete:
        log.info("%s: complete, skipping", name)
        return False
    return True


def _run(item: _WorkItem, slots: _Slots) -> None:
    """Run one item; a failure sets ``slots.abort`` (the other items stop at their next stage) and propagates."""
    try:
        item.action()
    except BuildAborted:
        log.info("%s %s stopped: %s", item.what, item.name, slots.reason)
        raise
    except BaseException:
        slots.stop(f"{item.what} {item.name} failed")
        log.exception("%s %s failed", item.what, item.name)
        raise


def _run_all(items: list[_WorkItem], slots: _Slots, *, max_workers: int) -> None:
    """Run every item in a thread pool of ``max_workers`` (the stage slots bound the real concurrency); the first
    failure (or an interrupt) cancels the items not started yet, lets the running ones stop at their next shard
    and is re-raised."""
    if not items:
        return
    running: list[str] = []
    lock = threading.Lock()
    with progress(total=len(items), desc="items", unit="item") as bar:

        def run_and_track(item: _WorkItem) -> None:
            with lock:
                running.append(item.name)
                bar.set_postfix({"running": ", ".join(running)}, refresh=False)
            try:
                _run(item, slots)
            finally:
                with lock:
                    running.remove(item.name)
                    bar.set_postfix({"running": ", ".join(running)}, refresh=False)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="build") as pool:
            futures: dict[Future[None], _WorkItem] = {pool.submit(run_and_track, item): item for item in items}
            failures: list[BaseException] = []
            try:
                for future in as_completed(futures):
                    if future.cancelled():
                        continue  # never started: cancelled after another item failed
                    error = future.exception()
                    if error is None:
                        bar.update(1)
                        continue
                    slots.stop(f"{futures[future].what} {futures[future].name} failed")
                    for other in futures:
                        other.cancel()
                    failures.append(error)
            except BaseException:  # KeyboardInterrupt while waiting: the running items stop at their next shard
                slots.stop("build interrupted")
                for future in futures:
                    future.cancel()
                raise
    # every item has finished or been cancelled: re-raise the failure that caused it (items that merely stopped
    # because of it raised BuildAborted, which is only reported when nothing else went wrong)
    for error in failures:
        if not isinstance(error, BuildAborted):
            raise error
    if failures:
        raise failures[0]


def _log_plan(result: Plan) -> None:
    for item in result.sources + result.validations:
        if not (item.complete and item.exhausted):
            continue
        if item.reason == "ok":
            log.warning(
                "%s: source exhausted at %d rows (%d tokens, budget %d)",
                item.name, item.rows_present, item.tokens_present, item.budget_tokens,
            )
        else:
            log.warning("%s: %s", item.name, item.reason)
    log.info("dataset status:\n%s", result.summary(), extra={"keep": True})  # keep: printed unwrapped into the scrollback


# --- building the individual items ----------------------------------------------------------------------------------


def _remove_broken_stages(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> None:
    """Delete stage directories whose manifest is stale or whose shards do not verify, so the stage rebuilds — except
    a raw directory with a broken shard, which is truncated to its good prefix (the next download resumes there)
    rather than downloaded again."""
    for stage, problem in stage_problems(cfg, name, layout).items():
        directory = layout.validation_dir(name) if stage == "validation" else layout.source_dir(name, stage)
        if stage == "raw" and not problem.endswith("manifest stale"):
            manifest = Manifest.load(directory)
            if manifest is not None and truncate_raw_to_good_prefix(directory, manifest):
                log.warning("%s: %s; kept the %d good shard(s) of %s", name, problem, len(manifest.shards), directory)
                continue
        log.warning("%s: %s; removing %s", name, problem, directory)
        shutil.rmtree(directory)


def _build_pretrain_source(
    cfg: DatasetConfig,
    source_plan: SourcePlan,
    layout: DatasetLayout,
    active_steps: set[str],
    num_workers: int,
    hf_token: str | None,
    max_rounds: int,
    slots: _Slots,
) -> None:
    """download -> process, repeated with a refined tokens/row until the processed tokens reach the budget."""
    name, budget = source_plan.name, source_plan.budget_tokens
    _remove_broken_stages(cfg, name, layout)
    tokens_per_row = source_plan.tokens_per_row

    for round_index in range(max_rounds):
        raw = None
        if "download" in active_steps:
            rows_needed = rows_for_budget(budget, tokens_per_row)
            with slots.downloading():
                raw = download(cfg, name, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=slots.should_stop)
        refined = _process_round(cfg, name, layout, active_steps, num_workers, raw, budget, round_index, slots)
        if refined is None:
            return
        tokens_per_row = refined

    raise RuntimeError(f"{name}: token budget {budget} not reached after {max_rounds} rounds")


def _build_github_code_group(
    cfg: DatasetConfig,
    group: list[SourcePlan],
    layout: DatasetLayout,
    active_steps: set[str],
    num_workers: int,
    hf_token: str | None,
    max_rounds: int,
    slots: _Slots,
) -> None:
    """`_build_pretrain_source` for the `github_code` sources of one repo: each round downloads every source that
    still needs rows in one pass over the repo files, then processes them one by one."""
    budgets = {p.name: p.budget_tokens for p in group}
    tokens_per_row = {p.name: p.tokens_per_row for p in group}
    for source_plan in group:
        _remove_broken_stages(cfg, source_plan.name, layout)
    pending = [p.name for p in group]  # sources whose budget is not reached yet

    for round_index in range(max_rounds):
        raws: dict[str, Manifest] = {}
        if "download" in active_steps:
            rows_needed = {name: rows_for_budget(budgets[name], tokens_per_row[name]) for name in pending}
            with slots.downloading():
                raws = download_github_code_group(cfg, pending, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=slots.should_stop)
        still_pending: list[str] = []
        for name in pending:
            refined = _process_round(cfg, name, layout, active_steps, num_workers, raws.get(name), budgets[name], round_index, slots)
            if refined is not None:
                tokens_per_row[name] = refined
                still_pending.append(name)
        pending = still_pending
        if not pending:
            return

    raise RuntimeError(f"{', '.join(pending)}: token budget not reached after {max_rounds} rounds")


def _process_round(
    cfg: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    active_steps: set[str],
    num_workers: int,
    raw: Manifest | None,
    budget: int,
    round_index: int,
    slots: _Slots,
) -> float | None:
    """process after one download round of ``name`` (``raw``: its raw manifest, None without a download step).
    Returns None when the source is done (budget reached, exhausted, or nothing more to do) or the refined
    tokens/row for the next round."""
    if "process" not in active_steps:
        return None
    with slots.processing():
        processed = process(cfg, name, layout, num_workers=num_workers, should_stop=slots.should_stop)

    tokens = processed.tokens() or 0
    if tokens >= budget:
        return None
    exhausted = raw is not None and bool(raw.extra.get("exhausted"))
    if exhausted or raw is None:
        # nothing more can be fetched; the source stays below its budget
        why = "exhausted" if exhausted else "no download step"
        log.warning("%s: %s at %d tokens, budget is %d", name, why, tokens, budget)
        return None

    tokens_per_row = max(tokens / max(raw.rows(), 1), 1.0)  # floor: never more than `budget` rows per round
    log.info("%s: round %d: %d of %d tokens after processing, refining to %.1f tokens/row", name, round_index + 1, tokens, budget, tokens_per_row)
    return tokens_per_row


def _build_validation(cfg: DatasetConfig, validation_plan: SourcePlan, layout: DatasetLayout, slots: _Slots) -> None:
    name = validation_plan.name
    _remove_broken_stages(cfg, name, layout)  # `rows` is part of the source hash, so a stale manifest covers row changes
    with slots.downloading():
        validation(cfg, name, layout)


def _build_instruct_mixture(
    cfg: DatasetConfig, mixture_plan: InstructMixturePlan, layout: DatasetLayout, hf_token: str | None, max_rounds: int, slots: _Slots
) -> None:
    """Download every source's share, build the mixture, and repeat with refined tokens/row while a source is short."""
    name, budget = mixture_plan.name, mixture_plan.budget_tokens
    mixture = cfg.instruct_mixtures[name]

    for src in mixture.sources:
        _remove_broken_stages(cfg, src, layout)
    if not mixture_plan.current:
        _remove_stale_mixture_splits(cfg, name, layout)

    tokens_per_row = {src: float(cfg.sources[src].tokens_per_row_estimate) for src in mixture.sources}
    short: dict[str, object] = {}
    for round_index in range(max_rounds):
        exhausted: set[str] = set()
        with slots.downloading():
            for src, share in mixture.sources.items():
                rows_needed = rows_for_budget(budget * share, tokens_per_row[src])
                raw = download(cfg, src, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=slots.should_stop)
                if raw.extra.get("exhausted"):
                    exhausted.add(src)

        with slots.processing():
            train = build_instruct_mixture(cfg, name, layout, budget_tokens=budget, should_stop=slots.should_stop)["train"]
        short_sources = train.extra["short_sources"]
        short = {src: info for src, info in short_sources.items() if src not in exhausted}
        if not short:
            for src in exhausted & set(short_sources):
                log.warning("%s: source %s exhausted before its share of the budget", name, src)
            return

        for src in short:
            tokens_per_row[src] = max(train.extra["tokens_per_row"][src], 1.0)
        log.info("%s: round %d: short sources %s, refining tokens/row", name, round_index + 1, sorted(short))

    raise RuntimeError(f"{name}: sources {sorted(short)} still short after {max_rounds} rounds")


def _remove_stale_mixture_splits(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> None:
    for split in INSTRUCT_MIXTURE_SPLITS:
        directory = layout.instruct_mixture_dir(cfg.name, name, split)
        if directory.exists():
            _remove_dir(directory)


def _remove_dir(directory: Path) -> None:
    log.warning("removing stale %s", directory)
    shutil.rmtree(directory)


__all__ = ["DEFAULT_MAX_PARALLEL_DOWNLOADS", "DEFAULT_MAX_ROUNDS", "STEPS", "BuildAborted", "build", "status"]
