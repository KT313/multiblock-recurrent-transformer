# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build`` / ``status``: execute a :func:`planner.plan` (tokenizer, then every source: download -> build) and
report it. Task 9 rewrites this module around a readable top-level ``prepare()``; until then the structure is the
pre-restructure one minus the mixture and validation items.

Per source (both kinds) the estimate -> measured refinement loop runs download -> build until the processed tokens
reach the budget or the loader is exhausted (at most ``max_rounds`` rounds, then an error); a source used only for
validation is downloaded once with its ``rows``. ``github_code`` sources of one repo download together in a single
pass over its files (one work item per repo). Folders whose manifest is stale or whose shards do not verify are
removed before their step reruns.

The tokenizer is prepared first (downloads count tokens with it); every other work item then runs in a thread of a
bounded pool so that downloads (network) and builds (CPU) of *different* items overlap: at most
``max_parallel_downloads`` items download and at most ``num_workers`` items build at any time (:class:`_Slots`).
Each item is its own download -> build sequence writing only its own folders, so the files on disk do not depend on
the interleaving. Every failure propagates after logging which item failed; the other items stop at their next shard
(``_Slots.should_stop`` is handed to the steps) and nothing is swallowed; the same happens on Ctrl-C, with everything
published so far kept on disk.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, check_stop
from data_preparation.lib.build.lock import build_lock
from data_preparation.lib.build.planner import Plan, SourcePlan, plan, rows_for_budget, stage_dir, stage_problems
from data_preparation.lib.log import get_logger
from data_preparation.lib.sources import github_code_repo_key
from data_preparation.lib.stages import (
    build_source,
    download,
    download_github_code_group,
    prepare_tokenizer,
    truncate_raw_to_good_prefix,
)
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.ui.dashboard import progress

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "process")  # task 9 renames `process` -> `build`
DEFAULT_MAX_ROUNDS = 5
DEFAULT_MAX_PARALLEL_DOWNLOADS = 2
DEFAULT_NUM_WORKERS = 2  # items built at a time; also the pool size of each decontamination / minhash pass


@dataclass
class _Slots:
    """The bounded concurrency of one ``build``: a download slot and a build slot are held for the duration of a
    step; ``stop(reason)`` makes every item stop — at its next step boundary, and inside a running step at its next
    shard (the steps take ``should_stop``)."""

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


def status(cfg: DatasetConfig, layout: DatasetLayout, *, log_summary: bool = True) -> Plan:
    """The plan for ``cfg`` under ``layout``, logged as a summary (warnings for exhausted sources); ``log_summary``
    False logs only the warnings (``prepare.py status`` prints the table itself)."""
    result = plan(cfg, layout)
    _log_plan(result, summary=log_summary)
    _warn_overlaps(cfg)
    return result


def _warn_overlaps(cfg: DatasetConfig) -> None:
    for warning in cfg.overlap_warnings():
        log.warning(warning)


def build(
    cfg: DatasetConfig,
    layout: DatasetLayout,
    *,
    sources: list[str] | None = None,
    steps: set[str] | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    hf_token: str | None = None,
    dry_run: bool = False,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_parallel_downloads: int = DEFAULT_MAX_PARALLEL_DOWNLOADS,
) -> Plan:
    """Materialise what the plan says is missing; returns the re-computed plan.

    ``steps`` (subset of :data:`STEPS`) limits which steps run, ``sources`` limits to the named sources. ``dry_run``
    logs the plan and returns it without writing anything. ``max_parallel_downloads`` items download and
    ``num_workers`` items build concurrently (see the module docstring). Holds the dataset directory's build lock
    (``lib/build/lock.py``): a concurrent ``prepare.py build`` / ``train.py`` auto-prepare on the same directory
    fails fast with :class:`BuildLocked`.
    """
    active_steps = set(STEPS) if steps is None else set(steps)
    _check_steps(active_steps)
    _check_sources(cfg, sources)
    selected = None if sources is None else set(sources)

    current = plan(cfg, layout)
    _warn_overlaps(cfg)
    if dry_run:
        log.info("dry run, plan for %s under %s:\n%s", cfg.name, layout.root, current.summary(), extra={"keep": True})
        return current
    log.info("building %s under %s (%d item(s) missing)", cfg.name, layout.root, len(current.missing()))

    with build_lock(layout.root):  # one build per dataset directory, across processes
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
    unknown = set(sources) - set(cfg.sources)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}")


@dataclass
class _WorkItem:
    what: str  # "tokenizer" | "source" | "github_code group" (for the log line on failure)
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
    """Every source that is incomplete and selected, in plan order (the tokenizer is not an item: ``build``
    prepares it before anything else)."""
    items: list[_WorkItem] = []
    wanted_sources = [p for p in current.sources if _wanted(p.name, p.complete, selected)]
    for group in _github_code_groups(cfg, wanted_sources):
        action = partial(_build_github_code_group, cfg, group, layout, active_steps, num_workers, hf_token, max_rounds, slots)
        items.append(_WorkItem("github_code group", ", ".join(p.name for p in group), action))
        wanted_sources = [p for p in wanted_sources if p not in group]
    for source_plan in wanted_sources:
        action = partial(_download_and_build_source, cfg, source_plan, layout, active_steps, num_workers, hf_token, max_rounds, slots)
        items.append(_WorkItem("source", source_plan.name, action))
    return items


def _remove_orphaned_filtered_dirs(layout: DatasetLayout) -> None:
    """Delete ``sources/*/filtered/`` directories left behind by builds from before the length filter moved into
    the build step (they were a third copy of the text; raw and processed are the only copies now)."""
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
    """Run one item; a failure sets ``slots.abort`` (the other items stop at their next step) and propagates."""
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
    """Run every item in a thread pool of ``max_workers`` (the step slots bound the real concurrency); the first
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


def _log_plan(result: Plan, *, summary: bool = True) -> None:
    for item in result.sources:
        if not (item.complete and item.exhausted):
            continue
        if item.reason == "ok":
            log.warning(
                "%s: source exhausted at %d rows (%d tokens, budget %d)",
                item.name, item.rows_present, item.tokens_present, item.budget_tokens,
            )
        else:
            log.warning("%s: %s", item.name, item.reason)
    if summary:
        log.info("dataset status:\n%s", result.summary(), extra={"keep": True})  # keep: printed unwrapped into the scrollback


# --- building the individual items ----------------------------------------------------------------------------------


def _remove_broken_stages(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> None:
    """Delete folders whose manifest is stale or whose shards do not verify, so the step rebuilds — except a raw
    folder with a broken shard, which is truncated to its good prefix (the next download resumes there) rather than
    downloaded again. (Task 7 turns this into the repair step; deleting a raw folder then needs confirmation.)"""
    for stage, problem in stage_problems(cfg, name, layout).items():
        directory = stage_dir(layout, name, stage)
        if stage == "raw" and not problem.endswith("manifest stale"):
            manifest = Manifest.load(directory)
            if manifest is not None and truncate_raw_to_good_prefix(directory, manifest):
                log.warning("%s: %s; kept the %d good shard(s) of %s", name, problem, len(manifest.shards), directory)
                continue
        log.warning("%s: %s; removing %s", name, problem, directory)
        shutil.rmtree(directory)


def _download_and_build_source(
    cfg: DatasetConfig,
    source_plan: SourcePlan,
    layout: DatasetLayout,
    active_steps: set[str],
    num_workers: int,
    hf_token: str | None,
    max_rounds: int,
    slots: _Slots,
) -> None:
    """download -> build, repeated with a refined tokens/row until the processed tokens reach the budget (a source
    used only for validation is downloaded once with its ``rows``)."""
    name, budget = source_plan.name, source_plan.budget_tokens
    _remove_broken_stages(cfg, name, layout)
    rows_needed = source_plan.rows_needed

    for round_index in range(max_rounds):
        raw = None
        if "download" in active_steps:
            with slots.downloading():
                raw = download(cfg, name, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=slots.should_stop)
        refined = _build_round(cfg, name, layout, active_steps, num_workers, raw, budget, round_index, slots)
        if refined is None:
            return
        rows_needed = rows_for_budget(budget, refined)

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
    """`_download_and_build_source` for the `github_code` sources of one repo: each round downloads every source
    that still needs rows in one pass over the repo files, then builds them one by one."""
    budgets = {p.name: p.budget_tokens for p in group}
    rows_needed = {p.name: p.rows_needed for p in group}
    for source_plan in group:
        _remove_broken_stages(cfg, source_plan.name, layout)
    pending = [p.name for p in group]  # sources whose budget is not reached yet

    for round_index in range(max_rounds):
        raws: dict[str, Manifest] = {}
        if "download" in active_steps:
            wanted = {name: rows_needed[name] for name in pending}
            with slots.downloading():
                raws = download_github_code_group(cfg, pending, layout, rows_needed=wanted, hf_token=hf_token, should_stop=slots.should_stop)
        still_pending: list[str] = []
        for name in pending:
            refined = _build_round(cfg, name, layout, active_steps, num_workers, raws.get(name), budgets[name], round_index, slots)
            if refined is not None:
                rows_needed[name] = rows_for_budget(budgets[name], refined)
                still_pending.append(name)
        pending = still_pending
        if not pending:
            return

    raise RuntimeError(f"{', '.join(pending)}: token budget not reached after {max_rounds} rounds")


def _build_round(
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
    """The build after one download round of ``name`` (``raw``: its raw manifest, None without a download step).
    Returns None when the source is done (budget reached, exhausted, used only for validation, or nothing more to
    do) or the refined tokens/row for the next round."""
    if "process" not in active_steps:
        return None
    with slots.processing():
        processed = build_source(cfg, name, layout, num_workers=num_workers, should_stop=slots.should_stop)
    if not cfg.used_in_train(name):
        return None  # sized by `rows`, not by a token budget: one round

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
    log.info("%s: round %d: %d of %d tokens after the build, refining to %.1f tokens/row", name, round_index + 1, tokens, budget, tokens_per_row)
    return tokens_per_row


__all__ = ["DEFAULT_MAX_PARALLEL_DOWNLOADS", "DEFAULT_MAX_ROUNDS", "STEPS", "BuildAborted", "build", "status"]
