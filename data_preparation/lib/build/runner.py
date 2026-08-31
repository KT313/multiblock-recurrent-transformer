# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``prepare`` / ``status``: the top-level data pipeline, readable top to bottom.

``prepare`` = tokenizer → repair → (download → build) rounds → report, under the dataset directory's build lock::

    prepare_tokenizer                                  tokenizers/<name>/ (downloads count tokens with it)
    repair_broken_and_stale_folders                    truncate broken raw, delete stale processed, confirm before any raw
                                                       folder is deleted (lib/build/repair.py)
    for round in 1..MAX_ROUNDS:
        plan_downloads                                 rows still missing per source (lib/build/planner.py)
        download_all_missing_rows                      parallel, per-shard resumable, writes sources/<name>/raw only
        build_all_pending_raw_shards                   parallel, per-shard resumable, writes processed/<name> only
        stop when every source serves its budget, or when nothing more can be fetched
    summarize_dataset_state                            the status table

The two parallel helpers are the only places with thread-pool code: ``max_parallel_downloads`` download jobs
(the ``github_code`` sources of one repo form one job, read in a single pass over the repo files) and
``num_workers`` build jobs run at a time. A failing job stops every running job at its next shard (the steps take
``should_stop``; :class:`StopFlag`) and is re-raised after they stopped — a failed source is a failed build, never a
silently smaller dataset. Ctrl-C while waiting does the same and raises :class:`BuildAborted` (``prepare.py`` exits
130); everything published so far is kept and the next run resumes at shard granularity.

``status`` is read-only: the repair step's dry report ("would repair: …") plus the same status table.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from data_preparation.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, StopCheck, check_stop
from data_preparation.lib.build.lock import build_lock
from data_preparation.lib.build.planner import (
    DatasetReport,
    DownloadPlan,
    every_source_satisfies_its_budget,
    plan_downloads,
    sources_with_pending_raw_shards,
    summarize_dataset_state,
)
from data_preparation.lib.build.repair import Confirm, RepairReport, repair_broken_and_stale_folders
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources import github_code_repo_key
from data_preparation.lib.stages import build_source, download, download_github_code_group, prepare_tokenizer
from data_preparation.lib.ui.dashboard import progress, set_status

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "build")
MAX_ROUNDS = 5  # download → build rounds; a source still short afterwards is reported, not looped on forever
DEFAULT_MAX_PARALLEL_DOWNLOADS = 2
DEFAULT_NUM_WORKERS = 2  # sources built at a time; also the pool size of each decontamination / minhash pass


# --- prepare / status ------------------------------------------------------------------------------------------------


def prepare(
    config_path: str | Path,
    dataset_dir: str | Path,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_parallel_downloads: int = DEFAULT_MAX_PARALLEL_DOWNLOADS,
    assume_yes: bool,
    dry_run: bool = False,
    steps: Iterable[str] = STEPS,
    sources: Iterable[str] | None = None,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
    confirm: Confirm | None = None,
) -> DatasetReport:
    """Materialise the dataset config at ``config_path`` under ``dataset_dir`` (see the module docstring) and return
    its status.

    ``assume_yes`` confirms the deletion of stale / outdated raw folders without asking; otherwise ``confirm`` (or
    the terminal) is asked once and a refusal raises :class:`ConfirmationRequired` before anything is changed.
    ``dry_run`` reports what the repair and the first round would do and writes nothing (not even the lock file).
    ``steps`` (a subset of :data:`STEPS`) and ``sources`` restrict the work — and the satisfaction check — to the
    named steps / sources; the returned report always covers the whole config.
    """
    config = load_dataset_config(config_path)
    layout = DatasetLayout(Path(dataset_dir))
    active_steps = checked_steps(steps)
    selected = checked_sources(config, sources)
    check_worker_counts(num_workers, max_parallel_downloads)
    warn_about_overlaps(config)

    with build_lock(layout.root) if not dry_run else nullcontext():
        if "tokenizer" in active_steps and not dry_run:
            prepare_tokenizer(config, layout)
        repair_report = repair_broken_and_stale_folders(config, layout, assume_yes=assume_yes, dry_run=dry_run, confirm=confirm)
        log_repair(repair_report)
        for round_number in range(1, MAX_ROUNDS + 1):
            download_plan = plan_downloads(config, layout, sources=selected)
            if dry_run:
                log.info("dry run, downloads planned:\n%s", download_plan.describe(), extra={"keep": True})
                break
            log.info("round %d: %s", round_number, download_plan.summary())
            set_status(round=f"{round_number}/{MAX_ROUNDS}", step="download")
            if "download" in active_steps:
                download_all_missing_rows(download_plan, config, layout, max_parallel_downloads=max_parallel_downloads, hf_token=hf_token, should_stop=should_stop)
            set_status(step="build")
            if "build" in active_steps:
                build_all_pending_raw_shards(config, layout, num_workers=num_workers, sources=selected, should_stop=should_stop)
            if every_source_satisfies_its_budget(config, layout, sources=selected):
                break
            if not another_round_can_fetch_more(config, layout, active_steps, selected):
                break  # still short, but nothing left to download: the report names the sources
        set_status(step="status")
        report = summarize_dataset_state(config, layout)
    log_report(report)
    return report


def status(config_path: str | Path, dataset_dir: str | Path) -> DatasetReport:
    """Read-only: what the repair step would do ("would repair: …"; such sources count as incomplete) and the
    status table, logged and returned."""
    config = load_dataset_config(config_path)
    layout = DatasetLayout(Path(dataset_dir))
    warn_about_overlaps(config)
    repair_report = repair_broken_and_stale_folders(config, layout, assume_yes=False, dry_run=True)
    if repair_report.actions:
        log.warning("would repair:\n%s", repair_report.describe(), extra={"keep": True})
    report = summarize_dataset_state(config, layout, needs_repair=[action.source for action in repair_report.actions])
    log_report(report)
    return report


# --- the two parallel helpers --------------------------------------------------------------------------------------------


def download_all_missing_rows(
    download_plan: DownloadPlan,
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    max_parallel_downloads: int,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
) -> None:
    """Download the rows :func:`plan_downloads` found missing, ``max_parallel_downloads`` sources at a time. The
    ``github_code`` sources of one repo are one job (:func:`download_github_code_group`: a single pass over the repo
    files), every other source its own :func:`download` call. Writes ``sources/<name>/raw`` only."""
    run_jobs(download_jobs(download_plan, config, layout, hf_token), max_workers=max_parallel_downloads, description="downloads", should_stop=should_stop)


def build_all_pending_raw_shards(
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    num_workers: int,
    sources: Iterable[str] | None = None,
    should_stop: StopCheck | None = None,
) -> None:
    """Build every source (all, or ``sources``) whose raw shards are not all covered by its processed manifest,
    ``num_workers`` sources at a time (:func:`build_source`, resumable per raw shard). Writes ``processed/<name>``
    only."""
    names = sources_with_pending_raw_shards(config, layout, sources)
    jobs = [build_source_job(config, name, layout, num_workers) for name in names]
    run_jobs(jobs, max_workers=num_workers, description="builds", should_stop=should_stop)


# --- jobs --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One unit of work of a parallel helper; ``action`` takes the stop check the steps poll between shards."""

    what: str  # "source" | "github_code group" (for the log line on failure)
    name: str
    action: Callable[[StopCheck], object]


def download_jobs(download_plan: DownloadPlan, config: DatasetConfig, layout: DatasetLayout, hf_token: str | None) -> list[Job]:
    """One job per source with rows to fetch — the ``github_code`` sources of one repo grouped into one."""
    to_fetch = download_plan.to_fetch()
    rows_needed = {source.name: source.rows_needed for source in to_fetch}
    jobs: list[Job] = []
    grouped: set[str] = set()
    for names in github_code_groups(config, list(rows_needed)):
        jobs.append(github_code_group_job(config, names, layout, {name: rows_needed[name] for name in names}, hf_token))
        grouped.update(names)
    for name, needed in rows_needed.items():
        if name not in grouped:
            jobs.append(download_source_job(config, name, layout, needed, hf_token))
    return jobs


def github_code_groups(config: DatasetConfig, names: list[str]) -> list[list[str]]:
    """The ``github_code`` sources among ``names`` that share a repo (:func:`github_code_repo_key`), two or more
    per group, in config order; a single source of a repo goes through the ordinary per-source download."""
    groups: dict[tuple[str | None, str | None, str], list[str]] = {}
    for name in names:
        source = config.sources[name]
        if source.loader == "github_code":
            groups.setdefault(github_code_repo_key(source), []).append(name)
    return [group for group in groups.values() if len(group) >= 2]


def download_source_job(config: DatasetConfig, name: str, layout: DatasetLayout, rows_needed: int, hf_token: str | None) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download(config, name, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop)

    return Job("source", name, action)


def github_code_group_job(config: DatasetConfig, names: list[str], layout: DatasetLayout, rows_needed: dict[str, int], hf_token: str | None) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download_github_code_group(config, names, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop)

    return Job("github_code group", ", ".join(names), action)


def build_source_job(config: DatasetConfig, name: str, layout: DatasetLayout, num_workers: int) -> Job:
    def action(should_stop: StopCheck) -> object:
        return build_source(config, name, layout, num_workers=num_workers, should_stop=should_stop)

    return Job("source", name, action)


# --- running jobs in a pool ---------------------------------------------------------------------------------------------


class StopFlag:
    """The shared stop request of one pool of jobs: ``stop(reason)`` makes every job stop at its next shard (the
    steps poll :meth:`should_stop`); the first reason wins. An outer ``should_stop`` (Ctrl-C handling of the caller,
    a training run shutting down) is polled too."""

    def __init__(self, outer: StopCheck | None = None) -> None:
        self._event = threading.Event()
        self._outer = outer
        self.reason = ""

    def stop(self, reason: str) -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def should_stop(self) -> bool:
        return self._event.is_set() or (self._outer is not None and self._outer())


def run_jobs(jobs: list[Job], *, max_workers: int, description: str, should_stop: StopCheck | None = None) -> None:
    """Run ``jobs`` in a thread pool of ``max_workers``. The first failure stops the running jobs at their next
    shard (:class:`StopFlag`), cancels the jobs not started yet and is re-raised once every job has finished; a
    ``KeyboardInterrupt`` while waiting does the same and raises :class:`BuildAborted` (published shards are kept).
    The pool's bar is the summary task of the dashboard panel named ``description`` (the jobs' own bars are its rows)."""
    if not jobs:
        return
    flag = StopFlag(should_stop)
    running = RunningJobs()
    with (
        progress(total=len(jobs), desc=description, unit="job", panel=description, summary=True) as bar,
        ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=description) as pool,
    ):
        futures = {pool.submit(run_job, job, flag, running, bar): job for job in jobs}
        failures = wait_for_jobs(futures, flag, bar)  # the pool's exit waits for the running jobs to stop
    raise_first_failure(failures)


class RunningJobs:
    """The names of the jobs running right now (the pool bar's postfix)."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._lock = threading.Lock()

    def add(self, name: str) -> str:
        with self._lock:
            self._names.append(name)
            return ", ".join(self._names)

    def remove(self, name: str) -> str:
        with self._lock:
            self._names.remove(name)
            return ", ".join(self._names)


def run_job(job: Job, flag: StopFlag, running: RunningJobs, bar: Progress) -> None:
    """Run one job in a pool thread; a failure raises the flag (the other jobs stop at their next shard, queued
    jobs never start), is logged with its traceback and propagates to :func:`wait_for_jobs`."""
    bar.set_postfix({"running": running.add(job.name)}, refresh=False)
    try:
        check_stop(flag.should_stop)  # a job dequeued after a failure or an interrupt does not start
        job.action(flag.should_stop)
    except BuildAborted:
        log.info("%s %s stopped: %s", job.what, job.name, flag.reason or "stop requested")
        raise
    except BaseException:
        flag.stop(f"{job.what} {job.name} failed")
        log.exception("%s %s failed", job.what, job.name)
        raise
    finally:
        bar.set_postfix({"running": running.remove(job.name)}, refresh=False)


def wait_for_jobs(futures: dict[Future[None], Job], flag: StopFlag, bar: Progress) -> list[BaseException]:
    """Wait for every future; returns the failures (a ``KeyboardInterrupt`` becomes a :class:`BuildAborted`). Jobs
    not started yet are cancelled on the first failure or interrupt; the running ones stop at their next shard."""
    failures: list[BaseException] = []
    try:
        for future in as_completed(futures):
            if future.cancelled():
                continue  # never started: cancelled after another job failed
            error = future.exception()
            if error is None:
                bar.update(1)
                continue
            flag.stop(f"{futures[future].what} {futures[future].name} failed")
            cancel_all(futures)
            failures.append(error)
    except KeyboardInterrupt:
        flag.stop("interrupted")
        cancel_all(futures)
        log.warning("interrupted; the running jobs stop at their next shard, everything published so far is kept")
        failures.append(BuildAborted("interrupted; everything published so far is kept, rerun to resume"))
    return failures


def cancel_all(futures: dict[Future[None], Job]) -> None:
    for future in futures:
        future.cancel()


def raise_first_failure(failures: list[BaseException]) -> None:
    """Re-raise the failure that caused the stop: jobs that merely stopped because of it raised
    :class:`BuildAborted`, which is only raised when nothing else went wrong."""
    for error in failures:
        if not isinstance(error, BuildAborted):
            raise error
    if failures:
        raise failures[0]


# --- small checks and log lines -----------------------------------------------------------------------------------------


def checked_steps(steps: Iterable[str]) -> set[str]:
    active = set(steps)
    unknown = active - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; expected a subset of {STEPS}")
    return active


def checked_sources(config: DatasetConfig, sources: Iterable[str] | None) -> list[str] | None:
    """``sources`` as a list (None for every source); unknown names are an error."""
    if sources is None:
        return None
    selected = list(sources)
    unknown = set(selected) - set(config.sources)
    if unknown:
        raise ValueError(f"unknown sources {sorted(unknown)}")
    return selected


def check_worker_counts(num_workers: int, max_parallel_downloads: int) -> None:
    if num_workers < 1 or max_parallel_downloads < 1:
        raise ValueError(f"num_workers and max_parallel_downloads must be >= 1, got {num_workers} and {max_parallel_downloads}")


def another_round_can_fetch_more(config: DatasetConfig, layout: DatasetLayout, active_steps: set[str], selected: list[str] | None) -> bool:
    """Whether a further round would download anything: the download step is active and, after this round, some
    selected source still has rows to fetch (a loader that returned fewer rows than asked without being exhausted)."""
    if "download" not in active_steps:
        return False
    return plan_downloads(config, layout, sources=selected).total_rows_to_fetch() > 0


def warn_about_overlaps(config: DatasetConfig) -> None:
    for warning in config.overlap_warnings():
        log.warning(warning)


def log_repair(report: RepairReport) -> None:
    if report.actions:
        log.info("repair:\n%s", report.describe(), extra={"keep": True})


def log_report(report: DatasetReport) -> None:
    """The status table (kept in the scrollback) plus one warning per exhausted or unsatisfied source."""
    for source in report.sources:
        if source.satisfied and source.exhausted:
            log.warning("%s: source exhausted (%s); the training sampler cycles the rows on disk", source.name, source.reason)
        elif not source.satisfied:
            log.warning("%s: %s", source.name, source.reason)
    log.info("dataset status:\n%s", report.describe(), extra={"keep": True})  # keep: printed unwrapped into the scrollback


__all__ = [
    "DEFAULT_MAX_PARALLEL_DOWNLOADS",
    "DEFAULT_NUM_WORKERS",
    "MAX_ROUNDS",
    "STEPS",
    "BuildAborted",
    "Job",
    "StopFlag",
    "build_all_pending_raw_shards",
    "download_all_missing_rows",
    "prepare",
    "run_jobs",
    "status",
]
