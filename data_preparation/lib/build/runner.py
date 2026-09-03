# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``prepare`` / ``status``: the top-level data pipeline, readable top to bottom.

``prepare`` runs tokenizer, repair, (download + build) rounds, report, under the dataset directory's build lock::

    prepare_tokenizer                                  tokenizers/<name>/ (downloads count tokens with it)
    repair_broken_and_stale_folders                    truncate broken raw, delete stale processed, confirm before any raw
                                                       folder is deleted (lib/build/repair.py)
    for round in 1..MAX_ROUNDS:
        plan_downloads                                 rows still missing per source (lib/build/planner.py: one
                                                       SourceLedger per source answers both "what to download" and
                                                       "is it done", so a round 2 tops a short source up)
        download_and_build_missing                     downloads (sources/<name>/raw) and builds (processed/<name>) at the
                                                       same time: a source is built as soon as its download finished
        stop when every source serves its budget, or when nothing more can be fetched
    assess_dataset_state                               the status table and its verdict, counting the repairs this
                                                       run left undone (a dry run leaves all of them) as incomplete

:func:`download_and_build_missing` is the only place with thread-pool code: a pool of ``max_parallel_downloads``
download jobs (the ``github_code`` sources of one repo form one job) and a pool of ``num_workers`` build jobs run
side by side (:class:`JobPool`). A source is built the moment its download job finished, sources with nothing to
download are built right away, and a source is never built while its own download runs. Each build job may hold a
spawn process pool of ``pass_workers`` for its optional cleaning passes (decontamination / minhash), so the worst
case is ``num_workers × pass_workers`` worker processes next to the threads. A failing job stops every running job
of both pools at its next shard (:class:`StopFlag`) and is re-raised after they stopped: a failed source is a
failed build. Ctrl-C while waiting does the same and raises :class:`BuildAborted` (``prepare.py`` exits 130);
everything published so far is kept and the next run resumes at shard granularity.

``status`` is read-only: the repair step's dry report ("would repair: …") plus the same
:func:`assess_dataset_state` ending, so it and ``prepare --dry_run`` cannot call the same tree differently.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from data_preparation.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, StopCheck, check_stop
from data_preparation.lib.build.lock import build_lock
from data_preparation.lib.build.planner import (
    DatasetReport,
    DownloadPlan,
    build_is_pending,
    every_source_satisfies_its_budget,
    plan_downloads,
    selected_sources,
    sources_with_pending_raw_shards,
    summarize_dataset_state,
)
from data_preparation.lib.build.repair import Confirm, RepairAction, RepairReport, repair_broken_and_stale_folders
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.loaders import github_code_repo_key
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import download, download_github_code_group, prepare_tokenizer
from data_preparation.lib.ui.dashboard import progress, set_status

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "build")
MAX_ROUNDS = 5  # download + build rounds; a source still short afterwards is reported, not looped on forever
DEFAULT_MAX_PARALLEL_DOWNLOADS = 2
DEFAULT_NUM_WORKERS = 2  # sources built at a time (threads; pyarrow/tokenizers release the GIL)
DEFAULT_PASS_WORKERS = 4  # spawn processes per build for the optional cleaning passes (decontamination / minhash)


# --- prepare / status ------------------------------------------------------------------------------------------------


def prepare(
    config_path: str | Path,
    dataset_dir: str | Path,
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pass_workers: int = DEFAULT_PASS_WORKERS,
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

    ``assume_yes`` answers the repair confirmation (stale / outdated raw folders, processed folders whose manifest
    cannot be parsed) without asking; otherwise ``confirm`` (or the terminal) is asked once and a refusal raises
    :class:`ConfirmationRequired` before anything is changed. ``dry_run`` reports what the repair and the first
    round would do and writes nothing (not even the lock file); its report is the one :func:`status` gives for the
    same tree. ``steps`` (a subset of :data:`STEPS`) and ``sources`` restrict the work, and the satisfaction check,
    to the named steps / sources; the returned report always covers the whole config.
    """
    config = load_dataset_config(config_path)
    layout = DatasetLayout(Path(dataset_dir))
    active_steps = checked_steps(steps)
    selected = checked_sources(config, sources)
    check_worker_counts(num_workers, max_parallel_downloads, pass_workers)
    warn_about_overlaps(config)

    with build_lock(layout.root) if not dry_run else nullcontext():
        if "tokenizer" in active_steps and not dry_run:
            prepare_tokenizer(config, layout)
        repair_report = repair_broken_and_stale_folders(config, layout, assume_yes=assume_yes, dry_run=dry_run, confirm=confirm, sources=selected)
        log_repair(repair_report)
        for round_number in range(1, MAX_ROUNDS + 1):
            download_plan = plan_downloads(config, layout, sources=selected)
            if dry_run:
                log.info("dry run, downloads planned:\n%s", download_plan.describe(), extra={"keep": True})
                break
            log.info("round %d: %s", round_number, download_plan.summary())
            set_status(round=f"{round_number}/{MAX_ROUNDS}", step="download + build")
            download_and_build_missing(
                download_plan, config, layout, steps=active_steps, sources=selected, max_parallel_downloads=max_parallel_downloads,
                num_workers=num_workers, pass_workers=pass_workers, hf_token=hf_token, should_stop=should_stop,
            )
            if every_source_satisfies_its_budget(config, layout, sources=selected):
                break
            if not another_round_can_fetch_more(config, layout, active_steps, selected):
                break  # still short, but nothing left to download: the report names the sources
        set_status(step="status")
        report = assess_dataset_state(config, layout, repair_report)
    return report


def status(config_path: str | Path, dataset_dir: str | Path) -> DatasetReport:
    """Read-only: what the repair step would do ("would repair: …"; such sources count as incomplete) and the
    status table, logged and returned."""
    config = load_dataset_config(config_path)
    layout = DatasetLayout(Path(dataset_dir))
    warn_about_overlaps(config)
    repair_report = repair_broken_and_stale_folders(config, layout, assume_yes=False, dry_run=True)
    log_repair(repair_report)
    return assess_dataset_state(config, layout, repair_report)


# --- one round: downloads and builds side by side --------------------------------------------------------------------


def download_and_build_missing(
    download_plan: DownloadPlan,
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    steps: set[str],
    sources: list[str] | None,
    max_parallel_downloads: int,
    num_workers: int,
    pass_workers: int,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
) -> None:
    """One round: download the rows :func:`plan_downloads` found missing and build the sources whose raw shards are
    not all processed yet, at the same time. A pool of ``max_parallel_downloads`` download jobs (the ``github_code``
    sources of one repo are one job, :func:`download_github_code_group`) and a pool of ``num_workers`` build jobs
    (:func:`build_source`, resumable per raw shard) run under one :class:`StopFlag`; each build hands
    ``pass_workers`` to its optional cleaning passes. Sources with nothing to download are built right away; every
    other source is built as soon as its download job finished, so a source is never built while its own download
    runs. ``steps`` restricts the round to its download / build part, ``sources`` to the named sources."""
    downloads = download_jobs(download_plan, config, layout, hf_token) if "download" in steps else []
    downloading = {name for job in downloads for name in job.sources}
    pending = sources_with_pending_raw_shards(config, layout, sources) if "build" in steps else []
    builds = [build_source_job(config, name, layout, pass_workers) for name in pending if name not in downloading]

    flag = StopFlag(should_stop)
    build_pool = JobPool("builds", max_workers=num_workers, flag=flag, total=len(builds) + len(downloading))

    def build_when_downloaded(job: Job) -> None:
        """The follow-up of a finished download job (called in the main thread): build what it fetched."""
        for name in job.sources:
            if "build" in steps and build_is_pending(config, name, layout):
                build_pool.submit(build_source_job(config, name, layout, pass_workers))
            else:
                build_pool.bar.update(1)  # nothing to build for this source: it counts as done

    download_pool = JobPool("downloads", max_workers=max_parallel_downloads, flag=flag, total=len(downloads), on_success=build_when_downloaded)
    with download_pool, build_pool:  # the exits wait for the running jobs to stop
        for job in builds:
            build_pool.submit(job)
        for job in downloads:
            download_pool.submit(job)
        failures = wait_for_jobs([download_pool, build_pool], flag)
    raise_first_failure(failures)


# --- jobs --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One unit of work of a parallel helper; ``action`` takes the stop check the steps poll between shards."""

    what: str  # "source" | "github_code group" (for the log line on failure)
    name: str
    sources: tuple[str, ...]  # the source(s) the job writes; a finished download job's sources are built next
    action: Callable[[StopCheck], object]


def download_jobs(download_plan: DownloadPlan, config: DatasetConfig, layout: DatasetLayout, hf_token: str | None) -> list[Job]:
    """One job per source with rows to fetch; the ``github_code`` sources of one repo are grouped into one. The
    download takes a target (``rows_needed=``), so every job is asked for :attr:`SourceLedger.rows_target`: the
    rows already on disk plus the ones the plan wants added (more than the budget in a top-up round)."""
    to_fetch = download_plan.to_fetch()
    rows_needed = {source.name: source.rows_target for source in to_fetch}
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
    per group, in config order; a single source of a repo goes through the ordinary per-source download (the same
    pass over its own loader)."""
    groups: dict[tuple[str | None, str | None, str], list[str]] = {}
    for name in names:
        source = config.sources[name]
        if source.loader == "github_code":
            groups.setdefault(github_code_repo_key(source), []).append(name)
    return [group for group in groups.values() if len(group) >= 2]


def download_source_job(config: DatasetConfig, name: str, layout: DatasetLayout, rows_needed: int, hf_token: str | None) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download(config, name, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop)

    return Job("source", name, (name,), action)


def github_code_group_job(config: DatasetConfig, names: list[str], layout: DatasetLayout, rows_needed: dict[str, int], hf_token: str | None) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download_github_code_group(config, names, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop)

    return Job("github_code group", ", ".join(names), tuple(names), action)


def build_source_job(config: DatasetConfig, name: str, layout: DatasetLayout, pass_workers: int) -> Job:
    def action(should_stop: StopCheck) -> object:
        return build_source(config, name, layout, pass_workers=pass_workers, should_stop=should_stop)

    return Job("source", name, (name,), action)


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


class JobPool:
    """A thread pool of ``max_workers`` running :class:`Job` objects under a shared :class:`StopFlag`, with the summary
    bar of the dashboard panel named ``description`` (the jobs' own bars are its rows; ``total`` = the jobs expected).
    Jobs may be submitted while the pool runs (:meth:`submit`); :func:`wait_for_jobs` waits on :attr:`futures` and
    calls ``on_success`` (main thread) for every job that finished without an error; that is where the download
    pool submits the build of what it fetched. Leaving the ``with`` block waits for the running jobs (they stop at
    their next shard once the flag is raised), then closes the bar."""

    def __init__(
        self, description: str, *, max_workers: int, flag: StopFlag, total: int, on_success: Callable[[Job], None] | None = None
    ) -> None:
        self.description = description
        self.flag = flag
        self.on_success = on_success
        self.futures: dict[Future[None], Job] = {}
        self._running = RunningJobs()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=description)
        self._bar: Progress | None = None
        self._total = total

    @property
    def bar(self) -> Progress:
        if self._bar is None:
            raise RuntimeError(f"{self.description}: the pool is not entered")
        return self._bar

    def submit(self, job: Job) -> None:
        self.futures[self._executor.submit(run_job, job, self.flag, self._running, self.bar)] = job

    def cancel_queued(self) -> None:
        """Cancel the jobs not started yet (the running ones stop at their next shard through the flag)."""
        for future in self.futures:
            future.cancel()

    def __enter__(self) -> JobPool:
        self._bar = progress(total=self._total, desc=self.description, unit="job", panel=self.description, summary=True).__enter__()
        self._executor.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        try:
            self._executor.__exit__(exc_type, exc, tb)  # waits for the running jobs
        finally:
            self.bar.__exit__(exc_type, exc, tb)


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


def wait_for_jobs(pools: list[JobPool], flag: StopFlag) -> list[BaseException]:
    """Wait until every job of every pool finished, including the jobs an ``on_success`` follow-up submits while
    waiting. Returns the failures (a ``KeyboardInterrupt`` becomes a :class:`BuildAborted`). The first failure or
    interrupt raises the flag (the running jobs of every pool stop at their next shard), cancels the jobs not started
    yet and ends the follow-ups; the jobs that merely stopped are not collected (their ``BuildAborted`` is implied)."""
    failures: list[BaseException] = []
    handled: set[Future[None]] = set()
    try:
        while unhandled := [future for pool in pools for future in pool.futures if future not in handled]:
            done, _ = wait(unhandled, return_when=FIRST_COMPLETED)
            for future in done:
                handled.add(future)
                pool = next(pool for pool in pools if future in pool.futures)
                job = pool.futures[future]
                if future.cancelled():
                    continue  # never started: cancelled after another job failed
                error = future.exception()
                if error is not None:
                    flag.stop(f"{job.what} {job.name} failed")
                    cancel_all(pools)
                    failures.append(error)
                    continue
                pool.bar.update(1)
                if pool.on_success is not None and not flag.should_stop():
                    pool.on_success(job)
    except KeyboardInterrupt:
        flag.stop("interrupted")
        cancel_all(pools)
        log.warning("interrupted; the running jobs stop at their next shard, everything published so far is kept")
        failures.append(BuildAborted("interrupted; everything published so far is kept, rerun to resume"))
    except BaseException:
        # a crash in an `on_success` follow-up (main-thread code) must stop the running jobs like a job
        # failure would; otherwise the pool exits block on downloads polling a flag nobody raised
        flag.stop("a follow-up after a finished job failed")
        cancel_all(pools)
        log.exception("follow-up after a finished job failed; the running jobs stop at their next shard")
        raise
    return failures


def cancel_all(pools: list[JobPool]) -> None:
    for pool in pools:
        pool.cancel_queued()


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
    """``sources`` in config order, each once (None for every source); unknown names are an error. A name listed
    twice (``prepare.py --sources a a``) would otherwise be inspected twice by the repair step, appear twice in its
    confirmation list and make the second deletion of the same folder fail on a directory that is no longer there."""
    return None if sources is None else selected_sources(config, sources)


def check_worker_counts(num_workers: int, max_parallel_downloads: int, pass_workers: int) -> None:
    if num_workers < 1 or max_parallel_downloads < 1 or pass_workers < 1:
        raise ValueError(
            "num_workers, max_parallel_downloads and pass_workers must be >= 1, got "
            f"{num_workers}, {max_parallel_downloads} and {pass_workers}"
        )


def another_round_can_fetch_more(config: DatasetConfig, layout: DatasetLayout, active_steps: set[str], selected: list[str] | None) -> bool:
    """Whether a further round would download anything: the download step is active and the plan (the same
    :class:`~data_preparation.lib.build.planner.SourceLedger` objects the satisfaction check reads) still has rows
    to fetch for some selected source. That is a loader that returned fewer rows than asked without being exhausted,
    or a source whose build dropped more than the safety margin covers: the next round tops it up by the shortfall
    scaled with the yield it showed, instead of planning nothing and leaving the run stuck."""
    if "download" not in active_steps:
        return False
    return plan_downloads(config, layout, sources=selected).total_rows_to_fetch() > 0


def warn_about_overlaps(config: DatasetConfig) -> None:
    for warning in config.overlap_warnings():
        log.warning(warning)


def outstanding_repairs(report: RepairReport) -> list[RepairAction]:
    """The actions of a repair pass that were planned but not carried out: everything of a dry run, nothing of a
    pass that performed them. They are what still stands between the tree and a complete dataset."""
    return [] if report.performed else list(report.actions)


def log_repair(report: RepairReport) -> None:
    """One line per action of a repair pass: what it did, or what it would do (a dry run)."""
    if not report.actions:
        return
    if outstanding_repairs(report):
        log.warning("would repair:\n%s", report.describe(), extra={"keep": True})
    else:
        log.info("repair:\n%s", report.describe(), extra={"keep": True})


def assess_dataset_state(config: DatasetConfig, layout: DatasetLayout, repair_report: RepairReport) -> DatasetReport:
    """The verdict :func:`prepare` and :func:`status` both end with, so the two can never disagree about one tree:
    the status table, with the sources of the repairs ``repair_report`` left undone counted as incomplete. A
    ``status`` run and a ``prepare --dry_run`` change nothing, so their planned repairs are still outstanding; a real
    ``prepare`` performed them before it downloaded anything and leaves none."""
    report = summarize_dataset_state(config, layout, needs_repair=[action.source for action in outstanding_repairs(repair_report)])
    log_report(report)
    return report


def log_report(report: DatasetReport) -> None:
    """The status table (kept in the scrollback) plus one warning per exhausted or unsatisfied source."""
    for source in report.sources:
        satisfied, reason = source.satisfaction()
        if satisfied and source.exhausted:
            log.warning("%s: source exhausted (%s); the training sampler cycles the rows on disk", source.name, reason)
        elif not satisfied:
            log.warning("%s: %s", source.name, reason)
    log.info("dataset status:\n%s", report.describe(), extra={"keep": True})  # keep: printed unwrapped into the scrollback
