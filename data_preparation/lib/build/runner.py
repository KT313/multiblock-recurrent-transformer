# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
prepare / status: the top-level data pipeline, readable top to bottom.

prepare runs inspection, authorization, tokenizer publication, repairs, and download/build rounds under the
dataset directory's build lock::

    inspect_repairs / inspect_tokenizer                read-only plan over the existing published content
    authorize_repairs                                  foreign-data guard and single repair confirmation
    prepare_planned_tokenizer                          privately acquire/validate, then publish tokenizers/<name>/
    perform_repairs                                    truncate/adopt/delete raw, rebuild stale processed
    reopen_raw                                         clear the exhausted flag of the --reopen sources
    for round in 1..MAX_ROUNDS:
        plan_downloads                                 rows still missing per source (lib/build/planner.py: one
                                                       SourceLedger per source answers both "what to download" and
                                                       "is it done", so a round 2 tops a short source up)
        download_and_build_missing                     downloads (sources/<name>/raw) and builds (processed/<name>) at the
                                                       same time: a source is built as soon as its download finished
        stop when every source serves its budget, or when nothing more can be fetched
    assess_dataset_state                               the status table and its verdict, counting the repairs this
                                                       run left undone (a dry run leaves all of them) as incomplete

:func:`download_and_build_missing` is the only place with thread-pool code: a pool of max_parallel_downloads
download jobs (the github_code sources of one repo form one job) and a pool of num_workers build jobs run
side by side (:class:`JobPool`). A source is built the moment its download job finished, sources with nothing to
download are built right away, and a source is never built while its own download runs. Each build job may hold a
spawn process pool of pass_workers for its optional cleaning passes (decontamination / minhash), so the worst
case is num_workers × pass_workers worker processes next to the threads. A failing job stops every running job
of both pools at its next shard (:class:`StopFlag`) and is re-raised after they stopped: a failed source is a
failed build. Ctrl-C while waiting does the same and raises :class:`BuildAborted` (prepare.py exits 130);
everything published so far is kept and the next run resumes at shard granularity. A second Ctrl-C, while the
pools wait for the running jobs, ends the process right away (:meth:`JobPool._end_without_waiting`).

status is read-only: the repair step's dry report ("would repair: …") plus the same
:func:`assess_dataset_state` ending, so it and prepare --dry_run cannot call the same tree differently.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from data_preparation.lib.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, StopCheck, check_stop
from data_preparation.lib.build.lock import DatasetLease, dataset_lock
from data_preparation.lib.build.planner import (
    DatasetReport,
    DownloadPlan,
    UnreadableRawShardError,
    build_is_pending,
    every_source_satisfies_its_budget,
    plan_downloads,
    selected_sources,
    source_ledger,
    sources_with_pending_raw_shards,
    summarize_dataset_state,
)
from data_preparation.lib.build.repair import (
    Confirm, RepairAction, RepairReport, authorize_repairs, inspect_repairs, perform_repairs, repair_broken_and_stale_folders,
)
from data_preparation.lib.log import ROOT_LOGGER_NAME, get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.loaders import github_code_repo_key
from data_preparation.lib.stages.benchmark_seeds import load_benchmark_seeds
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import (
    download, download_github_code_group, inspect_tokenizer, prepare_planned_tokenizer, reopen_raw,
)
from data_preparation.lib.stages.global_dedup import global_policy, ordered_sources
from data_preparation.lib.stages.global_build import build_global_source, outputs_complete, source_frontier
from data_preparation.lib.storage.tokenizer_assessment import assess_tokenizer_folder
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import guarded_path
from data_preparation.lib.storage.snapshot import publish_snapshot, snapshot_problem
from data_preparation.lib.ui.dashboard import active_dashboard, progress, set_status

log = get_logger(__name__)

STEPS: tuple[str, ...] = ("tokenizer", "download", "build")
MAX_ROUNDS = 5  # download + build rounds; a source still short afterwards is reported, not looped on forever
DEFAULT_MAX_PARALLEL_DOWNLOADS = 2
DEFAULT_NUM_WORKERS = 2  # sources built at a time (threads; pyarrow/tokenizers release the GIL)
DEFAULT_PASS_WORKERS = 4  # spawn processes per build for the optional cleaning passes (decontamination / minhash)


# --- prepare / status ------------------------------------------------------------------------------------------------


def prepare_command(config_path: str | Path, dataset_dir: str | Path) -> str:
    """
    The command an error message tells the user to run: the same one training's auto-prepare prints
    (`training/data/dataset_resolver.py::build_command`), with the `--yes` that answers the repair confirmation.
    """

    return f"python data_preparation/prepare.py prepare --dataset_config {config_path} --dataset_dir {dataset_dir} --yes"


@contextmanager
def unreadable_shard_remedy(config_path: str | Path, dataset_dir: str | Path) -> Iterator[None]:
    """
    Give an :class:`UnreadableRawShardError` from the planner the remedy this level knows: the repair step and the
    command that runs it. The planner sees neither the config path nor the dataset directory, and a read-only
    caller (status, a dry run) does not repair anything itself, so the message has to say what will.
    """

    try:
        yield
    except UnreadableRawShardError as error:
        remedy = (
            f"the repair step truncates raw/{error.source} to its readable prefix, or deletes the folder when no "
            f"shard is readable; run\n  {prepare_command(config_path, dataset_dir)}\nthen status again"
        )
        raise error.with_remedy(remedy) from error


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
    reopen: Iterable[str] | None = None,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
    confirm: Confirm | None = None,
    allow_foreign_raw: bool = False,
    dataset_lease: DatasetLease | None = None,
) -> DatasetReport:
    """
    Materialise the dataset config at config_path under dataset_dir (see the module docstring) and return
    its status. Training auto-prepare may borrow an active dataset_lease; it never releases that ownership.

    assume_yes answers the repair confirmation (stale / outdated raw folders, processed folders whose manifest
    cannot be parsed) without asking; otherwise confirm (or the terminal) is asked once and a refusal raises
    :class:`ConfirmationRequired` before any dataset content changes (the lock holder metadata may be updated).
    A raw folder another dataset config downloaded
    (raw folders are shared by name) is deleted only with allow_foreign_raw on top, whatever the answer;
    raw manifests written by this run carry the config's file name for that. dry_run reports what the repair and the first
    round would do and writes nothing (not even the lock file); its report is the one :func:`status` gives for the
    same tree. steps (a subset of :data:`STEPS`) and sources restrict the work, and the satisfaction check,
    to the named steps / sources; the returned report always covers the whole config. reopen names sources
    whose exhausted flag is cleared before planning (:func:`reopen_raw`: their loader has more rows now).
    """

    config = load_dataset_config(config_path)
    config.validate_identifiers()
    config_name = Path(config_path).name
    layout = DatasetLayout(Path(dataset_dir))
    active_steps = checked_steps(steps)
    selected = checked_sources(config, sources)
    reopened = checked_sources(config, reopen) or []
    check_worker_counts(num_workers, max_parallel_downloads, pass_workers)
    warn_about_overlaps(config)

    with dataset_lock(layout.root, lease=dataset_lease) if not dry_run else nullcontext(), unreadable_shard_remedy(config_path, dataset_dir):
        if config.bloom_deduplicate_across_sources:
            return prepare_global(
                config, layout.for_config(config), config_name=config_name, steps=active_steps, selected=selected,
                reopened=reopened, dry_run=dry_run, assume_yes=assume_yes, confirm=confirm,
                allow_foreign_raw=allow_foreign_raw, hf_token=hf_token, num_workers=num_workers,
                pass_workers=pass_workers, max_parallel_downloads=max_parallel_downloads, should_stop=should_stop,
            )
        repair_report = inspect_repairs(config, layout, sources=selected, config_name=config_name)
        tokenizer_plan = inspect_tokenizer(config, layout) if "tokenizer" in active_steps else None
        if not dry_run:
            authorize_repairs(repair_report, assume_yes=assume_yes, confirm=confirm, allow_foreign_raw=allow_foreign_raw)
            # Validate and publish the intended tokenizer before changing raw/processed data. Staging failures
            # preserve all published content; after publication starts there is no multi-directory rollback.
            if tokenizer_plan is not None:
                prepare_planned_tokenizer(config, tokenizer_plan, hf_token=hf_token)
            perform_repairs(repair_report)
        log_repair(repair_report)
        reopen_sources(config, layout, reopened, dry_run=dry_run)
        for round_number in range(1, MAX_ROUNDS + 1):
            download_plan = plan_downloads(config, layout, sources=selected)
            if dry_run:
                log.info("dry run, downloads planned:\n%s", download_plan.describe(), extra={"keep": True})
                break
            log.info("round %d: %s", round_number, download_plan.summary())
            set_status(round=f"{round_number}/{MAX_ROUNDS}", step="download + build")
            download_and_build_missing(
                download_plan, config, layout, steps=active_steps, sources=selected, max_parallel_downloads=max_parallel_downloads,
                num_workers=num_workers, pass_workers=pass_workers, hf_token=hf_token, should_stop=should_stop, config_name=config_name,
            )
            if every_source_satisfies_its_budget(config, layout, sources=selected):
                break
            if not another_round_can_fetch_more(config, layout, active_steps, selected):
                break  # still short, but nothing left to download: the report names the sources
        set_status(step="status")
        report = assess_dataset_state(config, layout, repair_report, publish=not dry_run)
    return report


def status(config_path: str | Path, dataset_dir: str | Path) -> DatasetReport:
    """
    Read-only: what the repair step would do ("would repair: …"; such sources count as incomplete) and the
    status table, logged and returned.
    """

    config = load_dataset_config(config_path)
    config.validate_identifiers()
    layout = DatasetLayout(Path(dataset_dir)).for_config(config)
    warn_about_overlaps(config)
    with unreadable_shard_remedy(config_path, dataset_dir):
        repair_report = (inspect_repairs(config, layout, config_name=Path(config_path).name) if layout.processed_scope else
                         repair_broken_and_stale_folders(config, layout, assume_yes=False, dry_run=True, config_name=Path(config_path).name))
        log_repair(repair_report)
        return assess_dataset_state(config, layout, repair_report)


def inspect_global_repairs(config: DatasetConfig, layout: DatasetLayout, config_name: str) -> RepairReport:
    """Inspect local candidates and final output; shared raw actions execute only once."""
    candidate = inspect_repairs(config, DatasetLayout(layout.root), config_name=config_name)
    final = inspect_repairs(config, layout, config_name=config_name)
    folders = {action.folder for action in candidate.actions}
    return RepairReport([*candidate.actions, *(action for action in final.actions if action.folder not in folders)])


def prepare_global(
    config: DatasetConfig, layout: DatasetLayout, *, config_name: str, steps: set[str],
    selected: list[str] | None, reopened: list[str], dry_run: bool, assume_yes: bool,
    confirm: Confirm | None, allow_foreign_raw: bool, hf_token: str | None,
    num_workers: int, pass_workers: int, max_parallel_downloads: int, should_stop: StopCheck | None,
) -> DatasetReport:
    """Caller holds the exclusive lease until every worker and ordered writer stops."""
    from data_preparation.lib.build.planner import source_ledger

    for name in config.sources:
        guarded_path(layout.root, layout.processed_dir(name))
    repair = inspect_repairs(config, layout, config_name=config_name)
    current = summarize_dataset_state(config, layout, needs_repair=[action.source for action in repair.actions])
    current.snapshot_problem = snapshot_problem(config, layout, processing=global_policy(config))
    if current.complete and not reopened and outputs_complete(config, layout):
        if not dry_run:
            tokenizer = assess_tokenizer_folder(
                layout.tokenizer_dir(config.tokenizer.name), config.tokenizer_hash(), validate_payload=True,
            )
            if not tokenizer.ready and "tokenizer" not in steps:
                raise ValueError(f"{tokenizer.problem}; run prepare with the tokenizer step to repair it")
            current.tokenizer_complete = tokenizer.ready
            current.tokenizer_problem = tokenizer.problem
        if current.complete:
            log.info("dataset-wide Bloom snapshot already complete; no preparation changes needed")
            log_report(current)
            return current
    if selected is not None and set(selected) != set(config.sources):
        raise ValueError(
            "dataset-wide Bloom admission requires the complete dataset scope and priority prerequisites; "
            "omit --sources to prepare/replay all sources before requesting a partial no-op"
        )
    repair = inspect_global_repairs(config, layout, config_name)
    log.info("dataset-wide Bloom preparation/replay -> %s; source-local candidates and raw downloads are reusable", layout.processed_scope)
    if dry_run:
        log_repair(repair)
        log.info("dry run, downloads planned:\n%s", plan_downloads(config, layout, sources=selected).describe())
        return assess_dataset_state(config, layout, repair)
    authorize_repairs(repair, assume_yes=assume_yes, confirm=confirm, allow_foreign_raw=allow_foreign_raw)
    with load_benchmark_seeds(
        config.bloom_deduplicate_across_sources_add_benchmarks if "build" in steps else [],
        memory_mb=config.bloom_dedup_memory_mb, hf_token=hf_token, should_stop=should_stop,
    ) as seeds:
        if "tokenizer" in steps:
            prepare_planned_tokenizer(config, inspect_tokenizer(config, layout), hf_token=hf_token)
        perform_repairs(repair)
        log_repair(repair)
        reopen_sources(config, layout, reopened, dry_run=False)
        candidates = DatasetLayout(layout.root)
        plan = plan_downloads(config, candidates, sources=selected)
        log.info("round 1: %s", plan.summary())
        download_and_build_missing(
            plan, config, candidates, steps=steps, sources=selected,
            max_parallel_downloads=max_parallel_downloads, num_workers=num_workers, pass_workers=pass_workers,
            hf_token=hf_token, should_stop=should_stop, config_name=config_name,
        )
        if "build" not in steps:
            for _ in range(1, MAX_ROUNDS):
                if "download" not in steps:
                    break
                followup = plan_downloads(config, candidates, sources=selected)
                if followup.total_rows_to_fetch() == 0:
                    break
                download_and_build_missing(
                    followup, config, candidates, steps=steps, sources=selected,
                    max_parallel_downloads=max_parallel_downloads, num_workers=num_workers, pass_workers=pass_workers,
                    hf_token=hf_token, should_stop=should_stop, config_name=config_name,
                )
            return assess_dataset_state(config, layout, repair)
        frontier = seeds.frontier(ordered_sources(config), config.bloom_dedup_memory_mb)
        for name in ordered_sources(config):
            start = frontier
            complete = False
            for round_number in range(1, MAX_ROUNDS + 1):
                check_stop(should_stop)
                ledger = source_ledger(config, name, layout)
                if ledger.raw_state != "current":
                    break
                # Extend local candidates only by the global shortfall. Existing buffered
                # raw shards are consumed before any additional download is requested.
                local = Manifest.load(candidates.processed_dir(name))
                target = ledger.rows_sufficient
                retained = Manifest.load(layout.processed_dir(name))
                if local is not None and retained is not None:
                    target = local.rows() + max(0, ledger.rows_sufficient - ledger.processed_rows)
                    if (not retained.generation_complete and retained.extra.get("candidate_generation") == local.generation_id
                            and (source_frontier(retained).source_index > start.source_index
                                 or source_frontier(retained).source_candidates < local.rows())):
                        target = local.rows()  # recover pending candidates before planning a top-up
                local = build_source(config, name, candidates, pass_workers=pass_workers,
                                     should_stop=should_stop, rows_target=target)
                frontier, complete = build_global_source(
                    config, name, layout, start, rows_target=ledger.rows_sufficient,
                    exhausted=ledger.exhausted, should_stop=should_stop, preseed_keys=seeds.keys(),
                )
                if complete:
                    break
                if round_number == MAX_ROUNDS:
                    break
                raw = Manifest.load(candidates.raw_dir(name))
                if raw is not None and len(local.input_shards) < len(raw.shards):
                    continue
                if "download" not in steps:
                    break
                ledger = source_ledger(config, name, layout)
                increment = ledger.rows_to_fetch[0] or max(1, ledger.rows_needed)
                # A zero-yield source can have unique rows later; continue bounded top-ups
                # instead of treating global losses as proof that every later row is useless.
                log.info("%s: ordered global top-up %d/%d, fetching at most %d rows", name, round_number, MAX_ROUNDS, increment)
                download(config, name, candidates, rows_needed=ledger.raw_rows + increment,
                         hf_token=hf_token, should_stop=should_stop, config_name=config_name)
                after = source_ledger(config, name, layout)
                if after.raw_rows <= ledger.raw_rows and not after.exhausted:
                    log.warning("%s: global top-up made no raw progress; stopping before lower-priority admission", name)
                    break
            if not complete:
                log.warning("%s: global retained budget is unfinished; lower-priority sources remain pending", name)
                break
        return assess_dataset_state(config, layout, repair, publish=True)


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
    config_name: str | None = None,
) -> None:
    """
    Internal mutation helper: caller must own the dataset lock until all job pools have stopped.

    One round: download the rows :func:`plan_downloads` found missing and build the sources whose raw shards are
    not all processed yet, at the same time. A pool of max_parallel_downloads download jobs (the github_code
    sources of one repo are one job, :func:`download_github_code_group`) and a pool of num_workers build jobs
    (:func:`build_source`, resumable per raw shard) run under one :class:`StopFlag`; each build hands
    pass_workers to its optional cleaning passes. Sources with nothing to download are built right away; every
    other source is built as soon as its download job finished, so a source is never built while its own download
    runs. steps restricts the round to its download / build part, sources to the named sources; config_name
    (the dataset config's file name) is recorded in the raw manifests the downloads create.
    """

    downloads = download_jobs(download_plan, config, layout, hf_token, config_name) if "download" in steps else []
    downloading = {name for job in downloads for name in job.sources}
    pending = sources_with_pending_raw_shards(config, layout, sources) if "build" in steps else []
    builds = [build_source_job(config, name, layout, pass_workers) for name in pending if name not in downloading]

    flag = StopFlag(should_stop)
    build_pool = JobPool("builds", max_workers=num_workers, flag=flag, total=len(builds) + len(downloading))

    def build_when_downloaded(job: Job) -> None:
        """
        The follow-up of a finished download job (called in the main thread): build what it fetched.
        """

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
    """
    One unit of work of a parallel helper; action takes the stop check the steps poll between shards.
    """

    what: str  # "source" | "github_code group" (for the log line on failure)
    name: str
    sources: tuple[str, ...]  # the source(s) the job writes; a finished download job's sources are built next
    action: Callable[[StopCheck], object]


def download_jobs(
    download_plan: DownloadPlan, config: DatasetConfig, layout: DatasetLayout, hf_token: str | None, config_name: str | None = None
) -> list[Job]:
    """
    One job per source with rows to fetch; the github_code sources of one repo are one job as soon as any of
    them has rows to fetch, every member included (a member that has its rows joins passively and stores what
    the pass reads on for the others; the languages without a source get folders of their own, see
    stages/download.py). The download takes a target (rows_needed=), so every job is asked for
    :attr:`SourceLedger.rows_target`: the rows already on disk plus the ones the plan wants added (more than the
    budget in a top-up round; just the rows on disk for a member with nothing to fetch).
    """

    targets = {source.name: source.rows_target for source in download_plan.sources}
    to_fetch = [source.name for source in download_plan.to_fetch()]
    jobs: list[Job] = []
    grouped: set[str] = set()
    for names in github_code_groups(config, [source.name for source in download_plan.sources]):
        if not set(to_fetch).intersection(names):
            continue
        jobs.append(github_code_group_job(config, names, layout, {name: targets[name] for name in names}, hf_token, config_name))
        grouped.update(names)
    for name in to_fetch:
        if name not in grouped:
            jobs.append(download_source_job(config, name, layout, targets[name], hf_token, config_name))
    return jobs


def github_code_groups(config: DatasetConfig, names: list[str]) -> list[list[str]]:
    """
    The github_code sources among names grouped by repo (:func:`github_code_repo_key`), in config order; a
    single source of a repo is a group of its own (the group pass is what keeps the other languages).
    """

    groups: dict[tuple[str | None, str | None, str], list[str]] = {}
    for name in names:
        source = config.sources[name]
        if source.loader == "github_code":
            groups.setdefault(github_code_repo_key(source), []).append(name)
    return list(groups.values())


def download_source_job(
    config: DatasetConfig, name: str, layout: DatasetLayout, rows_needed: int, hf_token: str | None, config_name: str | None = None
) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download(config, name, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop, config_name=config_name)

    return Job("source", name, (name,), action)


def github_code_group_job(
    config: DatasetConfig, names: list[str], layout: DatasetLayout, rows_needed: dict[str, int], hf_token: str | None, config_name: str | None = None
) -> Job:
    def action(should_stop: StopCheck) -> object:
        return download_github_code_group(
            config, names, layout, rows_needed=rows_needed, hf_token=hf_token, should_stop=should_stop, config_name=config_name
        )

    return Job("github_code group", ", ".join(names), tuple(names), action)


def build_source_job(config: DatasetConfig, name: str, layout: DatasetLayout, pass_workers: int) -> Job:
    """
    The build of name, capped at the budget: the planner's rows_sufficient, read when the job is made (after
    the source's download finished), is the build's rows_target.
    """

    rows_target = source_ledger(config, name, layout).rows_sufficient

    def action(should_stop: StopCheck) -> object:
        return build_source(config, name, layout, pass_workers=pass_workers, should_stop=should_stop, rows_target=rows_target)

    return Job("source", name, (name,), action)


# --- running jobs in a pool ---------------------------------------------------------------------------------------------


class StopFlag:
    """
    The shared stop request of one pool of jobs: stop(reason) makes every job stop at its next shard (the
    steps poll :meth:`should_stop`); the first reason wins. An outer should_stop (Ctrl-C handling of the caller,
    a training run shutting down) is polled too.
    """

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
    """
    A thread pool of max_workers running :class:`Job` objects under a shared :class:`StopFlag`, with the summary
    bar of the dashboard panel named description (the jobs' own bars are its rows; total = the jobs expected).
    Jobs may be submitted while the pool runs (:meth:`submit`); :func:`wait_for_jobs` waits on :attr:`futures` and
    calls on_success (main thread) for every job that finished without an error; that is where the download
    pool submits the build of what it fetched. Leaving the with block waits for the running jobs (they stop at
    their next shard once the flag is raised), then closes the bar; a Ctrl-C during that wait ends the process
    (:meth:`_end_without_waiting`).
    """

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
        """
        Cancel the jobs not started yet (the running ones stop at their next shard through the flag).
        """

        for future in self.futures:
            future.cancel()

    def __enter__(self) -> JobPool:
        self._bar = progress(total=self._total, desc=self.description, unit="job", panel=self.description, summary=True).__enter__()
        self._executor.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        try:
            self._executor.__exit__(exc_type, exc, tb)  # waits for the running jobs
        except KeyboardInterrupt:
            self._end_without_waiting()
        finally:
            self.bar.__exit__(exc_type, exc, tb)

    def _end_without_waiting(self) -> None:
        """
        Ctrl-C while the pool already waits for its running jobs (the second one of a run): end the process now.
        The running transfer may be a row group of hundreds of MB, and raising out of the with block would not
        skip it: the interpreter joins every executor thread at exit. So the executor stops handing out jobs,
        the dashboard closes (the terminal restored, the kept lines printed), the log is flushed and the process
        exits with prepare.py's interrupted code; everything published so far is on disk already.
        """

        self._executor.shutdown(wait=False, cancel_futures=True)
        log.warning("second interrupt: ending without waiting for the running transfer; everything published so far is kept")
        dashboard = active_dashboard()
        if dashboard is not None:
            dashboard.__exit__(None, None, None)
        for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
            handler.flush()
        sys.stderr.flush()
        os._exit(130)


class RunningJobs:
    """
    The names of the jobs running right now (the pool bar's postfix).
    """

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
    """
    Run one job in a pool thread; a failure raises the flag (the other jobs stop at their next shard, queued
    jobs never start), is logged with its traceback and propagates to :func:`wait_for_jobs`.
    """

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
    """
    Wait until every job of every pool finished, including the jobs an on_success follow-up submits while
    waiting. Returns the failures (a KeyboardInterrupt becomes a :class:`BuildAborted`). The first failure or
    interrupt raises the flag (the running jobs of every pool stop at their next shard), cancels the jobs not started
    yet and ends the follow-ups; the jobs that merely stopped are not collected (their BuildAborted is implied).
    """

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
    """
    Re-raise the failure that caused the stop: jobs that merely stopped because of it raised
    :class:`BuildAborted`, which is only raised when nothing else went wrong.
    """

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
    """
    sources in config order, each once (None for every source); unknown names are an error. A name listed
    twice (prepare.py --sources a a) would otherwise be inspected twice by the repair step, appear twice in its
    confirmation list and make the second deletion of the same folder fail on a directory that is no longer there.
    """

    return None if sources is None else selected_sources(config, sources)


def check_worker_counts(num_workers: int, max_parallel_downloads: int, pass_workers: int) -> None:
    if num_workers < 1 or max_parallel_downloads < 1 or pass_workers < 1:
        raise ValueError(
            "num_workers, max_parallel_downloads and pass_workers must be >= 1, got "
            f"{num_workers}, {max_parallel_downloads} and {pass_workers}"
        )


def reopen_sources(config: DatasetConfig, layout: DatasetLayout, names: list[str], *, dry_run: bool) -> None:
    """
    Clear the exhausted flag of the named sources (:func:`reopen_raw`); a dry run only says which it would.
    """

    for name in names:
        if dry_run:
            log.info("dry run, would reopen %s", name)
        elif not reopen_raw(config, name, layout):
            log.info("%s: nothing to reopen", name)


def another_round_can_fetch_more(config: DatasetConfig, layout: DatasetLayout, active_steps: set[str], selected: list[str] | None) -> bool:
    """
    Whether a further round would download anything: the download step is active and the plan (the same
    :class:`~data_preparation.lib.build.planner.SourceLedger` objects the satisfaction check reads) still has rows
    to fetch for some selected source. That is a source whose raw shards measured fewer tokens per row than the
    estimate its first download was sized with, or whose build dropped more than the safety margin covers: the
    next round tops it up (by the measured rate, or by the shortfall scaled with the yield it showed) instead of
    planning nothing and leaving the run stuck.
    """

    if "download" not in active_steps:
        return False
    return plan_downloads(config, layout, sources=selected).total_rows_to_fetch() > 0


def warn_about_overlaps(config: DatasetConfig) -> None:
    for warning in config.overlap_warnings():
        log.warning(warning)


def outstanding_repairs(report: RepairReport) -> list[RepairAction]:
    """
    The actions of a repair pass that were planned but not carried out: everything of a dry run, nothing of a
    pass that performed them. They are what still stands between the tree and a complete dataset (a folder the
    step leaves alone is not one of them: nothing of the step's stands there).
    """

    return [] if report.performed else [action for action in report.actions if action.action != "leave"]


def log_repair(report: RepairReport) -> None:
    """
    One line per action of a repair pass: what it did, or what it would do (a dry run).
    """

    if not report.actions:
        return
    if outstanding_repairs(report):
        log.warning("would repair:\n%s", report.describe(), extra={"keep": True})
    else:
        log.info("repair:\n%s", report.describe(), extra={"keep": True})


def assess_dataset_state(
    config: DatasetConfig, layout: DatasetLayout, repair_report: RepairReport, *, publish: bool = False
) -> DatasetReport:
    """
    The verdict :func:`prepare` and :func:`status` both end with, so the two can never disagree about one tree:
    the status table, with the sources of the repairs repair_report left undone counted as incomplete. A
    status run and a prepare --dry_run change nothing, so their planned repairs are still outstanding; a real
    prepare performed them before it downloaded anything and leaves none.
    """

    report = summarize_dataset_state(config, layout, needs_repair=[action.source for action in outstanding_repairs(repair_report)])
    processing = global_policy(config) if layout.processed_scope else None
    if publish and report.complete and outputs_complete(config, layout):
        publish_snapshot(config, layout, processing=processing)
    report.snapshot_problem = snapshot_problem(config, layout, processing=processing)
    if layout.processed_scope and report.snapshot_problem is None and not outputs_complete(config, layout):
        report.snapshot_problem = "dataset-wide Bloom frontier is incomplete; run prepare to continue ordered admission"
    elif layout.processed_scope and report.snapshot_problem is not None:
        report.snapshot_problem = "dataset-wide Bloom preparation/replay required: " + report.snapshot_problem
    log_report(report)
    return report


def log_report(report: DatasetReport) -> None:
    """
    The status table (kept in the scrollback), then one warning per exhausted or unsatisfied source, after the
    table so they stand next to the verdict instead of scrolling away above it.
    """

    log.info("dataset status:\n%s", report.describe(), extra={"keep": True})  # keep: printed unwrapped into the scrollback
    for source in report.sources:
        satisfied, reason = source.satisfaction()
        if satisfied and source.exhausted:
            log.warning(
                "%s: source exhausted (%s); the training sampler cycles the rows on disk; rerun with --reopen %s if the source has more rows",
                source.name, reason, source.name,
            )
        elif not satisfied:
            log.warning("%s: %s", source.name, reason)
