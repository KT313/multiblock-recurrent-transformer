# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Periodic, process/thread-labelled pipeline snapshots; workers send logs to the owning parent."""

from __future__ import annotations

import multiprocessing
import math
import os
import queue
import resource
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from multiprocessing.queues import Queue
from multiprocessing.util import Finalize
from typing import ParamSpec, TypeVar

from data_preparation.lib.download_profile import (
    DownloadProfile, Metric, current_source, install_worker_profile, measure, profile_source,
)
from data_preparation.lib.log import get_logger

log = get_logger(__name__)
P = ParamSpec("P")
T = TypeVar("T")
_SESSION: ContextVar[DebugSession | None] = ContextVar("download_debug_session", default=None)
_WORKER: WorkerReporter | None = None
WAITS = {"queue_put", "queue_get", "queue_drain", "wait_jobs", "prefetch_consumer_wait", "prefetch_discard_wait",
         "worker_result_wait", "worker_shutdown", "prefetch_shutdown",
         "token_worker_shutdown", "job_pool_shutdown"}


class Overview:
    """Difference cumulative snapshots, including active spans, without double-counting interval boundaries."""
    def __init__(self, profile: DownloadProfile) -> None:
        self.profile = profile
        self.previous_time, self.previous, _ = profile.snapshot()
        self.cpu = resource.getrusage(resource.RUSAGE_SELF)

    def render(self, *, final: bool = False) -> str:
        now, current, running = self.profile.snapshot()
        interval = now - self.previous_time
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        lines = [f"download debug pid={os.getpid()} ({multiprocessing.current_process().name}) "
                 f"sampled_at={time.time():.3f} "
                 f"{'final ' if final else ''}last={interval:.2f}s "
                 f"CPU user={cpu.ru_utime - self.cpu.ru_utime:.2f}s kernel={cpu.ru_stime - self.cpu.ru_stime:.2f}s; "
                 "inclusive thread-wall seconds (nested/parallel stages overlap)"]
        groups: dict[tuple[str, str], list[str]] = {}
        for key, value in sorted(current.items()):
            before = self.previous.get(key, Metric())
            seconds = max(0.0, value.seconds - before.seconds)
            amount = value.amount - before.amount
            ages = running.get(key, [])
            if seconds < 0.0005 and not amount and not ages:
                continue
            source, thread, phase = key
            label = f"WAIT:{phase}" if phase in WAITS else phase
            entry = f"{label}={seconds:.3f}s"
            if amount:
                entry += f"/{amount:,}"
            if ages:
                entry += f" [active={len(ages)}, oldest={max(ages):.2f}s]"
            groups.setdefault((thread, source), []).append(entry)
        for thread, source in sorted(groups, key=lambda key: (not key[0].startswith("MainThread["), key)):
            entries = groups[thread, source]
            lines.append(f"  {thread} source={source}: " + "; ".join(entries))
        if not groups:
            lines.append("  no instrumented pipeline activity in this interval")
        self.previous_time, self.previous, self.cpu = now, current, cpu
        return "\n".join(lines)


@dataclass
class WorkerDebugOptions:
    messages: Queue[str]
    interval: float
    source: str


class PeriodicReporter:
    def __init__(self, profile: DownloadProfile, interval: float, emit: Callable[[str], None]) -> None:
        self.overview = Overview(profile)
        self.interval = interval
        self.emit = emit
        self.stop = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="download-debug", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop.wait(self.interval):
                self.emit(self.overview.render())
        except Exception as error:
            self.failure = error

    def close(self) -> None:
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        if self.failure is not None:
            raise self.failure
        self.emit(self.overview.render(final=True))


class DebugSession(PeriodicReporter):
    def __init__(self, profile: DownloadProfile, interval: float) -> None:
        super().__init__(profile, interval, self._log)
        self.messages: Queue[str] = multiprocessing.get_context("spawn").Queue(maxsize=64)

    @staticmethod
    def _log(message: str) -> None:
        log.info("%s", message)  # dashboard and build.log use their existing handlers; no child writes to them

    def _drain(self) -> None:
        for _ in range(64):
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            self._log(message)

    def _run(self) -> None:
        deadline = time.monotonic() + self.interval
        try:
            while not self.stop.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
                self._drain()
                if time.monotonic() >= deadline:
                    self.emit(self.overview.render())
                    deadline = time.monotonic() + self.interval
        except Exception as error:
            self.failure = error

    def close(self) -> None:
        try:
            super().close()
            self._drain()
        finally:
            self.messages.close()
            self.messages.join_thread()


class WorkerReporter(PeriodicReporter):
    def __init__(self, profile: DownloadProfile, options: WorkerDebugOptions) -> None:
        super().__init__(profile, options.interval, self._send)
        self.messages = options.messages
        self.dropped = 0
        self.idle = profile.begin("pool_idle_or_dispatch")

    def _send(self, message: str) -> None:
        try:
            if self.dropped:
                message += f"\n  debug reports dropped due to full log queue: {self.dropped}"
            self.messages.put_nowait(message)
            self.dropped = 0
        except queue.Full:
            self.dropped += 1  # debug output must never block a process-pool result

    def close(self) -> None:
        self.overview.profile.finish(self.idle)
        super().close()


def worker_debug_options() -> WorkerDebugOptions | None:
    session = _SESSION.get()
    return None if session is None else WorkerDebugOptions(session.messages, session.interval, current_source())


def initialize_worker_debug(options: WorkerDebugOptions | None, stage: str) -> None:
    if options is None:
        return
    global _WORKER
    profile = DownloadProfile(live=True)
    install_worker_profile(profile, f"{options.source}/{stage}")
    options.messages.cancel_join_thread()  # a failed parent/logger must not deadlock worker exit
    _WORKER = WorkerReporter(profile, options)
    _WORKER.start()
    Finalize(_WORKER, _WORKER.close, exitpriority=10)


def measured_worker(phase: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorate(action: Callable[P, T]) -> Callable[P, T]:
        @wraps(action)
        def run(*args: P.args, **kwargs: P.kwargs) -> T:
            worker = _WORKER
            if worker is None:
                return action(*args, **kwargs)
            profile = worker.overview.profile
            profile.finish(worker.idle)
            try:
                with measure(phase):
                    return action(*args, **kwargs)
            finally:
                worker.idle = profile.begin("pool_idle_or_dispatch")
        return run
    return decorate


@contextmanager
def log_download_debug(profile: DownloadProfile | None, interval: float | None) -> Iterator[None]:
    if profile is None or interval is None:
        yield
        return
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("debug interval must be positive and finite")
    session = DebugSession(profile, interval)
    token = _SESSION.set(session)
    failure: BaseException | None = None
    try:
        session.start()
        with profile_source("main"):
            yield
    except BaseException as error:
        failure = error
        raise
    finally:
        _SESSION.reset(token)
        try:
            session.close()
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(f"Download debug reporting also failed: {error!r}")
