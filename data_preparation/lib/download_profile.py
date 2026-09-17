# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Opt-in, bounded download timings. Context follows owned threads, never dataset identity."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")
_ACTIVE: ContextVar[DownloadProfile | None] = ContextVar("download_profile", default=None)
_SOURCE: ContextVar[str] = ContextVar("download_profile_source", default="")
MAX_METRICS = 2048
MetricKey = tuple[str, str, str]


@dataclass
class Metric:
    calls: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    amount: int = 0


class DownloadProfile:
    def __init__(self, *, live: bool = False) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[MetricKey, Metric] = {}
        self._running: dict[int, tuple[MetricKey, float]] = {}
        self._sequence = 0
        self.live = live
        self.started = time.perf_counter()

    def record(self, phase: str, elapsed: float, *, calls: int = 1, amount: int = 0, maximum: float | None = None) -> None:
        """Merge a batch/operation, not individual row events; cap dynamic source cardinality."""
        key = (_SOURCE.get(), self._thread_label(), phase)
        with self._lock:
            metric = self._metrics[self._register(key)]
            metric.calls += calls
            metric.seconds += elapsed
            metric.max_seconds = max(metric.max_seconds, elapsed if maximum is None else maximum)
            metric.amount += amount

    @staticmethod
    def _thread_label() -> str:
        return f"{threading.current_thread().name}[{threading.get_native_id()}]"

    def _register(self, key: MetricKey) -> MetricKey:
        if key not in self._metrics and len(self._metrics) >= MAX_METRICS:
            key = ("overflow", "other", "additional_metrics")
        self._metrics.setdefault(key, Metric())
        return key

    def begin(self, phase: str) -> int:
        with self._lock:
            key = self._register((_SOURCE.get(), self._thread_label(), phase))
            self._sequence += 1
            self._running[self._sequence] = (key, time.perf_counter())
            return self._sequence

    def finish(self, span: int) -> None:
        with self._lock:
            key, started = self._running.pop(span)
            elapsed = time.perf_counter() - started
            metric = self._metrics[key]
            metric.calls += 1
            metric.seconds += elapsed
            metric.max_seconds = max(metric.max_seconds, elapsed)

    def snapshot(self) -> tuple[float, dict[MetricKey, Metric], dict[MetricKey, list[float]]]:
        """Include unfinished spans so interval deltas account for a stalled operation immediately."""
        with self._lock:
            now = time.perf_counter()
            metrics = {key: Metric(**asdict(value)) for key, value in self._metrics.items()}
            running: dict[MetricKey, list[float]] = {}
            for key, started in self._running.values():
                elapsed = now - started
                metrics[key].seconds += elapsed
                running.setdefault(key, []).append(elapsed)
        return now, metrics, running



def active_profile() -> DownloadProfile | None:
    return _ACTIVE.get() if _SOURCE.get() else None


def current_source() -> str:
    return _SOURCE.get()


def install_worker_profile(profile: DownloadProfile, source: str) -> None:
    """Initialize the dedicated spawn worker's lifetime context (never called in a parent thread)."""
    _ACTIVE.set(profile)
    _SOURCE.set(source)


def bind_profile(action: Callable[P, T]) -> Callable[P, T]:
    """Capture a distinct context per submission; disabled profiling returns the original callable."""
    if _ACTIVE.get() is None:
        return action
    context = copy_context()

    @wraps(action)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        return context.copy().run(action, *args, **kwargs)

    return run


@contextmanager
def profile_source(name: str) -> Iterator[None]:
    if _ACTIVE.get() is None:
        yield
        return
    token = _SOURCE.set(name)
    try:
        yield
    finally:
        _SOURCE.reset(token)


@contextmanager
def measure(phase: str) -> Iterator[None]:
    profile = active_profile()
    if profile is None:
        yield
        return
    if profile.live:
        span = profile.begin(phase)
        try:
            yield
        finally:
            profile.finish(span)
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        profile.record(phase, time.perf_counter() - started)


def measured(phase: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorate(action: Callable[P, T]) -> Callable[P, T]:
        @wraps(action)
        def run(*args: P.args, **kwargs: P.kwargs) -> T:
            if active_profile() is None:
                return action(*args, **kwargs)
            with measure(phase):
                return action(*args, **kwargs)
        return run
    return decorate


def measure_rows(rows: Iterator[T]) -> Iterator[T]:
    """Time iterator demand in batches, excluding time suspended while the consumer handles a row."""
    profile = active_profile()
    if profile is None:
        return rows

    def iterate() -> Iterator[T]:
        if profile.live:
            while True:
                with measure("input_next"):
                    try:
                        row = next(rows)
                    except StopIteration:
                        return
                with measure("fetch_dispatch"):
                    yield row
        elapsed = maximum = 0.0
        calls = 0
        try:
            while True:
                started = time.perf_counter()
                try:
                    row = next(rows)
                except StopIteration:
                    return
                finally:
                    duration = time.perf_counter() - started
                    elapsed += duration
                    maximum = max(maximum, duration)
                    calls += 1
                if calls >= 256:
                    profile.record("input_next", elapsed, calls=calls, maximum=maximum)
                    elapsed = maximum = 0.0
                    calls = 0
                yield row
        finally:
            if calls:
                profile.record("input_next", elapsed, calls=calls, maximum=maximum)
    return iterate()



@contextmanager
def profile_downloads(*, dry_run: bool = False, debug: float | None = None) -> Iterator[DownloadProfile | None]:
    """Enable live timing only for a requested debug run; leave dataset identity and output untouched."""
    if debug is None or dry_run:
        yield None
        return
    profile = DownloadProfile(live=True)
    token = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(token)
