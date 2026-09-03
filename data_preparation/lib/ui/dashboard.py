# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal dashboard for the preparation stages: one live layout (header, a **downloads** panel, a **builds**
panel, the **log** panel, a footer) and nothing else on the terminal while it is up.

A :class:`DataDashboard` is a :class:`ui.display.LiveDisplay` with one ``rich.live.Live`` display on stderr. Every
progress bar is a :class:`Task` rendered inside one of the panels: one row per running task (at most ``max_rows``,
then "… and k more"); finished rows disappear into the panel's summary line (jobs done, rows done / wanted, bytes
fetched, elapsed). A download row also shows its bytes and current speed (a live byte counter, sampled per frame
over the last :data:`DOWNLOAD_RATE_WINDOW` seconds). Updates from worker threads go through one lock; the display
refreshes on its own timer.

While the display is up nothing may print around it (a stray line shifts the frame), so ``__enter__`` also, through
:mod:`ui.capture`, routes every ``logging`` record into the log panel (the plain ``StreamHandler``\\s of libraries
such as ``huggingface_hub`` / ``datasets`` are detached for the duration), replaces ``sys.stdout`` / ``sys.stderr``
with line sinks that log what is written to them, and silences the tqdm bars of ``huggingface_hub`` / ``datasets``.

Records of WARNING and above, and records logged with ``extra={"keep": True}`` (the plan / status tables), are
*kept*: shown in the panel and printed once, unwrapped, after the display closed. The scrollback of a run is the
kept lines followed by the final table, never a frozen frame (the display is transient). Ctrl-C / an exception
leave through the same path.

Usage (``prepare.py`` / auto-prepare wrap the build once; the stages only create tasks)::

    with DataDashboard(title="prepare tiny") as dashboard, dashboard.attach(logging.getLogger("data_preparation")):
        with progress(total=1000, desc="fineweb_edu", unit="row", panel="downloads", bytes_fetched=lambda: stats.bytes_fetched) as bar:
            bar.update(1); bar.set_postfix({"file": "x.parquet"})

:class:`Task` implements ``lib.progress.Progress`` (``update`` / ``set_postfix`` / context manager / ``n`` /
``total``); :func:`progress` creates a task in ``panel`` of the active dashboard (``summary`` makes it the panel's
one summary task, e.g. the jobs of a pool) and returns the no-op ``NoProgress`` when none is active.
:func:`set_status` puts key/value pairs (round, step) into the header; :func:`suspended` clears the display around
a terminal prompt.

Disabled (``DATA_PREP_PROGRESS=0`` or stderr not a terminal, the rule of ``lib.progress``): no live display, no
capture, tasks are no-ops and the log handler writes plain lines to stderr. A terminal that dies mid-run closes the
display and the run continues headless with ``build.log`` as its output (:mod:`ui.display`).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO, TypeVar

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import NoProgress, Progress, progress_enabled
from ui.capture import DashboardLogHandler, LoggingCapture, StreamCapture, attach_logger
from ui.display import LiveDisplay

T = TypeVar("T")

DEFAULT_LOG_LINES = 12
DEFAULT_MAX_ROWS = 8  # running tasks shown per panel; the rest is "… and k more"
DEFAULT_REFRESH_PER_SECOND = 8
DEFAULT_PANELS: tuple[str, ...] = ("downloads", "builds")  # always shown, in this order; other panel names appear on demand
DEFAULT_PANEL = "tasks"  # tasks created without a panel name
DOWNLOAD_RATE_WINDOW = 5.0  # seconds of byte samples behind a download row's "current" speed
BUILD_LOG_NAME = "build.log"  # full log of every build, appended under the dataset root (`DataDashboard.attach(log_file=)`)
STDOUT_LOGGER = "data_preparation.stdout"  # lines written to sys.stdout while the display is up (INFO)
STDERR_LOGGER = "data_preparation.stderr"  # lines written to sys.stderr while the display is up (WARNING: kept)
FOOTER_HINT = "Ctrl-C stops at the next shard; everything published so far is kept"


def _format_postfix(values: dict[str, Any]) -> str:
    """``key=value`` pairs joined by commas (tqdm's postfix style)."""
    return ", ".join(f"{key}={value}" for key, value in values.items())


def _format_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def _format_bytes(count: int) -> str:
    """``320 MB``, ``1.25 GB`` from a gibibyte on."""
    megabytes = count / 2**20
    return f"{megabytes / 1024:,.2f} GB" if megabytes >= 1024 else f"{megabytes:,.0f} MB"


def _format_byte_rate(bytes_per_second: float) -> str:
    """``3.2 MB/s``, ``87 kB/s`` below a tenth of a megabyte per second."""
    megabytes = bytes_per_second / 2**20
    return f"{megabytes:,.1f} MB/s" if megabytes >= 0.1 else f"{bytes_per_second / 2**10:,.0f} kB/s"


def _download_cell(task: Task) -> str:
    """``3.2 MB/s · 320 MB`` for a task that fetched bytes (the speed once its samples span a second), else empty."""
    fetched = task.bytes_fetched
    if fetched <= 0:
        return ""
    rate = task.download_rate()
    total = _format_bytes(fetched)
    return total if rate is None else f"{_format_byte_rate(rate)} · {total}"


class Task:
    """One row of a dashboard panel; the ``lib.progress.Progress`` interface.

    ``n`` counts the updates. The row is shown while the task is open and disappears on ``close``; its counts then
    live on in the panel's summary line until the next summary task (round) starts. The bar may overshoot its
    ``total`` like the download bar does (a loader finishing a remote row group); it renders full, the count shows
    ``completed/total`` past 100 %. ``bytes_fetched`` is a download task's live byte counter: the row shows its
    value and the speed over the last frames, the summary line adds it up, ``close`` freezes it.
    """

    def __init__(
        self,
        dashboard: DataDashboard,
        panel: _PanelState,
        description: str,
        *,
        total: int | None,
        unit: str,
        summary: bool,
        bytes_fetched: Callable[[], int] | None = None,
    ) -> None:
        self._dashboard = dashboard
        self._panel = panel
        self.description = description
        self.total = total
        self.unit = unit
        self.summary = summary
        self.completed = 0
        self.postfix: dict[str, Any] = {}
        self.started = dashboard._clock()
        self.finished: float | None = None
        self._bytes_fetched = bytes_fetched
        self._bytes_at_close = 0
        self._byte_samples: deque[tuple[float, int]] = deque()  # (time, bytes) of the last frames, for `download_rate`

    @property
    def n(self) -> int:
        return self.completed

    @property
    def closed(self) -> bool:
        return self.finished is not None

    def update(self, n: int = 1) -> None:
        with self._dashboard._lock:
            self.completed += n

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        values: dict[str, Any] = dict(ordered_dict or {})
        values.update(kwargs)
        with self._dashboard._lock:
            self.postfix = values

    def close(self) -> None:
        with self._dashboard._lock:
            if self.finished is not None:
                return
            self._bytes_at_close = self.bytes_fetched
            self.finished = self._dashboard._clock()
            self._panel.finish(self)

    @property
    def bytes_fetched(self) -> int:
        """The bytes the task fetched so far (0 without a counter); frozen once closed."""
        if self._bytes_fetched is None:
            return 0
        return self._bytes_at_close if self.closed else self._bytes_fetched()

    def sample_bytes(self, now: float) -> None:
        """Record this frame's byte count; samples older than :data:`DOWNLOAD_RATE_WINDOW` drop out."""
        samples = self._byte_samples
        samples.append((now, self.bytes_fetched))
        while samples and now - samples[0][0] > DOWNLOAD_RATE_WINDOW:
            samples.popleft()

    def download_rate(self) -> float | None:
        """Bytes per second over the sampled frames; None until they span at least a second."""
        if len(self._byte_samples) < 2:
            return None
        (first_time, first_bytes), (last_time, last_bytes) = self._byte_samples[0], self._byte_samples[-1]
        if last_time - first_time < 1.0:
            return None
        return (last_bytes - first_bytes) / (last_time - first_time)

    def elapsed(self, now: float) -> float:
        return (self.finished if self.finished is not None else now) - self.started

    def speed(self, now: float) -> float | None:
        """Units per second over the task's lifetime; None before the first update."""
        elapsed = self.elapsed(now)
        if self.completed <= 0 or elapsed <= 0:
            return None
        return self.completed / elapsed

    def __enter__(self) -> Task:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


class _PanelState:
    """The tasks of one panel: the running rows, the finished ones of the current round, the round's summary task."""

    def __init__(self, name: str, max_rows: int) -> None:
        self.name = name
        self.max_rows = max_rows
        self.active: list[Task] = []
        self.done: list[Task] = []
        self.summary: Task | None = None

    def add(self, task: Task) -> None:
        if task.summary:  # a new round: the finished rows of the previous one are dropped
            self.summary = task
            self.done = []
        else:
            self.active.append(task)

    def finish(self, task: Task) -> None:
        if task.summary:
            return  # the summary stays until the next round's replaces it
        if task in self.active:
            self.active.remove(task)
            self.done.append(task)

    @property
    def idle(self) -> bool:
        return not self.active and not self.done and self.summary is None

    def height(self) -> int:
        """Lines the panel occupies: borders, rows (bounded), the overflow line, the summary line."""
        rows = min(len(self.active), self.max_rows) or (0 if self.idle else 1)
        return 2 + rows + (1 if len(self.active) > self.max_rows else 0) + 1

    # --- rendering ----------------------------------------------------------------------------------------------------

    def render(self, now: float) -> Panel:
        body: list[RenderableType] = []
        if self.active:
            body.append(self._rows_table(self.active[: self.max_rows], now))
            if len(self.active) > self.max_rows:
                body.append(Text(f"… and {len(self.active) - self.max_rows} more", style="dim"))
        elif not self.idle:
            body.append(Text("no running task", style="dim"))
        body.append(self._summary_line(now))  # "idle" when the panel never had a task
        return Panel(Group(*body), title=self.name, title_align="left", border_style="dim", padding=(0, 1))

    @staticmethod
    def _rows_table(tasks: list[Task], now: float) -> Table:
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(no_wrap=True, overflow="ellipsis", max_width=40, style="bold")  # description
        table.add_column(ratio=2, min_width=10)  # bar
        table.add_column(justify="right", no_wrap=True)  # completed/total
        table.add_column(justify="right", no_wrap=True, style="progress.data.speed")  # rate
        table.add_column(justify="right", no_wrap=True, style="progress.download")  # download speed · bytes fetched
        table.add_column(no_wrap=True, style="progress.elapsed")  # elapsed
        table.add_column(ratio=3, no_wrap=True, overflow="ellipsis", style="dim")  # postfix
        for task in tasks:
            speed = task.speed(now)
            task.sample_bytes(now)
            table.add_row(
                task.description,
                ProgressBar(total=task.total, completed=min(task.completed, task.total) if task.total is not None else task.completed, pulse=task.total is None),
                f"{task.completed:,}/{task.total:,}" if task.total is not None else f"{task.completed:,}",
                f"? {task.unit}/s" if speed is None else f"{speed:,.0f} {task.unit}/s",
                _download_cell(task),
                _format_elapsed(task.elapsed(now)),
                _format_postfix(task.postfix),
            )
        return table

    def _summary_line(self, now: float) -> Text:
        """``5/18 jobs done · 71,500/75,000 rows · 320 MB · 0:01:03``: the round's jobs, every row of every task
        of the round (running and finished), the bytes they fetched, the time since the round began."""
        if self.idle:
            return Text("idle", style="dim")
        tasks = [*self.done, *self.active]
        parts: list[str] = []
        summary = self.summary
        if summary is not None:
            jobs = f"{summary.completed:,}/{summary.total:,}" if summary.total is not None else f"{summary.completed:,}"
            parts.append(f"{jobs} {summary.unit}s done")
        if tasks:
            completed = sum(task.completed for task in tasks)
            totals = [task.total for task in tasks if task.total is not None]
            rows = f"{completed:,}/{sum(totals):,}" if len(totals) == len(tasks) else f"{completed:,}"
            parts.append(f"{rows} {tasks[0].unit}s")
        fetched = sum(task.bytes_fetched for task in tasks)
        if fetched > 0:
            parts.append(_format_bytes(fetched))
        started = min([task.started for task in tasks] + ([summary.started] if summary is not None else []))
        running = bool(self.active) or (summary is not None and not summary.closed)
        end = now if running else max([task.finished or now for task in tasks] + ([summary.finished or now] if summary is not None else []))
        parts.append(_format_elapsed(end - started))
        return Text(" · ".join(parts), style="bold" if running else "dim")


class DataDashboard(LiveDisplay):
    """Live terminal display: header, one panel per task group (``downloads``, ``builds``, …), the log panel (last
    ``log_lines`` lines), footer.

    ``enabled`` defaults to :func:`lib.progress.progress_enabled` (env var + TTY check); ``console`` is for tests
    (a ``rich.console.Console`` over a ``StringIO``), ``clock`` too (elapsed times and download speeds). Exactly one
    dashboard is active at a time (``with`` block); :func:`progress` and :func:`active_dashboard` find it, entering
    a second one raises.
    """

    _active: DataDashboard | None = None

    def __init__(
        self,
        *,
        title: str | None = None,
        panels: Iterable[str] = DEFAULT_PANELS,
        max_rows: int = DEFAULT_MAX_ROWS,
        log_lines: int = DEFAULT_LOG_LINES,
        refresh_per_second: float = DEFAULT_REFRESH_PER_SECOND,
        enabled: bool | None = None,
        console: Console | None = None,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(
            stream=stream if stream is not None else sys.stderr, console=console, refresh_per_second=refresh_per_second, log_lines=log_lines
        )
        self.enabled = progress_enabled(stream) if enabled is None else enabled
        self.logger = get_logger(__name__)
        self.title = title or "data preparation"
        self._opened = False  # the display was set up by `__enter__` (torn down by `__exit__` even once headless)
        self._max_rows = max_rows
        self._panels: dict[str, _PanelState] = {name: _PanelState(name, max_rows) for name in panels}
        self._status: dict[str, str] = {}
        self._clock = clock
        self._started_at = clock()
        self._saved_env: dict[str, str | None] = {}
        self._silenced_modules: list[str] = []
        self._logging_capture = LoggingCapture(self, already_attached=self.is_attached)
        self._stream_capture = StreamCapture(STDOUT_LOGGER, STDERR_LOGGER)

    # --- lifecycle --------------------------------------------------------------------------------------------------

    def __enter__(self) -> DataDashboard:
        if DataDashboard._active is not None:
            raise RuntimeError("a DataDashboard is already active")
        DataDashboard._active = self
        if self.enabled:
            self._opened = True
            self._started_at = self._clock()
            self._silence_third_party_bars()
            self._capture_logging()
            self._redirect_streams()
            self._start_live()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        DataDashboard._active = None
        if not self._opened:
            return
        try:
            self._stop_live()
        finally:
            self._release_streams()
            self._release_logging()
            self._restore_third_party_bars()
        self._print_kept()

    @property
    def is_active(self) -> bool:
        return DataDashboard._active is self

    # --- what else could reach the terminal ----------------------------------------------------------------------------

    _THIRD_PARTY_BAR_ENV = ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_DATASETS_DISABLE_PROGRESS_BARS")

    def _silence_third_party_bars(self) -> None:
        """Turn off the tqdm bars of ``huggingface_hub`` / ``datasets`` (file downloads, ``load_dataset``) while
        the live display is up. The env vars cover the not-yet-imported libraries (which then stay silent for the
        rest of the process: they read the variable once, at import), the function calls the already-imported
        ones; :meth:`_restore_third_party_bars` undoes both."""
        self._saved_env = {name: os.environ.get(name) for name in self._THIRD_PARTY_BAR_ENV}
        for name in self._THIRD_PARTY_BAR_ENV:
            os.environ[name] = "1"
        self._silenced_modules = []
        hub = sys.modules.get("huggingface_hub")
        if hub is not None:
            hub.utils.disable_progress_bars()
            self._silenced_modules.append("huggingface_hub")
        datasets = sys.modules.get("datasets")
        if datasets is not None:
            datasets.utils.logging.disable_progress_bar()
            self._silenced_modules.append("datasets")

    def _restore_third_party_bars(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if "huggingface_hub" in self._silenced_modules and self._saved_env["HF_HUB_DISABLE_PROGRESS_BARS"] is None:
            sys.modules["huggingface_hub"].utils.enable_progress_bars()
        if "datasets" in self._silenced_modules and self._saved_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] is None:
            sys.modules["datasets"].utils.logging.enable_progress_bar()

    def _capture_logging(self) -> None:
        """Every ``logging`` record into the panel: the dashboard's handler on the root logger, and every plain
        console ``StreamHandler`` of every logger (``huggingface_hub`` and ``datasets`` install one on theirs at
        import) detached until :meth:`_release_logging` (:class:`~ui.capture.LoggingCapture`)."""
        self._logging_capture.start()

    def _release_logging(self) -> None:
        self._logging_capture.stop()

    def _redirect_streams(self) -> None:
        """``sys.stdout`` / ``sys.stderr`` become line sinks that log (INFO / WARNING) what is written to them."""
        self._stream_capture.redirect()

    def _release_streams(self) -> None:
        self._stream_capture.release()

    # --- tasks, status and log lines ------------------------------------------------------------------------------------

    def task(
        self,
        desc: str,
        *,
        total: int | None = None,
        unit: str = "row",
        panel: str | None = None,
        summary: bool = False,
        bytes_fetched: Callable[[], int] | None = None,
    ) -> Progress:
        """A new row in ``panel`` (a no-op bar when the dashboard is disabled). ``summary`` makes it the panel's
        summary task (the pool's jobs) and starts a new round of the panel's counts; ``bytes_fetched`` is a download
        task's live byte counter (the row shows its bytes and current speed, the summary line adds it up)."""
        if not self.enabled:
            return NoProgress(total)
        with self._lock:
            state = self._panels.setdefault(panel or DEFAULT_PANEL, _PanelState(panel or DEFAULT_PANEL, self._max_rows))
            task = Task(self, state, desc, total=total, unit=unit, summary=summary, bytes_fetched=bytes_fetched)
            state.add(task)
        return task

    def set_status(self, **fields: object) -> None:
        """Header fields (``round="1/5"``, ``step="download"``), shown as ``key value`` after the title."""
        with self._lock:
            self._status.update({key: str(value) for key, value in fields.items()})

    def log_handler(self, level: int = logging.NOTSET, keep_level: int = logging.WARNING) -> DashboardLogHandler:
        """A logging handler for this dashboard (see :class:`DashboardLogHandler`); :meth:`attach` installs it."""
        return DashboardLogHandler(self, level, keep_level)

    @contextmanager
    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> Iterator[None]:
        """Route ``logger`` into this dashboard for the duration of the block: the plain stream handlers that
        ``lib.log.configure_logging`` installed are detached (their lines would print behind the live display) and
        restored afterwards; with ``log_file`` every record is also appended to that file (named in the footer)."""
        with attach_logger(self, logger, log_file):
            with self._lock:
                self._attached_logger_names.append(logger.name)
                if log_file is not None:
                    self._log_file = log_file
            try:
                yield
            finally:
                with self._lock:
                    self._attached_logger_names.remove(logger.name)

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        now = self._clock()
        with self._lock:
            panels = list(self._panels.values())
            fixed_height = 2 + sum(state.height() for state in panels)  # header + footer + panels
            log_height = max(3, min(self._panel_height, options.size.height - fixed_height - 2))
            status = "".join(f" · {key} {value}" for key, value in self._status.items())
            header = Text.assemble((self.title, "bold"), status, (f" · {_format_elapsed(now - self._started_at)}", "dim"), no_wrap=True, overflow="ellipsis")
            rendered = [state.render(now) for state in panels]
            log_panel = self._render_log(log_height)
            footer = self._footer(FOOTER_HINT)
        yield Group(header, *rendered, log_panel, footer)


def active_dashboard() -> DataDashboard | None:
    """The dashboard of the enclosing ``with DataDashboard()`` block, if any."""
    return DataDashboard._active


def progress(
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "row",
    panel: str | None = None,
    summary: bool = False,
    bytes_fetched: Callable[[], int] | None = None,
) -> Progress:
    """A task in ``panel`` of the active dashboard (see :meth:`DataDashboard.task`), else a :class:`NoProgress`
    (which counts, shows nothing)."""
    dashboard = active_dashboard()
    if dashboard is None:
        return NoProgress(total)
    return dashboard.task(desc, total=total, unit=unit, panel=panel, summary=summary, bytes_fetched=bytes_fetched)


def set_status(**fields: object) -> None:
    """Header fields of the active dashboard (no-op without one)."""
    dashboard = active_dashboard()
    if dashboard is not None:
        dashboard.set_status(**fields)


@contextmanager
def suspended() -> Iterator[None]:
    """The active dashboard's display cleared for the block (a terminal prompt); no-op without one."""
    dashboard = active_dashboard()
    if dashboard is None:
        yield
        return
    with dashboard.suspended():
        yield
