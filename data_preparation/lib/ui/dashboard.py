# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal dashboard for the preparation stages: several live progress bars plus a scrolling log section.

Built on ``rich`` (the only module that imports it). A :class:`Dashboard` owns one ``rich.live.Live`` display on
stderr that renders a fixed-height log panel (the last ``log_lines`` records) above the currently running tasks,
one line per task — so parallel downloads / processing stay readable and log records never garble a bar.

Usage (``prepare.py`` / auto-prepare wrap the build once; the stages only create tasks)::

    with Dashboard() as dashboard:
        logging.getLogger("data_preparation").addHandler(dashboard.log_handler())
        with dashboard.task("fineweb_edu: download", total=1000, unit="row") as bar:
            bar.update(1); bar.set_postfix({"file": "x.parquet"})

:class:`Task` implements the same interface as ``lib.progress.Progress`` (``update`` / ``set_postfix`` /
``set_description`` / ``close`` / iteration / context manager / ``n``), and :func:`progress` has the signature of
``lib.progress.progress``: it creates a task on the active dashboard, or falls back to the tqdm / no-op bar when
none is active. Switching a stage over is therefore a one-line import change; nothing else in the stages has to
know about the dashboard.

Disabled (``DATA_PREP_PROGRESS=0`` or stderr not a terminal — the same rule as ``lib.progress``): no live display,
tasks are no-ops and the log handler writes plain lines to stderr. Thread-safe: tasks may be created and updated
from worker threads; the display refreshes on its own timer.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from collections.abc import Iterable, Iterator, Sized
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO, TypeVar

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress as RichProgress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.progress import Task as RichTask
from rich.text import Text

from data_preparation.lib.log import LOG_FORMAT
from data_preparation.lib.progress import NoProgress, Progress, progress_enabled
from data_preparation.lib.progress import progress as fallback_progress

T = TypeVar("T")

DEFAULT_LOG_LINES = 12
DEFAULT_REFRESH_PER_SECOND = 8
BUILD_LOG_NAME = "build.log"  # full log of every build, appended under the dataset root (`Dashboard.attach(log_file=)`)


def _format_postfix(values: dict[str, Any]) -> str:
    """``key=value`` pairs joined by commas (tqdm's postfix style)."""
    return ", ".join(f"{key}={value}" for key, value in values.items())


class _RateColumn(ProgressColumn):
    """Rows (or whatever the task's unit is) per second, ``?`` until the first update."""

    def render(self, task: RichTask) -> Text:
        speed = task.finished_speed or task.speed
        unit = str(task.fields.get("unit", "it"))
        if speed is None:
            return Text(f"? {unit}/s", style="progress.data.speed")
        return Text(f"{speed:,.0f} {unit}/s", style="progress.data.speed")


class _PostfixColumn(ProgressColumn):
    """The task's postfix (``set_postfix``), dimmed, after the timing columns."""

    def render(self, task: RichTask) -> Text:
        return Text(str(task.fields.get("postfix", "")), style="dim")


class Task:
    """One progress bar of a :class:`Dashboard`; the ``lib.progress.Progress`` interface over a rich task.

    ``n`` mirrors tqdm's counter (updates done). ``leave=False`` removes the line on ``close``, ``leave=True`` keeps
    it (completed) until the dashboard stops. The bar may overshoot its ``total`` like the download bar does (a
    loader finishing a remote row group); rich renders that as a full bar with ``completed/total`` past 100 %.
    """

    def __init__(self, owner: RichProgress, task_id: TaskID, iterable: Iterable[Any] | None, leave: bool) -> None:
        self._owner = owner
        self._id = task_id
        self._iterable = iterable
        self._leave = leave
        self._closed = False
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n
        if not self._closed:  # like tqdm, a late update is ignored (rich would raise on a removed task)
            self._owner.update(self._id, advance=n)

    def set_postfix(self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any) -> None:
        values: dict[str, Any] = dict(ordered_dict or {})
        values.update(kwargs)
        if not self._closed:
            self._owner.update(self._id, postfix=_format_postfix(values))

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        if not self._closed:
            self._owner.update(self._id, description=desc or "")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._leave:
            task = self._owner.tasks[self._owner.task_ids.index(self._id)]
            if task.total is None:
                self._owner.update(self._id, total=task.completed)  # an indeterminate bar renders as done
            self._owner.stop_task(self._id)
        else:
            self._owner.remove_task(self._id)

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            return
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self) -> Task:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


class DashboardLogHandler(logging.Handler):
    """``logging.Handler`` whose records land in the dashboard's log panel (or on ``stream`` when it is disabled).

    Records of ``keep_level`` and above (default WARNING), and records logged with ``extra={"keep": True}`` (the
    plan / status tables), are additionally printed above the live display, unwrapped, where they scroll into the
    terminal's history and survive the run — the panel only shows the last few lines."""

    def __init__(self, dashboard: Dashboard, level: int = logging.NOTSET, keep_level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._dashboard = dashboard
        self._keep_level = keep_level
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            keep = record.levelno >= self._keep_level or bool(getattr(record, "keep", False))
            self._dashboard.write(text, keep=keep)
        except Exception:
            self.handleError(record)


class Dashboard:
    """Live terminal display: a log panel (last ``log_lines`` lines) above one progress line per running task.

    ``enabled`` defaults to :func:`lib.progress.progress_enabled` (env var + TTY check); ``console`` is for tests
    (a ``rich.console.Console`` over a ``StringIO``). Only one dashboard can be active at a time (``with`` block);
    :func:`progress` and :func:`active_dashboard` find it. Nested ``with`` blocks reuse the active one.
    """

    _active: Dashboard | None = None
    _active_lock = threading.RLock()  # re-entrant: a nested `with Dashboard()` delegates to the active one

    def __init__(
        self,
        *,
        log_lines: int = DEFAULT_LOG_LINES,
        refresh_per_second: float = DEFAULT_REFRESH_PER_SECOND,
        enabled: bool | None = None,
        console: Console | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.enabled = progress_enabled(stream) if enabled is None else enabled
        self._stream = stream if stream is not None else sys.stderr
        self._console = console if console is not None else Console(file=self._stream)
        self._refresh_per_second = refresh_per_second
        self._lines: deque[str] = deque(maxlen=log_lines)
        self._lines_lock = threading.Lock()
        self._progress = RichProgress(
            SpinnerColumn(finished_text="[green]✓[/green]"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>4.0f}%"),
            _RateColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            _PostfixColumn(),
            console=self._console,
            expand=True,
        )
        self._live: Live | None = None
        self._depth = 0
        self._saved_env: dict[str, str | None] = {}

    # --- lifecycle --------------------------------------------------------------------------------------------------

    def __enter__(self) -> Dashboard:
        with Dashboard._active_lock:
            if Dashboard._active is not None and Dashboard._active is not self:
                return Dashboard._active.__enter__()  # increments the active dashboard's depth; its __exit__ undoes it
            self._depth += 1
            if self._depth > 1:
                return self
            Dashboard._active = self
        if self.enabled:
            self._silence_third_party_bars()
            self._live = Live(
                self, console=self._console, refresh_per_second=self._refresh_per_second, transient=False, redirect_stderr=False,
                redirect_stdout=False,
            )
            self._live.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        with Dashboard._active_lock:
            active = Dashboard._active
            if active is not None and active is not self:  # this block was delegated to the active dashboard
                active.__exit__(exc_type, exc_value, traceback)
                return
            self._depth -= 1
            if self._depth > 0:
                return
            Dashboard._active = None
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._restore_third_party_bars()

    _THIRD_PARTY_BAR_ENV = ("HF_HUB_DISABLE_PROGRESS_BARS", "HF_DATASETS_DISABLE_PROGRESS_BARS")

    def _silence_third_party_bars(self) -> None:
        """Turn off the tqdm bars of ``huggingface_hub`` / ``datasets`` (file downloads, ``load_dataset``) while
        the live display is up — their carriage returns would garble it. The env vars cover the not-yet-imported
        libraries, the function calls the already-imported ones; :meth:`_restore_third_party_bars` undoes both."""
        self._saved_env = {name: os.environ.get(name) for name in self._THIRD_PARTY_BAR_ENV}
        for name in self._THIRD_PARTY_BAR_ENV:
            os.environ[name] = "1"
        hub = sys.modules.get("huggingface_hub")
        if hub is not None:
            hub.utils.disable_progress_bars()
        datasets = sys.modules.get("datasets")
        if datasets is not None:
            datasets.utils.logging.disable_progress_bar()

    def _restore_third_party_bars(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        hub = sys.modules.get("huggingface_hub")
        if hub is not None and self._saved_env["HF_HUB_DISABLE_PROGRESS_BARS"] is None:
            hub.utils.enable_progress_bars()
        datasets = sys.modules.get("datasets")
        if datasets is not None and self._saved_env["HF_DATASETS_DISABLE_PROGRESS_BARS"] is None:
            datasets.utils.logging.enable_progress_bar()

    @property
    def is_active(self) -> bool:
        return Dashboard._active is self

    # --- tasks and log lines --------------------------------------------------------------------------------------------

    def task(
        self,
        desc: str,
        *,
        total: int | None = None,
        unit: str = "row",
        leave: bool = True,
        iterable: Iterable[T] | None = None,
    ) -> Progress:
        """A new progress line (a no-op bar when the dashboard is disabled)."""
        if not self.enabled:
            return NoProgress(iterable)
        if total is None and iterable is not None and isinstance(iterable, Sized):
            total = len(iterable)  # like tqdm
        task_id = self._progress.add_task(desc, total=total, unit=unit, postfix="")
        return Task(self._progress, task_id, iterable, leave)

    def write(self, text: str, *, keep: bool = False) -> None:
        """Append ``text`` (one entry per line, so tracebacks stay readable) to the log panel; plain stderr when
        disabled. With ``keep`` the text is also printed above the live display, into the terminal's scrollback."""
        if not self.enabled:
            self._stream.write(text + "\n")
            self._stream.flush()
            return
        with self._lines_lock:
            self._lines.extend(text.splitlines() or [""])
        if keep:
            # rich prints above an active Live display; no wrapping / cropping so tables keep their columns
            self._console.print(Text(text, no_wrap=True, overflow="ignore"), crop=False, soft_wrap=True)

    def log_handler(self, level: int = logging.NOTSET, keep_level: int = logging.WARNING) -> DashboardLogHandler:
        """A logging handler for this dashboard (see :class:`DashboardLogHandler`); :meth:`attach` installs it."""
        return DashboardLogHandler(self, level, keep_level)

    @contextmanager
    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> Iterator[None]:
        """Route ``logger`` into this dashboard for the duration of the block: the plain stream handlers that
        ``lib.log.configure_logging`` installed are detached (their lines would print twice and garble the live
        display) and restored afterwards; with ``log_file`` every record is also appended to that file."""
        detached: list[logging.Handler] = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        added: list[logging.Handler] = [self.log_handler()]
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            added.append(file_handler)
        for handler in detached:
            logger.removeHandler(handler)
        for handler in added:
            logger.addHandler(handler)
        try:
            yield
        finally:
            for handler in added:
                logger.removeHandler(handler)
                handler.close()
            for handler in detached:
                logger.addHandler(handler)

    def lines(self) -> list[str]:
        """The log lines currently shown (newest last)."""
        with self._lines_lock:
            return list(self._lines)

    @property
    def tasks(self) -> list[RichTask]:
        """The rich tasks currently in the display (running and finished-but-left)."""
        return list(self._progress.tasks)

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich__(self) -> RenderableType:
        with self._lines_lock:
            body = Text("\n".join(self._lines) or "(no log output yet)", no_wrap=True, overflow="ellipsis")
        # bars first: a terminal too short for both crops the log panel, never the running tasks
        return Group(self._progress, Panel(body, title="log", title_align="left", border_style="dim"))

    def render_text(self, width: int = 120) -> str:
        """The current display as plain text (tests, or a snapshot for a log file)."""
        console = Console(width=width, force_terminal=False, color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()


def active_dashboard() -> Dashboard | None:
    """The dashboard of the enclosing ``with Dashboard()`` block, if any."""
    return Dashboard._active


def progress(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "row",
    leave: bool = True,
) -> Progress:
    """Drop-in for ``lib.progress.progress``: a task on the active dashboard, else the tqdm / no-op bar."""
    dashboard = active_dashboard()
    if dashboard is None:
        return fallback_progress(iterable, total=total, desc=desc, unit=unit, leave=leave)
    return dashboard.task(desc, total=total, unit=unit, leave=leave, iterable=iterable)
