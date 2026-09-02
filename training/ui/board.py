# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The live training dashboard: one transient ``rich.live.Live`` layout — header, stage bars, metrics, validation,
events, log panel, footer — with the terminal captured around it (:mod:`training.ui.capture`). The module docstring
of :mod:`training.ui.dashboard` describes the whole picture."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TextIO

from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from training.ui.capture import TerminalCapture, attach_logger
from training.ui.common import TRAINING_LOGGER_NAME, Clock, log
from training.ui.fallback import NoOpDashboard
from training.ui.format import (
    METRIC_COLUMNS,
    event_line,
    fit_panel_heights,
    floats,
    format_duration,
    format_metric,
    known_metrics,
    line,
    status_line,
    validation_line,
)

DEFAULT_LOG_LINES = 12
DEFAULT_EVENT_LINES = 6
DEFAULT_REFRESH_PER_SECOND = 4  # bounded: the live display redraws on its own timer, never per optimizer step


@dataclass
class StageBar:
    """One bar row: a stage (or the overall run) with its optimizer-step count. ``marker`` / ``style`` tell the
    stage's state (``▶`` current, ``✓`` done, blank pending); ``note`` is the dimmed text after the percentage."""

    name: str
    total: int
    completed: int = 0
    note: str = ""
    marker: str = "  "
    style: str = "dim"

    @property
    def percentage(self) -> float:
        return 100.0 if self.total <= 0 else 100.0 * min(self.completed, self.total) / self.total


class TrainingDashboard:
    """Live terminal display of one training run (layout and capture: see :mod:`training.ui.dashboard`).

    ``console`` is for tests (a ``rich.console.Console`` over a ``StringIO``); ``clock`` is injected by the ETA
    tests. ``enabled`` is True until the dashboard disables itself after an internal error; from then on the public
    methods delegate to a :class:`NoOpDashboard` built from the same arguments (the console fallback) and
    :meth:`write` prints plain lines to the stream, so a broken display costs one warning, never the run. That
    fallback writes to ``fallback_stream`` when one is given (the run's CLI passes stderr, so the lines of a
    display that disabled itself mid-run land on the same stream as the log handlers' lines), else to ``stream``.
    ``final_frame`` prints the static summary once the display closed.

    The log file :meth:`attach` is given receives the lines the fallback would log (one per ``log_step_interval``
    steps, one per validation, one per event) through :meth:`_log_line` — the file handler alone, never the panel or
    the terminal — so ``train.log`` reads the same whichever dashboard the run had.
    """

    def __init__(
        self,
        run_name: str,
        stage_names: Sequence[str],
        steps_per_stage: Sequence[int],
        total_steps: int,
        *,
        details: Mapping[str, str] | None = None,
        start_step: int = 0,
        log_step_interval: int = 1,
        log_lines: int = DEFAULT_LOG_LINES,
        event_lines: int = DEFAULT_EVENT_LINES,
        refresh_per_second: float = DEFAULT_REFRESH_PER_SECOND,
        final_frame: bool = True,
        console: Console | None = None,
        stream: TextIO | None = None,
        fallback_stream: TextIO | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._fallback = NoOpDashboard(
            run_name,
            stage_names,
            steps_per_stage,
            total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=log_step_interval,
            stream=fallback_stream if fallback_stream is not None else stream,
            clock=clock,
        )
        self.run_name = run_name
        self.stage_names = list(stage_names)
        self.steps_per_stage = list(steps_per_stage)
        self.total_steps = total_steps
        self.details = dict(details or {})
        self.enabled = True
        self._stream = stream if stream is not None else sys.stdout
        # where the plain lines go once the display is gone (`write`, and the fallback's own lines through the
        # handler `attach` installs): the run's fallback stream when it has one, else the display's own stream
        self._fallback_stream = fallback_stream if fallback_stream is not None else self._stream
        self._console = console if console is not None else Console(file=self._stream)
        self._refresh_per_second = refresh_per_second
        self._final_frame = final_frame
        # one estimate for the bars' ETA and the fallback's lines; it also carries on if the display gets disabled
        self._throughput = self._fallback.throughput
        self._lock = threading.RLock()  # every mutation (loop thread) and every render (Live thread)
        self._log_lines = log_lines
        self._lines: deque[str] = deque(maxlen=log_lines)
        self._kept: list[str] = []
        self._events: deque[str] = deque(maxlen=event_lines)
        self._status = "starting"
        self._step = start_step
        self._stage_index = 0
        self._transition: float | None = None  # progress of the running stage transition (the bar note), else None
        self._latest: dict[str, float] = {}
        self._validation: tuple[int, dict[str, float]] | None = None
        self._log_file: Path | None = None
        self._file_handler: logging.Handler | None = None  # the log file's handler while attached: `_log_line` feeds it
        self._render_error: BaseException | None = None  # set on the Live thread, handled on the caller's thread
        self._stage_starts = [sum(self.steps_per_stage[:i]) for i in range(len(self.steps_per_stage))]
        self._bars = [StageBar(name, steps) for name, steps in zip(self.stage_names, self.steps_per_stage)]
        self._overall = StageBar("overall", total_steps, marker="", style="bold")
        self._live: Live | None = None
        self._open = False
        self._attached: list[str] = []
        self._capture = TerminalCapture(self, skip=self.is_attached)
        self._refresh_bars(start_step, 0)

    @classmethod
    @contextmanager
    def open(
        cls,
        run_name: str,
        stage_names: Sequence[str],
        steps_per_stage: Sequence[int],
        total_steps: int,
        *,
        details: Mapping[str, str] | None = None,
        start_step: int = 0,
        log_step_interval: int = 1,
        log_file: Path | None = None,
        logger: logging.Logger | None = None,
        final_frame: bool = True,
        console: Console | None = None,
        stream: TextIO | None = None,
        fallback_stream: TextIO | None = None,
        clock: Clock = time.monotonic,
    ) -> Iterator[TrainingDashboard]:
        """A running dashboard with the ``training`` logger (or ``logger``) attached for the block; ``log_file``
        appended (``run_directory / TRAIN_LOG_NAME`` by convention)."""
        board = cls(
            run_name,
            stage_names,
            steps_per_stage,
            total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=log_step_interval,
            final_frame=final_frame,
            console=console,
            stream=stream,
            fallback_stream=fallback_stream,
            clock=clock,
        )
        # the logger first: the dashboard's own warning (a failing start, an internal error) always has a handler
        with board.attach(logger or logging.getLogger(TRAINING_LOGGER_NAME), log_file=log_file), board:
            yield board

    # --- lifecycle --------------------------------------------------------------------------------------------------

    def __enter__(self) -> TrainingDashboard:
        if self.enabled and not self._open:
            self._open = True
            try:
                self._pin_console_file()
                self._capture.start()
                self._start_live()
            except Exception as error:
                self._disable(error)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._check_render_error()
        self.close()

    def close(self) -> None:
        """End the display (idempotent): erase the frame, restore streams / handlers / environment, then print the
        kept lines and — with ``final_frame`` — the static summary. ``__exit__`` calls it on every way out."""
        if not self._open:
            return
        self._open = False
        if not self.enabled:
            return  # `_disable` already tore the display down and printed the kept lines
        try:
            self._teardown()
        finally:
            self._print_kept()
        if self._final_frame:
            self._console.print(self.render_summary())

    def _start_live(self) -> None:
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start(refresh=True)  # the first frame right away, not after the first refresh interval

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()  # transient: the frame is erased, the cursor is back where the display began

    def _teardown(self) -> None:
        """Undo ``__enter__``; never called with the lock held — ``Live.stop`` joins its refresh thread, which may
        be waiting for the lock."""
        try:
            self._stop_live()
        finally:
            self._capture.stop()

    def _pin_console_file(self) -> None:
        """A console created without a file follows ``sys.stdout`` dynamically — it would render into the sink."""
        if self._console.file is sys.stdout or self._console.file is sys.stderr:
            self._console.file = self._console.file

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Clear the display and give the terminal (streams included) back for a prompt; it comes back afterwards."""
        if self._live is None:
            yield
            return
        self._stop_live()
        self._capture.release_streams()
        try:
            yield
        finally:
            self._capture.redirect_streams()
            self._start_live()

    def _print_kept(self) -> None:
        """The kept records, unwrapped, once, after the display closed — on the console's own file, plainly."""
        with self._lock:
            kept, self._kept = self._kept, []
        file = self._console.file
        for text in kept:
            file.write(text + "\n")
        file.flush()

    def _disable(self, error: BaseException) -> None:
        """Close the display after an internal error; the fallback takes over. Logs one warning (the first error)."""
        if not self.enabled:
            return
        self.enabled = False
        with suppress(Exception):  # the display is already broken; nothing left to clean up if the teardown fails too
            self._teardown()
        with suppress(Exception):
            self._print_kept()
        log.warning(
            "training dashboard disabled after an internal error (training continues with the console fallback): %r",
            error,
        )

    def _check_render_error(self) -> None:
        """A failure of the render happened on the Live thread (which must not stop Live itself); handle it here."""
        error, self._render_error = self._render_error, None
        if error is not None:
            self._disable(error)

    def _guarded(self, action: Callable[[], None]) -> bool:
        """Run ``action``; on any exception disable the display and return False so the caller uses the fallback."""
        self._check_render_error()
        if not self.enabled:
            return False
        try:
            action()
        except Exception as error:
            self._disable(error)
            return False
        return True

    # --- the API train() drives ------------------------------------------------------------------------------------

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        """``step`` optimizer steps are done, the run is in stage ``stage_index``, ``transition`` is the progress
        (0-1) of the running transition out of it or None; ``metrics`` is the step dict (only the
        :data:`METRIC_COLUMNS` keys are read, missing keys keep their last value). O(1): a few bar updates and a dict
        merge; the display redraws on its own timer."""
        if not self._guarded(lambda: self._apply_step(step, stage_index, transition, metrics)):
            self._fallback.update_step(step, stage_index, transition, metrics)

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        """The validation losses measured after ``step`` (one entry per recurrence depth, e.g. ``val_loss_4``)."""
        if not self._guarded(lambda: self._apply_validation(step, losses)):
            self._fallback.update_validation(step, losses)

    def note_event(self, text: str) -> None:
        """Add a line to the events list (checkpoint written, resume point, stage transition, export)."""
        if not self._guarded(lambda: self._apply_event(text)):
            self._fallback.note_event(text)

    def set_status(self, text: str) -> None:
        """The status shown in the header (``training``, ``evaluating``, ``saving checkpoint`` ...)."""
        if not self._guarded(lambda: self._apply_status(text)):
            self._fallback.set_status(text)

    def _apply_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        known = known_metrics(metrics)
        with self._lock:
            self._transition = transition
            self._step = step
            self._stage_index = stage_index
            self._latest.update(known)
            self._refresh_bars(step, stage_index)
            file_line = self._fallback.step_line(step, stage_index, transition, metrics)  # also records the throughput
        if file_line is not None:
            self._log_line(logging.INFO, file_line)

    def _refresh_bars(self, step: int, stage_index: int) -> None:
        last = len(self._bars) - 1
        for index, (bar, start) in enumerate(zip(self._bars, self._stage_starts)):
            bar.completed = min(max(step - start, 0), bar.total)
            if bar.completed >= bar.total:  # done wins over current: the final summary shows every stage ticked
                bar.marker, bar.style = "✓ ", "green"
            elif index == stage_index:
                bar.marker, bar.style = "▶ ", "bold cyan"
            else:
                bar.marker, bar.style = "  ", "dim"
            bar.note = ""
            if index == stage_index and self._transition is not None and index < last:
                bar.note = f"transition → {self.stage_names[index + 1]} {self._transition:.0%}"
        self._overall.completed = min(step, self.total_steps)

    def _apply_validation(self, step: int, losses: Mapping[str, object]) -> None:
        with self._lock:
            self._validation = (step, floats(losses))
        self._log_line(logging.INFO, validation_line(step, losses))

    def _apply_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._events.append(f"{stamp}  step {self._step}: {text}")
        self._log_line(logging.INFO, event_line(text))

    def _apply_status(self, text: str) -> None:
        with self._lock:
            self._status = text
        self._log_line(logging.DEBUG, status_line(text))

    def _log_line(self, level: int, message: str) -> None:
        """The line the fallback would log, as a record of the ``training.ui.dashboard`` logger handed to the log
        file's handler directly: it never passes a logger, so no other handler (the panel, the root capture) sees it.
        Dropped when no log file is attached or the logger would not emit at ``level`` (the DEBUG status lines)."""
        handler = self._file_handler
        if handler is None or not log.isEnabledFor(level):
            return
        handler.handle(log.makeRecord(log.name, level, __file__, 0, message, (), None))

    # --- log lines ----------------------------------------------------------------------------------------------------

    def write(self, text: str, *, keep: bool = False) -> None:
        """Append ``text`` (one entry per line, so tracebacks stay readable) to the log panel; plain stream when
        disabled. With ``keep`` the text is also printed, unwrapped, once the display closed."""
        if not self.enabled:
            self._fallback_stream.write(text + "\n")
            self._fallback_stream.flush()
            return
        with self._lock:
            self._lines.extend(text.splitlines() or [""])
            if keep:
                self._kept.append(text)

    def is_attached(self, logger_name: str) -> bool:
        """Whether records of ``logger_name`` reach the panel through a handler :meth:`attach` installed."""
        with self._lock:
            return any(logger_name == name or logger_name.startswith(name + ".") for name in self._attached)

    @contextmanager
    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> Iterator[None]:
        """Route ``logger`` into this dashboard (and ``log_file``, named in the footer) for the duration of the
        block. ``logger`` must be the ``training`` logger or one of its ancestors for the dashboard's own warning /
        fallback lines to reach it."""
        with self._lock:
            self._attached.append(logger.name)
            if log_file is not None:
                self._log_file = log_file
        try:
            with attach_logger(self, logger, log_file) as file_handler:
                if file_handler is not None:
                    self._file_handler = file_handler
                yield
        finally:
            with self._lock:
                self._attached.remove(logger.name)
                if log_file is not None:
                    self._file_handler = None  # closed by `attach_logger`

    # --- state for tests ------------------------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        """The log lines currently shown (newest last)."""
        with self._lock:
            return list(self._lines)

    def kept(self) -> list[str]:
        """The kept records not yet printed (they are printed when the display closes)."""
        with self._lock:
            return list(self._kept)

    def events(self) -> list[str]:
        """The event lines currently shown (newest last)."""
        with self._lock:
            return list(self._events)

    @property
    def latest_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    @property
    def tasks(self) -> list[StageBar]:
        """The bars: one per stage, then the overall bar."""
        with self._lock:
            return [*self._bars, self._overall]

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        try:
            with self._lock:
                fixed = self._render_fixed()
                fixed_height = len(console.render_lines(fixed, options, pad=False))
                events_wanted = len(self._events) or 1
                events_shown, log_shown = fit_panel_heights(options.max_height - fixed_height - 1, events_wanted, self._log_lines)
                events = self._render_events(events_shown)
                log_panel = self._render_log(log_shown)
                footer = line(f"log: {self._log_file}" if self._log_file is not None else "", style="dim")
            yield Group(fixed, events, log_panel, footer)
        except Exception as error:  # runs on the Live thread: report, let the next public call disable the display
            self._render_error = error
            yield Text(f"training dashboard render failed: {error!r}", style="bold red")

    def render_summary(self) -> RenderableType:
        """The static summary printed after the display closed: header, bars, metrics, validation, events."""
        with self._lock:
            events = [line(event) for event in self._events] or [line("(no events)", style="dim")]
            return Group(self._render_fixed(), line("events", style="bold"), *events)

    def _render_fixed(self) -> Group:
        """Header, bars, metrics and validation: the part of the frame whose height only the run decides."""
        parts: list[RenderableType] = [self._render_header(), self._render_bars(), self._render_metrics()]
        validation = self._render_validation()
        if validation is not None:
            parts.append(validation)
        return Group(*parts)

    def _render_header(self) -> RenderableType:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        grid.add_column(justify="right", no_wrap=True, overflow="ellipsis")
        details = "  ".join(f"{key}={value}" for key, value in self.details.items())
        left = Text.assemble((self.run_name, "bold"), "  ", (details, "dim"), no_wrap=True, overflow="ellipsis")
        grid.add_row(left, line(self._status, style="bold yellow"))
        return grid

    def _render_bars(self) -> RenderableType:
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis", max_width=40)  # marker + stage name
        grid.add_column(ratio=1, min_width=10)  # bar
        grid.add_column(justify="right", no_wrap=True)  # completed/total
        grid.add_column(justify="right", no_wrap=True)  # percentage
        grid.add_column(no_wrap=True, overflow="ellipsis", style="dim")  # note
        rate = self._throughput.steps_per_second
        rate_text = "? steps/s" if rate is None else f"{rate:.2f} steps/s"
        remaining = format_duration(self._throughput.remaining(self._step))
        self._overall.note = f"{rate_text}, {format_duration(self._throughput.elapsed)} elapsed, ETA {remaining}"
        for bar in (*self._bars, self._overall):
            grid.add_row(
                line(f"{bar.marker}{bar.name}", style=bar.style),
                ProgressBar(total=max(bar.total, 1), completed=bar.completed if bar.total > 0 else 1),
                f"{bar.completed:,}/{bar.total:,}",
                f"{bar.percentage:>3.0f}%",
                bar.note,
            )
        return grid

    def _render_metrics(self) -> RenderableType:
        table = Table(
            box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, title=line(f"step {self._step}"), title_justify="left"
        )
        cells: list[str] = []
        for key, label in METRIC_COLUMNS:
            table.add_column(label, justify="right", no_wrap=True)
            value = self._latest.get(key)
            if value is None and key == "seconds/step":
                value = self._throughput.seconds_per_step  # the loop's own timing when the step dict has none
            cells.append("—" if value is None else format_metric(key, value))
        table.add_column("elapsed", justify="right", no_wrap=True)
        table.add_column("remaining", justify="right", no_wrap=True)
        cells += [format_duration(self._throughput.elapsed), format_duration(self._throughput.remaining(self._step))]
        table.add_row(*cells)
        return table

    def _render_validation(self) -> RenderableType | None:
        if self._validation is None:
            return None
        step, losses = self._validation
        table = Table(
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
            title=line(f"validation (step {step})"),
            title_justify="left",
        )
        for key in losses:
            table.add_column(key, justify="right", no_wrap=True)
        table.add_row(*(f"{value:.4f}" for value in losses.values()))
        return table

    def _render_events(self, height: int) -> RenderableType:
        events = list(self._events)[-height:]
        body = line("\n".join(events) or "(no events yet)")
        return Panel(body, title="events", title_align="left", border_style="dim", padding=(0, 1))

    def _render_log(self, height: int) -> RenderableType:
        lines = list(self._lines)[-height:]
        body = line("\n".join(lines) or "(no log output yet)")
        return Panel(body, title="log", title_align="left", border_style="dim", padding=(0, 1))

    def render_text(self, width: int = 120, height: int = 50) -> str:
        """The current display as plain text (tests, or a snapshot for a log file)."""
        console = Console(width=width, height=height, force_terminal=False, color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()
