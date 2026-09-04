# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The live training dashboard: one transient rich.live.Live layout (header, stage bars, metrics, validation,
events, log panel, footer) with the terminal captured around it (:mod:`training.ui.capture`). The module docstring
of :mod:`training.ui.dashboard` describes the whole picture.
"""

from __future__ import annotations

import logging
import sys
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
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ui.capture import attach_logger
from training.ui.capture import TerminalCapture, line_handler, run_log_handlers
from training.ui.common import TRAINING_LOGGER_NAME, Clock, lines_log, log
from training.ui.format import (
    depth_losses,
    METRIC_COLUMNS,
    event_line,
    fit_panel_heights,
    floats,
    format_duration,
    format_metric,
    known_metrics,
    status_line,
    step_line,
    validation_line,
)
from training.ui.throughput import Throughput
from ui.display import LiveDisplay, line

DEFAULT_LOG_LINES = 12
DEFAULT_EVENT_LINES = 6
DEFAULT_REFRESH_PER_SECOND = 4  # bounded: the live display redraws on its own timer, never per optimizer step


@dataclass
class StageBar:
    """
    One bar row: a stage (or the overall run) with its optimizer-step count. marker / style tell the
    stage's state (▶ current, ✓ done, blank pending); note is the dimmed text after the percentage.
    """

    name: str
    total: int
    completed: int = 0
    note: str = ""
    marker: str = "  "
    style: str = "dim"

    @property
    def percentage(self) -> float:
        return 100.0 if self.total <= 0 else 100.0 * min(self.completed, self.total) / self.total


class TrainingDashboard(LiveDisplay):
    """
    Live terminal display of one training run (layout and capture: see :mod:`training.ui.dashboard`).

    console is for tests; clock is injected by the ETA tests. enabled is True until the dashboard disables
    itself after an internal error or a dead terminal (:mod:`ui.display`); from then on the public methods only log
    the lines the console fallback logs and :meth:`write` prints plain lines to fallback_stream (the CLI passes
    stderr) or stream, so a broken display costs one warning, never the run. final_frame prints the static
    summary once the display closed.

    The step / validation / event lines go to :data:`~training.ui.common.lines_log` too, which only the log file
    reads, so train.log reads the same whichever dashboard the run had.
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
        if len(stage_names) != len(steps_per_stage):
            raise ValueError(f"{len(stage_names)} stage names for {len(steps_per_stage)} step counts")
        self.run_name = run_name
        self.stage_names = list(stage_names)
        self.steps_per_stage = list(steps_per_stage)
        self.total_steps = total_steps
        self.details = dict(details or {})
        self.log_step_interval = max(int(log_step_interval), 1)
        super().__init__(
            stream=stream if stream is not None else sys.stdout, console=console, refresh_per_second=refresh_per_second, log_lines=log_lines
        )
        # plain lines once the display is gone: the run's fallback stream when it has one, else the display's own stream
        if fallback_stream is not None:
            self._plain_stream = fallback_stream
        self._final_frame = final_frame
        self._throughput = Throughput(total_steps, start_step=start_step, clock=clock)  # the bars' ETA and the lines
        self._events: deque[str] = deque(maxlen=event_lines)
        self._status = "starting"
        self._step = start_step
        self._stage_index = 0
        self._transition: float | None = None  # progress of the running stage transition (the bar note), else None
        self._latest: dict[str, float] = {}
        self._validation: tuple[int, dict[str, float]] | None = None
        self._console_lines: logging.Handler | None = None  # the dashboard lines' way to the console once disabled
        self._torn_down = False  # `_disable` already closed the display (`close` has nothing left to do)
        self._render_error: BaseException | None = None  # set on the Live thread, handled on the caller's thread
        self._stage_starts = [sum(self.steps_per_stage[:i]) for i in range(len(self.steps_per_stage))]
        self._bars = [StageBar(name, steps) for name, steps in zip(self.stage_names, self.steps_per_stage)]
        self._overall = StageBar("overall", total_steps, marker="", style="bold")
        self._open = False
        self.logger = log
        self._capture = TerminalCapture(self, already_attached=self.is_attached)
        self._refresh_bars(start_step, 0)

    # --- lifecycle --------------------------------------------------------------------------------------------------

    @contextmanager
    def running(self, logger: logging.Logger | None = None, *, log_file: Path | None = None) -> Iterator[TrainingDashboard]:
        """
        The dashboard in service for the block: logger (default: the training logger) attached first, so
        the dashboard's own warnings always have a handler, then the display up; log_file appended.
        """

        with self.attach(logger, log_file=log_file), self:
            yield self

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
        """
        End the display (idempotent): erase the frame, restore streams / handlers / environment, print the kept
        lines and, with final_frame, the static summary. __exit__ calls it on every way out.
        """

        if not self._open:
            return
        self._open = False
        self._drop_console_lines()  # a demoted board's console handler (`_disable`) ends with the display
        if self._torn_down:
            return  # `_disable` already tore the display down and printed the kept lines
        try:
            self._teardown()
        finally:
            self._print_kept()
        if self._final_frame:
            self._console.print(self.render_summary())

    def _teardown(self) -> None:
        """
        Undo __enter__. Never called with the lock held: Live.stop joins its refresh thread, which may be
        waiting for the lock.
        """

        try:
            self._stop_live()
        finally:
            self._capture.stop()

    def _pin_console_file(self) -> None:
        """
        A console created without a file follows sys.stdout dynamically and would render into the sink.
        """

        if self._console.file is sys.stdout or self._console.file is sys.stderr:
            self._console.file = self._console.file

    def _release_streams(self) -> None:
        self._capture.release_streams()

    def _redirect_streams(self) -> None:
        self._capture.redirect_streams()

    def _disable(self, error: BaseException) -> None:
        """
        Close the display after an internal error and behave like the console fallback from now on. Logs one
        warning.
        """

        if not self.enabled:
            return
        self.enabled = False
        self._torn_down = True
        with suppress(Exception):  # the display is already broken; nothing left to clean up if the teardown fails too
            self._teardown()
        with suppress(Exception):
            self._print_kept()
        self._console_lines = line_handler(self._plain_stream)
        lines_log.addHandler(self._console_lines)
        log.warning(
            "training dashboard disabled after an internal error (training continues with the console fallback): %r",
            error,
        )

    def _drop_console_lines(self) -> None:
        handler, self._console_lines = self._console_lines, None
        if handler is not None:
            lines_log.removeHandler(handler)
            handler.close()

    def _check_render_error(self) -> None:
        """
        A failure of the render happened on the Live thread (which must not stop Live itself); handle it here.
        """

        error, self._render_error = self._render_error, None
        if error is not None:
            self._disable(error)

    def _guarded(self, action: Callable[[], None]) -> None:
        """
        Run action on the display; on any exception disable the display (the lines still get logged).
        """

        self._check_render_error()
        if not self.enabled:
            return
        try:
            action()
        except Exception as error:
            self._disable(error)

    # --- the API train() drives ------------------------------------------------------------------------------------

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        """
        step optimizer steps are done in stage stage_index; transition is the progress (0-1) of the
        running transition or None; only the :data:`METRIC_COLUMNS` keys of metrics are read. O(1); the display
        redraws on its own timer.
        """

        with self._lock:
            self._throughput.record(step)
        self._guarded(lambda: self._apply_step(step, stage_index, transition, metrics))
        text = step_line(
            step,
            stage_index,
            transition,
            metrics,
            total_steps=self.total_steps,
            stage_names=self.stage_names,
            log_step_interval=self.log_step_interval,
            throughput=self._throughput,
        )
        if text is not None:
            lines_log.info(text)

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        """
        The validation losses measured after step (per recurrence depth, e.g. val_loss_4, and per source,
        val_loss/<data id>); the table shows the depths, the log line everything.
        """

        self._guarded(lambda: self._apply_validation(step, losses))
        lines_log.info(validation_line(step, losses))

    def note_event(self, text: str) -> None:
        """
        Add a line to the events list (checkpoint written, resume point, stage transition, export).
        """

        self._guarded(lambda: self._apply_event(text))
        lines_log.info(event_line(text))

    def set_status(self, text: str) -> None:
        """
        The status shown in the header (training, evaluating, saving checkpoint ...).
        """

        self._guarded(lambda: self._apply_status(text))
        lines_log.debug(status_line(text))

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

    def _refresh_bars(self, step: int, stage_index: int) -> None:
        last_index = len(self._bars) - 1
        for index, (bar, start) in enumerate(zip(self._bars, self._stage_starts)):
            bar.completed = min(max(step - start, 0), bar.total)
            if bar.completed >= bar.total:  # done wins over current: the final summary shows every stage ticked
                bar.marker, bar.style = "✓ ", "green"
            elif index == stage_index:
                bar.marker, bar.style = "▶ ", "bold cyan"
            else:
                bar.marker, bar.style = "  ", "dim"
            bar.note = ""
            if index == stage_index and self._transition is not None and index < last_index:
                bar.note = f"transition → {self.stage_names[index + 1]} {self._transition:.0%}"
        self._overall.completed = min(step, self.total_steps)

    def _apply_validation(self, step: int, losses: Mapping[str, object]) -> None:
        with self._lock:
            self._validation = (step, depth_losses(floats(losses)))  # per-source losses: log line and wandb only

    def _apply_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._events.append(f"{stamp}  step {self._step}: {text}")

    def _apply_status(self, text: str) -> None:
        with self._lock:
            self._status = text

    # --- log lines ----------------------------------------------------------------------------------------------------

    @contextmanager
    def attach(self, logger: logging.Logger | None = None, *, log_file: Path | None = None) -> Iterator[None]:
        """
        Route logger (default: the training logger) into the panel and log_file (named in the footer),
        and this dashboard's own lines into log_file alone (one shared file handler) for the block. A logger
        above INFO is lowered to INFO for the block.
        """

        target = logger if logger is not None else logging.getLogger(TRAINING_LOGGER_NAME)
        with self._lock:
            self._attached_logger_names.append(target.name)
            if log_file is not None:
                self._log_file = log_file
        try:
            with attach_logger(self, target, None, ensure_info_level=True), run_log_handlers(target, log_file, None):
                yield
        finally:
            with self._lock:
                self._attached_logger_names.remove(target.name)
            self._drop_console_lines()

    # --- state for tests ------------------------------------------------------------------------------------------------

    def events(self) -> list[str]:
        """
        The event lines currently shown (newest last).
        """

        with self._lock:
            return list(self._events)

    @property
    def latest_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    @property
    def tasks(self) -> list[StageBar]:
        """
        The bars: one per stage, then the overall bar.
        """

        with self._lock:
            return [*self._bars, self._overall]

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        try:
            with self._lock:
                fixed = self._render_fixed()
                fixed_height = len(console.render_lines(fixed, options, pad=False))
                events_wanted = len(self._events) or 1
                events_shown, log_shown = fit_panel_heights(options.max_height - fixed_height - 1, events_wanted, self._panel_height)
                events = self._render_events(events_shown)
                log_panel = self._render_log(log_shown)
                footer = self._footer()
            yield Group(fixed, events, log_panel, footer)
        except Exception as error:  # runs on the Live thread: report, let the next public call disable the display
            self._render_error = error
            yield Text(f"training dashboard render failed: {error!r}", style="bold red")

    def render_summary(self) -> RenderableType:
        """
        The static summary printed after the display closed: header, bars, metrics, validation, events.
        """

        with self._lock:
            events = [line(event) for event in self._events] or [line("(no events)", style="dim")]
            return Group(self._render_fixed(), line("events", style="bold"), *events)

    def _render_fixed(self) -> Group:
        """
        Header, bars, metrics and validation: the part of the frame whose height only the run decides.
        """

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
