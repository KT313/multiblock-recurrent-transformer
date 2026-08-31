# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal dashboard for one training run: stage bars, the latest metrics, validation losses, events and a log panel.

Built on ``rich`` (the only training module that imports it), the sibling of ``data_preparation/lib/ui/dashboard.py``.
A :class:`TrainingDashboard` owns one ``rich.live.Live`` display on stdout that renders, top to bottom: a header (run
name, model / dataset config names, device / precision, the current status), one progress bar per training stage plus
an overall bar (optimizer steps, ETA from a smoothed steps-per-second estimate), a table with the metrics of the
latest optimizer step, the latest validation losses per recurrence depth, the last few events (checkpoints, resume
point, stage transitions, export) and the last ``log_lines`` ``logging`` records.

Usage (``RunLogger`` / ``train()`` wrap the run once — see ``tasks/training_pipeline_restructure.md``, task 10)::

    with training_dashboard(run_name, stage_names, steps_per_stage, total_steps, log_file=run_dir / TRAIN_LOG_NAME,
                            details={"model": "crow-300m-final", "dataset": "crow_300m_final", "device": "cuda:0",
                                     "precision": "bf16-mixed"}, log_step_interval=settings.log_step_interval) as board:
        board.note_event("resumed from step-00000100-run.pth at step 100")
        board.update_step(step, stage_index, metrics)            # every optimizer step, O(1), never raises
        board.update_validation(step, {"val_loss_4": 3.2, "val_loss": 3.1})
        board.set_status("saving checkpoint")

:func:`training_dashboard` returns the rich :class:`TrainingDashboard` when enabled (``TRAINING_DASHBOARD`` not ``0``
and stdout a terminal — the rule of ``DATA_PREP_PROGRESS``) and otherwise the :class:`NoOpDashboard` with the same
methods: the console fallback, which writes the same log file and logs one line per ``log_step_interval`` steps,
one per validation and one per event through ``logging`` (so a piped or ``nohup`` run still has a readable log).
Both attach the ``training`` logger for the duration of the block, like the data-prep dashboard attaches
``data_preparation``: the plain stream handlers are swapped for the dashboard's handler (records land in the panel,
warnings additionally scroll into the terminal history) plus a file handler for ``log_file``.

A dashboard failure must never end a training run: every public method of :class:`TrainingDashboard` catches its own
exceptions, stops the live display, logs one warning and from then on behaves like the :class:`NoOpDashboard`.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path
from types import TracebackType
from typing import Protocol, TextIO

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress as RichProgress, ProgressColumn, TaskID, TextColumn
from rich.progress import Task as RichTask
from rich.table import Table
from rich.text import Text

from data_preparation.lib.log import LOG_FORMAT
from data_preparation.lib.progress import DISABLING_VALUES

ENV_VAR = "TRAINING_DASHBOARD"
TRAINING_LOGGER_NAME = "training"  # the logger hierarchy of `training/`; `open()` attaches it by default
TRAIN_LOG_NAME = "train.log"  # full log of every run, appended under the run directory (`log_file=`)
DEFAULT_LOG_LINES = 12
DEFAULT_EVENT_LINES = 6
DEFAULT_REFRESH_PER_SECOND = 4  # bounded: the live display redraws on its own timer, never per optimizer step
RATE_SMOOTHING = 0.1  # weight of the newest seconds/step sample in the exponential moving average behind the ETA

# metric keys of the step dict (`RunLogger.log_step`, today's `train.py` names) shown in the metrics table, with labels
METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("loss", "loss"),
    ("ppl", "ppl"),
    ("lr", "lr"),
    ("grad_norm", "grad norm"),
    ("tokens/second", "tokens/s"),
    ("seconds/step", "s/step"),
    ("total_tokens", "tokens"),
)
TRANSITION_FLAG_KEY = "stage/in_transition"
TRANSITION_PROGRESS_KEY = "stage/transition_progress"

Clock = Callable[[], float]

# named explicitly (not `__name__`, which is `__main__` under `python -m`): it must sit under the attached `training` logger
_log = logging.getLogger(f"{TRAINING_LOGGER_NAME}.ui.dashboard")


def dashboard_enabled(stream: TextIO | None = None) -> bool:
    """False when ``TRAINING_DASHBOARD=0`` (or ``false``/``no``/``off``) or when ``stream`` (stdout) is not a TTY."""
    env_value = os.environ.get(ENV_VAR, "1").strip().lower()
    if env_value in DISABLING_VALUES:
        return False
    if stream is None:
        stream = sys.stdout
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    return bool(isatty())


# --- formatting -----------------------------------------------------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
    """``h:mm:ss`` (``Nd hh:mm:ss`` from one day on); ``—`` for unknown / non-finite / negative values."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    whole = int(seconds)
    days, rest = divmod(whole, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_tokens(count: float) -> str:
    """A token count with a k / M / B / T suffix (``1.23B``); plain below 1000."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(count) >= threshold:
            return f"{count / threshold:.2f}{suffix}"
    return f"{count:.0f}"


def format_metric(key: str, value: float) -> str:
    """The table / fallback-line rendering of one metric of the step dict."""
    if key == "loss":
        return f"{value:.4f}"
    if key == "ppl":
        return f"{value:.2f}"
    if key == "lr":
        return f"{value:.2e}"
    if key == "grad_norm":
        return f"{value:.3f}"
    if key == "tokens/second":
        return f"{value:,.0f}"
    if key == "seconds/step":
        return f"{value:.2f}s"
    if key == "total_tokens":
        return format_tokens(value)
    return f"{value:.4g}"


def _as_float(value: object) -> float | None:
    """``float(value)`` for numbers and one-element tensors, None for anything that does not convert."""
    try:
        return float(value)  # type: ignore[arg-type]  # the point of the helper is to accept anything float-like
    except (TypeError, ValueError):
        return None


def _floats(values: Mapping[str, object]) -> dict[str, float]:
    """The float-convertible entries of ``values`` (insertion order kept)."""
    return {key: value for key, raw in values.items() if (value := _as_float(raw)) is not None}


def _known_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """The metric-table keys present in ``metrics`` (in table order) as floats."""
    known: dict[str, float] = {}
    for key, _label in METRIC_COLUMNS:
        if key in metrics:
            value = _as_float(metrics[key])
            if value is not None:
                known[key] = value
    return known


def _transition_of(metrics: Mapping[str, object]) -> float | None:
    """The transition progress (0-1) when the step dict says the step is inside a transition, else None."""
    flag = _as_float(metrics.get(TRANSITION_FLAG_KEY, 0))
    if not flag:
        return None
    progress = _as_float(metrics.get(TRANSITION_PROGRESS_KEY, 0.0))
    return 0.0 if progress is None else progress


# --- throughput -----------------------------------------------------------------------------------------------------------


class Throughput:
    """Smoothed seconds per optimizer step from the times :meth:`record` is called with, and the ETA derived from it.

    The first interval sets the estimate, later ones move it by :data:`RATE_SMOOTHING` (an exponential moving average
    — a stall or one slow evaluation step does not swing the ETA). ``start_step`` is the step the run (re)starts at,
    so a resumed run does not count the checkpointed steps as done in zero seconds.
    """

    def __init__(self, total_steps: int, *, start_step: int = 0, clock: Clock = time.monotonic) -> None:
        self._total_steps = total_steps
        self._clock = clock
        self._started = clock()
        self._last_time = self._started
        self._last_step = start_step
        self.seconds_per_step: float | None = None

    def record(self, step: int) -> None:
        """Note that ``step`` optimizer steps are done now (a step not beyond the last recorded one is ignored)."""
        now = self._clock()
        advanced = step - self._last_step
        if advanced <= 0:
            return
        sample = (now - self._last_time) / advanced
        if self.seconds_per_step is None:
            self.seconds_per_step = sample
        else:
            self.seconds_per_step = (1 - RATE_SMOOTHING) * self.seconds_per_step + RATE_SMOOTHING * sample
        self._last_time = now
        self._last_step = step

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def steps_per_second(self) -> float | None:
        if not self.seconds_per_step:
            return None
        return 1.0 / self.seconds_per_step

    def remaining(self, step: int) -> float | None:
        """Estimated seconds until ``total_steps`` (None before the first interval)."""
        if self.seconds_per_step is None:
            return None
        return self.seconds_per_step * max(self._total_steps - step, 0)


# --- logging -------------------------------------------------------------------------------------------------------------


class LogSink(Protocol):
    """What :class:`DashboardLogHandler` writes to (both dashboards)."""

    def write(self, text: str, *, keep: bool = False) -> None: ...


class DashboardLogHandler(logging.Handler):
    """``logging.Handler`` whose records land in the dashboard's log panel (or on its stream, for the fallback).

    Records of ``keep_level`` and above (default WARNING), and records logged with ``extra={"keep": True}`` (stage
    summaries, the final report), are additionally printed above the live display, unwrapped, where they scroll into
    the terminal's history and survive the run — the panel only shows the last few lines."""

    def __init__(self, sink: LogSink, level: int = logging.NOTSET, keep_level: int = logging.WARNING) -> None:
        super().__init__(level)
        self._sink = sink
        self._keep_level = keep_level
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
            keep = record.levelno >= self._keep_level or bool(getattr(record, "keep", False))
            self._sink.write(text, keep=keep)
        except Exception:
            self.handleError(record)


@contextmanager
def _attach(sink: LogSink, logger: logging.Logger, log_file: Path | None) -> Iterator[None]:
    """Route ``logger`` into ``sink`` for the duration of the block (the shared body of both ``attach`` methods).

    The plain stream handlers a CLI installed are detached (their lines would print twice and garble the live
    display) and restored afterwards; with ``log_file`` every record is also appended to that file. A logger whose
    effective level is above INFO is lowered to INFO for the block — the dashboard lives on INFO records — and
    restored afterwards."""
    detached: list[logging.Handler] = [
        h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    added: list[logging.Handler] = [DashboardLogHandler(sink)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        added.append(file_handler)
    for handler in detached:
        logger.removeHandler(handler)
    for handler in added:
        logger.addHandler(handler)
    previous_level = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.setLevel(previous_level)
        for handler in added:
            logger.removeHandler(handler)
            handler.close()
        for handler in detached:
            logger.addHandler(handler)


# --- the fallback ---------------------------------------------------------------------------------------------------------


class NoOpDashboard:
    """The console fallback: the :class:`TrainingDashboard` interface without a live display.

    ``update_step`` logs one line every ``log_step_interval`` steps (and at the last step), ``update_validation`` and
    ``note_event`` one line each, ``set_status`` a DEBUG line — all through ``logging`` on the ``training.ui.dashboard``
    logger, so :meth:`attach` (and :meth:`open`) decide where they go: the stream, the log file, or both.
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
        stream: TextIO | None = None,
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
        self._stream = stream if stream is not None else sys.stdout
        self._throughput = Throughput(total_steps, start_step=start_step, clock=clock)

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
        stream: TextIO | None = None,
        clock: Clock = time.monotonic,
    ) -> Iterator[NoOpDashboard]:
        """The fallback with the ``training`` logger (or ``logger``) attached for the block; ``log_file`` appended."""
        board = cls(
            run_name,
            stage_names,
            steps_per_stage,
            total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=log_step_interval,
            stream=stream,
            clock=clock,
        )
        with board, board.attach(logger or logging.getLogger(TRAINING_LOGGER_NAME), log_file=log_file):
            yield board

    def __enter__(self) -> NoOpDashboard:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        return None

    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> AbstractContextManager[None]:
        """Route ``logger`` to the stream (plain lines) and ``log_file`` for the duration of the block. ``logger``
        must be the ``training`` logger or one of its ancestors for the fallback's own lines to reach it."""
        return _attach(self, logger, log_file)

    def write(self, text: str, *, keep: bool = False) -> None:
        """A log record: one plain line on the stream."""
        self._stream.write(text + "\n")
        self._stream.flush()

    def _stage_name(self, stage_index: int) -> str:
        if 0 <= stage_index < len(self.stage_names):
            return self.stage_names[stage_index]
        return "?"

    def update_step(self, step: int, stage_index: int, metrics: Mapping[str, object]) -> None:
        self._throughput.record(step)
        if step % self.log_step_interval and step < self.total_steps:
            return
        parts = [f"step {step}/{self.total_steps}", f"stage {stage_index} {self._stage_name(stage_index)}"]
        transition = _transition_of(metrics)
        if transition is not None:
            parts.append(f"transition {transition:.0%}")
        known = _known_metrics(metrics)
        parts += [f"{label} {format_metric(key, known[key])}" for key, label in METRIC_COLUMNS if key in known]
        if "seconds/step" not in known and self._throughput.seconds_per_step is not None:
            parts.append(f"s/step {self._throughput.seconds_per_step:.2f}s")
        parts.append(f"elapsed {format_duration(self._throughput.elapsed)}")
        parts.append(f"ETA {format_duration(self._throughput.remaining(step))}")
        _log.info(" | ".join(parts))

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        values = ", ".join(f"{key} {value:.4f}" for key, value in _floats(losses).items())
        _log.info("step %d: validation %s", step, values or "(no losses)")

    def note_event(self, text: str) -> None:
        _log.info("event: %s", text)

    def set_status(self, text: str) -> None:
        _log.debug("status: %s", text)


# --- the live dashboard --------------------------------------------------------------------------------------------------


class _NoteColumn(ProgressColumn):
    """The task's ``note`` field (transition state of a stage bar, rate / elapsed / ETA of the overall bar), dimmed."""

    def render(self, task: RichTask) -> Text:
        return Text(str(task.fields.get("note", "")), style="dim")


class TrainingDashboard:
    """Live terminal display of one training run (see the module docstring for the layout).

    ``console`` is for tests (a ``rich.console.Console`` over a ``StringIO``); ``clock`` is injected by the ETA
    tests. ``enabled`` is True until the dashboard disables itself after an internal error; from then on the public
    methods delegate to a :class:`NoOpDashboard` built from the same arguments (the console fallback) and
    :meth:`write` prints plain lines to the stream, so a broken display costs one warning, never the run.
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
        console: Console | None = None,
        stream: TextIO | None = None,
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
            stream=stream,
            clock=clock,
        )
        self.run_name = run_name
        self.stage_names = list(stage_names)
        self.steps_per_stage = list(steps_per_stage)
        self.total_steps = total_steps
        self.details = dict(details or {})
        self.enabled = True
        self._stream = stream if stream is not None else sys.stdout
        self._console = console if console is not None else Console(file=self._stream)
        self._refresh_per_second = refresh_per_second
        self._throughput = Throughput(total_steps, start_step=start_step, clock=clock)
        self._state_lock = threading.Lock()  # metrics / validation / events / status: written by the loop, read by Live
        self._lines_lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=log_lines)
        self._events: deque[str] = deque(maxlen=event_lines)
        self._status = "starting"
        self._step = start_step
        self._stage_index = 0
        self._latest: dict[str, float] = {}
        self._validation: tuple[int, dict[str, float]] | None = None
        self._render_error: BaseException | None = None  # set on the Live thread, handled on the caller's thread
        self._progress = RichProgress(
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>4.0f}%"),
            _NoteColumn(),
            console=self._console,
            expand=True,
            get_time=clock,
        )
        self._stage_starts = [sum(self.steps_per_stage[:i]) for i in range(len(self.steps_per_stage))]
        self._stage_tasks: list[TaskID] = [self._progress.add_task("", total=steps, note="") for steps in self.steps_per_stage]
        self._overall_task = self._progress.add_task("[bold]overall", total=total_steps, note="")
        self._live: Live | None = None
        self._refresh_bars(start_step, 0, None)

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
        console: Console | None = None,
        stream: TextIO | None = None,
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
            console=console,
            stream=stream,
            clock=clock,
        )
        with board, board.attach(logger or logging.getLogger(TRAINING_LOGGER_NAME), log_file=log_file):
            yield board

    # --- lifecycle --------------------------------------------------------------------------------------------------

    def __enter__(self) -> TrainingDashboard:
        if self.enabled and self._live is None:
            try:
                self._live = Live(
                    self,
                    console=self._console,
                    refresh_per_second=self._refresh_per_second,
                    transient=False,
                    redirect_stderr=False,
                    redirect_stdout=False,
                )
                self._live.start()
            except Exception as error:
                self._disable(error)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._check_render_error()
        self._stop_live()

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()

    def _disable(self, error: BaseException) -> None:
        """Stop the display after an internal error; the fallback takes over. Logs one warning (the first error)."""
        if not self.enabled:
            return
        self.enabled = False
        with suppress(Exception):  # the display is already broken; nothing left to clean up if stopping fails too
            self._stop_live()
        _log.warning(
            "training dashboard disabled after an internal error (training continues with the console fallback): %r",
            error,
        )

    def _check_render_error(self) -> None:
        """A failure of ``__rich__`` happened on the Live thread (which must not stop Live itself); handle it here."""
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

    def update_step(self, step: int, stage_index: int, metrics: Mapping[str, object]) -> None:
        """``step`` optimizer steps are done, the run is in stage ``stage_index``; ``metrics`` is the step dict
        (only the :data:`METRIC_COLUMNS` keys and the ``stage/`` transition keys are read, missing keys keep their
        last value). O(1): a few task updates and a dict merge; the display redraws on its own timer."""
        if not self._guarded(lambda: self._apply_step(step, stage_index, metrics)):
            self._fallback.update_step(step, stage_index, metrics)

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

    def _apply_step(self, step: int, stage_index: int, metrics: Mapping[str, object]) -> None:
        self._throughput.record(step)
        known = _known_metrics(metrics)
        transition = _transition_of(metrics)
        with self._state_lock:
            self._step = step
            self._stage_index = stage_index
            self._latest.update(known)
        self._refresh_bars(step, stage_index, transition)

    def _refresh_bars(self, step: int, stage_index: int, transition: float | None) -> None:
        last = len(self.stage_names) - 1
        for index, (task_id, name, start, steps) in enumerate(
            zip(self._stage_tasks, self.stage_names, self._stage_starts, self.steps_per_stage)
        ):
            completed = min(max(step - start, 0), steps)
            if index == stage_index:
                description = f"[bold cyan]▶ {escape(name)}"
            elif completed >= steps:
                description = f"[green]✓ {escape(name)}"
            else:
                description = f"[dim]  {escape(name)}"
            note = ""
            if index == stage_index and transition is not None and index < last:
                note = f"transition → {self.stage_names[index + 1]} {transition:.0%}"
            self._progress.update(task_id, completed=completed, description=description, note=note)
        rate = self._throughput.steps_per_second
        rate_text = "? steps/s" if rate is None else f"{rate:.2f} steps/s"
        remaining = format_duration(self._throughput.remaining(step))
        note = f"{rate_text}, {format_duration(self._throughput.elapsed)} elapsed, ETA {remaining}"
        self._progress.update(self._overall_task, completed=min(step, self.total_steps), note=note)

    def _apply_validation(self, step: int, losses: Mapping[str, object]) -> None:
        with self._state_lock:
            self._validation = (step, _floats(losses))

    def _apply_event(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._state_lock:
            self._events.append(f"{stamp}  step {self._step}: {text}")

    def _apply_status(self, text: str) -> None:
        with self._state_lock:
            self._status = text

    # --- log lines ----------------------------------------------------------------------------------------------------

    def write(self, text: str, *, keep: bool = False) -> None:
        """Append ``text`` (one entry per line, so tracebacks stay readable) to the log panel; plain stream when
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

    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> AbstractContextManager[None]:
        """Route ``logger`` into this dashboard (and ``log_file``) for the duration of the block. ``logger`` must be
        the ``training`` logger or one of its ancestors for the dashboard's own warning / fallback lines to reach it."""
        return _attach(self, logger, log_file)

    # --- state for tests ------------------------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        """The log lines currently shown (newest last)."""
        with self._lines_lock:
            return list(self._lines)

    def events(self) -> list[str]:
        """The event lines currently shown (newest last)."""
        with self._state_lock:
            return list(self._events)

    @property
    def latest_metrics(self) -> dict[str, float]:
        with self._state_lock:
            return dict(self._latest)

    @property
    def tasks(self) -> list[RichTask]:
        """The rich tasks of the bars: one per stage, then the overall bar."""
        return list(self._progress.tasks)

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich__(self) -> RenderableType:
        try:
            with self._state_lock:
                header = self._render_header()
                metrics = self._render_metrics()
                validation = self._render_validation()
                events = self._render_events()
            parts: list[RenderableType] = [header, self._progress, metrics]
            if validation is not None:
                parts.append(validation)
            parts += [events, self._render_log()]
            return Group(*parts)
        except Exception as error:  # runs on the Live thread: report, let the next public call disable the display
            self._render_error = error
            return Text(f"training dashboard render failed: {error!r}", style="bold red")

    def _render_header(self) -> RenderableType:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        details = "  ".join(f"{key}={value}" for key, value in self.details.items())
        grid.add_row(Text.assemble((self.run_name, "bold"), "  ", (details, "dim")), Text(self._status, style="bold yellow"))
        return grid

    def _render_metrics(self) -> RenderableType:
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, title=f"step {self._step}", title_justify="left")
        cells: list[str] = []
        for key, label in METRIC_COLUMNS:
            table.add_column(label, justify="right")
            value = self._latest.get(key)
            if value is None and key == "seconds/step":
                value = self._throughput.seconds_per_step  # the loop's own timing when the step dict has none
            cells.append("—" if value is None else format_metric(key, value))
        table.add_column("elapsed", justify="right")
        table.add_column("remaining", justify="right")
        cells += [format_duration(self._throughput.elapsed), format_duration(self._throughput.remaining(self._step))]
        table.add_row(*cells)
        return table

    def _render_validation(self) -> RenderableType | None:
        if self._validation is None:
            return None
        step, losses = self._validation
        table = Table(
            box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, title=f"validation (step {step})", title_justify="left"
        )
        for key in losses:
            table.add_column(key, justify="right")
        table.add_row(*(f"{value:.4f}" for value in losses.values()))
        return table

    def _render_events(self) -> RenderableType:
        body = Text("\n".join(self._events) or "(no events yet)", no_wrap=True, overflow="ellipsis")
        return Panel(body, title="events", title_align="left", border_style="dim")

    def _render_log(self) -> RenderableType:
        with self._lines_lock:
            body = Text("\n".join(self._lines) or "(no log output yet)", no_wrap=True, overflow="ellipsis")
        return Panel(body, title="log", title_align="left", border_style="dim")

    def render_text(self, width: int = 120) -> str:
        """The current display as plain text (tests, or a snapshot for a log file)."""
        console = Console(width=width, force_terminal=False, color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()


RunDashboard = TrainingDashboard | NoOpDashboard  # what `training_dashboard` yields; the type of `RunLogger`'s field


@contextmanager
def training_dashboard(
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
    enabled: bool | None = None,
    console: Console | None = None,
    stream: TextIO | None = None,
    clock: Clock = time.monotonic,
) -> Iterator[RunDashboard]:
    """The dashboard of a run: :class:`TrainingDashboard` when enabled (default: :func:`dashboard_enabled` — the
    ``TRAINING_DASHBOARD`` env var and a TTY on stdout), else the :class:`NoOpDashboard` console fallback. Both
    attach the ``training`` logger (or ``logger``) for the block and append every record to ``log_file``."""
    if enabled is None:
        enabled = dashboard_enabled(stream)
    if enabled:
        with TrainingDashboard.open(
            run_name,
            stage_names,
            steps_per_stage,
            total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=log_step_interval,
            log_file=log_file,
            logger=logger,
            console=console,
            stream=stream,
            clock=clock,
        ) as board:
            yield board
        return
    with NoOpDashboard.open(
        run_name,
        stage_names,
        steps_per_stage,
        total_steps,
        details=details,
        start_step=start_step,
        log_step_interval=log_step_interval,
        log_file=log_file,
        logger=logger,
        stream=stream,
        clock=clock,
    ) as fallback:
        yield fallback


# --- demo -----------------------------------------------------------------------------------------------------------------


def demo(seconds: float = 5.0, *, enabled: bool | None = None) -> None:
    """A fake two-stage run (30 + 20 optimizer steps over ``seconds``) driving the whole API, for a look at the
    display: ``uv run python -m training.ui.dashboard``. Piped (or ``TRAINING_DASHBOARD=0``) it shows the fallback."""
    import random

    stage_names = ["pretrain", "instruct"]
    steps_per_stage = [30, 20]
    total_steps = sum(steps_per_stage)
    pause = seconds / total_steps
    rng = random.Random(0)
    details = {"model": "crow-tiny", "dataset": "tiny", "device": "cpu", "precision": "32"}
    with training_dashboard(
        "demo-run", stage_names, steps_per_stage, total_steps, details=details, log_step_interval=5, enabled=enabled
    ) as board:
        board.note_event("no checkpoint found, starting from scratch")
        board.set_status("training")
        _log.info("Total training steps: %d (4 micro-batches each)", total_steps)
        loss = 6.0
        for step in range(1, total_steps + 1):
            time.sleep(pause)
            loss = loss * 0.97 + rng.uniform(-0.05, 0.05)
            stage_index = 0 if step <= steps_per_stage[0] else 1
            in_transition = 27 <= step <= 30
            metrics: dict[str, float] = {
                "loss": loss,
                "ppl": math.exp(loss),
                "lr": 3e-4 * min(step / 10, 1.0),
                "grad_norm": rng.uniform(0.5, 1.5),
                "tokens/second": rng.uniform(9_000, 11_000),
                "total_tokens": step * 8_192,
                TRANSITION_FLAG_KEY: float(in_transition),
                TRANSITION_PROGRESS_KEY: (step - 27) / 4 if in_transition else 0.0,
            }
            board.update_step(step, stage_index, metrics)
            if step == 27:
                board.note_event("starting transition 0 -> 1 (pretrain -> instruct)")
            if step == 30:
                board.note_event("transition complete, now in stage 1 (instruct)")
                board.note_event("saved checkpoint outputs/demo/checkpoints/step-00000030-demo-run-stage-0_end.pth")
            if step % 10 == 0:
                board.set_status("evaluating")
                board.update_validation(step, {"val_loss_4": loss + 0.3, "val_loss_8": loss + 0.15, "val_loss": loss + 0.1})
                board.set_status("training")
            if step % 25 == 0 and step != 30:
                board.note_event(f"saved checkpoint outputs/demo/checkpoints/step-{step:08d}-demo-run.pth")
            if step % 7 == 0:
                _log.info("step %d: sample log record (grad metrics, data composition ...)", step)
            if step == 40:
                _log.warning("step %d: an example warning (kept in the scrollback)", step)
        board.set_status("exporting")
        board.note_event("exported HuggingFace model to outputs/demo/hf_export")
        board.set_status("finished")
        time.sleep(0.5)


if __name__ == "__main__":
    demo()
