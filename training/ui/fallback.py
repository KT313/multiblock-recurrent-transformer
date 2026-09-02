# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The console fallback of the training dashboard: the same interface, one log line per interval, no display."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import TracebackType
from typing import TextIO

from training.ui.capture import attach_logger
from training.ui.common import TRAINING_LOGGER_NAME, Clock, log
from training.ui.format import (
    METRIC_COLUMNS,
    event_line,
    format_duration,
    format_metric,
    known_metrics,
    status_line,
    validation_line,
)
from training.ui.throughput import Throughput


class NoOpDashboard:
    """The console fallback: the :class:`~training.ui.board.TrainingDashboard` interface without a live display.

    ``update_step`` logs one line every ``log_step_interval`` steps (and at the last step), ``update_validation`` and
    ``note_event`` one line each, ``set_status`` a DEBUG line — all through ``logging`` on the ``training.ui.dashboard``
    logger, so :meth:`attach` (and :meth:`open`) decide where they go: the stream, the log file, or both. It captures
    nothing: a piped or ``nohup`` run keeps its plain console.

    The text of the lines comes from :meth:`step_line` and the ``*_line`` functions of :mod:`training.ui.format`; the
    live dashboard writes the same lines to the log file, so ``train.log`` reads the same under both.
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
        self.throughput = Throughput(total_steps, start_step=start_step, clock=clock)  # the live dashboard shares it

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

    def attach(self, logger: logging.Logger, *, log_file: Path | None = None) -> AbstractContextManager[logging.FileHandler | None]:
        """Route ``logger`` to the stream (plain lines) and ``log_file`` for the duration of the block. ``logger``
        must be the ``training`` logger or one of its ancestors for the fallback's own lines to reach it."""
        return attach_logger(self, logger, log_file)

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Nothing to clear: the fallback owns no display (the interface of ``TrainingDashboard.suspended``)."""
        yield

    def write(self, text: str, *, keep: bool = False) -> None:
        """A log record: one plain line on the stream."""
        self._stream.write(text + "\n")
        self._stream.flush()

    def _stage_name(self, stage_index: int) -> str:
        if 0 <= stage_index < len(self.stage_names):
            return self.stage_names[stage_index]
        return "?"

    def step_line(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> str | None:
        """Record ``step`` for the throughput estimate and return the step's log line — None at a step that is not
        logged (every ``log_step_interval``\\ th step and the last one are). ``transition`` is the progress of the
        running stage transition, None outside one."""
        self.throughput.record(step)
        if step % self.log_step_interval and step < self.total_steps:
            return None
        parts = [f"step {step}/{self.total_steps}", f"stage {stage_index} {self._stage_name(stage_index)}"]
        if transition is not None:
            parts.append(f"transition {transition:.0%}")
        known = known_metrics(metrics)
        parts += [f"{label} {format_metric(key, known[key])}" for key, label in METRIC_COLUMNS if key in known]
        if "seconds/step" not in known and self.throughput.seconds_per_step is not None:
            parts.append(f"s/step {self.throughput.seconds_per_step:.2f}s")
        parts.append(f"elapsed {format_duration(self.throughput.elapsed)}")
        parts.append(f"ETA {format_duration(self.throughput.remaining(step))}")
        return " | ".join(parts)

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        text = self.step_line(step, stage_index, transition, metrics)
        if text is not None:
            log.info(text)

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        log.info(validation_line(step, losses))

    def note_event(self, text: str) -> None:
        log.info(event_line(text))

    def set_status(self, text: str) -> None:
        log.debug(status_line(text))
