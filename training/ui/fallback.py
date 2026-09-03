# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The console fallback of the training dashboard: the same four calls, one log line per interval, no display."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from ui.capture import attach_logger
from training.ui.capture import run_log_handlers
from training.ui.common import TRAINING_LOGGER_NAME, Clock, lines_log
from training.ui.format import event_line, status_line, step_line, validation_line
from training.ui.throughput import Throughput


class ConsoleFallbackDashboard:
    """The console fallback: the four calls of :class:`~training.ui.board.TrainingDashboard` without a live display.

    ``update_step`` logs one line every ``log_step_interval`` steps (and at the last step), ``update_validation`` and
    ``note_event`` one line each, ``set_status`` a DEBUG line, all on :data:`~training.ui.common.lines_log`, whose
    handlers :meth:`attach` installs for the block: the run's ``train.log`` and this dashboard's stream. It captures
    nothing. The live dashboard writes the same lines to the log file, so ``train.log`` reads the same under both.
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
        self.throughput = Throughput(total_steps, start_step=start_step, clock=clock)

    @contextmanager
    def attach(self, logger: logging.Logger | None = None, *, log_file: Path | None = None) -> Iterator[None]:
        """Route ``logger`` (default: the ``training`` logger) and this dashboard's own lines to the stream and
        ``log_file`` (one shared file handler) for the block. A logger above INFO is lowered to INFO for the block."""
        target = logger if logger is not None else logging.getLogger(TRAINING_LOGGER_NAME)
        with attach_logger(self, target, None, ensure_info_level=True), run_log_handlers(target, log_file, self._stream):
            yield

    @contextmanager
    def running(self, logger: logging.Logger | None = None, *, log_file: Path | None = None) -> Iterator[ConsoleFallbackDashboard]:
        """The dashboard in service for the block (:meth:`attach`); the interface of the live dashboard's
        ``running``."""
        with self.attach(logger, log_file=log_file):
            yield self

    def write(self, text: str, *, keep: bool = False) -> None:
        """A record of the attached logger: one plain line on the stream."""
        self._stream.write(text + "\n")
        self._stream.flush()

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        self.throughput.record(step)
        text = step_line(
            step,
            stage_index,
            transition,
            metrics,
            total_steps=self.total_steps,
            stage_names=self.stage_names,
            log_step_interval=self.log_step_interval,
            throughput=self.throughput,
        )
        if text is not None:
            lines_log.info(text)

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        lines_log.info(validation_line(step, losses))

    def note_event(self, text: str) -> None:
        lines_log.info(event_line(text))

    def set_status(self, text: str) -> None:
        lines_log.debug(status_line(text))
