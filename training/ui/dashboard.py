# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal dashboard for one training run: stage bars, the latest metrics, validation losses, events and a log panel
— one live layout, and nothing else on the terminal while it is up. This module opens it (:func:`training_dashboard`);
the parts live in the sibling modules (``common``, ``format``, ``throughput``, ``capture``, ``fallback``, ``board``, ``demo``).

Built on ``rich`` (the only training package that imports it), the sibling of ``data_preparation/lib/ui/dashboard.py``
(with which it shares the console capture of ``data_preparation/lib/ui/capture.py``: the line sinks, the log handler,
``attach_logger`` and the logging / stream captures). A :class:`TrainingDashboard` owns exactly one *transient*
``rich.live.Live`` display on stdout that renders, top to bottom: a header (run name, model / dataset config names,
device / precision, the current status), one progress bar per training stage plus an overall bar (optimizer steps,
ETA from a smoothed steps-per-second estimate), a table with the metrics of the latest optimizer step, the latest
validation losses per recurrence depth, the last few events (checkpoints, resume point, stage transitions, export),
a panel with the last ``logging`` records and a footer naming the log file. Every row is a single line (``no_wrap``
+ ellipsis, so row heights never jump), the log panel — and then the events panel — shrink on a short terminal so
the frame always fits, every mutation and every render holds one lock (the loop writes, the Live thread reads), and
the display redraws on its own bounded timer: :meth:`TrainingDashboard.update_step` is O(1) per optimizer step.

While the display is up nothing may print around it (a stray line between two frames shifts the frame and leaves
its top behind in the scrollback — duplicated bars), so ``__enter__`` also (:class:`~training.ui.capture.TerminalCapture`)

* routes *every* ``logging`` record into the log panel: a handler on the root logger, while the plain
  ``StreamHandler``\\s that libraries such as ``transformers`` / ``datasets`` / ``huggingface_hub`` put on their own
  loggers are detached for the duration (they would write to the real stdout / stderr behind the display),
* replaces ``warnings.showwarning`` so that every ``warnings.warn`` is one kept record on ``training.warnings``
  (``UserWarning: text (file:line)`` — one line, the message first — whether the process writes warnings to stderr or
  routes them through ``logging.captureWarnings``),
* replaces ``sys.stdout`` / ``sys.stderr`` with line sinks that log what is written to them (stray prints, bare
  stderr writes, the final line of a tqdm bar, handlers created later) on ``training.stdout`` (INFO) and
  ``training.stderr`` (WARNING, i.e. kept), and
* sets ``WANDB_CONSOLE=off`` / ``WANDB_SILENT=true`` in the environment for a ``wandb.init`` inside the block. A wandb
  run created *before* the dashboard opens must pass ``wandb.Settings(**WANDB_QUIET_SETTINGS)`` itself: wandb's
  default ``console="wrap"`` replaces ``sys.stdout`` / ``sys.stderr`` on its own and prints its banner to stderr.

Only what bypasses Python's streams (C++ warnings of torch written to file descriptor 2) is not captured.

Records of WARNING and above, and records logged with ``extra={"keep": True}`` (stage summaries, the final report),
are *kept*: shown in the panel like everything else and printed once, unwrapped, after the display closed. When the
block ends — normally, by an exception, or by Ctrl-C — the frame is erased (the display is transient), the streams
and handlers are restored, the kept lines are printed and then one static final summary (header, bars, metrics,
validation, events; ``final_frame=False`` turns it off): the scrollback of a run is exactly the kept lines followed
by that summary, never a frozen or duplicated frame. A SIGTERM must be turned into ``KeyboardInterrupt`` by the CLI
(as ``prepare.py`` does) to leave through the same path. :meth:`TrainingDashboard.suspended` clears the display
around a terminal prompt.

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
one per validation and one per event through ``logging`` (so a piped or ``nohup`` run still has a readable log; it
captures nothing). Both attach the ``training`` logger for the duration of the block: its plain stream handlers are
swapped for the dashboard's handler plus a file handler for ``log_file`` and restored afterwards.

A dashboard failure must never end a training run: every public method of :class:`TrainingDashboard` catches its own
exceptions, closes the display (restoring the terminal), logs one warning and from then on behaves like the
:class:`NoOpDashboard`.

``uv run python -m training.ui.dashboard [seconds]`` runs the scripted demo of :mod:`training.ui.demo`.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from rich.console import Console

from training.ui.board import TrainingDashboard
from training.ui.common import Clock, dashboard_enabled
from training.ui.fallback import NoOpDashboard

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
    final_frame: bool = True,
    console: Console | None = None,
    stream: TextIO | None = None,
    fallback_stream: TextIO | None = None,
    clock: Clock = time.monotonic,
) -> Iterator[RunDashboard]:
    """The dashboard of a run: :class:`TrainingDashboard` when enabled (default: :func:`dashboard_enabled` — the
    ``TRAINING_DASHBOARD`` env var and a TTY on stdout), else the :class:`NoOpDashboard` console fallback. Both
    attach the ``training`` logger (or ``logger``) for the block and append every record to ``log_file``.
    ``fallback_stream`` is where the plain-line fallback writes (default: ``stream``, then stdout) — the run's CLI
    passes stderr so a piped run's step lines land on the same stream as the log handlers' lines; the live display
    is given it too, for the fallback it switches to when it disables itself after an internal error."""
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
            final_frame=final_frame,
            console=console,
            stream=stream,
            fallback_stream=fallback_stream,  # a display that disables itself mid-run falls back to the same stream
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
        stream=fallback_stream if fallback_stream is not None else stream,
        clock=clock,
    ) as fallback:
        yield fallback


if __name__ == "__main__":
    from training.ui.demo import main

    sys.exit(main())
