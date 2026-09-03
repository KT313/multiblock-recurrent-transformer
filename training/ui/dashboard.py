# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Terminal dashboard for one training run: stage bars, the latest metrics, validation losses, events and a log panel
in one live layout, and nothing else on the terminal while it is up. `training.logger.open_dashboard` opens it; the
parts live in the sibling modules (common, format, throughput, capture, fallback, board,
demo). This module holds the overview and the demo entry point.

Built on rich as the sibling of data_preparation/lib/ui/dashboard.py: both subclass
:class:`ui.display.LiveDisplay` and share the console capture of ui/capture.py. A
:class:`~training.ui.board.TrainingDashboard` owns one *transient* rich.live.Live display on stdout that renders
a header (run, model / dataset config, device / precision, status), one bar per stage plus an overall bar with an
ETA, the latest step's metrics, the latest validation losses per depth, the last events, the last logging
records and a footer naming the log file. Every row is one line, the panels shrink on a short terminal, every
mutation and render holds one lock, and the display redraws on its own timer: update_step is O(1).

While the display is up nothing may print around it, so __enter__ (:class:`~training.ui.capture.TerminalCapture`)
routes every logging record into the panel (third-party stream handlers detached for the duration), turns every
warnings.warn into one kept record, replaces sys.stdout / sys.stderr with line sinks, and quiets wandb
through the environment. Only writes that bypass Python's streams (C++ warnings on fd 2) are not captured.

Records of WARNING and above, and records logged with extra={"keep": True}, are *kept*: printed once, unwrapped,
after the display closed. When the block ends (normally, by an exception or by Ctrl-C) the frame is erased, streams
and handlers are restored, the kept lines and one static summary are printed (final_frame=False turns the
summary off). suspended clears the display around a terminal prompt. A terminal that dies mid-run closes the
display and the run continues headless with train.log as its output (:mod:`ui.display`).

Usage (RunLogger opens it through training.logger.open_dashboard)::

    board = TrainingDashboard(run_name, stage_names, steps_per_stage, total_steps, details={...}, log_step_interval=5)
    with board.running(log_file=run_dir / TRAIN_LOG_NAME):
        board.note_event("resumed from step-00000100-run.pth at step 100")
        board.update_step(step, stage_index, transition, metrics)  # every optimizer step, O(1), never raises
        board.update_validation(step, {"val_loss_4": 3.2, "val_loss": 3.1})
        board.set_status("saving checkpoint")

open_dashboard builds the live dashboard when enabled (TRAINING_DASHBOARD not 0 and stdout a terminal),
else the :class:`~training.ui.fallback.ConsoleFallbackDashboard` with the same four methods: one log line per
log_step_interval steps, per validation and per event, the same log file, no capture. Their own lines go through
training.ui.common.lines_log to the log file (and, for the fallback, the console).

A dashboard failure never ends a training run: every public method of TrainingDashboard catches its own
exceptions, closes the display, logs one warning and behaves like the console fallback from then on.

uv run python -m training.ui.dashboard [seconds] runs the scripted demo of :mod:`training.ui.demo`.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from training.ui.demo import main

    sys.exit(main())
