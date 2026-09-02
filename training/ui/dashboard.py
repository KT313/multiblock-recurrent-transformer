# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Terminal dashboard for one training run: stage bars, the latest metrics, validation losses, events and a log panel
— one live layout, and nothing else on the terminal while it is up. `training.logger.open_dashboard` opens it for a
run; the parts live in the sibling modules (``common``, ``format``, ``throughput``, ``capture``, ``fallback``,
``board``, ``demo``); this module holds the overview and the demo entry point.

Built on ``rich``, the sibling of ``data_preparation/lib/ui/dashboard.py``: both subclass
:class:`ui.display.LiveDisplay` (the live display, the log panel, the kept lines, ``suspended``) and share the console
capture of ``ui/capture.py`` (the line sinks, the log handler, ``attach_logger`` and the logging / stream captures).
A :class:`~training.ui.board.TrainingDashboard` owns exactly
one *transient* ``rich.live.Live`` display on stdout (redrawn from a cleared screen after a terminal resize) that
renders, top to bottom: a header (run name, model / dataset
config names, device / precision, the current status), one progress bar per training stage plus an overall bar
(optimizer steps, ETA from a smoothed steps-per-second estimate), a table with the metrics of the latest optimizer
step, the latest validation losses per recurrence depth, the last few events (checkpoints, resume point, stage
transitions, export), a panel with the last ``logging`` records and a footer naming the log file. Every row is a
single line (``no_wrap`` + ellipsis, so row heights never jump), the log panel — and then the events panel — shrink
on a short terminal so the frame always fits, every mutation and every render holds one lock (the loop writes, the
Live thread reads), and the display redraws on its own bounded timer: ``TrainingDashboard.update_step`` is O(1) per
optimizer step.

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
(as ``prepare.py`` does) to leave through the same path. ``TrainingDashboard.suspended`` clears the display around
a terminal prompt. A terminal that dies mid-run (closed window, dropped SSH session, SIGHUP) closes the display
and the run continues headless with ``train.log`` as its output (:mod:`ui.display`).

Usage (``RunLogger`` / ``train()`` open it once through ``training.logger.open_dashboard``)::

    board = TrainingDashboard(run_name, stage_names, steps_per_stage, total_steps, details={...}, log_step_interval=5)
    with board.running(log_file=run_dir / TRAIN_LOG_NAME):           # the `training` logger attached, the display up
        board.note_event("resumed from step-00000100-run.pth at step 100")
        board.update_step(step, stage_index, transition, metrics)  # every optimizer step, O(1), never raises
        board.update_validation(step, {"val_loss_4": 3.2, "val_loss": 3.1})
        board.set_status("saving checkpoint")

``open_dashboard`` builds the live dashboard when enabled (``TRAINING_DASHBOARD`` not ``0`` and stdout a terminal —
the rule of ``DATA_PREP_PROGRESS``) and otherwise the :class:`~training.ui.fallback.ConsoleFallbackDashboard` with
the same four methods: the console fallback, which writes the same log file and logs one line per
``log_step_interval`` steps, one per validation and one per event (so a piped or ``nohup`` run still has a readable
log; it captures nothing). Both attach the ``training`` logger for the duration of the block: its plain stream
handlers are swapped for the dashboard's handler plus one file handler for ``log_file`` and restored afterwards; their
own lines go through ``training.ui.common.lines_log`` — which shares that file handler — to the log file (and, for
the fallback, the console).

A dashboard failure must never end a training run: every public method of ``TrainingDashboard`` catches its own
exceptions, closes the display (restoring the terminal), logs one warning and from then on behaves like the
console fallback.

``uv run python -m training.ui.dashboard [seconds]`` runs the scripted demo of :mod:`training.ui.demo`.
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from training.ui.demo import main

    sys.exit(main())
