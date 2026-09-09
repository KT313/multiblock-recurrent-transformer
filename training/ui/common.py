# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Names shared by the dashboard modules: the env vars, the logger hierarchy, the log file, the enabling rules.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TextIO

from ui.enabled import display_enabled

ENV_VAR = "TRAINING_DASHBOARD"
MICRO_BATCHES_ENV = "DASHBOARD_SHOW_MICRO_BATCHES"  # 1/true/yes/on: the live dashboard adds a micro-batch bar (off by default)
ENABLING_VALUES = ("1", "true", "yes", "on")
TRAINING_LOGGER_NAME = "training"  # the logger hierarchy of `training/`; `open()` attaches it by default
TRAIN_LOG_NAME = "train.log"  # full log of every run, appended under the run directory (`log_file=`)
TRAIN_REPORT_NAME = "train_report.json"  # the `TrainingReport` of the last process, next to train.log
DASHBOARD_LOGGER_NAME = f"{TRAINING_LOGGER_NAME}.ui.dashboard"  # the dashboard's own records
LINES_LOGGER_NAME = f"{TRAINING_LOGGER_NAME}.ui.lines"  # the dashboards' step / validation / event lines
KEEP = {"keep": True}  # `extra=` of the records that must survive in the terminal scrollback under a live dashboard

Clock = Callable[[], float]

# named explicitly (not `__name__`): it must sit under the attached `training` logger whatever module logs through it
log = logging.getLogger(DASHBOARD_LOGGER_NAME)

# The step / validation / event lines of both dashboards. Only the handlers a dashboard's `attach` installs see them
# (the run's `train.log`, and the console for the fallback). Never propagated: the live display would otherwise
# print its own lines behind itself.
lines_log = logging.getLogger(LINES_LOGGER_NAME)
lines_log.propagate = False


def dashboard_enabled(stream: TextIO | None = None) -> bool:
    """
    False when TRAINING_DASHBOARD=0 (or false/no/off) or when stream (stdout) is not a TTY.
    """

    return display_enabled(ENV_VAR, sys.stdout if stream is None else stream)


def micro_batches_shown() -> bool:
    """
    True when DASHBOARD_SHOW_MICRO_BATCHES is 1 / true / yes / on: the live dashboard then shows a bar of the
    micro-batches of the running optimizer step (rank 0's share), for runs whose steps take long enough to watch.
    """

    return os.environ.get(MICRO_BATCHES_ENV, "0").strip().lower() in ENABLING_VALUES
