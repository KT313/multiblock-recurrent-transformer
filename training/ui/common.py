# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Names shared by the dashboard modules: the env var, the logger hierarchy, the log file, the enabling rule.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import TextIO

from ui.enabled import display_enabled

ENV_VAR = "TRAINING_DASHBOARD"
TRAINING_LOGGER_NAME = "training"  # the logger hierarchy of `training/`; `open()` attaches it by default
TRAIN_LOG_NAME = "train.log"  # full log of every run, appended under the run directory (`log_file=`)
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
