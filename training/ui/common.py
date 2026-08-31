# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Names shared by the dashboard modules: the env var, the logger hierarchy, the log file, the enabling rule."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from typing import TextIO

from data_preparation.lib.progress import DISABLING_VALUES

ENV_VAR = "TRAINING_DASHBOARD"
TRAINING_LOGGER_NAME = "training"  # the logger hierarchy of `training/`; `open()` attaches it by default
TRAIN_LOG_NAME = "train.log"  # full log of every run, appended under the run directory (`log_file=`)
DASHBOARD_LOGGER_NAME = f"{TRAINING_LOGGER_NAME}.ui.dashboard"  # the dashboard's own records (fallback lines, its warning)

Clock = Callable[[], float]

# named explicitly (not `__name__`): it must sit under the attached `training` logger whatever module logs through it
log = logging.getLogger(DASHBOARD_LOGGER_NAME)


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
