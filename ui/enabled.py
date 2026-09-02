# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The rule a live display opens under: its environment variable not set to a disabling value, and its stream a
terminal. ``data_preparation.lib.progress.progress_enabled`` (``DATA_PREP_PROGRESS``, stderr) and
``training.ui.common.dashboard_enabled`` (``TRAINING_DASHBOARD``, stdout) are this rule with their names filled in."""

from __future__ import annotations

import os
from typing import TextIO

DISABLING_VALUES = ("0", "false", "no", "off")


def display_enabled(env_var: str, stream: TextIO) -> bool:
    """False when ``env_var`` is set to ``0`` / ``false`` / ``no`` / ``off`` or when ``stream`` is not a TTY."""
    value = os.environ.get(env_var, "1").strip().lower()
    if value in DISABLING_VALUES:
        return False
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    return bool(isatty())
