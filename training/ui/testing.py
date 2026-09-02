# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Helpers of the training dashboard tests: a step dict, the box characters that must not
survive a run, and the two dashboards opened in one call (constructor arguments plus `logger` / `log_file` of
`running`). The hand-advanced clock, the StringIO console and the VT emulator are the shared ones of ``ui.testing``."""

from __future__ import annotations

import logging
import math
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from training.ui.board import TrainingDashboard
from training.ui.fallback import ConsoleFallbackDashboard

BOX_CHARACTERS = ("╭", "╰", "│")  # the events / log panels; the static summary has none
STAGES = ["pretrain", "instruct"]
STEPS = [20, 10]
TOTAL = 30
LOGGER_NAME = "training.test_dashboard"


def live_board(
    *args: Any, logger: logging.Logger | None = None, log_file: Path | None = None, **kwargs: Any
) -> AbstractContextManager[TrainingDashboard]:
    """A `TrainingDashboard(*args, **kwargs)` running for the block: `logger` attached, `log_file` appended, the
    display up."""
    return TrainingDashboard(*args, **kwargs).running(logger, log_file=log_file)


def fallback_board(
    *args: Any, logger: logging.Logger | None = None, log_file: Path | None = None, **kwargs: Any
) -> AbstractContextManager[ConsoleFallbackDashboard]:
    """A `ConsoleFallbackDashboard(*args, **kwargs)` running for the block: `logger` attached, `log_file` appended."""
    return ConsoleFallbackDashboard(*args, **kwargs).running(logger, log_file=log_file)


def metrics(step: int, loss: float = 3.0, **extra: float) -> dict[str, float]:
    """A step dict like ``RunLogger.log_step`` passes on."""
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "lr": 3e-4,
        "grad_norm": 1.25,
        "tokens/second": 12_345.6,
        "total_tokens": step * 8_192,
        **extra,
    }
