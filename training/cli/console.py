# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Console logging shared by the CLI and the training dashboard."""

from __future__ import annotations

import logging
import sys

from data_preparation.lib.log import LOG_FORMAT, ProgressStreamHandler, configure_logging
from training.ui.common import TRAINING_LOGGER_NAME


def configure_console_logging(level: int = logging.INFO, rank: int = 0) -> logging.Logger:
    """
    Attach one stderr handler each to the `training` and `data_preparation` logger hierarchies (same handler type
    and line format); idempotent. Returns the `training` logger. The CLI's job: library code configures no logging.

    `rank` above 0 (a non-main rank under torchrun) logs WARNING and above only, every line prefixed with
    `[rank N]`: the main rank tells the story of the run, the others only speak up when something is wrong.
    """

    if rank > 0:
        level = max(level, logging.WARNING)
    line_format = f"[rank {rank}] {LOG_FORMAT}" if rank > 0 else LOG_FORMAT
    data_logger = configure_logging(level)  # the `data_preparation` hierarchy: the status table, split and build lines
    training_logger = logging.getLogger(TRAINING_LOGGER_NAME)
    training_logger.setLevel(level)
    handler = next((h for h in training_logger.handlers if isinstance(h, ProgressStreamHandler)), None)
    if handler is None:
        handler = ProgressStreamHandler(sys.stderr)
        training_logger.addHandler(handler)
    handler.setLevel(level)
    for owner in (training_logger, data_logger):
        for existing in owner.handlers:
            if isinstance(existing, ProgressStreamHandler):
                existing.setFormatter(logging.Formatter(line_format))
    return training_logger
