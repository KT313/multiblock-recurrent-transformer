# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Logging for the data preparation package.

Every module logs under the ``data_preparation`` hierarchy (``get_logger(__name__)``); ``configure_logging`` attaches
one stream handler to that root, so ``prepare.py`` and ``training/train.py`` share the same configuration.
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER_NAME = "data_preparation"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Logger under the ``data_preparation`` hierarchy (``name`` is prefixed unless it already is)."""
    if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a single stderr stream handler to the ``data_preparation`` logger; idempotent."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    handler = next((h for h in root.handlers if getattr(h, "_data_preparation_handler", False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._data_preparation_handler = True  # type: ignore[attr-defined]  # marker to find our handler again
        root.addHandler(handler)
    handler.setLevel(level)
    return root
