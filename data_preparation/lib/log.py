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
    already_under_root = name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + ".")
    if not already_under_root:
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


class ProgressStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]  # stdlib generic only in stubs
    """The stderr handler :func:`configure_logging` attaches. Its own class so a second call finds it again and the
    dashboard, which swaps it out during a run, can tell it from a library's handler."""


def _our_handler(root: logging.Logger) -> ProgressStreamHandler | None:
    """The handler a previous ``configure_logging`` call attached, if any."""
    for handler in root.handlers:
        if isinstance(handler, ProgressStreamHandler):
            return handler
    return None


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a single stderr stream handler to the ``data_preparation`` logger; idempotent."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)
    handler = _our_handler(root)
    if handler is None:
        handler = ProgressStreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    handler.setLevel(level)
    return root
