# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the shared names and the enabling rule of the training dashboard.
"""

from __future__ import annotations

import io
import logging

import pytest

from training.ui.common import (
    DASHBOARD_LOGGER_NAME,
    ENV_VAR,
    MICRO_BATCHES_ENV,
    TRAINING_LOGGER_NAME,
    dashboard_enabled,
    log,
    micro_batches_shown,
)


def test_enabled_follows_env_and_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "0")
    assert dashboard_enabled(io.StringIO()) is False
    monkeypatch.setenv(ENV_VAR, "1")
    assert dashboard_enabled(io.StringIO()) is False, "StringIO is not a TTY"
    monkeypatch.delenv(ENV_VAR)
    assert dashboard_enabled(io.StringIO()) is False

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert dashboard_enabled(Tty()) is True
    monkeypatch.setenv(ENV_VAR, "off")
    assert dashboard_enabled(Tty()) is False


def test_micro_batches_shown_is_off_unless_the_env_var_enables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MICRO_BATCHES_ENV, raising=False)
    assert micro_batches_shown() is False
    for value in ("1", "true", " YES ", "on"):
        monkeypatch.setenv(MICRO_BATCHES_ENV, value)
        assert micro_batches_shown() is True, value
    for value in ("0", "false", "off", "", "2"):
        monkeypatch.setenv(MICRO_BATCHES_ENV, value)
        assert micro_batches_shown() is False, value


def test_the_dashboard_logger_sits_under_the_training_hierarchy() -> None:
    assert log.name == DASHBOARD_LOGGER_NAME == "training.ui.dashboard"
    assert log.parent is not None and log.name.startswith(TRAINING_LOGGER_NAME + ".")
    assert logging.getLogger(TRAINING_LOGGER_NAME) in _ancestors(log)


def _ancestors(logger: logging.Logger) -> list[logging.Logger]:
    ancestors: list[logging.Logger] = []
    parent = logger.parent
    while parent is not None:
        ancestors.append(parent)
        parent = parent.parent
    return ancestors
