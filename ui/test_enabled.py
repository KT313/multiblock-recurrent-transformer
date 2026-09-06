# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the enabling rule on a real pseudo-terminal: what rich refuses to draw on stays disabled too.
"""

from __future__ import annotations

import os
import pty
from collections.abc import Iterator
from typing import TextIO

import pytest

from ui.enabled import display_enabled

ENV_VAR = "UI_TEST_DISPLAY"


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[TextIO]:
    """
    The writing end of a pty, with the environment a plain terminal has.
    """

    for name in (ENV_VAR, "TTY_COMPATIBLE", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    master, slave = pty.openpty()
    with os.fdopen(slave, "w") as stream:
        yield stream
    os.close(master)


def test_a_plain_terminal_enables(terminal: TextIO) -> None:
    assert display_enabled(ENV_VAR, terminal) is True


def test_a_dumb_terminal_does_not(terminal: TextIO, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "dumb")  # an Emacs shell: rich's Live draws nothing there, the capture would swallow everything
    assert display_enabled(ENV_VAR, terminal) is False


def test_tty_compatible_zero_does_not(terminal: TextIO, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTY_COMPATIBLE", "0")
    assert display_enabled(ENV_VAR, terminal) is False
