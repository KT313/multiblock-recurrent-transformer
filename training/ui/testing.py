# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Helpers of the dashboard tests: a hand-advanced clock, StringIO consoles, the screen a real terminal would show
(the VT emulator of the data-prep dashboard tests), a step dict, the box characters that must not survive a run."""

from __future__ import annotations

import io
import math
import re

from rich.console import Console

from data_preparation.lib.ui.test_dashboard import _Screen  # the VT emulator: what the control codes leave on screen

BOX_CHARACTERS = ("╭", "╰", "│")  # the events / log panels; the static summary has none
STAGES = ["pretrain", "instruct"]
STEPS = [20, 10]
TOTAL = 30
LOGGER_NAME = "training.test_dashboard"


class FakeClock:
    """A clock the tests advance by hand (injected as ``clock=``)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def string_console(width: int = 120, height: int | None = None) -> Console:
    """A terminal-like console writing into a StringIO (colours and cursor codes included)."""
    return Console(file=io.StringIO(), force_terminal=True, width=width, height=height)


def console_output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def strip_ansi(text: str) -> str:
    """Terminal output without its control sequences (colours split the box titles: ``╭─`` + reset + `` log ``)."""
    return _ANSI.sub("", text)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def screen_text(console: Console, width: int) -> str:
    """What a terminal of ``width`` columns shows after everything the console wrote (its unbounded scrollback)."""
    screen = _Screen(width)
    screen.feed(console_output(console))
    return screen.text()


def screen_of(raw: str, width: int) -> str:
    """:func:`screen_text` for raw terminal output (a pty transcript)."""
    screen = _Screen(width)
    screen.feed(raw)
    return screen.text()


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
