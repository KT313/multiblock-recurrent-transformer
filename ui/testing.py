# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Helpers of the dashboard tests, shared by data_preparation/lib/ui and training/ui: a hand-advanced clock,
a StringIO console that behaves like a terminal, its output with and without control codes, and :class:`Screen`, the VT emulator that
shows what those control codes leave on a real terminal.
"""

from __future__ import annotations

import errno
import io
import re

from rich.console import Console

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class FakeClock:
    """
    A clock the tests advance by hand (injected as clock=).
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class DyingFile(io.StringIO):
    """
    A terminal that goes away: after :meth:`die` every write raises EIO, as a closed pty does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dead = False
        self.refused = 0  # writes attempted after the death

    def die(self) -> None:
        self.dead = True

    def write(self, text: str) -> int:
        if self.dead:
            self.refused += 1
            raise OSError(errno.EIO, "Input/output error")
        return super().write(text)


def string_console(width: int = 120, height: int | None = None) -> Console:
    """
    A terminal-like console writing into a StringIO (colours and cursor codes included).
    """

    return Console(file=io.StringIO(), force_terminal=True, width=width, height=height)


def console_output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def strip_ansi(text: str) -> str:
    """
    Terminal output without its control sequences (colours split the box titles: ╭─ + reset +  log ).
    """

    return _ANSI.sub("", text)


class Screen:
    """
    A minimal terminal emulator (CR, LF, cursor up/down, home, erase line / screen, SGR ignored) with unbounded
    scrollback: what the dashboard's control codes leave on the screen, as a real terminal would show it.
    """

    _CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

    def __init__(self, width: int) -> None:
        self.width = width
        self.lines: list[list[str]] = [[]]
        self.row = 0
        self.col = 0

    def _line(self, row: int) -> list[str]:
        while len(self.lines) <= row:
            self.lines.append([])
        return self.lines[row]

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                match = self._CSI.match(data, i)
                if match is None:
                    i += 1
                    continue
                params, command = match.group(1), match.group(2)
                n = int(params) if params.isdigit() else 1
                if command == "A":
                    self.row = max(0, self.row - n)
                elif command == "B":
                    self.row += n
                elif command == "C":
                    self.col += n
                elif command == "D":
                    self.col = max(0, self.col - n)
                elif command == "K":
                    line = self._line(self.row)
                    if params in ("", "0"):
                        del line[self.col :]
                    else:
                        line.clear()
                elif command == "J":
                    if params == "2":  # erase the whole screen (the cursor stays)
                        for line in self.lines:
                            line.clear()
                    else:
                        del self.lines[self.row + 1 :]
                        del self._line(self.row)[self.col :]
                elif command == "H":  # cursor to `row;col` (1-based), the top left by default
                    row, _, col = params.partition(";")
                    self.row, self.col = max(int(row or 1) - 1, 0), max(int(col or 1) - 1, 0)
                    self._line(self.row)
                i = match.end()
                continue
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self.row += 1
                self.col = 0
                self._line(self.row)
            elif ch == "\b":
                self.col = max(0, self.col - 1)
            elif ch >= " ":
                if self.col >= self.width:
                    self.row += 1
                    self.col = 0
                line = self._line(self.row)
                while len(line) <= self.col:
                    line.append(" ")
                line[self.col] = ch
                self.col += 1
            i += 1

    def text(self) -> str:
        return "\n".join("".join(line).rstrip() for line in self.lines).rstrip("\n")


def screen_text(console: Console, width: int) -> str:
    """
    What a terminal of width columns shows after everything the console wrote (its unbounded scrollback).
    """

    return screen_of(console_output(console), width)


def screen_of(raw: str, width: int) -> str:
    """
    :func:`screen_text` for raw terminal output (a pty transcript).
    """

    screen = Screen(width)
    screen.feed(raw)
    return screen.text()
