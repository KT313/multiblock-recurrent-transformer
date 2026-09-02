# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""What the two live dashboards — ``DataDashboard`` (``data_preparation/lib/ui/dashboard.py``) and
``TrainingDashboard`` (``training/ui/board.py``) — share: one transient ``rich.live.Live`` display on a console, the
log panel (the last ``log_lines`` lines of everything logged while the display is up), the *kept* lines printed once
the display closed, :meth:`LiveDisplay.suspended` around a terminal prompt, and the current frame as plain text.

A subclass renders the frame (``__rich_console__``, under ``_lock``), opens and closes the display (its ``__enter__``
/ ``__exit__`` call :meth:`LiveDisplay._start_live` / :meth:`LiveDisplay._stop_live` around its own terminal
capture) and hands the streams back and forth around a prompt (:meth:`LiveDisplay._release_streams` /
:meth:`LiveDisplay._redirect_streams`). Every mutation and every render holds ``_lock``: the caller's threads write,
the Live thread reads.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.text import Text


def line(text: str, style: str = "") -> Text:
    """One terminal row: never wraps, cropped with an ellipsis; markup in ``text`` is not interpreted."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


class LiveDisplay:
    """The live display, log panel and kept lines of a dashboard; see the module docstring.

    ``stream`` is the terminal the display draws on (stderr for data preparation, stdout for training); ``console``
    replaces the console built on it (tests: a ``rich.console.Console`` over a ``StringIO``). ``enabled`` is what a
    subclass decides: while False, :meth:`write` prints plain lines to ``_plain_stream`` instead of the panel.
    """

    def __init__(self, *, stream: TextIO, console: Console | None, refresh_per_second: float, log_lines: int) -> None:
        self.enabled = True
        self._stream = stream
        self._plain_stream = stream  # where `write` goes while the display is disabled
        self._console = console if console is not None else Console(file=stream)
        self._refresh_per_second = refresh_per_second
        self._lock = threading.RLock()  # every mutation and every render
        self._log_lines = log_lines
        self._lines: deque[str] = deque(maxlen=log_lines)
        self._kept: list[str] = []
        self._log_file: Path | None = None  # named in the footer once `attach` is given one
        self._attached: list[str] = []  # the loggers `attach` routes into the panel directly (see `is_attached`)
        self._live: Live | None = None

    # --- the display ------------------------------------------------------------------------------------------------

    def _start_live(self) -> None:
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start(refresh=True)  # the first frame right away, not after the first refresh interval

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        if live is not None:
            live.stop()  # transient: the frame is erased, the cursor is back where the display began

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Clear the display and give the terminal (streams included) back for a prompt; it comes back afterwards."""
        if self._live is None:
            yield
            return
        self._stop_live()
        self._release_streams()
        try:
            yield
        finally:
            self._redirect_streams()
            self._start_live()

    def _release_streams(self) -> None:
        """The real ``sys.stdout`` / ``sys.stderr`` back (a subclass's capture)."""
        raise NotImplementedError

    def _redirect_streams(self) -> None:
        """``sys.stdout`` / ``sys.stderr`` captured again (a subclass's capture)."""
        raise NotImplementedError

    # --- log lines and kept records -----------------------------------------------------------------------------------

    def write(self, text: str, *, keep: bool = False) -> None:
        """Append ``text`` (one entry per line, so tracebacks stay readable) to the log panel; plain stream when
        disabled. With ``keep`` the text is also printed, unwrapped, once the display closed."""
        if not self.enabled:
            self._plain_stream.write(text + "\n")
            self._plain_stream.flush()
            return
        with self._lock:
            self._lines.extend(text.splitlines() or [""])
            if keep:
                self._kept.append(text)

    def _print_kept(self) -> None:
        """The kept records, unwrapped, once, after the display closed — on the console's own file, plainly."""
        with self._lock:
            kept, self._kept = self._kept, []
        file = self._console.file
        for text in kept:
            file.write(text + "\n")
        file.flush()

    def is_attached(self, logger_name: str) -> bool:
        """Whether records of ``logger_name`` reach the panel through a handler ``attach`` installed (the root-logger
        handler of the capture skips those, or the panel would show them twice)."""
        with self._lock:
            return any(logger_name == name or logger_name.startswith(name + ".") for name in self._attached)

    def lines(self) -> list[str]:
        """The log lines currently shown (newest last)."""
        with self._lock:
            return list(self._lines)

    def kept(self) -> list[str]:
        """The kept records not yet printed (they are printed when the display closes)."""
        with self._lock:
            return list(self._kept)

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """The frame (a subclass's layout; rendered by the Live thread, under ``_lock``)."""
        raise NotImplementedError

    def _render_log(self, height: int) -> Panel:
        """The log panel: the newest ``height`` lines, one row each."""
        lines = list(self._lines)[-height:]
        body = line("\n".join(lines) or "(no log output yet)")
        return Panel(body, title="log", title_align="left", border_style="dim", padding=(0, 1))

    def _footer(self, *parts: str) -> Text:
        """The dim last row: ``log: <file>`` (once ``attach`` named one) and ``parts``, dot-separated."""
        named = [f"log: {self._log_file}"] if self._log_file is not None else []
        return line(" · ".join([*named, *parts]), style="dim")

    def render_text(self, width: int = 120, height: int = 50) -> str:
        """The current display as plain text (tests, or a snapshot for a log file)."""
        console = Console(width=width, height=height, force_terminal=False, color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()
