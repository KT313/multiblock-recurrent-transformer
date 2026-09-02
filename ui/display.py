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

The display survives a terminal resize (:class:`ResizeAwareLive`): rich re-reads the terminal size at every refresh,
so the next frame already fits the new size; what breaks is the erase of the previous frame, which rich does by
moving the cursor up as many lines as that frame had — after a resize the terminal has wrapped or reflowed those
lines, the count is wrong and the new frame lands over leftovers. A frame drawn for a size other than the previous
frame's is therefore preceded by a clear-screen + cursor-home instead: it starts at the top of a clean screen, and
every frame after it is erased correctly again. The check runs on the refresh timer (no ``SIGWINCH`` handler, so
it works from any thread and in tests); its one visible cost is that a resize wipes what stood above the frame.

A dead terminal never ends the run (:meth:`LiveDisplay._terminal_lost`): when a frame cannot be written any more
(``OSError``, the window was closed or the SSH session dropped) or ``SIGHUP`` arrives, the display closes itself
without touching the terminal, the process's stdout / stderr are pointed at ``/dev/null`` (later prints and C-level
writes must not raise anywhere), one WARNING names the log file, and the run continues headless — its log file is
its output from then on (``tail -f`` it; start long runs under tmux to come back to a live display). The SIGHUP
handler itself only silences the streams and leaves a note for the refresh thread: the teardown must not run
inside a signal handler, where the main thread may hold the display lock the refresh thread is waiting for.
Ctrl-C and SIGTERM keep their meaning (the CLIs handle them).
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

from rich.console import Console, ConsoleDimensions, ConsoleOptions, ConsoleRenderable, RenderResult
from rich.control import Control
from rich.live import Live
from rich.panel import Panel
from rich.segment import ControlType
from rich.text import Text


def _fileno(stream: TextIO) -> int | None:
    try:
        return stream.fileno()
    except (OSError, ValueError, AttributeError):  # a StringIO, a closed file, a sink without one
        return None


def line(text: str, style: str = "") -> Text:
    """One terminal row: never wraps, cropped with an ellipsis; markup in ``text`` is not interpreted."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


class ResizeAwareLive(Live):
    """``rich.live.Live`` whose frame is redrawn from a cleared screen when the terminal size changed since the
    previous frame (the module docstring says why); otherwise rich's own cursor-up erase is kept.

    ``on_terminal_lost`` is called (with the reason) instead of letting a failing write kill the refresh thread,
    and when :attr:`terminal_lost_pending` was set (a SIGHUP handler) before a refresh.
    """

    def __init__(self, *args: Any, on_terminal_lost: Callable[[str], None] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frame_size: ConsoleDimensions | None = None  # the terminal size the previous frame was drawn for
        self._on_terminal_lost = on_terminal_lost
        self.terminal_lost_pending: str | None = None  # the reason left by a signal handler; acted on at the next refresh

    def refresh(self) -> None:
        reason = self.terminal_lost_pending
        if reason is None:
            try:
                super().refresh()
                return
            except OSError as error:  # the terminal is gone: EIO / EBADF from the write
                reason = str(error)
        if self._on_terminal_lost is not None:
            self._on_terminal_lost(reason)

    def process_renderables(self, renderables: list[ConsoleRenderable]) -> list[ConsoleRenderable]:
        renderables = super().process_renderables(renderables)  # interactive: [erase the previous frame, ..., the frame]
        if not self.console.is_interactive:
            return renderables
        size = self.console.size
        if self._frame_size is not None and size != self._frame_size:
            renderables[0] = Control(ControlType.CLEAR, ControlType.HOME)
        self._frame_size = size
        return renderables


class LiveDisplay:
    """The live display, log panel and kept lines of a dashboard; see the module docstring.

    ``stream`` is the terminal the display draws on (stderr for data preparation, stdout for training); ``console``
    replaces the console built on it (tests: a ``rich.console.Console`` over a ``StringIO``). ``enabled`` is what a
    subclass decides: while False, :meth:`write` prints plain lines to ``_plain_stream`` instead of the panel.
    """

    def __init__(self, *, stream: TextIO, console: Console | None, refresh_per_second: float, log_lines: int) -> None:
        self.enabled = True
        self.logger = logging.getLogger(__name__)  # a subclass sets its own: where the "terminal gone" warning goes
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
        self._live: ResizeAwareLive | None = None
        self._headless = False  # the terminal is gone; the display closed itself (`_terminal_lost`)
        self._null_file: TextIO | None = None  # /dev/null, where the console and the plain stream write once headless
        self._previous_sighup: Any = None  # the SIGHUP handler found when the display started (restored on stop)

    # --- the display ------------------------------------------------------------------------------------------------

    def _start_live(self) -> None:
        self._live = ResizeAwareLive(
            self,
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
            on_terminal_lost=self._terminal_lost,
        )
        self._install_sighup_handler()
        self._live.start(refresh=True)  # the first frame right away, not after the first refresh interval

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        self._restore_sighup_handler()
        if live is None:
            return
        try:
            live.stop()  # transient: the frame is erased, the cursor is back where the display began
        except OSError as error:  # the terminal died since the last frame (`_live` is already None: no second stop)
            self._terminal_lost(str(error))

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

    # --- a dead terminal ----------------------------------------------------------------------------------------------

    @property
    def headless(self) -> bool:
        """Whether the terminal went away and the display closed itself; the run goes on without one."""
        return self._headless

    def _terminal_lost(self, reason: str) -> None:
        """The terminal is gone (``reason``: the failing write, or SIGHUP): silence it, close the display, say so in
        the log. Idempotent; safe on the refresh thread and on the caller's."""
        with self._lock:
            if self._headless:
                return
            self._headless = True
        self._silence_terminal()
        self.enabled = False
        self._stop_live()
        where = f"; its log: {self._log_file}" if self._log_file is not None else ""
        self.logger.warning("terminal gone (%s): the display is closed, the run continues headless%s", reason, where)

    def _silence_terminal(self) -> None:
        """Nothing writes to the dead terminal any more: the console and the plain stream go to ``/dev/null``, and
        when the display drew on the process's own stdout / stderr those file descriptors do too (a later print or a
        C-level write must not raise). Lock-free: a signal handler calls it."""
        if self._null_file is None:
            self._null_file = open(os.devnull, "w")  # noqa: SIM115  # stays open for the rest of the process
            if self._console.file is self._stream and _fileno(self._stream) in (1, 2):
                for fd in (1, 2):
                    os.dup2(self._null_file.fileno(), fd)
        self._console.file = self._null_file
        self._plain_stream = self._null_file

    def _install_sighup_handler(self) -> None:
        """SIGHUP (the terminal closed) leaves a note for the refresh thread instead of ending the process; main
        thread only (Python's rule for signal handlers)."""
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGHUP"):
            return
        live = self._live

        def on_sighup(signum: int, frame: FrameType | None) -> None:
            self._silence_terminal()
            if live is not None:
                live.terminal_lost_pending = "SIGHUP: the terminal closed"

        self._previous_sighup = signal.signal(signal.SIGHUP, on_sighup)

    def _restore_sighup_handler(self) -> None:
        if self._previous_sighup is None or threading.current_thread() is not threading.main_thread():
            return  # off the main thread (the refresh thread noticed the loss) the handler stays; it is harmless
        signal.signal(signal.SIGHUP, self._previous_sighup)
        self._previous_sighup = None

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
