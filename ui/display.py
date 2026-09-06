# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Shared base of the two live dashboards (`DataDashboard`, `TrainingDashboard`): a transient `rich.live.Live`
display, a log panel with the last `log_lines` logged lines, *kept* lines printed after the display closes,
`suspended()` around a terminal prompt, and the frame as plain text.

A subclass renders the frame in `__rich_console__`, calls `_start_live` / `_stop_live` around its own terminal
capture and implements `_release_streams` / `_redirect_streams` for prompts. Every mutation and render holds
`_lock`.

Terminal resize: `ResizeAwareLive` clears the screen before the first frame at a new size. The check runs on
the refresh timer, so no SIGWINCH handler is needed.

Dead terminal (`_terminal_lost`): on a failed frame write or SIGHUP the display closes, the terminal's descriptor
goes to `/dev/null`, one WARNING names the log file and the run continues headless. The SIGHUP handler only silences
the streams and leaves a note for the refresh thread; a teardown inside the handler could deadlock on `_lock`.
Ctrl-C and SIGTERM are unchanged.
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
    """
    One terminal row: never wraps, cropped with an ellipsis; markup in text is not interpreted.
    """

    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


class ResizeAwareLive(Live):
    """
    A rich.live.Live that clears the screen before a frame drawn for a new terminal size and reports a
    dead terminal, or a frame that fails to render, through on_terminal_lost(reason) instead of crashing the
    refresh thread (the display then closes and the run goes on with plain logging).
    """

    def __init__(self, *args: Any, on_terminal_lost: Callable[[str], None] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frame_size: ConsoleDimensions | None = None  # the terminal size the previous frame was drawn for
        self._on_terminal_lost = on_terminal_lost
        self._starting = False  # inside `start`, where rich handles a failed first frame itself (see `refresh`)
        self.terminal_lost_pending: str | None = None  # set by the SIGHUP handler, handled at the next refresh

    def start(self, refresh: bool = False) -> None:
        self._starting = True
        try:
            super().start(refresh)
        finally:
            self._starting = False

    def refresh(self) -> None:
        reason = self.terminal_lost_pending
        if reason is None:
            try:
                super().refresh()
                return
            except Exception as error:  # the terminal is gone (OSError: EIO / EBADF), or a frame failed to render
                if self._starting:  # let rich's start stop the display and raise, else it would still start its refresh thread
                    raise
                reason = str(error) if isinstance(error, OSError) else f"render failed: {error!r}"
        if self._on_terminal_lost is not None:
            self._on_terminal_lost(reason)

    def process_renderables(self, renderables: list[ConsoleRenderable]) -> list[ConsoleRenderable]:
        renderables = super().process_renderables(renderables)  # interactive: [erase previous frame, ..., frame]
        if not self.console.is_interactive:
            return renderables
        size = self.console.size
        if self._frame_size is not None and size != self._frame_size:
            renderables[0] = Control(ControlType.CLEAR, ControlType.HOME)  # rich's cursor-up erase assumes the old size
        self._frame_size = size
        return renderables


class LiveDisplay:
    """
    The live display, log panel and kept lines of a dashboard.

    stream is the terminal to draw on (stderr for data preparation, stdout for training); console replaces
    the console built on it (tests pass one over a StringIO). While enabled is False, :meth:`write` prints
    plain lines to _plain_stream instead of the panel.
    """

    def __init__(self, *, stream: TextIO, console: Console | None, refresh_per_second: float, log_lines: int) -> None:
        self.enabled = True
        self.logger = logging.getLogger(__name__)  # subclasses set their own; receives the "terminal gone" warning
        self._stream = stream
        self._plain_stream = stream  # where `write` goes while the display is disabled
        self._console = console if console is not None else Console(file=stream)
        self._refresh_per_second = refresh_per_second
        self._lock = threading.RLock()  # held by every mutation and render
        self._panel_height = log_lines
        self._panel_lines: deque[str] = deque(maxlen=log_lines)
        self._kept: list[str] = []
        self._log_file: Path | None = None  # named in the footer once `attach` is given one
        self._attached_logger_names: list[str] = []  # logger names `attach` routes into the panel directly
        self._live: ResizeAwareLive | None = None
        self._headless = False  # set by `_terminal_lost`
        self._null_file: TextIO | None = None  # /dev/null once headless
        self._previous_sighup: Any = None  # SIGHUP handler replaced at start, restored at stop

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
        # a terminal gone before the first frame: rich's start either stopped the display itself (a failed frame) or
        # never got as far as its render hook (a failed hide-cursor write), so there is nothing left to stop
        try:
            self._live.start(refresh=True)  # first frame right away
        except OSError as error:
            self._live = None
            self._terminal_lost(str(error))

    def _stop_live(self) -> None:
        live, self._live = self._live, None
        self._restore_sighup_handler()
        if live is None:
            return
        try:
            live.stop()  # transient: erases the frame
        except OSError as error:  # terminal died since the last frame; `_live` is None already, so no second stop
            self._terminal_lost(str(error))

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """
        Close the display and hand the terminal back for a prompt; reopen it afterwards.
        """

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
        """
        Restore the real sys.stdout / sys.stderr (subclass capture).
        """

        raise NotImplementedError

    def _redirect_streams(self) -> None:
        """
        Capture sys.stdout / sys.stderr again (subclass capture).
        """

        raise NotImplementedError

    # --- a dead terminal ----------------------------------------------------------------------------------------------

    @property
    def headless(self) -> bool:
        """
        Whether the terminal went away and the display closed itself.
        """

        return self._headless

    def _terminal_lost(self, reason: str) -> None:
        """
        The terminal is gone: silence it, close the display, warn in the log. Idempotent; safe on any thread.
        """

        with self._lock:
            if self._headless:
                return
            self._headless = True
        self._silence_terminal()
        self.enabled = False
        self._stop_live()
        log_hint = f"; its log: {self._log_file}" if self._log_file is not None else ""
        self.logger.warning("terminal gone (%s): the display is closed, the run continues headless%s", reason, log_hint)

    def _silence_terminal(self) -> None:
        """
        Point the console and the plain stream at /dev/null; when the display drew on the process's own
        stdout or stderr, that file descriptor too (only that one: the other may be a redirected report).
        Lock-free, a signal handler calls it.
        """

        if self._null_file is None:
            self._null_file = open(os.devnull, "w")  # noqa: SIM115  # stays open for the rest of the process
            fd = _fileno(self._stream)
            if self._console.file is self._stream and fd in (1, 2):
                os.dup2(self._null_file.fileno(), fd)
        self._console.file = self._null_file
        self._plain_stream = self._null_file

    def _install_sighup_handler(self) -> None:
        """
        SIGHUP leaves a note for the refresh thread instead of ending the process. Main thread only, as Python
        requires for signal handlers.
        """

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
            return  # off the main thread the handler stays; it is harmless
        signal.signal(signal.SIGHUP, self._previous_sighup)
        self._previous_sighup = None

    # --- log lines and kept records -----------------------------------------------------------------------------------

    def write(self, text: str, *, keep: bool = False) -> None:
        """
        Append text to the log panel, one entry per line; to the plain stream when disabled. With keep it
        is also printed once the display closed.
        """

        if not self.enabled:
            self._plain_stream.write(text + "\n")
            self._plain_stream.flush()
            return
        with self._lock:
            self._panel_lines.extend(text.splitlines() or [""])
            if keep:
                self._kept.append(text)

    def _print_kept(self) -> None:
        """
        Print the kept records once, plainly, on the console's file.
        """

        with self._lock:
            kept, self._kept = self._kept, []
        file = self._console.file
        for text in kept:
            file.write(text + "\n")
        file.flush()

    def is_attached(self, logger_name: str) -> bool:
        """
        Whether logger_name already reaches the panel through a handler attach installed (the root handler
        skips those to avoid duplicates).
        """

        with self._lock:
            return any(logger_name == name or logger_name.startswith(name + ".") for name in self._attached_logger_names)

    def lines(self) -> list[str]:
        """
        The log lines currently shown (newest last).
        """

        with self._lock:
            return list(self._panel_lines)

    def kept(self) -> list[str]:
        """
        The kept records not yet printed (they are printed when the display closes).
        """

        with self._lock:
            return list(self._kept)

    # --- rendering ------------------------------------------------------------------------------------------------------

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """
        The frame (a subclass's layout), rendered by the Live thread under _lock.
        """

        raise NotImplementedError

    def _render_log(self, height: int) -> Panel:
        """
        The log panel: the newest height lines, one row each.
        """

        lines = list(self._panel_lines)[-height:]
        body = line("\n".join(lines) or "(no log output yet)")
        return Panel(body, title="log", title_align="left", border_style="dim", padding=(0, 1))

    def _footer(self, *parts: str) -> Text:
        """
        The dim last row: log: <file> (once attach named one) and parts, dot-separated.
        """

        log_name = [f"log: {self._log_file}"] if self._log_file is not None else []
        return line(" · ".join([*log_name, *parts]), style="dim")

    def render_text(self, width: int = 120, height: int = 50) -> str:
        """
        The current frame as plain text (tests, or a snapshot for a log file).
        """

        console = Console(width=width, height=height, force_terminal=False, color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()
