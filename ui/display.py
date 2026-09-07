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

Two failures the display survives, and only one of them touches the terminal.

Dead terminal (`_terminal_lost`): a frame write fails with an `OSError`, or a SIGHUP whose probe finds the terminal
gone. The display closes, the terminal's descriptor goes to `/dev/null`, one WARNING names the log file and the run
continues headless.

A frame that fails to render (`_display_failed`): a layout bug, on a terminal that is fine. The display closes and
one WARNING says so; nothing is silenced, so the plain stream stays the real one and log lines, tracebacks and the
exit message keep their terminal.

SIGHUP alone is not a closed terminal (`kill -HUP` is also a "reload" convention): the handler probes the display's
own descriptor and, when the terminal answers, logs one line and changes nothing. A confirmed hangup with a Live up
is left to the refresh thread; a teardown inside the handler could deadlock on rich's own lock. Ctrl-C and SIGTERM
are unchanged.

Fork (`_reset_in_child`): the training DataLoader forks its workers while the display is up. A forked child copies
`_lock` with its owner, but threads do not survive a fork; a worker whose first log line reaches `write` while the
render thread had the lock at the fork would block forever - and the training loop stalls as soon as it waits for
that worker's batch. The hook gives every display a
fresh lock in the child, disabled, its plain stream on /dev/null (the worker's lines must not land under the
parent's dashboard; the log file still gets them through logging).
"""

from __future__ import annotations

import errno
import logging
import os
import re
import signal
import threading
import weakref
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

from rich.console import Console, ConsoleDimensions, ConsoleOptions, ConsoleRenderable, RenderResult
from rich.control import Control
from rich.live import Live
from rich.panel import Panel
from rich.segment import ControlType
from rich.text import Text


_SIGHUP_REASON = "SIGHUP: the terminal closed"  # the reason of a hangup the probe confirmed


def _fileno(stream: TextIO) -> int | None:
    try:
        return stream.fileno()
    except (OSError, ValueError, AttributeError):  # a StringIO, a closed file, a sink without one
        return None


# an escape sequence: OSC (up to its terminator), CSI (parameters, intermediates, final byte), or a two-byte ESC
_ANSI_SEQUENCE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)?|\[[0-9;:<=>?]*[ -/]*[@-~]|[ -/]*[0-~])")


def strip_ansi(text: str) -> str:
    """
    text without its terminal control sequences.

    rich strips only BEL / BS / VT / FF / CR from a :class:`Text`, so a coloured library line or a half-written
    progress bar would put its escapes into the frame: the terminal reads them (colour bleed, cursor moves) and
    they count as printable columns (misaligned borders). Everything the display shows goes through here.
    """

    return _ANSI_SEQUENCE.sub("", text)


def line(text: str, style: str = "") -> Text:
    """
    One terminal row: never wraps, cropped with an ellipsis; markup in text is not interpreted.
    """

    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


class ResizeAwareLive(Live):
    """
    A rich.live.Live that clears the screen before a frame drawn for a new terminal size and reports a failed
    frame through one of two callbacks instead of crashing the refresh thread: on_terminal_lost(reason) for an
    OSError (the terminal is gone), on_render_failed(error) for anything else (the frame is broken, the terminal
    is not). Either way the display closes and the run goes on.
    """

    def __init__(
        self,
        *args: Any,
        on_terminal_lost: Callable[[str], None] | None = None,
        on_render_failed: Callable[[BaseException], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frame_size: ConsoleDimensions | None = None  # the terminal size the previous frame was drawn for
        self._on_terminal_lost = on_terminal_lost
        self._on_render_failed = on_render_failed
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
            except OSError as error:  # the terminal is gone (EIO / EBADF)
                if self._starting:  # let rich's start stop the display and raise, else it would still start its refresh thread
                    raise
                reason = str(error)
            except Exception as error:  # a frame that does not render: the terminal itself is fine
                if self._starting:
                    raise
                if self._on_render_failed is not None:
                    self._on_render_failed(error)
                return
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


_displays: weakref.WeakSet[LiveDisplay] = weakref.WeakSet()
_fork_hook_registered = False  # `register_at_fork` cannot be undone, so the hook is registered once


def _reset_in_child() -> None:
    """
    In a forked child (a DataLoader worker): a fresh lock, no display, plain writes to /dev/null.
    """

    for display in _displays:
        display._lock = threading.RLock()
        display._silence_lock = threading.RLock()
        display.enabled = False
        display._live = None
        display._plain_stream = open(os.devnull, "w")  # noqa: SIM115  # stays open for the rest of the child


def _register_fork_hook() -> None:
    global _fork_hook_registered
    if not _fork_hook_registered and hasattr(os, "register_at_fork"):
        _fork_hook_registered = True
        os.register_at_fork(after_in_child=_reset_in_child)


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
        self._display_broken = False  # set by `_display_failed`: a frame did not render, the terminal is fine
        self._null_file: TextIO | None = None  # /dev/null once headless
        self._silence_lock = threading.RLock()  # `_silence_terminal` runs on the refresh thread and in the SIGHUP handler
        self._previous_sighup: Any = None  # SIGHUP handler replaced at start, restored at stop
        _register_fork_hook()
        _displays.add(self)

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
            on_render_failed=self._display_failed,
        )
        self._install_sighup_handler()
        # a terminal gone before the first frame: rich's start either stopped the display itself (a failed frame) or
        # never got as far as its render hook (a failed hide-cursor write), so there is nothing left to stop
        try:
            self._live.start(refresh=True)  # first frame right away
        except OSError as error:
            self._live = None
            self._terminal_lost(str(error))
        except Exception as error:  # a broken first frame must not escape into a caller half-way through its setup
            self._live = None
            self._display_failed(error)

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

        The streams are handed back whenever the capture holds them, not only while a Live is up: after a render
        failure the display is gone but the sinks are not, and a question written to them would never reach the
        person answering it. The display comes back only if it was up and is still enabled.
        """

        was_live = self._live is not None
        captured = self._streams_captured
        if was_live:
            self._stop_live()
        if captured:
            self._release_streams()
        try:
            yield
        finally:
            if captured:
                self._redirect_streams()
            if was_live and self.enabled:
                self._start_live()

    @property
    def _streams_captured(self) -> bool:
        """
        Whether the subclass's capture holds sys.stdout / sys.stderr right now. Both dashboards ask their capture;
        the default answers for a display whose streams live and die with its Live.
        """

        return self._live is not None

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

    def _display_failed(self, error: BaseException) -> None:
        """
        A frame did not render on a terminal that is fine (a layout bug): close the display, say so once, silence
        nothing. `write` goes to the real plain stream from here on and the captured sys.stdout / sys.stderr feed
        it, so the run keeps its output. Idempotent; safe on any thread.
        """

        with self._lock:
            if self._display_broken:
                return
            self._display_broken = True  # before the teardown: rich's `Live.stop` draws one more frame, which fails again
        self.enabled = False
        with suppress(Exception):  # whatever breaks the frame breaks the last one `Live.stop` draws as well
            self._stop_live()
        self.logger.warning("display failed (%r): the run continues with plain output", error)

    def _silence_terminal(self) -> None:
        """
        Point the console and the plain stream at /dev/null; when the display drew on the process's own
        stdout or stderr, that file descriptor too (only that one: the other may be a redirected report).
        Idempotent under its own lock, which the refresh thread and the SIGHUP handler can both want.
        """

        with self._silence_lock:
            if self._null_file is None:
                null_file = open(os.devnull, "w")  # noqa: SIM115  # stays open for the rest of the process
                fd = _fileno(self._stream)
                if self._console.file is self._stream and fd in (1, 2):
                    os.dup2(null_file.fileno(), fd)
                self._null_file = null_file
            self._console.file = self._null_file
            self._plain_stream = self._null_file

    def _install_sighup_handler(self) -> None:
        """
        SIGHUP asks the terminal whether it is still there instead of ending the process. Main thread only, as
        Python requires for signal handlers.
        """

        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGHUP"):
            return

        def on_sighup(signum: int, frame: FrameType | None) -> None:
            self._sighup()

        self._previous_sighup = signal.signal(signal.SIGHUP, on_sighup)

    def _sighup(self) -> None:
        """
        SIGHUP: silence the terminal only once a probe says it is gone.

        A hangup is not the only sender - `kill -HUP` is a widespread "reload / notify" convention - and blinding a
        live terminal for one would cost the run every line it still has to print. The probe is
        `os.get_terminal_size` on the display's own descriptor, cheap enough for a handler (which runs on the main
        thread, between bytecodes); ENOTTY answers for a stream that never was a terminal (a redirected report),
        which has nothing to lose either.

        The handler asks the fd rather than reading a Live captured when it was installed: after a terminal loss or
        a render failure there is no Live, and a note nobody reads would leave a real hangup to surface as an EIO
        somewhere in the run.
        """

        fd = _fileno(self._stream)
        if fd is not None:
            try:
                os.get_terminal_size(fd)
            except OSError as error:
                if error.errno not in (errno.ENOTTY, errno.EINVAL):
                    self._terminal_hung_up()
                    return
        self.logger.info("SIGHUP received, terminal still open")

    def _terminal_hung_up(self) -> None:
        """
        A confirmed hangup: silence the terminal now and leave the teardown to the refresh thread while one runs.

        `Live.stop` waits for rich's own lock, which the refresh thread holds for the length of a frame - a frame
        that waits for `_lock`, which this thread may be holding where the signal interrupted it.
        """

        self._silence_terminal()
        live = self._live
        if live is not None:
            live.terminal_lost_pending = _SIGHUP_REASON
        else:
            self._terminal_lost(_SIGHUP_REASON)

    def _restore_sighup_handler(self) -> None:
        if self._previous_sighup is None or threading.current_thread() is not threading.main_thread():
            return  # off the main thread the handler stays; it is harmless
        signal.signal(signal.SIGHUP, self._previous_sighup)
        self._previous_sighup = None

    # --- log lines and kept records -----------------------------------------------------------------------------------

    def write(self, text: str, *, keep: bool = False) -> None:
        """
        Append text to the log panel, one entry per line; to the plain stream when disabled. With keep it
        is also printed once the display closed. Terminal control sequences are stripped (:func:`strip_ansi`).
        """

        text = strip_ansi(text)
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
