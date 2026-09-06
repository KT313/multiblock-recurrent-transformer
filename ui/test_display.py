# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the shared live display: the log panel and kept lines, the plain stream while disabled, the prompt
suspension, and the one-row text.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.text import Text

from ui.display import LiveDisplay, ResizeAwareLive, line
from ui.testing import DyingFile, console_output, screen_text, string_console


class MinimalDisplay(LiveDisplay):
    """
    A display with a one-line header, the log panel and the footer; the stream hooks only count.
    """

    def __init__(self, console: Console, stream: io.StringIO | None = None) -> None:
        super().__init__(stream=stream or io.StringIO(), console=console, refresh_per_second=50, log_lines=3)
        self.released = 0
        self.redirected = 0

    def _release_streams(self) -> None:
        self.released += 1

    def _redirect_streams(self) -> None:
        self.redirected += 1

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        with self._lock:
            yield Group(line("header", style="bold"), self._render_log(self._panel_height), self._footer("hint"))


def _live_of(display: LiveDisplay) -> ResizeAwareLive | None:
    return display._live  # through a call: mypy would otherwise keep the narrowing of an earlier assertion


@pytest.fixture
def display() -> Iterator[MinimalDisplay]:
    display = MinimalDisplay(string_console())
    display._start_live()
    try:
        yield display
    finally:
        display._stop_live()


def test_write_keeps_the_newest_lines_and_the_kept_records(display: MinimalDisplay) -> None:
    display.write("one")
    display.write("two\nthree", keep=True)
    display.write("four")
    assert display.lines() == ["two", "three", "four"], "the deque holds the last `log_lines` lines"
    assert display.kept() == ["two\nthree"]
    text = display.render_text(width=60)
    assert "header" in text and "three" in text and "four" in text and "one" not in text
    assert text.rstrip().splitlines()[-1] == "hint"
    assert "(no log output yet)" in MinimalDisplay(string_console()).render_text()


def test_print_kept_writes_the_records_once_to_the_console_file(display: MinimalDisplay) -> None:
    display.write("plain")
    display.write("warning line", keep=True)
    display._stop_live()
    display._print_kept()
    display._print_kept()
    assert screen_text(display._console, 120).strip() == "warning line"
    assert display.kept() == []


def test_write_goes_to_the_plain_stream_while_disabled() -> None:
    stream = io.StringIO()
    display = MinimalDisplay(string_console(), stream=stream)
    display.enabled = False
    display.write("late line", keep=True)
    assert stream.getvalue() == "late line\n" and display.lines() == [] and display.kept() == []


def test_suspended_stops_the_display_and_hands_the_streams_back(display: MinimalDisplay) -> None:
    console = display._console
    before = console_output(console)
    with display.suspended():
        assert _live_of(display) is None and display.released == 1 and display.redirected == 0
        assert "header" not in screen_text(console, 120), "the frame is erased while suspended"
    assert _live_of(display) is not None and display.redirected == 1
    assert len(console_output(console)) > len(before), "the display came back"
    idle = MinimalDisplay(string_console())
    with idle.suspended():  # without a display: a no-op
        assert idle.released == 0


def test_is_attached_covers_the_logger_and_its_children(display: MinimalDisplay) -> None:
    assert not display.is_attached("training")
    display._attached_logger_names.append("training")
    assert display.is_attached("training") and display.is_attached("training.ui") and not display.is_attached("trainingx")


def test_footer_names_the_log_file_first(display: MinimalDisplay) -> None:
    assert display._footer("a", "b").plain == "a · b"
    display._log_file = Path("/tmp/x/train.log")
    assert display._footer("hint").plain == "log: /tmp/x/train.log · hint"


def test_line_never_wraps_and_keeps_markup_literal() -> None:
    console = Console(width=20, force_terminal=False, color_system=None)
    with console.capture() as capture:
        # inside a Group, as in the frame: `console.print(Text)` itself re-joins the text and drops `no_wrap`
        console.print(Group(line("[bold]x[/bold] " + "y" * 40, style="dim")))
    (rendered,) = capture.get().splitlines()
    assert rendered.startswith("[bold]x[/bold] yyy") and rendered.endswith("…") and len(rendered) == 20


# --- the display after a terminal resize -------------------------------------------------------------------------------


def _clears(console: Console) -> int:
    return console_output(console).count("\x1b[2J\x1b[H")


def test_a_frame_for_a_new_terminal_size_is_preceded_by_a_clear_screen() -> None:
    console = string_console(80, height=24)
    live = ResizeAwareLive(Text("frame"), console=console, auto_refresh=False, transient=True)
    live.start(refresh=True)
    live.refresh()
    assert _clears(console) == 0 and "\x1b[2J" not in console_output(console), "the same size: rich's cursor-up erase"
    console.size = (60, 20)
    live.refresh()
    assert _clears(console) == 1
    assert console_output(console).rstrip().endswith("\x1b[2J\x1b[Hframe"), "the clear comes right before the frame"
    live.refresh()
    assert _clears(console) == 1, "the size is now the frame's: cursor-up erase again"
    live.stop()
    assert screen_text(console, 60) == "", "transient: the frame is gone"


def test_a_non_interactive_console_never_clears() -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=80, height=24)
    live = ResizeAwareLive(Text("frame"), console=console, auto_refresh=False, transient=True)
    live.start(refresh=True)
    console.size = (60, 20)
    live.refresh()
    live.stop()
    assert "\x1b[2J" not in console_output(console)


def test_the_display_of_a_dashboard_redraws_after_a_resize(display: MinimalDisplay) -> None:
    console = display._console
    live = _live_of(display)
    assert live is not None
    live.refresh()
    console.size = (100, 30)
    live.refresh()
    assert _clears(console) == 1
    assert screen_text(console, 100).count("header") == 1, "one frame on the screen: the old one is wiped"


# --- a dead terminal -----------------------------------------------------------------------------------------------------


def _dying_display() -> tuple[MinimalDisplay, DyingFile]:
    file = DyingFile()
    display = MinimalDisplay(Console(file=file, force_terminal=True, width=80, height=24))
    display._start_live()
    return display, file


def test_a_write_that_fails_closes_the_display_and_the_run_goes_on(caplog: pytest.LogCaptureFixture) -> None:
    display, file = _dying_display()
    live = _live_of(display)
    assert live is not None
    file.die()
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        live.refresh()  # what the refresh thread does 4-8 times a second
        assert display.headless and not display.enabled and _live_of(display) is None
        assert display._console.file is not file and display._plain_stream is not file
        display.write("later", keep=True)
        display._print_kept()
        display._stop_live()
        display._terminal_lost("again")
    assert [record.getMessage() for record in caplog.records] == [
        "terminal gone ([Errno 5] Input/output error): the display is closed, the run continues headless"
    ], "one warning, the second loss is a no-op"
    assert file.refused >= 1 and display.lines() == [], "nothing reached the dead terminal after the loss; `write` is plain"


def test_a_loss_noticed_while_stopping_is_handled_too() -> None:
    display, file = _dying_display()
    file.die()
    display._stop_live()  # rich's teardown writes: the frame erase, the cursor
    assert display.headless and _live_of(display) is None


def test_a_terminal_gone_before_the_display_opens_is_handled_without_raising() -> None:
    """
    `suspended()` reopens the display after a prompt; a terminal that died during the prompt fails rich's very
    first write (the hide-cursor code), which used to escape `_start_live`.
    """

    file = DyingFile()
    display = MinimalDisplay(Console(file=file, force_terminal=True, width=80, height=24))
    file.die()
    display._start_live()
    assert display.headless and not display.enabled and _live_of(display) is None and file.refused >= 1


def test_a_first_frame_that_fails_leaves_no_refresh_thread_behind() -> None:
    """
    Closing the display from inside rich's `start` (the first frame is written there) let `start` go on and
    start its refresh thread on the stopped Live, ticking for the rest of the process.
    """

    threads = set(threading.enumerate())
    file = DyingFile()
    display = MinimalDisplay(Console(file=file, force_terminal=True, width=80, height=24))
    file.die(after=1)  # the hide-cursor code gets through, the first frame does not
    display._start_live()
    assert "\x1b[?25l" in file.getvalue() and file.refused >= 1, "the death came between the two writes"
    assert display.headless and _live_of(display) is None
    assert not set(threading.enumerate()) - threads, "no refresh thread was started for the closed display"


def test_silencing_the_terminal_redirects_only_the_display_s_own_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `prepare.py > report.txt`: the data dashboard draws on stderr; stdout is the report and must survive the
    terminal's death.
    """

    class Stderr(io.StringIO):
        def fileno(self) -> int:
            return 2

    stream = Stderr()
    redirected: list[int] = []
    monkeypatch.setattr(os, "dup2", lambda _null, fd: redirected.append(fd))
    display = MinimalDisplay(Console(file=stream, force_terminal=True, width=80), stream=stream)
    display._silence_terminal()
    display._silence_terminal()
    assert redirected == [2], "once, and stdout is left alone"


def test_a_pending_loss_left_by_a_signal_handler_is_acted_on_at_the_next_refresh(caplog: pytest.LogCaptureFixture) -> None:
    display, file = _dying_display()
    live = _live_of(display)
    assert live is not None
    live.terminal_lost_pending = "SIGHUP: the terminal closed"
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        live.refresh()
    assert display.headless and "terminal gone (SIGHUP: the terminal closed)" in caplog.text


def test_sighup_marks_the_terminal_lost_instead_of_ending_the_process() -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers are installed on the main thread only")
    previous = signal.getsignal(signal.SIGHUP)
    display, file = _dying_display()
    live = _live_of(display)
    assert live is not None
    assert signal.getsignal(signal.SIGHUP) is not previous, "the display's handler is installed"
    os.kill(os.getpid(), signal.SIGHUP)
    assert live.terminal_lost_pending == "SIGHUP: the terminal closed"
    assert display._console.file is not file, "the handler already silenced the terminal"
    live.refresh()
    assert display.headless
    display._stop_live()
    assert signal.getsignal(signal.SIGHUP) is previous, "the previous handler is back once the display stopped"
