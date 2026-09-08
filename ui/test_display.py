# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the shared live display: the log panel and kept lines, the plain stream while disabled, the prompt
suspension, and the one-row text.
"""

from __future__ import annotations

import errno
import io
import logging
import multiprocessing
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
        self.captured = False  # what a subclass's capture would answer: set by the tests that need it

    def _release_streams(self) -> None:
        self.released += 1

    def _redirect_streams(self) -> None:
        self.redirected += 1

    @property
    def _streams_captured(self) -> bool:
        return self.captured or self._live is not None

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


def test_write_strips_terminal_control_sequences(display: MinimalDisplay) -> None:
    """
    U-M8: a library logging colours, or a stderr write carrying a title sequence, must not reach the frame: the
    terminal would act on the escapes and they would count as printable columns (misaligned borders).
    """

    display.write("\x1b[31mred\x1b[0m line")
    display.write("\x1b]0;a title\x07kept \x1b[1mline\x1b[0m", keep=True)
    assert display.lines() == ["red line", "kept line"]
    assert display.kept() == ["kept line"]
    assert "\x1b" not in display.render_text(width=60)
    stream = io.StringIO()
    disabled = MinimalDisplay(string_console(), stream=stream)
    disabled.enabled = False
    disabled.write("\x1b[2mdim\x1b[0m line")
    assert stream.getvalue() == "dim line\n", "the plain stream gets them stripped too"


def test_suspended_stops_the_display_and_hands_the_streams_back(display: MinimalDisplay) -> None:
    console = display._console
    before = console_output(console)
    with display.suspended():
        assert _live_of(display) is None and display.released == 1 and display.redirected == 0
        assert "header" not in screen_text(console, 120), "the frame is erased while suspended"
    assert _live_of(display) is not None and display.redirected == 1
    assert len(console_output(console)) > len(before), "the display came back"
    idle = MinimalDisplay(string_console())
    with idle.suspended():  # without a display and without a capture: a no-op
        assert idle.released == 0
    assert idle.redirected == 0 and _live_of(idle) is None


def test_suspended_hands_the_streams_back_while_the_display_is_off() -> None:
    """
    U-M1: a display closed by a render failure still has the capture on sys.stdout / sys.stderr; a prompt written
    into a line sink would become a log record and the run would wait for an answer nobody was asked for.
    """

    display = MinimalDisplay(string_console())
    display.captured = True
    display.enabled = False
    with display.suspended():
        assert display.released == 1 and display.redirected == 0, "the real streams are back for the prompt"
    assert display.redirected == 1 and _live_of(display) is None, "the capture is back, the closed display is not"


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


class _FdStream(io.StringIO):
    """
    A stream with a descriptor, like the real stderr the data dashboard draws on.
    """

    def fileno(self) -> int:
        return 2


def _display_on_a_descriptor() -> tuple[MinimalDisplay, _FdStream]:
    """
    A display whose plain stream has a descriptor, drawing on a console of its own (so nothing here can dup2 the
    test session's stderr away: `_silence_terminal` only touches the descriptor it also draws on).
    """

    stream = _FdStream()
    display = MinimalDisplay(Console(file=io.StringIO(), force_terminal=True, width=80, height=24), stream=stream)
    return display, stream


def test_silencing_the_terminal_redirects_only_the_display_s_own_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `prepare.py > report.txt`: the data dashboard draws on stderr; stdout is the report and must survive the
    terminal's death.
    """

    stream = _FdStream()
    redirected: list[int] = []
    monkeypatch.setattr(os, "dup2", lambda _null, fd: redirected.append(fd))
    display = MinimalDisplay(Console(file=stream, force_terminal=True, width=80), stream=stream)
    display._silence_terminal()
    display._silence_terminal()
    assert redirected == [2], "once, and stdout is left alone"


def test_a_frame_that_fails_to_render_closes_the_display_and_leaves_the_terminal_alone(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    U-H1: a layout bug is not a dead terminal. Silencing one for the other cost the run every line it had left:
    the log lines, the kept lines, the traceback of a later failure and the interpreter's own exit message.
    """

    redirected: list[int] = []
    monkeypatch.setattr(os, "dup2", lambda _null, fd: redirected.append(fd))
    display, stream = _display_on_a_descriptor()
    display._start_live()
    live = _live_of(display)
    assert live is not None
    display._console.file = _RaisingFile()  # not an OSError: a rich layout error looks like this to the refresh thread
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        live.refresh()
        live.refresh()  # the display is closed; a second failure is a no-op
    assert not display.enabled and not display.headless and _live_of(display) is None
    assert redirected == [] and display._console.file is not display._null_file, "the descriptor was not touched"
    assert display._plain_stream is stream, "plain writes go to the real stream"
    display.write("a line after the failure", keep=True)
    assert stream.getvalue() == "a line after the failure\n"
    assert [record.getMessage() for record in caplog.records] == [
        "display failed (RuntimeError('boom')): the run continues with plain output"
    ], "one warning, the second failure is silent"


def test_a_first_frame_that_does_not_render_closes_the_display_instead_of_escaping(caplog: pytest.LogCaptureFixture) -> None:
    """
    U-H2: `_start_live` runs half-way through `DataDashboard.__enter__`'s capture setup and from `suspended()`'s
    finally; an escaping layout error left the process with hijacked streams and a root handler for good.
    """

    threads = set(threading.enumerate())
    display = _BrokenFrameDisplay(string_console())
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        display._start_live()  # does not raise
    assert _live_of(display) is None and not display.enabled and not display.headless
    assert "display failed (ValueError('layout bug'))" in caplog.text
    assert not set(threading.enumerate()) - threads, "no refresh thread was left behind"


class _RaisingFile(io.StringIO):
    def write(self, text: str) -> int:
        raise RuntimeError("boom")


class _BrokenFrameDisplay(MinimalDisplay):
    """
    A display whose frame never renders (a layout bug, not a dead terminal).
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        raise ValueError("layout bug")


def test_a_pending_loss_left_by_a_signal_handler_is_acted_on_at_the_next_refresh(caplog: pytest.LogCaptureFixture) -> None:
    display, file = _dying_display()
    live = _live_of(display)
    assert live is not None
    live.terminal_lost_pending = "SIGHUP: the terminal closed"
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        live.refresh()
    assert display.headless and "terminal gone (SIGHUP: the terminal closed)" in caplog.text


def _probe_fails(_fd: int) -> os.terminal_size:
    raise OSError(errno.EIO, "Input/output error")


def _skip_off_the_main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers are installed on the main thread only")


def test_sighup_whose_terminal_answers_changes_nothing(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    U-M2: `kill -HUP` is a widespread "reload / notify" convention; taking it for a closed window blinded a
    terminal that was right there.
    """

    _skip_off_the_main_thread()
    monkeypatch.setattr(os, "dup2", lambda _null, _fd: None)
    monkeypatch.setattr(os, "get_terminal_size", lambda _fd: os.terminal_size((80, 24)))
    previous = signal.getsignal(signal.SIGHUP)
    display, stream = _display_on_a_descriptor()
    display._start_live()
    try:
        assert signal.getsignal(signal.SIGHUP) is not previous, "the display's handler is installed"
        with caplog.at_level(logging.INFO, logger="ui.display"):
            os.kill(os.getpid(), signal.SIGHUP)
        assert [record.getMessage() for record in caplog.records] == ["SIGHUP received, terminal still open"]
        assert not display.headless and display.enabled and display._plain_stream is stream
        assert _live_of(display) is not None and display._null_file is None, "no /dev/null was even opened"
    finally:
        display._stop_live()
    assert signal.getsignal(signal.SIGHUP) is previous, "the previous handler is back once the display stopped"


def test_sighup_whose_probe_fails_silences_the_terminal_and_closes_the_display(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_off_the_main_thread()
    monkeypatch.setattr(os, "dup2", lambda _null, _fd: None)
    monkeypatch.setattr(os, "get_terminal_size", _probe_fails)
    display, stream = _display_on_a_descriptor()
    display._start_live()
    live = _live_of(display)
    assert live is not None
    try:
        with caplog.at_level(logging.WARNING, logger="ui.display"):
            os.kill(os.getpid(), signal.SIGHUP)
            assert display._plain_stream is not stream, "the terminal is silenced right away"
            assert live.terminal_lost_pending == "SIGHUP: the terminal closed", "the teardown is the refresh thread's"
            live.refresh()  # what the refresh thread does 4-8 times a second
        assert display.headless and not display.enabled and _live_of(display) is None
        assert "terminal gone (SIGHUP: the terminal closed)" in caplog.text
    finally:
        display._stop_live()


def test_sighup_after_the_display_closed_is_confirmed_and_handled_by_the_handler_itself(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A render failure on the refresh thread leaves the handler installed (it is restored from the main thread only)
    and no Live to leave a note for. A real hangup afterwards must still be noticed here, not as an EIO deep in
    the run.
    """

    _skip_off_the_main_thread()
    monkeypatch.setattr(os, "dup2", lambda _null, _fd: None)
    previous = signal.getsignal(signal.SIGHUP)
    display, stream = _display_on_a_descriptor()
    display._start_live()
    closing = threading.Thread(target=display._display_failed, args=(ValueError("layout bug"),))  # the refresh thread's way
    closing.start()
    closing.join()
    assert _live_of(display) is None and display._plain_stream is stream
    assert signal.getsignal(signal.SIGHUP) is not previous, "the handler stays until the main thread stops the display"
    monkeypatch.setattr(os, "get_terminal_size", _probe_fails)
    with caplog.at_level(logging.WARNING, logger="ui.display"):
        os.kill(os.getpid(), signal.SIGHUP)
    assert display.headless and display._plain_stream is not stream
    assert "terminal gone (SIGHUP: the terminal closed)" in caplog.text
    assert signal.getsignal(signal.SIGHUP) is previous, "the handler restored itself on the main thread"


@pytest.mark.timeout(10)
def test_a_forked_child_gets_a_fresh_lock_and_writes_nowhere(display: MinimalDisplay) -> None:
    """
    The lock held by another thread at the fork (the render thread, in a run) must not block the child's first
    write, and that write must not reach the parent's panel.
    """

    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with display._lock:
            held.set()
            release.wait()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    held.wait()
    try:
        child = multiprocessing.get_context("fork").Process(target=display.write, args=("from the child",))
        child.daemon = True  # a child hung on the lock is killed at exit instead of hanging pytest
        child.start()
        child.join(timeout=3)
    finally:
        release.set()
        holder.join()
    assert child.exitcode == 0, "the child neither hung on the copied lock nor failed"
    assert "from the child" not in display.lines()
