# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the shared live display: the log panel and kept lines, the plain stream while disabled, the prompt
suspension, and the one-row text."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live

from ui.display import LiveDisplay, line
from ui.testing import console_output, screen_text, string_console


class MinimalDisplay(LiveDisplay):
    """A display with a one-line header, the log panel and the footer; the stream hooks only count."""

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
            yield Group(line("header", style="bold"), self._render_log(self._log_lines), self._footer("hint"))


def _live_of(display: LiveDisplay) -> Live | None:
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
    display._attached.append("training")
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
