# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Screen contents and terminal protocol emitted by synchronized, overwriting live frames."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.segment import Segment
from rich.text import Text

from ui.display import ResizeAwareLive
from ui.live_frame import BEGIN_SYNCHRONIZED_OUTPUT, END_SYNCHRONIZED_OUTPUT, OverwriteLiveRender, synchronize_output
from ui.testing import console_output, screen_text, string_console


def test_refresh_overwrites_shorter_lines_and_removed_rows_without_erasing() -> None:
    console = string_console(20, height=12)
    live = ResizeAwareLive(Text("long first line\nlong second line\nremoved row"), console=console,
                           auto_refresh=False, transient=True, overwrite_frames=True)
    live.start(refresh=True)
    start = len(console_output(console))
    live.update(Text("short\nx"), refresh=True)
    refresh = console_output(console)[start:]
    assert refresh.startswith(BEGIN_SYNCHRONIZED_OUTPUT)
    assert refresh.endswith(END_SYNCHRONIZED_OUTPUT)
    assert "\x1b[2K" not in refresh and "\x1b[2J" not in refresh
    assert screen_text(console, 20) == "short\nx"

    live.update(Text("next\nsecond\nthird\nfourth"), refresh=True)
    assert screen_text(console, 20) == "next\nsecond\nthird\nfourth"
    live.update(Text(""), refresh=True)
    assert screen_text(console, 20) == ""
    live.update(Text("restored"), refresh=True)
    assert screen_text(console, 20) == "restored"
    live.stop()
    assert screen_text(console, 20) == ""


def test_resize_clears_once_inside_synchronized_update() -> None:
    console = string_console(40, height=12)
    live = ResizeAwareLive(Text("long line\nsecond\nthird"), console=console, auto_refresh=False,
                           transient=True, overwrite_frames=True)
    live.start(refresh=True)
    start = len(console_output(console))
    console.size = (20, 8)
    live.update(Text("resized"), refresh=True)
    refresh = console_output(console)[start:]
    assert refresh.startswith(BEGIN_SYNCHRONIZED_OUTPUT + "\x1b[2J\x1b[H")
    assert refresh.endswith(END_SYNCHRONIZED_OUTPUT)
    assert screen_text(console, 20) == "resized"
    live.refresh()
    assert console_output(console).count("\x1b[2J") == 1
    live.stop()


def test_padding_counts_terminal_cells_for_wide_and_combining_characters() -> None:
    console = string_console(8, height=10)
    renderer = OverwriteLiveRender(Text("界e\u0301", style="red"))
    segments = list(console.render(renderer))
    assert Segment.get_line_length(segments) == 8
    assert "".join(part.text for part in segments) == "界e\u0301" + " " * 5


@pytest.mark.parametrize("overflow", ["crop", "ellipsis"])
def test_overwrite_preserves_rich_vertical_overflow(overflow: str) -> None:
    console = string_console(20, height=4)
    live = ResizeAwareLive(Text("\n".join(f"row {i}" for i in range(10))), console=console,
                           auto_refresh=False, transient=True, overwrite_frames=True, vertical_overflow=overflow)
    live.start(refresh=True)
    shown = screen_text(console, 20).splitlines()
    assert len(shown) == 4
    assert shown[:3] == ["row 0", "row 1", "row 2"]
    assert shown[-1].strip() == ("row 3" if overflow == "crop" else "...")
    live.update(Text("shorter"), refresh=True)
    assert screen_text(console, 20) == "shorter"
    live.stop()


def test_noninteractive_output_has_no_synchronization_or_padding() -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=20)
    with synchronize_output(console, enabled=True):
        console.print(OverwriteLiveRender(Text("plain")))
    output = console_output(console)
    assert output.strip() == "plain" and "\x1b" not in output
    assert " " * 5 not in output


def test_synchronization_ends_on_render_error() -> None:
    console = string_console()
    failure = ValueError("render failed")
    with pytest.raises(ValueError) as caught, synchronize_output(console, enabled=True):
        raise failure
    assert caught.value is failure
    assert console_output(console) == BEGIN_SYNCHRONIZED_OUTPUT + END_SYNCHRONIZED_OUTPUT


def test_sync_cleanup_failure_preserves_original_write_error() -> None:
    failure = OSError("terminal lost")

    class BrokenStream(io.StringIO):
        def write(self, text: str) -> int:
            raise failure

    console = Console(file=BrokenStream(), force_terminal=True)
    with pytest.raises(OSError) as caught, synchronize_output(console, enabled=True):
        pytest.fail("the first terminal write must fail")
    assert caught.value is failure
    assert "Resetting synchronized terminal output also failed" in " ".join(failure.__notes__)
