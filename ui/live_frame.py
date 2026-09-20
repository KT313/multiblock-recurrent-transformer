# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Overwrite live terminal frames without an intermediate erase, inside synchronized output."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console, ConsoleOptions, RenderResult
from rich.control import Control
from rich.live_render import LiveRender
from rich.segment import ControlType, Segment

BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"


@contextmanager
def synchronize_output(console: Console, *, enabled: bool) -> Iterator[None]:
    """Keep the last visible frame until this update ends; unsupported terminals ignore the mode.

    Use the console's output lock to keep another writer on this console outside the markers. Always attempt
    to reset the mode after an error, while preserving the original exception for terminal-loss handling.
    """

    if not enabled or not console.is_interactive:
        yield
        return
    with console._lock:
        stream = console.file
        try:
            stream.write(BEGIN_SYNCHRONIZED_OUTPUT)
            yield
        except BaseException as error:
            try:
                stream.write(END_SYNCHRONIZED_OUTPUT)
                stream.flush()
            except Exception as cleanup_error:
                error.add_note(f"Resetting synchronized terminal output also failed: {cleanup_error!r}")
            raise
        else:
            stream.write(END_SYNCHRONIZED_OUTPUT)
            stream.flush()


class OverwriteLiveRender(LiveRender):
    """Retain Rich's layout/overflow handling, but overwrite and pad rows instead of pre-clearing them."""

    def position_cursor(self) -> Control:
        height = self.last_render_height
        if not height:
            return Control()
        return Control(ControlType.CARRIAGE_RETURN, (ControlType.CURSOR_UP, height - 1)) if height > 1 else Control(ControlType.CARRIAGE_RETURN)

    def reset_height(self) -> None:
        self._shape = None  # after a terminal resize the old rows no longer describe the screen

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        previous_height = self.last_render_height
        segments: list[Segment] = []
        for part in super().__rich_console__(console, options):
            segments.extend([part] if isinstance(part, Segment) else console.render(part, options))
        if not console.is_interactive:
            yield from segments
            return

        # Paint complete rows, including spaces over text that disappeared or became shorter.
        height = self.last_render_height
        rows = list(Segment.split_lines(segments))
        painted_height = max(previous_height, height)
        for index in range(painted_height):
            row = rows[index] if index < len(rows) else []
            yield from Segment.adjust_line_length(row, options.max_width)
            if index + 1 < painted_height:
                yield Segment.line()

        # Removed rows are now blank; leave the cursor at the end of the new frame for Rich's next update.
        if previous_height > height:
            yield Control(ControlType.CARRIAGE_RETURN, (ControlType.CURSOR_UP, previous_height - max(height, 1)))
