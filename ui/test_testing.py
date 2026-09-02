# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the shared helpers of the dashboard tests."""

from __future__ import annotations

from ui.testing import console_output, screen_of, screen_text, string_console, strip_ansi


def test_strip_ansi_removes_colours_and_cursor_codes() -> None:
    assert strip_ansi("\x1b[2m╭─\x1b[0m\x1b[2m log \x1b[0m\x1b[?25l\x1b[1A\x1b[2Kx") == "╭─ log x"


def test_string_console_records_and_the_screen_replays_cursor_movement() -> None:
    console = string_console(width=20)
    console.print("first")
    console.file.write("\x1b[1A\x1b[2Ksecond\n")  # cursor up, erase the line, overwrite: what a transient Live does
    assert "first" in console_output(console)
    assert screen_text(console, 20) == "second"
    assert screen_of("a\r\nb\x1b[1A\x1b[Kc", 20) == "ac\nb"
