# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the helpers of the dashboard tests."""

from __future__ import annotations

from training.ui.testing import FakeClock, console_output, metrics, screen_of, screen_text, string_console, strip_ansi


def test_strip_ansi_removes_colours_and_cursor_codes() -> None:
    assert strip_ansi("\x1b[2m╭─\x1b[0m\x1b[2m log \x1b[0m\x1b[?25l\x1b[1A\x1b[2Kx") == "╭─ log x"


def test_fake_clock_advances_by_hand() -> None:
    clock = FakeClock()
    assert clock() == 0.0
    clock.advance(2.5)
    assert clock() == 2.5


def test_string_console_records_and_the_screen_replays_cursor_movement() -> None:
    console = string_console(width=20)
    console.print("first")
    console.file.write("\x1b[1A\x1b[2Ksecond\n")  # cursor up, erase the line, overwrite: what a transient Live does
    assert "first" in console_output(console)
    assert screen_text(console, 20) == "second"
    assert screen_of("a\r\nb\x1b[1A\x1b[Kc", 20) == "ac\nb"


def test_metrics_is_a_step_dict() -> None:
    step = metrics(3, loss=2.0, extra=1.0)
    assert step["loss"] == 2.0 and step["total_tokens"] == 3 * 8_192 and step["extra"] == 1.0
