# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Deterministic training redraw tests: count actual terminal writes without sleeping."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest

from training.ui.board import TrainingDashboard
from training.ui.testing import metrics
from ui import display
from ui.testing import FakeClock, console_output, string_console


@pytest.fixture(autouse=True)
def disable_timer_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(display, "ResizeAwareLive", partial(display.ResizeAwareLive, auto_refresh=False))


def test_idle_dashboard_writes_nothing_until_ten_second_heartbeat() -> None:
    clock = FakeClock()
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=clock) as board:
        live = board._live
        assert live is not None
        first = console_output(console)
        for _ in range(19):
            clock.advance(0.5)
            live.refresh()
            assert console_output(console) == first
        clock.advance(0.5)
        live.refresh()
        heartbeat = console_output(console)
        assert len(heartbeat) > len(first) and "0:00:10" in heartbeat[len(first):]
        clock.advance(0.5)
        live.refresh()
        assert console_output(console) == heartbeat  # rendering's overall-note mutation does not dirty the state


def test_burst_updates_are_coalesced_and_final_state_is_printed_immediately() -> None:
    clock = FakeClock()
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=clock) as board:
        live = board._live
        assert live is not None
        first = console_output(console)
        for step in range(1, 5):
            clock.advance(0.1)
            board.update_step(step, 0, None, metrics(step))
            live.refresh()
            assert console_output(console) == first
        clock.advance(0.1)
        live.refresh()
        updated = console_output(console)
        assert "4/10" in updated[len(first):]
        board.set_status("completed")
        live.refresh()
        assert console_output(console) == updated
    assert "completed" in console_output(console)[len(updated):]


@pytest.mark.parametrize("update", [
    lambda board: board.set_status("evaluating"),
    lambda board: board.update_step(1, 0, None, metrics(1)),
    lambda board: board.update_validation(0, {"val_loss": 2.0}),
    lambda board: board.update_micro_batch(1, 4),
    lambda board: board.note_event("checkpoint saved"),
    lambda board: board.write("compiler warning"),
])
def test_visible_updates_trigger_a_frame_but_identical_state_does_not(update: Callable[[TrainingDashboard], None]) -> None:
    clock = FakeClock()
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=clock, show_micro_batches=True) as board:
        live = board._live
        assert live is not None
        before = console_output(console)
        update(board)
        clock.advance(0.5)
        live.refresh()
        after = console_output(console)
        assert len(after) > len(before)
        board.set_status(board._status)
        clock.advance(0.5)
        live.refresh()
        assert console_output(console) == after


def test_hidden_microbatch_updates_do_not_redraw() -> None:
    clock = FakeClock()
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=clock) as board:
        live = board._live
        assert live is not None
        before = console_output(console)
        board.update_micro_batch(1, 4)
        clock.advance(0.5)
        live.refresh()
        assert console_output(console) == before


def test_resize_waits_for_rate_limit_but_not_idle_heartbeat() -> None:
    clock = FakeClock()
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=clock) as board:
        live = board._live
        assert live is not None
        before = console_output(console)
        console.size = (80, 30)
        live.refresh()
        assert console_output(console) == before
        clock.advance(0.5)
        live.refresh()
        assert "\x1b[2J\x1b[H" in console_output(console)[len(before):]


def test_pending_terminal_loss_bypasses_redraw_rate_limit() -> None:
    console = string_console()
    with TrainingDashboard("run", ["train"], [10], 10, console=console, clock=FakeClock()) as board:
        live = board._live
        assert live is not None
        live.terminal_lost_pending = "terminal disconnected"
        live.refresh()
        assert board.headless and not board.enabled
