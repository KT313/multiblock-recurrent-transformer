# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the rich dashboard (multiple live bars + log panel)."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from collections.abc import Iterator

import pytest
from rich.console import Console

from data_preparation.lib.ui.dashboard import Dashboard, Task, active_dashboard, progress
from data_preparation.lib.log import ProgressStreamHandler, configure_logging
from data_preparation.lib.progress import NoProgress


@pytest.fixture
def dashboard() -> Iterator[Dashboard]:
    """An enabled dashboard rendering into a StringIO console (no real terminal needed)."""
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    with Dashboard(enabled=True, console=console, log_lines=3, refresh_per_second=50) as board:
        yield board


def test_task_counts_and_renders(dashboard: Dashboard) -> None:
    bar = dashboard.task("src: download", total=10, unit="row")
    assert isinstance(bar, Task)
    bar.update(3)
    bar.update(1)
    bar.set_postfix({"file": "a.parquet"}, MB=7)
    assert bar.n == 4
    text = dashboard.render_text()
    assert "src: download" in text
    assert "4/10" in text
    assert "file=a.parquet, MB=7" in text


def test_several_tasks_each_have_a_line(dashboard: Dashboard) -> None:
    bars = [dashboard.task(f"source_{i}: download", total=5) for i in range(3)]
    for i, bar in enumerate(bars):
        bar.update(i + 1)
    text = dashboard.render_text()
    for i in range(3):
        assert f"source_{i}: download" in text
        assert f"{i + 1}/5" in text
    assert len(dashboard.tasks) == 3


def test_leave_false_removes_line_and_leave_true_keeps_it(dashboard: Dashboard) -> None:
    gone = dashboard.task("gone", total=1, leave=False)
    kept = dashboard.task("kept", total=1, leave=True)
    gone.update(1)
    kept.update(1)
    gone.close()
    kept.close()
    kept.close()  # idempotent
    text = dashboard.render_text()
    assert "gone" not in text
    assert "kept" in text
    assert [t.description for t in dashboard.tasks] == ["kept"]


def test_overshoot_past_total_renders(dashboard: Dashboard) -> None:
    bar = dashboard.task("src: download", total=11)
    bar.update(1000)
    assert "1000/11" in dashboard.render_text()


def test_indeterminate_task_without_total(dashboard: Dashboard) -> None:
    bar = dashboard.task("counting", total=None, unit="row")
    bar.update(42)
    assert "counting" in dashboard.render_text()


def test_iteration_updates_and_closes(dashboard: Dashboard) -> None:
    bar = dashboard.task("iter", total=3, leave=False, iterable=[1, 2, 3])
    assert list(bar) == [1, 2, 3]
    assert isinstance(bar, Task) and bar.n == 3
    assert dashboard.tasks == []


def test_set_description(dashboard: Dashboard) -> None:
    bar = dashboard.task("before", total=1)
    bar.set_description("after")
    text = dashboard.render_text()
    assert "after" in text and "before" not in text


def test_log_panel_keeps_last_lines(dashboard: Dashboard) -> None:
    for i in range(5):
        dashboard.write(f"line {i}")
    assert dashboard.lines() == ["line 2", "line 3", "line 4"]
    text = dashboard.render_text()
    assert "line 4" in text and "line 0" not in text


def test_log_handler_routes_records_into_panel(dashboard: Dashboard) -> None:
    logger = logging.getLogger("data_preparation.test_dashboard")
    handler = dashboard.log_handler()
    logger.addHandler(handler)
    try:
        logger.warning("hello %d", 7)
    finally:
        logger.removeHandler(handler)
    (line,) = dashboard.lines()
    assert "WARNING data_preparation.test_dashboard: hello 7" in line


def test_markup_in_log_lines_is_not_interpreted(dashboard: Dashboard) -> None:
    dashboard.write("path [bold]x[/bold] and [red]")
    assert "[bold]x[/bold]" in dashboard.render_text()


def test_disabled_dashboard_is_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    with Dashboard(enabled=False, stream=stream) as board:
        assert isinstance(board.task("x", total=1), NoProgress)
        handler = board.log_handler()
        record = logging.LogRecord("data_preparation.x", logging.INFO, __file__, 1, "plain %s", ("msg",), None)
        handler.emit(record)
    assert "INFO data_preparation.x: plain msg" in stream.getvalue()
    assert board.lines() == []


def test_enabled_follows_env_and_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PREP_PROGRESS", "0")
    assert Dashboard(stream=io.StringIO()).enabled is False
    monkeypatch.setenv("DATA_PREP_PROGRESS", "1")
    assert Dashboard(stream=io.StringIO()).enabled is False  # StringIO is not a TTY


def test_progress_uses_active_dashboard_else_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PREP_PROGRESS", "0")
    assert active_dashboard() is None
    assert isinstance(progress(total=1, desc="x"), NoProgress)
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    with Dashboard(enabled=True, console=console) as board:
        assert active_dashboard() is board
        bar = progress(total=2, desc="inside", unit="step", leave=False)
        assert isinstance(bar, Task)
        assert "inside" in board.render_text()
    assert active_dashboard() is None


def test_nested_with_reuses_active_dashboard() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    outer = Dashboard(enabled=True, console=console)
    with outer as a:
        with Dashboard(enabled=True, console=console) as b, a as c:
            assert a is b is c is outer
            assert outer.is_active
        assert outer.is_active
    assert active_dashboard() is None


def test_updates_from_threads(dashboard: Dashboard) -> None:
    bars = [dashboard.task(f"worker {i}", total=200) for i in range(4)]

    def work(bar_index: int) -> None:
        for _ in range(200):
            bars[bar_index].update(1)
            bars[bar_index].set_postfix(i=bar_index)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(t.completed == 200 for t in dashboard.tasks)
    assert dashboard.render_text().count("200/200") == 4


def test_multi_line_records_become_one_panel_line_each(dashboard: Dashboard) -> None:
    dashboard.write("Traceback:\n  File x\nValueError: boom")
    assert dashboard.lines() == ["Traceback:", "  File x", "ValueError: boom"]
    dashboard.write("")
    assert dashboard.lines()[-1] == ""


def test_warnings_are_also_printed_into_the_scrollback(dashboard: Dashboard) -> None:
    logger = logging.getLogger("data_preparation.test_dashboard_keep")
    logger.setLevel(logging.INFO)
    handler = dashboard.log_handler()
    logger.addHandler(handler)
    try:
        logger.info("quiet")
        logger.warning("loud")
    finally:
        logger.removeHandler(handler)
    console_output = dashboard._console.file.getvalue()  # type: ignore[attr-defined]  # StringIO console in the fixture
    scrollback = console_output.split("╭─")[0]  # what was printed before the first frame of the panel
    assert "WARNING data_preparation.test_dashboard_keep: loud" in scrollback and "quiet" not in scrollback
    assert [line.split(": ")[-1] for line in dashboard.lines()] == ["quiet", "loud"]


def test_attach_swaps_the_stream_handler_and_writes_the_log_file(tmp_path: Path) -> None:
    logger = logging.getLogger("data_preparation.test_dashboard_attach")
    logger.propagate = False
    stream_handler = ProgressStreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    log_file = tmp_path / "logs" / "build.log"
    with Dashboard(enabled=True, console=console) as board, board.attach(logger, log_file=log_file):
        assert stream_handler not in logger.handlers and len(logger.handlers) == 2
        logger.info("inside %d", 1)
    assert logger.handlers == [stream_handler], "the plain handler is restored, the dashboard handlers removed"
    assert "inside 1" in log_file.read_text() and board.lines()[-1].endswith("inside 1")
    assert stream_handler.stream.getvalue() == ""
    logger.removeHandler(stream_handler)


def test_attach_keeps_the_root_logger_usable_after_configure_logging(tmp_path: Path) -> None:
    root = configure_logging()
    before = list(root.handlers)
    with Dashboard(enabled=False, stream=io.StringIO()) as board, board.attach(root):
        assert all(not isinstance(h, ProgressStreamHandler) for h in root.handlers)
    assert root.handlers == before


def test_nested_delegated_block_keeps_the_outer_display_alive() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    outer = Dashboard(enabled=True, console=console)
    with outer:
        with Dashboard(enabled=True, console=console):
            pass
        assert outer.is_active and outer._live is not None, "the inner block must not stop the outer display"
    assert active_dashboard() is None and outer._live is None
