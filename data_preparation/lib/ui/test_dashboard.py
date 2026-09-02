# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the rich dashboard: one live layout (downloads / builds / log panels), nothing printed around it."""

from __future__ import annotations

import fcntl
import io
import logging
import os
import pty
import re
import select
import struct
import sys
import termios
import threading
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console
from rich.live import Live

from data_preparation.lib.log import ProgressStreamHandler, configure_logging
from data_preparation.lib.progress import NoProgress
from data_preparation.lib.ui.capture import LineSink
from data_preparation.lib.ui.dashboard import (
    Dashboard,
    Task,
    active_dashboard,
    progress,
    set_status,
    suspended,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def lines_of(board: Dashboard) -> list[str]:
    """The log lines the dashboard currently shows (newest last)."""
    with board._lock:
        return list(board._lines)


def kept_of(board: Dashboard) -> list[str]:
    """The kept records not printed yet (they are printed when the display closes)."""
    with board._lock:
        return list(board._kept)


def tasks_of(board: Dashboard) -> list[Task]:
    """The open tasks of every panel (summary tasks included)."""
    with board._lock:
        tasks: list[Task] = []
        for state in board._panels.values():
            if state.summary is not None and not state.summary.closed:
                tasks.append(state.summary)
            tasks.extend(state.active)
        return tasks


def panel_names_of(board: Dashboard) -> list[str]:
    with board._lock:
        return list(board._panels)


def render_text(board: Dashboard, width: int = 120, height: int = 50) -> str:
    """The dashboard's current display as plain text."""
    console = Console(width=width, height=height, force_terminal=False, color_system=None)
    with console.capture() as capture:
        console.print(board)
    return capture.get()


@pytest.fixture
def dashboard() -> Iterator[Dashboard]:
    """An enabled dashboard rendering into a StringIO console (no real terminal needed)."""
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    with Dashboard(enabled=True, console=console, log_lines=3, max_rows=4, refresh_per_second=50) as board:
        yield board


def _console_output(console: Console) -> str:
    file = console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


class _Screen:
    """A minimal terminal emulator (CR, LF, cursor up/down, erase line, SGR ignored) with unbounded scrollback:
    what the dashboard's control codes leave on the screen, as a real terminal would show it."""

    _CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

    def __init__(self, width: int) -> None:
        self.width = width
        self.lines: list[list[str]] = [[]]
        self.row = 0
        self.col = 0

    def _line(self, row: int) -> list[str]:
        while len(self.lines) <= row:
            self.lines.append([])
        return self.lines[row]

    def feed(self, data: str) -> None:
        i = 0
        while i < len(data):
            ch = data[i]
            if ch == "\x1b":
                match = self._CSI.match(data, i)
                if match is None:
                    i += 1
                    continue
                params, command = match.group(1), match.group(2)
                n = int(params) if params.isdigit() else 1
                if command == "A":
                    self.row = max(0, self.row - n)
                elif command == "B":
                    self.row += n
                elif command == "C":
                    self.col += n
                elif command == "D":
                    self.col = max(0, self.col - n)
                elif command == "K":
                    line = self._line(self.row)
                    if params in ("", "0"):
                        del line[self.col :]
                    else:
                        line.clear()
                elif command == "J":
                    del self.lines[self.row + 1 :]
                    del self._line(self.row)[self.col :]
                i = match.end()
                continue
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self.row += 1
                self.col = 0
                self._line(self.row)
            elif ch == "\b":
                self.col = max(0, self.col - 1)
            elif ch >= " ":
                if self.col >= self.width:
                    self.row += 1
                    self.col = 0
                line = self._line(self.row)
                while len(line) <= self.col:
                    line.append(" ")
                line[self.col] = ch
                self.col += 1
            i += 1

    def text(self) -> str:
        return "\n".join("".join(line).rstrip() for line in self.lines).rstrip("\n")


def _screen_text(console: Console, width: int) -> str:
    screen = _Screen(width)
    screen.feed(_console_output(console))
    return screen.text()


def _outside_frames(screen_text: str) -> str:
    """The screen without the panel lines (frame content starts with a box character)."""
    return "\n".join(line for line in screen_text.splitlines() if not line.startswith(("│", "╭", "╰")))


# --- tasks and panels ---------------------------------------------------------------------------------------------------


def test_task_counts_and_renders_in_its_panel(dashboard: Dashboard) -> None:
    bar = dashboard.task("src", total=10, unit="row", panel="downloads")
    assert isinstance(bar, Task)
    bar.update(3)
    bar.update(1)
    bar.set_postfix({"file": "a.parquet"}, MB=7)
    assert bar.n == 4
    text = render_text(dashboard)
    downloads = text[text.index("downloads") : text.index("builds")]
    assert "src" in downloads and "4/10" in downloads and "file=a.parquet, MB=7" in downloads


def test_panels_keep_their_order_and_unknown_panels_appear_on_demand(dashboard: Dashboard) -> None:
    dashboard.task("b", total=1, panel="builds")
    dashboard.task("d", total=1, panel="downloads")
    dashboard.task("t", total=1)
    text = render_text(dashboard)
    assert text.index("─ downloads") < text.index("─ builds") < text.index("─ tasks") < text.index("─ log")
    assert panel_names_of(dashboard) == ["downloads", "builds", "tasks"]


def test_empty_panels_render_idle(dashboard: Dashboard) -> None:
    text = render_text(dashboard)
    assert text.count("idle") == 2 and "(no log output yet)" in text


def test_finished_rows_disappear_and_count_in_the_summary(dashboard: Dashboard) -> None:
    gone = dashboard.task("gone", total=10, panel="downloads")
    kept = dashboard.task("kept", total=10, panel="downloads")
    gone.update(10)
    gone.set_postfix(MB=3)
    kept.update(4)
    gone.close()
    gone.close()  # idempotent
    text = render_text(dashboard)
    assert "gone" not in text and "kept" in text
    assert "14/20 rows · 3 MB" in text, "the summary counts finished and running rows (and the MB they reported)"
    assert [task.description for task in tasks_of(dashboard)] == ["kept"]


def test_summary_task_starts_a_round_and_is_updated_in_place(dashboard: Dashboard) -> None:
    summary = dashboard.task("downloads", total=3, unit="job", panel="downloads", summary=True)
    first = dashboard.task("a", total=5, panel="downloads")
    first.update(5)
    first.close()
    summary.update(1)
    text = render_text(dashboard)
    assert text.count("jobs done") == 1 and "1/3 jobs done · 5/5 rows" in text
    summary.update(2)
    summary.close()
    text = render_text(dashboard)
    assert text.count("jobs done") == 1 and "3/3 jobs done · 5/5 rows" in text, "closed: the round's summary stays"
    dashboard.task("downloads", total=2, unit="job", panel="downloads", summary=True)
    text = render_text(dashboard)
    assert text.count("jobs done") == 1 and "0/2 jobs done" in text and "5/5 rows" not in text, "a new round resets the counts"


def test_rows_are_bounded_per_panel(dashboard: Dashboard) -> None:
    bars = [dashboard.task(f"src_{i}", total=10, panel="downloads") for i in range(6)]
    text = render_text(dashboard)
    assert "src_3" in text and "src_4" not in text and "… and 2 more" in text
    bars[0].close()
    text = render_text(dashboard)
    assert "src_4" in text and "… and 1 more" in text


def test_overshoot_past_total_renders(dashboard: Dashboard) -> None:
    bar = dashboard.task("src", total=11, panel="downloads")
    bar.update(1000)
    assert "1,000/11" in render_text(dashboard)


def test_indeterminate_task_without_total(dashboard: Dashboard) -> None:
    bar = dashboard.task("counting", total=None, unit="row")
    bar.update(42)
    text = render_text(dashboard)
    assert "counting" in text and "42 " in text and "42 rows" in text


def test_updates_after_close_keep_counting_but_the_row_is_gone(dashboard: Dashboard) -> None:
    bar = dashboard.task("gone", total=2)
    assert isinstance(bar, Task)
    bar.close()
    bar.update(1)
    bar.set_postfix(a=1)
    assert bar.n == 1 and tasks_of(dashboard) == [] and "gone" not in render_text(dashboard)


def test_updates_from_threads(dashboard: Dashboard) -> None:
    bars = [dashboard.task(f"worker {i}", total=200, panel="builds") for i in range(4)]

    def work(bar_index: int) -> None:
        for _ in range(200):
            bars[bar_index].update(1)
            bars[bar_index].set_postfix(i=bar_index)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(t.completed == 200 for t in tasks_of(dashboard))
    assert render_text(dashboard).count("200/200") == 4


def test_header_shows_title_status_and_footer_the_log_file(short_tmp_path: Path) -> None:
    """(`short_tmp_path`: the footer shows the log file's path unabridged at width 120.)"""
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    logger = logging.getLogger("data_preparation.test_dashboard_header")
    log_file = short_tmp_path / "build.log"
    with Dashboard(title="prepare tiny", enabled=True, console=console) as board, board.attach(logger, log_file=log_file):
        set_status(round="1/5", step="download")
        board.set_status(step="build")
        first_line, *rest = render_text(board).splitlines()
        assert first_line.startswith("prepare tiny · round 1/5 · step build · 0:00:0")
        assert rest[-1].startswith(f"log: {log_file} · Ctrl-C stops at the next shard")


def test_short_terminal_shrinks_the_log_panel_not_the_task_rows() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    with Dashboard(enabled=True, console=console, log_lines=12, max_rows=8) as board:
        for i in range(20):
            board.write(f"line {i}")
        bars = [board.task(f"src_{i}", total=1, panel="downloads") for i in range(8)]
        tall = render_text(board, width=100, height=60)
        short = render_text(board, width=100, height=24)  # header + footer + 8 rows in panels leave 6 log lines
        assert "line 8" in tall and "line 19" in tall
        assert "line 13" not in short and "line 14" in short and "line 19" in short, "the log panel keeps the newest lines"
        assert all(f"src_{i}" in short for i in range(8))
        for bar in bars:
            bar.close()


# --- log panel and kept records ---------------------------------------------------------------------------------------


def test_log_panel_keeps_last_lines(dashboard: Dashboard) -> None:
    for i in range(5):
        dashboard.write(f"line {i}")
    assert lines_of(dashboard) == ["line 2", "line 3", "line 4"]
    text = render_text(dashboard)
    assert "line 4" in text and "line 0" not in text


def test_every_record_reaches_the_panel_once_while_the_display_is_up(dashboard: Dashboard) -> None:
    logging.getLogger("data_preparation.test_dashboard").warning("hello %d", 7)  # no handler of its own: the root one
    logging.getLogger("some_library").warning("careful")
    first, second = lines_of(dashboard)
    assert "WARNING data_preparation.test_dashboard: hello 7" in first and "WARNING some_library: careful" in second
    assert kept_of(dashboard) == [first, second]


def test_markup_in_log_lines_is_not_interpreted(dashboard: Dashboard) -> None:
    dashboard.write("path [bold]x[/bold] and [red]")
    assert "[bold]x[/bold]" in render_text(dashboard)


def test_multi_line_records_become_one_panel_line_each(dashboard: Dashboard) -> None:
    dashboard.write("Traceback:\n  File x\nValueError: boom")
    assert lines_of(dashboard) == ["Traceback:", "  File x", "ValueError: boom"]
    dashboard.write("")
    assert lines_of(dashboard)[-1] == ""


def test_kept_records_are_printed_once_after_the_display_closed_not_during() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    logger = logging.getLogger("data_preparation.test_dashboard_keep")
    logger.setLevel(logging.INFO)
    table = "a  b  c" + " " * 200 + "end"
    with Dashboard(enabled=True, console=console, refresh_per_second=50) as board, board.attach(logger):
        logger.info("quiet")
        logger.warning("loud")
        logger.info("table:\n%s", table, extra={"keep": True})
        board.task("src", total=1, panel="downloads").update(1)
        time.sleep(0.1)  # a few live frames
        assert [line.split(": ")[-1] for line in lines_of(board)[:2]] == ["quiet", "loud"]
        assert len(kept_of(board)) == 2
        assert "loud" not in _outside_frames(_screen_text(console, 100)), "nothing is printed while the display is up"
    screen = _screen_text(console, 100)
    assert screen.count("WARNING data_preparation.test_dashboard_keep: loud") == 1 and "quiet" not in screen
    assert _console_output(console).count(table) == 1 and screen.count("end") == 1, "kept multi-line records are written unwrapped, once"
    assert "╭" not in screen and "src" not in screen, "the display is transient: no frame is left behind"
    assert screen.index("loud") < screen.index("end")
    assert kept_of(board) == []


def test_disabled_dashboard_is_plain() -> None:
    stream = io.StringIO()
    real_out = sys.stdout
    with Dashboard(enabled=False, stream=stream) as board:
        assert isinstance(board.task("x", total=1), NoProgress)
        handler = board.log_handler()
        record = logging.LogRecord("data_preparation.x", logging.INFO, __file__, 1, "plain %s", ("msg",), None)
        handler.emit(record)
        assert sys.stdout is real_out, "disabled: no capture"
    assert "INFO data_preparation.x: plain msg" in stream.getvalue()
    assert lines_of(board) == []


def test_enabled_follows_env_and_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PREP_PROGRESS", "0")
    assert Dashboard(stream=io.StringIO()).enabled is False
    monkeypatch.setenv("DATA_PREP_PROGRESS", "1")
    assert Dashboard(stream=io.StringIO()).enabled is False  # StringIO is not a TTY


# --- the active dashboard and the progress() drop-in --------------------------------------------------------------------


def test_progress_uses_active_dashboard_else_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_PREP_PROGRESS", "0")
    assert active_dashboard() is None
    assert isinstance(progress(total=1, desc="x", panel="downloads", summary=True), NoProgress)
    set_status(round="none")  # no-op without a dashboard
    with suspended():
        pass
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    with Dashboard(enabled=True, console=console) as board:
        assert active_dashboard() is board
        bar = progress(total=2, desc="inside", unit="step", panel="builds")
        assert isinstance(bar, Task)
        assert "inside" in render_text(board)
    assert active_dashboard() is None


def test_a_second_dashboard_inside_the_block_raises() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    outer = Dashboard(enabled=True, console=console)
    with outer:
        with pytest.raises(RuntimeError, match="already active"), Dashboard(enabled=True, console=console):
            pass
        with pytest.raises(RuntimeError, match="already active"), outer:
            pass
        assert outer.is_active and outer._live is not None, "the refused blocks leave the outer display alone"
    assert active_dashboard() is None and outer._live is None


# --- attach ---------------------------------------------------------------------------------------------------------------


def test_attach_swaps_the_stream_handler_and_writes_the_log_file(tmp_path: Path) -> None:
    logger = logging.getLogger("data_preparation.test_dashboard_attach")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    stream_handler = ProgressStreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    log_file = tmp_path / "logs" / "build.log"
    with Dashboard(enabled=True, console=console) as board, board.attach(logger, log_file=log_file):
        assert stream_handler not in logger.handlers and len(logger.handlers) == 2
        assert board.is_attached("data_preparation.test_dashboard_attach.child") and not board.is_attached("data_preparation")
        logger.info("inside %d", 1)
    assert logger.handlers == [stream_handler], "the plain handler is restored, the dashboard handlers removed"
    assert "inside 1" in log_file.read_text() and lines_of(board)[-1].endswith("inside 1")
    assert stream_handler.stream.getvalue() == ""
    logger.removeHandler(stream_handler)


def test_attach_keeps_the_root_logger_usable_after_configure_logging(tmp_path: Path) -> None:
    root = configure_logging()
    before = list(root.handlers)
    with Dashboard(enabled=False, stream=io.StringIO()) as board, board.attach(root):
        assert all(not isinstance(h, ProgressStreamHandler) for h in root.handlers)
    assert root.handlers == before


def test_attach_restores_handlers_when_the_body_raises() -> None:
    logger = logging.getLogger("data_preparation.test_dashboard_raise")
    logger.propagate = False
    stream_handler = ProgressStreamHandler(io.StringIO())
    logger.addHandler(stream_handler)
    board = Dashboard(enabled=False, stream=io.StringIO())
    with pytest.raises(RuntimeError, match="boom"), board, board.attach(logger):
        assert stream_handler not in logger.handlers
        raise RuntimeError("boom")
    assert logger.handlers == [stream_handler] and active_dashboard() is None
    logger.removeHandler(stream_handler)


def test_disabled_dashboard_with_threads_logs_each_record_once() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("data_preparation.test_dashboard_threads")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(ProgressStreamHandler(io.StringIO()))  # the plain handler `attach` swaps out
    with Dashboard(enabled=False, stream=stream) as board, board.attach(logger):
        threads = [threading.Thread(target=logger.info, args=("record %d", i)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    lines = stream.getvalue().splitlines()
    assert sorted(line.split(": ")[-1] for line in lines) == sorted(f"record {i}" for i in range(8))
    logger.handlers.clear()


def test_records_of_attached_loggers_land_in_the_panel_once() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    logger = logging.getLogger("data_preparation.test_dashboard_once")
    logger.setLevel(logging.INFO)
    with Dashboard(enabled=True, console=console) as board, board.attach(logging.getLogger("data_preparation")):
        logger.info("just once")
        assert [line for line in lines_of(board) if "just once" in line] == lines_of(board)[-1:]


# --- what else could reach the terminal -------------------------------------------------------------------------------------


def test_third_party_console_handlers_are_detached_while_the_display_is_up() -> None:
    library = logging.getLogger("fake_hub_library")  # like huggingface_hub / datasets: a StreamHandler on stderr
    library.setLevel(logging.WARNING)
    plain = logging.StreamHandler(sys.stderr)
    library.addHandler(plain)
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    try:
        with Dashboard(enabled=True, console=console) as board:
            assert library.handlers == [], "the plain handler would print behind the display"
            library.warning("repo card missing")
            assert lines_of(board)[-1].endswith("WARNING fake_hub_library: repo card missing")
            assert kept_of(board)[-1].endswith("repo card missing")
        assert library.handlers == [plain]
    finally:
        library.removeHandler(plain)


def test_stdout_and_stderr_are_captured_while_the_display_is_up() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    real_out, real_err = sys.stdout, sys.stderr
    with Dashboard(enabled=True, console=console) as board:
        streams: tuple[object, object] = (sys.stdout, sys.stderr)  # object: the stubs type them TextIO, the sink is not one
        assert all(isinstance(stream, LineSink) for stream in streams)
        assert not sys.stderr.isatty()
        print("stray print")
        sys.stderr.write("\rbar 10%\rbar 100%\n")
        # what `warnings.showwarning` writes to sys.stderr (pytest records warnings itself, so it is written by hand)
        sys.stderr.write(warnings.formatwarning("careful", UserWarning, "x.py", 1))
        print("partial", end="")  # no newline: flushed when the display closes
        lines = lines_of(board)
        assert any(line.endswith("INFO data_preparation.stdout: stray print") for line in lines)
        assert any(line.endswith("WARNING data_preparation.stderr: bar 100%") for line in lines), "a carriage return discards the line so far"
        assert any("WARNING data_preparation.stderr: x.py:1: UserWarning: careful" in line for line in lines)
        assert kept_of(board) and all("stray print" not in text for text in kept_of(board)), "stdout lines are not kept, stderr lines are"
    assert sys.stdout is real_out and sys.stderr is real_err
    assert lines_of(board)[-1].endswith("INFO data_preparation.stdout: partial")
    screen = _screen_text(console, 100)
    assert "bar 100%" in screen and "careful" in screen and "stray print" not in screen


def test_third_party_bars_are_silenced_while_the_display_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setenv("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    with Dashboard(enabled=True, console=console):
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1" and os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ and os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "0"


def test_suspended_clears_the_display_for_a_prompt_and_brings_it_back(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=80)
    real_out = sys.stdout
    with Dashboard(enabled=True, console=console) as board:
        board.task("src", total=1, panel="downloads")
        live = _live_of(board)
        assert live is not None
        with suspended():
            assert _live_of(board) is None and sys.stdout is real_out, "the terminal belongs to the prompt"
            assert "src" not in _screen_text(console, 80), "the frame is erased while suspended"
        stdout: object = sys.stdout
        restarted = _live_of(board)
        assert restarted is not None and restarted is not live and isinstance(stdout, LineSink)
        assert "src" in render_text(board)


def _live_of(board: Dashboard) -> Live | None:
    return board._live  # through a call: mypy would otherwise keep the narrowing of an earlier assertion


# --- the whole thing: threads, refreshes, scrollback ------------------------------------------------------------------------


def _check_frame(frame: str, log_lines: int) -> None:
    """One rendered frame: the summary row once, every bar inside a panel, the log panel bounded."""
    assert frame.count("jobs done") == 1, frame
    for line in frame.splitlines():
        if "row/s" in line or "jobs done" in line or "rows" in line:
            assert line.startswith("│"), f"bar text outside the panels: {line!r}"
    log_panel = frame[frame.index("─ log") :]
    assert sum(1 for line in log_panel.splitlines() if " INFO " in line or " WARNING " in line) <= log_lines


def test_three_threads_drive_the_panels_and_the_scrollback_is_clean() -> None:
    console = Console(file=io.StringIO(), record=True, force_terminal=True, width=120, height=40)
    logger = logging.getLogger("data_preparation.test_dashboard_scenario")
    logger.setLevel(logging.INFO)
    log_lines = 4
    frames: list[str] = []
    with Dashboard(title="prepare tiny", enabled=True, console=console, log_lines=log_lines, refresh_per_second=100) as board, board.attach(logger):
        summary = progress(total=2, desc="downloads", unit="job", panel="downloads", summary=True)

        def download(name: str) -> None:
            with progress(total=40, desc=name, unit="row", panel="downloads") as bar:
                for i in range(40):
                    bar.update(1)
                    bar.set_postfix({"consumed": i + 1, "file": f"{name}-{i:05d}.parquet", "MB": 1})
                    logger.info("%s: row %d", name, i)
                    time.sleep(0.002)
            logger.info("%s: kept 40 of 40 fetched rows", name)
            summary.update(1)

        def build() -> None:
            with progress(total=30, desc="peso", unit="row", panel="builds") as bar:
                for i in range(30):
                    bar.update(1)
                    bar.set_postfix({"shard": f"{i + 1}/30"})
                    time.sleep(0.002)

        threads = [threading.Thread(target=download, args=("fineweb_edu",)), threading.Thread(target=download, args=("wikipedia",)), threading.Thread(target=build)]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            frame = render_text(board, width=120, height=40)  # a snapshot under the lock, like a refresh
            frames.append(frame)
            _check_frame(frame, log_lines)
            time.sleep(0.005)
        for thread in threads:
            thread.join()
        summary.close()
        final = render_text(board, width=120, height=40)
        _check_frame(final, log_lines)
        panels = final[final.index("─ downloads") : final.index("─ log")]
        assert "peso" not in panels and "fineweb_edu" not in panels, "finished rows are removed"
        assert "2/2 jobs done · 80/80 rows · 2 MB" in final and "30/30 rows" in final
        assert final.splitlines()[0].startswith("prepare tiny")
        assert lines_of(board)[-1].endswith("wikipedia: kept 40 of 40 fetched rows") or lines_of(board)[-1].endswith("fineweb_edu: kept 40 of 40 fetched rows")
        logger.info("dataset status:\nsource  kind\na  b\ndataset complete", extra={"keep": True})
    assert any("consumed=" in frame and "shard=" in frame for frame in frames), "downloads and the build were live at the same time"
    assert len(console.export_text()) > 0
    screen = _screen_text(console, 120)
    assert screen.count("dataset complete") == 1 and screen.count("source  kind") == 1, screen
    assert "╭" not in screen and "jobs done" not in screen and "row/s" not in screen and "consumed=" not in screen, screen


# --- end to end in a pseudo-terminal -----------------------------------------------------------------------------------------

_PTY_CHILD = """
import os, sys, time
sys.path.insert(0, {root!r})
from data_preparation.lib.sources import loaders
original = loaders.LOADERS["synthetic"]
def slow(source, offset, count, **kwargs):
    for row in original(source, offset, count, **kwargs):
        time.sleep({row_delay})
        yield row
loaders.LOADERS["synthetic"] = slow
from data_preparation import prepare
prepare.main(["prepare", "--dataset_config", "config/datasets/tiny.yaml", "--dataset_dir", {dataset_dir!r}])
"""


def _run_in_pty(script: str, *, width: int, height: int, timeout: float = 120.0, terminate_after: float | None = None) -> tuple[int, bytes]:
    """Run ``python -c script`` on a pseudo-terminal of the given size; the exit code and everything it wrote.
    ``terminate_after`` sends SIGTERM that many seconds after the first dashboard frame (a byte-capped run)."""
    pid, fd = pty.fork()
    if pid == 0:  # child: the pty is its controlling terminal (stdin/stdout/stderr)
        fcntl.ioctl(1, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        os.chdir(REPO_ROOT)
        env = {key: value for key, value in os.environ.items() if key != "DATA_PREP_PROGRESS"}
        env["TERM"] = "xterm-256color"
        os.execve(sys.executable, [sys.executable, "-c", script], env)
    output = bytearray()
    deadline = time.monotonic() + timeout
    terminate_at: float | None = None
    while True:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:  # EIO: the child closed its side
                break
            if not chunk:
                break
            output += chunk
            if terminate_after is not None and terminate_at is None and b"downloads" in output:  # the first frame is up
                terminate_at = time.monotonic() + terminate_after
        if terminate_at is not None and time.monotonic() > terminate_at:
            os.kill(pid, 15)
            terminate_at = None
        if time.monotonic() > deadline:
            os.kill(pid, 9)
            break
    _, status = os.waitpid(pid, 0)
    os.close(fd)
    return os.waitstatus_to_exitcode(status), bytes(output)


@pytest.mark.slow
def test_prepare_tiny_in_a_pseudo_terminal_leaves_only_the_kept_lines_and_the_table(short_tmp_path: Path) -> None:
    """(`short_tmp_path`: the final `done: <dataset dir>` line must fit one 140-column screen line.)"""
    dataset_dir = short_tmp_path / "dataset"
    code, raw = _run_in_pty(_PTY_CHILD.format(root=str(REPO_ROOT), dataset_dir=str(dataset_dir), row_delay=0.03), width=140, height=45)
    text = raw.decode("utf-8", "replace")
    assert code == 0, text[-3000:]
    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    assert "╭─ downloads" in plain and "jobs done" in plain and "row/s" in plain, "the live dashboard did run, with download rows"
    assert re.search(r"│ synthetic_(pretrain|instruct) +[━╺╸ ]+ +\d+/\d+ ", plain), "a download row: name, bar, rows/wanted"
    screen = _Screen(140)
    screen.feed(text)
    shown = screen.text()
    assert shown.count("dataset status:") == 1 and shown.count("dataset complete") == 1, shown
    assert "╭" not in shown and "│" not in shown and "jobs done" not in shown and "row/s" not in shown, shown
    lines = [line for line in shown.splitlines() if line.strip()]
    assert lines[0].endswith("dataset status:") and lines[-1].endswith(f"done: {dataset_dir}"), shown
    build_log = (dataset_dir / "build.log").read_text()
    assert "round 1:" in build_log and "synthetic_pretrain: kept 76 of 76 fetched rows" in build_log


@pytest.mark.slow
def test_sigterm_in_a_pseudo_terminal_clears_the_display_and_exits_130(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    script = _PTY_CHILD.format(root=str(REPO_ROOT), dataset_dir=str(dataset_dir), row_delay=0.1)  # ~4 s of downloading
    code, raw = _run_in_pty(script, width=140, height=45, terminate_after=0.5)
    text = raw.decode("utf-8", "replace")
    assert code == 130, text[-3000:]
    screen = _Screen(140)
    screen.feed(text)
    shown = screen.text()
    assert "╭" not in shown and "│" not in shown and "jobs done" not in shown, shown
    assert "prepare interrupted; everything published so far is kept, rerun to resume" in shown, shown
    assert shown.count("interrupted; the running jobs stop at their next shard") == 1, shown
