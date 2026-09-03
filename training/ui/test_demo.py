# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the scripted demo: on a StringIO console, and end to end in a pseudo-terminal (python -m
training.ui.dashboard), where the screen afterwards must show only the kept lines and the final summary.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import pytest

from training.ui.demo import demo, main
from training.ui.testing import BOX_CHARACTERS
from ui.testing import console_output, screen_of, screen_text, string_console, strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[2]
# the fallback captures nothing: without the filter pytest would list the demo's deliberate `warnings.warn` in its summary
_IGNORE_THE_DEMO_WARNING = pytest.mark.filterwarnings("ignore:step 24")


def test_demo_on_a_string_console_runs_the_whole_script(tmp_path: Path) -> None:
    console = string_console(140, height=45)
    log_file = tmp_path / "train.log"
    demo(0.05, enabled=True, log_file=log_file, console=console)
    shown = screen_text(console, 140)
    assert not any(character in shown for character in BOX_CHARACTERS), shown
    assert shown.count("overall") == 1 and "50/50" in shown and "✓ instruct" in shown
    assert shown.count("Training finished after 50 steps") == 1 and shown.count("a bare stderr write (kept)") == 1
    assert shown.count("UserWarning: step 24: a warnings.warn (kept) (") == 1, "one kept line, the message before the location"
    assert "warnings.warn(f" not in shown, "the source line of the default warning format is not a second kept line"
    assert "stray print" not in shown and "sample log record" not in shown
    log_text = log_file.read_text()
    assert "a stray print" in log_text and "sample log record" in log_text and "Training finished" in log_text
    assert "╭─ log" in strip_ansi(console_output(console)), "the live display did run"


@_IGNORE_THE_DEMO_WARNING
def test_demo_falls_back_to_plain_lines_when_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    demo(0.02, enabled=False)
    out = capsys.readouterr().out
    assert "step 5/50 | stage 0 pretrain" in out and "event: exported HuggingFace model" in out and "╭" not in out


@_IGNORE_THE_DEMO_WARNING
def test_main_parses_the_seconds_and_returns_the_exit_code(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAINING_DASHBOARD", "0")
    assert main(["0.02"]) == 0
    assert "step 50/50" in capsys.readouterr().out


# --- end to end in a pseudo-terminal -----------------------------------------------------------------------------------------


def _run_demo_in_pty(*, width: int, height: int, seconds: float, interrupt_after: float | None = None, timeout: float = 60.0) -> tuple[int, str]:
    """
    Run python -m training.ui.dashboard <seconds> on a pseudo-terminal of the given size; the exit code and
    everything it wrote. interrupt_after sends SIGINT that many seconds after the first dashboard frame.
    """

    pid, fd = pty.fork()
    if pid == 0:  # child: the pty is its controlling terminal (stdin/stdout/stderr)
        fcntl.ioctl(1, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        os.chdir(REPO_ROOT)
        env = {key: value for key, value in os.environ.items() if key != "TRAINING_DASHBOARD"}
        env["TERM"] = "xterm-256color"
        os.execve(sys.executable, [sys.executable, "-m", "training.ui.dashboard", str(seconds)], env)
    output = bytearray()
    deadline = time.monotonic() + timeout
    interrupt_at: float | None = None
    interrupted = False  # exactly one SIGINT (as `training/test_train.py`): a second one would end the demo with a traceback
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
            if interrupt_after is not None and not interrupted and interrupt_at is None and b"overall" in output:
                interrupt_at = time.monotonic() + interrupt_after  # the first frame is up
        if interrupt_at is not None and time.monotonic() > interrupt_at:
            os.kill(pid, signal.SIGINT)
            interrupt_at, interrupted = None, True
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            break
    _, status = os.waitpid(pid, 0)
    os.close(fd)
    return os.waitstatus_to_exitcode(status), bytes(output).decode("utf-8", "replace")


def _assert_clean_terminal(text: str, width: int) -> str:
    """
    The screen after the run: no panel remnants, the bars once (the static summary), the cursor shown again.
    """

    plain = strip_ansi(text)
    assert "╭─ log" in plain and "╭─ events" in plain and plain.count("overall") > 1, "the live dashboard did run"
    shown = screen_of(text, width)
    assert not any(character in shown for character in BOX_CHARACTERS), shown
    assert shown.count("overall") == 1 and shown.count("grad norm") == 1, shown
    assert "sample log record" not in shown and "stray print" not in shown, shown
    assert text.rfind("\x1b[?25h") > text.rfind("\x1b[?25l"), "the cursor is visible again"
    return shown


@pytest.mark.slow
def test_demo_in_a_pseudo_terminal_leaves_only_the_kept_lines_and_the_summary() -> None:
    code, text = _run_demo_in_pty(width=140, height=45, seconds=2.0)
    assert code == 0, text[-3000:]
    shown = _assert_clean_terminal(text, 140)
    for kept in (
        "Total training steps: 50 (4 micro-batches each)",
        "step 18: a bare stderr write (kept)",
        "UserWarning: step 24: a warnings.warn (kept)",
        "WARNING demo_third_party_library: step 33: a third-party logger with its own stderr handler (kept)",
        "WARNING training.ui.dashboard: step 40: an example warning (kept in the scrollback)",
        "Training finished after 50 steps",
    ):
        assert shown.count(kept) == 1, (kept, shown)
    lines = [line for line in shown.splitlines() if line.strip()]
    assert lines[0].endswith("Total training steps: 50 (4 micro-batches each)"), shown
    assert lines[-1].endswith("exported HuggingFace model to outputs/demo/hf_export"), shown
    assert "✓ pretrain" in shown and "✓ instruct" in shown and "50/50" in shown and "step 50" in shown and "finished" in shown, shown
    assert shown.index("Training finished after 50 steps") < shown.index("overall")


@pytest.mark.slow
def test_sigint_in_a_pseudo_terminal_leaves_a_clean_screen() -> None:
    code, text = _run_demo_in_pty(width=140, height=45, seconds=8.0, interrupt_after=0.7)
    assert code == 130, text[-3000:]
    shown = _assert_clean_terminal(text, 140)
    assert "Total training steps: 50" in shown and "▶ pretrain" in shown and "Training finished" not in shown, shown
