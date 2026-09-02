# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the training CLI (`training/train.py`): argv parsing, the exit codes 0 / 1 / 130 with a monkeypatched
`train`, the stop request set by the first Ctrl-C / SIGTERM (the second aborting), console logging, and the tiny run
end to end in a pseudo-terminal with the live dashboard (marked slow): a full run and one interrupted by SIGINT, both
leaving a clean screen. The run itself is tested in `test_run.py`."""

import fcntl
import logging
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import TRAIN_LOCK_NAME, Holder, RunLocked
from data_preparation.lib.log import ProgressStreamHandler
from training import train as train_module
from training.backend.base import Backend
from training.checkpoint import checkpoint_dir
from training.testing.golden import write_tiny_yaml
from training.logger import TrainingReport
from training.settings import Settings
from training.train import StopRequest, main, stop_on_interrupt
from training.ui.common import TRAIN_LOG_NAME, TRAINING_LOGGER_NAME
from training.ui.testing import BOX_CHARACTERS
from ui.testing import screen_of, strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[1]


def _report(out_dir: Path, **overrides: Any) -> TrainingReport:
    values: dict[str, Any] = dict(
        run_directory=out_dir,
        steps_completed=3,
        final_step=3,
        resumed_from=None,
        setup_seconds=1.0,
        train_seconds=2.0,
        last_loss=1.5,
        last_validation={},
        checkpoints_written=[],
        export_dir=None,
    )
    return TrainingReport(**(values | overrides))


@pytest.fixture(autouse=True)
def detached_training_handlers() -> Iterator[logging.Logger]:
    """The `training` logger (yielded) and the `data_preparation` logger without the handlers
    `configure_console_logging` adds to either of them — removed again afterwards, so a handler bound to a captured
    stderr never outlives its test (later tests of other modules would log into a closed stream). Autouse: every
    `main()` call configures both hierarchies."""
    training_logger = logging.getLogger(TRAINING_LOGGER_NAME)
    data_logger = logging.getLogger("data_preparation")
    before = {logger: (list(logger.handlers), logger.level) for logger in (training_logger, data_logger)}
    yield training_logger
    for logger, (handlers_before, level) in before.items():
        for handler in list(logger.handlers):
            if handler not in handlers_before:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(level)


@pytest.fixture
def yaml_path(tmp_path: Path, tiny_dataset_dir: Path) -> Path:
    return write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")


class FakeTrain:
    """Stands in for `training.run.train` in the CLI: records its arguments, returns `report` or raises `error`."""

    def __init__(self, report: TrainingReport | None = None, error: BaseException | None = None) -> None:
        self.report = report
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        settings: Settings,
        *,
        backend: Backend | None = None,
        should_stop: StopRequest | None = None,
        started_at: float | None = None,
    ) -> TrainingReport:
        self.calls.append({"settings": settings, "backend": backend, "should_stop": should_stop, "started_at": started_at})
        if self.error is not None:
            raise self.error
        assert self.report is not None
        return self.report


def test_main_parses_argv_trains_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, yaml_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], detached_training_handlers: logging.Logger
) -> None:
    """`--config` plus overrides reach `train()` as settings, together with the stop request and the start time; the
    report's summary is printed to stdout; exit code 0; the CLI configured the `training` console handler."""
    report = _report(tmp_path / "out")
    fake_train = FakeTrain(report)
    monkeypatch.setattr(train_module, "train", fake_train)
    before = time.time()
    assert main(["--config", str(yaml_path), "--seed", "5"]) == 0
    (call,) = fake_train.calls
    assert call["settings"].seed == 5 and call["settings"].out_dir == str(tmp_path / "out")
    assert call["backend"] is None  # the CLI lets `train()` create the backend from the settings
    assert isinstance(call["should_stop"], StopRequest) and call["should_stop"]() is False
    assert before <= call["started_at"] <= time.time()
    assert capsys.readouterr().out.strip() == report.summary()
    assert any(isinstance(h, ProgressStreamHandler) for h in detached_training_handlers.handlers)


def test_main_returns_3_while_another_training_run_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch, yaml_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    holder = Holder("training", 4242, "host", "2026-09-03T00:00:00+00:00")
    locked = RunLocked(tmp_path / "out" / TRAIN_LOCK_NAME, "training", holder)  # what `train()` raises off its lock
    monkeypatch.setattr(train_module, "train", FakeTrain(error=locked))
    assert main(["--config", str(yaml_path)]) == 3
    err = capsys.readouterr().err
    assert "training expects one run at a time on this system; one is already running (started " in err and "kill -INT 4242" in err


def test_main_returns_130_for_a_stopped_run(monkeypatch: pytest.MonkeyPatch, yaml_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _report(tmp_path / "out", stopped=True)
    monkeypatch.setattr(train_module, "train", FakeTrain(report))
    assert main(["--config", str(yaml_path)]) == 130
    assert capsys.readouterr().out.strip() == report.summary()  # the summary is printed for a stopped run too


@pytest.mark.parametrize("error", [BuildAborted("build cancelled"), KeyboardInterrupt()])
def test_main_returns_130_when_interrupted(
    monkeypatch: pytest.MonkeyPatch, yaml_path: Path, error: BaseException, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `BuildAborted` of the in-process dataset build and a `KeyboardInterrupt` (the second Ctrl-C) both end the
    CLI with 130 and one warning; nothing is printed."""
    monkeypatch.setattr(train_module, "train", FakeTrain(error=error))
    with caplog.at_level(logging.WARNING, logger="training"):
        assert main(["--config", str(yaml_path)]) == 130
    assert "training interrupted" in caplog.text and capsys.readouterr().out == ""


def test_main_returns_1_and_logs_the_traceback_on_failure(
    monkeypatch: pytest.MonkeyPatch, yaml_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train_module, "train", FakeTrain(error=RuntimeError("Loss is nan at step 3. Terminating.")))
    with caplog.at_level(logging.ERROR, logger="training"):
        assert main(["--config", str(yaml_path)]) == 1
    assert "training failed" in caplog.text
    assert "RuntimeError: Loss is nan at step 3. Terminating." in caplog.text  # the traceback is logged
    assert capsys.readouterr().out == ""


def test_main_without_a_config_exits_through_the_parser(capsys: pytest.CaptureFixture[str]) -> None:
    """Settings errors are the parser's (`SystemExit` 2, usage on stderr), not exit code 1."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "dataset_config" in capsys.readouterr().err


# --- the stop request --------------------------------------------------------------------------------------------------


def test_stop_request_is_a_stop_check() -> None:
    request = StopRequest()
    assert request() is False
    request.request_stop()
    assert request() is True
    request.request_stop()  # idempotent
    assert request() is True


def test_first_interrupt_sets_the_stop_request_and_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """SIGINT (Ctrl-C) once inside `stop_on_interrupt`: the request is set, a `keep` warning names the signal, the
    default handlers are back (a second SIGINT raises `KeyboardInterrupt`); on exit the previous handlers return."""
    previous_int, previous_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with caplog.at_level(logging.WARNING, logger="training"), stop_on_interrupt() as should_stop:
        assert should_stop() is False
        assert signal.getsignal(signal.SIGINT) is not previous_int
        signal.raise_signal(signal.SIGINT)
        assert should_stop() is True
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is previous_int and signal.getsignal(signal.SIGTERM) is previous_term
    (record,) = [r for r in caplog.records if r.name == "training.train"]
    assert record.getMessage().startswith("SIGINT received: stopping after this step, saving a checkpoint")
    assert getattr(record, "keep", False) is True


def test_sigterm_sets_the_stop_request_too() -> None:
    previous_term = signal.getsignal(signal.SIGTERM)
    with stop_on_interrupt() as should_stop:
        signal.raise_signal(signal.SIGTERM)  # would kill the process under the default handler
        assert should_stop() is True
    assert signal.getsignal(signal.SIGTERM) is previous_term


def test_stop_on_interrupt_outside_the_main_thread_yields_an_unarmed_request() -> None:
    """Signal handlers can only be installed from the main thread; elsewhere the request exists but no signal sets it."""
    previous_int = signal.getsignal(signal.SIGINT)
    seen: list[bool] = []

    def in_thread() -> None:
        with stop_on_interrupt() as should_stop:
            seen.append(should_stop())
            seen.append(signal.getsignal(signal.SIGINT) is previous_int)

    thread = threading.Thread(target=in_thread)
    thread.start()
    thread.join()
    assert seen == [False, True]


def test_main_maps_a_stop_during_training_to_130(monkeypatch: pytest.MonkeyPatch, yaml_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The whole path: Ctrl-C while `train()` runs sets the request the CLI passed in; `train()` returns a stopped
    report; the CLI prints it and exits 130; the signal handlers are restored afterwards."""
    previous_int = signal.getsignal(signal.SIGINT)

    def train_until_interrupted(settings: Settings, *, should_stop: StopRequest, started_at: float, **_: Any) -> TrainingReport:
        signal.raise_signal(signal.SIGINT)
        assert should_stop() is True
        return _report(tmp_path / "out", stopped=True, final_step=7, steps_completed=7)

    monkeypatch.setattr(train_module, "train", train_until_interrupted)
    assert main(["--config", str(yaml_path)]) == 130
    assert "stopped on request after step 7" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGINT) is previous_int


# --- console logging ---------------------------------------------------------------------------------------------------


@pytest.fixture
def detached_data_preparation_handlers() -> Iterator[logging.Logger]:
    """The `data_preparation` logger with no handler for the test (an earlier `configure_logging` of the session may
    have left one bound to a since-closed capture stream); its handlers and level are put back afterwards."""
    data_logger = logging.getLogger("data_preparation")
    before = list(data_logger.handlers)
    level = data_logger.level
    for handler in before:
        data_logger.removeHandler(handler)
    yield data_logger
    for handler in list(data_logger.handlers):
        data_logger.removeHandler(handler)
        handler.close()
    for handler in before:
        data_logger.addHandler(handler)
    data_logger.setLevel(level)


def test_configure_console_logging_routes_training_and_data_preparation_records_to_stderr(
    capsys: pytest.CaptureFixture[str],
    detached_training_handlers: logging.Logger,
    detached_data_preparation_handlers: logging.Logger,
) -> None:
    """One stderr handler each on the `training` and the `data_preparation` logger (idempotent) at INFO, so
    `RunLogger`'s `training.logger` records and the dataset resolver's `data_preparation.*` records both reach the
    terminal in the line format of the data-prep CLI (the resolver itself configures nothing)."""
    training_logger = train_module.configure_console_logging()
    train_module.configure_console_logging()
    assert training_logger is detached_training_handlers and training_logger.level == logging.INFO
    for configured in (training_logger, detached_data_preparation_handlers):
        handlers = [h for h in configured.handlers if isinstance(h, ProgressStreamHandler)]
        assert len(handlers) == 1 and configured.level == logging.INFO
    logging.getLogger("training.logger").info("Total training steps: 20 (2 micro-batches each)")
    logging.getLogger("data_preparation.training.data.dataset_resolver").info("source a: 40 processed rows, all training")
    lines = capsys.readouterr().err.rstrip().splitlines()
    assert lines[-2].endswith("INFO training.logger: Total training steps: 20 (2 micro-batches each)")
    assert lines[-1].endswith("INFO data_preparation.training.data.dataset_resolver: source a: 40 processed rows, all training")


# --- end to end in a pseudo-terminal: the live dashboard ---------------------------------------------------------------


def _run_cli_in_pty(
    arguments: list[str], *, width: int, height: int, interrupt_after: float | None = None, timeout: float = 240.0
) -> tuple[int, str]:
    """Run `python training/train.py <arguments>` on the CPU on a pseudo-terminal of the given size; the exit code
    and everything it wrote. `interrupt_after` sends SIGINT that many seconds after the first dashboard frame. (The
    pty runner of `training/ui/test_demo.py`, with the command line and the CPU pin of a training run.)"""
    pid, fd = pty.fork()
    if pid == 0:  # child: the pty is its controlling terminal (stdin/stdout/stderr)
        fcntl.ioctl(1, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        os.chdir(REPO_ROOT)
        env = {key: value for key, value in os.environ.items() if key != "TRAINING_DASHBOARD"}
        env["TERM"] = "xterm-256color"
        env["CUDA_VISIBLE_DEVICES"] = ""  # the CPU whatever the machine has; the assertions are about the terminal
        os.execve(sys.executable, [sys.executable, "training/train.py", *arguments], env)
    output = bytearray()
    deadline = time.monotonic() + timeout
    interrupt_at: float | None = None
    interrupted = False  # exactly one SIGINT: a second one would hit the default handler the first one put back
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


def _tiny_cli_arguments(tiny_dataset_dir: Path, out_dir: Path, *overrides: str) -> list[str]:
    """`config/tiny.yaml` on the prepared tiny dataset, fp32 (bf16 autocast is slow on the CPU), no wandb."""
    return ["--config", "config/tiny.yaml", "--dataset_dir", str(tiny_dataset_dir), "--out_dir", str(out_dir), "--precision", "32", *overrides]


def _assert_clean_terminal(text: str, width: int) -> str:
    """The screen after the run: the live dashboard did run, no panel remnants, the bars once (the dashboard's static
    summary), the cursor shown again; returns the screen text."""
    plain = strip_ansi(text)
    assert "╭─ log" in plain and "╭─ events" in plain and plain.count("overall") > 1, "the live dashboard did run"
    shown = screen_of(text, width)
    assert not any(character in shown for character in BOX_CHARACTERS), shown
    assert shown.count("overall") == 1 and shown.count("grad norm") == 1, shown
    assert text.rfind("\x1b[?25h") > text.rfind("\x1b[?25l"), "the cursor is visible again"
    return shown


@pytest.mark.slow
def test_tiny_run_in_a_pseudo_terminal_leaves_the_kept_lines_and_the_summaries(tiny_dataset_dir: Path, short_tmp_path: Path) -> None:
    """The whole CLI with the live dashboard: exit 0, the checkpoints and `train.log` written; the screen afterwards
    shows the kept header lines and the final line once, the dashboard's static summary once (every stage ticked)
    and then the report's summary, and nothing of the live frame. (`short_tmp_path`: the 140-column screen must show
    the checkpoint path unabridged in the events panel and the summary line.)"""
    out_dir = short_tmp_path / "out"
    code, text = _run_cli_in_pty(_tiny_cli_arguments(tiny_dataset_dir, out_dir), width=140, height=45)
    assert code == 0, text[-3000:]
    shown = _assert_clean_terminal(text, 140)
    for kept in ("Total training steps: 20 (2 micro-batches each)", "Training finished after 20 steps", "Training run in "):
        assert shown.count(kept) == 1, (kept, shown)
    assert "✓ pretrain_a" in shown and "✓ pretrain_b" in shown and "✓ finetune" in shown and "20/20" in shown, shown
    assert "step 20" in shown and "finished" in shown and "validation (step 20)" in shown, shown
    assert "saved checkpoint" in shown and "step-00000020-tiny.pth" in shown, shown
    assert shown.index("Training finished after 20 steps") < shown.index("overall") < shown.index("Training run in ")
    assert "20 optimizer steps completed" in shown and "3 checkpoints written" in shown  # the long path may wrap
    assert sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth")) == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    log_text = (out_dir / TRAIN_LOG_NAME).read_text()
    assert "Total training steps: 20" in log_text and "Training finished after 20 steps" in log_text


@pytest.mark.slow
def test_sigint_in_a_pseudo_terminal_saves_a_checkpoint_and_leaves_a_clean_screen(tiny_dataset_dir: Path, tmp_path: Path) -> None:
    """Ctrl-C once while the live dashboard is up (a longer run: 80 optimizer steps of one micro-batch each): the run
    finishes its step, saves a checkpoint, exits 130; the screen shows the signal's kept warning, the "stopped on
    request" line, the static summary and the report once — no frame remnants."""
    out_dir = tmp_path / "out"
    arguments = _tiny_cli_arguments(tiny_dataset_dir, out_dir, "--world_batch_size", "1", "--micro_batch_size", "1")
    code, text = _run_cli_in_pty(arguments, width=140, height=45, interrupt_after=0.5)
    assert code == 130, text[-3000:]
    shown = _assert_clean_terminal(text, 140)
    assert shown.count("SIGINT received: stopping after this step, saving a checkpoint") == 1, shown
    assert shown.count("Training stopped on request after ") == 1 and "Training finished" not in shown, shown
    assert "rerun with resume: true to continue" in shown and "checkpoints written, last:" in shown, shown
    assert "stopped on request" in shown and "saved checkpoint" in shown, shown
    checkpoints = sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth"))
    assert checkpoints and all(name.startswith("step-000000") for name in checkpoints), checkpoints
    assert checkpoints[-1].endswith("-tiny.pth") and "_end" not in checkpoints[-1], "the last one is the stop checkpoint"
    assert checkpoints[-1][len("step-") : len("step-00000000")].lstrip("0").isdigit(), "named after the stopped step"
    # events are dashboard-only under the live display (the fallback logs them); the records are in train.log
    log_text = (out_dir / TRAIN_LOG_NAME).read_text()
    assert "SIGINT received" in log_text and "Training stopped on request" in log_text
