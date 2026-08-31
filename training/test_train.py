# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the training CLI (`training/train.py`): argv parsing, the exit codes 0 / 1 / 130 with a monkeypatched
`train`, the stop request set by the first Ctrl-C / SIGTERM (the second aborting), console logging. The run itself is
tested in `test_run.py`."""

import logging
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.log import ProgressStreamHandler
from training import train as train_module
from training.backend import Backend
from training.golden import write_tiny_yaml
from training.logger import TrainingReport
from training.settings import Settings
from training.train import StopRequest, main, stop_on_interrupt


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
    """The `training` logger without the handlers `configure_console_logging` adds (removed again afterwards, so a
    handler bound to a captured stderr never outlives its test). Autouse: every `main()` call configures one."""
    training_logger = logging.getLogger(train_module.TRAINING_LOGGER_NAME)
    before = list(training_logger.handlers)
    level = training_logger.level
    yield training_logger
    for handler in training_logger.handlers:
        if handler not in before:
            training_logger.removeHandler(handler)
            handler.close()
    training_logger.setLevel(level)


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


def test_configure_console_logging_routes_training_records_to_stderr(
    capsys: pytest.CaptureFixture[str], detached_training_handlers: logging.Logger
) -> None:
    """One stderr handler on the `training` logger (idempotent) at INFO, so `RunLogger`'s `training.logger` records
    reach the terminal in the formatted line format of the data-prep CLI."""
    training_logger = train_module.configure_console_logging()
    train_module.configure_console_logging()
    assert training_logger is detached_training_handlers and training_logger.level == logging.INFO
    handlers = [h for h in training_logger.handlers if isinstance(h, ProgressStreamHandler)]
    assert len(handlers) == 1
    logging.getLogger("training.logger").info("Total training steps: 20 (2 micro-batches each)")
    err = capsys.readouterr().err
    assert err.rstrip().endswith("INFO training.logger: Total training steps: 20 (2 micro-batches each)")
