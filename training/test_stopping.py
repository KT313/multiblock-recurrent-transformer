# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Phase-driven stop regressions, including bounded real gloo completion ordering."""

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from threading import Event
from typing import Any

import pytest
import torch

from data_preparation import load_dataset_config
from training import run as run_module
from training.execution import loop as loop_helpers
from training.execution import checkpoints as checkpoint_helpers
from training.backend.single_device import SingleDeviceBackend
from training.run import train
from training.step import run_one_optimizer_step
from training.evaluation import evaluate
from training.checkpoint import save_training_checkpoint
from training.logger import RunLogger
from training.settings import Settings, parse_settings
from training.stopping import StopController, complete_main_phase
from training.testing.golden import write_tiny_yaml


@pytest.fixture
def settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(write_tiny_yaml(
        tmp_path, tiny_dataset_dir, tmp_path / "out", precision="32", export_to_hf=False,
        sample_at_training_progress=[], benchmark_at_training_progress=[],
    ))])


@pytest.mark.parametrize("train_count,eval_count", [(64, 1), (64, 12), (63, 8)])
def test_invalid_counts_precede_setup_and_cleanup(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, train_count: int, eval_count: int, tmp_path: Path,
) -> None:
    settings.micro_batches_per_step = train_count
    settings.eval_iters = eval_count
    # Keep the schedule valid at these larger global batch sizes so this test reaches count validation.
    config = load_dataset_config(settings.dataset_config)
    for stage in config.stages:
        stage.tokens = 20 * settings.tokens_per_optimizer_step
    config_path = tmp_path / "valid_schedule.yaml"
    config_path.write_text(json.dumps(asdict(config)))
    settings.dataset_config = str(config_path)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    backend.world_size = 8
    closed = []

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid counts reached setup")

    for name in ("prepare_run_directory", "resolve_dataset", "build_run_model", "build_run_optimizer"):
        monkeypatch.setattr(run_module, name, unexpected)
    monkeypatch.setattr(backend, "seed_everything", unexpected)
    monkeypatch.setattr(backend, "shutdown", lambda: closed.append(True))
    with pytest.raises(ValueError, match="must be a multiple"):
        train(settings, backend=backend)
    assert closed == [True]


@pytest.mark.parametrize("eval_count", [8, 64])
def test_valid_distributed_counts(settings: Settings, eval_count: int) -> None:
    settings.micro_batches_per_step = 64
    settings.eval_iters = eval_count
    settings.validate_world_size(8)


def test_controller_completes_before_sampling_and_latches() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    events = []
    requested = Event()
    original_vote = backend.any_flag

    def vote(flag: bool) -> bool:
        events.append(("vote", flag))
        return original_vote(flag)

    backend.any_flag = vote  # type: ignore[method-assign]

    def local() -> bool:
        events.append(("sample", requested.is_set()))
        return requested.is_set()

    def work() -> None:
        events.append(("work", False))
        requested.set()

    stop = StopController(backend, local)
    complete_main_phase(backend, "samples", work)
    assert stop.poll("after samples")
    requested.clear()
    assert stop.poll("before update")
    assert events[:4] == [("work", False), ("vote", False), ("sample", True), ("vote", True)]


@pytest.mark.slow
@pytest.mark.parametrize("phase", ["step", "validation", "checkpoint", "samples", "benchmark", "final_step"])
def test_request_at_each_completed_phase(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    requested = Event()
    events: list[str] = []
    backend = SingleDeviceBackend(device="cpu", precision="32")
    settings.save_step_interval = 1
    settings.eval_step_interval = 1
    settings.sample_step_interval = 1
    settings.benchmark_step_interval = 1
    settings.benchmark_tasks = ["stub"]
    settings.export_to_hf = True
    original_step = run_one_optimizer_step
    original_evaluate = evaluate
    original_save = save_training_checkpoint
    step_count = 0

    def step(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_count
        result = original_step(*args, **kwargs)
        step_count += 1
        events.append("step")
        if phase == "step" or (phase == "final_step" and step_count == 20):
            requested.set()
        return result

    def evaluation(*args: Any, **kwargs: Any) -> Any:
        result = original_evaluate(*args, **kwargs)
        events.append("validation")
        if phase == "validation":
            requested.set()
        return result

    def save(*args: Any, **kwargs: Any) -> None:
        original_save(*args, **kwargs)
        events.append("checkpoint")
        if phase == "checkpoint":
            requested.set()

    def optional(name: str) -> Callable[..., None]:
        def work(*args: Any, **kwargs: Any) -> None:
            events.append(name)
            if name == phase:
                # Same step, changed RNG: the final checkpoint must refresh the scheduled one.
                torch.rand(3)
                requested.set()
        return work

    monkeypatch.setattr(loop_helpers, "run_one_optimizer_step", step)
    monkeypatch.setattr(loop_helpers, "evaluate", evaluation)
    monkeypatch.setattr(checkpoint_helpers, "save_training_checkpoint", save)
    monkeypatch.setattr(loop_helpers, "write_samples", optional("samples"))
    monkeypatch.setattr(loop_helpers, "run_benchmarks", optional("benchmark"))
    monkeypatch.setattr(loop_helpers, "export_if_requested", optional("export"))
    report = train(settings, backend=backend, should_stop=requested.is_set, keep_history=True)
    completed = 20 if phase == "final_step" else 1
    assert report.completed_steps == completed and report.steps_this_process == completed
    assert report.stopped == (phase != "final_step")
    assert "export" not in events
    assert len(report.history) == completed
    if phase in ("step", "final_step"):
        assert "val_loss" not in report.history[completed]
    else:
        assert "val_loss" in report.history[completed]
    if phase == "final_step":
        assert events[-2:] == ["step", "checkpoint"]
    else:
        expected = {
            "step": ["step", "checkpoint"],
            "validation": ["step", "validation", "checkpoint"],
            "checkpoint": ["step", "validation", "checkpoint"],
            "samples": ["step", "validation", "checkpoint", "samples", "checkpoint"],
            "benchmark": ["step", "validation", "checkpoint", "samples", "benchmark", "checkpoint"],
        }
        assert events == expected[phase]
    stored = torch.load(report.checkpoints_written[-1], weights_only=False)
    assert torch.equal(stored["rng_states"][0]["torch"], torch.get_rng_state())
    assert stored["step"] == completed


@pytest.mark.slow
def test_step_zero_stop_and_immediately_stopped_resume_preserve_continuation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    original_step = run_one_optimizer_step

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("an already requested stop must not consume a training pack")

    monkeypatch.setattr(loop_helpers, "run_one_optimizer_step", unexpected)
    report = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), should_stop=lambda: True)
    assert report.stopped and report.completed_steps == report.steps_this_process == 0
    checkpoint = report.checkpoints_written[0]
    assert checkpoint.name == "step-00000000-tiny.pth"
    initial = torch.load(checkpoint, weights_only=False)
    assert initial["step"] == initial["stage"] == 0
    settings.resume = True
    resumed_stop = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), should_stop=lambda: True)
    assert resumed_stop.completed_steps == resumed_stop.steps_this_process == 0
    restored = torch.load(resumed_stop.checkpoints_written[0], weights_only=False)
    assert torch.equal(initial["rng_states"][0]["torch"], restored["rng_states"][0]["torch"])
    assert initial["data_stream"] == restored["data_stream"]
    monkeypatch.setattr(loop_helpers, "run_one_optimizer_step", original_step)
    resumed = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)
    settings.resume = False
    settings.out_dir = str(tmp_path / "fresh")
    full = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)
    for done in full.history:
        for key in ("loss", "grad_norm", "lr"):
            assert resumed.history[done][key] == full.history[done][key]


@pytest.mark.slow
def test_two_rank_stop_agreement_completion_errors_and_exact_resume(
    tmp_path: Path, tiny_dataset_dir: Path,
) -> None:
    import os
    import signal
    import subprocess
    import sys

    from training.testing.network import free_port

    config = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, tmp_path / "out", backend="ddp", precision="32", export_to_hf=False,
        sample_at_training_progress=[], benchmark_at_training_progress=[],
    )
    for mode in ("stop", "provenance_failure", "resume", "full"):
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
                   "--max-restarts=0", "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{free_port()}",
                   "-m", "training.testing.stopping_worker", str(config), str(tmp_path), mode]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   start_new_session=True, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
        def kill_owned_workers(worker_mode: str = mode) -> None:
            # torchrun workers may own separate process groups. Check command-line ownership as well
            # as this launch's PID files, so a reused PID never targets an unrelated process.
            for rank in (0, 1):
                pid_file = tmp_path / f"{worker_mode}-rank-{rank}.pid"
                if pid_file.exists():
                    pid = int(pid_file.read_text())
                    try:
                        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                        if b"training.testing.stopping_worker" in arguments and str(tmp_path).encode() in arguments:
                            os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except FileNotFoundError:
                        pass

        try:
            output, _ = process.communicate(timeout=60)
            assert process.returncode == 0, output[-12000:]
        except subprocess.TimeoutExpired:
            kill_owned_workers()
            process.kill()
            output, _ = process.communicate(timeout=10)
            pytest.fail(f"{mode} workers exceeded 60 seconds:\n{output[-12000:]}")
        finally:
            kill_owned_workers()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        for rank in (0, 1):
            assert (tmp_path / f"{mode}-rank-{rank}-passed").exists()


@pytest.mark.slow
def test_checkpoint_publication_failure_is_visible_and_not_logged_as_saved(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    logged: list[Path] = []
    closed: list[bool] = []

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("atomic publication refused")

    monkeypatch.setattr(backend, "save_checkpoint", fail)
    monkeypatch.setattr(backend, "shutdown", lambda: closed.append(True))
    monkeypatch.setattr(RunLogger, "log_checkpoint", lambda self, path: logged.append(path))
    with pytest.raises(OSError, match="atomic publication refused"):
        train(settings, backend=backend, should_stop=lambda: True)
    assert logged == [] and closed == [True]
    assert list((Path(settings.out_dir) / settings.run_name / "checkpoints").glob("*.pth")) == []


@pytest.mark.slow
def test_request_at_final_export_boundary_suppresses_export_but_keeps_completion(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = Event()
    poll = StopController.poll

    def at_export(self: StopController, boundary: str) -> bool:
        if boundary == "before final export":
            request.set()
        return poll(self, boundary)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("export started after the shared stop decision")

    monkeypatch.setattr(StopController, "poll", at_export)
    monkeypatch.setattr(loop_helpers, "export_if_requested", unexpected)
    report = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), should_stop=request.is_set)
    assert report.completed_steps == report.steps_this_process == 20 and not report.stopped
    assert report.export_dir is None
    assert len([p for p in report.checkpoints_written if "00000020" in p.name]) == 1
