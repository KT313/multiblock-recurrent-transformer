# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Resume history records describe successful setup without changing old evidence or replay."""

import json
import os
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from training import provenance, run as run_module
from training.backend.single_device import SingleDeviceBackend
from training.logger import RunLogger
from training.settings import Settings, parse_settings
from training.step import RankBatches
from training.testing.golden import write_tiny_yaml


@pytest.fixture
def settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(write_tiny_yaml(
        tmp_path, tiny_dataset_dir, tmp_path / "out", precision="32", export_to_hf=False,
        sample_at_training_progress=[], benchmark_at_training_progress=[],
    ))])


def _train(settings: Settings, *, step: bool = False) -> None:
    polls = 0

    def stop() -> bool:
        nonlocal polls
        polls += 1
        return polls > int(step)

    run_module.train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), should_stop=stop)


@pytest.fixture
def checkpoint_run(settings: Settings) -> tuple[Path, Path]:
    _train(settings, step=True)
    root = run_module.run_directory_of(settings)
    checkpoint = root / "checkpoints" / "step-00000001-tiny.pth"
    assert checkpoint.is_file()
    settings.resume = True
    settings.resume_checkpoint_path = str(checkpoint)
    return root, checkpoint


def _sidecars(root: Path) -> dict[str, bytes]:
    return {name: (root / name).read_bytes() for name in ("run_config.json", "model_config.json")}


def _records(root: Path) -> list[Path]:
    return sorted((root / "resumes").glob("*.json"))


@pytest.mark.parametrize("failure", ["shape", "settings", "dataset"])
def test_rejected_resume_preserves_originals(
    settings: Settings, checkpoint_run: tuple[Path, Path], failure: str,
) -> None:
    root, _ = checkpoint_run
    original = _sidecars(root)
    if failure == "shape":
        settings.model_overwrite = {"n_embd": 32}
    elif failure == "settings":
        settings.grad_clip += 0.5
    else:
        checkpoint = checkpoint_run[1]
        stored = torch.load(checkpoint, weights_only=False)
        stored["dataset_config_hash"] = "incompatible"
        torch.save(stored, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        _train(settings)
    assert _sidecars(root) == original
    assert _records(root) == []


def test_accepted_resumes_append_and_report_restored_optimizer(
    settings: Settings, checkpoint_run: tuple[Path, Path],
) -> None:
    root, checkpoint = checkpoint_run
    original = _sidecars(root)
    for changed in (False, True):
        if changed:
            settings.grad_clip += 0.5
            settings.allow_settings_change = True
        _train(settings)
        assert _sidecars(root) == original
    paths = _records(root)
    assert len(paths) == 2
    records = [json.loads(path.read_text()) for path in paths]
    assert len({record["attempt_id"] for record in records}) == 2
    assert all(record["source_checkpoint"] == {"path": str(checkpoint.resolve()), "step": 1} for record in records)
    assert all(record["completed_optimizer_updates_this_attempt"] == 0 for record in records)
    for record in records:
        groups = record["effective"]["optimizer"]["parameter_groups_at_acceptance"]
        checkpoint_state = torch.load(checkpoint, weights_only=False)
        stored = checkpoint_state["optimizer"]["param_groups"]
        assert record["dataset"]["identity_status"] == record["dataset"]["checkpoint_identity_status"] == "available"
        assert record["dataset"]["build_id"] == record["dataset"]["checkpoint_build_id"] == checkpoint_state["dataset_build_id"]
        assert [group["hyperparameters"]["lr"] for group in groups] == [group["lr"] for group in stored]
        assert groups[0]["hyperparameters"]["lr"] != record["requested_settings"]["optim_config"]["lr"]
        assert "optim_config" not in record["effective"]["run_settings"]
        assert all("params" not in group["hyperparameters"] for group in groups)
    assert records[1]["acknowledgements"]["allow_settings_change"]
    assert records[1]["differences_from_checkpoint"]["settings"]["grad_clip"]["requested"] == settings.grad_clip
    previous = {path.name: path.read_bytes() for path in paths}
    _train(settings)
    assert len(_records(root)) == 3
    assert all((root / "resumes" / name).read_bytes() == content for name, content in previous.items())


@pytest.mark.parametrize("new_directory", [False, True])
def test_missing_originals_are_explicit_checkpoint_evidence(
    settings: Settings, checkpoint_run: tuple[Path, Path], new_directory: bool, tmp_path: Path,
) -> None:
    root, checkpoint = checkpoint_run
    stored = torch.load(checkpoint, weights_only=False)
    if new_directory:
        settings.out_dir = str(tmp_path / "new")
        root = run_module.run_directory_of(settings)
    else:
        for name in _sidecars(root):
            (root / name).unlink()
    settings.grad_clip += 0.5
    settings.allow_settings_change = True
    _train(settings)
    for filename, key in (("run_config.json", "settings"), ("model_config.json", "model_config")):
        record = json.loads((root / filename).read_text())
        origin = record.pop("_provenance")
        assert record == json.loads(json.dumps(stored[key]))
        assert origin["origin"] == "checkpoint_reconstruction"
        assert origin["original_run_configuration"] == "unknown"
    assert len(_records(root)) == 1


@pytest.mark.parametrize("failure", ["write", "rename"])
def test_publication_failure_has_no_partial_record_or_update(
    settings: Settings, checkpoint_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    root, _ = checkpoint_run
    _train(settings)
    old = _sidecars(root) | {str(path.relative_to(root)): path.read_bytes() for path in _records(root)}
    real_write = Path.write_text
    real_replace = os.replace

    def write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.parent.name == "resumes":
            real_write(path, "partial")
            raise OSError("disk full")
        return real_write(path, *args, **kwargs)

    def rename(source: Any, destination: Any) -> None:
        if Path(destination).parent.name == "resumes":
            raise OSError("rename refused")
        real_replace(source, destination)

    def update(*args: Any, **kwargs: Any) -> None:
        pytest.fail("publication failure reached optimizer update")

    monkeypatch.setattr(run_module, "run_one_optimizer_step", update)
    monkeypatch.setattr(Path, "write_text", write if failure == "write" else real_write)
    monkeypatch.setattr(os, "replace", rename if failure == "rename" else real_replace)
    with pytest.raises(OSError, match="disk full|rename refused"):
        run_module.train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"))
    assert all((root / name).read_bytes() == content for name, content in old.items())
    assert len(_records(root)) == 1
    assert not list((root / "resumes").glob("*.tmp"))


@pytest.mark.parametrize("phase", ["logger", "stream"])
def test_prerequisite_failure_publishes_nothing(
    settings: Settings, checkpoint_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    root, _ = checkpoint_run
    old = _sidecars(root)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("setup failed")

    if phase == "logger":
        monkeypatch.setattr(RunLogger, "open", fail)
    else:
        monkeypatch.setattr(RankBatches, "load_state_dict", fail)
    with pytest.raises(OSError, match="setup failed"):
        _train(settings)
    assert _sidecars(root) == old
    assert not _records(root)


def test_publication_preserves_training_rng_and_settings(
    settings: Settings, checkpoint_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = provenance.publish_configuration
    observed: list[bool] = []

    def publish(*args: Any, **kwargs: Any) -> Path | None:
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(legacy=True), torch.get_rng_state()
        requested = asdict(settings)
        result = original_publish(*args, **kwargs)
        assert random.getstate() == python_rng
        current_numpy = np.random.get_state(legacy=True)
        assert isinstance(numpy_rng, tuple) and isinstance(current_numpy, tuple)
        assert current_numpy[0] == numpy_rng[0] and np.array_equal(current_numpy[1], numpy_rng[1])
        assert current_numpy[2:] == numpy_rng[2:]
        assert torch.equal(torch.get_rng_state(), torch_rng)
        assert asdict(settings) == requested
        observed.append(True)
        return result

    monkeypatch.setattr(run_module, "publish_configuration", publish)
    _train(settings)
    assert observed == [True]


def test_ellis_tensor_hyperparameters_are_recorded_as_scalars(settings: Settings) -> None:
    settings.optimizer = "ELLISAdam"
    _train(settings, step=True)
    settings.resume = True
    _train(settings)
    record = json.loads(_records(run_module.run_directory_of(settings))[0].read_text())
    group = record["effective"]["optimizer"]["parameter_groups_at_acceptance"][0]["hyperparameters"]
    assert group["lr"] == 0.0
    assert group["init_lr"] == settings.optim_config.lr
    assert group["eps"] == 1e-6  # resolved ELLIS default; requested eps is null
    assert record["requested_settings"]["optim_config"]["eps"] is None


def test_unavailable_git_revision_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("git is unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert provenance.code_revision() == {"status": "unavailable", "revision": None}


@pytest.mark.parametrize("resume_without_checkpoint", [False, True])
def test_fresh_start_cannot_reuse_established_run(
    settings: Settings, checkpoint_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
    resume_without_checkpoint: bool,
) -> None:
    root, checkpoint = checkpoint_run
    settings.resume = resume_without_checkpoint
    settings.resume_checkpoint_path = None
    settings.grad_clip += 0.5
    if resume_without_checkpoint:
        checkpoint.unlink()
    original = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("fresh reuse reached dataset, model, optimizer, or logger setup")

    for name in ("resolve_dataset", "build_run_dataloaders", "build_run_model", "build_run_optimizer"):
        monkeypatch.setattr(run_module, name, unexpected)
    monkeypatch.setattr(RunLogger, "open", unexpected)
    with pytest.raises(ValueError, match="refusing a fresh start.*new run_name or out_dir.*valid checkpoint"):
        _train(settings)
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("evidence", ["run_config.json", "model_config.json", "train_report.json",
                                      "resumes/accepted.json", "checkpoints/step-00000001-tiny-failed.pth"])
def test_each_original_history_or_checkpoint_artifact_blocks_fresh_reuse(tmp_path: Path, evidence: str) -> None:
    artifact = tmp_path / evidence
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("existing evidence")
    with pytest.raises(ValueError, match="established run directory"):
        provenance.check_fresh_run_directory(tmp_path)
    assert artifact.read_text() == "existing evidence"


@pytest.mark.parametrize("resume", [False, True])
def test_precreated_empty_run_directory_allows_fresh_start(settings: Settings, resume: bool) -> None:
    root = run_module.prepare_run_directory(settings)
    (root / "resumes").mkdir()
    settings.resume = resume
    _train(settings)
    assert _sidecars(root)
    assert _records(root) == []
