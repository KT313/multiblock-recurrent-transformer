# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the golden-run helpers (`training.testing.golden`): the tiny-yaml writer, the comparison and the
single-thread / deterministic block. The golden run itself is `test_run.py::test_golden_tiny_run`.
"""

import json
from pathlib import Path

import pytest
import torch
import yaml

from training.testing.golden import (
    GOLDEN_RUN_PATH,
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    single_thread_deterministic,
    write_tiny_yaml,
)
from training.settings import parse_settings


def test_write_tiny_yaml_rewrites_paths_and_overrides(tmp_path: Path) -> None:
    """
    `out_dir` / `dataset_dir` are rewritten, an existing key is replaced in place (a string value that looks like
    a number stays a string), a new key is appended, and the result parses as settings.
    """

    path = write_tiny_yaml(tmp_path, tmp_path / "data", tmp_path / "out", precision="32", resume_warmup_steps=2)
    assert path == tmp_path / "tiny.yaml"
    text = path.read_text()
    written = yaml.safe_load(text)
    assert written["out_dir"] == str(tmp_path / "out") and written["dataset_dir"] == str(tmp_path / "data")
    assert text.count("precision:") == 1 and written["precision"] == "32"
    assert text.rstrip().endswith("resume_warmup_steps: 2")  # appended: tiny.yaml has no such key
    settings = parse_settings(["--config", str(path)])
    assert settings.precision == "32" and settings.resume_warmup_steps == 2 and settings.run_name == "tiny"


def test_golden_exact_requested_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOLDEN_EXACT", raising=False)
    assert golden_exact_requested() is False
    monkeypatch.setenv("GOLDEN_EXACT", "1")
    assert golden_exact_requested() is True
    monkeypatch.setenv("GOLDEN_EXACT", "0")
    assert golden_exact_requested() is False


def test_single_thread_deterministic_restores_the_torch_settings() -> None:
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with single_thread_deterministic():
        assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == threads and torch.are_deterministic_algorithms_enabled() == deterministic
    with pytest.raises(RuntimeError, match="boom"), single_thread_deterministic():
        raise RuntimeError("boom")
    assert torch.get_num_threads() == threads and torch.are_deterministic_algorithms_enabled() == deterministic


def test_golden_run_json_round_trips_floats() -> None:
    metrics = {"steps": {"1": {"loss": 1.0000001, "lr": 3e-4}}, "checkpoints": ["a.pth"]}
    text = golden_run_json(metrics)
    assert text.endswith("\n") and text.index('"checkpoints"') < text.index('"steps"')  # sorted keys
    assert json.loads(text) == metrics


def test_golden_mismatches_reports_every_difference() -> None:
    expected = {"steps": {"1": {"loss": 1.0, "lr": 2.0}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, json.loads(golden_run_json(expected)), exact=True) == []
    # rel 1e-5 on plain floats, but never on the learning rate, the checkpoint names or the step count
    close = {"steps": {"1": {"loss": 1.0 + 1e-7, "lr": 2.0}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, close, exact=False) == []
    assert golden_mismatches(expected, close, exact=True) == ["/steps/1/loss: 1.0 != 1.0000001"]
    lr_off = {"steps": {"1": {"loss": 1.0, "lr": 2.0 + 1e-7}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, lr_off, exact=False) == ["/steps/1/lr: 2.0 != 2.0000001"]
    off = {"steps": {"1": {"loss": 1.1, "grad_norm": 0.0}}, "checkpoints": ["b.pth"], "optimizer_steps": 18}
    assert golden_mismatches(expected, off, exact=False) == [
        "/checkpoints: ['a.pth'] != ['b.pth']",
        "/optimizer_steps: 19 != 18",
        "/steps/1/lr: missing",
        "/steps/1/grad_norm: unexpected",
        "/steps/1/loss: 1.0 != 1.1",
    ]
    # the relative tolerance has no absolute part, and a non-number never agrees with a float
    assert golden_mismatches({"x": 0.0}, {"x": 1e-12}, exact=False) == ["/x: 0.0 != 1e-12"]
    assert golden_mismatches({"x": 1.0}, {"x": "1.0"}, exact=False) == ["/x: 1.0 != '1.0'"]
    assert golden_mismatches({"x": 1.0}, {"x": True}, exact=False) == ["/x: 1.0 != True"]
    assert golden_mismatches({"x": {"y": 1}}, {"x": [1]}, exact=False) == ["/x: expected a mapping, got list"]


def test_golden_fixture_is_committed() -> None:
    golden = json.loads(GOLDEN_RUN_PATH.read_text())
    assert set(golden) == {"steps", "checkpoints", "optimizer_steps", "parameter_norms"}
    assert golden["optimizer_steps"] == 19 and len(golden["steps"]) == 20
