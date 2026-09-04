# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests of the checkpoint CLI on a hand-made tiny checkpoint.
"""

import json
from pathlib import Path

import pytest
import torch

from evaluation.prompts import DEFAULT_PROMPTS
from evaluation.evaluate import load_checkpoint_model, main, parse_recurrences, tokenizer_dir_of
from evaluation.test_benchmarks import stub_lm_eval
from model.model import RecurrentGPT

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def checkpoint(tiny_model: RecurrentGPT, tiny_dataset_dir: Path, tmp_path: Path) -> Path:
    """
    `<tmp>/run/checkpoints/step-00000003-tiny.pth` holding the tiny model; `model_config` carries a stale `name`
    key as checkpoints written before its removal do.
    """

    path = tmp_path / "run" / "checkpoints" / "step-00000003-tiny.pth"
    path.parent.mkdir(parents=True)
    state = {
        "model": tiny_model.state_dict(),
        "model_config": {**tiny_model.config.to_dict(), "name": "old"},
        "step": 3,
        "settings": {"dataset_config": str(REPO_ROOT / "config" / "datasets" / "tiny.yaml"), "dataset_dir": str(tiny_dataset_dir)},
    }
    torch.save(state, path)
    return path


def test_parse_recurrences() -> None:
    assert parse_recurrences(None) == [None] and parse_recurrences([]) == [None]
    assert parse_recurrences(["4,12,4", "8"]) == [[4, 12, 4], [8]]


def test_checkpoint_model_and_tokenizer_dir(checkpoint: Path, tiny_model: RecurrentGPT, tiny_tokenizer_dir: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_checkpoint_model(state, "cpu")
    assert model.config == tiny_model.config
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, tiny_model.state_dict()[name]), name
    assert tokenizer_dir_of(state) == tiny_tokenizer_dir


def test_cli_writes_samples_and_benchmarks_into_the_run_directory(
    checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    assert main(["--checkpoint", str(checkpoint), "--device", "cpu", "--max_new_tokens", "3", "--tasks", ""]) == 0
    samples = run_dir / "samples" / "step-00000003.jsonl"
    assert samples.is_file() and len(samples.read_text().split("\n")[:-1]) == len(DEFAULT_PROMPTS)
    assert not (run_dir / "benchmarks").exists()
    assert f"{len(DEFAULT_PROMPTS)} samples written to" in capsys.readouterr().out

    calls = stub_lm_eval(monkeypatch)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Only one prompt")
    argv = ["--checkpoint", str(checkpoint), "--device", "cpu", "--max_new_tokens", "2", "--prompts_file", str(prompts)]
    argv += ["--recurrence", "1,1", "--recurrence", "2,2"]
    assert main([*argv, "--tasks", "arc_easy,hellaswag", "--limit", "5", "--out_dir", str(tmp_path / "elsewhere")]) == 0
    written = json.loads((tmp_path / "elsewhere" / "benchmarks" / "step-00000003.json").read_text())
    assert written["tasks"] == ["arc_easy", "hellaswag"] and calls["evaluate"]["limit"] == 5
    assert written["recurrences"] == [[1, 1], [2, 2]] and set(written["results"]) == {"1-1", "2-2"}
    lines = [json.loads(line) for line in (tmp_path / "elsewhere" / "samples" / "step-00000003.jsonl").read_text().split("\n")[:-1]]
    assert [(line["prompt"], line["recurrence"]) for line in lines] == [("Only one prompt", [1, 1]), ("Only one prompt", [2, 2])]
    out = capsys.readouterr().out
    assert "benchmark/1-1/arc_easy/acc" in out and "benchmark/2-2/hellaswag/acc" in out and "2 samples written" in out

    assert main([*argv, "--no_samples", "--out_dir", str(tmp_path / "none")]) == 0
    assert not (tmp_path / "none").exists()
