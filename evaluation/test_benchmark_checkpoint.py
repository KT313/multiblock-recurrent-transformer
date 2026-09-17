# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Immediate benchmark checks restore real weights and use the scheduled inference path."""

import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
import yaml

from evaluation.benchmark_checkpoint import main
from evaluation.cli.benchmark_checkpoint import (
    BenchmarkOptions, create_benchmark_output, load_benchmark_inputs, parse_benchmark_settings, select_benchmark_checkpoint,
)
from evaluation.test_distributed import run_launcher
from evaluation.testing import build_offline_task
from model.model import RecurrentGPT


@pytest.fixture
def benchmark_run(tmp_path: Path, tiny_model: RecurrentGPT, tiny_dataset_dir: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "tiny" / "checkpoints" / "step-00000003-tiny.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model": tiny_model.state_dict(), "model_config": tiny_model.config.to_dict(), "step": 3,
                "settings": {}, "optimizer": {"unused": torch.ones(4)}, "world_size": 8}, checkpoint)
    settings = yaml.safe_load(Path("config/tiny.yaml").read_text())
    settings.update(out_dir=str(tmp_path), dataset_dir=str(tiny_dataset_dir), use_custom_kernels=False,
                    precision="32", benchmark_tasks=["offline_multiple_choice", "offline_generate_until"],
                    benchmark_limit=1, benchmark_batch_size=1, benchmark_num_fewshot=0,
                    benchmark_recurrences=[[1, 1], [2, 2]], benchmark_at_training_progress=[100.0],
                    resume=True, resume_checkpoint_path="intentionally-ignored.pth")
    config = tmp_path / "training.yaml"
    config.write_text(yaml.safe_dump(settings))
    return config, checkpoint


def test_select_restore_and_validate(
    benchmark_run: tuple[Path, Path], tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint = benchmark_run
    settings, options = parse_benchmark_settings(["--config", str(config), "--benchmark_limit", "2"])
    assert settings.benchmark_limit == 2 and options == BenchmarkOptions()
    assert select_benchmark_checkpoint(settings, None) == checkpoint
    settings.resume = False
    settings.resume_checkpoint_path = None
    assert select_benchmark_checkpoint(settings, None) == checkpoint
    failed = checkpoint.with_name("step-00000004-tiny-failed.pth")
    failed.write_text("not automatically selected")
    assert select_benchmark_checkpoint(settings, None) == checkpoint
    assert select_benchmark_checkpoint(settings, str(failed)) == failed
    with pytest.raises(FileNotFoundError):
        select_benchmark_checkpoint(settings, "missing.pth")

    def forbid_init(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("checkpoint restoration must skip expensive matrix initialization")
    monkeypatch.setattr("model.layers.init.trunc_orthogonal_", forbid_init)
    model, tokenizer, step = load_benchmark_inputs(settings, options, checkpoint, torch.device("cpu"))
    assert step == model.step == 3 and len(tokenizer) <= tiny_model.config.vocab_size
    for name, tensor in tiny_model.state_dict().items():
        assert torch.equal(model.state_dict()[name], tensor)
    settings.model_overwrite = {"n_embd": 96}
    with pytest.raises(ValueError, match="architecture differs"):
        load_benchmark_inputs(settings, options, checkpoint, torch.device("cpu"))


def test_check_output_never_overwrites(benchmark_run: tuple[Path, Path]) -> None:
    config, _ = benchmark_run
    settings, options = parse_benchmark_settings(["--config", str(config)])
    first = create_benchmark_output(settings, options, 3)
    second = create_benchmark_output(settings, options, 3)
    assert first != second and first.parent.name == "benchmark_checks"
    with pytest.raises(FileExistsError):
        create_benchmark_output(settings, BenchmarkOptions(output_dir=str(first)), 3)


def test_cli_runs_real_offline_benchmarks(
    benchmark_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lm_eval.tasks import TaskManager

    config, checkpoint = benchmark_run
    before = checkpoint.read_bytes()
    load_tasks = TaskManager.load
    def load_offline(manager: Any, specs: Any) -> Any:
        return load_tasks(manager, [build_offline_task(name.removeprefix("offline_"), 2) for name in specs])
    monkeypatch.setattr(TaskManager, "load", load_offline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--config", str(config)]) == 0
    result = next((checkpoint.parent.parent / "benchmark_checks").glob("*/step-00000003.json"))
    saved = json.loads(result.read_text())
    assert saved["step"] == 3 and saved["limit"] == 1 and saved["execution_precision"] == "32"
    assert saved["recurrences"] == [[1, 1], [2, 2]]
    assert saved["tasks"] == ["offline_multiple_choice", "offline_generate_until"]
    assert saved["execution_metadata"]["recurrences"]["1-1"]["protocol"] == "fixed_document_jobs_v1"
    assert checkpoint.read_bytes() == before
    assert not (checkpoint.parent.parent / "benchmarks").exists()
    assert not (checkpoint.parent.parent / "samples").exists()


def test_stop_and_missing_checkpoint(benchmark_run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.cli.benchmark_checkpoint import run_checkpoint_benchmark_check

    config, checkpoint = benchmark_run
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings, options = parse_benchmark_settings(["--config", str(config)])
    assert run_checkpoint_benchmark_check(settings, options, should_stop=lambda: True) == 130
    assert not list((checkpoint.parent.parent / "benchmark_checks").glob("*/*.json"))
    assert main(["--config", str(config), "--checkpoint", "missing.pth"]) == 1


@pytest.mark.slow
@pytest.mark.parametrize("limit", [1, 9])
def test_checkpoint_cli_two_ranks(benchmark_run: tuple[Path, Path], tmp_path: Path, limit: int) -> None:
    config, checkpoint = benchmark_run
    worker = tmp_path / "worker.py"
    worker.write_text('''import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from unittest.mock import patch
from lm_eval.tasks import TaskManager
from evaluation.testing import build_offline_task
from evaluation.benchmark_checkpoint import main
load_tasks = TaskManager.load
def load_offline(manager, specs):
    return load_tasks(manager, [build_offline_task(name.removeprefix("offline_"), 9) for name in specs])
with patch.object(TaskManager, "load", load_offline):
    sys.exit(main())
''')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(worker),
               "--config", str(config), "--backend", "ddp", "--benchmark_limit", str(limit)]
    log = tmp_path / "workers.log"
    assert run_launcher(command, environment, log, 120) == 0, log.read_text()[-16000:]
    outputs = list((checkpoint.parent.parent / "benchmark_checks").glob("*/*.json"))
    assert len(outputs) == 1
    saved = json.loads(outputs[0].read_text())
    for metadata in saved["execution_metadata"]["recurrences"].values():
        assert metadata["inference_world_size"] == 2 and metadata["evaluator_world_size"] == 1
        assert metadata["workers"][1]["jobs"] > 0 if limit == 9 else metadata["workers"][1]["jobs"] == 0
