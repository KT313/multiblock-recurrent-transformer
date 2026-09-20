# (c) 2026 Tobias Kerner. Apache-2.0.
"""Opt-in validation and real two-rank, CPU-only optimizer compatibility."""

import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from training.optim import build_optimizer
from training.settings import OptimizerConfig, parse_settings
from training.testing.network import free_port

ROOT = Path(__file__).resolve().parent.parent


def test_sharding_requires_ddp_and_valid_selector() -> None:
    settings = parse_settings(["--config", str(ROOT / "config/tiny.yaml")])
    assert settings.optimizer_sharding == "none"
    with pytest.raises(ValueError, match="requires backend"):
        replace(settings, optimizer_sharding="zero1")
    with pytest.raises(ValueError, match="must be"):
        replace(settings, optimizer_sharding="invalid")  # type: ignore[arg-type]
    for name in ("ELLISAdam", "ELLISAdam8bit", "AdamW"):
        assert replace(settings, backend="ddp", optimizer=name, optimizer_sharding="zero1").optimizer == name


def test_sharding_requires_initialized_process_group() -> None:
    with pytest.raises(ValueError, match="initialized DDP"):
        build_optimizer("ELLISAdam", [torch.nn.Parameter(torch.ones(2))], OptimizerConfig(), sharding="zero1")


@pytest.mark.slow
def test_two_rank_optimizer_update_metrics_and_resume() -> None:
    completed = _run_ranks(["--module", "training.testing.optimizer_sharding_worker"])
    assert completed.returncode == 0, completed.stdout[-12000:]
    assert completed.stdout.count("PASS ") == 3, completed.stdout


def _run_ranks(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--max-restarts=0",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{free_port()}",
    ] + arguments
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=120)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)
            raise
    return subprocess.CompletedProcess(command, process.returncode, output)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ELLISAdam", "ELLISAdam8bit"])
def test_sharded_training_matches_ddp_and_resumes(
    name: str, tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    from training.test_distributed import two_rank_yaml
    from training.testing.optimizer_sharding_worker import compare

    def run(directory: str, **options: object) -> tuple[Path, dict[str, object]]:
        yaml_path, run_directory = two_rank_yaml(
            tmp_path / directory, tiny_dataset_dir, optimizer=name,
            tokens_per_micro_batch=256, micro_batches_per_step=4, **options,
        )  # two accumulated microbatches per rank, preserving the fixture's global token budget
        completed = _run_ranks([str(ROOT / "training/train.py"), "--config", str(yaml_path)])
        assert completed.returncode == 0, completed.stdout[-12000:]
        path = run_directory / "checkpoints" / "step-00000020-tiny.pth"
        return run_directory, dict(torch.load(path, map_location="cpu", weights_only=False))

    _, baseline = run("baseline")
    directory, sharded = run("sharded", optimizer_sharding="zero1")
    compare(baseline["model"], sharded["model"])
    compare(baseline["optimizer"], sharded["optimizer"])
    _, resumed = run(
        "resumed", optimizer_sharding="zero1", resume=True,
        resume_checkpoint_path=str(directory / "checkpoints" / "step-00000010-tiny.pth"),
    )
    for key in ("model", "optimizer", "rng_states", "data_stream"):
        compare(sharded[key], resumed[key])
