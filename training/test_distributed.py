# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Two-rank runs of the tiny config through a real torchrun launch on the CPU (gloo): the multi-rank loop end to end.

Every test here launches `python -m torch.distributed.run --nproc_per_node=2 training/train.py` as a subprocess
with `backend: ddp`, fp32, one thread per rank, CUDA hidden (`CUDA_VISIBLE_DEVICES=""`, so the backend takes the
gloo path also on a machine with a GPU) and the dashboard off. The base run is a module-scoped fixture; the tests
read its checkpoints and reports, launch a second run for determinism and a third for the resume. Slow (each launch
is a few seconds of torchrun start-up plus the run); the module shares one xdist worker through its fixture.

What is checked: the run finishes with the files on rank 0 only, the checkpoint carries two RNG states and one
stream state, the run has the golden run's shape (the same packs reach the same optimizer step, split over two ranks
instead of accumulated on one; the numbers differ because every rank draws its own latent noise), two identical
launches are bit-identical, a resume reproduces the uninterrupted run exactly, a one-rank resume of the checkpoint is
refused, and rank 1 says nothing below WARNING.
"""

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from training.checkpoint import checkpoint_dir
from training.testing.golden import GOLDEN_RUN_PATH, optimizer_steps_taken, write_tiny_yaml
from training.ui.common import TRAIN_LOG_NAME, TRAIN_REPORT_NAME

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
RANKS = 2
LAUNCH_TIMEOUT_SECONDS = 600
DDP_OPTIONS = {"backend": "ddp", "precision": "32", "wandb_enabled": False, "export_to_hf": False, "resume": False}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class Launch:
    """
    One finished torchrun launch: its exit code, both ranks' stderr (torchrun interleaves them), stdout, and the run
    directory of the config it ran.
    """

    returncode: int
    stdout: str
    stderr: str
    run_directory: Path

    def final_checkpoint(self) -> dict[str, object]:
        path = checkpoint_dir(self.run_directory) / "step-00000020-tiny.pth"
        assert path.exists(), sorted(p.name for p in checkpoint_dir(self.run_directory).glob("*.pth"))
        return dict(torch.load(path, map_location="cpu", weights_only=False))


def launch(yaml_path: Path, run_directory: Path, *arguments: str, ranks: int = RANKS) -> Launch:
    """
    `torchrun --nproc_per_node=<ranks> training/train.py --config <yaml> <arguments>` on the CPU, waited for.
    """

    env = os.environ | {
        "TRAINING_DASHBOARD": "0",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",  # the gloo path, also on a machine with a GPU (two ranks cannot share one NCCL device)
        "PYTHONWARNINGS": "ignore::UserWarning",  # the CPU fallback warning of every rank
    }
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes=1", f"--nproc_per_node={ranks}", "--max-restarts=0",
        "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{free_port()}",
        str(REPO_ROOT / "training" / "train.py"), "--config", str(yaml_path), *arguments,
    ]  # fmt: skip
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT_SECONDS
    )
    return Launch(completed.returncode, completed.stdout, completed.stderr, run_directory)


def single_device_launch(yaml_path: Path, run_directory: Path, *arguments: str) -> Launch:
    """
    `python training/train.py --config <yaml> <arguments>` on the CPU (no torchrun), waited for.
    """

    env = os.environ | {"TRAINING_DASHBOARD": "0", "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, str(REPO_ROOT / "training" / "train.py"), "--config", str(yaml_path), *arguments]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=LAUNCH_TIMEOUT_SECONDS)
    return Launch(completed.returncode, completed.stdout, completed.stderr, run_directory)


def two_rank_yaml(directory: Path, tiny_dataset_dir: Path, **overrides: object) -> tuple[Path, Path]:
    """
    The tiny yaml for a two-rank CPU run into `directory / "out"`, checkpoints every 10 steps; (yaml, run directory).
    """

    directory.mkdir(parents=True, exist_ok=True)
    out_dir = directory / "out"
    yaml_path = write_tiny_yaml(directory, tiny_dataset_dir, out_dir, save_step_interval=10, **(DDP_OPTIONS | overrides))
    return yaml_path, out_dir / "tiny"


@pytest.fixture(scope="module")
def base_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_dir: Path) -> Iterator[Launch]:
    """
    The two-rank tiny run every test reads; launched once per module.
    """

    yaml_path, run_directory = two_rank_yaml(tmp_path_factory.mktemp("ddp_base"), tiny_dataset_dir)
    result = launch(yaml_path, run_directory)
    assert result.returncode == 0, result.stderr[-4000:]
    yield result


def test_two_ranks_finish_with_the_files_on_rank_zero_and_a_two_rank_checkpoint(base_run: Launch) -> None:
    run_directory = base_run.run_directory
    report = json.loads((run_directory / TRAIN_REPORT_NAME).read_text())
    assert report["completed_steps"] == 20 and report["stopped"] is False
    assert (run_directory / TRAIN_LOG_NAME).exists() and (run_directory / "run_config.json").exists()
    assert base_run.stdout.count("last loss") == 1, base_run.stdout  # one summary: rank 0's
    names = sorted(p.name for p in checkpoint_dir(run_directory).glob("*.pth"))
    assert names == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000010-tiny.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    final = base_run.final_checkpoint()
    assert final["world_size"] == RANKS
    rng_states = final["rng_states"]
    assert isinstance(rng_states, list) and len(rng_states) == RANKS
    assert not torch.equal(rng_states[0]["torch"], rng_states[1]["torch"]), "every rank draws its own latent noise"
    stream = final["data_stream"]
    assert isinstance(stream, dict) and set(stream) == {"consumed_rows", "pool_loaded", "pool_target", "buffers", "pool"}
    model_state = final["model"]
    assert isinstance(model_state, dict) and not any(key.startswith("module.") for key in model_state)


def test_two_ranks_train_the_same_run_up_to_the_per_rank_latent_noise(base_run: Launch) -> None:
    """
    The two ranks train on the packs the one-device golden run accumulates (pack 0 on rank 0, pack 1 on rank 1, per
    micro-batch index), with the same optimizer steps and checkpoint names. The numbers are not the golden's: every
    rank draws its own latent noise (`seed + rank`), so the forward passes differ by design and only the shape of
    the run can be compared. The loss after 20 steps lands within a few percent of the golden's.
    """

    final = base_run.final_checkpoint()
    assert optimizer_steps_taken(final["optimizer"]) == 19  # type: ignore[arg-type]  # the optimizer state dict
    golden = json.loads(GOLDEN_RUN_PATH.read_text())
    report = json.loads((base_run.run_directory / TRAIN_REPORT_NAME).read_text())
    assert report["last_loss"] == pytest.approx(golden["steps"]["20"]["loss"], rel=0.05)
    assert report["last_loss"] != golden["steps"]["20"]["loss"], "rank 1's latent noise differs from the one-device run's"


def test_two_identical_launches_are_bit_identical(base_run: Launch, tmp_path: Path, tiny_dataset_dir: Path) -> None:
    yaml_path, run_directory = two_rank_yaml(tmp_path / "again", tiny_dataset_dir)
    again = launch(yaml_path, run_directory)
    assert again.returncode == 0, again.stderr[-4000:]
    _assert_same_model_and_optimizer(base_run.final_checkpoint(), again.final_checkpoint())


def test_a_resumed_two_rank_run_reproduces_the_uninterrupted_one(base_run: Launch, tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    Resume from the base run's step-10 checkpoint into a new run directory: rank 0's stream state and every rank's
    RNG state continue where they stood, so the final checkpoint equals the uninterrupted run's bit for bit.
    """

    middle = checkpoint_dir(base_run.run_directory) / "step-00000010-tiny.pth"
    yaml_path, run_directory = two_rank_yaml(tmp_path / "resumed", tiny_dataset_dir, resume=True, resume_checkpoint_path=str(middle))
    resumed = launch(yaml_path, run_directory)
    assert resumed.returncode == 0, resumed.stderr[-4000:]
    report = json.loads((run_directory / TRAIN_REPORT_NAME).read_text())
    assert report["completed_steps"] == 20 and report["resumed_from"] == str(middle)
    names = sorted(p.name for p in checkpoint_dir(run_directory).glob("*.pth"))
    assert names == ["step-00000014-tiny-stage-1_end.pth", "step-00000020-tiny.pth"]
    _assert_same_model_and_optimizer(base_run.final_checkpoint(), resumed.final_checkpoint())


def test_a_one_rank_resume_of_a_two_rank_checkpoint_is_refused(base_run: Launch, tmp_path: Path, tiny_dataset_dir: Path) -> None:
    middle = checkpoint_dir(base_run.run_directory) / "step-00000010-tiny.pth"
    yaml_path, run_directory = two_rank_yaml(
        tmp_path / "one_rank", tiny_dataset_dir, backend="single_device", resume=True, resume_checkpoint_path=str(middle)
    )
    result = single_device_launch(yaml_path, run_directory)
    assert result.returncode == 1
    assert "written by a run with 2 rank(s), this run has 1: resume with the same number of ranks" in result.stderr
    assert not (run_directory / TRAIN_REPORT_NAME).exists()


def test_rank_one_says_nothing_below_warning(base_run: Launch) -> None:
    """
    torchrun interleaves both ranks' stderr: rank 0's INFO lines carry no prefix, and every `[rank 1]` line is a
    WARNING or worse (there should be none in a healthy run).
    """

    lines = base_run.stderr.splitlines()
    assert any(" INFO training.logger: Total training steps: 20 (1 micro-batches each)" in line for line in lines), base_run.stderr[-3000:]
    rank_one = [line for line in lines if line.startswith("[rank 1]")]
    assert all(" WARNING " in line or " ERROR " in line or " CRITICAL " in line for line in rank_one), rank_one
    assert not any("[rank 1]" in line and " INFO " in line for line in lines)


def _assert_same_model_and_optimizer(expected: dict[str, object], actual: dict[str, object]) -> None:
    expected_model, actual_model = expected["model"], actual["model"]
    assert isinstance(expected_model, dict) and isinstance(actual_model, dict)
    assert expected_model.keys() == actual_model.keys()
    for name, tensor in expected_model.items():
        assert torch.equal(tensor, actual_model[name]), name
    expected_optimizer, actual_optimizer = expected["optimizer"], actual["optimizer"]
    assert isinstance(expected_optimizer, dict) and isinstance(actual_optimizer, dict)
    for index, entry in expected_optimizer["state"].items():
        for key, value in entry.items():
            assert torch.equal(value, actual_optimizer["state"][index][key]), (index, key)
