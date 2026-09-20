# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Real CPU/Gloo workers; no mock collectives and bounded launcher lifetimes."""
from contextlib import suppress
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch


@pytest.mark.slow
@pytest.mark.parametrize("world,mode", [(2, "normal"), (3, "normal"), (2, "fail"), (2, "taskfail")])
def test_real_distributed_inference(tmp_path: Path, tiny_tokenizer_dir: Path, world: int, mode: str) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
               "-m", "training.testing.inference_worker", str(tmp_path), str(tiny_tokenizer_dir), mode]
    # Capture only to disk: a failed launcher must not fill the parent's RAM with repeated traces.
    log = tmp_path / "workers.log"
    returncode = run_launcher(command, environment, log, 120)
    text = log.read_text()
    if mode in ("fail", "taskfail"):
        message = "injected inference worker failure" if mode == "fail" else "injected rank-zero task preparation failure"
        assert returncode != 0 and message in text, text[-10000:]
    else:
        assert returncode == 0, text[-16000:]
        assert len(list(tmp_path.glob("rank-*.json"))) == world


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("world", [2, 8])
def test_real_nccl_inference(tmp_path: Path, tiny_tokenizer_dir: Path, world: int) -> None:
    if not os.environ.get("RUN_DISTRIBUTED_GPU_TESTS") or torch.cuda.device_count() < world:
        pytest.skip(f"opt in with RUN_DISTRIBUTED_GPU_TESTS=1 on {world} free GPUs")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
               "-m", "training.testing.inference_worker", str(tmp_path), str(tiny_tokenizer_dir), "normal", "cuda"]
    log = tmp_path / "workers.log"
    returncode = run_launcher(command, dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"), log, 180)
    assert returncode == 0, log.read_text()[-16000:]


def run_launcher(command: list[str], environment: dict[str, str], log: Path, timeout: int) -> int:
    with log.open("w") as output:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            # Kill this test's entire process group, including workers blocked in a collective.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise
