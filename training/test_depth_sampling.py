# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Actual two-rank CPU accumulation verifies the rank-invariant local depth schedule."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from training.backend.ddp import DDPBackend
import training.backend.ddp as ddp_module
from training.test_step import fresh_optimizer, fresh_tiny_model, reference_settings, run_steps
from training.testing.network import free_port


def _depth_worker(rank: int, port: int, out_path: str) -> None:
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    ddp_module.DDP_TIMEOUT = timedelta(seconds=60)
    backend = DDPBackend(device="cpu", precision="32")
    try:
        backend.seed_everything(42)
        settings = reference_settings(
            micro_batches_per_step=4, training_max_sequence_length=16, tokens_per_micro_batch=16,
        )
        model = fresh_tiny_model(backend, seed=42 + rank)
        plain = backend.plain_model(model)
        calls: list[tuple[int, int, int, int, int]] = []
        original = plain.sample_block_depths

        def record(block_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
            n, k = original(block_idx)
            calls.append((plain.step, plain.micro_batch_index, block_idx, int(n.item()), int(k.item())))
            return n, k

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(plain, "sample_block_depths", record)
            run_steps(settings, backend, model, fresh_optimizer(settings, model, backend), steps=3)
        gathered = backend.all_gather_object(calls)
        assert gathered[0] == gathered[1]
        # DDP's final synchronized backwards must leave matching model parameters.
        parameters = torch.cat([parameter.detach().reshape(-1) for parameter in plain.parameters()])
        gathered_parameters = backend.all_gather_object(parameters)
        assert torch.equal(gathered_parameters[0], gathered_parameters[1])
        if backend.is_main:
            Path(out_path).write_text(json.dumps(gathered))
    finally:
        backend.shutdown()


@pytest.mark.slow
def test_two_rank_accumulation_shares_each_local_depth_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    out_path = tmp_path / "depths.json"
    # torch exposes spawn at runtime, but does not declare the re-export or annotate it.
    mp.spawn(_depth_worker, args=(free_port(), str(out_path)), nprocs=2, join=True)  # type: ignore[attr-defined, no-untyped-call]
    schedules = json.loads(out_path.read_text())
    assert schedules[0] == schedules[1]
    assert [entry[:3] for entry in schedules[0]] == [
        [step, micro, core] for step in range(3) for micro in range(2) for core in range(2)
    ]
    assert len({tuple(entry[3:]) for entry in schedules[0]}) > 1
