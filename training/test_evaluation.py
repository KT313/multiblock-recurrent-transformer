# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `training.evaluation`: the per-depth validation loss and the evaluation-step rule."""

from typing import Any

import pytest
import torch

from model import RecurrentGPT
from training.backend import SingleDeviceBackend
from training.evaluation import evaluate, is_evaluation_step
from training.settings import Settings
from training.stage_manager import StageManager, TrainingStage
from training.step import TrainingProgress
from training.test_step import TINY_MODEL_ARCHITECTURE


@pytest.fixture
def cpu_backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        dataset_config="config/datasets/tiny.yaml",
        model_architecture_config=str(TINY_MODEL_ARCHITECTURE),
        stage_base_lrs=[3e-4],
        block_size=256,
        micro_batch_size=2,
        world_batch_size=4,
        eval_step_interval=8,
    )


def test_evaluate_reports_every_depth(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.partial_depth_eval = [1, 3]
    settings.eval_iters = 2
    torch.manual_seed(0)
    batches = [(torch.randint(1, 512, (2, 16)), torch.randint(1, 512, (2, 16)), ["v", "v"]) for _ in range(5)]
    seen: list[tuple[bool, Any]] = []
    forward = RecurrentGPT.forward

    def spy(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        seen.append((self.training, kwargs.get("num_steps_pair")))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(RecurrentGPT, "forward", spy)
    torch.manual_seed(1)  # the latent state is drawn from the global RNG; depth 1 is evaluated first
    metrics = evaluate(settings, cpu_backend, tiny_model, batches)
    expected = {"val_loss", "val_ppl"} | {f"val_{k}_{d}" for k in ("loss", "ppl") for d in (1, 3, "[2, 2]")}
    assert set(metrics) == expected
    assert all(torch.isfinite(v) for v in metrics.values())
    assert metrics["val_loss"] == metrics["val_loss_[2, 2]"]
    assert torch.allclose(metrics["val_ppl_1"], metrics["val_loss_1"].exp())
    assert tiny_model.training  # restored to train mode
    # every depth is evaluated on exactly `eval_iters` batches in eval mode, as one (depth, 0) pair per core block
    assert seen == [(False, [(1, 0), (1, 0)])] * 2 + [(False, [(3, 0), (3, 0)])] * 2 + [(False, [(2, 0), (2, 0)])] * 2
    # the depth actually changes the computation
    assert metrics["val_loss_1"] != metrics["val_loss_3"] != metrics["val_loss_[2, 2]"]
    # each column is the mean over the eval_iters batches (same RNG stream as the depth-1 column above)
    with torch.no_grad():
        torch.manual_seed(1)
        tiny_model.eval()
        per_batch = [tiny_model(x, labels=y, num_steps_pair=[(1, 0), (1, 0)])["loss"] for x, y, _ in batches[:2]]
    assert metrics["val_loss_1"].item() == pytest.approx(torch.stack(per_batch).mean().item(), rel=1e-6)


def test_evaluate_re_iterates_the_loader_per_depth(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """Every depth iterates the validation loader from its start (numerics: each `iter()` of a DataLoader draws a base
    seed from the global torch RNG, so the number of iterations is part of the RNG stream)."""
    settings.partial_depth_eval = [1]
    settings.eval_iters = 1
    iterations = 0
    batch = (torch.randint(1, 512, (2, 16)), torch.randint(1, 512, (2, 16)), ["v", "v"])

    class CountingLoader:
        def __iter__(self) -> Any:
            nonlocal iterations
            iterations += 1
            yield batch
            yield batch

    evaluate(settings, cpu_backend, tiny_model, CountingLoader())
    assert iterations == 2  # depth 1 and the mean recurrence


def test_is_evaluation_step_table(settings: Settings) -> None:
    """Every `eval_step_interval` completed steps and after the last step (here a 20-step run, interval 8)."""
    stage = TrainingStage("only", tokens=20 * settings.world_batch_size * settings.block_size, base_lr=3e-4, transition_pct=0.0)
    stage_manager = StageManager([stage], settings.world_batch_size, settings.block_size)
    assert stage_manager.total_steps == 20
    evaluated = [done for done in range(1, 21) if is_evaluation_step(settings, TrainingProgress(step=done), stage_manager)]
    assert evaluated == [8, 16, 20]
    settings.eval_step_interval = 7
    evaluated = [done for done in range(1, 21) if is_evaluation_step(settings, TrainingProgress(step=done), stage_manager)]
    assert evaluated == [7, 14, 20]
