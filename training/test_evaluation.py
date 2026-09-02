# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `training.evaluation`: the per-depth validation loss (batch-major, averaged over the batches actually
delivered), the empty-loader error and the evaluation-step rule."""

from typing import Any

import pytest
import torch

from model import RecurrentGPT
from training.backend.single_device import SingleDeviceBackend
from training.data.collate import Batch
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


def _batches(count: int) -> list[Batch]:
    """`count` fixed validation micro-batches of two rows each."""
    return [(torch.randint(1, 512, (2, 16)), torch.randint(1, 512, (2, 16)), ["v", "v"]) for _ in range(count)]


def test_evaluate_reports_every_depth(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.partial_depth_eval = [1, 3]
    settings.eval_iters = 2
    torch.manual_seed(0)
    batches = _batches(5)
    seen: list[tuple[bool, Any]] = []
    forward = RecurrentGPT.forward

    def spy(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        seen.append((self.training, kwargs.get("num_steps_pair")))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(RecurrentGPT, "forward", spy)
    torch.manual_seed(1)  # the latent state is drawn from the global RNG; batch 1 is scored at every depth first
    metrics = evaluate(settings, cpu_backend, tiny_model, batches)
    expected = {"val_loss", "val_ppl"} | {f"val_{k}_{d}" for k in ("loss", "ppl") for d in (1, 3, "[2, 2]")}
    assert set(metrics) == expected
    assert all(torch.isfinite(v) for v in metrics.values())
    assert metrics["val_loss"] == metrics["val_loss_[2, 2]"]
    assert torch.allclose(metrics["val_ppl_1"], metrics["val_loss_1"].exp())
    assert tiny_model.training  # restored to train mode
    # CHANGED (was depth-major, `eval_iters` forwards per depth in a row): the loop is now batch-major, every depth
    # scoring the batch in hand before the next batch is fetched, still one (depth, 0) pair per core block
    per_batch = [(False, [(1, 0), (1, 0)]), (False, [(3, 0), (3, 0)]), (False, [(2, 0), (2, 0)])]
    assert seen == per_batch * 2  # `eval_iters` = 2 batches
    # the depth actually changes the computation
    assert metrics["val_loss_1"] != metrics["val_loss_3"] != metrics["val_loss_[2, 2]"]
    # each depth is the mean over the eval_iters batches (replaying the same RNG stream batch by batch)
    with torch.no_grad():
        torch.manual_seed(1)
        tiny_model.eval()
        replay = [
            [tiny_model(x, labels=y, num_steps_pair=steps)["loss"] for _, steps in per_batch] for x, y, _ in batches[:2]
        ]
    for depth_idx, depth in enumerate((1, 3, "[2, 2]")):
        expected_loss = torch.stack([losses[depth_idx] for losses in replay]).mean().item()
        assert metrics[f"val_loss_{depth}"].item() == pytest.approx(expected_loss, rel=1e-6)


def test_evaluate_averages_the_batches_actually_delivered(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """A validation loader shorter than `eval_iters` (the tiny config's finetune split is one micro-batch) is
    averaged over the batches it delivered, not over the planned `eval_iters` — the bug this replaces divided by
    `eval_iters` and reported a loss scaled down by the missing rows."""
    settings.partial_depth_eval = [1]
    settings.eval_iters = 4
    torch.manual_seed(0)
    batches = _batches(3)  # fewer than eval_iters
    torch.manual_seed(1)
    metrics = evaluate(settings, cpu_backend, tiny_model, batches)
    with torch.no_grad():
        torch.manual_seed(1)
        tiny_model.eval()
        replay = [
            [tiny_model(x, labels=y, num_steps_pair=[(d, 0), (d, 0)])["loss"] for d in (1, 2)] for x, y, _ in batches
        ]
    tiny_model.train()
    for depth_idx, depth in enumerate((1, "[2, 2]")):
        total = sum(losses[depth_idx].item() for losses in replay)
        assert metrics[f"val_loss_{depth}"].item() == pytest.approx(total / len(batches), rel=1e-6)
        assert metrics[f"val_loss_{depth}"].item() != pytest.approx(total / settings.eval_iters, rel=1e-3)  # the bug
    assert metrics["val_loss"].item() > 1.0  # a plausible loss, not one scaled down by the batches that never came


def test_evaluate_on_an_empty_loader_raises(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """No batch at all is an error naming the situation — never a NaN or a silent zero."""
    settings.partial_depth_eval = [1]
    settings.eval_iters = 2
    with pytest.raises(RuntimeError, match="validation loader yielded no batch"):
        evaluate(settings, cpu_backend, tiny_model, [])
    assert tiny_model.training


def test_evaluate_scores_every_depth_on_the_same_batches(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paired comparison: the depths are compared on exactly the same batches, and every batch is taken from
    the loader once (the recording loader hands out distinguishable batches)."""
    settings.partial_depth_eval = [1, 3]
    settings.eval_iters = 3
    torch.manual_seed(0)
    batches = _batches(5)
    delivered: list[int] = []
    scored: dict[str, list[int]] = {}
    forward = RecurrentGPT.forward

    class RecordingLoader:
        def __iter__(self) -> Any:
            for index, batch in enumerate(batches):
                delivered.append(index)
                yield batch

    def spy(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        input_ids = args[0] if args else kwargs["input_ids"]
        index = next(i for i, (x, _, _) in enumerate(batches) if torch.equal(x, input_ids))
        scored.setdefault(str(kwargs.get("num_steps_pair")), []).append(index)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(RecurrentGPT, "forward", spy)
    evaluate(settings, cpu_backend, tiny_model, RecordingLoader())
    assert delivered == [0, 1, 2]  # `eval_iters` batches, each pulled from the loader exactly once
    assert len(scored) == 3  # two partial depths plus the mean recurrence
    assert all(indices == [0, 1, 2] for indices in scored.values()), scored


def test_evaluate_iterates_the_loader_once(
    tiny_model: RecurrentGPT, settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """CHANGED (was one `iter()` per depth): the batch-major loop creates exactly ONE iterator per evaluation.
    Numerics: each `iter()` of a DataLoader draws a base seed from the global torch RNG, so the number of
    iterations is part of the RNG stream (`training/golden_tiny_run.json` was re-recorded for it)."""
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
    assert iterations == 1  # depth 1 and the mean recurrence share the one pass over the loader


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
