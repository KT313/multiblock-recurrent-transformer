# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the disabled logger (no-ops), the stubbed wandb path and the gradient/parameter metric helpers."""

import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT
from training.logger import (
    Logger,
    _qkv_dims,
    _reverse_engineer_adam_effective_lr,
    _to_scalar,
    num_parameters,
    track_gradient_metrics,
)
from training.optim import ELLISAdam, get_param_groups


def test_disabled_logger_is_a_no_op(tmp_path: Path) -> None:
    logger = Logger("proj", "run", tmp_path, enabled=False)
    assert logger.run is None and logger.enabled is False
    logger.log({"loss": torch.tensor(1.0)}, step=1)
    logger.log_hyperparams({"a": 1})
    logger.log_summary({"b": torch.tensor(2)})
    logger.finish()
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_enabled_logger_forwards_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wandb calls are exercised against a stub run so no wandb import/network happens."""
    calls: list[tuple[Any, ...]] = []

    class _Config:
        @staticmethod
        def update(params: dict[str, Any], allow_val_change: bool) -> None:
            calls.append(("config", params, allow_val_change))

    class _Run:
        summary: dict[str, Any] = {}
        config = _Config()

        def log(self, metrics: dict[str, Any], step: int) -> None:
            calls.append(("log", metrics, step))

        def finish(self) -> None:
            calls.append(("finish",))

    class _WandbStub:
        @staticmethod
        def init(**kwargs: Any) -> _Run:
            calls.append(("init", kwargs))
            return _Run()

    monkeypatch.setitem(sys.modules, "wandb", _WandbStub())  # any object works as a module
    logger = Logger("proj", "run", tmp_path / "out", offline=True, enabled=True)
    assert calls[0][1]["mode"] == "offline" and calls[0][1]["project"] == "proj"
    assert calls[0][1]["name"] == "run" and calls[0][1]["dir"] == str(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    logger.log({"loss": torch.tensor(1.5), "n": 3}, step=4)
    assert calls.pop() == ("log", {"loss": 1.5, "n": 3}, 4)
    logger.log_hyperparams({"seed": 1})
    assert calls.pop() == ("config", {"seed": 1}, True)
    logger.log_summary({"p": torch.tensor(7)})
    assert _Run.summary == {"p": 7}
    logger.finish()
    assert calls.pop() == ("finish",) and logger.run is None
    logger.finish()  # idempotent
    assert calls == [("init", calls[0][1])]

    Logger("proj", "run", tmp_path / "online", offline=False, enabled=True)
    assert calls[-1][1]["mode"] == "online"


def test_to_scalar() -> None:
    assert _to_scalar(torch.tensor([2.5])) == 2.5
    assert _to_scalar(torch.tensor(4)) == 4
    assert _to_scalar(3) == 3
    t = torch.zeros(2)
    assert _to_scalar(t) is t


def test_num_parameters_counts_tied_weights_once(tiny_model: RecurrentGPT) -> None:
    total = num_parameters(tiny_model)
    assert total == sum(p.numel() for p in tiny_model.parameters())
    with_duplicates = sum(p.numel() for _, p in tiny_model.named_parameters(remove_duplicate=False))
    assert with_duplicates == total + tiny_model.transformer.wte.weight.numel()  # lm_head is tied
    tiny_model.transformer.wte.weight.requires_grad_(False)
    assert num_parameters(tiny_model, only_trainable=True) == total - tiny_model.transformer.wte.weight.numel()


def test_reverse_engineer_adam_effective_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    param.grad = torch.tensor([0.5, 0.0])  # second element: |g| <= eps branch
    state = {"exp_avg": torch.tensor([0.05, 0.02]), "exp_avg_sq": torch.tensor([0.0025, 0.0004])}
    group = {"eps": 1e-6}
    lr = _reverse_engineer_adam_effective_lr(param, state, group)
    assert lr[0].item() == pytest.approx(0.05 / (0.05 + 1e-6) / 0.5, rel=1e-5)
    assert lr[1].item() == pytest.approx(0.02 / (0.02 + 1e-6) / 1e-6, rel=1e-5)


def test_qkv_dims(tiny_model: RecurrentGPT) -> None:
    assert _qkv_dims(tiny_model) == (64, 64, 64)
    assert _qkv_dims(torch.nn.Linear(2, 2)) is None

    class Partial(torch.nn.Module):
        config = type("Cfg", (), {"n_embd": 8})()

    assert _qkv_dims(Partial()) is None


def _step_tiny(tiny_model: RecurrentGPT) -> ELLISAdam:
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    loss = tiny_model(x, labels=x, num_steps_pair=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    opt.step()
    return opt


N_ATTN_LAYERS = 2 + 2 + 1  # prelude + core blocks (1 layer each) + coda


def test_track_gradient_metrics_on_tiny_model(tiny_model: RecurrentGPT) -> None:
    opt = _step_tiny(tiny_model)
    metrics = track_gradient_metrics(tiny_model, opt)

    for i in range(N_ATTN_LAYERS):
        for key in (
            f"query_grad_{i}",
            f"ffn2_grad_{i}",
            f"q_effective_lr_{i}",
            f"k_effective_lr_{i}",
            f"v_effective_lr_{i}",
            f"ffn2_effective_lr_{i}",
        ):
            assert key in metrics, key
    assert f"ffn2_grad_{N_ATTN_LAYERS}" not in metrics and f"query_grad_{N_ATTN_LAYERS}" not in metrics
    for key in (
        "avg_RMS",
        "embed_RMS",
        "local_l1_grad_norm",
        "l2_param_norm",
        "l1_param_norm",
        "core_block_0_l2_param_norm",
        "core_block_1_l2_param_norm",
        "word_embed_l2_param_norm",
        "model_l2_param_norm",
    ):
        assert key in metrics, key
    for key, value in metrics.items():
        assert torch.is_tensor(value) and value.numel() == 1, key
        assert math.isfinite(value.item()), key

    # hand-checkable values
    expected_l2 = torch.norm(torch.stack([p.norm() for p in tiny_model.parameters()]))
    assert torch.allclose(metrics["l2_param_norm"], expected_l2)
    assert torch.allclose(metrics["word_embed_l2_param_norm"], tiny_model.transformer.wte.weight.norm())
    assert metrics["ffn2_grad_0"] > 0 and metrics["query_grad_0"] > 0
    grad = dict(tiny_model.named_parameters())["transformer.prelude.0.attn.Wqkv.weight"].grad
    assert grad is not None and torch.allclose(metrics["query_grad_0"], grad[:64].norm())
    # right after the first step exp_avg_sq == (1 - beta2) * g^2, so the per-element RMS is 1/sqrt(1 - beta2)
    # (= 10) wherever |g| > eps and 0 where the gradient vanishes: the average is bounded by 10 from above
    assert 5.0 < metrics["avg_RMS"].item() <= 1 / math.sqrt(1 - 0.99) + 1e-4


def test_track_gradient_metrics_without_gradients_or_state(tiny_model: RecurrentGPT) -> None:
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    metrics = track_gradient_metrics(tiny_model, opt)
    assert "avg_RMS" not in metrics and "query_grad_0" not in metrics and "local_l1_grad_norm" not in metrics
    assert math.isfinite(metrics["l2_param_norm"].item())


def test_track_gradient_metrics_on_plain_module() -> None:
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 4)).sum().backward()
    opt.step()
    metrics = track_gradient_metrics(model, opt)
    assert set(metrics) == {"avg_RMS", "local_l1_grad_norm", "l2_param_norm", "l1_param_norm"}


def test_non_finite_gradient_is_reported_as_nan(tiny_model: RecurrentGPT) -> None:
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    loss = tiny_model(x, labels=x, num_steps_pair=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    params = dict(tiny_model.named_parameters())
    proj_grad = params["transformer.prelude.0.mlp.proj.weight"].grad
    qkv_grad = params["transformer.prelude.0.attn.Wqkv.weight"].grad
    assert proj_grad is not None and qkv_grad is not None
    proj_grad[0, 0] = float("inf")
    qkv_grad[0, 0] = float("inf")
    metrics = track_gradient_metrics(tiny_model, opt)
    assert math.isnan(metrics["ffn2_grad_0"].item())
    assert math.isnan(metrics["query_grad_0"].item())
    assert math.isfinite(metrics["ffn2_grad_1"].item())
    assert "ffn2_effective_lr_0" not in metrics  # params with non-finite grads are skipped for effective LRs
