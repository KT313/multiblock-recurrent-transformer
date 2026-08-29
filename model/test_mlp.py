# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `model.mlp.GatedMLP`."""

from typing import Any

import pytest
import torch

from model.config import RecurrentConfig
from model.mlp import GatedMLP


def make_mlp(**overrides: Any) -> tuple[GatedMLP, RecurrentConfig]:
    cfg = RecurrentConfig.from_name("tiny", **overrides)
    torch.manual_seed(0)
    return GatedMLP(cfg), cfg


def test_shapes_and_no_bias() -> None:
    mlp, cfg = make_mlp()
    assert cfg.intermediate_size is not None
    assert mlp.fc.weight.shape == (2 * cfg.intermediate_size, cfg.n_embd)
    assert mlp.proj.weight.shape == (cfg.n_embd, cfg.intermediate_size)
    assert {n for n, _ in mlp.named_parameters()} == {"fc.weight", "proj.weight"}  # no biases
    x = torch.randn(2, 7, cfg.n_embd)
    assert mlp(x).shape == x.shape


def test_matches_hand_computed_swiglu() -> None:
    mlp, cfg = make_mlp()
    x = torch.randn(3, cfg.n_embd)
    w1, w2 = mlp.fc.weight.chunk(2, dim=0)
    expected = (torch.nn.functional.silu(x @ w1.T) * (x @ w2.T)) @ mlp.proj.weight.T
    torch.testing.assert_close(mlp(x), expected)


def test_position_wise() -> None:
    mlp, cfg = make_mlp()
    x = torch.randn(1, 6, cfg.n_embd)
    y = mlp(x)
    x2 = x.clone()
    x2[:, 3] += 1.0
    y2 = mlp(x2)
    keep = torch.tensor([True, True, True, False, True, True])
    torch.testing.assert_close(y[:, keep], y2[:, keep])
    assert not torch.allclose(y[:, 3], y2[:, 3])


def test_silu_is_applied_to_the_first_half_only() -> None:
    """SiLU is not symmetric in its two factors: applying it to the second half instead must give a different result."""
    mlp, cfg = make_mlp()
    x = torch.randn(3, cfg.n_embd)
    w1, w2 = mlp.fc.weight.chunk(2, dim=0)
    swapped = ((x @ w1.T) * torch.nn.functional.silu(x @ w2.T)) @ mlp.proj.weight.T
    assert not torch.allclose(mlp(x), swapped)
    # gate rows zero -> silu(0) = 0 -> exact zero output
    with torch.no_grad():
        mlp.fc.weight[: w1.shape[0]].zero_()
    assert torch.equal(mlp(x), torch.zeros(3, cfg.n_embd))


def test_output_projection_uses_depth_scaled_init() -> None:
    mlp, cfg = make_mlp(n_embd=256, intermediate_size=1024, num_attention_heads=4)
    assert mlp.fc.weight.std().item() == pytest.approx(cfg.init.table["std"], rel=0.1)
    assert mlp.proj.weight.std().item() == pytest.approx(cfg.init.table["out_proj"], rel=0.1)
