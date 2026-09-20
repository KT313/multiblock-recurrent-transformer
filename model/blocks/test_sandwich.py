# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.blocks.sandwich.SandwichBlock`.
"""

import copy
from typing import Any

import pytest
import torch
from torch import Tensor

from model.blocks.sandwich import SandwichBlock
from model.config import RecurrentConfig
from model.test_config import tiny_config
from model.layers.attention import precompute_freqs_cis
from model.layers.mlp import GatedMLP
from model.layers.norms import RMSNorm


def make_block(*, bf16_stream: bool = False, **overrides: Any) -> tuple[SandwichBlock, RecurrentConfig, Tensor]:
    cfg = tiny_config(**overrides)
    torch.manual_seed(0)
    block = SandwichBlock(cfg, bf16_stream=bf16_stream)
    freqs = precompute_freqs_cis(cfg.head_size, cfg.model_max_sequence_length, cfg.rope_settings.rope_base)
    return block, cfg, freqs


def test_structure() -> None:
    block, cfg, _ = make_block()
    assert all(isinstance(getattr(block, f"norm_{i}"), RMSNorm) for i in (1, 2, 3, 4))
    assert isinstance(block.mlp, GatedMLP)
    assert all(n.eps == cfg.norm_eps for n in (block.norm_1, block.norm_2, block.norm_3, block.norm_4))


def test_shape_and_output_is_normalized() -> None:
    block, cfg, freqs = make_block()
    x = torch.randn(2, 9, cfg.n_embd) * 5
    y = block(x, freqs[:, :9])
    assert y.shape == x.shape
    # the last op is norm_4 with unit weight: every output vector has unit RMS
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(2, 9), atol=1e-4, rtol=0)


def test_matches_hand_composed_sandwich() -> None:
    block, cfg, freqs = make_block()
    x = torch.randn(1, 5, cfg.n_embd)
    h = block.norm_2(block.attn(block.norm_1(x), freqs[:, :5]) + x)
    expected = block.norm_4(block.mlp(block.norm_3(h)) + h)
    torch.testing.assert_close(block(x, freqs[:, :5]), expected)


@pytest.mark.parametrize('bf16,bf16_stream', [(False, False), (True, False), (True, True)])
def test_scaled_sandwich_matches_explicit_outputs_and_gradients(bf16: bool, bf16_stream: bool) -> None:
    block, cfg, freqs = make_block(residual_scaling='inverse_sqrt_depth', bf16_stream=bf16_stream)
    reference = copy.deepcopy(block)
    x = torch.randn(1, 5, cfg.n_embd, dtype=torch.bfloat16 if bf16_stream else torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    alpha = 1 / cfg.effective_expected_depth**.5
    upstream = torch.randn_like(x)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        y = block(x, freqs[:, :5])
        u = reference.norm_2(x_ref + alpha*reference.attn(reference.norm_1(x_ref), freqs[:, :5]))
        expected = reference.norm_4(u + alpha*reference.mlp(reference.norm_3(u)))
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    y.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
    for actual, ref in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, ref.grad, rtol=0, atol=0)


def test_scaled_sandwich_aot_eager_forward_and_backward() -> None:
    block, cfg, freqs = make_block(residual_scaling='inverse_sqrt_depth')
    reference = copy.deepcopy(block)
    compiled = torch.compile(block, backend='aot_eager', fullgraph=True)
    x = torch.randn(1, 4, cfg.n_embd)
    actual = compiled(x, freqs[:, :4])
    expected = reference(x, freqs[:, :4])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(x)
    actual.backward(upstream)
    expected.backward(upstream)
    for parameter, ref in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref.grad, atol=1e-6, rtol=1e-5)


def test_causality() -> None:
    block, cfg, freqs = make_block()
    S, t = 10, 4
    x = torch.randn(1, S, cfg.n_embd)
    x2 = x.clone()
    x2[:, t] += 1.0
    y, y2 = block(x, freqs[:, :S]), block(x2, freqs[:, :S])
    torch.testing.assert_close(y[:, :t], y2[:, :t])
    assert not torch.allclose(y[:, t:], y2[:, t:])


def test_all_parameters_receive_gradient() -> None:
    block, cfg, freqs = make_block()
    x = torch.randn(2, 6, cfg.n_embd)
    block(x, freqs[:, :6]).pow(2).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_mask_argument_is_forwarded_to_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The block passes its `mask` argument through to attention (a sentinel object is used so a dropped argument
    cannot be confused with the default None).
    """

    block, cfg, freqs = make_block()
    seen: list[Tensor | None] = []
    orig = block.attn.forward
    sentinel = torch.zeros(4, 4, dtype=torch.bool)

    def spy(x: Tensor, freqs_cis: Tensor, mask: Tensor | None = None) -> Tensor:
        seen.append(mask)
        return orig(x, freqs_cis, None)

    monkeypatch.setattr(block.attn, "forward", spy)
    block(torch.randn(1, 4, cfg.n_embd), freqs[:, :4], sentinel)
    assert len(seen) == 1 and seen[0] is sentinel
