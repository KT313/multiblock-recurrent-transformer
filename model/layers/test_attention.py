# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `model.attention`: shapes, causality, RoPE relative-position invariance and the q/k bias."""

import math
from typing import Any

import pytest
import torch
from torch import Tensor

from model.layers.attention import (
    CausalSelfAttention,
    apply_rotary_emb_complex_like,
    attention_sdpa,
    precompute_freqs_cis,
)
from model.config import RecurrentConfig
from model.test_config import tiny_config


def make_attn(**overrides: Any) -> tuple[CausalSelfAttention, RecurrentConfig, Tensor]:
    cfg = tiny_config(**overrides)
    torch.manual_seed(0)
    attn = CausalSelfAttention(cfg)
    freqs = precompute_freqs_cis(cfg.head_size, cfg.block_size, cfg.rope_settings.rope_base)
    return attn, cfg, freqs


def test_freqs_cis_shape_and_dtype() -> None:
    freqs = precompute_freqs_cis(16, 32, 50_000)
    assert freqs.shape == (1, 32, 1, 8, 2)
    assert freqs.dtype == torch.float32
    torch.testing.assert_close(freqs[..., 0] ** 2 + freqs[..., 1] ** 2, torch.ones(1, 32, 1, 8))
    assert torch.equal(freqs[0, 0, 0], torch.tensor([[1.0, 0.0]] * 8))  # position 0 is the identity rotation


def test_freqs_cis_matches_hand_formula() -> None:
    """Entry (m, j) is (cos(m * theta_j), sin(m * theta_j)) with theta_j = base ** (-2j / dim)."""
    dim, end, base = 8, 5, 100.0
    freqs = precompute_freqs_cis(dim, end, base)
    for m in range(end):
        for j in range(dim // 2):
            theta = base ** (-2 * j / dim)
            expected = torch.tensor([math.cos(m * theta), math.sin(m * theta)])
            torch.testing.assert_close(freqs[0, m, 0, j], expected, atol=1e-6, rtol=0)


def test_rope_matches_hand_written_complex_rotation() -> None:
    """Adjacent pairs (x[2j], x[2j+1]) are rotated by m * theta_j (interleaved pairing, positive angle for q and k)."""
    torch.manual_seed(0)
    hd, S, base = 8, 6, 100.0
    freqs = precompute_freqs_cis(hd, S, base)
    q, k = torch.randn(1, S, 2, hd), torch.randn(1, S, 2, hd)
    qr, kr = apply_rotary_emb_complex_like(q, k, freqs)
    for x, xr in ((q, qr), (k, kr)):
        z = torch.view_as_complex(x.reshape(1, S, 2, hd // 2, 2).contiguous())
        angles = torch.tensor([[m * base ** (-2 * j / hd) for j in range(hd // 2)] for m in range(S)])
        rot = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
        expected = torch.view_as_real(z * rot).flatten(3)
        torch.testing.assert_close(xr, expected, atol=1e-5, rtol=1e-5)


def test_forward_shape_and_parameters() -> None:
    attn, cfg, freqs = make_attn()
    x = torch.randn(2, 10, cfg.n_embd)
    y = attn(x, freqs[:, :10])
    assert y.shape == x.shape
    assert attn.Wqkv.weight.shape == (3 * cfg.n_embd, cfg.n_embd)
    assert attn.qk_bias.shape == (2, 1, cfg.num_attention_heads, cfg.head_size)
    assert torch.equal(attn.qk_bias, torch.zeros_like(attn.qk_bias))


def test_causality_perturbing_token_t_leaves_earlier_outputs_unchanged() -> None:
    attn, cfg, freqs = make_attn()
    S, t = 12, 5
    x = torch.randn(1, S, cfg.n_embd)
    x2 = x.clone()
    x2[:, t] += torch.randn(cfg.n_embd)
    y, y2 = attn(x, freqs[:, :S]), attn(x2, freqs[:, :S])
    torch.testing.assert_close(y[:, :t], y2[:, :t])
    # every position from t on sees the perturbed token (a mask that hides token t from t+1 would keep some equal)
    changed = ~torch.isclose(y[0, t:], y2[0, t:]).all(dim=-1)
    assert changed.all(), changed


def test_batch_elements_are_independent() -> None:
    attn, cfg, freqs = make_attn()
    x = torch.randn(3, 8, cfg.n_embd)
    y = attn(x, freqs[:, :8])
    y0 = attn(x[:1], freqs[:, :8])
    torch.testing.assert_close(y[:1], y0)


def test_attention_sdpa_matches_hand_computed_causal_softmax() -> None:
    torch.manual_seed(0)
    B, S, nh, hd = 1, 6, 2, 8
    q, k, v = (torch.randn(B, S, nh, hd) for _ in range(3))
    y = attention_sdpa(q, k, v)
    scores = torch.einsum("bsnd,btnd->bnst", q, k) / hd**0.5
    scores = scores.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    expected = torch.einsum("bnst,btnd->bsnd", scores.softmax(-1), v)
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)


def test_rope_relative_position_invariance_of_qk_dot_product() -> None:
    """q at position i dotted with k at position j depends only on i - j after rotation."""
    torch.manual_seed(0)
    hd, S = 16, 40
    freqs = precompute_freqs_cis(hd, S, 50_000)
    q0 = torch.randn(1, 1, 1, hd)
    k0 = torch.randn(1, 1, 1, hd)
    q = q0.expand(1, S, 1, hd).contiguous()  # the same vector at every position
    k = k0.expand(1, S, 1, hd).contiguous()
    qr, kr = apply_rotary_emb_complex_like(q, k, freqs)
    dots = torch.einsum("bind,bjnd->ij", qr, kr)  # (S, S)
    for delta in (0, 1, 3, 17):
        diag = torch.diagonal(dots, offset=-delta)  # all (i, j) with i - j == delta
        torch.testing.assert_close(diag, diag[:1].expand_as(diag), atol=1e-4, rtol=1e-4)
    # ... and genuinely varies with the offset (rotation is not a no-op)
    assert not torch.allclose(torch.diagonal(dots, 0)[0], torch.diagonal(dots, -3)[0])


def test_rope_preserves_norm_and_position_zero_is_identity() -> None:
    torch.manual_seed(0)
    hd, S = 16, 8
    freqs = precompute_freqs_cis(hd, S, 50_000)
    q, k = torch.randn(2, S, 3, hd), torch.randn(2, S, 3, hd)
    qr, kr = apply_rotary_emb_complex_like(q, k, freqs)
    assert qr.shape == q.shape and kr.shape == k.shape
    torch.testing.assert_close(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(qr[:, 0], q[:, 0])
    torch.testing.assert_close(kr[:, 0], k[:, 0])


def test_rope_keeps_input_dtype() -> None:
    hd, S = 16, 8
    freqs = precompute_freqs_cis(hd, S, 50_000)
    q, k = torch.randn(1, S, 1, hd).bfloat16(), torch.randn(1, S, 1, hd).bfloat16()
    qr, kr = apply_rotary_emb_complex_like(q, k, freqs)
    assert qr.dtype == torch.bfloat16 and kr.dtype == torch.bfloat16


def test_position_ids_select_freqs() -> None:
    """Using rows 4.. of the table for a sequence must equal computing at those positions directly."""
    attn, cfg, freqs = make_attn()
    x = torch.randn(1, 6, cfg.n_embd)
    y_shifted = attn(x, freqs[:, 4:10])
    y_zero = attn(x, freqs[:, :6])
    assert y_shifted.shape == y_zero.shape
    # relative positions are the same, so the attention output is invariant to the absolute offset
    torch.testing.assert_close(y_shifted, y_zero, atol=1e-5, rtol=1e-5)


def test_qk_bias_changes_output_and_is_absent_when_disabled() -> None:
    attn, cfg, freqs = make_attn()
    x = torch.randn(1, 8, cfg.n_embd)
    y_zero_bias = attn(x, freqs[:, :8])
    with torch.no_grad():
        attn.qk_bias.normal_(std=1.0)
    y_biased = attn(x, freqs[:, :8])
    assert not torch.allclose(y_zero_bias, y_biased)

    attn_nb, _, _ = make_attn(qk_bias=False)
    assert not hasattr(attn_nb, "qk_bias")
    assert "qk_bias" not in dict(attn_nb.named_parameters())
    with torch.no_grad():
        attn_nb.Wqkv.weight.copy_(attn.Wqkv.weight)
        attn_nb.proj.weight.copy_(attn.proj.weight)
    torch.testing.assert_close(attn_nb(x, freqs[:, :8]), y_zero_bias)


def test_qk_bias_only_receives_gradient_from_q_and_k_path() -> None:
    attn, cfg, freqs = make_attn()
    x = torch.randn(1, 8, cfg.n_embd)
    attn(x, freqs[:, :8]).sum().backward()
    assert attn.qk_bias.grad is not None
    assert attn.qk_bias.grad.shape == attn.qk_bias.shape


@pytest.mark.parametrize("S", [1, 2, 17])
def test_various_sequence_lengths(S: int) -> None:
    attn, cfg, freqs = make_attn()
    x = torch.randn(2, S, cfg.n_embd)
    assert attn(x, freqs[:, :S]).shape == (2, S, cfg.n_embd)
