# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Causal self-attention (fused qkv projection, optional q/k bias, RoPE) on top of `scaled_dot_product_attention`."""

import torch
from torch import Tensor

from .config import RecurrentConfig
from .init import Linear


def precompute_freqs_cis(dim: int, end: int, theta: float) -> Tensor:
    """RoPE cos/sin table of shape (1, end, 1, dim // 2, 2), always in float32."""
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device)
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)


def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate q and k (B, S, nh, hd) in float32 and cast back to the input dtype."""
    with torch.autocast("cuda", enabled=False):
        # https://github.com/t-vi/lit-llama/blob/9e5eb8b1b376d8ae24e79278008b7190961062e3/lit_llama/model.py
        qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()  # type: ignore[no-untyped-call]  # torch stub gap
        rotated_qk_r2 = torch.stack(
            [
                qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        ).flatten(3)
        q_out, k_out = torch.split(rotated_qk_r2.type_as(q), q.shape[2], dim=2)
        return q_out, k_out


def attention_sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Causal attention; inputs and output are (B, S, nh, hd)."""
    q = q.transpose(1, 2)  # (B, nh, S, hs)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=True)
    return y.transpose(1, 2)


class CausalSelfAttention(torch.nn.Module):
    __constants__ = ("n_head", "head_dim")

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_size
        self.Wqkv = Linear(config.n_embd, 3 * config.n_embd, bias=False, init_method=config.init.fn("qkv"))
        self.use_qk_bias = config.qk_bias
        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.proj = Linear(config.n_embd, config.n_embd, bias=False, init_method=config.init.fn("out_attn"))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None = None) -> Tensor:
        B, S, E = x.shape
        q, k, v = self.Wqkv(x).split(E, dim=2)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)
        if self.use_qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)  # type: ignore[no-untyped-call]  # torch stub gap
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        y = attention_sdpa(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()
        out: Tensor = self.proj(y)
        return out
