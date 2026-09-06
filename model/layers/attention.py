# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Causal self-attention (fused qkv projection, optional q/k bias, RoPE) on top of `scaled_dot_product_attention`, or
of FlexAttention when the batch carries a document `BlockMask` (packed sequences).

Tensor shape names used throughout: B = batch, S = sequence length, E = n_embd, nh = number of heads,
hd = head dimension (E == nh * hd).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union, cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask, flex_attention

from .init import Linear

if TYPE_CHECKING:
    from ..config import RecurrentConfig

# What an attention layer accepts as its mask: None (plain causal), a broadcastable bool tensor that already contains
# the causal triangle (sdpa's explicit-mask path), or a FlexAttention `BlockMask` (packed sequences on CUDA).
AttentionMask = Union[Tensor, BlockMask, None]

_compiled_flex_attention: Callable[..., Tensor] | None = None
_compiled_create_block_mask: Callable[..., BlockMask] | None = None


def precompute_freqs_cis(dim: int, end: int, theta: float) -> Tensor:
    """
    RoPE table for head dimension `dim` and positions `0 .. end-1`: entry `[0, m, 0, j]` is
    `(cos(m * theta_j), sin(m * theta_j))` with `theta_j = theta ** (-2j / dim)`. Shape (1, end, 1, dim // 2, 2),
    always float32; the singleton axes broadcast against (B, S, nh, hd // 2, 2).
    """

    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))  # (dim // 2,)
        positions = torch.arange(end, dtype=torch.float32, device=inv_freqs.device)  # (end,)
        angles = torch.outer(positions, inv_freqs).float()  # (end, dim // 2)
        cos = torch.cos(angles)[None, :, None, :]
        sin = torch.sin(angles)[None, :, None, :]
        return torch.stack([cos, sin], dim=4)


def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Rotate q and k (B, S, nh, hd) by RoPE in float32 and cast back to the input dtype.

    Adjacent feature pairs `(x[2j], x[2j+1])` are treated as complex numbers and multiplied by `e^(i * m * theta_j)`
    (https://github.com/t-vi/lit-llama/blob/9e5eb8b1b376d8ae24e79278008b7190961062e3/lit_llama/model.py).
    """

    with torch.autocast("cuda", enabled=False):
        # q and k are rotated in one go: concatenated along the head axis, features split into pairs.
        qk_pairs = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()  # type: ignore[no-untyped-call]  # torch stub gap
        real, imag = qk_pairs[..., 0], qk_pairs[..., 1]  # (B, S, 2 * nh, hd // 2)
        cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]  # (1, S, 1, hd // 2)
        rotated = torch.stack([real * cos - imag * sin, imag * cos + real * sin], -1)
        rotated = rotated.flatten(3).type_as(q)  # pairs back to (B, S, 2 * nh, hd)
        q_out, k_out = torch.split(rotated, q.shape[2], dim=2)
        return q_out, k_out


def attention_sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """
    Causal attention; inputs and output are (B, S, nh, hd).

    Without `mask`, sdpa's own `is_causal=True` applies the causal triangle (the training path). A `mask` must be a
    broadcastable bool mask that already contains the causal triangle (`prepare_attention_inputs` builds one); it is
    passed with `is_causal=False` because some sdpa backends reject an explicit mask together with `is_causal=True`.
    """

    # scaled_dot_product_attention wants the head axis before the sequence axis: (B, nh, S, hd).
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None
    )
    return y.transpose(1, 2)


def _flex_attention_kernel(device: torch.device) -> Callable[..., Tensor]:
    """
    `flex_attention` as called from an uncompiled model: compiled once on CUDA (eager FlexAttention on CUDA runs a
    slow fallback), plain on other devices. Inside a `torch.compile`d forward the call is traced either way.
    """

    global _compiled_flex_attention
    if device.type != "cuda" or torch.compiler.is_dynamo_compiling():
        return flex_attention
    if _compiled_flex_attention is None:
        # torch.compile is typed after the overloaded flex_attention; the call below uses the plain-Tensor overload
        _compiled_flex_attention = cast(Callable[..., Tensor], torch.compile(flex_attention))
    return _compiled_flex_attention


def attention_flex(q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask) -> Tensor:
    """
    Attention under a FlexAttention `BlockMask`; inputs and output are (B, S, nh, hd) like `attention_sdpa`.

    The block mask carries the whole pattern (causal within a document, nothing across documents, see
    `document_attention_mask`); blocks that are fully masked are skipped, so a packed row costs about what its
    documents would cost on their own.
    """

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y: Tensor = _flex_attention_kernel(q.device)(q, k, v, block_mask=block_mask)
    return y.transpose(1, 2)


def attention(q: Tensor, k: Tensor, v: Tensor, mask: AttentionMask = None) -> Tensor:
    """
    `attention_flex` for a `BlockMask`, `attention_sdpa` otherwise (a bool mask or None).
    """

    if isinstance(mask, BlockMask):
        return attention_flex(q, k, v, mask)
    return attention_sdpa(q, k, v, mask)


@torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
def document_attention_mask(document_ids: Tensor) -> Tensor | BlockMask:
    """
    The attention mask of packed sequences from their `(B, S)` document ids: causal within a document, blocked
    across documents. True means attend; every query attends at least to itself, so no row is fully masked.

    On CUDA a FlexAttention `BlockMask` (block-sparse, built here OUTSIDE the model's forward: `create_block_mask`
    is data-dependent and must not sit inside a compiled or CUDA-graph-captured region). On other devices the same
    pattern as a dense `(B, 1, S, S)` bool mask for sdpa's explicit-mask path, because FlexAttention has no backward
    on the CPU; that dense path is O(S^2) memory and the correctness reference, not a training path.
    """

    if document_ids.dim() != 2:
        raise ValueError(f"document_ids must be (B, S), got shape {tuple(document_ids.shape)}")
    batch_size, sequence_length = document_ids.shape

    def same_document_causal(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return (q_idx >= kv_idx) & (document_ids[b, q_idx] == document_ids[b, kv_idx])

    if document_ids.device.type == "cuda":
        # The compiled builder computes the block grid tile by tile instead of materialising the dense (B, S, S) mask
        # first: measured 640 MiB -> 0.1 MiB and 7.8 ms -> 0.8 ms per call at S = 8192 (the dense build runs out of
        # memory at 32768); a few seconds of compile once per shape, dynamic after the second shape.
        global _compiled_create_block_mask
        if _compiled_create_block_mask is None:
            _compiled_create_block_mask = cast(Callable[..., BlockMask], torch.compile(create_block_mask))
        return _compiled_create_block_mask(
            same_document_causal, batch_size, None, sequence_length, sequence_length, device=document_ids.device
        )
    dense: Tensor = create_mask(
        same_document_causal, batch_size, None, sequence_length, sequence_length, device=document_ids.device
    )
    return dense


class CausalSelfAttention(torch.nn.Module):
    """
    Multi-head causal self-attention: fused qkv projection, optional q/k bias, RoPE, sdpa, output projection.
    """

    __constants__ = ("n_head", "head_dim")

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_size
        self.Wqkv = Linear(config.n_embd, 3 * config.n_embd, bias=False, init_method=config.init.fn("qkv"))
        self.use_qk_bias = config.qk_bias
        if config.qk_bias:
            # One bias vector per head for q and one for k (index 0 / 1 of the first axis), added before RoPE.
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.proj = Linear(config.n_embd, config.n_embd, bias=False, init_method=config.init.fn("out_attn"))

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: AttentionMask = None) -> Tensor:
        B, S, E = x.shape
        q, k, v = self.Wqkv(x).split(E, dim=2)  # each (B, S, E)
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_head, self.head_dim)
        v = v.view(B, S, self.n_head, self.head_dim)
        if self.use_qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)  # type: ignore[no-untyped-call]  # torch stub gap
            # The bias is a float32 parameter; the sum is cast back so q/k keep the activation dtype under autocast.
            q = (q + q_bias).to(q.dtype)
            k = (k + k_bias).to(q.dtype)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        y = attention(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()
        out: Tensor = self.proj(y)
        return out
