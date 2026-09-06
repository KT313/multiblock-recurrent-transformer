# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The building blocks a transformer block is made of: norms, attention, MLP and the parameter initialization.
"""

from .attention import (
    AttentionMask,
    CausalSelfAttention,
    apply_rotary_emb_complex_like,
    attention,
    attention_flex,
    attention_sdpa,
    document_attention_mask,
    precompute_freqs_cis,
)
from .init import Init, InitFn, Linear, init_glu, init_qkv, trunc_orthogonal_, wrapped_trunc_ortho
from .mlp import GatedMLP
from .norms import RMSNorm

__all__ = [
    "AttentionMask",
    "CausalSelfAttention",
    "GatedMLP",
    "Init",
    "InitFn",
    "Linear",
    "RMSNorm",
    "apply_rotary_emb_complex_like",
    "attention",
    "attention_flex",
    "attention_sdpa",
    "document_attention_mask",
    "init_glu",
    "init_qkv",
    "precompute_freqs_cis",
    "trunc_orthogonal_",
    "wrapped_trunc_ortho",
]
