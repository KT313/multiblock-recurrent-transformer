# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Transformer layer used in prelude, core blocks and coda.
"""

import torch
from torch import Tensor

from ..config import RecurrentConfig
from ..generation import KVCache
from ..layers.attention import AttentionMask, CausalSelfAttention
from ..layers.mlp import GatedMLP
from ..layers.norms import RMSNorm


class SandwichBlock(torch.nn.Module):
    """
    Attention and MLP sub-layer, each "sandwiched" between a pre-norm and a post-norm:

        x = norm_2(alpha * attn(norm_1(x)) + x)
        x = norm_4(alpha * mlp(norm_3(x)) + x)

    alpha is 1 by default, or 1/sqrt(config.effective_expected_depth) with inverse_sqrt_depth scaling. It is a
    fixed float, not a parameter, and scales only the branch before residual addition and the outer norm.

    `bf16_stream` makes the four norms round their output to the autocast dtype (see `RMSNorm`), which makes the
    residual stream through this block bf16 under autocast; `RecurrentGPT` sets it per `bf16_residual_stream`.
    """

    def __init__(self, config: RecurrentConfig, bf16_stream: bool = False) -> None:
        super().__init__()
        self.residual_scale = config.residual_scale
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps, autocast_output=bf16_stream)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps, autocast_output=bf16_stream)
        self.mlp = GatedMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps, autocast_output=bf16_stream)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps, autocast_output=bf16_stream)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: AttentionMask = None, cache: KVCache | None = None,
    ) -> Tensor:
        if cache is None:
            attn_out = self.attn(self.norm_1(x), freqs_cis, mask)
        else:
            attn_out = self.attn(self.norm_1(x), freqs_cis, mask, cache)
        if self.residual_scale != 1.0:
            attn_out = attn_out * self.residual_scale
        x = self.norm_2.residual(attn_out, x)
        mlp_out = self.mlp(self.norm_3(x))
        if self.residual_scale != 1.0:
            mlp_out = mlp_out * self.residual_scale
        x = self.norm_4.residual(mlp_out, x)
        return x
