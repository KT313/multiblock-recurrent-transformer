# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Transformer layer used in prelude, core blocks and coda.
"""

import torch
from torch import Tensor

from ..config import RecurrentConfig
from ..layers.attention import AttentionMask, CausalSelfAttention
from ..layers.mlp import GatedMLP
from ..layers.norms import RMSNorm


class SandwichBlock(torch.nn.Module):
    """
    Attention and MLP sub-layer, each "sandwiched" between a pre-norm and a post-norm:

        x = norm_2(attn(norm_1(x)) + x)
        x = norm_4(mlp(norm_3(x)) + x)
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: AttentionMask = None) -> Tensor:
        attn_out = self.attn(self.norm_1(x), freqs_cis, mask)
        x = self.norm_2(attn_out + x)
        mlp_out = self.mlp(self.norm_3(x))
        x = self.norm_4(mlp_out + x)
        return x
