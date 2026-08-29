# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Transformer layer used in prelude, core blocks and coda."""

import torch
from torch import Tensor

from .attention import CausalSelfAttention
from .config import RecurrentConfig
from .mlp import GatedMLP
from .norms import RMSNorm


class SandwichBlock(torch.nn.Module):
    """Pre- and post-norm around both the attention and the MLP sub-layer."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config)
        self.norm_3 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.norm_2(self.attn(self.norm_1(x), freqs_cis, mask) + x)
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x
