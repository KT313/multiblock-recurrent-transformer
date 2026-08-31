# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Gated (SwiGLU) MLP with a fused gate/up projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .init import Linear

if TYPE_CHECKING:
    from ..config import RecurrentConfig


class GatedMLP(torch.nn.Module):
    """`proj(silu(gate(x)) * up(x))`; `fc` computes gate and up in one matmul (gate = first half of its rows)."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        intermediate_size = config.intermediate_size
        assert intermediate_size is not None  # filled in by RecurrentConfig.__post_init__
        self.fc = Linear(config.n_embd, intermediate_size * 2, bias=False, init_method=config.init.fn("glu"))
        self.proj = Linear(intermediate_size, config.n_embd, bias=False, init_method=config.init.fn("out_proj"))
        self.nonlin = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.fc(x).chunk(2, dim=-1)  # each (..., intermediate_size)
        hidden = self.nonlin(gate) * up
        out: Tensor = self.proj(hidden)
        return out
