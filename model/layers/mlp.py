# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Gated (SwiGLU) MLP with a fused gate/up projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .init import Linear
from ..kernels.runtime import load_mlp

if TYPE_CHECKING:
    from ..config import RecurrentConfig


class GatedMLP(torch.nn.Module):
    """
    `proj(silu(gate(x)) * up(x))`; `fc` computes gate and up in one matmul (gate = first half of its rows).
    """

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        intermediate_size = config.intermediate_size
        assert intermediate_size is not None  # filled in by RecurrentConfig.__post_init__
        self.fc = Linear(config.n_embd, intermediate_size * 2, bias=False, init_method=config.init.fn("glu"))
        self.proj = Linear(intermediate_size, config.n_embd, bias=False, init_method=config.init.fn("out_proj"))
        self.nonlin = torch.nn.SiLU()
        self._custom_mlp = load_mlp() if config.use_custom_kernels else None

    def forward(self, x: Tensor) -> Tensor:
        if self._custom_mlp is not None:
            return self._custom_mlp(x, self.fc, self.proj, self.nonlin)
        return mlp_projection(x, self.fc, self.proj, self.nonlin)


def mlp_projection(x: Tensor, fc: torch.nn.Module, proj: torch.nn.Module, nonlin: torch.nn.Module) -> Tensor:
    """The complete gated MLP through the original projection modules; shared experimental integration point."""
    hidden = swiglu(fc(x), nonlin)
    out: Tensor = proj(hidden)
    return out


def swiglu(hidden: Tensor, nonlin: torch.nn.Module) -> Tensor:
    """
    Gate/up activation using the module's original nonlinearity, shared with component measurements.
    """

    gate, up = hidden.chunk(2, dim=-1)
    result: Tensor = nonlin(gate) * up
    return result
