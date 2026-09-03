# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""RMS normalization (upstream `RMSNorm_llama`): the statistics are always computed in float32."""

import torch


class RMSNorm(torch.nn.Module):
    """`x / rms(x) * weight` over the last dimension, with `rms(x) = sqrt(mean(x^2) + eps)`."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """The normalization without the learned weight."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Statistics in float32 (autocast off: a half-precision `x^2` would underflow), cast back, then the weight.
        # The op order is part of the reference numerics.
        with torch.autocast(enabled=False, device_type=x.device.type):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
