# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
RMS normalization (upstream `RMSNorm_llama`): the statistics are always computed in float32.

`autocast_output` is the bf16-residual-stream switch (`RecurrentConfig.bf16_residual_stream`): the sandwich block
computes `norm(sublayer(norm(x)) + x)`, so the norm output's dtype is the dtype of the residual stream. By default a
bf16 input times the fp32 weight promotes to fp32 and the stream between layers stays fp32. With the switch the
norm rounds its output to the autocast dtype whenever autocast is active, so the residual adds and the next norm
read and write bf16 (the GEMMs already did under autocast). Without autocast nothing changes: a bf16 output would
meet fp32 GEMM weights there, and the fp32 golden numerics stay as they are.
"""

import torch


class RMSNorm(torch.nn.Module):
    """
    `x / rms(x) * weight` over the last dimension, with `rms(x) = sqrt(mean(x^2) + eps)`.
    """

    def __init__(self, dim: int, eps: float = 1e-6, autocast_output: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.autocast_output = autocast_output
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        The normalization without the learned weight.
        """

        return rms_normalize(x, self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps, self.autocast_output)

    def residual(self, branch: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply this norm after a residual addition, through the shared functional operation."""
        return residual_norm(self, branch, residual)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, autocast_output: bool = False) -> torch.Tensor:
    """
    Functional RMSNorm, including the residual-stream rounding contract. Kernel experiments replace this operation.
    """

    device_type = x.device.type
    round_to = autocast_output and torch.is_autocast_enabled(device_type)
    with torch.autocast(enabled=False, device_type=device_type):
        value = x.float()
        normalized = rms_normalize(value, eps)
        if round_to:
            return (normalized * weight).to(torch.get_autocast_dtype(device_type))
        return normalized.type_as(x) * weight


def residual_norm(norm: RMSNorm, branch: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """
    The sandwich post-norm: add the branch to the residual, then apply the original norm module.
    """

    result: torch.Tensor = norm(branch + residual)
    return result


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize in the supplied dtype. The weighted operation selects fp32 before calling this."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
