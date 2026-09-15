# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Keep the ELLISAdam update mathematics and its CUDA compilation boundary together."""

from math import sqrt
from typing import Callable

import torch
from torch import Tensor

from training.optim.state import dequantized_state, is_quantized_state


def _single_tensor_modded_adamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Tensor | float,
    init_lr: Tensor | float,
    weight_decay: float,
    eps: float,
    update_clipping: bool = False,
    atan_adam: bool = False,
    decouple_wd: bool = False,
) -> None:
    """
    One ELLISAdam step for one parameter group (AdamW plus the ELLIS extras: update clipping, the atan update,
    decoupled weight decay).

    Two halves. First the scalars, here on the CPU: each parameter's step counter, the bias corrections, the
    effective step size and the weight-decay factor. Then the tensor math for all parameters in
    `_adamw_group_update`, compiled on CUDA and eager on CPU. The scalars travel as 0-d CPU tensors, so the
    compiled graph takes them as inputs and a changed learning rate or step count never recompiles.
    """

    if not params:
        return
    step_sizes: list[Tensor] = []
    bias_corrections1: list[Tensor] = []
    bias_corrections2_sqrt: list[Tensor] = []
    decays: list[Tensor] = []
    for step_t in state_steps:
        step_t += 1
        bias_correction1 = 1 - beta1**step_t
        bias_correction2 = 1 - beta2**step_t
        bias_correction2_sqrt = sqrt(bias_correction2)
        # 0-d CPU tensors: scalars to eager kernels, plain inputs to the compiled update (no recompile per step)
        step_sizes.append(torch.as_tensor(lr, dtype=torch.float32))
        bias_corrections1.append(bias_correction1)
        bias_corrections2_sqrt.append(torch.as_tensor(bias_correction2_sqrt, dtype=torch.float32))
        decay = 1 - lr / init_lr * weight_decay if decouple_wd else 1 - lr * weight_decay
        decays.append(torch.as_tensor(decay, dtype=torch.float32))
    update = _compiled_adamw_group_update() if params[0].is_cuda else _adamw_group_update
    update(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        step_sizes,
        bias_corrections1,
        bias_corrections2_sqrt,
        decays,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        update_clipping=update_clipping,
        atan_adam=atan_adam,
    )


def _adamw_group_update(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    step_sizes: list[Tensor],
    bias_corrections1: list[Tensor],
    bias_corrections2_sqrt: list[Tensor],
    decays: list[Tensor],
    *,
    beta1: float,
    beta2: float,
    eps: float,
    update_clipping: bool,
    atan_adam: bool,
) -> None:
    """
    The tensor math of the ELLISAdam step, one parameter after the other, with the per-parameter scalars prepared
    by `_single_tensor_modded_adamw`.

    Per parameter: update the two moments, optionally shrink the step by the RMS of `grad / sqrt(exp_avg_sq)`
    (`update_clipping`), apply the decoupled weight decay, then subtract the update: `atan2` of the moments
    (`atan_adam`) or the usual Adam quotient. `param.sub_(update * step_size)` rather than `add_(alpha=...)` on
    purpose: a tensor `alpha` is read back to the host, one sync per parameter. On CUDA this runs compiled
    (`_compiled_adamw_group_update`), so each parameter's chain of elementwise ops becomes two or three kernels.

    8-bit moments (ELLISAdam8bit) are dequantised into fp32 temporaries first and written back at the end, so the
    maths in between is the same for both optimizers; under torch.compile the dequantise, the update and the
    re-quantise fuse into the same few kernels. fp32 moments are updated in place as before.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        quantized = is_quantized_state(exp_avgs[i])
        exp_avg = dequantized_state(exp_avgs[i])
        exp_avg_sq = dequantized_state(exp_avg_sqs[i])
        step_size = step_sizes[i]

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        step_size = step_size / bias_corrections1[i]

        # Perform stepweight decay
        param.mul_(decays[i])
        denom = exp_avg_sq.sqrt().div_(bias_corrections2_sqrt[i])
        if atan_adam:
            param.sub_(torch.atan2(exp_avg, denom) * step_size)
        else:
            param.sub_(exp_avg.div(denom.add_(eps)) * step_size)
        if quantized:
            exp_avgs[i].copy_(exp_avg)
            exp_avg_sqs[i].copy_(exp_avg_sq)


_compiled_group_update: Callable[..., None] | None = None


def _compiled_adamw_group_update() -> Callable[..., None]:
    """
    `_adamw_group_update` under `torch.compile`, built once per process and reused by every step.

    `dynamic=False`: a parameter group's shapes never change during a run, so each group gets one specialized
    graph (three for v2_small). The scalars are tensor inputs, so learning-rate and step changes do not recompile.
    CUDA only: the CPU path (tests, the golden run) calls the plain function.
    """

    global _compiled_group_update
    if _compiled_group_update is None:
        _compiled_group_update = torch.compile(_adamw_group_update, dynamic=False)
    return _compiled_group_update
