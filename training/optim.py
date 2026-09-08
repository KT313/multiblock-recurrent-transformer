# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
The optimizer of the thesis runs, ELLISAdam (a port of the upstream implementation with its four options:
`update_clipping`, `atan_adam`, `running_init`, `decouple_wd`), plus the parameter-group split.
"""

from math import sqrt
from typing import Any, Callable, Iterable, overload

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from training.settings import OptimizerConfig


def get_param_groups(
    model: Module, weight_decay: float, no_wd_for_bias_and_norm: bool = True
) -> list[dict[str, Any]]:
    """
    Split parameters into weights / embeddings / scale-and-norm groups, as upstream did.

    Group order matters for checkpoints: 0 = matrices, 1 = embeddings (+ tied lm_head), 2 = norms and biases.
    """

    weights_group: list[Tensor] = []
    embedding_group: list[Tensor] = []
    scale_and_norm_group: list[Tensor] = []
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if "wte" in name_lower or "embedding" in name_lower or "lm_head" in name_lower:
            embedding_group.append(param)
        elif "ln_f" in name_lower or "norm" in name_lower or "bias" in name_lower:
            scale_and_norm_group.append(param)
        elif "proj" in name_lower or "qkv" in name_lower or "fc" in name_lower or param.ndim == 2:
            weights_group.append(param)
        else:
            raise ValueError(f"param {name} could not be matched to an optim group")

    param_groups = [
        {"params": weights_group, "weight_decay": weight_decay},
        {"params": embedding_group, "weight_decay": weight_decay},
        {"params": scale_and_norm_group, "weight_decay": weight_decay},
    ]
    if no_wd_for_bias_and_norm:
        param_groups[-1]["weight_decay"] = 0.0
    return param_groups


# `OptimizerConfig` fields that are ELLISAdam constructor arguments only; a non-default value with another
# optimizer is a config mistake and fails loudly instead of being dropped.
ELLIS_ONLY_OPTIONS = ("update_clipping", "atan_adam", "running_init", "decouple_wd")

# The `optimizer:` values `build_optimizer` knows; `Settings.__post_init__` keeps a torch-free copy (OPTIMIZERS in
# training/settings.py, a settings test keeps the two equal) so an unknown name fails before the dataset is touched.
OPTIMIZERS = ("AdamW", "ELLISAdam")


def build_optimizer(name: str, params: Iterable[Tensor] | list[dict[str, Any]], config: OptimizerConfig) -> Optimizer:
    """
    Construct "AdamW" (torch) or "ELLISAdam" from the run's `optim_config`.

    `eps: None` is left out of the constructor call so each optimizer keeps its own default (ELLISAdam 1e-6,
    torch AdamW 1e-8), exactly as a config that never mentioned `eps` did.
    """

    common: dict[str, Any] = {"lr": config.lr, "betas": config.betas, "weight_decay": config.weight_decay}
    if config.eps is not None:
        common["eps"] = config.eps
    if name == "ELLISAdam":
        return ELLISAdam(
            params,
            **common,
            update_clipping=config.update_clipping,
            atan_adam=config.atan_adam,
            running_init=config.running_init,
            decouple_wd=config.decouple_wd,
        )
    defaults = OptimizerConfig()
    ellis_only_set = [option for option in ELLIS_ONLY_OPTIONS if getattr(config, option) != getattr(defaults, option)]
    if ellis_only_set:
        raise ValueError(f"optim_config option(s) {ellis_only_set} apply only to 'ELLISAdam', not {name!r}")
    if name == "AdamW":
        return torch.optim.AdamW(params, **common)
    raise ValueError(f"Invalid optimizer {name!r} requested (use one of {', '.join(map(repr, OPTIMIZERS))}).")


def set_lr(optimizer: Optimizer, lr: float) -> None:
    """
    Apply the scheduled learning rate to every group. ELLISAdam stores its LR as a float32 tensor and clones it in
    its step; torch AdamW gets the plain float, a tensor LR silently drops its foreach/fused path on CUDA and calls
    `lr.item()` per parameter every step.
    """

    value: torch.Tensor | float = torch.as_tensor(lr) if isinstance(optimizer, ELLISAdam) else lr
    for group in optimizer.param_groups:
        group["lr"] = value


class ELLISAdam(Optimizer):
    """
    AdamW variant with optional RMS update clipping, atan2 update, running init and decoupled weight decay.

    `lr` is stored as a float32 tensor; `init_lr` (the constructor LR) is the reference for decoupled weight decay,
    i.e. the decay applied per step is `lr / init_lr * weight_decay`.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | list[dict[str, Any]],
        lr: float | Tensor = 3e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        *,
        update_clipping: bool = False,
        running_init: bool = False,
        atan_adam: bool = False,
        decouple_wd: bool = True,
    ) -> None:
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32),
            init_lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_clipping=update_clipping,
            running_init=running_init,
            atan_adam=atan_adam,
            decouple_wd=decouple_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _init_group(
        self,
        group: dict[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
        running_init: bool = False,
    ) -> None:
        for param in group["params"]:
            if param.grad is None:
                continue
            params_with_grad.append(param)
            grads.append(param.grad)

            state = self.state[param]
            if len(state) == 0:
                # `step` lives on the CPU: kernel launches are costly on CUDA
                state["step"] = torch.tensor(0, dtype=torch.long)
                if running_init:
                    state["exp_avg"] = param.grad.clone().to(memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = param.grad.pow(2).clone().to(memory_format=torch.preserve_format)
                else:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step.
        """

        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, running_init=group["running_init"]
            )
            _single_tensor_modded_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                init_lr=group["init_lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                update_clipping=group["update_clipping"],
                atan_adam=group["atan_adam"],
                decouple_wd=group["decouple_wd"],
            )

        return loss


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
    """

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
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
