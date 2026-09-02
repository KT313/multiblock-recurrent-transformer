# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Optimizer used by the thesis runs: ELLISAdam (verbatim port) plus the parameter-group split.

Only the options the final config sets are kept (`update_clipping`, `atan_adam`, `running_init`, `decouple_wd`);
the other experimental switches of the upstream implementation were never enabled and are gone.
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
    """Split parameters into weights / embeddings / scale-and-norm groups, as upstream did.

    Group order matters for checkpoints: 0 = matrices, 1 = embeddings (+ tied lm_head), 2 = norms and biases.
    Every group carries `base_lr` (a per-group multiplier applied to the scheduled LR by `set_lr`).
    """
    weights_group: list[Tensor] = []
    embedding_group: list[Tensor] = []
    scale_and_norm_group: list[Tensor] = []
    for name, param in model.named_parameters():
        lname = name.lower()
        if "wte" in lname or "embedding" in lname or "lm_head" in lname:
            embedding_group.append(param)
        elif "ln_f" in lname or "norm" in lname or "bias" in lname:
            scale_and_norm_group.append(param)
        elif "proj" in lname or "qkv" in lname or "fc" in lname or param.ndim == 2:
            weights_group.append(param)
        else:
            raise ValueError(f"param {name} could not be matched to an optim group")

    param_groups = [
        {"params": weights_group, "base_lr": 1.0, "weight_decay": weight_decay},
        {"params": embedding_group, "base_lr": 1.0, "weight_decay": weight_decay},
        {"params": scale_and_norm_group, "base_lr": 1.0, "weight_decay": weight_decay},
    ]
    if no_wd_for_bias_and_norm:
        param_groups[-1]["weight_decay"] = 0.0
    return param_groups


# `OptimizerConfig` fields that are ELLISAdam constructor arguments only; a non-default value with another
# optimizer is a config mistake and fails loudly instead of being dropped.
ELLIS_ONLY_OPTIONS = ("update_clipping", "atan_adam", "running_init", "decouple_wd")


def build_optimizer(name: str, params: Iterable[Tensor] | list[dict[str, Any]], config: OptimizerConfig) -> Optimizer:
    """Construct "AdamW" (torch) or "ELLISAdam" from the run's `optim_config`.

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
    ellis_only_set = [k for k in ELLIS_ONLY_OPTIONS if getattr(config, k) != getattr(defaults, k)]
    if ellis_only_set:
        raise ValueError(f"optim_config option(s) {ellis_only_set} apply only to 'ELLISAdam', not {name!r}")
    if name == "AdamW":
        return torch.optim.AdamW(params, **common)
    raise ValueError(f"Invalid optimizer {name!r} requested (use 'AdamW' or 'ELLISAdam').")


def set_lr(optimizer: Optimizer, lr: float) -> None:
    """Apply the scheduled learning rate to every group, scaled by the group's `base_lr` multiplier."""
    for group in optimizer.param_groups:
        group["lr"] = torch.as_tensor(lr * group.get("base_lr", 1.0))


class ELLISAdam(Optimizer):
    """AdamW variant with optional RMS update clipping, atan2 update, running init and decoupled weight decay.

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
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                # `step` deliberately lives on the CPU: kernel launches are costly on CUDA.
                state["step"] = torch.tensor(0, dtype=torch.long)
                if running_init:
                    state["exp_avg"] = p.grad.clone().to(memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = p.grad.pow(2).clone().to(memory_format=torch.preserve_format)
                else:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
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
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr.clone() if isinstance(lr, torch.Tensor) else lr

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        step_t += 1
        bias_correction1 = 1 - beta1**step_t
        bias_correction2 = 1 - beta2**step_t
        bias_correction2_sqrt = sqrt(bias_correction2)

        step_size = step_size / bias_correction1

        # Perform stepweight decay
        if decouple_wd:
            param.mul_(1 - lr / init_lr * weight_decay)
        else:
            param.mul_(1 - lr * weight_decay)
        if atan_adam:
            update = torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt))
            # torch accepts a 0-d tensor for `alpha`/`value` at runtime; the stubs only list python numbers
            param.add_(update, alpha=-step_size)  # type: ignore[arg-type]
        else:
            param.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)  # type: ignore[arg-type]
