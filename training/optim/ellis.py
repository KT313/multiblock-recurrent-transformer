# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""ELLISAdam implementations, parameter-group validation, and lazy state initialization."""

from typing import Any, Callable, Iterable, overload

import torch
from torch import Tensor
from torch.optim import Optimizer

from training.optim.state import _quantizable, _quantized_zeros
from training.optim.update import _single_tensor_modded_adamw
from training.optimizer_validation import validate_adam_hyperparameters, validate_scalar


def _validate_lr(value: object, name: str, *, positive: bool) -> None:
    # Tensor LR is a public ELLIS API. A one-time scalar read is allowed at construction/restore,
    # never in set_lr or step (which would introduce a device synchronization on every update).
    if isinstance(value, Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar learning rate, got shape {tuple(value.shape)}")
        value = value.item()
    validate_scalar(value, name, positive=positive)


class ELLISAdam(Optimizer):
    """
    AdamW variant with optional RMS update clipping, atan2 update, running init and decoupled weight decay.

    `lr` is stored as a float32 tensor; `init_lr` (the constructor LR) is the reference for decoupled weight decay,
    i.e. the decay applied per step is `lr / init_lr * weight_decay`. Constructors and added/restored groups
    validate numerical metadata before registration. Constructor/reference LRs must be positive; a restored
    scheduled LR may be zero. Scheduled updates incur no extra scalar reads or validation.
    """

    __module__ = "training.optim"  # Preserve checkpoint provenance and the public pickle path.

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
        super().__init__(params, self._defaults(**locals()))

    @staticmethod
    def _defaults(
        lr: float | Tensor,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        update_clipping: bool,
        running_init: bool,
        atan_adam: bool,
        decouple_wd: bool,
        **_: Any,  # `self`, `params`, `__class__` of the caller's `locals()`
    ) -> dict[str, Any]:
        _validate_lr(lr, "ELLISAdam.lr", positive=True)
        validate_adam_hyperparameters(betas=betas, eps=eps, weight_decay=weight_decay, prefix="ELLISAdam")
        return dict(
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

    def _validate_group(self, group: dict[str, Any], *, restored: bool) -> None:
        prefix = f"{type(self).__name__} parameter group"
        required = ("lr", "init_lr", "betas", "eps", "weight_decay")
        missing = [key for key in required if key not in group]
        if missing:
            raise ValueError(f"{prefix} is missing hyperparameters: {missing}")
        # Restored LR is scheduled and may be zero; init_lr is always the positive decay reference.
        _validate_lr(group["lr"], f"{prefix}.lr", positive=not restored)
        _validate_lr(group["init_lr"], f"{prefix}.init_lr", positive=True)
        validate_adam_hyperparameters(
            betas=group["betas"], eps=group["eps"], weight_decay=group["weight_decay"], prefix=prefix
        )
        if "state_bits" in self.defaults and group.get("state_bits") not in (8, 32):
            raise ValueError(f"{prefix}.state_bits must be 8 or 32, not {group.get('state_bits')!r}")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Merge metadata only. Do not materialize or traverse the caller's parameter generator twice.
        self._validate_group(self.defaults | param_group, restored=False)
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Validate the checkpoint evidence before PyTorch replaces live groups or casts state tensors.
        # Do not fill missing saved metadata from current defaults: that would conceal an invalid checkpoint.
        for group in state_dict["param_groups"]:
            self._validate_group(group, restored=True)
        super().load_state_dict(state_dict)

    def _new_state(self, param: Tensor, group: dict[str, Any], running_init: bool) -> tuple[Tensor, Tensor]:
        """
        The fresh (exp_avg, exp_avg_sq) of `param`: zeros, or the first gradient and its square (`running_init`).
        """

        grad = param.grad
        assert grad is not None
        if running_init:
            return (
                grad.clone().to(memory_format=torch.preserve_format),
                grad.pow(2).clone().to(memory_format=torch.preserve_format),
            )
        return (
            torch.zeros_like(param, memory_format=torch.preserve_format),
            torch.zeros_like(param, memory_format=torch.preserve_format),
        )

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
                state["exp_avg"], state["exp_avg_sq"] = self._new_state(param, group, running_init)

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


class ELLISAdam8bit(ELLISAdam):
    """
    ELLISAdam with the two moments stored block-wise quantised to 8 bits (`OptimState8bit`) wherever
    `group["state_bits"]` is 8 (the default; `build_optimizer` sets 32 on the embedding group) and the tensor is
    large enough for the block quantisation. Every other part is inherited: the update reads the moments through
    `_dequantized`, so the ELLIS options run on fp32 values, and writes the new moments back with `copy_`, which
    quantises. The state dict holds the quantised tensors as they are; `torch.load` knows the class (torchao
    registers it as a safe global) and `Optimizer.load_state_dict` moves it to the parameter's device.
    """

    __module__ = "training.optim"  # Preserve checkpoint provenance and the public pickle path.

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
        state_bits: int = 8,
    ) -> None:
        if state_bits not in (8, 32):
            raise ValueError(f"state_bits must be 8 or 32, not {state_bits!r}")
        defaults = self._defaults(**locals()) | {"state_bits": state_bits}
        Optimizer.__init__(self, params, defaults)

    def _new_state(self, param: Tensor, group: dict[str, Any], running_init: bool) -> tuple[Tensor, Tensor]:
        if group["state_bits"] != 8 or not _quantizable(param):
            return super()._new_state(param, group, running_init)
        exp_avg = _quantized_zeros(param, signed=True)
        exp_avg_sq = _quantized_zeros(param, signed=False)
        if running_init:
            grad = param.grad
            assert grad is not None
            exp_avg.copy_(grad)
            exp_avg_sq.copy_(grad.pow(2))
        return exp_avg, exp_avg_sq
