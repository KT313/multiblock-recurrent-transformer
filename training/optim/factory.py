# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Construct the configured optimizer and apply its scheduled learning rate."""

from typing import Any, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from training.optim.ellis import ELLISAdam, ELLISAdam8bit
from training.optim.groups import _pin_embedding_group_to_fp32
from training.settings import OptimizerConfig

# `OptimizerConfig` fields that are ELLISAdam constructor arguments only; a non-default value with another
# optimizer is a config mistake and fails loudly instead of being dropped.
ELLIS_ONLY_OPTIONS = ("update_clipping", "atan_adam", "running_init", "decouple_wd")

# The `optimizer:` values `build_optimizer` knows; `Settings.__post_init__` keeps a torch-free copy (OPTIMIZERS in
# training/settings.py, a settings test keeps the two equal) so an unknown name fails before the dataset is touched.
OPTIMIZERS = ("AdamW", "ELLISAdam", "ELLISAdam8bit")
ELLIS_OPTIMIZERS = ("ELLISAdam", "ELLISAdam8bit")


def build_optimizer(name: str, params: Iterable[Tensor] | list[dict[str, Any]], config: OptimizerConfig) -> Optimizer:
    """
    Construct "AdamW" (torch) or "ELLISAdam" from the run's `optim_config`.

    `eps: None` is left out of the constructor call so each optimizer keeps its own default (ELLISAdam 1e-6,
    torch AdamW 1e-8), exactly as a config that never mentioned `eps` did.
    """

    # Validate the configuration and retain each optimizer's default epsilon.
    config.validate(name)
    common: dict[str, Any] = {"lr": config.lr, "betas": config.betas, "weight_decay": config.weight_decay}
    if config.eps is not None:
        common["eps"] = config.eps

    # Construct an ELLIS optimizer with its optional update settings.
    if name in ELLIS_OPTIMIZERS:
        ellis_options = {option: getattr(config, option) for option in ELLIS_ONLY_OPTIONS}
        if name == "ELLISAdam8bit":
            return ELLISAdam8bit(_pin_embedding_group_to_fp32(params), **common, **ellis_options)
        return ELLISAdam(params, **common, **ellis_options)

    # Reject ELLIS-only settings before constructing native AdamW.
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
