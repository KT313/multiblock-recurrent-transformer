"""
Shared compiler policy. Backends call this after device placement and, for DDP, after wrapping the model.
"""

from typing import cast

import torch
from torch.nn import Module

DYNAMO_RECOMPILE_LIMIT = 32
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def compile_module(model: Module, mode: str = "default") -> Module:
    """
    Compile with dynamic shapes and sufficient cache entries for the recurrent forward/backward variants.
    """

    if mode not in COMPILE_MODES:
        raise ValueError(f"unknown compile mode {mode!r}; choose from {COMPILE_MODES}")
    torch._dynamo.config.recompile_limit = DYNAMO_RECOMPILE_LIMIT
    if mode == "default":
        return cast(Module, torch.compile(model, dynamic=True))
    return cast(Module, torch.compile(model, dynamic=True, mode=mode))
