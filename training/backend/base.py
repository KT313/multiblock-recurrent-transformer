# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Device/strategy/precision abstraction used by the training loop.

The loop never touches CUDA, autocast, `torch.distributed` or rank checks directly; everything goes through a
`Backend`. `single_device.py` is the only implementation for now; a DDP/FSDP/Fabric backend implements the same
protocol later.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model import RecurrentGPT


def unwrap_compiled(model: Module) -> Module:
    """The plain module behind a `torch.compile` wrapper (state-dict keys stay stable across compiled/uncompiled
    runs)."""
    return getattr(model, "_orig_mod", model)


def plain_model(model: Module) -> RecurrentGPT:
    """The `RecurrentGPT` behind whatever `Backend.setup_model` wrapped around it (today: `torch.compile`); the loop
    reads `.step` / `.config` on it and the export writes it."""
    return cast(RecurrentGPT, unwrap_compiled(model))


class Backend(Protocol):
    """Everything that is device-, strategy- or precision-specific."""

    device: torch.device
    world_size: int
    rank: int
    is_main: bool
    pin_memory: bool  # whether dataloaders should pin host memory (true on CUDA)

    def setup_model(self, model: Module, compile_model: bool = False) -> Module:
        """Move the model to the device, optionally compile it, and wrap it (DDP/FSDP later)."""
        ...

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Hook for optimizer sharding later; identity on a single device."""
        ...

    def autocast(self) -> AbstractContextManager[None]:
        """Mixed-precision context for forward passes."""
        ...

    def backward(self, loss: Tensor) -> None: ...

    def no_sync(self, model: Module) -> AbstractContextManager[None]:
        """Skip gradient synchronisation while accumulating; no-op on a single device."""
        ...

    def all_reduce(self, tensor: Tensor, op: str = "mean") -> Tensor: ...

    def barrier(self) -> None: ...

    def clip_grad_norm(self, model: Module, max_norm: float) -> Tensor:
        """Clip gradients in place and return the pre-clip total norm."""
        ...

    def save_checkpoint(self, path: str | Path, state: dict[str, Any]) -> None: ...

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]: ...

    def seed_everything(self, seed: int) -> None: ...

    def rng_state(self) -> dict[str, Any]:
        """Python + torch (+ device) RNG state, as stored in a checkpoint."""
        ...

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore what `rng_state` collected."""
        ...

    def to_device(self, tensor: Tensor) -> Tensor:
        """Move a host tensor (a batch) to the device."""
        ...
