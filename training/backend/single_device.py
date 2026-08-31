# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Single-device backend: one CUDA GPU (CPU fallback), bf16 autocast, plain `torch.save`/`torch.load`."""

import os
import random
import warnings
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

PRECISIONS = ("bf16-mixed", "32")


def _set_torch_flags() -> None:
    """Global matmul/cuDNN settings the training runs used (TF32 accumulation, benchmark mode)."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class SingleDeviceBackend:
    """Runs everything on `cuda:0` if available, otherwise on the CPU (with a warning)."""

    world_size: int = 1
    rank: int = 0
    is_main: bool = True

    def __init__(self, device: str | None = None, precision: str = "bf16-mixed") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                warnings.warn("No CUDA device available, falling back to CPU.", stacklevel=2)
                device = "cpu"
        self.device = torch.device(device)
        self.precision = precision
        self.pin_memory = self.device.type == "cuda"
        _set_torch_flags()

    def setup_model(self, model: Module, compile: bool = False) -> Module:
        model = model.to(self.device)
        if compile:
            # dynamic=True: variable sequence lengths (padding multiples) must not trigger recompiles
            # torch.compile is typed as returning a bare callable; at runtime it is an OptimizedModule (a Module)
            model = cast(Module, torch.compile(model, dynamic=True))
        return model

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        return optimizer

    def autocast(self) -> AbstractContextManager[None]:
        if self.precision == "bf16-mixed":
            # torch.autocast is a context manager by protocol only (no AbstractContextManager base in the stubs)
            return cast(
                AbstractContextManager[None], torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            )
        return nullcontext()

    def backward(self, loss: Tensor) -> None:
        loss.backward()  # type: ignore[no-untyped-call]  # Tensor.backward is unannotated in torch

    def no_sync(self, model: Module) -> AbstractContextManager[None]:
        return nullcontext()

    def all_reduce(self, tensor: Tensor, op: str = "mean") -> Tensor:
        return tensor

    def barrier(self) -> None:
        return None

    def clip_grad_norm(self, model: Module, max_norm: float) -> Tensor:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)

    def save_checkpoint(self, path: str | Path, state: dict[str, Any]) -> None:
        # Written to a sibling temp file and renamed: a crash (or the second Ctrl-C) mid-save never leaves a
        # truncated file under the final name, which `find_latest_checkpoint` would otherwise pick.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        # Our own trusted checkpoints contain plain python objects (configs, RNG states), hence weights_only=False.
        # Loaded on the CPU: `load_state_dict` moves what belongs on the device, and the optimizer's CPU-hosted
        # step counters stay on the CPU as in a fresh run.
        return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))

    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rng_state(self) -> dict[str, Any]:
        # Only this backend's device: a CPU run on a GPU box must not initialise CUDA at every checkpoint, and a
        # checkpoint must not depend on how many GPUs the machine has.
        state = {"python": random.getstate(), "torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"].cpu(), self.device)

    def to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(self.device, non_blocking=True)
