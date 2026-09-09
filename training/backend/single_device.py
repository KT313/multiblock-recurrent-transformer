# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Single-device backend: one CUDA GPU (CPU fallback), bf16 autocast, plain `torch.save`/`torch.load`. Every collective
of the protocol is the identity here. A launch under `torchrun` with more than one rank is refused at construction:
with this backend every rank would train its own copy of the run on the same device.
"""

import os
import random
import warnings
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from data_preparation.lib.storage.atomic import write_atomically
from model import RecurrentGPT
from training.backend.base import unwrap_model

T = TypeVar("T")

PRECISIONS = ("bf16-mixed", "32")
WORLD_SIZE_ENV = "WORLD_SIZE"  # set by torchrun for every rank; more than 1 needs a multi-rank backend
DYNAMO_RECOMPILE_LIMIT = 32  # per compiled frame; the default 8 is one above what the recurrence iteration needs


def _set_torch_flags() -> None:
    """
    Global matmul/cuDNN settings the training runs used (TF32 accumulation, benchmark mode).
    """

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


class SingleDeviceBackend:
    """
    Runs everything on `cuda:0` if available, otherwise on the CPU (with a warning).
    """

    world_size: int = 1
    rank: int = 0
    is_main: bool = True

    def __init__(self, device: str | None = None, precision: str = "bf16-mixed") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self._check_launch_environment()
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            else:
                warnings.warn("No CUDA device available, falling back to CPU.", stacklevel=2)
                device = "cpu"
        self.device = torch.device(device)
        self.precision = precision
        self.pin_memory = self.device.type == "cuda"
        self.wrappers: tuple[str, ...] = ()
        _set_torch_flags()

    def _check_launch_environment(self) -> None:
        """
        Refuse a torchrun launch with several ranks: every rank would build this backend on the same device and
        train its own copy of the run. The multi-rank backend (`training/backend/ddp.py`) overrides this.
        """

        launched_world = int(os.environ.get(WORLD_SIZE_ENV, "1"))
        if launched_world > 1:
            raise RuntimeError(
                f"{WORLD_SIZE_ENV}={launched_world} in the environment (a torchrun launch with {launched_world} ranks) "
                "but the run uses the single_device backend, which would train an independent copy of the run per "
                "rank on the same device; set `backend: ddp` in the run config for a multi-GPU run"
            )

    def setup_model(self, model: Module, compile_model: bool = False) -> Module:
        model = model.to(self.device)
        if compile_model:
            # The recurrence iteration is compiled as one frame with several legitimate variants (no-grad and grad
            # iterations, the latent with and without gradient), the checkpointed iteration (`checkpointed_iteration`,
            # its own frame holding the checkpoint call) likewise; past dynamo's default limit of 8 recompiles it
            # silently runs the frame eagerly (a warning in the log, a 10 percent slower step).
            torch._dynamo.config.recompile_limit = DYNAMO_RECOMPILE_LIMIT
            # dynamic=True: variable sequence lengths (padding multiples) must not trigger recompiles. Packed
            # training has one shape, but its validation is still padded: a static compile recompiled the forward for
            # every validation length and ran past the limit into eager (measured: validation three times slower).
            # torch.compile is typed as returning a bare callable; at runtime it is an OptimizedModule (a Module)
            model = cast(Module, torch.compile(model, dynamic=True))
        self.wrappers = ("compile",) if compile_model else ()
        return model

    def plain_model(self, model: Module) -> RecurrentGPT:
        return unwrap_model(model, self.wrappers)

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

    def all_gather_object(self, obj: T) -> list[T]:
        return [obj]

    def any_flag(self, flag: bool) -> bool:
        return flag

    def scatter_packs(self, packs: Tensor | None, slice_shape: tuple[int, ...]) -> Tensor:
        if packs is None:
            raise ValueError("the single device is the main rank and must pass the packs")
        return packs[0]

    def shutdown(self) -> None:
        return None

    def clip_grad_norm(self, model: Module, max_norm: float) -> Tensor:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=False)

    def save_checkpoint(self, path: str | Path, state: dict[str, Any]) -> None:
        # `write_atomically`: a crash mid-save never leaves a truncated file under the final name
        with write_atomically(path) as temporary:
            torch.save(state, temporary)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        # weights_only=False: our checkpoints hold plain python objects (configs, RNG states)
        # map_location="cpu": `load_state_dict` moves what belongs on the device; optimizer step counters stay there
        return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))

    def seed_everything(self, seed: int) -> None:
        seed += self.rank  # rank 0 keeps `seed`; other ranks (a multi-rank subclass) draw other latent noise
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rng_state(self) -> dict[str, Any]:
        # only this backend's device: a CPU run must not initialise CUDA, a checkpoint must not depend on the GPU count
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
