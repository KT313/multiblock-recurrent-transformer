# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
DDP backend: one process per GPU launched by torchrun, `torch.distributed` collectives, the model wrapped in
`DistributedDataParallel` (then compiled, when asked).

The process group comes from the torchrun environment (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`,
`MASTER_PORT`): NCCL on CUDA with the device `cuda:<LOCAL_RANK>`, gloo on the CPU (the CPU path is how the
multi-rank loop is tested on a laptop; `device="cpu"` forces it). The group's timeout is `DDP_TIMEOUT`, hours rather
than NCCL's ten minutes: the other ranks wait at a barrier while the main rank builds a dataset or runs a benchmark,
and a wait that long is legitimate. A genuine hang therefore surfaces late; torchrun still ends every rank when one
of them dies.

What differs from the single device: `setup_model` wraps in DDP (which broadcasts rank 0's parameters, so the
per-rank seeds of `seed_everything` never reach the model), `no_sync` is DDP's, the collectives are real,
`scatter_packs` hands every rank its slice of the main rank's packs, and `save_checkpoint` writes on the main rank
only. Everything else (precision, checkpoint loading, RNG state, clipping) is inherited.
"""

import os
import warnings
from contextlib import AbstractContextManager
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from training.backend.single_device import DYNAMO_RECOMPILE_LIMIT, SingleDeviceBackend

T = TypeVar("T")

TORCHRUN_VARIABLES = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
# The process group's collective timeout. Hours, not NCCL's default ten minutes: while the main rank builds a dataset
# (`auto_prepare`) or runs the lm-eval benchmarks, every other rank waits at a barrier, and that wait is legitimate.
DDP_TIMEOUT = timedelta(hours=4)


class DDPBackend(SingleDeviceBackend):
    """
    One rank of a torchrun launch: `cuda:<LOCAL_RANK>` (or the CPU, with gloo), the model in DDP.

    `device` forces a device (`"cpu"` for the gloo path in tests); None picks `cuda:<LOCAL_RANK>` when CUDA is
    available and falls back to the CPU with a warning otherwise.
    """

    wrappers: tuple[str, ...]  # declared here too: the type checkers otherwise narrow it to the literals below

    def __init__(self, device: str | None = None, precision: str = "bf16-mixed") -> None:
        missing = [name for name in TORCHRUN_VARIABLES if name not in os.environ]
        if missing:
            raise RuntimeError(
                f"backend ddp needs the torchrun environment but {missing} are not set; launch the run with "
                "`torchrun --standalone --nproc_per_node=<gpus> training/train.py --config <run.yaml>` "
                "(`make training-ddp`), or use `backend: single_device` for one GPU"
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{local_rank}"
            else:
                warnings.warn("No CUDA device available, falling back to CPU (gloo).", stacklevel=2)
                device = "cpu"
        super().__init__(device=device, precision=precision)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend="nccl" if self.device.type == "cuda" else "gloo", init_method="env://", timeout=DDP_TIMEOUT
        )
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.is_main = self.rank == 0

    def _check_launch_environment(self) -> None:
        return None  # several ranks are the point here

    def setup_model(self, model: Module, compile_model: bool = False) -> Module:
        """
        Move to the device, wrap in `DistributedDataParallel` (rank 0's parameters are broadcast to every rank
        here), then compile when asked: compile OUTSIDE the DDP wrapper, the order PyTorch recommends (its DDP
        optimizer splits the graph at the bucket boundaries so the gradient all-reduce overlaps the backward).
        `wrappers` records the layering for `plain_model`.
        """

        model = model.to(self.device)
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        wrapped: Module = DistributedDataParallel(model, device_ids=device_ids)
        if compile_model:
            torch._dynamo.config.recompile_limit = DYNAMO_RECOMPILE_LIMIT  # see the single-device backend
            wrapped = cast(Module, torch.compile(wrapped, dynamic=True))
        self.wrappers = ("compile", "ddp") if compile_model else ("ddp",)
        return wrapped

    def _ddp_module(self, model: Module) -> DistributedDataParallel:
        """
        The DDP wrapper inside `model` (behind the compile wrapper when there is one).
        """

        candidate = getattr(model, "_orig_mod", model) if "compile" in self.wrappers else model
        if not isinstance(candidate, DistributedDataParallel):
            raise TypeError(f"expected the DDP wrapper, found {type(candidate).__name__}; layering {self.wrappers}")
        return candidate

    def no_sync(self, model: Module) -> AbstractContextManager[None]:
        return cast(AbstractContextManager[None], self._ddp_module(model).no_sync())

    def all_reduce(self, tensor: Tensor, op: str = "mean") -> Tensor:
        """
        The sum over the ranks, in place in `tensor` (which must live on this rank's device), divided by the world
        size for `op="mean"`.
        """

        if op not in ("mean", "sum"):
            raise ValueError(f"op must be 'mean' or 'sum', got {op!r}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / self.world_size if op == "mean" else tensor

    def barrier(self) -> None:
        dist.barrier()

    def all_gather_object(self, obj: T) -> list[T]:
        gathered: list[T | None] = [None] * self.world_size
        dist.all_gather_object(gathered, obj)
        return cast(list[T], gathered)

    def any_flag(self, flag: bool) -> bool:
        flags = torch.tensor([1 if flag else 0], dtype=torch.int64, device=self.device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        return bool(flags.item())

    def scatter_packs(self, packs: Tensor | None, slice_shape: tuple[int, ...]) -> Tensor:
        received = torch.empty(slice_shape, dtype=torch.int64, device=self.device)
        if self.is_main:
            if packs is None:
                raise ValueError("the main rank must pass the packs to scatter")
            if tuple(packs.shape) != (self.world_size, *slice_shape):
                raise ValueError(f"packs of shape {tuple(packs.shape)}, expected {(self.world_size, *slice_shape)}")
            slices: list[Tensor] | None = list(packs.to(self.device, dtype=torch.int64).contiguous())
        else:
            slices = None
        dist.scatter(received, slices, src=0)
        return received

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def save_checkpoint(self, path: str | Path, state: dict[str, Any]) -> None:
        if self.is_main:
            super().save_checkpoint(path, state)
