# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Device/strategy/precision abstraction used by the training loop.

The loop never touches CUDA, autocast, `torch.distributed` or rank checks directly; everything goes through a
`Backend`. Two implementations: `single_device.py` (one GPU or the CPU, every collective the identity) and `ddp.py`
(one process per GPU under torchrun, `torch.distributed` collectives). The loop uses the collectives at the places
a multi-rank run needs them: the checkpoint gathers every rank's RNG state (`all_gather_object`), the stop request
is decided by `any_flag` so all ranks stop after the same step, the loss and the validation losses are reduced
(`all_reduce`), and the main rank's packs reach the other ranks through `scatter_packs` (rank 0 owns the single
data stream, `training.step.RankBatches`).

Wrappers around the model (`torch.compile`, DDP later) are applied by `setup_model` and recorded in `wrappers`;
`plain_model` unwraps exactly that layering (`unwrap_model`) and raises on anything else, so a wrapper the backend
did not expect is an error and never a silently missed attribute (`model.step` set on the wrong object would leave
the recurrence sampler at step 0 forever).
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import Tensor
from torch._dynamo.eval_frame import OptimizedModule
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from model import RecurrentGPT

T = TypeVar("T")

# wrapper kind -> (the wrapper class, the attribute holding the wrapped module); the kinds `Backend.wrappers` lists
WRAPPERS: dict[str, tuple[type[Module], str]] = {
    "compile": (OptimizedModule, "_orig_mod"),
    "ddp": (DistributedDataParallel, "module"),
}


def unwrap_layers(model: Module, layers: Sequence[str]) -> Module:
    """
    The module behind exactly the wrappers `layers` names, outermost first (`WRAPPERS` kinds, for example
    `("compile", "ddp")` for a DDP-wrapped model under torch.compile). Every layer must be the expected wrapper
    class; anything else raises `TypeError` naming the expected layering and the chain of types found. Never peels
    off whatever wrapper happens to be there: a layering the backend did not apply is a bug worth seeing.
    """

    current = model
    chain: list[str] = [type(model).__name__]
    for kind in layers:
        if kind not in WRAPPERS:
            raise ValueError(f"unknown wrapper kind {kind!r}; known: {sorted(WRAPPERS)}")
        wrapper_class, attribute = WRAPPERS[kind]
        if not isinstance(current, wrapper_class):
            raise TypeError(
                f"expected a {kind} wrapper ({wrapper_class.__name__}) at this layer but found "
                f"{type(current).__name__}; expected layering {tuple(layers)}, found {' -> '.join(chain)}"
            )
        current = getattr(current, attribute)
        chain.append(type(current).__name__)
    return current


def unwrap_model(model: Module, layers: Sequence[str]) -> RecurrentGPT:
    """
    The `RecurrentGPT` behind exactly the wrappers `layers` names (`unwrap_layers`); a different innermost module
    raises `TypeError` too.
    """

    plain = unwrap_layers(model, layers)
    if not isinstance(plain, RecurrentGPT):
        raise TypeError(
            f"expected a RecurrentGPT behind the wrappers {tuple(layers)} but found {type(plain).__name__}"
        )
    return plain


class Backend(Protocol):
    """
    Everything that is device-, strategy- or precision-specific.
    """

    device: torch.device
    world_size: int
    rank: int
    is_main: bool
    pin_memory: bool  # whether dataloaders should pin host memory (true on CUDA)
    wrappers: tuple[str, ...]  # the wrapper kinds `setup_model` applied, outermost first (`WRAPPERS`)

    def setup_model(self, model: Module, compile_model: bool = False) -> Module:
        """
        Move the model to the device, optionally compile it, and wrap it (DDP later); records `wrappers`.
        """

        ...

    def plain_model(self, model: Module) -> RecurrentGPT:
        """
        The `RecurrentGPT` behind what `setup_model` wrapped around it (`unwrap_model` with `wrappers`); the loop
        reads `.step` / `.config` on it, the checkpoint stores its state dict, the export writes it.
        """

        ...

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """
        Hook for optimizer sharding later; identity on a single device.
        """

        ...

    def autocast(self) -> AbstractContextManager[None]:
        """
        Mixed-precision context for forward passes.
        """

        ...

    def backward(self, loss: Tensor) -> None: ...

    def no_sync(self, model: Module) -> AbstractContextManager[None]:
        """
        Skip gradient synchronisation while accumulating; no-op on a single device.
        """

        ...

    def all_reduce(self, tensor: Tensor, op: str = "mean") -> Tensor: ...

    def barrier(self) -> None: ...

    def all_gather_object(self, obj: T) -> list[T]:
        """
        Every rank's `obj`, indexed by rank (a one-element list on a single device). Picklable objects only, and
        tensors inside them on the CPU.
        """

        ...

    def any_flag(self, flag: bool) -> bool:
        """
        Whether any rank passed True (the flag itself on a single device). The stop request goes through this, so
        every rank stops after the same step.
        """

        ...

    def scatter_packs(self, packs: Tensor | None, slice_shape: tuple[int, ...]) -> Tensor:
        """
        This rank's slice of the main rank's packs: rank 0 passes a `(world_size, *slice_shape)` tensor on its
        device (the other ranks None) and every rank gets its `slice_shape` slice, on its device. The single device
        returns `packs[0]`. One collective per micro-batch index (`training.step.RankBatches`).
        """

        ...

    def shutdown(self) -> None:
        """
        Release what the backend holds (the process group later); no-op on a single device. Called on every way
        out of `train()`.
        """

        ...

    def clip_grad_norm(self, model: Module, max_norm: float) -> Tensor:
        """
        Clip gradients in place and return the pre-clip total norm.
        """

        ...

    def save_checkpoint(self, path: str | Path, state: dict[str, Any]) -> None: ...

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]: ...

    def seed_everything(self, seed: int) -> None:
        """
        Seed Python, numpy and torch with `seed + rank`: rank 0 gets `seed` itself, every other rank draws
        different latent noise (DDP broadcasts rank 0's parameters, so the init is shared anyway).
        """

        ...

    def rng_state(self) -> dict[str, Any]:
        """
        Python + torch (+ device) RNG state of this rank, as stored in a checkpoint.
        """

        ...

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """
        Restore what `rng_state` collected.
        """

        ...

    def to_device(self, tensor: Tensor) -> Tensor:
        """
        Move a host tensor (a batch) to the device.
        """

        ...
