# (c) 2026 Tobias Kerner. Apache-2.0.
"""Optimizer lifecycle helpers; the disabled path does not import PyTorch ZRO."""

from typing import TYPE_CHECKING

from torch.optim import Optimizer

if TYPE_CHECKING:
    from training.optim.zero1 import ShardedOptimizer


def sharded_optimizer(optimizer: Optimizer) -> "ShardedOptimizer | None":
    if not hasattr(optimizer, "optim"):
        return None
    from training.optim.zero1 import ShardedOptimizer

    return optimizer if isinstance(optimizer, ShardedOptimizer) else None


def local_optimizer(optimizer: Optimizer) -> Optimizer:
    sharded = sharded_optimizer(optimizer)
    return sharded.optim if sharded is not None else optimizer


def prepare_optimizer_state(optimizer: Optimizer) -> None:
    """All ranks participate BEFORE entering a rank-zero checkpoint publication callback."""
    sharded = sharded_optimizer(optimizer)
    if sharded is not None:
        sharded.consolidate_state_dict(to=0)


def release_optimizer_state(optimizer: Optimizer) -> None:
    sharded = sharded_optimizer(optimizer)
    if sharded is not None:
        sharded.release_consolidated_state()
