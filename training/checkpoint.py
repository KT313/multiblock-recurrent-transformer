# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Checkpoint save/load/resume through the backend. Steps in file names are OPTIMIZER steps.

State layout: `{"model", "optimizer", "step", "stage", "rng", "config", "dataset_config_hash", "dataset_validation_rows"}`;
the training loop supplies everything except "model"/"optimizer" via `extra` (the last two are verified on resume by
`training.data.dataset_resolver.check_checkpoint_dataset_hash` / `check_checkpoint_validation_rows`; "rng" is
`Backend.rng_state()`).
"""

import re
from pathlib import Path
from typing import Any, Optional

from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend

CHECKPOINT_SUBDIR = "checkpoints"
CHECKPOINT_SUFFIX = ".pth"


def checkpoint_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / CHECKPOINT_SUBDIR


def checkpoint_name(step: int, run_name: str, stage_end: Optional[int] = None) -> str:
    """`step-{step:08d}-{run_name}` plus `-stage-{i}_end` for the checkpoint written before a stage transition."""
    name = f"step-{step:08d}-{run_name}"
    if stage_end is not None:
        name += f"-stage-{stage_end}_end"
    return name + CHECKPOINT_SUFFIX


def _step_from_name(path: Path) -> int:
    return int(path.name.split("-")[1])


def find_latest_checkpoint(out_dir: str | Path, run_name: str) -> Optional[Path]:
    """Highest-step checkpoint of `run_name` under `out_dir/checkpoints`, or None."""
    base = checkpoint_dir(out_dir)
    pattern = re.compile(rf"^step-\d{{8}}-{re.escape(run_name)}(-stage-\d+_end)?{re.escape(CHECKPOINT_SUFFIX)}$")
    candidates = [p for p in base.glob(f"step-*{CHECKPOINT_SUFFIX}") if pattern.match(p.name)]
    if not candidates:
        return None
    return max(candidates, key=_step_from_name)


def should_save_checkpoint(
    step: int, *, max_steps: int, save_step_interval: int, save_last_step: bool, stage_end: bool = False
) -> bool:
    """Save at every `save_step_interval`, at the last step if requested, and before every stage transition."""
    save_at_interval = save_step_interval > 0 and step % save_step_interval == 0
    save_at_last_step = save_last_step and step >= max_steps
    return save_at_interval or save_at_last_step or stage_end


def _unwrap(model: Module) -> Module:
    """Strip the `torch.compile` wrapper so state-dict keys stay stable across compiled/uncompiled runs."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optimizer, extra: dict[str, Any]
) -> None:
    """Write model + optimizer state dicts and `extra` (step, stage, dataloader, rng, config) to `path`."""
    state: dict[str, Any] = {"model": _unwrap(model).state_dict(), "optimizer": optimizer.state_dict()}
    state.update(extra)
    backend.save_checkpoint(path, state)


def load_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optional[Optimizer] = None
) -> dict[str, Any]:
    """Load model (and optimizer) state in place; return the remaining entries (step, stage, dataloader, rng, ...)."""
    state = backend.load_checkpoint(path)
    _unwrap(model).load_state_dict(state.pop("model"))
    optimizer_state = state.pop("optimizer")
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    return state
