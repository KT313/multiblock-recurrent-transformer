# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Checkpoint schema, naming, search, save/load through the backend. Steps in file names are OPTIMIZER steps.

A checkpoint is one `torch.save` dict: the two state dicts `"model"` / `"optimizer"` plus the fields of
`CheckpointMetadata` (`step`, `stage`, `rng`, `settings`, `model_config`, `dataset_config_hash`, `validation_rows`,
`data_stream`). `dataset_config_hash` and `validation_rows` are verified on resume by
`training.data.dataset_resolver.check_dataset_unchanged`; `rng` is `Backend.rng_state()`. There is no loader for
older layouts (clean break, a standing decision): `CheckpointMetadata.from_state` raises on a missing key.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend
from training.settings import Settings
from training.stage_manager import StageManager

CHECKPOINT_SUBDIR = "checkpoints"
CHECKPOINT_SUFFIX = ".pth"


@dataclass
class CheckpointMetadata:
    """Everything in a checkpoint besides the two state dicts."""

    step: int  # optimizer steps completed when the checkpoint was written
    stage: int  # stage the run is in at `step` (the one it enters next when written before a transition)
    rng: dict[str, Any]  # `Backend.rng_state()` after evaluation and logging of `step`
    settings: dict[str, Any]  # `asdict(Settings)` of the run
    model_config: dict[str, Any]  # `RecurrentConfig.to_dict()` of the trained model
    dataset_config_hash: str  # `ResolvedDataset.config_hash`
    validation_rows: dict[str, int]  # `ResolvedDataset.validation_rows`, {source: rows held out for validation}
    data_stream: dict[str, Any]  # `training.step.BatchStream.state_dict()`: consumed rows + the transition RNG

    def to_state(self) -> dict[str, Any]:
        """The metadata as the flat dict merged into the checkpoint (a shallow copy, tensors are not copied)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CheckpointMetadata":
        """Read the metadata fields out of a loaded checkpoint dict; other keys (the state dicts) are ignored."""
        missing = [f.name for f in fields(cls) if f.name not in state]
        if missing:
            raise KeyError(
                f"checkpoint is missing the metadata key(s) {missing}; it was written by an older version of the "
                "training code and cannot be resumed (checkpoint formats are a clean break)"
            )
        return cls(**{f.name: state[f.name] for f in fields(cls)})


def checkpoint_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / CHECKPOINT_SUBDIR


def checkpoint_name(step: int, run_name: str, stage_end: Optional[int] = None) -> str:
    """`step-{step:08d}-{run_name}` plus `-stage-{i}_end` for the checkpoint written before a stage transition."""
    name = f"step-{step:08d}-{run_name}"
    if stage_end is not None:
        name += f"-stage-{stage_end}_end"
    return name + CHECKPOINT_SUFFIX


def checkpoint_path(run_directory: str | Path, run_name: str, step: int, stage_end: Optional[int] = None) -> Path:
    """`run_directory/checkpoints/<checkpoint_name>`; `stage_end` is `StageManager.stage_ending_at(step - 1)`."""
    return checkpoint_dir(run_directory) / checkpoint_name(step, run_name, stage_end)


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


# Settings whose value changes the numbers a run produces: a resume that silently mixes two configurations of these
# is a chimera, so `restore_checkpoint_if_resuming` compares them against the checkpoint (`allow_settings_change`
# overrides). The evaluation knobs are here because every forward consumes the global torch RNG (the meta check and
# the latent `randn_like` of the recurrence): how often validation runs, how many batches it draws and at how many
# depths it scores each of them all change the training stream itself, not just the reported numbers — and so does
# `dataloader_num_workers`, which changes how the workers hand batches over. Deliberately absent: paths, run_name,
# resume/logging/export knobs, `resume_warmup_steps` (a resume feature by design) and the dataset config (its own
# hash check).
NUMERICS_SETTINGS = (
    "world_batch_size",
    "micro_batch_size",
    "seed",
    "stage_base_lrs",
    "lr_schedule",
    "warmup_steps",
    "cooldown_steps",
    "min_lr",
    "grad_clip",
    "optimizer",
    "optim_config",
    "no_weight_decay_for_bias_and_norm_params",
    "block_size",
    "dataloader_num_workers",
    "sort_batches_by_length",
    "sequence_padding_multiple",
    "precision",
    "eval_step_interval",
    "eval_iters",
    "partial_depth_eval",
)


def check_settings_unchanged(
    metadata: CheckpointMetadata, settings: "Settings", model_config: dict[str, Any], allow_settings_change: bool
) -> None:
    """Fail a resume whose numerics-relevant settings (:data:`NUMERICS_SETTINGS`) or model config differ from what
    the checkpoint was written with, unless `allow_settings_change` is set. `model_config` is the current model's
    `RecurrentConfig.to_dict()`, compared whole against the stored one."""
    current = asdict(settings)
    changed = sorted(key for key in NUMERICS_SETTINGS if current.get(key) != metadata.settings.get(key))
    if model_config != metadata.model_config:
        changed.append("model_config")
    if changed and not allow_settings_change:
        raise ValueError(
            f"resuming with changed {changed}: the checkpoint was written with different values; "
            "set allow_settings_change: true to continue anyway (the run becomes a mix of two configurations)"
        )



def is_checkpoint_step(settings: Settings, done: int, stage_manager: StageManager) -> bool:
    """Whether to write a checkpoint after `done` completed optimizer steps.

    Three rules: every `save_step_interval` steps (0 disables), at the last step (`stage_manager.total_steps`) if
    `save_last_step`, and before every stage transition (`done` follows the last plain step of a stage,
    `StageManager.stage_ending_at(done - 1)`).
    """
    save_at_interval = settings.save_step_interval > 0 and done % settings.save_step_interval == 0
    save_at_last_step = settings.save_last_step and done >= stage_manager.total_steps
    save_at_stage_end = stage_manager.stage_ending_at(done - 1) is not None
    return save_at_interval or save_at_last_step or save_at_stage_end


def unwrap_compiled(model: Module) -> Module:
    """The plain module behind a `torch.compile` wrapper (state-dict keys stay stable across compiled/uncompiled
    runs; the loop reads `.step` / `.config` on it)."""
    return getattr(model, "_orig_mod", model)


def save_training_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optimizer, metadata: CheckpointMetadata
) -> None:
    """Write the model + optimizer state dicts and `metadata` to `path`."""
    state: dict[str, Any] = {
        "model": unwrap_compiled(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        **metadata.to_state(),
    }
    backend.save_checkpoint(path, state)


def load_training_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optimizer
) -> CheckpointMetadata:
    """Load the model and optimizer state in place and return the checkpoint's metadata.

    The metadata is read first, so a checkpoint of an older layout fails before anything is modified.
    """
    state = backend.load_checkpoint(path)
    metadata = CheckpointMetadata.from_state(state)
    unwrap_compiled(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return metadata
