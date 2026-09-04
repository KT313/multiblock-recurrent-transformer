# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Checkpoint schema, naming, search, and save/load through the backend. Steps in file names are OPTIMIZER steps.

A checkpoint is one `torch.save` dict: the `"model"` and `"optimizer"` state dicts plus the `CheckpointMetadata`
fields. Older layouts have no loader (clean break): `CheckpointMetadata.from_state` raises on a missing key.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend, unwrap_compiled
from training.settings import Settings
from training.stage_manager import StageManager

CHECKPOINT_SUBDIR = "checkpoints"
CHECKPOINT_SUFFIX = ".pth"


@dataclass
class CheckpointMetadata:
    """
    Everything in a checkpoint besides the two state dicts.
    """

    step: int  # optimizer steps completed when the checkpoint was written
    stage: int  # stage the run is in at `step` (the one it enters next when written before a transition)
    rng: dict[str, Any]  # `Backend.rng_state()` after evaluation and logging of `step`
    settings: dict[str, Any]  # `asdict(Settings)` of the run (nested dataclasses like optim_config as plain dicts)
    model_config: dict[str, Any]  # `RecurrentConfig.to_dict()` of the trained model
    dataset_config_hash: str  # `ResolvedDataset.config_hash`
    validation_rows: dict[str, int]  # `ResolvedDataset.validation_rows`, {source: rows held out for validation}
    data_stream: dict[str, Any]  # `training.step.BatchStream.state_dict()`: rows read per source + the draw RNG

    def to_state(self) -> dict[str, Any]:
        """
        The metadata as the flat dict merged into the checkpoint (a shallow copy, tensors are not copied).
        """

        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CheckpointMetadata":
        """
        Read the metadata fields out of a loaded checkpoint dict; other keys (the state dicts) are ignored.
        """

        missing = [field.name for field in fields(cls) if field.name not in state]
        if missing:
            raise KeyError(
                f"checkpoint is missing the metadata key(s) {missing}; it was written by an older version of the "
                "training code and cannot be resumed (checkpoint formats are a clean break)"
            )
        return cls(**{field.name: state[field.name] for field in fields(cls)})


def checkpoint_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / CHECKPOINT_SUBDIR


def checkpoint_name(step: int, run_name: str, stage_end: Optional[int] = None) -> str:
    """
    `step-{step:08d}-{run_name}` plus `-stage-{i}_end` for the checkpoint written before a stage transition.
    """

    name = f"step-{step:08d}-{run_name}"
    if stage_end is not None:
        name += f"-stage-{stage_end}_end"
    return name + CHECKPOINT_SUFFIX


def checkpoint_path(run_directory: str | Path, run_name: str, step: int, stage_end: Optional[int] = None) -> Path:
    """
    `run_directory/checkpoints/<checkpoint_name>`; `stage_end` is `StageManager.stage_ending_at(step - 1)`.
    """

    return checkpoint_dir(run_directory) / checkpoint_name(step, run_name, stage_end)


def _step_from_name(path: Path) -> int:
    return int(path.name.split("-")[1])


def find_latest_checkpoint(out_dir: str | Path, run_name: str) -> Optional[Path]:
    """
    Most recently written checkpoint of `run_name` under `out_dir/checkpoints` (the step breaks ties), or None.
    """

    directory = checkpoint_dir(out_dir)
    pattern = re.compile(rf"^step-\d{{8}}-{re.escape(run_name)}(-stage-\d+_end)?{re.escape(CHECKPOINT_SUFFIX)}$")
    candidates = [path for path in directory.glob(f"step-*{CHECKPOINT_SUFFIX}") if pattern.match(path.name)]
    if not candidates:
        return None
    # the file time, not the step: after an explicit resume from an older checkpoint, a higher step of the
    # abandoned trajectory must not win the next plain resume
    return max(candidates, key=lambda path: (path.stat().st_mtime, _step_from_name(path)))


# `restore_checkpoint_if_resuming` compares EVERY `Settings` field against the checkpoint (`allow_settings_change`
# overrides), so a new field is checked until it is exempted here. Each group says why differing is harmless.
# Not exempt on purpose: the evaluation settings (`eval_step_interval`, `eval_iters`, `partial_depth_eval`). Every
# forward consumes the global torch RNG, so how often and how much validation runs changes the training stream.
SETTINGS_ALLOWED_TO_DIFFER_ON_RESUME = (
    # run identity and output location: where results go, not what is computed
    "run_name",
    "out_dir",
    # the resume feature's own knobs: they exist to differ between the original run and its resume
    "resume",
    "resume_checkpoint_path",
    # the override flags themselves: comparing them would make the escape hatches refuse their own use
    "allow_settings_change",
    "allow_dataset_change",
    # dataset reference: the dataset itself has its dedicated resume check (`check_dataset_unchanged` verifies the
    # dataset config HASH and the validation split), which a changed path or root alone does not trip
    "dataset_config",
    "dataset_dir",
    # model reference: the resolved `RecurrentConfig` is compared WHOLE (the `model_config` argument below), so the
    # built model is verified regardless of which architecture file / overrides produced it
    "model_architecture_config",
    "model_overwrite",
    # dataset-preparation conveniences: how missing data gets built, never what it contains
    "auto_prepare",
    "prepare_num_workers",
    "prepare_pass_workers",
    "prepare_max_parallel_downloads",
    # logging cadence: log steps read out metrics, they draw no RNG and change no state
    "log_step_interval",
    "log_gradient_metrics",
    # checkpoint cadence: when state is saved, not what it is
    "save_step_interval",
    "save_last_step",
    # wandb / export: reporting and post-run export only
    "logger_project",
    "wandb_offline",
    "wandb_enabled",
    "export_to_hf",
    "export_hf_path",
)

# Can never take effect on resume: the optimizer's parameter groups are restored from the checkpoint.
# `check_settings_unchanged` refuses a changed value even under `allow_settings_change`.
PARAM_GROUPING_SETTING = "no_weight_decay_for_bias_and_norm_params"


def check_settings_unchanged(
    metadata: CheckpointMetadata, settings: "Settings", model_config: dict[str, Any], allow_settings_change: bool
) -> None:
    """
    Fail a resume whose settings or model config differ from the checkpoint's, unless `allow_settings_change`.

    Fields in `SETTINGS_ALLOWED_TO_DIFFER_ON_RESUME` are skipped; a field the checkpoint did not store counts as
    changed. `model_config` is compared whole. A changed `PARAM_GROUPING_SETTING` is refused even with
    `allow_settings_change`: the restored optimizer keeps the checkpoint's groups.
    """

    current = asdict(settings)
    compared = [key for key in current if key not in SETTINGS_ALLOWED_TO_DIFFER_ON_RESUME]
    details = {
        key: f"checkpoint {metadata.settings[key]!r} != current {current[key]!r}"
        for key in compared
        if key in metadata.settings and metadata.settings[key] != current[key]
    }
    if PARAM_GROUPING_SETTING in details:
        raise ValueError(
            f"resuming with changed {PARAM_GROUPING_SETTING} ({details[PARAM_GROUPING_SETTING]}): the optimizer's "
            "parameter groups are restored from the checkpoint, so the new value would be silently ignored; "
            "allow_settings_change cannot override this; keep the checkpoint's value or start a fresh run"
        )
    details |= {
        key: "not stored in the checkpoint (written by an older version of the training code)"
        for key in compared
        if key not in metadata.settings
    }
    if model_config != metadata.model_config:
        differing = sorted(
            key
            for key in model_config.keys() | metadata.model_config.keys()
            if model_config.get(key) != metadata.model_config.get(key)
        )
        details["model_config"] = f"differs from the stored model config in {differing}"
    if details and not allow_settings_change:
        listed = "; ".join(f"{key}: {details[key]}" for key in sorted(details))
        raise ValueError(
            f"resuming with changed {sorted(details)}: {listed}; set allow_settings_change: true to continue "
            "anyway (the run becomes a mix of two configurations)"
        )



def is_checkpoint_step(settings: Settings, completed_steps: int, stage_manager: StageManager) -> bool:
    """
    Whether to write a checkpoint after `completed_steps` completed optimizer steps.

    Three rules: every `save_step_interval` steps (0 disables), at the last step (`stage_manager.total_steps`) if
    `save_last_step`, and before every stage transition (`completed_steps` follows the last plain step of a stage,
    `StageManager.stage_ending_at(completed_steps - 1)`).
    """

    save_at_interval = settings.save_step_interval > 0 and completed_steps % settings.save_step_interval == 0
    save_at_last_step = settings.save_last_step and completed_steps >= stage_manager.total_steps
    save_at_stage_end = stage_manager.stage_ending_at(completed_steps - 1) is not None
    return save_at_interval or save_at_last_step or save_at_stage_end


def save_training_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optimizer, metadata: CheckpointMetadata
) -> None:
    """
    Write the model + optimizer state dicts and `metadata` to `path`.
    """

    state: dict[str, Any] = {
        "model": unwrap_compiled(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        **metadata.to_state(),
    }
    backend.save_checkpoint(path, state)


def load_training_checkpoint(
    backend: Backend, path: str | Path, model: Module, optimizer: Optimizer
) -> CheckpointMetadata:
    """
    Load the model and optimizer state in place and return the checkpoint's metadata.

    The metadata is read first, so a checkpoint of an older layout fails before anything is modified. The
    optimizer's parameter-group hyperparameters must match the checkpoint's (`check_param_groups_unchanged`).
    """

    state = backend.load_checkpoint(path)
    metadata = CheckpointMetadata.from_state(state)
    unwrap_compiled(model).load_state_dict(state["model"])
    expected = _group_hyperparameters(optimizer)
    optimizer.load_state_dict(state["optimizer"])
    check_param_groups_unchanged(expected, _group_hyperparameters(optimizer))
    return metadata


UNCOMPARED_GROUP_KEYS = ("params", "lr")  # `lr` is rewritten by the schedule every step


def _group_hyperparameters(optimizer: Optimizer) -> list[dict[str, Any]]:
    return [
        {key: _plain(value) for key, value in group.items() if key not in UNCOMPARED_GROUP_KEYS}
        for group in optimizer.param_groups
    ]


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value


def check_param_groups_unchanged(expected: list[dict[str, Any]], loaded: list[dict[str, Any]]) -> None:
    """
    Fail when the optimizer state of a checkpoint carried other parameter-group hyperparameters than the current
    `optim_config` built (`optimizer.load_state_dict` replaces them, so the checkpoint's would silently win).
    """

    if len(expected) != len(loaded):
        raise ValueError(
            f"resuming with a different number of optimizer parameter groups ({len(loaded)} in the checkpoint, "
            f"{len(expected)} built); start a fresh run"
        )
    differing = [
        f"group {index}: {key}: checkpoint {after.get(key)!r} != current {before.get(key)!r}"
        for index, (before, after) in enumerate(zip(expected, loaded))
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    ]
    if differing:
        raise ValueError(
            "resuming with changed optimizer hyperparameters: " + "; ".join(differing) + "; the optimizer state is "
            "restored from the checkpoint and would silently keep the checkpoint's values; allow_settings_change "
            "cannot override this; keep the checkpoint's optim_config or start a fresh run"
        )
