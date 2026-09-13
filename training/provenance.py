# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Immutable configuration sidecars and append-only records of accepted resume setup.

Publication belongs under the training run lock, on rank zero after all setup checks. A resume record means
setup succeeded, not that a new optimizer update completed. Checkpoints remain the evidence of completed work.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from torch import Tensor

from data_preparation.lib.storage.atomic import write_atomically

if TYPE_CHECKING:
    from training.run import ResumePoint, RunState
    from training.settings import Settings

# Only known optimizer hyperparameters: never serialize parameters, moments, arbitrary checkpoint extras, or env.
OPTIMIZER_GROUP_FIELDS = (
    "lr", "init_lr", "betas", "eps", "weight_decay", "update_clipping", "atan_adam", "running_init",
    "decouple_wd", "state_bits", "amsgrad", "maximize", "foreach", "capturable", "differentiable", "fused",
)


def check_fresh_run_directory(run_directory: Path) -> None:
    """Reject stale provenance before a fresh run mutates an established destination; hold the run lock."""
    evidence = next((run_directory / name for name in ("run_config.json", "model_config.json", "train_report.json")
                     if (run_directory / name).exists() or (run_directory / name).is_symlink()), None)
    if evidence is None:
        # Only inspect the two immediate artifact directories and stop at the first matching record. Empty
        # precreated directories and unpublished temporary files are not evidence of an established run.
        for directory, pattern in (("resumes", "*.json"), ("checkpoints", "*.pth")):
            evidence = next((run_directory / directory).glob(pattern), None)
            if evidence is not None:
                break
    if evidence is not None:
        raise ValueError(
            f"refusing a fresh start in established run directory {run_directory}: found {evidence}; "
            "choose a new run_name or out_dir, or resume from a valid checkpoint with resume: true "
            "(and resume_checkpoint_path when needed). Existing configuration and history must be preserved"
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("optimizer provenance expects scalar hyperparameters, not tensor arrays")
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported optimizer provenance value: {type(value).__name__}")


def _differences(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Keep missing distinguishable from explicitly null; retain the checkpoint's actual schema."""
    return {
        key: {
            "checkpoint_present": key in previous, "checkpoint": previous.get(key),
            "requested_present": key in current, "requested": current.get(key),
        }
        for key in sorted(previous.keys() | current.keys())
        if key not in previous or key not in current or previous[key] != current[key]
    }


def code_revision() -> dict[str, Any]:
    """Read only the code checkout's HEAD; a commit alone does not attest to a clean working tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "revision": None}
    return {"status": "available", "revision": result.stdout.strip(), "working_tree_clean": "not_checked"}


def build_resume_record(
    state: RunState, resume: ResumePoint, *, accepted_at: str, attempt_id: str, revision: dict[str, Any],
) -> dict[str, Any]:
    """Construct a record without publication, RNG draws, model initialization, or settings mutation."""
    metadata = resume.metadata
    requested = asdict(state.settings)
    model_config = state.backend.plain_model(state.model).config.to_dict()
    groups = [
        {"index": index, "parameter_count": len(group["params"]),
         "hyperparameters": {key: _json_value(group[key]) for key in OPTIMIZER_GROUP_FIELDS if key in group}}
        for index, group in enumerate(state.optimizer.param_groups)
    ]
    # The optional build-ID interface also supports legacy checkpoints and pre-ID resolved datasets.
    dataset_id = getattr(state.dataset, "dataset_build_id", None)
    checkpoint_dataset_id = getattr(metadata, "dataset_build_id", None)
    return {
        "schema_version": 1,
        "event": "resume_setup_accepted",
        "accepted_at": accepted_at,
        "attempt_id": attempt_id,
        "completed_optimizer_updates_this_attempt": 0,
        "source_checkpoint": {"path": str(resume.checkpoint.resolve()), "step": metadata.step},
        "destination": {"run_name": state.settings.run_name, "run_directory": str(state.run_directory.resolve())},
        "requested_settings": requested,
        "effective": {
            # optim_config is a constructor request; restored parameter groups are authoritative at acceptance.
            "run_settings": {key: value for key, value in requested.items() if key != "optim_config"},
            "model_config": model_config,
            "stage_schedule": [
                {"name": stage.name, "tokens": stage.tokens, "base_lr": stage.base_lr,
                 "transition_pct": stage.transition_pct, "boundaries": asdict(boundary)}
                for stage, boundary in zip(state.stage_manager.stages, state.stage_manager.boundaries, strict=True)
            ],
            "optimizer": {
                "implementation": f"{type(state.optimizer).__module__}.{type(state.optimizer).__qualname__}",
                "parameter_groups_at_acceptance": groups,
                "learning_rate_policy": "each update replaces group lr with the current run's stage schedule",
                "unlisted_group_fields": "not recorded",
            },
        },
        "differences_from_checkpoint": {
            "settings": _differences(metadata.settings, requested),
            "model_config": _differences(metadata.model_config, model_config),
            "dataset": _differences(
                {"build_id": checkpoint_dataset_id, "config_hash": metadata.dataset_config_hash,
                 "source_rows": metadata.source_rows, "validation_rows": metadata.validation_rows},
                {"build_id": dataset_id, "config_hash": state.dataset.config_hash,
                 "source_rows": state.dataset.source_rows, "validation_rows": state.dataset.validation_rows},
            ),
        },
        "acknowledgements": {
            "allow_settings_change": state.settings.allow_settings_change,
            "allow_dataset_change": state.settings.allow_dataset_change,
        },
        "dataset": {"build_id": dataset_id, "identity_status": "available" if dataset_id else "unavailable",
                    "checkpoint_build_id": checkpoint_dataset_id,
                    "checkpoint_identity_status": "available" if checkpoint_dataset_id else "unavailable"},
        "code": revision,
        "original_history": "existing sidecars are preserved; missing sidecars contain checkpoint evidence only",
    }


def _publish_new_json(path: Path, record: dict[str, Any]) -> None:
    """The run lock serializes publishers; never replace an existing public record."""
    if path.exists():
        raise FileExistsError(f"refusing to replace existing configuration history: {path}")
    with write_atomically(path) as temporary:
        temporary.write_text(json.dumps(record, indent=4, allow_nan=False) + "\n", encoding="utf-8")


def record_run_config(settings: Settings, run_directory: Path) -> None:
    """Publish fresh settings only if absent. Called under the run lock after successful setup."""
    path = run_directory / "run_config.json"
    if not path.exists():
        _publish_new_json(path, asdict(settings))


def publish_configuration(state: RunState, resume: ResumePoint | None) -> Path | None:
    """Publish original evidence first and the accepted record last, after all setup prerequisites succeed."""
    record = None
    path = None
    if resume is not None:
        accepted_at = datetime.now(timezone.utc)
        attempt_id = uuid4().hex  # OS randomness, independent of training's Python/NumPy/Torch RNG streams.
        path = state.run_directory / "resumes" / f"{accepted_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{attempt_id}.json"
        record = build_resume_record(
            state, resume, accepted_at=accepted_at.isoformat(), attempt_id=attempt_id, revision=code_revision(),
        )
        # A checkpoint can itself be from a prior changed-settings resume. It proves configuration at that
        # checkpoint, not the original run's birth configuration. Keep this marker inside the atomic artifact.
        origin = {
            "schema_version": 1, "origin": "checkpoint_reconstruction", "original_run_configuration": "unknown",
            "source_checkpoint": str(resume.checkpoint.resolve()), "checkpoint_step": resume.metadata.step,
        }
        originals = {
            "run_config.json": resume.metadata.settings | {"_provenance": origin},
            "model_config.json": resume.metadata.model_config | {"_provenance": origin},
        }
    else:
        originals = {
            "run_config.json": asdict(state.settings),
            "model_config.json": state.backend.plain_model(state.model).config.to_dict(),
        }
    for name, original in originals.items():
        if not (state.run_directory / name).exists():
            _publish_new_json(state.run_directory / name, original)
    if path is not None and record is not None:
        _publish_new_json(path, record)
    return path
