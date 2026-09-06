# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the checkpoint schema (`CheckpointMetadata`), naming/search, the save decision, save→load→forward
bit-identity and optimizer state.
"""

import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT, build_model
from training.backend.base import unwrap_compiled
from training.backend.single_device import SingleDeviceBackend
from training.checkpoint import (
    CHECKPOINT_SUBDIR,
    PARAM_GROUPING_SETTING,
    SETTINGS_ALLOWED_TO_DIFFER_ON_RESUME,
    CheckpointMetadata,
    _step_from_name,
    check_param_groups_unchanged,
    check_settings_unchanged,
    checkpoint_dir,
    checkpoint_name,
    checkpoint_path,
    find_latest_checkpoint,
    is_checkpoint_step,
    load_training_checkpoint,
    save_training_checkpoint,
)
from training.optim import ELLISAdam, get_param_groups
from training.settings import OptimizerConfig, Settings
from training.stage_manager import StageManager
from training.testing.stages import resolved_stage

TINY_MODEL_ARCHITECTURE = Path(__file__).resolve().parent.parent / "config" / "model_architecture" / "tiny.yaml"
TINY_DATASET_CONFIG = Path(__file__).resolve().parent.parent / "config" / "datasets" / "tiny.yaml"


@pytest.fixture
def backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "dataset_config": str(TINY_DATASET_CONFIG),
        "model_architecture_config": str(TINY_MODEL_ARCHITECTURE),
        "stage_base_lrs": [1e-3],
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]  # heterogeneous kwargs for a test helper


def _metadata(backend: SingleDeviceBackend, model: RecurrentGPT, step: int = 1, **overrides: Any) -> CheckpointMetadata:
    values: dict[str, Any] = {
        "step": step,
        "stage": 0,
        "rng": backend.rng_state(),
        "settings": asdict(_settings(run_name="tiny", seed=42)),
        "model_config": model.config.to_dict(),
        "dataset_config_hash": "abc123",
        "validation_rows": {"synthetic_pretrain": 3, "synthetic_instruct": 1},
        "data_stream": {"consumed_rows": {"synthetic_pretrain": 12}, "draw_rng": (3, (1, 2), None)},
    }
    return CheckpointMetadata(**(values | overrides))


# --- schema ------------------------------------------------------------------------------------------------------------


def test_metadata_round_trip(backend: SingleDeviceBackend, tiny_model: RecurrentGPT) -> None:
    metadata = _metadata(backend, tiny_model, step=7)
    state = metadata.to_state()
    assert set(state) == {
        "step", "stage", "rng", "settings", "model_config", "dataset_config_hash", "validation_rows", "data_stream"
    }
    assert state["rng"] is metadata.rng  # a shallow copy: the RNG tensors are not duplicated
    restored = CheckpointMetadata.from_state({"model": {}, "optimizer": {}, **state})  # state dicts are ignored
    assert restored == metadata
    assert restored.step == 7 and restored.settings["seed"] == 42 and restored.model_config["block_size"] == 256


def test_metadata_from_state_missing_key_raises(backend: SingleDeviceBackend, tiny_model: RecurrentGPT) -> None:
    """
    No tolerance for older layouts (clean break): a missing key names itself instead of becoming a default.
    """

    state = _metadata(backend, tiny_model).to_state()
    del state["validation_rows"]
    state["config"] = state.pop("settings")  # the pre-restructure key name
    with pytest.raises(KeyError, match=r"missing the metadata key\(s\) \['settings', 'validation_rows'\]") as excinfo:
        CheckpointMetadata.from_state(state)
    assert "older version" in str(excinfo.value)


def test_metadata_field_order_matches_the_documented_layout() -> None:
    assert [f.name for f in fields(CheckpointMetadata)] == [
        "step",
        "stage",
        "rng",
        "settings",
        "model_config",
        "dataset_config_hash",
        "validation_rows",
        "data_stream",
    ]


# --- names, search, save decision ----------------------------------------------------------------------------------


def test_checkpoint_name_and_dir(tmp_path: Path) -> None:
    assert checkpoint_name(6, "tiny", stage_end=0) == "step-00000006-tiny-stage-0_end.pth"
    assert checkpoint_name(20, "tiny") == "step-00000020-tiny.pth"
    assert checkpoint_name(123456789, "r") == "step-123456789-r.pth"
    assert checkpoint_dir(tmp_path) == tmp_path / CHECKPOINT_SUBDIR
    assert checkpoint_dir(str(tmp_path)) == tmp_path / "checkpoints"


def test_checkpoint_path_names(tmp_path: Path) -> None:
    assert (
        checkpoint_path(tmp_path, "tiny", 6, stage_end=0)
        == tmp_path / "checkpoints" / "step-00000006-tiny-stage-0_end.pth"
    )
    assert checkpoint_path(tmp_path, "tiny", 20, stage_end=None) == tmp_path / "checkpoints" / "step-00000020-tiny.pth"
    assert checkpoint_path(str(tmp_path), "tiny", 20) == tmp_path / "checkpoints" / "step-00000020-tiny.pth"
    assert _step_from_name(checkpoint_path(tmp_path, "my-run", 14, stage_end=1)) == 14


def test_step_from_name() -> None:
    assert _step_from_name(Path("/x/step-00000014-tiny-stage-1_end.pth")) == 14
    assert _step_from_name(Path("step-00000020-my-run.pth")) == 20


def test_find_latest_checkpoint_picks_the_newest_file_including_stage_end_names(tmp_path: Path) -> None:
    """
    The most recently written checkpoint of the run wins, whatever its step; equal times fall back to the step.
    """

    assert find_latest_checkpoint(tmp_path, "tiny") is None
    d = checkpoint_dir(tmp_path)
    d.mkdir()
    names = [
        checkpoint_name(6, "tiny", stage_end=0),
        checkpoint_name(10, "tiny"),
        checkpoint_name(14, "tiny", stage_end=1),
        checkpoint_name(99, "other"),
        "step-00000500-tiny.txt",
    ]
    for n in names:
        (d / n).touch()
        os.utime(d / n, (1_000_000, 1_000_000))  # all written "at the same time"

    def latest(run_name: str) -> str:
        found = find_latest_checkpoint(tmp_path, run_name)
        assert found is not None
        return found.name

    assert latest("tiny") == "step-00000014-tiny-stage-1_end.pth"
    (d / checkpoint_name(9, "tiny")).touch()  # lexically later ("9" > "1") but a lower step
    os.utime(d / checkpoint_name(9, "tiny"), (1_000_000, 1_000_000))
    assert latest("tiny") == "step-00000014-tiny-stage-1_end.pth"
    (d / checkpoint_name(20, "tiny")).touch()
    os.utime(d / checkpoint_name(20, "tiny"), (1_000_000, 1_000_000))
    assert latest("tiny") == "step-00000020-tiny.pth"
    assert latest("other") == "step-00000099-other.pth"
    assert find_latest_checkpoint(tmp_path, "nothing") is None

    # an explicit resume from step 6 wrote a new step 10 later: that file is the newest and wins over 14 and 20
    os.utime(d / checkpoint_name(10, "tiny"), (2_000_000, 2_000_000))
    assert latest("tiny") == "step-00000010-tiny.pth"


@pytest.mark.parametrize("foreign", ["step-00000099-tiny-v2.pth", "step-00000099-other-tiny.pth"])
def test_find_latest_checkpoint_ignores_runs_with_a_longer_name(tmp_path: Path, foreign: str) -> None:
    d = checkpoint_dir(tmp_path)
    d.mkdir()
    (d / checkpoint_name(1, "tiny")).touch()
    (d / foreign).touch()
    found = find_latest_checkpoint(tmp_path, "tiny")
    assert found is not None and found.name == "step-00000001-tiny.pth"


def test_is_checkpoint_step_table() -> None:
    """
    Three rules: every `save_step_interval` steps, the last step (`total_steps`) if `save_last_step`, and the
    step after the last plain step of a stage (`stage_ending_at`). Two stages of 12 + 8 optimizer steps (4 × 256
    tokens each, no transition): stage 0 ends with step 12, the run with step 20.
    """

    stages = [
        resolved_stage("a", tokens=12 * 4 * 256, base_lr=1e-3, transition_pct=0.0),
        resolved_stage("b", tokens=8 * 4 * 256, base_lr=1e-3, transition_pct=0.0),
    ]
    stage_manager = StageManager(stages, world_batch_size=4, block_size=256)
    assert stage_manager.total_steps == 20 and stage_manager.stage_ending_at(11) == 0
    settings = _settings(save_step_interval=8, save_last_step=True)
    assert [s for s in range(1, 21) if is_checkpoint_step(settings, s, stage_manager)] == [8, 12, 16, 20]
    assert is_checkpoint_step(settings, 25, stage_manager)  # past the end still counts as the last step
    no_interval = _settings(save_step_interval=0, save_last_step=False)
    assert [s for s in range(1, 21) if is_checkpoint_step(no_interval, s, stage_manager)] == [12]  # stage end only
    last_only = _settings(save_step_interval=0, save_last_step=True)
    assert [s for s in range(1, 26) if is_checkpoint_step(last_only, s, stage_manager)] == [12, *range(20, 26)]


# --- the resume compatibility check ------------------------------------------------------------------------------------

# One differing value per compared setting, i.e. per Settings field NOT in the exemption tuple (the defaults are in
# `training/settings.py`); `test_every_settings_field_is_classified` keeps this table complete.
CHANGED_COMPARED_VALUES: dict[str, Any] = {
    "stage_base_lrs": [2e-3],
    "seed": 7,
    "block_size": 128,
    "sort_batches_by_length": False,
    "sequence_padding_multiple": 64,
    "backend": "future_ddp",
    "precision": "32",
    "compile_model": True,
    "gradient_checkpointing": True,
    "micro_batch_size": 2,
    "world_batch_size": 2048,
    "optimizer": "AdamW",
    "optim_config": OptimizerConfig(lr=2e-4, weight_decay=4e-5, betas=(0.9, 0.95)),
    "no_weight_decay_for_bias_and_norm_params": False,
    "grad_clip": 0.5,
    "lr_schedule": "cosine",
    "warmup_steps": 5,
    "cooldown_steps": 5,
    "min_lr": 1e-6,
    "eval_step_interval": 7,
    "eval_iters": 3,
    "partial_depth_eval": [2],
}

# Compared like the fields above, but only valid together (`Settings._check_packing`): a resume that switches to
# sequence packing changes all three at once, so they are tested as one switch.
CHANGED_TOGETHER: dict[str, Any] = {"pack_sequences": True, "tokens_per_micro_batch": 4096, "tokens_per_step": 8192}


def test_every_settings_field_is_classified() -> None:
    """
    Every Settings field is either exempted or compared on resume (with a differing value in the table above),
    and every exempted name is a real field; a typo in the exemption tuple would silently compare nothing.
    """

    field_names = {f.name for f in fields(Settings)}
    exempt = set(SETTINGS_ALLOWED_TO_DIFFER_ON_RESUME)
    assert exempt <= field_names
    assert set(CHANGED_COMPARED_VALUES) | set(CHANGED_TOGETHER) == field_names - exempt
    assert not set(CHANGED_COMPARED_VALUES) & set(CHANGED_TOGETHER)
    assert PARAM_GROUPING_SETTING in field_names and PARAM_GROUPING_SETTING not in exempt


def test_check_settings_unchanged_catches_a_switch_to_sequence_packing(
    backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    The packing fields are compared too: a resume that turns packing on (all three fields change together) is
    refused by name unless `allow_settings_change`.
    """

    metadata = _metadata(backend, tiny_model)
    config = tiny_model.config.to_dict()
    packed = _settings(run_name="tiny", seed=42, **CHANGED_TOGETHER)
    with pytest.raises(
        ValueError, match=r"resuming with changed \['pack_sequences', 'tokens_per_micro_batch', 'tokens_per_step'\]"
    ):
        check_settings_unchanged(metadata, packed, config, False)
    check_settings_unchanged(metadata, packed, config, True)


def test_check_settings_unchanged_catches_every_compared_setting(
    backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    Each compared field fails a resume on its own, the eval knobs included: every forward draws from the global
    torch RNG, so how often and how widely validation runs changes the training stream itself. The weight-decay
    grouping flag is refused even under `allow_settings_change`: the restored optimizer keeps the checkpoint's
    parameter groups, so the new value could never take effect.
    """

    metadata = _metadata(backend, tiny_model)
    config = tiny_model.config.to_dict()
    check_settings_unchanged(metadata, _settings(run_name="tiny", seed=42), config, False)  # nothing changed
    for key, value in CHANGED_COMPARED_VALUES.items():
        changed = _settings(**({"run_name": "tiny", "seed": 42} | {key: value}))
        if key == PARAM_GROUPING_SETTING:
            for allow in (False, True):  # non-overridable
                with pytest.raises(ValueError, match="parameter groups are restored from the checkpoint"):
                    check_settings_unchanged(metadata, changed, config, allow)
        else:
            with pytest.raises(ValueError, match=rf"resuming with changed \['{key}'\].*checkpoint .* != current "):
                check_settings_unchanged(metadata, changed, config, False)
            check_settings_unchanged(metadata, changed, config, True)  # allow_settings_change


def test_check_settings_unchanged_ignores_the_exempt_settings(
    backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    Paths, run bookkeeping, logging and the resume features themselves may differ from the checkpoint.
    """

    metadata = _metadata(backend, tiny_model)
    harmless = _settings(
        run_name="tiny", seed=42, out_dir="elsewhere", log_step_interval=4, save_step_interval=3,
        wandb_enabled=False, export_to_hf=True, auto_prepare=False,
        model_architecture_config="moved/elsewhere/tiny.yaml",  # the resolved model config is what gets compared
    )
    check_settings_unchanged(metadata, harmless, tiny_model.config.to_dict(), False)
    with pytest.raises(ValueError, match=r"resuming with changed \['model_config'\].*n_embd"):
        check_settings_unchanged(metadata, harmless, {"n_embd": 1}, False)


def test_check_settings_unchanged_treats_a_missing_stored_key_as_changed(
    backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    A checkpoint from an older code version that did not store a (newer) compared field fails the resume naming
    it (fail fast, `allow_settings_change` overrides); a stored key that is no longer a Settings field is ignored.
    """

    metadata = _metadata(backend, tiny_model)
    del metadata.settings["grad_clip"]
    metadata.settings["a_removed_setting"] = 123  # not a field any more: ignored
    config = tiny_model.config.to_dict()
    current = _settings(run_name="tiny", seed=42)
    with pytest.raises(ValueError, match=r"resuming with changed \['grad_clip'\].*not stored in the checkpoint"):
        check_settings_unchanged(metadata, current, config, False)
    check_settings_unchanged(metadata, current, config, True)  # allow_settings_change


# --- save / load -----------------------------------------------------------------------------------------------------


def test_unwrap_compiled_strips_compile_wrapper(tiny_model: RecurrentGPT) -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    assert unwrap_compiled(tiny_model) is tiny_model
    assert unwrap_compiled(Wrapper(tiny_model)) is tiny_model


def _train_one_step(model: RecurrentGPT) -> tuple[ELLISAdam, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(model, 4e-5), lr=1e-3, betas=(0.9, 0.95))
    loss = model(x, labels=x, num_steps=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    opt.step()
    opt.zero_grad()
    return opt, x


def test_save_load_forward_bit_identical(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    opt, x = _train_one_step(tiny_model)
    path = checkpoint_path(tmp_path, "tiny", 1)
    metadata = _metadata(backend, tiny_model, step=1)
    save_training_checkpoint(backend, path, tiny_model, opt, metadata)
    assert path.exists()
    raw = backend.load_checkpoint(path)
    assert set(raw) == {"model", "optimizer", *metadata.to_state()}

    torch.manual_seed(999)
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    fresh_opt = ELLISAdam(get_param_groups(fresh, 4e-5), lr=1e-3, betas=(0.9, 0.95))
    restored = load_training_checkpoint(backend, path, fresh, fresh_opt)
    assert restored.step == 1 and restored.stage == 0 and restored.settings["seed"] == 42
    # the nested optim_config dataclass round-trips as the plain dict `asdict` made of it, so the resume
    # compatibility check compares it against `asdict(current_settings)["optim_config"]` value by value
    assert restored.settings["optim_config"] == asdict(OptimizerConfig())
    assert restored.dataset_config_hash == "abc123" and restored.validation_rows == metadata.validation_rows
    assert restored.model_config == tiny_model.config.to_dict()
    assert torch.equal(restored.rng["torch"], metadata.rng["torch"])

    tiny_model.eval()
    fresh.eval()
    with torch.no_grad():  # the recurrent state is drawn from the global RNG: seed before each forward
        torch.manual_seed(7)
        a = tiny_model(x, return_logits=True, num_steps=(0, 2))["logits"]
        torch.manual_seed(7)
        b = fresh(x, return_logits=True, num_steps=(0, 2))["logits"]
    assert a is not None and b is not None and torch.equal(a, b)
    for (n1, p1), (n2, p2) in zip(tiny_model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)

    # optimizer state restored: the next step on both optimizers is identical
    for p, q in zip(tiny_model.parameters(), fresh.parameters()):
        assert torch.equal(opt.state[p]["exp_avg"], fresh_opt.state[q]["exp_avg"])
        assert torch.equal(opt.state[p]["exp_avg_sq"], fresh_opt.state[q]["exp_avg_sq"])
        assert opt.state[p]["step"].item() == fresh_opt.state[q]["step"].item() == 1
    assert fresh_opt.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    tiny_model.train()
    fresh.train()
    for model, optimizer in ((tiny_model, opt), (fresh, fresh_opt)):
        torch.manual_seed(3)
        loss = model(x, labels=x, num_steps=(0, 2))["loss"]
        assert loss is not None
        loss.backward()
        optimizer.step()
    for p, q in zip(tiny_model.parameters(), fresh.parameters()):
        assert torch.equal(p, q)


def test_load_of_an_older_layout_fails_before_touching_the_model(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    opt, _ = _train_one_step(tiny_model)
    path = tmp_path / "old.pth"
    state = {"model": tiny_model.state_dict(), "optimizer": opt.state_dict(), "step": 1, "config": {}}
    backend.save_checkpoint(path, state)
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    before = [p.detach().clone() for p in fresh.parameters()]
    with pytest.raises(KeyError, match="missing the metadata key"):
        load_training_checkpoint(backend, path, fresh, ELLISAdam(get_param_groups(fresh, 4e-5), lr=1e-3))
    assert all(torch.equal(a, b) for a, b in zip(before, fresh.parameters()))


def test_compiled_wrapper_is_unwrapped_for_state_dict(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    State-dict keys must not carry an `_orig_mod.` prefix when the model is a torch.compile wrapper.
    """

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    opt, _ = _train_one_step(tiny_model)
    path = tmp_path / "w.pth"
    save_training_checkpoint(backend, path, Wrapper(tiny_model), opt, _metadata(backend, tiny_model))
    keys = set(backend.load_checkpoint(path)["model"].keys())
    assert keys == set(tiny_model.state_dict().keys())
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    load_training_checkpoint(backend, path, Wrapper(fresh), ELLISAdam(get_param_groups(fresh, 4e-5), lr=1e-3, betas=(0.9, 0.95)))
    assert all(torch.equal(a, b) for a, b in zip(tiny_model.parameters(), fresh.parameters()))


def test_load_refuses_changed_optimizer_hyperparameters(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """
    `optimizer.load_state_dict` replaces the parameter groups with the checkpoint's, so an `optim_config` that
    differs from the checkpoint's would silently lose: the load fails and names the differing keys.
    """

    opt, _ = _train_one_step(tiny_model)  # weight_decay 4e-5, betas (0.9, 0.95)
    path = checkpoint_path(tmp_path, "tiny", 1)
    save_training_checkpoint(backend, path, tiny_model, opt, _metadata(backend, tiny_model, step=1))

    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    changed = ELLISAdam(get_param_groups(fresh, 0.5), lr=1e-3, betas=(0.8, 0.9))
    with pytest.raises(ValueError, match=r"group 0: betas: checkpoint \(0\.9, 0\.95\) != current \(0\.8, 0\.9\)") as info:
        load_training_checkpoint(backend, path, fresh, changed)
    assert "weight_decay: checkpoint 4e-05 != current 0.5" in str(info.value)
    assert "allow_settings_change cannot override this" in str(info.value)

    same = ELLISAdam(get_param_groups(fresh, 4e-5), lr=1e-3, betas=(0.9, 0.95))
    load_training_checkpoint(backend, path, fresh, same)  # equal hyperparameters load fine, whatever the LR is


def test_check_param_groups_unchanged() -> None:
    check_param_groups_unchanged([{"betas": (0.9, 0.95)}], [{"betas": (0.9, 0.95)}])
    with pytest.raises(ValueError, match="different number of optimizer parameter groups"):
        check_param_groups_unchanged([{}], [{}, {}])
    with pytest.raises(ValueError, match="group 1: eps: checkpoint 1e-08 != current 1e-06"):
        check_param_groups_unchanged([{"eps": 1e-6}, {"eps": 1e-6}], [{"eps": 1e-6}, {"eps": 1e-8}])
