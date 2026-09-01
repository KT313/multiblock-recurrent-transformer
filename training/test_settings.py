# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the settings schema: YAML loading (`config/tiny.yaml`, the complete `config/crow_300m_final.yaml`), CLI
overrides, validation, derived values."""

from dataclasses import MISSING, asdict, fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from training.settings import (
    NON_NEGATIVE_SETTINGS,
    POSITIVE_SETTINGS,
    REQUIRED_SETTINGS,
    Settings,
    parse_settings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
CROW_YAML = REPO_ROOT / "config" / "crow_300m_final.yaml"
TINY_DATASET_CONFIG = "config/datasets/tiny.yaml"
TINY_MODEL_ARCHITECTURE = "config/model_architecture/tiny.yaml"

# The values config/crow_300m_final.yaml set explicitly before it listed every key; every other key = the default.
CROW_EXPLICIT: dict[str, Any] = {
    "run_name": "crow-300m-final",
    "out_dir": "outputs",
    "resume": True,
    "seed": 233,
    "model_architecture_config": "config/model_architecture/crow_300m_final.yaml",
    "block_size": 2048,
    "dataset_config": "config/datasets/crow_300m_final.yaml",
    "dataset_dir": "dataset",
    "auto_prepare": True,
    "prepare_num_workers": 4,
    "stage_base_lrs": [3e-4, 1e-4, 5e-5],
    "backend": "single_device",
    "precision": "bf16-mixed",
    "compile_model": True,
    "gradient_checkpointing": False,
    "micro_batch_size": 4,
    "world_batch_size": 1024,
    "dataloader_num_workers": 8,
    "sort_batches_by_length": True,
    "sequence_padding_multiple": 128,
    "optimizer": "ELLISAdam",
    "optim_config": {
        "lr": 1e-4,
        "weight_decay": 4e-5,
        "betas": [0.9, 0.95],
        "update_clipping": True,
        "atan_adam": True,
        "running_init": True,
        "decouple_wd": True,
    },
    "no_weight_decay_for_bias_and_norm_params": True,
    "grad_clip": 1.0,
    "lr_schedule": "trapezoid",
    "warmup_steps": 64,
    "cooldown_steps": 64,
    "min_lr": 0.0,
    "resume_warmup_steps": 8,
    "log_step_interval": 1,
    "log_gradient_metrics": True,
    "eval_step_interval": 16,
    "eval_iters": 50,
    "partial_depth_eval": [1, 2, 4, 8, 16],
    "save_step_interval": 128,
    "save_last_step": True,
    "logger_project": "recurtrain-baseline",
    "wandb_offline": True,
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "dataset_config": TINY_DATASET_CONFIG,
        "model_architecture_config": TINY_MODEL_ARCHITECTURE,
        "stage_base_lrs": [1e-3],
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]  # heterogeneous kwargs for a test helper


def _field_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(Settings):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
    return out


def test_parse_tiny_yaml() -> None:
    cfg = parse_settings(["--config", str(TINY_YAML)])
    assert isinstance(cfg, Settings)
    assert cfg.run_name == "tiny" and cfg.block_size == 256
    assert cfg.model_architecture_config == TINY_MODEL_ARCHITECTURE and cfg.model_overwrite == {}
    assert cfg.dataset_config == TINY_DATASET_CONFIG and cfg.dataset_dir == "dataset"
    assert cfg.auto_prepare is True and cfg.prepare_num_workers == 1 and cfg.allow_dataset_change is False
    assert cfg.stage_base_lrs == pytest.approx([3e-4, 1e-4, 5e-5])
    assert cfg.backend == "single_device" and cfg.precision == "bf16-mixed"
    assert cfg.resume is False and cfg.wandb_enabled is False
    assert (cfg.micro_batch_size, cfg.world_batch_size) == (2, 4)
    assert cfg.optimizer == "AdamW"
    assert cfg.optim_config["lr"] == pytest.approx(3e-4) and list(cfg.optim_config["betas"]) == [0.9, 0.95]
    assert (cfg.warmup_steps, cfg.cooldown_steps, cfg.eval_step_interval, cfg.eval_iters) == (2, 2, 8, 2)
    assert cfg.partial_depth_eval == [1]


def test_parse_crow_yaml() -> None:
    cfg = parse_settings(["--config", str(CROW_YAML)])
    assert cfg.dataset_config == "config/datasets/crow_300m_final.yaml"
    assert cfg.stage_base_lrs == pytest.approx([3e-4, 1e-4, 5e-5])
    assert cfg.block_size == 2048 and cfg.optimizer == "ELLISAdam"
    assert cfg.model_architecture_config == "config/model_architecture/crow_300m_final.yaml"
    assert (cfg.warmup_steps, cfg.cooldown_steps, cfg.save_step_interval, cfg.eval_step_interval) == (64, 64, 128, 16)
    assert not hasattr(cfg, "tokenizer_path") and not hasattr(cfg, "training_stages")


def test_crow_yaml_lists_every_settings_field() -> None:
    """The thesis run config is the template: exactly the set of `Settings` fields, nothing stale, nothing missing."""
    with open(CROW_YAML, encoding="utf-8") as fp:
        keys = set(yaml.safe_load(fp))
    assert keys == {f.name for f in fields(Settings)}


def test_crow_yaml_effective_values_are_unchanged() -> None:
    """Listing every key must not change the run: explicit values as before, every other key at its default."""
    cfg = parse_settings(["--config", str(CROW_YAML)])
    expected = _field_defaults() | CROW_EXPLICIT
    assert set(expected) == {f.name for f in fields(Settings)}
    assert asdict(cfg) == expected
    assert cfg.model_overwrite == {} and cfg.wandb_enabled is True and cfg.export_to_hf is False
    assert cfg.prepare_max_parallel_downloads == 2 and cfg.allow_dataset_change is False
    assert cfg.resume_checkpoint_path is None and cfg.export_hf_path is None


def test_run_configs_reference_existing_architecture_and_dataset_configs() -> None:
    for path in (TINY_YAML, CROW_YAML):
        cfg = parse_settings(["--config", str(path)])
        assert (REPO_ROOT / cfg.model_architecture_config).is_file(), cfg.model_architecture_config
        assert (REPO_ROOT / cfg.dataset_config).is_file(), cfg.dataset_config
        assert cfg.model_architecture_config.startswith("config/model_architecture/")


def test_parse_without_model_architecture_config_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--dataset_config", TINY_DATASET_CONFIG, "--stage_base_lrs", "[1e-3]"])


def test_validation_empty_model_architecture_config() -> None:
    with pytest.raises(ValueError, match="model_architecture_config is required"):
        _settings(model_architecture_config="")


def test_cli_overrides_win_over_yaml() -> None:
    cfg = parse_settings(
        [
            "--config",
            str(TINY_YAML),
            "--seed",
            "7",
            "--micro_batch_size",
            "4",
            "--partial_depth_eval",
            "[1, 2]",
            "--out_dir",
            "/nowhere/x",
            "--warmup_steps",
            "3",
            "--model_overwrite",
            '{"n_embd": 32}',
            "--export_to_hf",
            "true",
            "--stage_base_lrs",
            "[1e-3, 2e-3, 3e-3]",
            "--auto_prepare",
            "false",
            "--dataset_dir",
            "/data/x",
        ]
    )
    assert cfg.seed == 7 and cfg.micro_batch_size == 4 and cfg.partial_depth_eval == [1, 2]
    assert cfg.out_dir == "/nowhere/x" and cfg.warmup_steps == 3
    assert cfg.model_overwrite == {"n_embd": 32} and cfg.export_to_hf is True
    assert cfg.stage_base_lrs == pytest.approx([1e-3, 2e-3, 3e-3])
    assert cfg.auto_prepare is False and cfg.dataset_dir == "/data/x"
    assert cfg.dataset_config == TINY_DATASET_CONFIG  # untouched by the overrides


def test_parse_settings_reads_sys_argv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train.py", "--config", str(TINY_YAML), "--seed", "3"])
    assert parse_settings().seed == 3


def test_parse_without_config_requires_dataset_config() -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--stage_base_lrs", "[1e-3]"])


def test_parse_without_stage_base_lrs_is_rejected() -> None:
    with pytest.raises(ValueError, match="stage_base_lrs"):
        parse_settings(
            ["--dataset_config", TINY_DATASET_CONFIG, "--model_architecture_config", TINY_MODEL_ARCHITECTURE]
        )


def test_unknown_cli_key_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--config", str(TINY_YAML), "--nope", "1"])


def test_defaults_are_a_single_gpu_config() -> None:
    cfg = _settings()
    assert cfg.backend == "single_device" and cfg.world_batch_size % cfg.micro_batch_size == 0
    assert cfg.model_overwrite == {} and cfg.optimizer == "ELLISAdam"
    assert cfg.optim_config == {"lr": 1e-4, "weight_decay": 4e-5, "betas": (0.9, 0.95)}
    assert cfg.out_dir == "outputs" and cfg.resume is True
    assert cfg.export_to_hf is False and cfg.export_hf_path is None
    assert cfg.dataset_dir == "dataset" and cfg.auto_prepare is True and cfg.prepare_num_workers == 2
    assert cfg.prepare_max_parallel_downloads == 2
    assert cfg.allow_dataset_change is False
    assert cfg.gradient_accumulation_steps == 1024 // 4


def test_validation_empty_dataset_config() -> None:
    with pytest.raises(ValueError, match="dataset_config is required"):
        _settings(dataset_config="")


def test_validation_empty_stage_base_lrs() -> None:
    with pytest.raises(ValueError, match="stage_base_lrs"):
        _settings(stage_base_lrs=[])


def test_validation_negative_stage_base_lr() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _settings(stage_base_lrs=[1e-3, -1.0])


def test_validation_batch_divisibility() -> None:
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        _settings(micro_batch_size=3, world_batch_size=8)


def test_validation_of_nonsensical_batch_sizes() -> None:
    """A `micro_batch_size` of 0 used to die with a raw ZeroDivisionError and a negative one made the micro-batch
    loop of a step run zero times — the run "trained" and reported loss 0.0. Both fail at settings time now, and so
    does a world batch smaller than one micro-batch."""
    for bad in (0, -4):
        with pytest.raises(ValueError, match="micro_batch_size must be positive"):
            _settings(micro_batch_size=bad, world_batch_size=8)
        with pytest.raises(ValueError, match="world_batch_size must be positive"):
            _settings(micro_batch_size=4, world_batch_size=bad)
    with pytest.raises(ValueError, match=r"world_batch_size \(4\) must be >= micro_batch_size \(8\)"):
        _settings(micro_batch_size=8, world_batch_size=4)
    assert _settings(micro_batch_size=8, world_batch_size=8).gradient_accumulation_steps == 1


def test_validation_runs_for_yaml_configs_too(tmp_path: Path) -> None:
    """`parse_settings` goes through `Settings.__post_init__`, so a bad YAML value is rejected the same way."""
    yaml = tmp_path / "bad.yaml"
    yaml.write_text(TINY_YAML.read_text().replace("micro_batch_size: 2", "micro_batch_size: 3"))
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        parse_settings(["--config", str(yaml)])


def test_settings_do_not_touch_the_filesystem(tmp_path: Path) -> None:
    """Existence of the dataset config / data is checked by the resolver, not by the schema."""
    cfg = _settings(dataset_config=str(tmp_path / "missing.yaml"), dataset_dir=str(tmp_path / "nowhere"))
    assert cfg.dataset_config.endswith("missing.yaml")


@pytest.mark.parametrize("micro,world,expected", [(2, 4, 2), (1, 4, 4), (4, 4, 1), (2, 64, 32)])
def test_gradient_accumulation_steps(micro: int, world: int, expected: int) -> None:
    assert _settings(micro_batch_size=micro, world_batch_size=world).gradient_accumulation_steps == expected


# --- the value-rule tables ---------------------------------------------------------------------------------------


def test_value_rule_tables_name_real_fields_and_do_not_overlap() -> None:
    """The three tables `Settings.__post_init__` loops over hold field names, each field in at most one of them."""
    names = {f.name for f in fields(Settings)}
    tables = [set(REQUIRED_SETTINGS), set(POSITIVE_SETTINGS), set(NON_NEGATIVE_SETTINGS)]
    for table in tables:
        assert table and table <= names
    assert sum(len(t) for t in tables) == len(set().union(*tables))


@pytest.mark.parametrize("name", sorted(POSITIVE_SETTINGS))
def test_positive_settings_are_rejected_at_zero(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        _settings(**{name: 0})


@pytest.mark.parametrize("name", sorted(NON_NEGATIVE_SETTINGS))
def test_non_negative_settings_are_rejected_below_zero(name: str) -> None:
    assert _settings(**{name: 0}) is not None
    with pytest.raises(ValueError, match=f"{name} must be >= 0"):
        _settings(**{name: -1})


@pytest.mark.parametrize("name", sorted(REQUIRED_SETTINGS))
def test_required_settings_are_rejected_when_empty(name: str) -> None:
    empty: Any = [] if name == "stage_base_lrs" else ""
    with pytest.raises(ValueError, match=f"{name} is required"):
        _settings(**{name: empty})
