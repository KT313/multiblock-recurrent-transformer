# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the settings schema: YAML loading (`config/tiny.yaml`), CLI overrides, validation, derived values."""

from pathlib import Path

import pytest

from training.settings import DataEntry, Settings, parse_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
CROW_YAML = REPO_ROOT / "config" / "crow_300m_final.yaml"
TINY_DATASET_CONFIG = "config/datasets/tiny.yaml"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"dataset_config": TINY_DATASET_CONFIG, "stage_base_lrs": [1e-3]}
    return Settings(**(base | overrides))  # type: ignore[arg-type]  # heterogeneous kwargs for a test helper


def test_data_entry_defaults() -> None:
    entry = DataEntry(prefix="p", data_dir="d")
    assert entry.weight == 1.0 and entry.data_signature is None


def test_parse_tiny_yaml() -> None:
    cfg = parse_settings(["--config", str(TINY_YAML)])
    assert isinstance(cfg, Settings)
    assert cfg.run_name == "tiny" and cfg.model_name == "tiny" and cfg.block_size == 256
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
    assert cfg.block_size == 2048 and cfg.model_name == "crow-300m-final" and cfg.optimizer == "ELLISAdam"
    assert (cfg.warmup_steps, cfg.cooldown_steps, cfg.save_step_interval, cfg.eval_step_interval) == (64, 64, 128, 16)
    assert not hasattr(cfg, "tokenizer_path") and not hasattr(cfg, "training_stages")


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
        parse_settings(["--dataset_config", TINY_DATASET_CONFIG])


def test_unknown_cli_key_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--config", str(TINY_YAML), "--nope", "1"])


def test_defaults_are_a_single_gpu_config() -> None:
    cfg = _settings()
    assert cfg.backend == "single_device" and cfg.world_batch_size % cfg.micro_batch_size == 0
    assert cfg.model_name == "crow-300m-final" and cfg.optimizer == "ELLISAdam"
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
