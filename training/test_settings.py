# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the settings schema: YAML loading (`config/tiny.yaml`), CLI overrides, validation, derived values."""

from pathlib import Path

import pytest

from training.settings import DataEntry, Settings, StageConfig, parse_settings
from training.stage_manager import StageManager, TrainingStage

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"


def _stage(
    name: str = "s", tokens: int = 1024, lr: float = 1e-3, data_dir: str = "dataset/tiny/pretrain"
) -> StageConfig:
    entry = DataEntry(prefix="p", data_dir=data_dir)
    return StageConfig(name=name, tokens=tokens, base_lr=lr, train_data=[entry], val_data=[entry])


def test_data_entry_and_stage_config_defaults() -> None:
    entry = DataEntry(prefix="p", data_dir="d")
    assert entry.weight == 1.0 and entry.data_signature is None
    assert _stage().transition_pct == 0.0


def test_parse_tiny_yaml(tiny_tokenizer_path: Path) -> None:
    cfg = parse_settings(["--config", str(TINY_YAML), "--tokenizer_path", str(tiny_tokenizer_path)])
    assert isinstance(cfg, Settings)
    assert cfg.run_name == "tiny" and cfg.model_name == "tiny" and cfg.block_size == 256
    assert cfg.tokenizer_path == str(tiny_tokenizer_path)
    assert cfg.backend == "single_device" and cfg.precision == "bf16-mixed"
    assert cfg.resume is False and cfg.wandb_enabled is False
    assert (cfg.micro_batch_size, cfg.world_batch_size) == (2, 4)
    assert cfg.optimizer == "AdamW"
    assert cfg.optim_config["lr"] == pytest.approx(3e-4) and list(cfg.optim_config["betas"]) == [0.9, 0.95]
    assert (cfg.warmup_steps, cfg.cooldown_steps, cfg.eval_step_interval, cfg.eval_iters) == (2, 2, 8, 2)
    assert cfg.partial_depth_eval == [1]
    assert len(cfg.training_stages) == 3
    s0, s1, s2 = cfg.training_stages
    assert all(isinstance(s, StageConfig) for s in cfg.training_stages)
    assert [s.name for s in cfg.training_stages] == ["pretrain_a", "pretrain_b", "finetune"]
    assert [s.tokens for s in cfg.training_stages] == [8192, 8192, 4096]
    assert [s.transition_pct for s in cfg.training_stages] == [0.25, 0.25, 0.0]
    assert s0.base_lr == pytest.approx(3e-4) and s2.base_lr == pytest.approx(5e-5)
    assert isinstance(s1.train_data[0], DataEntry)
    assert s1.train_data[0].prefix == "pretrain-train" and s1.train_data[0].weight == 1.0
    assert s2.val_data[0].data_dir == "dataset/mixtures/tiny/tiny_mixture/validation"
    assert s2.val_data[0].data_signature == {
        "keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output",
    }  # fmt: skip


def test_cli_overrides_win_over_yaml(tiny_tokenizer_path: Path) -> None:
    cfg = parse_settings(
        [
            "--config",
            str(TINY_YAML),
            "--tokenizer_path",
            str(tiny_tokenizer_path),
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
        ]
    )
    assert cfg.seed == 7 and cfg.micro_batch_size == 4 and cfg.partial_depth_eval == [1, 2]
    assert cfg.out_dir == "/nowhere/x" and cfg.warmup_steps == 3
    assert cfg.model_overwrite == {"n_embd": 32} and cfg.export_to_hf is True
    assert cfg.training_stages[0].base_lr == pytest.approx(3e-4)  # untouched by the overrides


def test_parse_settings_reads_sys_argv_by_default(tiny_tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", str(TINY_YAML), "--tokenizer_path", str(tiny_tokenizer_path), "--seed", "3"],
    )
    assert parse_settings().seed == 3


def test_parse_without_config_uses_defaults_but_requires_stages(tiny_tokenizer_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        parse_settings(["--tokenizer_path", str(tiny_tokenizer_path)])


def test_unknown_cli_key_is_rejected(tiny_tokenizer_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_settings(["--config", str(TINY_YAML), "--tokenizer_path", str(tiny_tokenizer_path), "--nope", "1"])


def test_defaults_are_a_single_gpu_config(tiny_tokenizer_path: Path) -> None:
    cfg = Settings(training_stages=[_stage()], tokenizer_path=str(tiny_tokenizer_path))
    assert cfg.backend == "single_device" and cfg.world_batch_size % cfg.micro_batch_size == 0
    assert cfg.model_name == "crow-300m-final" and cfg.optimizer == "ELLISAdam"
    assert cfg.optim_config == {"lr": 1e-4, "weight_decay": 4e-5, "betas": (0.9, 0.95)}
    assert cfg.out_dir == "outputs" and cfg.resume is True
    assert cfg.export_to_hf is False and cfg.export_hf_path is None
    assert cfg.gradient_accumulation_steps == 1024 // 4


def test_validation_empty_stages(tiny_tokenizer_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        Settings(training_stages=[], tokenizer_path=str(tiny_tokenizer_path))


def test_validation_batch_divisibility(tiny_tokenizer_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        Settings(
            training_stages=[_stage()], tokenizer_path=str(tiny_tokenizer_path), micro_batch_size=3, world_batch_size=8
        )


def test_validation_runs_for_yaml_configs_too(tmp_path: Path, tiny_tokenizer_path: Path) -> None:
    """`parse_settings` goes through `Settings.__post_init__`, so a bad YAML value is rejected the same way."""
    yaml = tmp_path / "bad.yaml"
    yaml.write_text(TINY_YAML.read_text().replace("micro_batch_size: 2", "micro_batch_size: 3"))
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        parse_settings(["--config", str(yaml), "--tokenizer_path", str(tiny_tokenizer_path)])


def test_validation_missing_tokenizer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer_path"):
        Settings(training_stages=[_stage()], tokenizer_path=str(tmp_path / "missing"))


@pytest.mark.parametrize("micro,world,expected", [(2, 4, 2), (1, 4, 4), (4, 4, 1), (2, 64, 32)])
def test_gradient_accumulation_steps(tiny_tokenizer_path: Path, micro: int, world: int, expected: int) -> None:
    cfg = Settings(
        training_stages=[_stage()],
        tokenizer_path=str(tiny_tokenizer_path),
        micro_batch_size=micro,
        world_batch_size=world,
    )
    assert cfg.gradient_accumulation_steps == expected


def test_stage_manager_stages_conversion(tiny_tokenizer_path: Path) -> None:
    entry = DataEntry(prefix="a", data_dir="d", weight=0.3, data_signature={"keys": ["text"]})
    stage = StageConfig(name="x", tokens=2048, base_lr=2e-3, train_data=[entry], val_data=[], transition_pct=0.1)
    cfg = Settings(training_stages=[stage], tokenizer_path=str(tiny_tokenizer_path))
    stages = cfg.stage_manager_stages()
    assert len(stages) == 1 and isinstance(stages[0], TrainingStage)
    ts = stages[0]
    assert (ts.name, ts.tokens, ts.base_lr, ts.transition_pct) == ("x", 2048, 2e-3, 0.1)
    assert ts.train_data == [{"prefix": "a", "data_dir": "d", "weight": 0.3, "data_signature": {"keys": ["text"]}}]
    assert ts.val_data == []


def test_tiny_yaml_stage_steps_match_the_documented_curriculum(tiny_tokenizer_path: Path) -> None:
    cfg = parse_settings(["--config", str(TINY_YAML), "--tokenizer_path", str(tiny_tokenizer_path)])
    sm = StageManager(
        cfg.stage_manager_stages(),
        cfg.world_batch_size,
        cfg.block_size,
        warmup_steps=cfg.warmup_steps,
        cooldown_steps=cfg.cooldown_steps,
        micro_batch_size=cfg.micro_batch_size,
    )
    assert [b.end_step for b in sm.boundaries] == [8, 16, 20]
