# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the run setup helpers (fast) and end-to-end runs of `training.train.train` on the tiny 3-stage config
with synthetic data (marked slow), including the golden 20-step run. One optimizer step is tested in `test_step.py`,
evaluation in `test_evaluation.py`."""

import json
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from data_preparation.lib.build import prepare
from data_preparation.lib.log import ProgressStreamHandler
from model import RecurrentConfig, RecurrentGPT
from training import train as train_module
from training.backend import SingleDeviceBackend
from training.checkpoint import checkpoint_dir, find_latest_checkpoint
from training.data import IGNORE_INDEX
from training.data.dataset_resolver import ResolvedDataset, resolve_dataset
from training.logger import Logger, TrainingReport
from training.optim import build_optimizer
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.train import (
    build_run_model,
    build_run_optimizer,
    build_stage_manager,
    check_block_sizes_agree,
    prepare_run_directory,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


def _write_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: str) -> Path:
    """`config/tiny.yaml` with dataset_dir/out_dir rewritten and optional `key: value` line replacements."""
    lines = []
    for line in TINY_YAML.read_text().splitlines():
        key = line.split(":")[0].strip() if ":" in line and not line.startswith(" ") else None
        if key == "out_dir":
            line = f"out_dir: {out_dir}"
        elif key == "dataset_dir":
            line = f"dataset_dir: {tiny_dataset_dir}"
        elif key in overrides:
            line = f"{key}: {overrides.pop(key)}"
        lines.append(line)
    lines += [f"{k}: {v}" for k, v in overrides.items()]
    path = tmp_path / "tiny.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


Logged = dict[int, dict[str, Any]]


def _capture_logs(monkeypatch: pytest.MonkeyPatch) -> Logged:
    logged: Logged = {}

    def capture(self: Logger, metrics: dict[str, Any], step: int) -> None:
        logged.setdefault(step, {}).update({k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()})

    monkeypatch.setattr(Logger, "log", capture)
    return logged


def _run(yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> Logged:
    """Run training on the yaml and return `{step: metrics}` as handed to `Logger.log`."""
    logged = _capture_logs(monkeypatch)
    train_module.train(parse_settings(["--config", str(yaml_path)]))
    return logged


# --------------------------------------------------------------------------------------------------------------
# fast helper tests


@pytest.fixture
def tiny_settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(_write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])


@pytest.fixture
def tiny_resolved(tiny_settings: Settings) -> ResolvedDataset:
    return resolve_dataset(tiny_settings)


@pytest.fixture
def cpu_backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


def test_build_stage_manager(tiny_settings: Settings, tiny_resolved: ResolvedDataset) -> None:
    """`build_stage_manager` is today's seven-argument constructor call: budgets of the resolved stages, batch and
    block size, world size, warmup / cooldown and the micro-batch divisibility check from the settings."""
    sm = build_stage_manager(tiny_settings, tiny_resolved, world_size=1)
    assert isinstance(sm, StageManager)
    assert sm.stages == tiny_resolved.training_stages()
    assert (sm.world_batch_size, sm.block_size, sm.world_size) == (tiny_settings.world_batch_size, tiny_settings.block_size, 1)
    assert (sm.warmup_steps, sm.cooldown_steps) == (tiny_settings.warmup_steps, tiny_settings.cooldown_steps)
    assert sm.total_steps == 20  # tiny: (8192 + 8192 + 4096) // (4 * 256)
    with pytest.raises(ValueError, match="divisible by world_size"):
        build_stage_manager(tiny_settings, tiny_resolved, world_size=3)


def test_prepare_run_directory_writes_run_config(tiny_settings: Settings) -> None:
    run_directory = prepare_run_directory(tiny_settings)
    assert run_directory == Path(tiny_settings.out_dir)
    assert checkpoint_dir(run_directory).is_dir()
    assert json.loads((run_directory / "run_config.json").read_text()) == json.loads(json.dumps(asdict(tiny_settings)))
    prepare_run_directory(tiny_settings)  # idempotent (a resumed run reuses the directory)


def test_check_block_sizes_agree_message(tiny_settings: Settings) -> None:
    model_config = RecurrentConfig.from_yaml(tiny_settings.model_architecture_config)
    check_block_sizes_agree(tiny_settings, model_config)  # tiny: both 256
    mismatched = RecurrentConfig.from_yaml(tiny_settings.model_architecture_config, block_size=128)
    with pytest.raises(ValueError) as excinfo:
        check_block_sizes_agree(tiny_settings, mismatched)
    assert str(excinfo.value) == (
        "block_size 256 of the run config does not match block_size 128 of the model architecture config "
        "config/model_architecture/tiny.yaml (with model_overwrite applied)"
    )


def test_build_run_model_on_tiny(tiny_settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """The architecture yaml with `model_overwrite` applied, `ignore_index` / gradient checkpointing from the
    settings, `model_config.json` next to the checkpoints, the model on the backend's device."""
    tiny_settings.model_overwrite = {"n_embd": 32}
    run_directory = prepare_run_directory(tiny_settings)
    model = build_run_model(tiny_settings, cpu_backend, run_directory)
    assert isinstance(model, RecurrentGPT)
    assert model.config.n_embd == 32 and model.config.block_size == 256
    assert model.ignore_index == IGNORE_INDEX
    assert model.gradient_checkpointing is tiny_settings.gradient_checkpointing
    assert all(p.device == cpu_backend.device for p in model.parameters())
    written = json.loads((run_directory / "model_config.json").read_text())
    assert written == model.config.to_dict() and written["n_embd"] == 32
    tiny_settings.model_overwrite = {"block_size": 128}
    with pytest.raises(ValueError, match="does not match block_size 128 of the model architecture"):
        build_run_model(tiny_settings, cpu_backend, run_directory)


def test_build_run_model_is_seeded_by_the_global_rng(tiny_settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """The parameter init consumes the global torch RNG (why `build_run_model` runs after the loaders): the same seed
    gives the same weights, and the init advances the RNG."""
    run_directory = prepare_run_directory(tiny_settings)
    torch.manual_seed(3)
    first = build_run_model(tiny_settings, cpu_backend, run_directory)
    after_first = torch.get_rng_state()
    torch.manual_seed(3)
    second = build_run_model(tiny_settings, cpu_backend, run_directory)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert torch.equal(after_first, torch.get_rng_state())
    torch.manual_seed(3)
    assert not torch.equal(after_first, torch.get_rng_state())


def test_build_run_optimizer_groups(tiny_settings: Settings, tiny_model: RecurrentGPT, cpu_backend: SingleDeviceBackend) -> None:
    """Three parameter groups (matrices, embeddings, norms + biases) with `base_lr` 1.0; the third has no weight
    decay under `no_weight_decay_for_bias_and_norm_params`; the constructor LR is `optim_config.lr`."""
    optimizer = build_run_optimizer(tiny_settings, tiny_model, cpu_backend)
    assert isinstance(optimizer, torch.optim.AdamW)  # tiny.yaml
    assert len(optimizer.param_groups) == 3
    assert [g["base_lr"] for g in optimizer.param_groups] == [1.0, 1.0, 1.0]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.1, 0.1, 0.0]
    assert all(float(g["lr"]) == tiny_settings.optim_config["lr"] for g in optimizer.param_groups)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(tiny_model.parameters()))


@pytest.fixture
def detached_training_handlers() -> Iterator[logging.Logger]:
    """The `training` logger without the handlers `configure_console_logging` adds (removed again afterwards, so a
    handler bound to a captured stderr never outlives its test)."""
    training_logger = logging.getLogger(train_module.TRAINING_LOGGER_NAME)
    before = list(training_logger.handlers)
    level = training_logger.level
    yield training_logger
    for handler in training_logger.handlers:
        if handler not in before:
            training_logger.removeHandler(handler)
            handler.close()
    training_logger.setLevel(level)


def test_main_parses_argv_trains_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch,
    tiny_dataset_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    detached_training_handlers: logging.Logger,
) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    seen: list[Settings] = []
    report = TrainingReport(
        run_directory=tmp_path / "out",
        steps_completed=3,
        final_step=3,
        resumed_from=None,
        setup_seconds=1.0,
        train_seconds=2.0,
        last_loss=1.5,
        last_validation={},
        checkpoints_written=[],
        export_dir=None,
    )

    def fake_train(settings: Settings) -> TrainingReport:
        seen.append(settings)
        return report

    monkeypatch.setattr(train_module, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(yaml_path), "--seed", "5"])
    train_module.main()
    assert len(seen) == 1 and seen[0].seed == 5 and seen[0].out_dir == str(tmp_path / "out")
    assert capsys.readouterr().out.strip() == report.summary()
    assert any(isinstance(h, ProgressStreamHandler) for h in detached_training_handlers.handlers)  # the CLI configured it


def test_configure_console_logging_routes_training_records_to_stderr(
    capsys: pytest.CaptureFixture[str], detached_training_handlers: logging.Logger
) -> None:
    """One stderr handler on the `training` logger (idempotent) at INFO, so `RunLogger`'s `training.logger` records
    reach the terminal in the formatted line format of the data-prep CLI."""
    training_logger = train_module.configure_console_logging()
    train_module.configure_console_logging()
    assert training_logger is detached_training_handlers and training_logger.level == logging.INFO
    handlers = [h for h in training_logger.handlers if isinstance(h, ProgressStreamHandler)]
    assert len(handlers) == 1
    logging.getLogger("training.logger").info("Total training steps: 20 (2 micro-batches each)")
    err = capsys.readouterr().err
    assert err.rstrip().endswith("INFO training.logger: Total training steps: 20 (2 micro-batches each)")


def test_block_size_mismatch_with_the_dataset_config_raises(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", block_size="128")
    with pytest.raises(ValueError, match="block_size 128 of the run config does not match block_size 256 of dataset config") as excinfo:
        train_module.train(parse_settings(["--config", str(yaml_path)]))
    assert "'config/datasets/tiny.yaml'" in str(excinfo.value)  # the dataset config as the run config names it


def test_block_size_mismatch_with_the_model_architecture_raises(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", model_overwrite="{block_size: 128}")
    with pytest.raises(ValueError, match="block_size 256 of the run config does not match block_size 128 of the model architecture"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


def test_non_finite_loss_terminates(tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    forward = RecurrentGPT.forward

    def nan_forward(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        out = forward(self, *args, **kwargs)
        assert out["loss"] is not None
        out["loss"] = out["loss"] * torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(RecurrentGPT, "forward", nan_forward)
    with pytest.raises(RuntimeError, match="Loss is nan at step 0"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


def test_non_finite_grad_norm_terminates(tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    monkeypatch.setattr(SingleDeviceBackend, "clip_grad_norm", lambda self, model, max_norm: torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="Gradient norm is non-finite at step 0"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


# --------------------------------------------------------------------------------------------------------------
# end-to-end runs


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_dir: Path) -> dict[str, Any]:
    """One uninterrupted tiny run shared by the assertions below (module-scoped: a few seconds on CPU)."""
    tmp = tmp_path_factory.mktemp("full_run")
    out_dir = tmp / "out"
    yaml_path = _write_yaml(tmp, tiny_dataset_dir, out_dir, export_to_hf="true")
    mp = pytest.MonkeyPatch()
    optimizer_steps: list[int] = []  # `model.step` at every optimizer.step() call (tiny.yaml uses AdamW)
    adamw_step = torch.optim.AdamW.step

    def counting_step(self: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        optimizer_steps.append(len(optimizer_steps))
        return adamw_step(self, *args, **kwargs)

    mp.setattr(torch.optim.AdamW, "step", counting_step)
    try:
        logged = _run(yaml_path, mp)
    finally:
        mp.undo()
    resolved = resolve_dataset(parse_settings(["--config", str(yaml_path)]))
    return {
        "out_dir": out_dir,
        "yaml": yaml_path,
        "logged": logged,
        "optimizer_steps": len(optimizer_steps),
        "dataset_hash": resolved.config_hash,
        "validation_rows": resolved.validation_rows,
    }


@pytest.mark.slow
def test_tiny_multistage_run_finishes_and_writes_checkpoints(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    assert sorted(logged) == list(range(1, 21))
    names = sorted(p.name for p in checkpoint_dir(full_run["out_dir"]).glob("*.pth"))
    assert names == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert (full_run["out_dir"] / "run_config.json").exists()
    assert (full_run["out_dir"] / "model_config.json").exists()
    for step, m in logged.items():
        assert m["step"] == step and m["total_tokens"] == step * 4 * 256
        assert torch.isfinite(torch.tensor(m["loss"])) and m["grad_norm"] >= 0
    assert full_run["optimizer_steps"] == 19  # the very first update (step 0) is skipped
    # the stage-end checkpoints carry the step they were written at, the stage the run enters next and the dataset
    # identity: the config hash and the validation split (the run validates on the held-out first 5 % of each source)
    validation_rows: dict[str, int] = full_run["validation_rows"]
    assert set(validation_rows) == {"synthetic_pretrain", "synthetic_instruct"} and min(validation_rows.values()) >= 1
    for name, step, stage in (("step-00000006-tiny-stage-0_end.pth", 6, 1), ("step-00000014-tiny-stage-1_end.pth", 14, 2)):
        extra = torch.load(checkpoint_dir(full_run["out_dir"]) / name, map_location="cpu", weights_only=False)
        assert (extra["step"], extra["stage"]) == (step, stage)
        assert extra["settings"]["run_name"] == "tiny" and set(extra["rng"]) >= {"python", "torch"}
        assert extra["model_config"]["block_size"] == 256 and extra["model_config"]["mean_recurrence"] == [2, 2]
        assert extra["dataset_config_hash"] == full_run["dataset_hash"]
        assert extra["validation_rows"] == validation_rows


@pytest.mark.slow
def test_logged_lr_follows_the_multistage_schedule(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    # metrics at `done = step + 1` carry the LR used for optimizer step `step`
    expected = {1: 0.0, 2: 1.5e-4, 3: 3e-4, 7: 3e-4, 8: 2e-4, 9: 1e-4, 15: 1e-4, 16: 7.5e-5, 17: 5e-5, 20: 2.5e-5}
    for done, lr in expected.items():
        assert logged[done]["lr"] == pytest.approx(lr), done
    # inside a transition the stage info already names the next stage (steps 6-7 -> stage 1, 14-15 -> stage 2)
    assert [logged[d]["stage/current_stage"] for d in (1, 6, 7, 8, 9, 14, 15, 17)] == [0, 0, 1, 1, 1, 1, 2, 2]
    assert [logged[d]["stage/in_transition"] for d in (6, 7, 8, 9, 15, 16, 17)] == [0, 1, 1, 0, 1, 1, 0]
    assert logged[8]["stage/transition_progress"] == pytest.approx(0.5)


@pytest.mark.slow
def test_evaluates_at_every_partial_depth(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    eval_steps = [s for s, m in logged.items() if "val_loss" in m]
    assert eval_steps == [8, 16, 20]
    for s in eval_steps:
        m = logged[s]
        for depth in (1, "[2, 2]"):  # partial_depth_eval [1] plus the model's mean recurrence
            assert f"val_loss_{depth}" in m and f"val_ppl_{depth}" in m, (s, depth)
            assert torch.isfinite(torch.tensor(m[f"val_loss_{depth}"]))
        assert m["val_loss"] == pytest.approx(m["val_loss_[2, 2]"])
        assert m["val_ppl"] == pytest.approx(torch.tensor(m["val_loss"]).exp().item(), rel=1e-4)


@pytest.mark.slow
def test_data_composition_follows_the_stages(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    assert logged[3]["data_composition/pretrain_a-synthetic_pretrain"] == pytest.approx(1.0)
    assert logged[18]["data_composition/finetune-synthetic_instruct"] == pytest.approx(1.0)
    for done in range(15, 17):  # inside the 1 -> 2 transition both stages' sources may appear, weights sum to 1
        total = sum(v for k, v in logged[done].items() if k.startswith("data_composition/"))
        assert total == pytest.approx(1.0)


@pytest.mark.slow
def test_export_to_hf_produces_loadable_folder(full_run: dict[str, Any]) -> None:
    export_dir = full_run["out_dir"] / "hf_export"
    assert (export_dir / "config.json").exists() and (export_dir / "model.safetensors").exists()
    model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    ids = tokenizer("tok_1 tok_2 tok_3", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (1, ids.shape[1], 512)
    assert torch.isfinite(out.logits).all()

    # exported weights are the final checkpoint's weights
    final = torch.load(
        checkpoint_dir(full_run["out_dir"]) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False
    )["model"]
    wte = final["transformer.wte.weight"]
    assert torch.equal(model.model.transformer.wte.weight.detach(), wte)


@pytest.mark.slow
def test_same_seed_is_deterministic(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    logged = _run(yaml_path, monkeypatch)
    for done in range(1, 21):
        assert logged[done]["loss"] == pytest.approx(full_run["logged"][done]["loss"], rel=1e-5), done


@pytest.mark.slow
def test_resume_picks_latest_checkpoint_and_restores_the_schedule(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume: true` continues from the latest checkpoint of the run (here the stage-1_end one at step 14).

    Losses cannot be compared exactly here: the resumed run re-seeds the transition sampler with
    `seed + step` and rebuilds the loaders, so the mixed batches of the 1 -> 2 transition (steps 14, 15) differ
    by design. Exact equivalence is asserted in `test_resume_is_bit_exact_without_transitions`."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    latest = find_latest_checkpoint(out_dir, "tiny")
    assert latest is not None and latest.name == "step-00000014-tiny-stage-1_end.pth"
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    logged = _run(yaml_path, monkeypatch)

    assert sorted(logged) == list(range(15, 21))  # steps 14..19 ran, nothing before
    assert (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").exists()
    full: Logged = full_run["logged"]
    for done in range(15, 21):  # schedule state is restored exactly
        assert logged[done]["lr"] == pytest.approx(full[done]["lr"])
        assert logged[done]["stage/current_stage"] == full[done]["stage/current_stage"]
        assert logged[done]["stage/in_transition"] == full[done]["stage/in_transition"]
        assert logged[done]["total_tokens"] == full[done]["total_tokens"]
    assert [s for s, m in logged.items() if "val_loss" in m] == [16, 20]
    assert logged[16]["data_composition/finetune-synthetic_instruct"] == pytest.approx(0.5, abs=0.5)  # transition mix
    final = torch.load(checkpoint_dir(out_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the resumed run kept the split


def _no_transition_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: str) -> Path:
    """tiny.yaml without transitions; fp32 because bf16 autocast is very slow on the CPU and precision is
    irrelevant for the bit-exactness claim."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_yaml = tmp_path / "tiny_dataset.yaml"
    dataset_yaml.write_text(TINY_DATASET_YAML.read_text().replace("transition_pct: 0.25", "transition_pct: 0.0"))
    return _write_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision='"32"', dataset_config=str(dataset_yaml), **overrides
    )


@pytest.mark.slow
def test_resume_is_bit_exact_without_transitions(
    tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without transitions (no rng-driven mixing) resuming from the stage-0_end checkpoint reproduces the
    uninterrupted run exactly: every logged loss, the validation losses and the final model + optimizer state."""
    full_dir = tmp_path / "full" / "out"
    logged_full = _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir), monkeypatch)
    names = sorted(p.name for p in checkpoint_dir(full_dir).glob("*.pth"))
    assert names == ["step-00000008-tiny-stage-0_end.pth", "step-00000016-tiny-stage-1_end.pth", "step-00000020-tiny.pth"]

    resumed_dir = tmp_path / "resumed" / "out"
    yaml_path = _no_transition_yaml(
        tmp_path / "resumed",
        tiny_dataset_dir,
        resumed_dir,
        resume="true",
        resume_checkpoint_path=str(checkpoint_dir(full_dir) / "step-00000008-tiny-stage-0_end.pth"),
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(9, 21))
    for done in range(9, 21):
        assert logged[done]["loss"] == logged_full[done]["loss"], done  # exact, not approx
        assert logged[done]["lr"] == logged_full[done]["lr"]
        assert logged[done].get("val_loss") == logged_full[done].get("val_loss")
    assert "val_loss" in logged[16] and "val_loss" in logged[20]

    final_full = torch.load(checkpoint_dir(full_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    final_res = torch.load(checkpoint_dir(resumed_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final_full["model"].keys() == final_res["model"].keys()
    assert all(torch.equal(final_full["model"][k], final_res["model"][k]) for k in final_full["model"])
    for sa, sb in zip(final_full["optimizer"]["state"].values(), final_res["optimizer"]["state"].values()):
        assert all(torch.equal(sa[k], sb[k]) for k in sa if torch.is_tensor(sa[k]))
    assert final_res["step"] == 20


@pytest.mark.slow
def test_resume_from_explicit_checkpoint_path_with_resume_warmup(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = checkpoint_dir(full_run["out_dir"]) / "step-00000006-tiny-stage-0_end.pth"
    yaml_path = _write_yaml(
        tmp_path,
        tiny_dataset_dir,
        tmp_path / "fresh_out",
        resume="true",
        resume_checkpoint_path=str(ckpt),
        resume_warmup_steps="2",
        export_to_hf="false",
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(7, 21))
    assert logged[7]["lr"] == pytest.approx(0.0)  # step 6: ramp starts at min_lr
    assert logged[8]["lr"] == pytest.approx(0.5 * 2e-4)  # step 7: halfway to the schedule's 2e-4
    assert logged[9]["lr"] == pytest.approx(1e-4)  # step 8: back on the schedule


@pytest.mark.slow
def test_resume_with_changed_dataset_config_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset config whose hash differs from the checkpoint's (here: transition_pct changed, data unchanged)."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    yaml_path = _no_transition_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    with pytest.raises(RuntimeError, match="dataset config hash"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))
    yaml_path = _no_transition_yaml(
        tmp_path / "allowed", tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", allow_dataset_change="true"
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(15, 21))


@pytest.mark.slow
def test_resume_with_changed_validation_split_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose stored validation split differs from the freshly resolved one (as if the data had grown
    since): the error names the source and both numbers; `allow_dataset_change` resumes anyway."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    latest = checkpoint_dir(out_dir) / "step-00000014-tiny-stage-1_end.pth"
    state = torch.load(latest, map_location="cpu", weights_only=False)
    k = state["validation_rows"]["synthetic_pretrain"]
    state["validation_rows"]["synthetic_pretrain"] = k + 1
    torch.save(state, latest)
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    with pytest.raises(RuntimeError, match=f"validation rows per source: 'synthetic_pretrain': checkpoint {k + 1}, now {k}"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))
    (tmp_path / "allowed").mkdir()
    yaml_path = _write_yaml(tmp_path / "allowed", tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", allow_dataset_change="true")
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(15, 21))
    final = torch.load(checkpoint_dir(out_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the new checkpoint stores the current split


# --------------------------------------------------------------------------------------------------------------
# golden run: the numerics oracle of the training-pipeline restructure (tasks/training_pipeline_restructure.md)

GOLDEN_RUN_PATH = Path(__file__).resolve().parent / "golden_tiny_run.json"
GOLDEN_EXACT_ENV = "GOLDEN_EXACT"  # `GOLDEN_EXACT=1`: compare every float with `==` instead of rel 1e-5
GOLDEN_PER_STEP_KEYS = ("loss", "grad_norm", "lr")
GOLDEN_ALWAYS_EXACT_KEYS = ("lr", "checkpoints", "optimizer_steps")


@contextmanager
def _single_thread_deterministic() -> Iterator[None]:
    """One intra-op thread and deterministic algorithms for the block; both restored afterwards, also on failure."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def golden_run_metrics(tiny_dataset_dir: Path) -> dict[str, Any]:
    """The 20-step tiny run in fp32 on the CPU (one thread, deterministic algorithms), reduced to its numerics.

    `config/tiny.yaml` with `precision: "32"`, `wandb_enabled: false`, `export_to_hf: false`,
    `dataloader_num_workers: 0`, `resume: false` and `out_dir` in a temporary directory, on a
    `SingleDeviceBackend(device="cpu", precision="32")` injected through `training.train.get_backend`. Returns
    `{"steps": {"<done>": {loss, grad_norm, lr[, val_loss, val_loss_<depth>...]}}, "checkpoints": [file names],
    "optimizer_steps": number of optimizer.step() calls, "parameter_norms": {name: L2 norm in the final checkpoint}}`.
    """
    with tempfile.TemporaryDirectory() as tmp, _single_thread_deterministic():
        tmp_path = Path(tmp)
        out_dir = tmp_path / "out"
        yaml_path = _write_yaml(
            tmp_path,
            tiny_dataset_dir,
            out_dir,
            precision='"32"',
            wandb_enabled="false",
            export_to_hf="false",
            dataloader_num_workers="0",
            resume="false",
        )
        settings = parse_settings(["--config", str(yaml_path)])
        mp = pytest.MonkeyPatch()
        optimizer_step_calls = 0

        def cpu_backend(name: str, precision: str) -> SingleDeviceBackend:
            assert name == settings.backend and precision == "32"
            return SingleDeviceBackend(device="cpu", precision=precision)

        def counting_build_optimizer(name: str, params: Any, **cfg: Any) -> torch.optim.Optimizer:
            optimizer = build_optimizer(name, params, **cfg)
            optimizer_class = type(optimizer)
            original_step = optimizer_class.step

            def counting_step(self: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:
                nonlocal optimizer_step_calls
                optimizer_step_calls += 1
                return original_step(self, *args, **kwargs)

            mp.setattr(optimizer_class, "step", counting_step)
            return optimizer

        mp.setattr(train_module, "get_backend", cpu_backend)
        mp.setattr(train_module, "build_optimizer", counting_build_optimizer)
        try:
            logged = _run(yaml_path, mp)
        finally:
            mp.undo()

        steps: dict[str, dict[str, float]] = {}
        for done, metrics in sorted(logged.items()):
            step_metrics = {key: float(metrics[key]) for key in GOLDEN_PER_STEP_KEYS}
            step_metrics |= {key: float(value) for key, value in metrics.items() if key.startswith("val_loss")}
            steps[str(done)] = step_metrics
        final_checkpoint = find_latest_checkpoint(out_dir, settings.run_name)
        assert final_checkpoint is not None
        final_state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)["model"]
        return {
            "steps": steps,
            "checkpoints": sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth")),
            "optimizer_steps": optimizer_step_calls,
            "parameter_norms": {
                name: float(torch.linalg.vector_norm(tensor.float())) for name, tensor in final_state.items()
            },
        }


def golden_run_json(metrics: dict[str, Any]) -> str:
    """The fixture text: sorted keys, indent 2, floats as `repr` (json's default, round-trips exactly)."""
    return json.dumps(metrics, sort_keys=True, indent=2) + "\n"


def record_golden_run() -> Path:
    """Re-record `training/golden_tiny_run.json`. ONLY do this in a commit whose purpose is a numerics change of the
    training loop, or when the tiny dataset changes (the fixture depends on `config/datasets/tiny.yaml` and the data
    pipeline: the synthetic rows, dedup, the instruct shuffle and input inversions, the 5 % validation split):

        uv run python -c "from training.test_train import record_golden_run; record_golden_run()"

    Builds the tiny dataset into a temporary directory first (as the `tiny_dataset_dir` fixture does). The committed
    fixture was recorded with torch 2.13.0+cu130 on the author's machine (CPU, fp32, one thread, deterministic
    algorithms); two consecutive recordings there are byte-identical.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset_root = Path(tmp) / "tiny_dataset"
        prepare(TINY_DATASET_YAML, dataset_root, assume_yes=False)
        metrics = golden_run_metrics(dataset_root)
    GOLDEN_RUN_PATH.write_text(golden_run_json(metrics))
    return GOLDEN_RUN_PATH


def golden_mismatches(expected: Any, actual: Any, *, exact: bool, path: str = "") -> list[str]:
    """Every difference between a recorded golden structure and a fresh one, as `path: expected != actual` lines.

    Floats are compared with `pytest.approx(rel=1e-5, abs=0)`, or with `==` when `exact`; values under a key in
    `GOLDEN_ALWAYS_EXACT_KEYS` (learning rates, checkpoint names, the optimizer-step count), ints, strings and key
    sets are always compared exactly.
    """
    key = path.rsplit("/", 1)[-1]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a mapping, got {type(actual).__name__}"]
        mismatches = [f"{path}/{k}: missing" for k in sorted(set(expected) - set(actual))]
        mismatches += [f"{path}/{k}: unexpected" for k in sorted(set(actual) - set(expected))]
        for k in sorted(set(expected) & set(actual)):
            mismatches += golden_mismatches(expected[k], actual[k], exact=exact, path=f"{path}/{k}")
        return mismatches
    if isinstance(expected, list):
        return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, float) and not exact and key not in GOLDEN_ALWAYS_EXACT_KEYS:
        return [] if actual == pytest.approx(expected, rel=1e-5, abs=0) else [f"{path}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def test_golden_mismatches_reports_every_difference() -> None:
    expected = {"steps": {"1": {"loss": 1.0, "lr": 2.0}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, json.loads(golden_run_json(expected)), exact=True) == []
    # rel 1e-5 on plain floats, but never on the learning rate, the checkpoint names or the step count
    close = {"steps": {"1": {"loss": 1.0 + 1e-7, "lr": 2.0}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, close, exact=False) == []
    assert golden_mismatches(expected, close, exact=True) == ["/steps/1/loss: 1.0 != 1.0000001"]
    lr_off = {"steps": {"1": {"loss": 1.0, "lr": 2.0 + 1e-7}}, "checkpoints": ["a.pth"], "optimizer_steps": 19}
    assert golden_mismatches(expected, lr_off, exact=False) == ["/steps/1/lr: 2.0 != 2.0000001"]
    off = {"steps": {"1": {"loss": 1.1, "grad_norm": 0.0}}, "checkpoints": ["b.pth"], "optimizer_steps": 18}
    assert golden_mismatches(expected, off, exact=False) == [
        "/checkpoints: ['a.pth'] != ['b.pth']",
        "/optimizer_steps: 19 != 18",
        "/steps/1/lr: missing",
        "/steps/1/grad_norm: unexpected",
        "/steps/1/loss: 1.0 != 1.1",
    ]


@pytest.mark.slow
def test_golden_tiny_run(tiny_dataset_dir: Path) -> None:
    """Numerics regression guard for the training loop: the 20-step tiny run reproduces `golden_tiny_run.json`.

    The golden is a refactor guard, not a promise about CPU training: it was recorded in fp32 on the CPU with one
    thread and deterministic algorithms (torch 2.13.0+cu130, see `record_golden_run`), so it catches a changed
    operation order, an extra RNG draw or a moved forward pass in the loop. It does NOT exercise the bf16 autocast
    path used for real training (the bf16 "finite / same seed" tests above are the only cover there). Every float
    is compared with `rel=1e-5`; `GOLDEN_EXACT=1` compares with `==` (bit-identical on the recording machine);
    learning rates, checkpoint names and the optimizer-step count are always exact. Re-record only in a commit whose
    purpose is a numerics change or a change of the tiny dataset. If it fails on another machine for float-order
    reasons only, loosen the tolerance rather than chase it.
    """
    assert GOLDEN_RUN_PATH.exists(), "golden run missing; record it with record_golden_run() in a numerics commit"
    expected = json.loads(GOLDEN_RUN_PATH.read_text())
    actual = golden_run_metrics(tiny_dataset_dir)
    assert sorted(actual["steps"], key=int) == [str(s) for s in range(1, 21)]
    assert actual["optimizer_steps"] == 19  # the very first update (step 0) is skipped
    assert all("val_loss_1" in actual["steps"][str(s)] for s in (8, 16, 20))
    exact = os.environ.get(GOLDEN_EXACT_ENV) == "1"
    mismatches = golden_mismatches(expected, json.loads(golden_run_json(actual)), exact=exact)
    assert not mismatches, "golden run changed:\n" + "\n".join(mismatches)
