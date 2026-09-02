# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `training.run`: the setup helpers (fast) and end-to-end runs of `train()` on the tiny 3-stage config with
synthetic data (marked slow) — checkpoints, schedule, evaluation, export, determinism, resume, the stop request and
the golden 20-step run. One optimizer step is tested in `test_step.py`, evaluation in `test_evaluation.py`, the
CLI in `test_train.py`."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from model import RecurrentConfig, RecurrentGPT
from training.backend.single_device import SingleDeviceBackend
from training.checkpoint import checkpoint_dir, find_latest_checkpoint
from training.data.collate import IGNORE_INDEX
from training.data.dataset_resolver import ResolvedDataset, resolve_dataset
from training.golden import (
    GOLDEN_RUN_PATH,
    TINY_DATASET_YAML,
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    golden_run_metrics,
    write_tiny_yaml,
)
from training import logger as logger_module
from training import run as run_module
from training.logger import TrainingReport
from training.run import (
    build_run_model,
    build_run_optimizer,
    build_stage_manager,
    check_block_sizes_agree,
    create_backend,
    prepare_run_directory,
    record_run_config,
    restore_checkpoint_if_resuming,
    stop_requested,
    train,
)
from training.run_lock import RunDirectoryLocked, run_directory_lock
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.step import TrainingProgress
from training.ui.common import TRAIN_LOG_NAME

History = dict[int, dict[str, float]]


def _run(yaml_path: Path, backend: SingleDeviceBackend | None = None) -> TrainingReport:
    """Run training on the yaml (the backend of the settings unless one is given) and return its report."""
    return train(parse_settings(["--config", str(yaml_path)]), backend=backend)


# --------------------------------------------------------------------------------------------------------------
# fast helper tests


@pytest.fixture
def tiny_settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])


@pytest.fixture
def tiny_resolved(tiny_settings: Settings) -> ResolvedDataset:
    return resolve_dataset(tiny_settings)


@pytest.fixture
def cpu_backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


def test_create_backend_follows_the_settings(tiny_settings: Settings) -> None:
    """`settings.backend` at `settings.precision`; the device is the backend's default (`cuda:0` or the CPU)."""
    tiny_settings.precision = "32"
    backend = create_backend(tiny_settings)
    assert isinstance(backend, SingleDeviceBackend) and backend.precision == "32"
    assert backend.device.type in ("cuda", "cpu") and backend.world_size == 1
    tiny_settings.backend = "nope"
    with pytest.raises(ValueError, match="Unknown backend 'nope'"):
        create_backend(tiny_settings)


def test_stop_requested() -> None:
    assert stop_requested(None) is False
    assert stop_requested(lambda: False) is False
    assert stop_requested(lambda: True) is True


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


def test_prepare_run_directory_creates_dirs_and_record_run_config_writes_the_record(tiny_settings: Settings) -> None:
    run_directory = prepare_run_directory(tiny_settings)
    assert run_directory == Path(tiny_settings.out_dir)
    assert checkpoint_dir(run_directory).is_dir()
    assert not (run_directory / "run_config.json").exists(), "written only for a FRESH run, by record_run_config"
    record_run_config(tiny_settings, run_directory)
    assert json.loads((run_directory / "run_config.json").read_text()) == json.loads(json.dumps(asdict(tiny_settings)))
    prepare_run_directory(tiny_settings)  # idempotent (a resumed run reuses the directory)


def test_train_refuses_a_run_directory_another_run_holds(
    tiny_settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """`train()` takes the run-directory lock right after creating the directory and holds it for the whole run: a
    second run pointed at the same `out_dir` fails before it resolves the dataset, instead of sharing checkpoints,
    `train.log` and `run_config.json` with the first one."""
    with run_directory_lock(Path(tiny_settings.out_dir)), pytest.raises(RunDirectoryLocked, match="already using"):
        train(tiny_settings, backend=cpu_backend)
    assert list(checkpoint_dir(Path(tiny_settings.out_dir)).glob("*.pth")) == [], "nothing ran"


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
    assert all(float(g["lr"]) == tiny_settings.optim_config.lr for g in optimizer.param_groups)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(tiny_model.parameters()))


def test_restore_checkpoint_if_resuming_starts_fresh_without_a_checkpoint(
    tiny_settings: Settings, tiny_resolved: ResolvedDataset, tiny_model: RecurrentGPT, cpu_backend: SingleDeviceBackend
) -> None:
    """`resume: false`, and `resume: true` with no checkpoint of the run in the directory: step 0, no path, no
    data-stream state."""
    run_directory = prepare_run_directory(tiny_settings)
    optimizer = build_run_optimizer(tiny_settings, tiny_model, cpu_backend)
    for resume in (False, True):
        tiny_settings.resume = resume
        progress, resumed_from, data_stream_state = restore_checkpoint_if_resuming(
            tiny_settings, run_directory, cpu_backend, tiny_model, optimizer, tiny_resolved
        )
        assert progress == TrainingProgress(step=0, resume_step=-1)
        assert resumed_from is None and data_stream_state is None


def test_block_size_mismatch_with_the_dataset_config_raises(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", block_size="128")
    with pytest.raises(ValueError, match="block_size 128 of the run config does not match block_size 256 of dataset config") as excinfo:
        _run(yaml_path, cpu_backend)
    assert "'config/datasets/tiny.yaml'" in str(excinfo.value)  # the dataset config as the run config names it


def test_block_size_mismatch_with_the_model_architecture_raises(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", model_overwrite="{block_size: 128}")
    with pytest.raises(ValueError, match="block_size 256 of the run config does not match block_size 128 of the model architecture"):
        _run(yaml_path, cpu_backend)


def test_non_finite_loss_terminates(
    tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", precision='"32"')
    forward = RecurrentGPT.forward

    def nan_forward(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        out = forward(self, *args, **kwargs)
        assert out["loss"] is not None
        out["loss"] = out["loss"] * torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(RecurrentGPT, "forward", nan_forward)
    with pytest.raises(RuntimeError, match="Loss is nan at step 0"):
        _run(yaml_path, cpu_backend)


def test_non_finite_grad_norm_terminates(
    tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", precision='"32"')
    monkeypatch.setattr(SingleDeviceBackend, "clip_grad_norm", lambda self, model, max_norm: torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="Gradient norm is non-finite at step 0"):
        _run(yaml_path, cpu_backend)


# --------------------------------------------------------------------------------------------------------------
# end-to-end runs (on the backend of the settings: the GPU when there is one, bf16 autocast — the only cover of
# that path; the golden run and the stop tests inject the fp32 CPU backend)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_dir: Path) -> dict[str, Any]:
    """One uninterrupted tiny run shared by the assertions below (module-scoped: a few seconds on CPU)."""
    tmp = tmp_path_factory.mktemp("full_run")
    out_dir = tmp / "out"
    yaml_path = write_tiny_yaml(tmp, tiny_dataset_dir, out_dir, export_to_hf="true")
    mp = pytest.MonkeyPatch()
    optimizer_steps: list[int] = []  # one entry per optimizer.step() call (tiny.yaml uses AdamW)
    adamw_step = torch.optim.AdamW.step

    def counting_step(self: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        optimizer_steps.append(len(optimizer_steps))
        return adamw_step(self, *args, **kwargs)

    mp.setattr(torch.optim.AdamW, "step", counting_step)
    try:
        report = _run(yaml_path)
    finally:
        mp.undo()
    resolved = resolve_dataset(parse_settings(["--config", str(yaml_path)]))
    return {
        "out_dir": out_dir,
        "yaml": yaml_path,
        "report": report,
        "history": report.history,
        "optimizer_steps": len(optimizer_steps),
        "dataset_hash": resolved.config_hash,
        "validation_rows": resolved.validation_rows,
    }


@pytest.mark.slow
def test_tiny_multistage_run_finishes_and_writes_checkpoints(full_run: dict[str, Any]) -> None:
    history: History = full_run["history"]
    assert sorted(history) == list(range(1, 21))
    names = sorted(p.name for p in checkpoint_dir(full_run["out_dir"]).glob("*.pth"))
    assert names == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert (full_run["out_dir"] / "run_config.json").exists()
    assert (full_run["out_dir"] / "model_config.json").exists()
    for step, m in history.items():
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
def test_training_report_of_a_full_run(full_run: dict[str, Any]) -> None:
    """Every field of the report of a fresh, uninterrupted, exporting run, and its summary."""
    report: TrainingReport = full_run["report"]
    history: History = full_run["history"]
    assert report.run_directory == full_run["out_dir"]
    assert (report.steps_completed, report.final_step, report.resumed_from, report.stopped) == (20, 20, None, False)
    assert report.setup_seconds == 0.0  # no `started_at` given
    assert report.train_seconds > 0.0
    assert report.last_loss == history[20]["loss"]
    assert report.last_validation["val_loss"] == history[20]["val_loss"] and report.last_validation["val_time"] >= 0.0
    assert [p.name for p in report.checkpoints_written] == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert report.export_dir == full_run["out_dir"] / "hf_export"
    summary = report.summary()
    assert summary.startswith(f"Training run in {full_run['out_dir']}: 20 optimizer steps completed (final step 20, fresh start)")
    assert "3 checkpoints written, last:" in summary and "HuggingFace export:" in summary
    assert "stopped on request" not in summary


@pytest.mark.slow
def test_train_log_of_a_full_run(full_run: dict[str, Any]) -> None:
    """Under pytest stdout is not a TTY, so `RunLogger` opened the console fallback of the dashboard: the run left
    `out_dir / train.log` with the header lines, one line per optimizer step (`log_step_interval: 1`), the
    validation lines, the events (checkpoints, transitions, export) and the final line."""
    log_text = (full_run["out_dir"] / TRAIN_LOG_NAME).read_text()
    assert "Total training steps: 20 (2 micro-batches each)" in log_text
    assert "event: no checkpoint found, starting from scratch" in log_text
    assert "step 1/20 | stage 0 pretrain_a | " in log_text and "step 20/20 | stage 2 finetune | " in log_text
    assert "step 7/20 | stage 0 pretrain_a | transition 50% | " in log_text
    assert "event: starting transition 0 -> 1 (pretrain_a -> pretrain_b), LR 3.00e-04 -> 1.00e-04" in log_text
    assert "event: transition complete, now in stage 1 (pretrain_b)" in log_text
    assert "step 8: validation val_loss " in log_text and "val_loss_1 " in log_text
    for name in ("step-00000006-tiny-stage-0_end.pth", "step-00000014-tiny-stage-1_end.pth", "step-00000020-tiny.pth"):
        assert f"event: saved checkpoint {checkpoint_dir(full_run['out_dir']) / name}" in log_text
    assert f"event: exported HuggingFace model to {full_run['out_dir'] / 'hf_export'}" in log_text
    assert "Training finished after 20 steps" in log_text


def test_train_and_logger_never_print() -> None:
    """Only the CLI (`train.py`) prints; `run.py` and `logger.py` log and drive the dashboard."""
    for module in (run_module, logger_module):
        source = Path(cast(str, module.__file__)).read_text()
        assert "print(" not in source, module.__name__


@pytest.mark.slow
def test_logged_lr_follows_the_multistage_schedule(full_run: dict[str, Any]) -> None:
    history: History = full_run["history"]
    # metrics at `done = step + 1` carry the LR used for optimizer step `step`
    expected = {1: 0.0, 2: 1.5e-4, 3: 3e-4, 7: 3e-4, 8: 2e-4, 9: 1e-4, 15: 1e-4, 16: 7.5e-5, 17: 5e-5, 20: 2.5e-5}
    for done, lr in expected.items():
        assert history[done]["lr"] == pytest.approx(lr), done
    # inside a transition the stage info already names the next stage (steps 6-7 -> stage 1, 14-15 -> stage 2)
    assert [history[d]["stage/current_stage"] for d in (1, 6, 7, 8, 9, 14, 15, 17)] == [0, 0, 1, 1, 1, 1, 2, 2]
    assert [history[d]["stage/in_transition"] for d in (6, 7, 8, 9, 15, 16, 17)] == [0, 1, 1, 0, 1, 1, 0]
    assert history[8]["stage/transition_progress"] == pytest.approx(0.5)


@pytest.mark.slow
def test_evaluates_at_every_partial_depth(full_run: dict[str, Any]) -> None:
    history: History = full_run["history"]
    eval_steps = [s for s, m in history.items() if "val_loss" in m]
    assert eval_steps == [8, 16, 20]
    for s in eval_steps:
        m = history[s]
        for depth in (1, "[2, 2]"):  # partial_depth_eval [1] plus the model's mean recurrence
            assert f"val_loss_{depth}" in m and f"val_ppl_{depth}" in m, (s, depth)
            assert torch.isfinite(torch.tensor(m[f"val_loss_{depth}"]))
        assert m["val_loss"] == pytest.approx(m["val_loss_[2, 2]"])
        assert m["val_ppl"] == pytest.approx(torch.tensor(m["val_loss"]).exp().item(), rel=1e-4)


@pytest.mark.slow
def test_data_composition_follows_the_stages(full_run: dict[str, Any]) -> None:
    """Data ids are plain SOURCE names (the run-wide readers): all pretrain until the transition into finetune, all
    instruct after it, a per-sample mix inside the window."""
    history: History = full_run["history"]
    assert history[3]["data_composition/synthetic_pretrain"] == pytest.approx(1.0)
    assert history[18]["data_composition/synthetic_instruct"] == pytest.approx(1.0)
    for done in range(15, 17):  # inside the 1 -> 2 transition both sources may appear, weights sum to 1
        total = sum(v for k, v in history[done].items() if k.startswith("data_composition/"))
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
def test_same_seed_is_deterministic(full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    history = _run(yaml_path).history
    for done in range(1, 21):
        assert history[done]["loss"] == pytest.approx(full_run["history"][done]["loss"], rel=1e-5), done


def test_resume_with_changed_numerics_settings_is_refused_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """A resume that silently mixes two configurations is a chimera: numerics-relevant settings are compared
    against the checkpoint; `allow_settings_change: true` overrides."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()  # leave steps to run after the resume
    changed = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", grad_clip="0.5")
    with pytest.raises(ValueError, match=r"resuming with changed \['grad_clip'\]"):
        _run(changed)
    allowed = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", grad_clip="0.5", allow_settings_change="true"
    )
    assert _run(allowed).final_step == 20


def test_resume_keeps_the_original_run_config_json(full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """`run_config.json` is the historical record of what the run was started with; a resume must not overwrite it."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    original = json.loads((out_dir / "run_config.json").read_text())
    assert original["log_step_interval"] != 4
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", log_step_interval="4")
    _run(yaml_path)  # log_step_interval is not numerics-relevant: the resume runs
    assert json.loads((out_dir / "run_config.json").read_text()) == original



@pytest.mark.slow
def test_resume_picks_latest_checkpoint_and_restores_the_schedule(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """`resume: true` continues from the latest checkpoint of the run (here the stage-1_end one at step 14).

    Losses cannot be compared exactly here: the resumed run rebuilds the loaders, whose fresh iterators draw base
    seeds from the global torch RNG at points the uninterrupted run does not (see
    `test_stage_boundary_resume_continues_schedule_and_stream` for what a resume does promise)."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    latest = find_latest_checkpoint(out_dir, "tiny")
    assert latest is not None and latest.name == "step-00000014-tiny-stage-1_end.pth"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    report = _run(yaml_path)
    history = report.history

    assert sorted(history) == list(range(15, 21))  # steps 14..19 ran, nothing before
    assert (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").exists()
    assert report.resumed_from == latest  # the report names the checkpoint the run continued from
    assert (report.steps_completed, report.final_step, report.stopped) == (6, 20, False)
    assert [p.name for p in report.checkpoints_written] == ["step-00000020-tiny.pth"]
    assert f"resumed from {latest}" in report.summary()
    full: History = full_run["history"]
    for done in range(15, 21):  # schedule state is restored exactly
        assert history[done]["lr"] == pytest.approx(full[done]["lr"])
        assert history[done]["stage/current_stage"] == full[done]["stage/current_stage"]
        assert history[done]["stage/in_transition"] == full[done]["stage/in_transition"]
        assert history[done]["total_tokens"] == full[done]["total_tokens"]
    assert [s for s, m in history.items() if "val_loss" in m] == [16, 20]
    assert history[16]["data_composition/synthetic_instruct"] == pytest.approx(0.5, abs=0.5)  # transition mix
    final = torch.load(checkpoint_dir(out_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the resumed run kept the split


def _no_transition_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: str) -> Path:
    """tiny.yaml without transitions; fp32 because bf16 autocast is very slow on the CPU and precision is
    irrelevant for the bit-exactness claim."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_yaml = tmp_path / "tiny_dataset.yaml"
    dataset_yaml.write_text(TINY_DATASET_YAML.read_text().replace("transition_pct: 0.25", "transition_pct: 0.0"))
    return write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision='"32"', dataset_config=str(dataset_yaml), **overrides
    )


@pytest.mark.slow
def test_stage_boundary_resume_continues_schedule_and_stream(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """Resuming from the stage-0_end checkpoint continues the run: the same remaining steps, the exact LR schedule,
    the evaluation cadence, and the data stream picks up where the checkpoint stood — the resumed run ends with
    exactly the uninterrupted run's per-source consumed-row counters, having repeated no row.

    It is deliberately NOT bit-exact any more: the run-wide readers live across stage boundaries, so a resumed
    run's freshly created loader iterators draw base seeds from the global torch RNG at points the uninterrupted
    run does not, and the losses diverge (the old per-stage loaders happened to make a stage-boundary resume
    bit-exact because the next stage's loader had not been created yet)."""
    full_dir = tmp_path / "full" / "out"
    history_full = _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir)).history
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
    history = _run(yaml_path).history
    assert sorted(history) == list(range(9, 21))
    for done in range(9, 21):
        assert history[done]["lr"] == history_full[done]["lr"]
        assert ("val_loss" in history[done]) == ("val_loss" in history_full[done])
        assert torch.isfinite(torch.tensor(history[done]["loss"]))
    assert "val_loss" in history[16] and "val_loss" in history[20]

    final_full = torch.load(checkpoint_dir(full_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    final_res = torch.load(checkpoint_dir(resumed_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final_res["step"] == 20
    # the data stream continued: rows consumed per source add up to the uninterrupted run's counters exactly
    # (no-transition config, worker batches of micro_batch_size 2 divide each step's draws: no buffered leftovers)
    assert final_res["data_stream"]["consumed_rows"] == final_full["data_stream"]["consumed_rows"]
    assert final_res["model"].keys() == final_full["model"].keys()


@pytest.mark.slow
def test_mid_stage_resume_continues_the_data_stream(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """A resume in the middle of a stage picks the data stream up where the checkpoint left it: the per-source row
    counters continue instead of restarting at row 0, so the resumed run trains on rows the interrupted run had not
    reached (`BatchStream.load_state_dict` says exactly what that does and does not promise)."""

    def consumed(directory: Path, name: str) -> dict[str, int]:
        state = torch.load(checkpoint_dir(directory) / name, map_location="cpu", weights_only=False)
        return dict(state["data_stream"]["consumed_rows"])

    full_dir = tmp_path / "full" / "out"
    options = {"save_step_interval": "4", "export_to_hf": "false"}
    _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir, **options))
    # 12 steps of 4 rows, all from the ONE run-wide synthetic_pretrain reader (stages 0 and 1 share the source and
    # only change its weight, so the counter keeps counting across the stage boundary at step 8)
    assert consumed(full_dir, "step-00000012-tiny.pth") == {"synthetic_pretrain": 48}

    resumed_dir = tmp_path / "resumed" / "out"
    mid = checkpoint_dir(full_dir) / "step-00000012-tiny.pth"
    _run(
        _no_transition_yaml(
            tmp_path / "resumed", tiny_dataset_dir, resumed_dir, resume="true", resume_checkpoint_path=str(mid), **options
        )
    )
    # the resumed run added its 8 steps on top of the stored counters instead of counting from zero
    assert consumed(resumed_dir, "step-00000020-tiny.pth") == consumed(full_dir, "step-00000020-tiny.pth")


@pytest.mark.slow
def test_resume_from_explicit_checkpoint_path_with_resume_warmup(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    ckpt = checkpoint_dir(full_run["out_dir"]) / "step-00000006-tiny-stage-0_end.pth"
    yaml_path = write_tiny_yaml(
        tmp_path,
        tiny_dataset_dir,
        tmp_path / "fresh_out",
        resume="true",
        resume_checkpoint_path=str(ckpt),
        resume_warmup_steps="2",
        export_to_hf="false",
    )
    report = _run(yaml_path)
    history = report.history
    assert sorted(history) == list(range(7, 21))
    assert report.resumed_from == ckpt
    assert history[7]["lr"] == pytest.approx(0.0)  # step 6: ramp starts at min_lr
    assert history[8]["lr"] == pytest.approx(0.5 * 2e-4)  # step 7: halfway to the schedule's 2e-4
    assert history[9]["lr"] == pytest.approx(1e-4)  # step 8: back on the schedule


@pytest.mark.slow
def test_resume_with_changed_dataset_config_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """A dataset config whose hash differs from the checkpoint's (here: transition_pct changed, data unchanged)."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    yaml_path = _no_transition_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    with pytest.raises(RuntimeError, match="dataset config hash"):
        _run(yaml_path)
    yaml_path = _no_transition_yaml(
        tmp_path / "allowed", tiny_dataset_dir, out_dir, resume="true", export_to_hf="false",
        allow_dataset_change="true", allow_settings_change="true",  # the fixture's checkpoint is bf16, this yaml fp32
    )
    history = _run(yaml_path).history
    assert sorted(history) == list(range(15, 21))


@pytest.mark.slow
def test_resume_with_changed_validation_split_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
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
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    with pytest.raises(RuntimeError, match=f"validation rows per source: 'synthetic_pretrain': checkpoint {k + 1}, now {k}"):
        _run(yaml_path)
    (tmp_path / "allowed").mkdir()
    yaml_path = write_tiny_yaml(tmp_path / "allowed", tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", allow_dataset_change="true")
    history = _run(yaml_path).history
    assert sorted(history) == list(range(15, 21))
    final = torch.load(checkpoint_dir(out_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the new checkpoint stores the current split


# --------------------------------------------------------------------------------------------------------------
# the stop request (the CLI's Ctrl-C)


class StopAfterPolls:
    """A `StopCheck` that says stop from its n-th poll on; `train()` polls once per completed optimizer step, so
    `StopAfterPolls(5)` stops the run after step 5."""

    def __init__(self, polls: int) -> None:
        self.polls = polls
        self.count = 0

    def __call__(self) -> bool:
        self.count += 1
        return self.count >= self.polls


@pytest.mark.slow
def test_stop_request_saves_a_checkpoint_and_the_run_resumes_from_it(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """A stop request after step 5 (inside stage 0): the loop saves `step-00000005-tiny.pth`, skips the export and
    returns a stopped report of 5 steps; `resume: true` then continues from that checkpoint to the end and exports."""
    out_dir = tmp_path / "out"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, precision='"32"', export_to_hf="true")
    should_stop = StopAfterPolls(5)
    report = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, should_stop=should_stop)
    assert should_stop.count == 5  # polled once per completed step, nothing before the loop
    assert report.stopped is True
    assert (report.steps_completed, report.final_step, report.resumed_from) == (5, 5, None)
    assert sorted(report.history) == [1, 2, 3, 4, 5]
    assert [p.name for p in report.checkpoints_written] == ["step-00000005-tiny.pth"]
    assert sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth")) == ["step-00000005-tiny.pth"]
    assert report.export_dir is None and not (out_dir / "hf_export").exists()
    assert "stopped on request after step 5; rerun with resume: true to continue" in report.summary()
    stored = torch.load(checkpoint_dir(out_dir) / "step-00000005-tiny.pth", map_location="cpu", weights_only=False)
    assert (stored["step"], stored["stage"]) == (5, 0)

    (tmp_path / "resumed").mkdir()
    resumed_yaml = write_tiny_yaml(tmp_path / "resumed", tiny_dataset_dir, out_dir, precision='"32"', export_to_hf="true", resume="true")
    resumed = train(parse_settings(["--config", str(resumed_yaml)]), backend=cpu_backend)
    assert resumed.resumed_from == checkpoint_dir(out_dir) / "step-00000005-tiny.pth"
    assert (resumed.steps_completed, resumed.final_step, resumed.stopped) == (15, 20, False)
    assert sorted(resumed.history) == list(range(6, 21))
    assert [p.name for p in resumed.checkpoints_written] == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert resumed.export_dir == out_dir / "hf_export" and (out_dir / "hf_export" / "config.json").exists()


@pytest.mark.slow
def test_stop_request_at_a_checkpoint_step_saves_once(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """A stop request after step 6, the last plain step of stage 0: the stage-end checkpoint is the one written,
    not a second file."""
    out_dir = tmp_path / "out"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, precision='"32"')
    report = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, should_stop=StopAfterPolls(6))
    assert report.stopped and report.final_step == 6
    assert [p.name for p in report.checkpoints_written] == ["step-00000006-tiny-stage-0_end.pth"]
    assert sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth")) == ["step-00000006-tiny-stage-0_end.pth"]


# --------------------------------------------------------------------------------------------------------------
# golden run: the numerics oracle of the training-pipeline restructure (tasks/training_pipeline_restructure.md)


@pytest.mark.slow
def test_golden_tiny_run(tiny_dataset_dir: Path) -> None:
    """Numerics regression guard for the training loop: the 20-step tiny run reproduces `golden_tiny_run.json`.

    The golden is a refactor guard, not a promise about CPU training: it was recorded in fp32 on the CPU with one
    thread and deterministic algorithms (torch 2.13.0+cu130, see `training.golden.record_golden_run`), so it catches
    a changed operation order, an extra RNG draw or a moved forward pass in the loop. It does NOT exercise the bf16
    autocast path used for real training (the bf16 "finite / same seed" tests above are the only cover there).
    Every float is compared with `rel=1e-5`; `GOLDEN_EXACT=1` compares with `==` (bit-identical on the recording
    machine); learning rates, checkpoint names and the optimizer-step count are always exact. Re-record only in a
    commit whose purpose is a numerics change or a change of the tiny dataset. If it fails on another machine for
    float-order reasons only, loosen the tolerance rather than chase it.
    """
    assert GOLDEN_RUN_PATH.exists(), "golden run missing; record it with record_golden_run() in a numerics commit"
    expected = json.loads(GOLDEN_RUN_PATH.read_text())
    actual = golden_run_metrics(tiny_dataset_dir)
    assert sorted(actual["steps"], key=int) == [str(s) for s in range(1, 21)]
    assert actual["optimizer_steps"] == 19  # the very first update (step 0) is skipped
    assert all("val_loss_1" in actual["steps"][str(s)] for s in (8, 16, 20))
    mismatches = golden_mismatches(expected, json.loads(golden_run_json(actual)), exact=golden_exact_requested())
    assert not mismatches, "golden run changed:\n" + "\n".join(mismatches)
