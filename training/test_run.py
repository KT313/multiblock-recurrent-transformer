# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `training.run`: the setup helpers (fast) and end-to-end runs of `train()` on the tiny 3-stage config with
synthetic data (marked slow): checkpoints, schedule, evaluation, export, determinism, resume, the stop request and
the golden 20-step run. One optimizer step is tested in `test_step.py`, evaluation in `test_evaluation.py`, the
CLI in `test_train.py`.
"""

import json
import logging
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from model import RecurrentConfig, RecurrentGPT
from training.backend.base import plain_model
from training.backend.single_device import SingleDeviceBackend
from training.checkpoint import checkpoint_dir, find_latest_checkpoint
from training.data.collate import IGNORE_INDEX
from training.data.dataset_resolver import ResolvedDataset, resolve_dataset
from training.data.loader import TRAIN_LOADER_BATCH_ROWS
from training.testing.golden import (
    GOLDEN_RUN_PATH,
    PADDED_ROWS,
    TINY_DATASET_YAML,
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    golden_run_metrics,
    single_thread_deterministic,
    write_tiny_yaml,
)
from training import logger as logger_module
from training import run as run_module
from training.logger import TrainingReport
from training.run import (
    RunState,
    build_run_model,
    build_run_optimizer,
    build_stage_manager,
    check_sequence_lengths,
    create_backend,
    prepare_run_directory,
    record_run_config,
    restore_checkpoint_if_resuming,
    run_directory_of,
    stop_requested,
    train,
)
from data_preparation.lib.build.lock import TRAIN_LOCK_NAME, RunLocked, run_lock
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.step import TrainingProgress
from evaluation.prompts import DEFAULT_PROMPTS
from training.ui.common import TRAIN_LOG_NAME, TRAIN_REPORT_NAME

History = dict[int, dict[str, float]]


def _run(
    yaml_path: Path, backend: SingleDeviceBackend | None = None, should_stop: Callable[[], bool] | None = None
) -> TrainingReport:
    """
    Run training on the yaml (the backend of the settings unless one is given) and return its report.
    """

    return train(parse_settings(["--config", str(yaml_path)]), backend=backend, should_stop=should_stop, keep_history=True)


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
    """
    `settings.backend` at `settings.precision`; the device is the backend's default (`cuda:0` or the CPU).
    """

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
    """
    `build_stage_manager` is the seven-argument constructor call: budgets of the resolved stages, batch and
    sequence length, world size, warmup / cooldown and the micro-batch divisibility check from the settings.
    """

    sm = build_stage_manager(tiny_settings, tiny_resolved, world_size=1)
    assert isinstance(sm, StageManager)
    assert sm.stages is tiny_resolved.stages
    assert (sm.world_batch_size, sm.training_max_sequence_length, sm.world_size) == (tiny_settings.world_batch_size, tiny_settings.training_max_sequence_length, 1)
    assert (sm.warmup_steps, sm.cooldown_steps) == (tiny_settings.warmup_steps, tiny_settings.cooldown_steps)
    assert sm.total_steps == 20  # tiny: (8192 + 8192 + 4096) // (4 * 256)
    assert build_stage_manager(tiny_settings, tiny_resolved, world_size=2).total_steps == 20  # 2 packed micro-batches, one each
    with pytest.raises(ValueError, match=r"micro_batches_per_step \(2\) must be a multiple of the number of devices \(3\)"):
        build_stage_manager(tiny_settings, tiny_resolved, world_size=3)
    with pytest.raises(ValueError, match="divisible by world_size"):
        build_stage_manager(replace(tiny_settings, **PADDED_ROWS), tiny_resolved, world_size=3)


def test_prepare_run_directory_creates_dirs_and_record_run_config_writes_the_record(tiny_settings: Settings) -> None:
    run_directory = prepare_run_directory(tiny_settings)
    assert run_directory == Path(tiny_settings.out_dir) / "tiny" == run_directory_of(tiny_settings)
    assert checkpoint_dir(run_directory).is_dir()
    assert not (run_directory / "run_config.json").exists(), "written only for a FRESH run, by record_run_config"
    record_run_config(tiny_settings, run_directory)
    assert json.loads((run_directory / "run_config.json").read_text()) == json.loads(json.dumps(asdict(tiny_settings)))
    prepare_run_directory(tiny_settings)  # idempotent (a resumed run reuses the directory)


def test_train_refuses_a_run_directory_another_run_holds(
    tiny_settings: Settings, cpu_backend: SingleDeviceBackend
) -> None:
    """
    `train()` takes the `out_dir` lock right after creating the run directory and holds it for the whole run: a
    second run pointed at the same `out_dir` fails before it resolves the dataset, instead of sharing checkpoints,
    `train.log` and `run_config.json` with the first one.
    """

    with run_lock(Path(tiny_settings.out_dir) / TRAIN_LOCK_NAME, "training"), pytest.raises(RunLocked, match="one is already running"):
        train(tiny_settings, backend=cpu_backend)
    assert list(checkpoint_dir(run_directory_of(tiny_settings)).glob("*.pth")) == [], "nothing ran"


def test_check_sequence_lengths_nest(tiny_settings: Settings, tiny_resolved: ResolvedDataset, caplog: pytest.LogCaptureFixture) -> None:
    """
    training_max_sequence_length <= model_max_sequence_length and <= dataset_max_sequence_length (the two bounds
    are independent); a longer run is refused with the three numbers and their files. A run cut at another length
    than the dataset config's training_target_sequence_length is a warning, not an error.
    """

    model_config = RecurrentConfig.from_yaml(tiny_settings.model_architecture_config)
    with caplog.at_level(logging.WARNING, logger="data_preparation"):
        check_sequence_lengths(tiny_settings, tiny_resolved.config, model_config)  # tiny: all 256
        assert "differs from training_target_sequence_length" not in caplog.text
        tiny_settings.training_max_sequence_length = 128
        check_sequence_lengths(tiny_settings, tiny_resolved.config, model_config)  # training shorter than the data: fine
        assert "training_max_sequence_length 128 differs from training_target_sequence_length 256 of config/datasets/tiny.yaml" in caplog.text
    tiny_settings.training_max_sequence_length = 256
    smaller = RecurrentConfig.from_yaml(tiny_settings.model_architecture_config, model_max_sequence_length=128)
    with pytest.raises(ValueError) as excinfo:
        check_sequence_lengths(tiny_settings, tiny_resolved.config, smaller)
    assert str(excinfo.value) == (
        "training_max_sequence_length 256 (the run config) must be at most model_max_sequence_length 128 "
        "(config/model_architecture/tiny.yaml, with model_overwrite applied) and dataset_max_sequence_length 256 "
        "(config/datasets/tiny.yaml)"
    )
    larger = RecurrentConfig.from_yaml(tiny_settings.model_architecture_config, model_max_sequence_length=1024)
    check_sequence_lengths(tiny_settings, tiny_resolved.config, larger)  # the model may cover more than the rows hold
    tiny_settings.training_max_sequence_length = 512
    with pytest.raises(ValueError, match="training_max_sequence_length 512 .* dataset_max_sequence_length 256"):
        check_sequence_lengths(tiny_settings, tiny_resolved.config, model_config)


def test_build_run_model_on_tiny(tiny_settings: Settings, tiny_resolved: ResolvedDataset, cpu_backend: SingleDeviceBackend) -> None:
    """
    The architecture yaml with `model_overwrite` applied, `ignore_index` / gradient checkpointing from the
    settings, `model_config.json` next to the checkpoints, the model on the backend's device.
    """

    tiny_settings.model_overwrite = {"n_embd": 32}
    run_directory = prepare_run_directory(tiny_settings)
    model = build_run_model(tiny_settings, tiny_resolved, cpu_backend, run_directory)
    assert isinstance(model, RecurrentGPT)
    assert model.config.n_embd == 32 and model.config.model_max_sequence_length == 256
    assert model.ignore_index == IGNORE_INDEX
    assert model.gradient_checkpointing is tiny_settings.gradient_checkpointing
    assert all(p.device == cpu_backend.device for p in model.parameters())
    written = json.loads((run_directory / "model_config.json").read_text())
    assert written == model.config.to_dict() and written["n_embd"] == 32
    tiny_settings.model_overwrite = {"model_max_sequence_length": 128}
    with pytest.raises(ValueError, match="training_max_sequence_length 256 .* must be at most model_max_sequence_length 128 "):
        build_run_model(tiny_settings, tiny_resolved, cpu_backend, run_directory)


def test_build_run_model_is_seeded_by_the_global_rng(tiny_settings: Settings, tiny_resolved: ResolvedDataset, cpu_backend: SingleDeviceBackend) -> None:
    """
    The parameter init consumes the global torch RNG (why `build_run_model` runs after the loaders): the same seed
    gives the same weights, and the init advances the RNG.
    """

    run_directory = prepare_run_directory(tiny_settings)
    torch.manual_seed(3)
    first = build_run_model(tiny_settings, tiny_resolved, cpu_backend, run_directory)
    after_first = torch.get_rng_state()
    torch.manual_seed(3)
    second = build_run_model(tiny_settings, tiny_resolved, cpu_backend, run_directory)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert torch.equal(after_first, torch.get_rng_state())
    torch.manual_seed(3)
    assert not torch.equal(after_first, torch.get_rng_state())


def test_build_run_optimizer_groups(tiny_settings: Settings, tiny_model: RecurrentGPT, cpu_backend: SingleDeviceBackend) -> None:
    """
    Three parameter groups (matrices, embeddings, norms + biases); the third has no weight
    decay under `no_weight_decay_for_bias_and_norm_params`; the constructor LR is `optim_config.lr`.
    """

    optimizer = build_run_optimizer(tiny_settings, tiny_model, cpu_backend)
    assert isinstance(optimizer, torch.optim.AdamW)  # tiny.yaml
    assert len(optimizer.param_groups) == 3
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.1, 0.1, 0.0]
    assert all(float(g["lr"]) == tiny_settings.optim_config.lr for g in optimizer.param_groups)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(tiny_model.parameters()))


def test_restore_checkpoint_if_resuming_starts_fresh_without_a_checkpoint(
    tiny_settings: Settings, tiny_resolved: ResolvedDataset, tiny_model: RecurrentGPT, cpu_backend: SingleDeviceBackend
) -> None:
    """
    `resume: false`, and `resume: true` with no checkpoint of the run in the directory: no resume point, the
    progress stays at step 0.
    """

    run_directory = prepare_run_directory(tiny_settings)
    optimizer = build_run_optimizer(tiny_settings, tiny_model, cpu_backend)
    stage_manager = build_stage_manager(tiny_settings, tiny_resolved, cpu_backend.world_size)
    for resume in (False, True):
        tiny_settings.resume = resume
        state = RunState(tiny_settings, run_directory, cpu_backend, tiny_model, optimizer, tiny_resolved, stage_manager, TrainingProgress())
        assert restore_checkpoint_if_resuming(state) is None
        assert state.progress == TrainingProgress(step=0, resume_step=-1)


def test_training_longer_than_the_dataset_rows_is_refused(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", training_max_sequence_length=512)
    with pytest.raises(ValueError, match=r"training_max_sequence_length \(512\) exceeds dataset_max_sequence_length \(256\) of config/datasets/tiny.yaml"):
        _run(yaml_path, cpu_backend)  # refused where the dataset config is loaded, before any data is touched


def test_a_model_shorter_than_the_training_length_is_refused(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", model_overwrite={"model_max_sequence_length": 128})
    with pytest.raises(ValueError, match="training_max_sequence_length 256 .* must be at most model_max_sequence_length 128 "):
        _run(yaml_path, cpu_backend)


def test_non_finite_loss_terminates(
    tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch, cpu_backend: SingleDeviceBackend
) -> None:
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", precision="32")
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
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", precision="32")
    monkeypatch.setattr(SingleDeviceBackend, "clip_grad_norm", lambda self, model, max_norm: torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="Gradient norm is non-finite at step 0"):
        _run(yaml_path, cpu_backend)


# --------------------------------------------------------------------------------------------------------------
# end-to-end runs (on the backend of the settings: the GPU when there is one, bf16 autocast, the only cover of
# that path; the golden run and the stop tests inject the fp32 CPU backend)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_dir: Path) -> dict[str, Any]:
    """
    One uninterrupted tiny run shared by the assertions below (module-scoped: a few seconds on CPU).
    """

    tmp = tmp_path_factory.mktemp("full_run")
    out_dir = tmp / "out"
    yaml_path = write_tiny_yaml(tmp, tiny_dataset_dir, out_dir, export_to_hf=True)
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
    settings = parse_settings(["--config", str(yaml_path)])
    resolved = resolve_dataset(settings)
    return {
        "out_dir": out_dir,
        "run_dir": run_directory_of(settings),
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
    names = sorted(p.name for p in checkpoint_dir(full_run["run_dir"]).glob("*.pth"))
    assert names == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert (full_run["run_dir"] / "run_config.json").exists()
    assert (full_run["run_dir"] / "model_config.json").exists()
    for step, m in history.items():
        assert m["step"] == step and m["total_tokens"] == step * 4 * 256
        assert torch.isfinite(torch.tensor(m["loss"])) and m["grad_norm"] >= 0
    assert full_run["optimizer_steps"] == 19  # the very first update (step 0) is skipped
    # the stage-end checkpoints carry the step they were written at, the stage the run enters next and the dataset
    # identity: the config hash and the validation split (the run validates on the held-out first 5 % of each source)
    validation_rows: dict[str, int] = full_run["validation_rows"]
    assert set(validation_rows) == {"synthetic_pretrain", "synthetic_instruct"} and min(validation_rows.values()) >= 1
    for name, step, stage in (("step-00000006-tiny-stage-0_end.pth", 6, 1), ("step-00000014-tiny-stage-1_end.pth", 14, 2)):
        extra = torch.load(checkpoint_dir(full_run["run_dir"]) / name, map_location="cpu", weights_only=False)
        assert (extra["step"], extra["stage"]) == (step, stage)
        assert extra["settings"]["run_name"] == "tiny" and set(extra["rng"]) >= {"python", "torch"}
        assert extra["model_config"]["model_max_sequence_length"] == 256 and extra["model_config"]["mean_recurrence"] == [2, 2]
        assert extra["dataset_config_hash"] == full_run["dataset_hash"]
        assert extra["validation_rows"] == validation_rows


@pytest.mark.slow
def test_training_report_of_a_full_run(full_run: dict[str, Any]) -> None:
    """
    Every field of the report of a fresh, uninterrupted, exporting run, and its summary.
    """

    report: TrainingReport = full_run["report"]
    history: History = full_run["history"]
    assert report.run_directory == full_run["run_dir"]
    assert (report.steps_this_process, report.completed_steps, report.resumed_from, report.stopped) == (20, 20, None, False)
    assert report.setup_seconds == 0.0  # no `started_at` given
    assert report.train_seconds > 0.0
    assert report.last_loss == history[20]["loss"]
    assert report.last_validation["val_loss"] == history[20]["val_loss"] and report.last_validation["val_time"] >= 0.0
    per_source = {key for key in report.last_validation if key.startswith("val_loss/")}
    assert per_source == {"val_loss/finetune-synthetic_instruct"}  # the finetune stage validates on one source
    written = json.loads((full_run["run_dir"] / TRAIN_REPORT_NAME).read_text())
    assert written["completed_steps"] == 20 and written["last_validation"] == report.last_validation
    assert "history" not in written and written["checkpoints_written"][-1].endswith("step-00000020-tiny.pth")
    assert [p.name for p in report.checkpoints_written] == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert report.export_dir == full_run["run_dir"] / "hf_export"
    summary = report.summary()
    assert summary.startswith(f"Training run in {full_run['run_dir']}: 20 optimizer steps completed (final step 20, fresh start)")
    assert "3 checkpoints written, last:" in summary and "HuggingFace export:" in summary
    assert "stopped on request" not in summary


@pytest.mark.slow
def test_train_log_of_a_full_run(full_run: dict[str, Any]) -> None:
    """
    Under pytest stdout is not a TTY, so `RunLogger` opened the console fallback of the dashboard: the run left
    the run directory's `train.log` with the header lines, one line per optimizer step (`log_step_interval: 1`), the
    validation lines, the events (checkpoints, transitions, export) and the final line.
    """

    log_text = (full_run["run_dir"] / TRAIN_LOG_NAME).read_text()
    assert "Total training steps: 20 (2 micro-batches each)" in log_text
    assert "event: no checkpoint found, starting from scratch" in log_text
    assert "step 1/20 | stage 0 pretrain_a | " in log_text and "step 20/20 | stage 2 finetune | " in log_text
    assert "step 7/20 | stage 0 pretrain_a | transition 50% | " in log_text
    assert "event: starting transition 0 -> 1 (pretrain_a -> pretrain_b), LR 3.00e-04 -> 1.00e-04" in log_text
    assert "event: transition complete, now in stage 1 (pretrain_b)" in log_text
    assert "step 8: validation val_loss " in log_text and "val_loss_1 " in log_text
    for name in ("step-00000006-tiny-stage-0_end.pth", "step-00000014-tiny-stage-1_end.pth", "step-00000020-tiny.pth"):
        assert f"event: saved checkpoint {checkpoint_dir(full_run['run_dir']) / name}" in log_text
    assert f"event: exported HuggingFace model to {full_run['run_dir'] / 'hf_export'}" in log_text
    assert "Training finished after 20 steps" in log_text


def test_train_and_logger_never_print() -> None:
    """
    Only the CLI (`train.py`) prints; `run.py` and `logger.py` log and drive the dashboard.
    """

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
    # `stage/current_stage` is the stage containing the step, inside its transition too (steps 6-7 -> stage 0,
    # 14-15 -> stage 1); the metrics at `done` describe step `done - 1`
    assert [history[d]["stage/current_stage"] for d in (1, 6, 7, 8, 9, 14, 15, 17)] == [0, 0, 0, 0, 1, 1, 1, 2]
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
    """
    Data ids are plain SOURCE names (the run-wide readers): all pretrain until the transition into finetune, all
    instruct after it, a per-sample mix inside the window.
    """

    history: History = full_run["history"]
    assert history[3]["data_composition/synthetic_pretrain"] == pytest.approx(1.0)
    assert history[18]["data_composition/synthetic_instruct"] == pytest.approx(1.0)
    for done in range(15, 17):  # inside the 1 -> 2 transition both sources may appear, weights sum to 1
        total = sum(v for k, v in history[done].items() if k.startswith("data_composition/"))
        assert total == pytest.approx(1.0)


@pytest.mark.slow
def test_export_to_hf_produces_loadable_folder(full_run: dict[str, Any]) -> None:
    export_dir = full_run["run_dir"] / "hf_export"
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
        checkpoint_dir(full_run["run_dir"]) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False
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
    """
    A resume that silently mixes two configurations is a chimera: numerics-relevant settings are compared
    against the checkpoint; `allow_settings_change: true` overrides.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").unlink()  # leave steps to run after the resume
    changed = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False, grad_clip=0.5)
    with pytest.raises(ValueError, match=r"resuming with changed \['grad_clip'\]"):
        _run(changed)
    allowed = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False, grad_clip=0.5, allow_settings_change=True
    )
    assert _run(allowed).completed_steps == 20


def test_resume_keeps_the_original_run_config_json(full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    `run_config.json` is the historical record of what the run was started with; a resume must not overwrite it.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").unlink()
    original = json.loads((run_dir / "run_config.json").read_text())
    assert original["log_step_interval"] != 4
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False, log_step_interval=4)
    _run(yaml_path)  # log_step_interval is not numerics-relevant: the resume runs
    assert json.loads((run_dir / "run_config.json").read_text()) == original



@pytest.mark.slow
def test_resume_picks_latest_checkpoint_and_restores_the_schedule(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    `resume: true` continues from the newest checkpoint of the run (here the stage-1_end one at step 14).

    Losses are not compared here: `full_run` uses the settings' backend (bf16, a GPU when there is one);
    `test_resume_reproduces_the_uninterrupted_run` asserts the bit-exactness on the CPU.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").unlink()
    latest = find_latest_checkpoint(run_dir, "tiny")
    assert latest is not None and latest.name == "step-00000014-tiny-stage-1_end.pth"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False)
    report = _run(yaml_path)
    history = report.history

    assert sorted(history) == list(range(15, 21))  # steps 14..19 ran, nothing before
    assert (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").exists()
    assert report.resumed_from == latest  # the report names the checkpoint the run continued from
    assert (report.steps_this_process, report.completed_steps, report.stopped) == (6, 20, False)
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
    final = torch.load(checkpoint_dir(run_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the resumed run kept the split


def _no_transition_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: Any) -> Path:
    """
    tiny.yaml without transitions; fp32 because bf16 autocast is very slow on the CPU and precision is
    irrelevant for the bit-exactness claim.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_yaml = tmp_path / "tiny_dataset.yaml"
    dataset_yaml.write_text(TINY_DATASET_YAML.read_text().replace("transition_pct: 0.25", "transition_pct: 0.0"))
    return write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision="32", dataset_config=str(dataset_yaml), **overrides
    )


@pytest.mark.slow
def test_stage_boundary_resume_continues_schedule_and_stream(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    Resuming from the stage-0_end checkpoint continues the run: the same remaining steps, the exact LR schedule,
    the evaluation cadence, and the data stream picks up where the checkpoint stood. The resumed run ends with
    exactly the uninterrupted run's per-source consumed-row counters, having repeated no row.

    Losses are not compared here (settings' backend, possibly a GPU); `test_resume_reproduces_the_uninterrupted_run`
    does that on the CPU.
    """

    full_dir = tmp_path / "full" / "out"
    history_full = _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir)).history
    names = sorted(p.name for p in checkpoint_dir(full_dir / "tiny").glob("*.pth"))
    assert names == ["step-00000008-tiny-stage-0_end.pth", "step-00000016-tiny-stage-1_end.pth", "step-00000020-tiny.pth"]

    resumed_dir = tmp_path / "resumed" / "out"
    yaml_path = _no_transition_yaml(
        tmp_path / "resumed",
        tiny_dataset_dir,
        resumed_dir,
        resume=True,
        resume_checkpoint_path=str(checkpoint_dir(full_dir / "tiny") / "step-00000008-tiny-stage-0_end.pth"),
    )
    history = _run(yaml_path).history
    assert sorted(history) == list(range(9, 21))
    for done in range(9, 21):
        assert history[done]["lr"] == history_full[done]["lr"]
        assert ("val_loss" in history[done]) == ("val_loss" in history_full[done])
        assert torch.isfinite(torch.tensor(history[done]["loss"]))
    assert "val_loss" in history[16] and "val_loss" in history[20]

    final_full = torch.load(checkpoint_dir(full_dir / "tiny") / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    final_res = torch.load(checkpoint_dir(resumed_dir / "tiny") / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final_res["step"] == 20
    # the data stream continued: rows consumed per source add up to the uninterrupted run's counters exactly
    # (no-transition config, worker batches of micro_batch_size 2 divide each step's draws: no buffered leftovers)
    assert final_res["data_stream"]["consumed_rows"] == final_full["data_stream"]["consumed_rows"]
    assert final_res["model"].keys() == final_full["model"].keys()


@pytest.mark.slow
def test_mid_stage_resume_continues_the_data_stream(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    A resume in the middle of a stage picks the data stream up where the checkpoint left it: the per-source row
    counters continue instead of restarting at row 0, so the resumed run trains on rows the interrupted run had not
    reached (`BatchStream.load_state_dict` says exactly what that does and does not promise).
    """

    def data_stream(directory: Path, name: str) -> dict[str, Any]:
        state = torch.load(checkpoint_dir(directory / "tiny") / name, map_location="cpu", weights_only=False)
        return cast(dict[str, Any], state["data_stream"])

    def consumed(directory: Path, name: str) -> dict[str, int]:
        return dict(data_stream(directory, name)["consumed_rows"])

    full_dir = tmp_path / "full" / "out"
    options: dict[str, Any] = {"save_step_interval": 4, "export_to_hf": False}
    _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir, **options))
    # 12 steps of 1024 packed tokens, all from the ONE run-wide synthetic_pretrain reader (stages 0 and 1 share the
    # source and only change its weight, so the counter keeps counting across the stage boundary at step 8). The
    # counter is rows READ: the one worker batch of `TRAIN_LOADER_BATCH_ROWS` rows that covers the documents of the
    # 12 steps, the rest of it sitting in the checkpoint's packing pool and buffers
    mid_state = data_stream(full_dir, "step-00000012-tiny.pth")
    assert mid_state["consumed_rows"] == {"synthetic_pretrain": TRAIN_LOADER_BATCH_ROWS}
    unconsumed = len(mid_state["buffers"]["synthetic_pretrain"]) + len(mid_state["pool"])
    assert 0 < unconsumed < TRAIN_LOADER_BATCH_ROWS

    resumed_dir = tmp_path / "resumed" / "out"
    mid = checkpoint_dir(full_dir / "tiny") / "step-00000012-tiny.pth"
    _run(
        _no_transition_yaml(
            tmp_path / "resumed", tiny_dataset_dir, resumed_dir, resume=True, resume_checkpoint_path=str(mid), **options
        )
    )
    # the resumed run added its 8 steps on top of the stored counters instead of counting from zero
    assert consumed(resumed_dir, "step-00000020-tiny.pth") == consumed(full_dir, "step-00000020-tiny.pth")


@pytest.mark.slow
def test_resume_from_explicit_checkpoint_path(full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    `resume_checkpoint_path` resumes from that file into another run directory, on the plain schedule.
    """

    ckpt = checkpoint_dir(full_run["run_dir"]) / "step-00000006-tiny-stage-0_end.pth"
    yaml_path = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, tmp_path / "fresh_out", resume=True, resume_checkpoint_path=str(ckpt), export_to_hf=False
    )
    report = _run(yaml_path)
    history = report.history
    assert sorted(history) == list(range(7, 21))
    assert report.resumed_from == ckpt
    full: History = full_run["history"]
    for done in range(7, 10):
        assert history[done]["lr"] == pytest.approx(full[done]["lr"])  # no ramp: the scheduled LR right away
    fresh_dir = tmp_path / "fresh_out" / "tiny"  # the new run directory holds everything a fresh one does
    assert report.run_directory == fresh_dir
    assert sorted(p.name for p in checkpoint_dir(fresh_dir).glob("*.pth")) == [
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    for name in ("run_config.json", "model_config.json", TRAIN_LOG_NAME, TRAIN_REPORT_NAME):
        assert (fresh_dir / name).is_file(), name
    assert json.loads((fresh_dir / "run_config.json").read_text())["resume_checkpoint_path"] == str(ckpt)


@pytest.mark.slow
def test_resume_with_changed_dataset_config_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    A dataset config whose hash differs from the checkpoint's (here: transition_pct changed, data unchanged).
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").unlink()
    yaml_path = _no_transition_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False)
    with pytest.raises(RuntimeError, match="dataset config hash"):
        _run(yaml_path)
    yaml_path = _no_transition_yaml(
        tmp_path / "allowed", tiny_dataset_dir, out_dir, resume=True, export_to_hf=False,
        allow_dataset_change=True, allow_settings_change=True,  # the fixture's checkpoint is bf16, this yaml fp32
    )
    history = _run(yaml_path).history
    assert sorted(history) == list(range(15, 21))


@pytest.mark.slow
def test_resume_with_changed_validation_split_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    A checkpoint whose stored validation split differs from the freshly resolved one (as if the data had grown
    since): the error names the source and both numbers; `allow_dataset_change` resumes anyway.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    (checkpoint_dir(run_dir) / "step-00000020-tiny.pth").unlink()
    latest = checkpoint_dir(run_dir) / "step-00000014-tiny-stage-1_end.pth"
    state = torch.load(latest, map_location="cpu", weights_only=False)
    k = state["validation_rows"]["synthetic_pretrain"]
    state["validation_rows"]["synthetic_pretrain"] = k + 1
    torch.save(state, latest)
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False)
    with pytest.raises(RuntimeError, match=f"validation rows per source: 'synthetic_pretrain': checkpoint {k + 1}, now {k}"):
        _run(yaml_path)
    (tmp_path / "allowed").mkdir()
    yaml_path = write_tiny_yaml(tmp_path / "allowed", tiny_dataset_dir, out_dir, resume=True, export_to_hf=False, allow_dataset_change=True)
    history = _run(yaml_path).history
    assert sorted(history) == list(range(15, 21))
    final = torch.load(checkpoint_dir(run_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final["validation_rows"] == full_run["validation_rows"]  # the new checkpoint stores the current split


# --------------------------------------------------------------------------------------------------------------
# the stop request (the CLI's Ctrl-C)


class StopAfterPolls:
    """
    A `StopCheck` that says stop from its n-th poll on; `train()` polls once per completed optimizer step, so
    `StopAfterPolls(5)` stops the run after step 5.
    """

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
    """
    A stop request after step 5 (inside stage 0): the loop saves `step-00000005-tiny.pth`, skips the export and
    returns a stopped report of 5 steps; `resume: true` then continues from that checkpoint to the end and exports.
    """

    out_dir = tmp_path / "out"
    run_dir = out_dir / "tiny"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, precision="32", export_to_hf=True)
    should_stop = StopAfterPolls(5)
    report = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, should_stop=should_stop, keep_history=True)
    assert should_stop.count == 5  # polled once per completed step, nothing before the loop
    assert report.stopped is True
    assert (report.steps_this_process, report.completed_steps, report.resumed_from) == (5, 5, None)
    assert sorted(report.history) == [1, 2, 3, 4, 5]
    assert [p.name for p in report.checkpoints_written] == ["step-00000005-tiny.pth"]
    assert sorted(p.name for p in checkpoint_dir(run_dir).glob("*.pth")) == ["step-00000005-tiny.pth"]
    assert report.export_dir is None and not (run_dir / "hf_export").exists()
    assert "stopped on request after step 5; rerun with resume: true to continue" in report.summary()
    stored = torch.load(checkpoint_dir(run_dir) / "step-00000005-tiny.pth", map_location="cpu", weights_only=False)
    assert (stored["step"], stored["stage"]) == (5, 0)

    (tmp_path / "resumed").mkdir()
    resumed_yaml = write_tiny_yaml(tmp_path / "resumed", tiny_dataset_dir, out_dir, precision="32", export_to_hf=True, resume=True)
    resumed = train(parse_settings(["--config", str(resumed_yaml)]), backend=cpu_backend, keep_history=True)
    assert resumed.resumed_from == checkpoint_dir(run_dir) / "step-00000005-tiny.pth"
    assert (resumed.steps_this_process, resumed.completed_steps, resumed.stopped) == (15, 20, False)
    assert sorted(resumed.history) == list(range(6, 21))
    assert [p.name for p in resumed.checkpoints_written] == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert resumed.export_dir == run_dir / "hf_export" and (run_dir / "hf_export" / "config.json").exists()


@pytest.mark.slow
def test_stop_request_at_a_checkpoint_step_saves_once(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    A stop request after step 6, the last plain step of stage 0: the stage-end checkpoint is the one written,
    not a second file.
    """

    out_dir = tmp_path / "out"
    run_dir = out_dir / "tiny"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, precision="32")
    report = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, should_stop=StopAfterPolls(6))
    assert report.stopped and report.completed_steps == 6
    assert [p.name for p in report.checkpoints_written] == ["step-00000006-tiny-stage-0_end.pth"]
    assert sorted(p.name for p in checkpoint_dir(run_dir).glob("*.pth")) == ["step-00000006-tiny-stage-0_end.pth"]


# --------------------------------------------------------------------------------------------------------------
# golden run: the numerics oracle of the training loop (`training/testing/golden.py`)


@pytest.mark.slow
def test_golden_tiny_run(tiny_dataset_dir: Path) -> None:
    """
    Numerics regression guard for the training loop: the 20-step tiny run reproduces `golden_tiny_run.json`.

    The golden is a refactor guard, not a promise about CPU training: it was recorded in fp32 on the CPU with one
    thread and deterministic algorithms (torch 2.13.0+cu130, see `training.testing.golden.record_golden_run`), so it catches
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


# --------------------------------------------------------------------------------------------------------------
# a resume reproduces the uninterrupted run (CPU, fp32, one thread, deterministic algorithms)


@pytest.mark.slow
@pytest.mark.parametrize("stop_after", [5, 7, 14])  # a plain step, inside the 0 -> 1 transition, the stage-1 end
def test_resume_reproduces_the_uninterrupted_run(
    stop_after: int, tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    A run stopped after `stop_after` steps and resumed from that checkpoint logs, from the first resumed step on,
    exactly the losses, gradient norms and validation losses of the uninterrupted run: the checkpoint restores the
    model, the optimizer, every RNG and the data stream including its buffered samples, and the loaders draw their
    iterator seeds from a private generator instead of the global RNG.
    """

    with single_thread_deterministic():
        full_dir = tmp_path / "full"
        full_dir.mkdir()
        yaml_path = write_tiny_yaml(full_dir, tiny_dataset_dir, full_dir / "out", precision="32", export_to_hf=False)
        full = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, keep_history=True).history
        stopped_dir = tmp_path / "stopped"
        stopped_dir.mkdir()
        yaml_path = write_tiny_yaml(stopped_dir, tiny_dataset_dir, stopped_dir / "out", precision="32", export_to_hf=False)
        settings = parse_settings(["--config", str(yaml_path)])
        stopped = train(settings, backend=cpu_backend, should_stop=StopAfterPolls(stop_after), keep_history=True)
        assert stopped.stopped and stopped.completed_steps == stop_after
        resumed_yaml = write_tiny_yaml(
            stopped_dir, tiny_dataset_dir, stopped_dir / "out", precision="32", export_to_hf=False, resume=True
        )
        resumed = train(parse_settings(["--config", str(resumed_yaml)]), backend=cpu_backend, keep_history=True)

    assert resumed.resumed_from == stopped.checkpoints_written[-1]
    assert sorted(resumed.history) == list(range(stop_after + 1, 21))
    for done in range(1, stop_after + 1):  # the stopped run itself matches the uninterrupted one
        assert stopped.history[done]["loss"] == full[done]["loss"], done
    for done in range(stop_after + 1, 21):
        assert resumed.history[done]["loss"] == full[done]["loss"], done
        assert resumed.history[done]["grad_norm"] == full[done]["grad_norm"], done
        assert resumed.history[done]["lr"] == full[done]["lr"], done
        for key in (k for k in full[done] if k.startswith("val_loss")):
            assert resumed.history[done][key] == full[done][key], (done, key)


@pytest.mark.slow
def test_packed_tiny_run_finishes_and_resumes_exactly(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    The tiny run with sequence packing (packs of 256 tokens, 1024 tokens per step: the same 20 steps): it trains,
    logs the packing efficiency, checkpoints the packing pool, and a run resumed from its step-12 checkpoint logs
    exactly the losses and gradient norms of the uninterrupted run from step 13 on. Validation stays padded.
    """

    options: dict[str, Any] = dict(
        pack_sequences=True,
        tokens_per_micro_batch=256,
        micro_batches_per_step=4,
        save_step_interval=4,
        export_to_hf=False,
        precision="32",
    )
    with single_thread_deterministic():
        full_dir = tmp_path / "full"
        full_dir.mkdir()
        full_yaml = write_tiny_yaml(full_dir, tiny_dataset_dir, full_dir / "out", **options)
        full = train(parse_settings(["--config", str(full_yaml)]), backend=cpu_backend, keep_history=True)
        assert sorted(full.history) == list(range(1, 21))
        for done, metrics in full.history.items():
            assert 0.0 <= metrics["packing/padding_fraction"] < 0.5, done
            assert math.isfinite(metrics["loss"])
        assert "val_loss" in full.history[8], "validation (padded batches) runs as before"
        mid = checkpoint_dir(full_dir / "out" / "tiny") / "step-00000012-tiny.pth"
        stream_state = torch.load(mid, map_location="cpu", weights_only=False)["data_stream"]
        assert stream_state["pool"], "the packing pool travels in the checkpoint"
        assert stream_state["consumed_rows"]["synthetic_pretrain"] > 0

        resumed_dir = tmp_path / "resumed"
        resumed_dir.mkdir()
        resumed_yaml = write_tiny_yaml(
            resumed_dir, tiny_dataset_dir, resumed_dir / "out", resume=True, resume_checkpoint_path=str(mid), **options
        )
        resumed = train(parse_settings(["--config", str(resumed_yaml)]), backend=cpu_backend, keep_history=True)

    assert sorted(resumed.history) == list(range(13, 21))
    for done in range(13, 21):
        assert resumed.history[done]["loss"] == full.history[done]["loss"], done
        assert resumed.history[done]["grad_norm"] == full.history[done]["grad_norm"], done
        assert resumed.history[done]["packing/padding_fraction"] == full.history[done]["packing/padding_fraction"], done


# --------------------------------------------------------------------------------------------------------------
# resume edge cases: an abandoned trajectory, changed optimizer hyperparameters, the final checkpoint, the GPU path


@pytest.mark.slow
def test_plain_resume_picks_the_newest_file_over_an_abandoned_higher_step(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    An explicit resume from step 6 into a directory that still holds steps 14 and 20 writes step 9 and stops; the
    next plain `resume: true` continues from 9 (the newest file), not from the abandoned step 20, and overwrites
    the old files as it passes their steps.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)  # copy2 keeps the modification times
    run_dir = out_dir / "tiny"
    step6 = checkpoint_dir(run_dir) / "step-00000006-tiny-stage-0_end.pth"
    (tmp_path / "redo").mkdir()
    redo_yaml = write_tiny_yaml(
        tmp_path / "redo", tiny_dataset_dir, out_dir, resume=True, resume_checkpoint_path=str(step6), export_to_hf=False
    )
    redo = _run(redo_yaml, should_stop=StopAfterPolls(3))
    assert redo.stopped and redo.completed_steps == 9 and redo.resumed_from == step6
    assert [p.name for p in redo.checkpoints_written] == ["step-00000009-tiny.pth"]

    (tmp_path / "again").mkdir()
    again = _run(write_tiny_yaml(tmp_path / "again", tiny_dataset_dir, out_dir, resume=True, export_to_hf=False))
    assert again.resumed_from == checkpoint_dir(run_dir) / "step-00000009-tiny.pth"
    assert sorted(again.history) == list(range(10, 21))
    assert [p.name for p in again.checkpoints_written] == ["step-00000014-tiny-stage-1_end.pth", "step-00000020-tiny.pth"]
    assert sorted(p.name for p in checkpoint_dir(run_dir).glob("*.pth")) == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000009-tiny.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]


@pytest.mark.slow
def test_resume_with_changed_optim_config_is_refused_even_when_settings_changes_are_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    A resume with another weight decay fails with the parameter-group message although `allow_settings_change` is
    set (the restored optimizer would silently keep the checkpoint's value); the checkpoint's value resumes.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir / "tiny") / "step-00000020-tiny.pth").unlink()
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=False, allow_settings_change=True)
    original = parse_settings(["--config", str(yaml_path)]).optim_config.weight_decay
    changed = parse_settings(["--config", str(yaml_path), "--optim_config.weight_decay", str(original * 2 + 0.01)])
    with pytest.raises(ValueError, match="resuming with changed optimizer hyperparameters") as excinfo:
        train(changed)
    assert "weight_decay" in str(excinfo.value) and "allow_settings_change cannot override" in str(excinfo.value)
    report = train(parse_settings(["--config", str(yaml_path)]), keep_history=True)
    assert sorted(report.history) == list(range(15, 21))


@pytest.mark.slow
def test_resume_from_the_final_checkpoint_runs_no_step_and_exports_again(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path
) -> None:
    """
    `resume: true` on a finished run restores step 20: no step runs, no checkpoint is written, the report says so,
    and the export is written again.
    """

    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    run_dir = out_dir / "tiny"
    exported_config = run_dir / "hf_export" / "config.json"
    written_before = exported_config.stat().st_mtime_ns
    report = _run(write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, resume=True, export_to_hf=True))
    assert report.resumed_from == checkpoint_dir(run_dir) / "step-00000020-tiny.pth"
    assert (report.steps_this_process, report.completed_steps, report.stopped) == (0, 20, False)
    assert report.history == {} and report.checkpoints_written == [] and report.last_loss is None
    assert report.export_dir == run_dir / "hf_export" and exported_config.stat().st_mtime_ns > written_before
    assert "0 optimizer steps completed (final step 20, resumed from" in report.summary()
    assert "no step logged" in report.summary()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_resume_with_compile_and_bf16_restores_the_state_and_finishes(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    The real run path (settings backend on CUDA, bf16 autocast, `compile_model`): a run stopped after step 5 leaves
    a checkpoint whose model and optimizer tensors are what a fresh setup restores from it, and the resumed run
    finishes with finite losses. Losses are not compared: cudnn benchmark and TF32 make the GPU path nondeterministic.
    """

    out_dir = tmp_path / "out"
    run_dir = out_dir / "tiny"
    yaml_path = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision="bf16-mixed", compile_model=True, export_to_hf=False
    )
    stopped = train(parse_settings(["--config", str(yaml_path)]), should_stop=StopAfterPolls(5), keep_history=True)
    assert stopped.stopped and stopped.completed_steps == 5
    checkpoint = checkpoint_dir(run_dir) / "step-00000005-tiny.pth"
    assert stopped.checkpoints_written == [checkpoint]

    settings = parse_settings(["--config", str(yaml_path), "--resume", "true"])
    backend = create_backend(settings)
    assert backend.device.type == "cuda"
    backend.seed_everything(settings.seed)
    dataset = resolve_dataset(settings, backend)
    stage_manager = build_stage_manager(settings, dataset, backend.world_size)
    model = build_run_model(settings, dataset, backend, run_dir)
    optimizer = build_run_optimizer(settings, model, backend)
    state = RunState(settings, run_dir, backend, model, optimizer, dataset, stage_manager, TrainingProgress())
    resume = restore_checkpoint_if_resuming(state)
    assert resume is not None and resume.checkpoint == checkpoint and state.progress.step == 5
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored_model = plain_model(model).state_dict()
    assert restored_model.keys() == stored["model"].keys()
    for name, tensor in restored_model.items():
        assert torch.equal(tensor.cpu(), stored["model"][name]), name
    restored_optimizer = optimizer.state_dict()["state"]
    assert restored_optimizer.keys() == stored["optimizer"]["state"].keys()
    for index, entry in restored_optimizer.items():
        for key, value in entry.items():
            assert torch.equal(value.cpu(), stored["optimizer"]["state"][index][key]), (index, key)
    del state, model, optimizer

    resumed = train(parse_settings(["--config", str(yaml_path), "--resume", "true"]), keep_history=True)
    assert resumed.resumed_from == checkpoint and resumed.completed_steps == 20 and not resumed.stopped
    assert sorted(resumed.history) == list(range(6, 21))
    assert all(math.isfinite(metrics["loss"]) for metrics in resumed.history.values())


# --------------------------------------------------------------------------------------------------------------
# samples and benchmarks during training (evaluation/)


@pytest.mark.slow
def test_samples_during_and_after_training_leave_the_numerics_unchanged(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    Sampling every 10 steps and at 0, 50 and 100 percent of the 20-step run writes `samples/step-00000001.jsonl`,
    `step-00000010.jsonl` (50 % and the interval coincide: once) and `step-00000020.jsonl`, and every training loss
    equals the run without sampling: the generations run RNG-isolated.
    """

    for name in ("plain", "sampling"):
        (tmp_path / name).mkdir()
    plain_yaml = write_tiny_yaml(tmp_path / "plain", tiny_dataset_dir, tmp_path / "plain" / "out", precision="32", export_to_hf=False)
    sampling_yaml = write_tiny_yaml(
        tmp_path / "sampling", tiny_dataset_dir, tmp_path / "sampling" / "out", precision="32", export_to_hf=False,
        sample_step_interval=10, sample_at_training_progress=[0, 50, 100], sample_max_new_tokens=4,
        sample_recurrences=[[1, 1], [2, 2]],
    )
    with single_thread_deterministic():
        plain = train(parse_settings(["--config", str(plain_yaml)]), backend=cpu_backend, keep_history=True)
        sampling = train(parse_settings(["--config", str(sampling_yaml)]), backend=cpu_backend, keep_history=True)
    assert sorted(plain.history) == sorted(sampling.history) == list(range(1, 21))
    for done in plain.history:
        assert sampling.history[done]["loss"] == plain.history[done]["loss"], done
    run_dir = tmp_path / "sampling" / "out" / "tiny"
    expected = ["step-00000001.jsonl", "step-00000010.jsonl", "step-00000020.jsonl"]
    assert [p.name for p in sampling.samples_written] == expected
    assert sorted(p.name for p in (run_dir / "samples").glob("*.jsonl")) == expected
    lines = [json.loads(line) for line in (run_dir / "samples" / "step-00000010.jsonl").read_text().split("\n")[:-1]]
    prompts = [prompt.text for prompt in DEFAULT_PROMPTS]  # the built-in prompts, once per recurrence setting
    assert [line["prompt"] for line in lines] == prompts + prompts
    assert [line["recurrence"] for line in lines] == len(prompts) * [[1, 1]] + len(prompts) * [[2, 2]]
    assert all(line["step"] == 10 and line["new_tokens"] <= 4 for line in lines)
    assert plain.samples_written == [] and not (tmp_path / "plain" / "out" / "tiny" / "samples").exists()
    log_text = (run_dir / TRAIN_LOG_NAME).read_text()
    assert f"event: wrote {2 * len(prompts)} samples to" in log_text and "event: sample: 'The Eiffel Tower" in log_text
    assert "3 samples files written, last:" in sampling.summary()
    assert "samples after steps: 1, 10, 20" in log_text
    written = json.loads((run_dir / TRAIN_REPORT_NAME).read_text())
    assert [Path(p).name for p in written["samples_written"]] == expected


@pytest.mark.slow
def test_benchmarks_during_training_use_the_harness_and_survive_its_failure(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    With a stubbed lm_eval: benchmarks every 10 steps and at the end write `benchmarks/step-XXXXXXXX.json`, the
    scores reach the report, the JSON report and the log; a harness that raises leaves the run finishing with a
    warning and no scores.
    """

    from evaluation.benchmarks import flatten_results
    from evaluation.test_benchmarks import RESULTS, stub_lm_eval

    calls = stub_lm_eval(monkeypatch)
    out_dir = tmp_path / "out"
    yaml_path = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision="32", export_to_hf=False, sample_at_training_progress=[],
        benchmark_step_interval=10, benchmark_at_training_progress=[100], benchmark_tasks=["arc_easy", "hellaswag"], benchmark_limit=5,
    )
    report = train(parse_settings(["--config", str(yaml_path)]), backend=cpu_backend, keep_history=True)
    run_dir = out_dir / "tiny"
    assert sorted(p.name for p in (run_dir / "benchmarks").glob("*.json")) == ["step-00000010.json", "step-00000020.json"]
    assert report.last_benchmarks == flatten_results(RESULTS, "mean") and report.completed_steps == 20
    assert calls["evaluate"]["tasks"] == ["arc_easy", "hellaswag"] and calls["evaluate"]["limit"] == 5
    assert json.loads((run_dir / "benchmarks" / "step-00000020.json").read_text())["step"] == 20
    assert json.loads((run_dir / TRAIN_REPORT_NAME).read_text())["last_benchmarks"] == report.last_benchmarks
    log_text = (run_dir / TRAIN_LOG_NAME).read_text()
    assert "event: benchmark arc_easy (recurrence mean) at step 10: acc 0.2500, acc_norm 0.3000" in log_text
    assert "benchmarks: mean/arc_easy/acc 0.2500" in report.summary()

    stub_lm_eval(monkeypatch, error=RuntimeError("no network"))
    (tmp_path / "failing").mkdir()
    failing_yaml = write_tiny_yaml(
        tmp_path / "failing", tiny_dataset_dir, tmp_path / "failing" / "out", precision="32", export_to_hf=False,
        sample_at_training_progress=[], benchmark_at_training_progress=[100], benchmark_tasks=["arc_easy"],
    )
    failing = train(parse_settings(["--config", str(failing_yaml)]), backend=cpu_backend, keep_history=True)
    assert failing.completed_steps == 20 and failing.last_benchmarks == {} and not failing.stopped
    assert not (tmp_path / "failing" / "out" / "tiny" / "benchmarks").exists()
    assert "benchmark evaluation failed: no network" in (tmp_path / "failing" / "out" / "tiny" / TRAIN_LOG_NAME).read_text()


def test_evaluation_recurrences_must_match_the_architecture_before_anything_runs(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    """
    A recurrence setting with the wrong number of entries fails `train()` before the run directory exists.
    """

    out_dir = tmp_path / "out"
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, out_dir, sample_recurrences=[[4, 4], [4, 4, 4]])
    with pytest.raises(ValueError, match=r"sample_recurrences\[1\] = \[4, 4, 4\] has 3 entries .* 2 recurrent blocks"):
        train(parse_settings(["--config", str(yaml_path)]))
    assert not out_dir.exists()
    (tmp_path / "b").mkdir()
    yaml_path = write_tiny_yaml(tmp_path / "b", tiny_dataset_dir, out_dir, benchmark_recurrences=[[1]])
    with pytest.raises(ValueError, match=r"benchmark_recurrences\[0\]"):
        train(parse_settings(["--config", str(yaml_path)]))
