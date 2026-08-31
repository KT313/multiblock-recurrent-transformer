# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the disabled logger (no-ops), the stubbed wandb path, the gradient/parameter metric helpers and
`RunLogger` (console lines via `caplog` on `training.logger`, wandb dict, timers on a fake clock, data composition,
history, `TrainingReport`)."""

import logging
import math
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from data_preparation.dataset_config import DatasetConfig
from model import RecurrentGPT
from training.backend import SingleDeviceBackend
from training.data.dataset_resolver import ResolvedDataset
from training.logger import (
    CONSOLE_LOGGER_NAME,
    Logger,
    RunLogger,
    TrainingReport,
    _qkv_dims,
    _reverse_engineer_adam_effective_lr,
    _to_scalar,
    describe_parameters,
    num_parameters,
    track_gradient_metrics,
)
from training.optim import ELLISAdam, get_param_groups
from training.settings import Settings
from training.stage_manager import StageManager, TrainingStage
from training.step import StepResult, TrainingProgress
from training.test_step import reference_settings, reference_stage_manager


def test_disabled_logger_is_a_no_op(tmp_path: Path) -> None:
    logger = Logger("proj", "run", tmp_path, enabled=False)
    assert logger.run is None and logger.enabled is False
    logger.log({"loss": torch.tensor(1.0)}, step=1)
    logger.log_hyperparams({"a": 1})
    logger.log_summary({"b": torch.tensor(2)})
    logger.finish()
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_enabled_logger_forwards_scalars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wandb calls are exercised against a stub run so no wandb import/network happens."""
    calls: list[tuple[Any, ...]] = []

    class _Config:
        @staticmethod
        def update(params: dict[str, Any], allow_val_change: bool) -> None:
            calls.append(("config", params, allow_val_change))

    class _Run:
        summary: dict[str, Any] = {}
        config = _Config()

        def log(self, metrics: dict[str, Any], step: int) -> None:
            calls.append(("log", metrics, step))

        def finish(self) -> None:
            calls.append(("finish",))

    class _WandbStub:
        @staticmethod
        def init(**kwargs: Any) -> _Run:
            calls.append(("init", kwargs))
            return _Run()

    monkeypatch.setitem(sys.modules, "wandb", _WandbStub())  # any object works as a module
    logger = Logger("proj", "run", tmp_path / "out", offline=True, enabled=True)
    assert calls[0][1]["mode"] == "offline" and calls[0][1]["project"] == "proj"
    assert calls[0][1]["name"] == "run" and calls[0][1]["dir"] == str(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    logger.log({"loss": torch.tensor(1.5), "n": 3}, step=4)
    assert calls.pop() == ("log", {"loss": 1.5, "n": 3}, 4)
    logger.log_hyperparams({"seed": 1})
    assert calls.pop() == ("config", {"seed": 1}, True)
    logger.log_summary({"p": torch.tensor(7)})
    assert _Run.summary == {"p": 7}
    logger.finish()
    assert calls.pop() == ("finish",) and logger.run is None
    logger.finish()  # idempotent
    assert calls == [("init", calls[0][1])]

    Logger("proj", "run", tmp_path / "online", offline=False, enabled=True)
    assert calls[-1][1]["mode"] == "online"


def test_to_scalar() -> None:
    assert _to_scalar(torch.tensor([2.5])) == 2.5
    assert _to_scalar(torch.tensor(4)) == 4
    assert _to_scalar(3) == 3
    t = torch.zeros(2)
    assert _to_scalar(t) is t


def test_describe_parameters_counts_total_recurrent_and_unrolled(tiny_model: RecurrentGPT) -> None:
    """The line names the total, the parameters of the core blocks and the unrolled count at the mean recurrence
    (tiny: mean_recurrence [2, 2], so unrolled = total + recurrent)."""
    total = num_parameters(tiny_model)
    recurrent = sum(p.numel() for block in tiny_model.transformer.core_blocks for p in block.parameters())
    assert 0 < recurrent < total
    expected = f"Model: {total:,} parameters, {recurrent:,} in recurrent blocks, unfolds to {total + recurrent:,} at mean recurrence."
    assert describe_parameters(tiny_model) == expected
    compiled = cast(torch.nn.Module, torch.compile(tiny_model))  # typed as a bare callable, is an OptimizedModule
    assert describe_parameters(compiled) == expected  # the compiled wrapper is unwrapped


def test_num_parameters_counts_tied_weights_once(tiny_model: RecurrentGPT) -> None:
    total = num_parameters(tiny_model)
    assert total == sum(p.numel() for p in tiny_model.parameters())
    with_duplicates = sum(p.numel() for _, p in tiny_model.named_parameters(remove_duplicate=False))
    assert with_duplicates == total + tiny_model.transformer.wte.weight.numel()  # lm_head is tied
    tiny_model.transformer.wte.weight.requires_grad_(False)
    assert num_parameters(tiny_model, only_trainable=True) == total - tiny_model.transformer.wte.weight.numel()


def test_reverse_engineer_adam_effective_lr() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    param.grad = torch.tensor([0.5, 0.0])  # second element: |g| <= eps branch
    state = {"exp_avg": torch.tensor([0.05, 0.02]), "exp_avg_sq": torch.tensor([0.0025, 0.0004])}
    group = {"eps": 1e-6}
    lr = _reverse_engineer_adam_effective_lr(param, state, group)
    assert lr[0].item() == pytest.approx(0.05 / (0.05 + 1e-6) / 0.5, rel=1e-5)
    assert lr[1].item() == pytest.approx(0.02 / (0.02 + 1e-6) / 1e-6, rel=1e-5)


def test_qkv_dims(tiny_model: RecurrentGPT) -> None:
    assert _qkv_dims(tiny_model) == (64, 64, 64)
    assert _qkv_dims(torch.nn.Linear(2, 2)) is None

    class Partial(torch.nn.Module):
        config = type("Cfg", (), {"n_embd": 8})()

    assert _qkv_dims(Partial()) is None


def _step_tiny(tiny_model: RecurrentGPT) -> ELLISAdam:
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    loss = tiny_model(x, labels=x, num_steps_pair=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    opt.step()
    return opt


N_ATTN_LAYERS = 2 + 2 + 1  # prelude + core blocks (1 layer each) + coda


def test_track_gradient_metrics_on_tiny_model(tiny_model: RecurrentGPT) -> None:
    opt = _step_tiny(tiny_model)
    metrics = track_gradient_metrics(tiny_model, opt)

    for i in range(N_ATTN_LAYERS):
        for key in (
            f"query_grad_{i}",
            f"ffn2_grad_{i}",
            f"q_effective_lr_{i}",
            f"k_effective_lr_{i}",
            f"v_effective_lr_{i}",
            f"ffn2_effective_lr_{i}",
        ):
            assert key in metrics, key
    assert f"ffn2_grad_{N_ATTN_LAYERS}" not in metrics and f"query_grad_{N_ATTN_LAYERS}" not in metrics
    for key in (
        "avg_RMS",
        "embed_RMS",
        "local_l1_grad_norm",
        "l2_param_norm",
        "l1_param_norm",
        "core_block_0_l2_param_norm",
        "core_block_1_l2_param_norm",
        "word_embed_l2_param_norm",
        "model_l2_param_norm",
    ):
        assert key in metrics, key
    for key, value in metrics.items():
        assert torch.is_tensor(value) and value.numel() == 1, key
        assert math.isfinite(value.item()), key

    # hand-checkable values
    expected_l2 = torch.norm(torch.stack([p.norm() for p in tiny_model.parameters()]))
    assert torch.allclose(metrics["l2_param_norm"], expected_l2)
    assert torch.allclose(metrics["word_embed_l2_param_norm"], tiny_model.transformer.wte.weight.norm())
    assert metrics["ffn2_grad_0"] > 0 and metrics["query_grad_0"] > 0
    grad = dict(tiny_model.named_parameters())["transformer.prelude.0.attn.Wqkv.weight"].grad
    assert grad is not None and torch.allclose(metrics["query_grad_0"], grad[:64].norm())
    # right after the first step exp_avg_sq == (1 - beta2) * g^2, so the per-element RMS is 1/sqrt(1 - beta2)
    # (= 10) wherever |g| > eps and 0 where the gradient vanishes: the average is bounded by 10 from above
    assert 5.0 < metrics["avg_RMS"].item() <= 1 / math.sqrt(1 - 0.99) + 1e-4


def test_track_gradient_metrics_without_gradients_or_state(tiny_model: RecurrentGPT) -> None:
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    metrics = track_gradient_metrics(tiny_model, opt)
    assert "avg_RMS" not in metrics and "query_grad_0" not in metrics and "local_l1_grad_norm" not in metrics
    assert math.isfinite(metrics["l2_param_norm"].item())


def test_track_gradient_metrics_on_plain_module() -> None:
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 4)).sum().backward()
    opt.step()
    metrics = track_gradient_metrics(model, opt)
    assert set(metrics) == {"avg_RMS", "local_l1_grad_norm", "l2_param_norm", "l1_param_norm"}


def test_non_finite_gradient_is_reported_as_nan(tiny_model: RecurrentGPT) -> None:
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(tiny_model, 4e-5), lr=1e-3)
    loss = tiny_model(x, labels=x, num_steps_pair=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    params = dict(tiny_model.named_parameters())
    proj_grad = params["transformer.prelude.0.mlp.proj.weight"].grad
    qkv_grad = params["transformer.prelude.0.attn.Wqkv.weight"].grad
    assert proj_grad is not None and qkv_grad is not None
    proj_grad[0, 0] = float("inf")
    qkv_grad[0, 0] = float("inf")
    metrics = track_gradient_metrics(tiny_model, opt)
    assert math.isnan(metrics["ffn2_grad_0"].item())
    assert math.isnan(metrics["query_grad_0"].item())
    assert math.isfinite(metrics["ffn2_grad_1"].item())
    assert "ffn2_effective_lr_0" not in metrics  # params with non-finite grads are skipped for effective LRs


# --------------------------------------------------------------------------------------------------------------
# RunLogger: a run without a dataset — in-memory settings, a stage manager, fake step results, a fake clock


class FakeClock:
    """A clock that only moves when a test says so (`clock=` of `RunLogger.open`)."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


TOKENS_PER_STEP = 4 * 256  # reference_settings: world_batch_size 4, block_size 256
STEP_KEYS = {
    "loss", "ppl", "lr", "grad_norm", "step", "seconds/step", "tokens/second", "total_tokens", "total_time",
    "remaining_time", "stage/current_stage", "stage/base_lr", "stage/in_transition", "stage/transition_progress",
    "stage/stage_progress",
}  # fmt: skip


def two_stage_manager(settings: Settings) -> StageManager:
    """Two stages with a transition between them (stage a: 8 steps, the last 25 % transitioning; stage b: 4 steps)."""
    stages = [
        TrainingStage("a", tokens=8 * TOKENS_PER_STEP, base_lr=3e-4, transition_pct=0.25),
        TrainingStage("b", tokens=4 * TOKENS_PER_STEP, base_lr=1e-4, transition_pct=0.0),
    ]
    return StageManager(stages, settings.world_batch_size, settings.block_size, warmup_steps=2, cooldown_steps=2)


def fake_result(
    stage_manager: StageManager,
    step: int,
    *,
    loss: float = 2.0,
    data_ids: list[str] | None = None,
    metrics: dict[str, torch.Tensor] | None = None,
    validation: dict[str, torch.Tensor] | None = None,
) -> StepResult:
    """A `StepResult` as `run_one_optimizer_step` returns it, with tensors where the step has tensors."""
    return StepResult(
        step=step,
        learning_rate=1e-4 * step,
        loss=torch.tensor(loss),
        log_ppl=torch.tensor(loss),
        grad_norm=torch.tensor(0.5),
        stage=stage_manager.get_stage_info(step),
        next_stage=stage_manager.get_stage_info(step + 1),
        data_ids=data_ids if data_ids is not None else ["source_a"] * 4,
        metrics=metrics or {},
        validation=validation,
    )


@pytest.fixture
def resolved(tiny_dataset_config: DatasetConfig) -> ResolvedDataset:
    """Only `config_hash` is read by `RunLogger.open` (the wandb hyperparameters)."""
    return ResolvedDataset(config=tiny_dataset_config, config_hash="hash-1", tokenizer_dir="unused", stages=[], validation_rows={})


@pytest.fixture
def console_records(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger=CONSOLE_LOGGER_NAME)
    return caplog


def open_run_logger(
    settings: Settings,
    stage_manager: StageManager,
    model: RecurrentGPT,
    resolved: ResolvedDataset,
    run_directory: Path,
    clock: FakeClock,
    *,
    start_step: int = 0,
    setup_started: float | None = None,
) -> RunLogger:
    progress = TrainingProgress(step=start_step, resume_step=start_step if start_step else -1)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    return RunLogger.open(
        settings, run_directory, resolved, model, stage_manager, progress, backend, clock=clock, setup_started=setup_started
    )


def run_fake_steps(
    run_logger: RunLogger, stage_manager: StageManager, progress: TrainingProgress, clock: FakeClock, steps: int, seconds_per_step: float
) -> None:
    """`steps` fake optimizer steps, each taking `seconds_per_step`, logged as `train()` logs them."""
    for _ in range(steps):
        result = fake_result(stage_manager, progress.step)
        progress.advance()
        clock.advance(seconds_per_step)
        run_logger.log_step(result, progress)


def _record_wandb_logs(monkeypatch: pytest.MonkeyPatch) -> dict[int, dict[str, Any]]:
    recorded: dict[int, dict[str, Any]] = {}

    def capture(self: Logger, metrics: dict[str, Any], step: int) -> None:
        recorded[step] = dict(metrics)

    monkeypatch.setattr(Logger, "log", capture)
    return recorded


def test_open_logs_the_run_header_and_ends_the_setup_timer(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """The stage summary, the total-steps line, the parameter line and the setup line — all INFO records on
    `training.logger` marked `keep` — and the setup timer measured from `setup_started` to `open`."""
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock(1000.0)
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock, setup_started=987.5)
    assert run_logger.wandb.enabled is False and run_logger.setup_seconds == 12.5
    assert [record.getMessage() for record in console_records.records] == [
        stage_manager.get_stage_summary(),
        "Total training steps: 10 (2 micro-batches each)",
        describe_parameters(tiny_model),
        "Setup took 12.5s, starting training at step 0 on cpu (32).",
    ]
    assert all(getattr(record, "keep", False) is True for record in console_records.records)
    assert all(record.name == CONSOLE_LOGGER_NAME for record in console_records.records)


def test_log_step_history_wandb_dict_and_throughput(
    tiny_model: RecurrentGPT,
    resolved: ResolvedDataset,
    tmp_path: Path,
    console_records: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every log step: the same metric dict (already scalars) to wandb and, as floats, to `history[done]`; the
    throughput arithmetic on a fake clock ticking 2 s per step; the one-line console summary."""
    recorded = _record_wandb_logs(monkeypatch)
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    gradient_metrics = {"l2_param_norm": torch.tensor(7.0)}
    for _ in range(3):
        result = fake_result(stage_manager, progress.step, loss=2.0, metrics=gradient_metrics)
        progress.advance()
        clock.advance(2.0)
        run_logger.log_step(result, progress)

    assert sorted(run_logger.history) == [1, 2, 3]
    for done, metrics in run_logger.history.items():
        assert set(metrics) == STEP_KEYS | {"l2_param_norm", "data_composition/source_a"}
        assert all(isinstance(value, float) for value in metrics.values())
        assert metrics["step"] == done and metrics["loss"] == 2.0 and metrics["ppl"] == pytest.approx(math.exp(2.0))
        assert metrics["lr"] == 1e-4 * (done - 1) and metrics["grad_norm"] == 0.5 and metrics["l2_param_norm"] == 7.0
        assert metrics["seconds/step"] == 2.0 and metrics["tokens/second"] == TOKENS_PER_STEP / 2.0
        assert metrics["total_tokens"] == done * TOKENS_PER_STEP and metrics["total_time"] == 2.0 * done
        assert metrics["remaining_time"] == 2.0 * (10 - done)
        assert (metrics["stage/current_stage"], metrics["stage/in_transition"], metrics["stage/base_lr"]) == (0, 0, 3e-4)
        assert metrics["data_composition/source_a"] == 1.0
        assert recorded[done] == metrics and not any(torch.is_tensor(v) for v in recorded[done].values())
    summaries = [r.getMessage() for r in console_records.records if r.getMessage().startswith("step ")]
    assert summaries == [
        "step 1/10 | loss 2.0000 | lr 0.00e+00 | grad_norm 0.500 | 2.00s/step",
        "step 2/10 | loss 2.0000 | lr 1.00e-04 | grad_norm 0.500 | 2.00s/step",
        "step 3/10 | loss 2.0000 | lr 2.00e-04 | grad_norm 0.500 | 2.00s/step",
    ]
    assert not any(getattr(r, "keep", False) for r in console_records.records if r.getMessage().startswith("step "))


def test_log_interval_composition_fractions_sum_to_one_and_reset(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`log_step_interval: 2`: only even steps are logged (no `.item()` in between), `seconds/step` is the interval
    time per step, the composition counts every world batch since the last log step and starts over afterwards."""
    recorded = _record_wandb_logs(monkeypatch)
    settings = reference_settings(log_step_interval=2)
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    batches = [["a", "a", "b", "b"], ["b", "b", "b", "b"], ["a"] * 4, ["a"] * 4]
    for data_ids in batches:
        result = fake_result(stage_manager, progress.step, data_ids=data_ids)
        progress.advance()
        clock.advance(1.0)
        run_logger.log_step(result, progress)
    assert sorted(run_logger.history) == [2, 4] and sorted(recorded) == [2, 4]
    second, fourth = run_logger.history[2], run_logger.history[4]
    assert second["seconds/step"] == 1.0 and second["tokens/second"] == TOKENS_PER_STEP
    assert second["data_composition/a"] == 0.25 and second["data_composition/b"] == 0.75
    assert fourth["data_composition/a"] == 1.0 and "data_composition/b" not in fourth
    for metrics in (second, fourth):
        assert sum(v for k, v in metrics.items() if k.startswith("data_composition/")) == pytest.approx(1.0)


def test_log_step_logs_the_transition_lines(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """One "starting transition" line after the last plain step of stage a and one "transition complete" line after
    the last transition step, worded as the thesis loop printed them; nothing at any other step."""
    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    run_fake_steps(run_logger, stage_manager, progress, clock, stage_manager.total_steps, 1.0)

    expected = []
    for step in range(stage_manager.total_steps):
        before, after = stage_manager.get_stage_info(step), stage_manager.get_stage_info(step + 1)
        if after.in_transition and not before.in_transition:
            expected.append(
                f"Step {step + 1}: starting transition {after.prev_stage_idx} -> {after.stage_idx} ({after.stage_name}), "
                f"LR {cast(float, after.prev_base_lr):.2e} -> {after.base_lr:.2e}"
            )
        elif before.in_transition and not after.in_transition:
            expected.append(f"Step {step + 1}: transition complete, now in stage {after.stage_idx} ({after.stage_name})")
    assert len(expected) == 2 and "starting transition 0 -> 1 (b), LR 3.00e-04 -> 1.00e-04" in expected[0]
    assert expected[1].endswith("transition complete, now in stage 1 (b)")
    transition_lines = [r.getMessage() for r in console_records.records if "transition" in r.getMessage()]
    assert transition_lines == expected
    assert [run_logger.history[d]["stage/in_transition"] for d in range(1, 13)] == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]


def test_evaluating_times_the_validation_and_log_step_reports_it(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """The `evaluating()` block's duration becomes `val_time` of that step's validation metrics (floats in the
    metric dict, `history` and the report), with the `Step N: val loss ...` line; a validation without a timed block
    reports 0 s (the timer is consumed, never stale)."""
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    validation = {"val_loss": torch.tensor(2.5), "val_ppl": torch.tensor(2.5).exp(), "val_loss_1": torch.tensor(2.6)}

    result = fake_result(stage_manager, 0, validation=validation)
    progress.advance()
    with run_logger.evaluating():
        clock.advance(3.0)
    run_logger.log_step(result, progress)
    assert run_logger.history[1]["val_time"] == 3.0 and run_logger.history[1]["val_loss"] == 2.5
    assert run_logger.history[1]["val_loss_1"] == pytest.approx(2.6)

    result = fake_result(stage_manager, 1, validation=validation)
    progress.advance()
    clock.advance(1.0)
    run_logger.log_step(result, progress)
    assert run_logger.history[2]["val_time"] == 0.0
    validation_lines = [r.getMessage() for r in console_records.records if "val loss" in r.getMessage()]
    assert validation_lines == ["Step 1: val loss 2.5000 (stage 0, 3.0s)", "Step 2: val loss 2.5000 (stage 0, 0.0s)"]
    report = run_logger.close(progress, None)
    assert report.last_validation == {"val_loss": 2.5, "val_ppl": pytest.approx(math.exp(2.5)), "val_loss_1": pytest.approx(2.6), "val_time": 0.0}
    assert all(isinstance(v, float) for v in report.last_validation.values())


def test_close_returns_the_report_of_a_resumed_run_and_is_idempotent(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """A run resumed at step 4 that ran 3 steps, wrote 2 checkpoints and exported: every report field, the summary
    text, the `keep` lines of resume / checkpoint / export / finish, and `close()` + `__exit__` releasing the
    resources exactly once."""
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock(500.0)
    released: list[str] = []
    resume_path = tmp_path / "checkpoints" / "step-00000004-steps.pth"
    with open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock, start_step=4, setup_started=490.0) as run_logger:
        run_logger.resources.callback(released.append, "released")
        run_logger.log_resume(resume_path, 4)
        progress = TrainingProgress(step=4, resume_step=4)
        run_fake_steps(run_logger, stage_manager, progress, clock, 3, 2.0)
        first, second = tmp_path / "checkpoints" / "a.pth", tmp_path / "checkpoints" / "b.pth"
        run_logger.log_checkpoint(first)
        run_logger.log_checkpoint(second)
        export_dir = tmp_path / "hf_export"
        run_logger.log_export(export_dir)
        report = run_logger.close(progress, export_dir)
        assert released == ["released"]
        run_logger.close(progress, export_dir)  # a second close changes nothing
    assert released == ["released"]  # `__exit__` after `close()` is a no-op
    with run_logger:
        pass
    assert released == ["released"]

    assert isinstance(report, TrainingReport)
    assert (report.run_directory, report.steps_completed, report.final_step) == (tmp_path, 3, 7)
    assert report.resumed_from == resume_path
    assert report.setup_seconds == 10.0 and report.train_seconds == 6.0
    assert report.last_loss == 2.0 and report.last_validation == {}
    assert report.checkpoints_written == [first, second] and report.export_dir == export_dir
    assert sorted(report.history) == [5, 6, 7] and report.history is run_logger.history
    assert report.summary() == "\n".join(
        [
            f"Training run in {tmp_path}: 3 optimizer steps completed (final step 7, resumed from {resume_path})",
            "  setup 10.0s, training 6.0s",
            "  last loss 2.0000 | no validation",
            f"  2 checkpoints written, last: {second}",
            f"  HuggingFace export: {export_dir}",
        ]
    )
    kept = {r.getMessage() for r in console_records.records if getattr(r, "keep", False)}
    assert {
        f"Resumed from {resume_path} at step 4",
        f"Saved checkpoint {first}",
        f"Saved checkpoint {second}",
        f"Exported HuggingFace model to {export_dir}",
        "Training finished after 7 steps in 6.0s.",
    } <= kept


def test_fresh_start_report_summary_without_steps(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, FakeClock())
    run_logger.log_fresh_start()
    report = run_logger.close(TrainingProgress(), None)
    fresh = [r for r in console_records.records if r.getMessage() == "No checkpoint loaded, starting from scratch."]
    assert len(fresh) == 1 and getattr(fresh[0], "keep", False) is True
    assert (report.steps_completed, report.final_step, report.resumed_from, report.last_loss) == (0, 0, None, None)
    assert report.summary() == "\n".join(
        [
            f"Training run in {tmp_path}: 0 optimizer steps completed (final step 0, fresh start)",
            "  setup 0.0s, training 0.0s",
            "  no step logged | no validation",
            "  no checkpoint written",
            "  no HuggingFace export",
        ]
    )


def test_run_logger_never_prints(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every console line is a logging record; nothing reaches stdout / stderr without a handler on `training`."""
    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    clock = FakeClock()
    with open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock) as run_logger:
        run_logger.log_fresh_start()
        progress = TrainingProgress()
        run_fake_steps(run_logger, stage_manager, progress, clock, stage_manager.total_steps, 1.0)
        run_logger.log_checkpoint(tmp_path / "c.pth")
        run_logger.close(progress, None)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
