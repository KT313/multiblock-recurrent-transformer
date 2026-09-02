# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the disabled logger (no-ops), the stubbed wandb path (quiet settings included), the gradient/parameter
metric helpers and `RunLogger` (the header records via `caplog` on `training.logger`, the dashboard calls on a
recording fake and on a real `TrainingDashboard` over a StringIO console, the fallback picked under pytest and its
`train.log`, wandb dict, timers on a fake clock, data composition, history, `TrainingReport`)."""

import io
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from rich.console import Console

from data_preparation.dataset_config import DatasetConfig
from model import RecurrentGPT
from training.backend.single_device import SingleDeviceBackend
from training.data.dataset_resolver import ResolvedDataset
from training.logger import (
    CONSOLE_LOGGER_NAME,
    Dashboard,
    Logger,
    RunLogger,
    TrainingReport,
    _qkv_dims,
    _reverse_engineer_adam_effective_lr,
    _to_scalar,
    describe_parameters,
    num_parameters,
    open_dashboard,
    track_gradient_metrics,
)
from training.optim import ELLISAdam, get_param_groups
from training.settings import Settings
from training.stage_manager import StageManager
from training.testing.stages import resolved_stage
from training.step import StepResult, TrainingProgress
from training.test_step import reference_settings, reference_stage_manager
from training.ui.board import TrainingDashboard
from training.ui.capture import WANDB_QUIET_SETTINGS
from training.ui.common import TRAIN_LOG_NAME
from training.ui.fallback import ConsoleFallbackDashboard


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

    class _Settings:
        """Stands in for `wandb.Settings`: records what `Logger` asks for."""

        def __init__(self, **values: Any) -> None:
            self.values = values

    class _WandbStub:
        Settings = _Settings

        @staticmethod
        def init(**kwargs: Any) -> _Run:
            calls.append(("init", kwargs))
            return _Run()

    monkeypatch.setitem(sys.modules, "wandb", _WandbStub())  # any object works as a module
    logger = Logger("proj", "run", tmp_path / "out", offline=True, enabled=True)
    assert calls[0][1]["mode"] == "offline" and calls[0][1]["project"] == "proj"
    assert calls[0][1]["name"] == "run" and calls[0][1]["dir"] == str(tmp_path / "out")
    quiet = calls[0][1]["settings"]  # no stdout / stderr wrapping and no banner under the dashboard
    assert isinstance(quiet, _Settings) and quiet.values == WANDB_QUIET_SETTINGS == {"console": "off", "silent": True}
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
        resolved_stage("a", tokens=8 * TOKENS_PER_STEP, base_lr=3e-4, transition_pct=0.25),
        resolved_stage("b", tokens=4 * TOKENS_PER_STEP, base_lr=1e-4, transition_pct=0.0),
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
        data_ids=data_ids if data_ids is not None else ["source_a"] * 4,
        metrics=metrics or {},
        validation=validation,
    )


@pytest.fixture
def resolved(tiny_dataset_config: DatasetConfig) -> ResolvedDataset:
    """Only `config_hash` is read by `RunLogger.open` (the wandb hyperparameters)."""
    return ResolvedDataset(
        config=tiny_dataset_config, config_hash="hash-1", tokenizer_dir="unused", stages=[], train_sources=[], validation_rows={}, rows_on_disk={}
    )


@pytest.fixture
def console_records(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger=CONSOLE_LOGGER_NAME)
    return caplog


class RecordingDashboard:
    """A `Dashboard` that records every call (`RunLogger`'s side of the dashboard API, without a display)."""

    def __init__(self) -> None:
        # (step, stage index, transition progress or None, the step dict as passed)
        self.steps: list[tuple[int, int, float | None, dict[str, object]]] = []
        self.validations: list[tuple[int, dict[str, object]]] = []
        self.events: list[str] = []
        self.statuses: list[str] = []

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None:
        self.steps.append((step, stage_index, transition, dict(metrics)))

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None:
        self.validations.append((step, dict(losses)))

    def note_event(self, text: str) -> None:
        self.events.append(text)

    def set_status(self, text: str) -> None:
        self.statuses.append(text)


def string_console_dashboard(stage_manager: StageManager, log_step_interval: int = 1) -> TrainingDashboard:
    """A real `TrainingDashboard` rendering into a StringIO (never entered: no live display, no terminal capture)."""
    return TrainingDashboard(
        "steps",
        [s.name for s in stage_manager.stages],
        [b.end_step - b.start_step for b in stage_manager.boundaries],
        stage_manager.total_steps,
        details={"model": "tiny", "dataset": "tiny", "device": "cpu", "precision": "32"},
        log_step_interval=log_step_interval,
        console=Console(file=io.StringIO(), force_terminal=True, width=120),
    )


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
    dashboard: Dashboard | None = None,
) -> RunLogger:
    """`RunLogger.open` on the CPU backend; `dashboard` defaults to a fresh `RecordingDashboard` (available as
    `run_logger.dashboard`), so no test opens the real factory unless it asks for it (`dashboard=None` explicitly is
    not possible here — call `RunLogger.open` directly for that)."""
    progress = TrainingProgress(step=start_step, resume_step=start_step if start_step else -1)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    return RunLogger.open(
        settings,
        run_directory,
        resolved,
        model,
        stage_manager,
        progress,
        backend,
        dashboard=dashboard if dashboard is not None else RecordingDashboard(),
        clock=clock,
        setup_started=setup_started,
        keep_history=True,
    )


def recording(run_logger: RunLogger) -> RecordingDashboard:
    """The `RecordingDashboard` behind a logger opened by `open_run_logger` without a dashboard of its own."""
    assert isinstance(run_logger.dashboard, RecordingDashboard)
    return run_logger.dashboard


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
    """Every log step: the same metric dict (already scalars) to wandb, as floats to `history[done]` and to the
    dashboard's `update_step` (no tensor in it); the throughput arithmetic on a fake clock ticking 2 s per step; no
    per-step console record (the fallback dashboard's line is the one console line of a step)."""
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
    shown = recording(run_logger).steps
    assert [(step, stage, transition) for step, stage, transition, _ in shown] == [(1, 0, None), (2, 0, None), (3, 0, None)]
    for (_, _, _, step_dict), metrics in zip(shown, run_logger.history.values()):
        assert step_dict == metrics  # exactly the wandb dict
        assert not any(torch.is_tensor(value) for value in step_dict.values())
    assert not any(r.getMessage().startswith("step ") for r in console_records.records)


def test_history_is_kept_only_on_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without `keep_history` (the CLI's default) `history` stays empty while wandb and the dashboard still get every
    log step's metric dict: a long run does not hold its metrics in memory."""
    recorded = _record_wandb_logs(monkeypatch)
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    run_logger = _logger_with(Logger("p", "r", tmp_path, enabled=False), stage_manager, settings, tmp_path)
    assert run_logger.keep_history is False
    progress = TrainingProgress()
    for _ in range(2):
        result = fake_result(stage_manager, progress.step, loss=2.0)
        progress.advance()
        run_logger.log_step(result, progress)
    assert run_logger.history == {} and sorted(recorded) == [1, 2]
    assert [step for step, _, _, _ in recording(run_logger).steps] == [1, 2]
    assert run_logger.close(progress, None).history == {}


def test_log_interval_composition_fractions_sum_to_one_and_reset(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`log_step_interval: 2`: only even steps are logged (no `.item()` in between — the dashboard gets an empty step
    dict at the odd steps, which only moves its bars), `seconds/step` is the interval time per step, the composition
    counts every world batch since the last log step and starts over afterwards."""
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
    shown = recording(run_logger).steps
    assert [(step, stage) for step, stage, _, _ in shown] == [(1, 0), (2, 0), (3, 0), (4, 0)], "the bars move every step"
    assert shown[0][3] == {} and shown[2][3] == {}, "nothing is read from the step's tensors at a non-log step"
    assert shown[1][3]["loss"] == 2.0 and shown[3][3]["step"] == 4
    second, fourth = run_logger.history[2], run_logger.history[4]
    assert second["seconds/step"] == 1.0 and second["tokens/second"] == TOKENS_PER_STEP
    assert second["data_composition/a"] == 0.25 and second["data_composition/b"] == 0.75
    assert fourth["data_composition/a"] == 1.0 and "data_composition/b" not in fourth
    for metrics in (second, fourth):
        assert sum(v for k, v in metrics.items() if k.startswith("data_composition/")) == pytest.approx(1.0)


def test_log_step_notes_the_transition_events_and_moves_the_bars_with_the_stage_at_done(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """Stage a: 8 steps, the last two (6, 7) transitioning to b. One "starting transition" event after step 6 is
    done and one "transition complete" event after step 8 is done (worded as the thesis loop printed them, with the
    stage names); no console record for them. The bars get the stage containing `done` (a until 8 steps are done, b
    from then on) and the transition keys of `done` — while `history` keeps the `stage/*` metrics of the step trained
    on, one step behind: `stage/current_stage` is the stage containing that step, also inside its transition."""
    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    run_fake_steps(run_logger, stage_manager, progress, clock, stage_manager.total_steps, 1.0)

    assert recording(run_logger).events == [
        "starting transition 0 -> 1 (a -> b), LR 3.00e-04 -> 1.00e-04",
        "transition complete, now in stage 1 (b)",
    ]
    assert not any("transition" in r.getMessage() for r in console_records.records)
    shown = recording(run_logger).steps
    assert [stage for _, stage, _, _ in shown] == [0] * 7 + [1] * 5  # done 6, 7: a's transition steps count for a
    assert [transition for _, _, transition, _ in shown] == [None] * 5 + [0.0, 0.5] + [None] * 5
    assert [run_logger.history[d]["stage/in_transition"] for d in range(1, 13)] == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert [run_logger.history[d]["stage/transition_progress"] for d in (7, 8)] == [0.0, 0.5]
    assert [run_logger.history[d]["stage/current_stage"] for d in range(1, 13)] == [0] * 8 + [1] * 4


def test_evaluating_times_the_validation_and_log_step_reports_it(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """The `evaluating()` block shows `evaluating` as the status (the previous status afterwards) and its duration
    becomes `val_time` of that step's validation metrics (floats in the metric dict, `history` and the report); the
    dashboard gets the `val_loss*` entries as floats; a validation without a timed block reports 0 s (the timer is
    consumed, never stale)."""
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    validation = {"val_loss": torch.tensor(2.5), "val_ppl": torch.tensor(2.5).exp(), "val_loss_1": torch.tensor(2.6)}

    result = fake_result(stage_manager, 0, validation=validation)
    progress.advance()
    run_logger.status("training")
    with run_logger.evaluating():
        assert recording(run_logger).statuses == ["training", "evaluating"]
        clock.advance(3.0)
    assert recording(run_logger).statuses == ["training", "evaluating", "training"]
    run_logger.log_step(result, progress)
    assert run_logger.history[1]["val_time"] == 3.0 and run_logger.history[1]["val_loss"] == 2.5
    assert run_logger.history[1]["val_loss_1"] == pytest.approx(2.6)

    result = fake_result(stage_manager, 1, validation=validation)
    progress.advance()
    clock.advance(1.0)
    run_logger.log_step(result, progress)
    assert run_logger.history[2]["val_time"] == 0.0
    shown = recording(run_logger).validations
    assert [step for step, _ in shown] == [1, 2] and list(shown[0][1]) == ["val_loss", "val_loss_1"]
    assert shown[0][1]["val_loss"] == 2.5 and shown[0][1]["val_loss_1"] == pytest.approx(2.6)
    assert all(isinstance(value, float) for _, losses in shown for value in losses.values())
    assert not any("val loss" in r.getMessage() for r in console_records.records)
    report = run_logger.close(progress, None)
    assert report.last_validation == {"val_loss": 2.5, "val_ppl": pytest.approx(math.exp(2.5)), "val_loss_1": pytest.approx(2.6), "val_time": 0.0}
    assert all(isinstance(v, float) for v in report.last_validation.values())


def test_close_returns_the_report_of_a_resumed_run_and_is_idempotent(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """A run resumed at step 4 that ran 3 steps, wrote 2 checkpoints and exported: every report field, the summary
    text, the events of resume / checkpoint / export, the `keep` line and the status of the finish, and `close()` +
    `__exit__` releasing the resources exactly once."""
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
    assert recording(run_logger).events == [
        f"resumed from {resume_path} at step 4",
        f"saved checkpoint {first}",
        f"saved checkpoint {second}",
        f"exported HuggingFace model to {export_dir}",
    ]
    assert recording(run_logger).statuses == ["finished", "finished"]  # once per `close()`
    kept = [r.getMessage() for r in console_records.records if getattr(r, "keep", False)]
    assert kept.count("Training finished after 7 steps in 6.0s.") == 2 and not any("checkpoint" in k for k in kept)


def test_fresh_start_report_summary_without_steps(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, FakeClock())
    run_logger.log_fresh_start()
    report = run_logger.close(TrainingProgress(), None)
    assert recording(run_logger).events == ["no checkpoint found, starting from scratch"]
    assert not any("scratch" in r.getMessage() for r in console_records.records)
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
    """Every console line is a logging record or a dashboard call; with a recording dashboard nothing reaches stdout /
    stderr (there is no handler on `training`), and the module has no `print` at all."""
    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    clock = FakeClock()
    with open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock) as run_logger:
        run_logger.log_fresh_start()
        run_logger.status("training")
        progress = TrainingProgress()
        run_fake_steps(run_logger, stage_manager, progress, clock, stage_manager.total_steps, 1.0)
        with run_logger.saving_checkpoint():
            run_logger.log_checkpoint(tmp_path / "c.pth")
        run_logger.close(progress, None)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert "print(" not in Path(sys.modules[RunLogger.__module__].__file__ or "").read_text()


def test_status_and_saving_checkpoint_forward_to_the_dashboard(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path
) -> None:
    """`status(text)` sets the header status; `saving_checkpoint()` shows `saving checkpoint` for the block and puts
    the status from before it back (also after an exception in the block)."""
    settings = reference_settings()
    run_logger = open_run_logger(settings, reference_stage_manager(settings), tiny_model, resolved, tmp_path, FakeClock())
    run_logger.status("training")
    with run_logger.saving_checkpoint():
        pass
    run_logger.status("stopping after this step, saving a checkpoint")
    with pytest.raises(OSError), run_logger.saving_checkpoint():
        raise OSError("disk full")
    assert recording(run_logger).statuses == [
        "training",
        "saving checkpoint",
        "training",
        "stopping after this step, saving a checkpoint",
        "saving checkpoint",
        "stopping after this step, saving a checkpoint",
    ]


def test_run_logger_drives_a_real_training_dashboard(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path
) -> None:
    """The same calls on a `TrainingDashboard` over a StringIO console (not entered: rendering only): after a log
    step the frame shows the loss and the bar counts, `log_checkpoint` shows up in the events, `evaluating()` in the
    status, the validation losses in their table."""
    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    board = string_console_dashboard(stage_manager)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock, dashboard=board)
    assert run_logger.dashboard is board
    progress = TrainingProgress()
    run_logger.status("training")
    run_fake_steps(run_logger, stage_manager, progress, clock, 3, 2.0)
    text = board.render_text()
    assert "3/8" in text and "0/4" in text and "3/12" in text, "the stage bars and the overall bar count optimizer steps"
    assert "▶ a" in text and "2.0000" in text and "2.00e-04" in text and "0.500" in text and "training" in text
    assert "2.00s" in text, "seconds/step from the step dict"

    run_logger.log_checkpoint(tmp_path / "checkpoints" / "step-00000003-steps.pth")
    assert board.events()[-1].endswith("step 3: saved checkpoint " + str(tmp_path / "checkpoints" / "step-00000003-steps.pth"))
    with run_logger.evaluating():
        assert "evaluating" in board.render_text()
    result = fake_result(stage_manager, progress.step, validation={"val_loss": torch.tensor(2.5), "val_loss_1": torch.tensor(2.6)})
    progress.advance()
    run_logger.log_step(result, progress)
    text = board.render_text()
    assert "validation (step 4)" in text and "val_loss_1" in text and "2.6000" in text and "2.5000" in text
    run_fake_steps(run_logger, stage_manager, progress, clock, 3, 1.0)  # steps 5-7: the transition starts after 6
    text = board.render_text()
    assert "▶ a" in text and "transition → b 50%" in text and "7/8" in text
    run_logger.close(progress, None)
    assert "finished" in board.render_text()


def test_open_picks_the_console_fallback_under_pytest_and_writes_train_log(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without an injected dashboard `open` goes through `open_dashboard`: stdout is not a TTY under pytest, so the
    `ConsoleFallbackDashboard` is chosen, built from the run (stage names and step counts from the boundaries, the header
    details, the log interval, the resume step) with the `training` logger attached for the block and
    `run_directory / train.log` appended — the header records, the fallback's step lines and events all end up there
    and on stderr (where the CLI's log handlers write too, so a piped run's story stays in one stream);
    `close()` detaches it again."""
    monkeypatch.setenv("TRAINING_DASHBOARD", "1")
    settings = reference_settings(log_step_interval=2)
    stage_manager = two_stage_manager(settings)
    progress = TrainingProgress(step=4, resume_step=4)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    training_logger = logging.getLogger("training")
    handlers_before = list(training_logger.handlers)
    with RunLogger.open(settings, tmp_path, resolved, tiny_model, stage_manager, progress, backend, clock=FakeClock()) as run_logger:
        board = run_logger.dashboard
        assert isinstance(board, ConsoleFallbackDashboard)
        assert board.stage_names == ["a", "b"] and board.steps_per_stage == [8, 4] and board.total_steps == 12
        assert board.details == {"model": "tiny", "dataset": "tiny", "device": "cpu", "precision": "32"}
        assert board.log_step_interval == 2
        assert len(training_logger.handlers) == len(handlers_before) + 2, "the dashboard handler and the file handler"
        run_logger.log_resume(tmp_path / "step-00000004-steps.pth", 4)
        run_fake_steps(run_logger, stage_manager, progress, FakeClock(), 8, 1.0)
        run_logger.log_checkpoint(tmp_path / "checkpoints" / "step-00000012-steps.pth")
        run_logger.close(progress, None)
    assert training_logger.handlers == handlers_before
    log_text = (tmp_path / TRAIN_LOG_NAME).read_text()
    assert "Total training steps: 12 (2 micro-batches each)" in log_text
    assert "event: resumed from" in log_text and "event: saved checkpoint" in log_text
    assert "step 6/12 | stage 0 a | " in log_text and "step 12/12 | stage 1 b | " in log_text and "step 5/12" not in log_text
    assert "Training finished after 12 steps" in log_text
    err = capsys.readouterr().err
    assert "step 12/12 | stage 1 b | " in err and "event: saved checkpoint" in err, "the fallback's lines go to stderr"


def test_open_dashboard_arguments(tmp_path: Path) -> None:
    """`open_dashboard` passes the run to the factory: one bar per stage, the config file names as the header
    details, the log interval and the resume step; the fallback is chosen when the display is disabled."""
    settings = reference_settings(log_step_interval=3, eval_step_interval=99)  # eval must be a multiple of log
    stage_manager = two_stage_manager(settings)
    with open_dashboard(settings, tmp_path, stage_manager, start_step=5, device="cuda:0") as board:
        assert isinstance(board, ConsoleFallbackDashboard)  # stdout is not a TTY under pytest
        assert (board.run_name, board.stage_names, board.steps_per_stage, board.total_steps) == ("steps", ["a", "b"], [8, 4], 12)
        assert board.details == {"model": "tiny", "dataset": "tiny", "device": "cuda:0", "precision": "32"}
        assert board.log_step_interval == 3
        board.update_step(6, 0, None, {"loss": 1.0})
    assert (tmp_path / TRAIN_LOG_NAME).exists() and "step 6/12" in (tmp_path / TRAIN_LOG_NAME).read_text()


def _display_is_up(board: TrainingDashboard) -> bool:
    """Through a call: mypy would otherwise keep the narrowing of ``board._live`` across the ``with`` block."""
    return board._live is not None


def test_open_dashboard_builds_the_live_display_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the display enabled (`dashboard_enabled`: a terminal and `TRAINING_DASHBOARD` not `0`) `open_dashboard`
    builds the live `TrainingDashboard` from the same run description, up for the block and closed after it, with
    stderr as the stream a display that disables itself falls back to (where the CLI's log handlers write)."""
    monkeypatch.setattr("training.logger.dashboard_enabled", lambda: True)
    settings = reference_settings(log_step_interval=3, eval_step_interval=99)
    stage_manager = two_stage_manager(settings)
    with open_dashboard(settings, tmp_path, stage_manager, start_step=5, device="cpu") as board:
        assert isinstance(board, TrainingDashboard) and _display_is_up(board) and board.enabled
        assert (board.run_name, board.stage_names, board.steps_per_stage, board.total_steps) == ("steps", ["a", "b"], [8, 4], 12)
        assert board.details == {"model": "tiny", "dataset": "tiny", "device": "cpu", "precision": "32"}
        assert board.log_step_interval == 3
        board.update_step(6, 0, None, {"loss": 1.0})
    assert not _display_is_up(board) and board._plain_stream is sys.stderr
    assert "step 6/12" in (tmp_path / TRAIN_LOG_NAME).read_text()


class RaisingTracker(Logger):
    """A `Logger` whose `finish()` fails, as a broken `wandb.finish()` would (disabled: wandb is never imported)."""

    def __init__(self, out_dir: Path) -> None:
        super().__init__("proj", "run", out_dir, enabled=False)
        self.finish_calls = 0

    def finish(self) -> None:
        self.finish_calls += 1
        raise RuntimeError("wandb finish failed")


def _logger_with(tracker: Logger, stage_manager: StageManager, settings: Settings, run_directory: Path) -> RunLogger:
    return RunLogger(
        settings, run_directory, stage_manager, tracker, start_step=0, device="cpu",
        dashboard=RecordingDashboard(), clock=FakeClock(),
    )


def test_exit_releases_every_resource_even_when_the_tracker_raises(tmp_path: Path) -> None:
    """A failing `wandb.finish()` used to leave the terminal with the dashboard's redirected streams and a hidden
    cursor: the resources are released whatever the tracker does."""
    settings = reference_settings()
    tracker = RaisingTracker(tmp_path)
    run_logger = _logger_with(tracker, reference_stage_manager(settings), settings, tmp_path)
    released: list[str] = []
    run_logger.resources.callback(released.append, "dashboard")
    with pytest.raises(RuntimeError, match="wandb finish failed"), run_logger:
        pass
    assert tracker.finish_calls == 1 and released == ["dashboard"]


def test_exit_raises_the_first_failure_and_still_releases_the_rest(tmp_path: Path) -> None:
    """Both teardowns fail: everything is released and the first failure is the one raised, not the last."""
    settings = reference_settings()
    tracker = RaisingTracker(tmp_path)
    run_logger = _logger_with(tracker, reference_stage_manager(settings), settings, tmp_path)
    released: list[str] = []

    def failing_release() -> None:
        released.append("dashboard")
        raise ValueError("dashboard teardown failed")

    run_logger.resources.callback(failing_release)
    with pytest.raises(RuntimeError, match="wandb finish failed"):
        run_logger.__exit__(None, None, None)
    assert tracker.finish_calls == 1 and released == ["dashboard"]


def test_close_of_a_stopped_run(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """`close(..., stopped=True)` (the stop request of `train()`): the report says so, its summary tells how to
    continue, the final `keep` line and the final dashboard status read "stopped on request"."""
    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    run_fake_steps(run_logger, stage_manager, progress, clock, 5, 1.0)
    run_logger.log_checkpoint(tmp_path / "checkpoints" / "step-00000005-steps.pth")
    report = run_logger.close(progress, None, stopped=True)
    assert recording(run_logger).statuses == ["stopped on request"]
    assert report.stopped is True and (report.steps_completed, report.final_step) == (5, 5)
    assert report.export_dir is None
    lines = report.summary().splitlines()
    assert lines[2] == "  stopped on request after step 5; rerun with resume: true to continue"
    final = [r for r in console_records.records if r.getMessage() == "Training stopped on request after 5 steps in 5.0s."]
    assert len(final) == 1 and getattr(final[0], "keep", False) is True
    assert TrainingReport(**{**report.__dict__, "stopped": False}).stopped is False  # default when not given
