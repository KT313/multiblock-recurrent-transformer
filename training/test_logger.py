# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the disabled logger (no-ops), the stubbed wandb path (quiet settings included), the gradient/parameter
metric helpers and `RunLogger` (the header records via `caplog` on `training.logger`, the dashboard calls on a
recording fake and on a real `TrainingDashboard` over a StringIO console, the fallback picked under pytest and its
`train.log`, wandb dict, timers on a fake clock, data composition, history, `TrainingReport`).
"""

import io
import json
import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from rich.console import Console

from data_preparation.dataset_config import DatasetConfig
from evaluation.samples import GeneratedSample
from model import RecurrentGPT
from training.backend.single_device import SingleDeviceBackend
from training.data.dataset_resolver import ResolvedDataset
from training.logger import (
    DATA_WAIT_WARNING_INTERVAL_SECONDS,
    NullDashboard,
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
from training.test_step import PACK_LENGTH, reference_settings, reference_stage_manager
from training.ui.board import TrainingDashboard
from training.ui.capture import WANDB_QUIET_SETTINGS
from training.ui.common import TRAIN_LOG_NAME, TRAIN_REPORT_NAME
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
    """
    The wandb calls are exercised against a stub run so no wandb import/network happens.
    """

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
        """
        Stands in for `wandb.Settings`: records what `Logger` asks for.
        """

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
    assert calls[0][1]["group"] == "run" and calls[0][1]["tags"] == [] and calls[0][1]["config"] == {"resume_step": None}
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

    Logger("proj", "run", tmp_path / "resumed", enabled=True, resume_step=1000)  # linked to the first process's run
    resumed = calls[-1][1]
    assert (resumed["name"], resumed["group"], resumed["tags"]) == ("run-from-1000", "run", ["resumed"])
    assert resumed["config"] == {"resume_step": 1000}


def test_to_scalar() -> None:
    assert _to_scalar(torch.tensor([2.5])) == 2.5
    assert _to_scalar(torch.tensor(4)) == 4
    assert _to_scalar(3) == 3
    t = torch.zeros(2)
    assert _to_scalar(t) is t


def test_describe_parameters_counts_total_recurrent_and_unrolled(tiny_model: RecurrentGPT) -> None:
    """
    The line names the total, the parameters of the core blocks and the unrolled count at the mean recurrence
    (tiny: mean_recurrence [2, 2], so unrolled = total + recurrent).
    """

    total = num_parameters(tiny_model)
    recurrent = sum(p.numel() for block in tiny_model.transformer.core_blocks for p in block.parameters())
    assert 0 < recurrent < total
    expected = f"Model: {total:,} parameters, {recurrent:,} in recurrent blocks, unfolds to {total + recurrent:,} at mean recurrence."
    assert describe_parameters(tiny_model) == expected


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
    loss = tiny_model(x, labels=x, num_steps=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    opt.step()
    return opt


N_ATTN_LAYERS = 2 + 2 + 1  # prelude + core blocks (1 layer each) + coda


def test_track_gradient_metrics_on_tiny_model(tiny_model: RecurrentGPT) -> None:
    opt = _step_tiny(tiny_model)
    state_before = {id(p): {k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for p, s in opt.state.items()}
    metrics = track_gradient_metrics(tiny_model, opt)
    for param, state in opt.state.items():  # a log step changes no state (the resume checks rely on it)
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, state_before[id(param)][key]), key

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
    loss = tiny_model(x, labels=x, num_steps=(0, 2))["loss"]
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


def test_nan_gradient_metrics_match_the_per_parameter_values() -> None:
    torch.manual_seed(0)
    proj = torch.nn.Linear(3, 2)
    model = torch.nn.Sequential()
    model.add_module("mlp", torch.nn.Sequential())
    model[0].add_module("proj", proj)  # named `mlp.proj.weight`: counted as an `ffn2_grad_<i>` weight
    model.add_module("head", torch.nn.Linear(2, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, eps=1e-8)
    model(torch.randn(4, 3)).sum().backward()
    opt.step()
    assert proj.weight.grad is not None
    proj.weight.grad.fill_(float("NaN"))
    metrics = track_gradient_metrics(model, opt)

    assert set(metrics) == {"ffn2_grad_0", "avg_RMS", "local_l1_grad_norm", "l2_param_norm", "l1_param_norm"}
    assert math.isnan(metrics["ffn2_grad_0"].item())
    finite = [p for p in model.parameters() if p is not proj.weight]  # in optimizer-group order
    rms = [
        p.grad.pow(2).div(opt.state[p]["exp_avg_sq"].clamp(min=1e-16)).mean().sqrt() for p in finite if p.grad is not None
    ]
    assert torch.equal(metrics["avg_RMS"], torch.as_tensor(sum(rms) / len(rms)))
    l1_norms = [p.grad.norm(1.0) for p in finite if p.grad is not None]
    assert torch.equal(metrics["local_l1_grad_norm"], torch.stack(l1_norms).mean())
    assert torch.equal(metrics["l2_param_norm"], torch.stack([p.norm() for p in model.parameters()]).norm())
    assert torch.equal(metrics["l1_param_norm"], torch.stack([p.norm(1.0) for p in model.parameters()]).mean())


# --------------------------------------------------------------------------------------------------------------
# RunLogger: a run without a dataset (in-memory settings, a stage manager, fake step results, a fake clock)


class FakeClock:
    """
    A clock that only moves when a test says so (`clock=` of `RunLogger.open`).
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


TOKENS_PER_STEP = 2 * PACK_LENGTH  # reference_settings: two packs of PACK_LENGTH tokens per optimizer step
STEP_KEYS = {
    "loss", "ppl", "lr", "grad_norm", "step", "seconds/step", "tokens/second", "total_tokens", "total_time",
    "remaining_time", "stage/current_stage", "stage/base_lr", "stage/in_transition", "stage/transition_progress",
    "stage/stage_progress", "data/wait_seconds", "data/wait_fraction",
}  # fmt: skip


def two_stage_manager(settings: Settings) -> StageManager:
    """
    Two stages with a transition between them (stage a: 8 steps, the last 25 % transitioning; stage b: 4 steps).
    """

    stages = [
        resolved_stage("a", tokens=8 * TOKENS_PER_STEP, base_lr=3e-4, transition_pct=0.25),
        resolved_stage("b", tokens=4 * TOKENS_PER_STEP, base_lr=1e-4, transition_pct=0.0),
    ]
    return StageManager(stages, settings.tokens_per_optimizer_step, warmup_steps=2, cooldown_steps=2)


def fake_result(
    stage_manager: StageManager,
    step: int,
    *,
    loss: float = 2.0,
    data_ids: list[str] | None = None,
    data_tokens: dict[str, int] | None = None,
    metrics: dict[str, torch.Tensor] | None = None,
    validation: dict[str, torch.Tensor] | None = None,
) -> StepResult:
    """
    A `StepResult` as `run_one_optimizer_step` returns it, with tensors where the step has tensors. Without
    `data_tokens` every document counts 64 tokens, so the token composition equals the document composition.
    """

    ids = data_ids if data_ids is not None else ["source_a"] * 4
    if data_tokens is None:
        data_tokens = {}
        for data_id in ids:
            data_tokens[data_id] = data_tokens.get(data_id, 0) + 64
    return StepResult(
        step=step,
        learning_rate=1e-4 * step,
        loss=torch.tensor(loss),
        grad_norm=torch.tensor(0.5),
        stage=stage_manager.get_stage_info(step),
        data_ids=ids,
        data_tokens=data_tokens,
        metrics=metrics or {},
        validation=validation,
    )


@pytest.fixture
def resolved(tiny_dataset_config: DatasetConfig) -> ResolvedDataset:
    """
    Only `config_hash` is read by `RunLogger.open` (the wandb hyperparameters).
    """

    return ResolvedDataset(
        config=tiny_dataset_config, config_hash="hash-1", tokenizer_dir="unused", stages=[], train_sources=[], validation_rows={}, source_rows={}, rows_on_disk={}
    )


@pytest.fixture
def console_records(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger=CONSOLE_LOGGER_NAME)
    return caplog


class RecordingDashboard:
    """
    A `Dashboard` that records every call (`RunLogger`'s side of the dashboard API, without a display).
    """

    def __init__(self) -> None:
        # (step, stage index, transition progress or None, the step dict as passed)
        self.steps: list[tuple[int, int, float | None, dict[str, object]]] = []
        self.validations: list[tuple[int, dict[str, object]]] = []
        self.events: list[str] = []
        self.statuses: list[str] = []
        self.discounted: list[float] = []  # the seconds of every block that was not a training step
        self.micro_batches: list[tuple[int, int]] = []  # (completed, total) of every micro-batch report

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

    def discount_time(self, seconds: float) -> None:
        self.discounted.append(seconds)

    def update_micro_batch(self, completed: int, total: int) -> None:
        self.micro_batches.append((completed, total))


def string_console_dashboard(stage_manager: StageManager, log_step_interval: int = 1) -> TrainingDashboard:
    """
    A real `TrainingDashboard` rendering into a StringIO (never entered: no live display, no terminal capture).
    """

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
    """
    `RunLogger.open` on the CPU backend; `dashboard` defaults to a fresh `RecordingDashboard` (available as
    `run_logger.dashboard`), so no test opens the real factory unless it asks for it (`dashboard=None` explicitly is
    not possible here; call `RunLogger.open` directly for that).
    """

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
        wall_clock=clock,  # the fake timeline stands in for the CLI's wall clock too (`setup_started`)
        setup_started=setup_started,
        keep_history=True,
    )


def recording(run_logger: RunLogger) -> RecordingDashboard:
    """
    The `RecordingDashboard` behind a logger opened by `open_run_logger` without a dashboard of its own.
    """

    assert isinstance(run_logger.dashboard, RecordingDashboard)
    return run_logger.dashboard


def run_fake_steps(
    run_logger: RunLogger, stage_manager: StageManager, progress: TrainingProgress, clock: FakeClock, steps: int, seconds_per_step: float
) -> None:
    """
    `steps` fake optimizer steps, each taking `seconds_per_step`, logged as `train()` logs them.
    """

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
    """
    The stage summary, the total-steps line, the parameter line and the setup line are INFO records on
    `training.logger` marked `keep`; the setup timer is measured from `setup_started` to `open`.
    """

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
    """
    Every log step: the same metric dict (already scalars) to wandb, as floats to `history[done]` and to the
    dashboard's `update_step` (no tensor in it); the throughput arithmetic on a fake clock ticking 2 s per step; no
    per-step console record (the fallback dashboard's line is the one console line of a step).
    """

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
    """
    Without `keep_history` (the CLI's default) `history` stays empty while wandb and the dashboard still get every
    log step's metric dict: a long run does not hold its metrics in memory.
    """

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
    """
    `log_step_interval: 2`: only even steps are logged (no `.item()` in between; the dashboard gets an empty step
    dict at the odd steps, which only moves its bars), `seconds/step` is the interval time per step, the composition
    counts the document TOKENS of every step since the last log step (pack tails excluded, `result.data_tokens`;
    the document counts play no part: step 1 has as many `a` as `b` documents but three times the `b` tokens) and
    starts over afterwards. The final step is logged whatever the interval (with `log_step_interval: 3` it would
    otherwise be dropped, its validation with it).
    """

    recorded = _record_wandb_logs(monkeypatch)
    settings = reference_settings(log_step_interval=2)
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    batches = [
        (["a", "a", "b", "b"], {"a": 100, "b": 300}),  # 2 + 2 documents, 1 : 3 in tokens
        (["b", "b", "b", "b"], {"b": 400}),
        (["a"] * 4, {"a": 500}),
        (["a"] * 4, {"a": 300}),
    ]
    for data_ids, data_tokens in batches:
        result = fake_result(stage_manager, progress.step, data_ids=data_ids, data_tokens=data_tokens)
        progress.advance()
        clock.advance(1.0)
        run_logger.log_step(result, progress)
    assert sorted(run_logger.history) == [2, 4] and sorted(recorded) == [2, 4]
    final_logger = open_run_logger(reference_settings(log_step_interval=3, eval_step_interval=3), stage_manager, tiny_model, resolved, tmp_path, clock)
    final_progress = TrainingProgress()
    while final_progress.step < stage_manager.total_steps:
        result = fake_result(stage_manager, final_progress.step, data_ids=["a"] * 4)
        final_progress.advance()
        final_logger.log_step(result, final_progress)
    assert stage_manager.total_steps % 3 != 0 and stage_manager.total_steps in final_logger.history
    shown = recording(run_logger).steps
    assert [(step, stage) for step, stage, _, _ in shown] == [(1, 0), (2, 0), (3, 0), (4, 0)], "the bars move every step"
    assert shown[0][3] == {} and shown[2][3] == {}, "nothing is read from the step's tensors at a non-log step"
    assert shown[1][3]["loss"] == 2.0 and shown[3][3]["step"] == 4
    second, fourth = run_logger.history[2], run_logger.history[4]
    assert second["seconds/step"] == 1.0 and second["tokens/second"] == TOKENS_PER_STEP
    assert second["data_composition/a"] == 0.125 and second["data_composition/b"] == 0.875  # 100 : 700 tokens
    assert fourth["data_composition/a"] == 1.0 and "data_composition/b" not in fourth
    for metrics in (second, fourth):
        assert sum(v for k, v in metrics.items() if k.startswith("data_composition/")) == pytest.approx(1.0)


def test_log_step_notes_the_transition_events_and_moves_the_bars_with_the_stage_at_done(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    Stage a: 8 steps, the last two (6, 7) transitioning to b. One "starting transition" event after step 6 is
    done and one "transition complete" event after step 8 is done (worded as the thesis loop printed them, with the
    stage names); no console record for them. The bars get the stage containing `done` (a until 8 steps are done, b
    from then on) and the transition keys of `done`, while `history` keeps the `stage/*` metrics of the step trained
    on, one step behind: `stage/current_stage` is the stage containing that step, also inside its transition.
    """

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
    """
    The `evaluating()` block shows `evaluating` as the status (the previous status afterwards) and its duration
    becomes `val_time` of that step's validation metrics (floats in the metric dict, `history` and the report); the
    dashboard gets the `val_loss*` entries as floats; a validation without a timed block reports 0 s (the timer is
    consumed, never stale).
    """

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


def test_side_blocks_are_kept_out_of_the_throughput_metrics(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path
) -> None:
    """
    L-H1: `seconds/step`, `tokens/second` and `remaining_time` are training only. A 30 s evaluation between two
    0.5 s steps used to read as a 60x slowdown; every block timed by the logger (evaluation, checkpoint, samples,
    benchmarks) comes off the interval and off the dashboard's own estimate (`discount_time`). `total_time` stays
    wall time.
    """

    settings = reference_settings()  # log_step_interval 1: every step carries a metric dict
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()

    run_fake_steps(run_logger, stage_manager, progress, clock, 1, 0.5)
    with run_logger.evaluating():
        clock.advance(30.0)
    run_fake_steps(run_logger, stage_manager, progress, clock, 1, 0.5)
    assert run_logger.history[2]["seconds/step"] == 0.5
    assert run_logger.history[2]["tokens/second"] == TOKENS_PER_STEP / 0.5
    assert run_logger.history[2]["remaining_time"] == 0.5 * (stage_manager.total_steps - 2)
    assert run_logger.history[2]["total_time"] == 31.0, "total_time is the wall time since `open`, evaluation included"

    with run_logger.saving_checkpoint():
        clock.advance(20.0)
    with run_logger.working("sampling"):
        clock.advance(4.0)
    run_fake_steps(run_logger, stage_manager, progress, clock, 1, 0.5)
    assert run_logger.history[3]["seconds/step"] == 0.5, "a checkpoint and a sampling block are not training either"
    assert recording(run_logger).discounted == [30.0, 20.0, 4.0]
    run_logger.note_micro_batch(1, 4)
    run_logger.note_micro_batch(4, 4)
    assert recording(run_logger).micro_batches == [(1, 4), (4, 4)], "handed to the dashboard as they are"


def test_data_wait_metrics_and_the_rate_limited_warning(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    The seconds the loaders blocked (`data_wait` of `log_step`) are summed per log interval into `data/wait_seconds`
    and, against the interval's training time, `data/wait_fraction`. Above `DATA_WAIT_WARNING_FRACTION` one kept
    WARNING names the slowest source and the dashboard gets an event; the warning repeats at most every
    `DATA_WAIT_WARNING_INTERVAL_SECONDS` while the wait persists, and a quiet interval reports zeros.
    """

    settings = reference_settings()  # log_step_interval 1
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    console_records.clear()

    def step(seconds: float, data_wait: dict[str, float] | None) -> None:
        result = fake_result(stage_manager, progress.step)
        progress.advance()
        clock.advance(seconds)
        run_logger.log_step(result, progress, data_wait=data_wait)

    step(1.0, {"a": 0.15, "b": 0.05})  # 20 % of the training time
    assert run_logger.history[1]["data/wait_seconds"] == pytest.approx(0.2)
    assert run_logger.history[1]["data/wait_fraction"] == pytest.approx(0.2)
    warnings = [record for record in console_records.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and getattr(warnings[0], "keep", False) is True
    assert "waited 0.2s for training data over the last 1 step(s), 20% of the training time" in warnings[0].getMessage()
    assert "(slowest: a 0.1s, b 0.1s)" in warnings[0].getMessage()  # per-source seconds, largest first
    assert recording(run_logger).events[-1] == "waiting for training data: 20% of the training time (a 0.1s, b 0.1s)"

    step(1.0, {"a": 0.5})  # still above the threshold, but inside the quiet interval: no second warning
    assert run_logger.history[2]["data/wait_fraction"] == pytest.approx(0.5)
    step(1.0, None)  # a quiet interval reports zeros
    assert (run_logger.history[3]["data/wait_seconds"], run_logger.history[3]["data/wait_fraction"]) == (0.0, 0.0)
    step(1.0, {"a": 0.01})  # below the threshold: a metric, no warning
    assert run_logger.history[4]["data/wait_fraction"] == pytest.approx(0.01)
    assert len([record for record in console_records.records if record.levelno == logging.WARNING]) == 1

    with run_logger.saving_checkpoint():  # a side block: not training time, so the fraction ignores it, but the
        clock.advance(DATA_WAIT_WARNING_INTERVAL_SECONDS)  # quiet interval of the warning has passed
    step(1.0, {"b": 0.3})
    assert run_logger.history[5]["data/wait_fraction"] == pytest.approx(0.3)
    warnings = [record for record in console_records.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2 and "(slowest: b 0.3s)" in warnings[1].getMessage()


def test_a_non_main_rank_logs_nothing_and_writes_no_file(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    On a rank that is not the main one `RunLogger.open` drives a `NullDashboard` (so `open_dashboard` never runs and
    no `train.log` appears), keeps wandb off whatever the settings say, sends its console records to the silent
    logger, and `close` returns the report without writing `train_report.json`. Every call the loop makes still works.
    """

    class NonMainBackend(SingleDeviceBackend):
        is_main = False

    settings = reference_settings(wandb_enabled=True)
    stage_manager = reference_stage_manager(settings)
    progress = TrainingProgress()
    run_logger = RunLogger.open(
        settings, tmp_path, resolved, tiny_model, stage_manager, progress, NonMainBackend(device="cpu", precision="32"),
        clock=FakeClock(), keep_history=True,
    )
    with run_logger:
        assert isinstance(run_logger.dashboard, NullDashboard) and run_logger.is_main is False
        assert run_logger.wandb.enabled is False
        run_logger.log_fresh_start()
        run_logger.log_triggers("samples", [5])
        with run_logger.evaluating():
            pass
        result = fake_result(stage_manager, 0)
        progress.advance()
        run_logger.log_step(result, progress, data_wait={"a": 5.0})
        with run_logger.saving_checkpoint():
            run_logger.log_checkpoint(tmp_path / "x.pth")
        run_logger.log_benchmark_failure(RuntimeError("no harness"))
        report = run_logger.close(progress, None)
    assert report.completed_steps == 1 and report.checkpoints_written == [tmp_path / "x.pth"]
    assert run_logger.history[1]["loss"] == 2.0  # the loop's bookkeeping runs as on the main rank
    assert console_records.records == []  # nothing on the `training.logger` logger, the header lines included
    assert not (tmp_path / TRAIN_LOG_NAME).exists() and not (tmp_path / TRAIN_REPORT_NAME).exists()


def test_a_failing_run_logs_its_traceback_and_leaves_no_report(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    L-H2: leaving the `with` by raising, without having reached `close()`, writes one kept ERROR record with the
    traceback into `train.log` - while the dashboard's file handler is still attached - and sets the status to
    `failed`. No `train_report.json`: it would read as the result of a run that has none.
    """

    settings = reference_settings()
    stage_manager = two_stage_manager(settings)
    progress = TrainingProgress()
    backend = SingleDeviceBackend(device="cpu", precision="32")
    statuses: list[str] = []
    opened = RunLogger.open(settings, tmp_path, resolved, tiny_model, stage_manager, progress, backend, clock=FakeClock())
    with pytest.raises(RuntimeError, match="the step exploded"), opened as run_logger:
        monkeypatch.setattr(run_logger.dashboard, "set_status", statuses.append)
        run_fake_steps(run_logger, stage_manager, progress, FakeClock(), 2, 1.0)
        raise RuntimeError("the step exploded")
    assert statuses == ["failed"]
    log_text = (tmp_path / TRAIN_LOG_NAME).read_text()
    assert "ERROR training.logger: Training failed: the step exploded" in log_text
    assert "Traceback (most recent call last)" in log_text and "RuntimeError: the step exploded" in log_text
    assert "step 2/12" in log_text, "the record is written before `__exit__` takes the file handler away"
    assert not (tmp_path / TRAIN_REPORT_NAME).exists(), "a failed run has no report"


def test_close_after_a_failure_is_still_the_run_that_reports(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    L-H2: a run that handled its exception and closed on its own (`close()` inside the block) is finished, not
    failed - `__exit__` adds nothing.
    """

    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    progress = TrainingProgress()
    opened = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    with pytest.raises(RuntimeError, match="handled"), opened as run_logger:
        run_logger.close(progress, None)
        raise RuntimeError("handled")
    assert recording(run_logger).statuses == ["finished"], "the status of the run that closed itself stands"
    assert not any("Training failed" in r.getMessage() for r in console_records.records)
    assert (tmp_path / TRAIN_REPORT_NAME).exists(), "the report `close()` wrote is the run's result"


def test_close_returns_the_report_of_a_resumed_run_and_is_idempotent(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    A run resumed at step 4 that ran 3 steps, wrote 2 checkpoints and exported: every report field, the summary
    text, the events of resume / checkpoint / export, the `keep` line and the status of the finish, and `close()` +
    `__exit__` releasing the resources exactly once.
    """

    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock(500.0)
    released: list[str] = []
    resume_path = tmp_path / "checkpoints" / "step-00000004-steps.pth"
    with open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock, start_step=4, setup_started=490.0) as run_logger:
        run_logger._exit_stack.callback(released.append, "released")
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
        assert run_logger.close(progress, export_dir) is report  # a second close changes nothing
    assert released == ["released"]  # `__exit__` after `close()` is a no-op
    with run_logger:
        pass
    assert released == ["released"]

    assert isinstance(report, TrainingReport)
    assert (report.run_directory, report.steps_this_process, report.completed_steps) == (tmp_path, 3, 7)
    assert report.resumed_from == resume_path
    assert report.setup_seconds == 10.0 and report.train_seconds == 6.0
    assert report.last_loss == 2.0 and report.last_validation == {}
    assert report.checkpoints_written == [first, second] and report.export_dir == export_dir
    assert sorted(report.history) == [5, 6, 7] and report.history is run_logger.history
    written = json.loads((tmp_path / TRAIN_REPORT_NAME).read_text())
    assert written.pop("written_at").startswith("20")
    assert written == {
        "run_directory": str(tmp_path),
        "steps_this_process": 3,
        "completed_steps": 7,
        "resumed_from": str(resume_path),
        "setup_seconds": 10.0,
        "train_seconds": 6.0,
        "last_loss": 2.0,
        "last_validation": {},
        "checkpoints_written": [str(first), str(second)],
        "export_dir": str(export_dir),
        "stopped": False,
        "samples_written": [],
        "last_benchmarks": {},
    }
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
    assert recording(run_logger).statuses == ["finished"]  # only the first `close()` sets it
    kept = [r.getMessage() for r in console_records.records if getattr(r, "keep", False)]
    assert kept.count("Training finished after 7 steps in 6.0s.") == 1 and not any("checkpoint" in k for k in kept)


def test_log_samples_and_benchmarks_reach_the_dashboard_wandb_and_report(
    tiny_model: RecurrentGPT,
    resolved: ResolvedDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    console_records: pytest.LogCaptureFixture,
) -> None:
    """
    `log_samples` notes the file and previews the first sample; `log_benchmarks` sends the scores to wandb at the
    step and notes one event per task; a failure is a kept warning plus an event; the report and its JSON carry
    the samples files and the last scores.
    """

    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    recorded = _record_wandb_logs(monkeypatch)
    metrics = {"benchmark/mean/arc_easy/acc": 0.25, "benchmark/mean/arc_easy/acc_norm": 0.3, "benchmark/4-4/hellaswag/acc": 0.26}
    first, second = tmp_path / "samples" / "step-00000010.jsonl", tmp_path / "samples" / "step-00000020.jsonl"
    with open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, FakeClock(0.0)) as run_logger:
        run_logger.log_samples(first, [GeneratedSample("Once", "continuation", "upon a time", 3, False)])
        run_logger.log_samples(second, [])
        with run_logger.working("benchmarking"):
            pass
        run_logger.log_benchmarks(metrics, tmp_path / "benchmarks" / "step-00000007.json", 7)
        run_logger.log_benchmark_failure(RuntimeError("no network"))
        report = run_logger.close(TrainingProgress(step=7), None)
    events = recording(run_logger).events
    assert f"wrote 1 samples to {first}" in events and "sample: 'Once' -> 'upon a time'" in events
    assert f"wrote 0 samples to {second}" in events
    assert "benchmark arc_easy (recurrence mean) at step 7: acc 0.2500, acc_norm 0.3000" in events
    assert "benchmark hellaswag (recurrence 4-4) at step 7: acc 0.2600" in events
    assert f"wrote benchmark results to {tmp_path / 'benchmarks' / 'step-00000007.json'}" in events
    assert "benchmark evaluation failed: no network" in events
    assert "benchmarking" in recording(run_logger).statuses
    assert recorded == {7: metrics}
    assert report.samples_written == [first, second] and report.last_benchmarks == metrics
    summary = report.summary()
    assert "  benchmarks: mean/arc_easy/acc 0.2500, mean/arc_easy/acc_norm 0.3000, 4-4/hellaswag/acc 0.2600" in summary
    assert f"  2 samples files written, last: {second}" in summary
    written = json.loads((tmp_path / TRAIN_REPORT_NAME).read_text())
    assert written["samples_written"] == [str(first), str(second)] and written["last_benchmarks"] == metrics
    kept = [r.getMessage() for r in console_records.records if getattr(r, "keep", False)]
    assert "benchmark evaluation failed, the run continues: no network" in kept


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
    assert (report.steps_this_process, report.completed_steps, report.resumed_from, report.last_loss) == (0, 0, None, None)
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
    """
    Every console line is a logging record or a dashboard call; with a recording dashboard nothing reaches stdout /
    stderr (there is no handler on `training`), and the module has no `print` at all.
    """

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
    """
    `status(text)` sets the header status; `saving_checkpoint()` shows `saving checkpoint` for the block and puts
    the status from before it back (also after an exception in the block).
    """

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
    """
    The same calls on a `TrainingDashboard` over a StringIO console (not entered: rendering only): after a log
    step the frame shows the loss and the bar counts, `log_checkpoint` shows up in the events, `evaluating()` in the
    status, the validation losses in their table.
    """

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
    """
    Without an injected dashboard `open` goes through `open_dashboard`: stdout is not a TTY under pytest, so the
    `ConsoleFallbackDashboard` is chosen, built from the run (stage names and step counts from the boundaries, the header
    details, the log interval, the resume step) with the `training` logger attached for the block and
    `run_directory / train.log` appended: the header records, the fallback's step lines and events all end up there
    and on stderr (where the CLI's log handlers write too, so a piped run's story stays in one stream);
    `close()` detaches it again.
    """

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
    """
    `open_dashboard` passes the run to the factory: one bar per stage, the config file names as the header
    details, the log interval and the resume step; the fallback is chosen when the display is disabled.
    """

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
    """
    Through a call: mypy would otherwise keep the narrowing of board._live across the with block.
    """

    return board._live is not None


def test_open_dashboard_builds_the_live_display_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    With the display enabled (`dashboard_enabled`: a terminal and `TRAINING_DASHBOARD` not `0`) `open_dashboard`
    builds the live `TrainingDashboard` from the same run description, up for the block and closed after it, with
    stderr as the stream a display that disables itself falls back to (where the CLI's log handlers write).
    """

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
    """
    A `Logger` whose `finish()` fails, as a broken `wandb.finish()` would (disabled: wandb is never imported).
    """

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
    """
    A failing `wandb.finish()` used to leave the terminal with the dashboard's redirected streams and a hidden
    cursor: the resources are released whatever the tracker does.
    """

    settings = reference_settings()
    tracker = RaisingTracker(tmp_path)
    run_logger = _logger_with(tracker, reference_stage_manager(settings), settings, tmp_path)
    released: list[str] = []
    run_logger._exit_stack.callback(released.append, "dashboard")
    with pytest.raises(RuntimeError, match="wandb finish failed"), run_logger:
        pass
    assert tracker.finish_calls == 1 and released == ["dashboard"]


def test_exit_raises_the_first_failure_and_still_releases_the_rest(tmp_path: Path) -> None:
    """
    Both teardowns fail: everything is released and the first failure is the one raised, not the last.
    """

    settings = reference_settings()
    tracker = RaisingTracker(tmp_path)
    run_logger = _logger_with(tracker, reference_stage_manager(settings), settings, tmp_path)
    released: list[str] = []

    def failing_release() -> None:
        released.append("dashboard")
        raise ValueError("dashboard teardown failed")

    run_logger._exit_stack.callback(failing_release)
    with pytest.raises(RuntimeError, match="wandb finish failed"):
        run_logger.__exit__(None, None, None)
    assert tracker.finish_calls == 1 and released == ["dashboard"]


def test_close_of_a_stopped_run(
    tiny_model: RecurrentGPT, resolved: ResolvedDataset, tmp_path: Path, console_records: pytest.LogCaptureFixture
) -> None:
    """
    `close(..., stopped=True)` (the stop request of `train()`): the report says so, its summary tells how to
    continue, the final `keep` line and the final dashboard status read "stopped on request".
    """

    settings = reference_settings()
    stage_manager = reference_stage_manager(settings)
    clock = FakeClock()
    run_logger = open_run_logger(settings, stage_manager, tiny_model, resolved, tmp_path, clock)
    progress = TrainingProgress()
    run_fake_steps(run_logger, stage_manager, progress, clock, 5, 1.0)
    run_logger.log_checkpoint(tmp_path / "checkpoints" / "step-00000005-steps.pth")
    report = run_logger.close(progress, None, stopped=True)
    assert recording(run_logger).statuses == ["stopped on request"]
    assert report.stopped is True and (report.steps_this_process, report.completed_steps) == (5, 5)
    assert report.export_dir is None
    lines = report.summary().splitlines()
    assert lines[2] == "  stopped on request after step 5; rerun with resume: true to continue"
    final = [r for r in console_records.records if r.getMessage() == "Training stopped on request after 5 steps in 5.0s."]
    assert len(final) == 1 and getattr(final[0], "keep", False) is True
    assert TrainingReport(**{**report.__dict__, "stopped": False}).stopped is False  # default when not given
