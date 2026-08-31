# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Logging of a training run: the thin wandb wrapper (`Logger`, offline by default), the gradient / parameter metric
helpers that were logged, and `RunLogger` — every console line, timer and counter of a run in one place, ending in a
`TrainingReport`.

`RunLogger` never prints: every console line is an INFO record on the `training.logger` logger (the `training`
hierarchy the CLI attaches a stream handler to and the dashboard of task 10 takes over for the run). Lines that must
survive in the terminal scrollback under a live dashboard carry `extra={"keep": True}`.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer

from model import RecurrentGPT
from training.checkpoint import unwrap_compiled
from training.settings import Settings
from training.stage_manager import StageManager

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from training.backend import Backend
    from training.data.dataset_resolver import ResolvedDataset
    from training.step import StepResult, TrainingProgress  # `step.py` imports `track_gradient_metrics` from here

CONSOLE_LOGGER_NAME = "training.logger"  # under the `training` hierarchy; named explicitly, not via `__name__`
KEEP = {"keep": True}  # `extra=` of the records that must survive in the terminal scrollback under a live dashboard

console = logging.getLogger(CONSOLE_LOGGER_NAME)


class Logger:
    """wandb run wrapper; every method is a no-op when `enabled=False` (wandb is then never imported)."""

    def __init__(
        self, project: str, run_name: str, out_dir: str | Path, offline: bool = True, enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self.run: Optional[Run] = None
        if not enabled:
            return
        import wandb

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project, name=run_name, dir=str(out_dir), mode="offline" if offline else "online"
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        self.run.log({k: _to_scalar(v) for k, v in metrics.items()}, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.config.update(params, allow_val_change=True)  # type: ignore[no-untyped-call]  # wandb Config.update is unannotated

    def log_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for k, v in values.items():
            self.run.summary[k] = _to_scalar(v)

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
        self.run = None


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def num_parameters(model: Module, only_trainable: bool = False) -> int:
    """Total number of parameters (tied weights counted once, as `parameters()` deduplicates them)."""
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    return sum(p.numel() for p in param_list)


def describe_parameters(model: Module) -> str:
    """The parameter-count line printed at the start of a run: total parameters, parameters inside the recurrent core
    blocks and the count of the unrolled model at the mean recurrence (`total - recurrent + recurrent * mean of
    mean_recurrence`). Accepts the compiled wrapper too (it is unwrapped)."""
    plain_model = cast(RecurrentGPT, unwrap_compiled(model))
    total_parameters = num_parameters(plain_model)
    core_blocks = cast(Iterable[Module], plain_model.transformer.core_blocks)
    recurrent_parameters = sum(p.numel() for block in core_blocks for p in block.parameters())
    mean_recurrence = cast(list[int], plain_model.config.mean_recurrence)  # a list after RecurrentConfig.__post_init__
    mean_of_means = sum(mean_recurrence) / len(mean_recurrence)
    unrolled_parameters = int(total_parameters - recurrent_parameters + recurrent_parameters * mean_of_means)
    return (
        f"Model: {total_parameters:,} parameters, {recurrent_parameters:,} in recurrent blocks, unfolds to "
        f"{unrolled_parameters:,} at mean recurrence."
    )


# --- the run logger --------------------------------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """What `train()` returns: the counts, times, last losses and files of one run.

    Defined here for now; task 8 of `tasks/training_pipeline_restructure.md` may move it to `training/run.py` next
    to `train()`.
    """

    run_directory: Path
    steps_completed: int  # optimizer steps run by this process (a resumed run counts from its resume step)
    final_step: int  # completed optimizer steps of the run in total (`progress.done` at the end)
    resumed_from: Path | None  # the checkpoint the run resumed from, None for a fresh start
    setup_seconds: float  # from the start of the run to `RunLogger.open` (backend, dataset, loaders, model, resume)
    train_seconds: float  # from `RunLogger.open` to `RunLogger.close`
    last_loss: float | None  # training loss of the last logged step, None if no step was logged
    last_validation: dict[str, float]  # `val_loss*`, `val_ppl*`, `val_time` of the last evaluation, {} if none ran
    checkpoints_written: list[Path]  # every checkpoint saved by this process, in order
    export_dir: Path | None  # the HuggingFace export folder, None without `export_to_hf` (and after a stop)
    stopped: bool = False  # the run stopped on request (`should_stop` of `train()`, the CLI's Ctrl-C) before its last step
    history: dict[int, dict[str, float]] = field(default_factory=dict)  # `RunLogger.log_step`'s metrics per logged step

    def summary(self) -> str:
        """The lines the CLI prints after `train()` returned."""
        origin = f"resumed from {self.resumed_from}" if self.resumed_from is not None else "fresh start"
        lines = [
            f"Training run in {self.run_directory}: {self.steps_completed} optimizer steps completed "
            f"(final step {self.final_step}, {origin})",
            f"  setup {self.setup_seconds:.1f}s, training {self.train_seconds:.1f}s",
        ]
        if self.stopped:
            lines.append(f"  stopped on request after step {self.final_step}; rerun with resume: true to continue")
        loss = f"last loss {self.last_loss:.4f}" if self.last_loss is not None else "no step logged"
        if self.last_validation:
            losses = ", ".join(f"{k} {v:.4f}" for k, v in self.last_validation.items() if k.startswith("val_loss"))
            lines.append(f"  {loss} | last validation: {losses}")
        else:
            lines.append(f"  {loss} | no validation")
        if self.checkpoints_written:
            lines.append(f"  {len(self.checkpoints_written)} checkpoints written, last: {self.checkpoints_written[-1]}")
        else:
            lines.append("  no checkpoint written")
        lines.append(f"  HuggingFace export: {self.export_dir}" if self.export_dir else "  no HuggingFace export")
        return "\n".join(lines)


class RunLogger:
    """Console lines, wandb metrics, timers, the data-composition counter and the metric history of one run.

    Create it with `open()` once the setup (resume included) is done; use it as a context manager so a failing loop
    still releases what it holds (task 10 enters the terminal dashboard on `resources`, the `ExitStack` kept here);
    `close()` returns the `TrainingReport`. Nothing here touches the numerics: tensors become floats only at log
    steps and for validation metrics, exactly where the thesis loop called `.item()`.
    """

    def __init__(
        self,
        settings: Settings,
        run_directory: Path,
        stage_manager: StageManager,
        wandb: Logger,
        *,
        start_step: int,
        device: str,
        clock: Callable[[], float] = time.time,
        setup_started: float | None = None,
    ) -> None:
        self.settings = settings
        self.run_directory = run_directory
        self.stage_manager = stage_manager
        self.wandb = wandb
        self.start_step = start_step  # `progress.step` when the logger opened (the resume step, 0 for a fresh run)
        self.device = device
        self.resources = ExitStack()  # closed by `close()` / `__exit__`; task 10 enters the dashboard on it
        self.history: dict[int, dict[str, float]] = {}  # per logged step: the metric dict as floats
        self.checkpoints_written: list[Path] = []
        self.resumed_from: Path | None = None
        self.tokens_per_step = settings.world_batch_size * settings.block_size
        self._clock = clock
        now = clock()
        self.setup_seconds = now - setup_started if setup_started is not None else 0.0
        self._train_started = now  # the train timer: `total_time` of the metrics, `train_time` of the wandb summary
        self._interval_started = now  # the log-interval timer behind `seconds/step`; reset at every log step
        self._sample_counter: Counter[str] = Counter()  # data ids of the world batches since the last log step
        self._evaluation_seconds: float | None = None  # duration of the last `evaluating()` block, read by `log_step`
        self._last_loss: float | None = None
        self._last_validation: dict[str, float] = {}

    @classmethod
    def open(
        cls,
        settings: Settings,
        run_directory: Path,
        dataset: ResolvedDataset,
        model: Module,
        stage_manager: StageManager,
        progress: TrainingProgress,
        backend: Backend,
        *,
        clock: Callable[[], float] = time.time,
        setup_started: float | None = None,
    ) -> RunLogger:
        """Open the run's logging once the setup is done: the wandb run with the hyperparameters (the settings plus
        `dataset_config_hash`) and the `num_parameters` summary; on the console the stage summary, the total-steps
        line, the parameter line and the setup line. The setup timer ends and the train timer starts here.

        `progress.step` is the step training starts at (the resume step), `backend.device` names the device;
        `setup_started` is the clock reading at the start of the run (`setup_seconds` of the report; 0 if not given).
        `clock` is `time.time` unless a test injects a fake.
        """
        wandb = Logger(
            settings.logger_project,
            settings.run_name,
            run_directory,
            offline=settings.wandb_offline,
            enabled=settings.wandb_enabled,
        )
        wandb.log_hyperparams(asdict(settings) | {"dataset_config_hash": dataset.config_hash})
        wandb.log_summary({"num_parameters": num_parameters(unwrap_compiled(model))})
        run_logger = cls(
            settings,
            run_directory,
            stage_manager,
            wandb,
            start_step=progress.step,
            device=str(backend.device),
            clock=clock,
            setup_started=setup_started,
        )
        console.info(stage_manager.get_stage_summary(), extra=KEEP)
        console.info(
            f"Total training steps: {stage_manager.total_steps:,} ({settings.gradient_accumulation_steps} micro-batches each)",
            extra=KEEP,
        )
        console.info(describe_parameters(model), extra=KEEP)
        console.info(
            f"Setup took {run_logger.setup_seconds:.1f}s, starting training at step {progress.step} "
            f"on {run_logger.device} ({settings.precision}).",
            extra=KEEP,
        )
        return run_logger

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """Release what the logger holds; idempotent (`close()` normally ran before)."""
        self.wandb.finish()
        self.resources.close()

    def log_resume(self, path: Path, step: int) -> None:
        """The run continues from checkpoint `path` at optimizer step `step` (the report's `resumed_from`)."""
        self.resumed_from = path
        console.info(f"Resumed from {path} at step {step}", extra=KEEP)

    def log_fresh_start(self) -> None:
        """No checkpoint was loaded; the run starts at step 0."""
        console.info("No checkpoint loaded, starting from scratch.", extra=KEEP)

    @contextmanager
    def evaluating(self) -> Iterator[None]:
        """Time one `evaluate` call; `log_step` reports the duration as `val_time` (seconds) next to the validation
        metrics of that step, as the thesis loop did. Task 10 sets the dashboard status here too."""
        started = self._clock()
        try:
            yield
        finally:
            self._evaluation_seconds = self._clock() - started

    def log_step(self, result: StepResult, progress: TrainingProgress) -> None:
        """Account one completed optimizer step (`progress.done`, i.e. after `progress.advance()`).

        Every step: the data ids join the composition counter, a stage transition starting or ending with this step
        gets its console line, and a set `result.validation` gets the validation line (`Step N: val loss ...`) and
        becomes the report's `last_validation`. At log steps (`done % log_step_interval == 0`) the metric dict goes
        to wandb, to `history[done]` (as floats) and to the one-line console summary:

        * `loss` (mean micro-batch loss), `ppl` (exp of the mean log-perplexity), `lr` (scheduled LR before the
          per-group `base_lr`), `grad_norm` (pre-clip), `step` (= done);
        * `seconds/step` (wall time of the last log interval per step), `tokens/second` (`world_batch_size ×
          block_size` per `seconds/step`; 0 if the interval took no measurable time), `total_tokens` (`done × tokens
          per step`, counted from step 0 also after a resume), `total_time` (seconds since `open`), `remaining_time`
          (`seconds/step × steps left`);
        * `stage/current_stage`, `stage/base_lr`, `stage/in_transition` (0/1), `stage/transition_progress`,
          `stage/stage_progress` — the stage info the step trained on (`result.stage`);
        * `data_composition/<data id>`: the fraction of world-batch samples since the last log step per data id (they
          sum to 1; the counter resets here);
        * the gradient / parameter metrics of `track_gradient_metrics` (`result.metrics`) and the validation metrics
          (`val_loss*`, `val_ppl*`, `val_time`) when this step evaluated.
        """
        self._sample_counter.update(result.data_ids)
        self._log_transition(result, progress)
        validation = self._log_validation(result, progress)
        if progress.done % self.settings.log_step_interval != 0:
            return
        metrics = self._step_metrics(result, progress, validation)
        self.wandb.log(metrics, step=progress.done)
        self.history[progress.done] = {name: float(value) for name, value in metrics.items()}
        self._last_loss = float(metrics["loss"])
        console.info(
            f"step {progress.done}/{self.stage_manager.total_steps} | loss {metrics['loss']:.4f} | "
            f"lr {metrics['lr']:.2e} | grad_norm {metrics['grad_norm']:.3f} | {metrics['seconds/step']:.2f}s/step"
        )

    def _log_transition(self, result: StepResult, progress: TrainingProgress) -> None:
        """The two transition lines: after the last plain step of a stage ("starting transition") and after the last
        transition step ("transition complete"). `result.stage` is the info at the step, `result.next_stage` at the
        step after; inside a transition the info already names the next stage."""
        before, after = result.stage, result.next_stage
        if after.in_transition and not before.in_transition:
            console.info(
                f"Step {progress.done}: starting transition {after.prev_stage_idx} -> {after.stage_idx} "
                f"({after.stage_name}), LR {cast(float, after.prev_base_lr):.2e} -> {after.base_lr:.2e}"
            )
        elif before.in_transition and not after.in_transition:
            console.info(
                f"Step {progress.done}: transition complete, now in stage {after.stage_idx} ({after.stage_name})"
            )

    def _log_validation(self, result: StepResult, progress: TrainingProgress) -> dict[str, float] | None:
        """The validation metrics of this step as floats plus `val_time`, and their console line; None if the step
        did not evaluate."""
        if result.validation is None:
            return None
        validation = {name: float(_to_scalar(value)) for name, value in result.validation.items()}
        validation["val_time"] = self._evaluation_seconds or 0.0
        self._evaluation_seconds = None
        self._last_validation = validation
        console.info(
            f"Step {progress.done}: val loss {validation['val_loss']:.4f} "
            f"(stage {result.next_stage.stage_idx}, {validation['val_time']:.1f}s)"
        )
        return validation

    def _step_metrics(
        self, result: StepResult, progress: TrainingProgress, validation: dict[str, float] | None
    ) -> dict[str, Any]:
        """The metric dict of a log step (documented in `log_step`); resets the interval timer and the composition
        counter."""
        now = self._clock()
        seconds_per_step = (now - self._interval_started) / self.settings.log_step_interval
        self._interval_started = now
        total_samples = sum(self._sample_counter.values())
        metrics: dict[str, Any] = {name: _to_scalar(value) for name, value in result.metrics.items()}
        metrics |= validation or {}
        metrics |= {
            "loss": _to_scalar(result.loss),
            "ppl": _to_scalar(result.log_ppl.exp()),
            "lr": result.learning_rate,
            "grad_norm": _to_scalar(result.grad_norm),
            "step": progress.done,
            "seconds/step": seconds_per_step,
            "tokens/second": self.tokens_per_step / seconds_per_step if seconds_per_step > 0 else 0.0,
            "total_tokens": progress.done * self.tokens_per_step,
            "total_time": now - self._train_started,
            "remaining_time": seconds_per_step * (self.stage_manager.total_steps - progress.done),
            "stage/current_stage": result.stage.stage_idx,
            "stage/base_lr": result.stage.base_lr,
            "stage/in_transition": int(result.stage.in_transition),
            "stage/transition_progress": result.stage.transition_progress,
            "stage/stage_progress": result.stage.stage_progress,
        }
        metrics |= {f"data_composition/{name}": count / total_samples for name, count in self._sample_counter.items()}
        self._sample_counter.clear()
        return metrics

    def log_checkpoint(self, path: Path) -> None:
        """A checkpoint was written to `path` (the report's `checkpoints_written`)."""
        self.checkpoints_written.append(path)
        console.info(f"Saved checkpoint {path}", extra=KEEP)

    def log_export(self, path: Path) -> None:
        """The HuggingFace export was written to `path`."""
        console.info(f"Exported HuggingFace model to {path}", extra=KEEP)

    def close(self, progress: TrainingProgress, export_dir: Path | None, *, stopped: bool = False) -> TrainingReport:
        """End the run's logging: `train_time` into the wandb summary, `finish()`, the final console line, the
        resources released; returns the report. `stopped` says the run ended on request before its last step.
        `__exit__` afterwards is a no-op (both are idempotent)."""
        train_seconds = self._clock() - self._train_started
        self.wandb.log_summary({"train_time": train_seconds})
        self.wandb.finish()
        ending = "stopped on request" if stopped else "finished"
        console.info(f"Training {ending} after {progress.done} steps in {train_seconds:.1f}s.", extra=KEEP)
        self.resources.close()
        return TrainingReport(
            run_directory=self.run_directory,
            steps_completed=progress.done - self.start_step,
            final_step=progress.done,
            resumed_from=self.resumed_from,
            setup_seconds=self.setup_seconds,
            train_seconds=train_seconds,
            last_loss=self._last_loss,
            last_validation=dict(self._last_validation),
            checkpoints_written=list(self.checkpoints_written),
            export_dir=export_dir,
            stopped=stopped,
            history=self.history,
        )


# --- gradient / parameter metrics -----------------------------------------------------------------------------------


def _reverse_engineer_adam_effective_lr(
    param: torch.Tensor, param_state: dict[str, torch.Tensor], group: dict[str, Any]
) -> torch.Tensor:
    """Recompute Adam's per-element effective LR (ignoring bias correction and the scheduled LR)."""
    grad = param.grad
    assert grad is not None, "effective LR needs a gradient"
    exp_avg = param_state["exp_avg"].float()
    denom = param_state["exp_avg_sq"].float().sqrt().add_(group["eps"])
    return torch.where(
        grad.float().abs() > group["eps"],
        exp_avg / denom / grad.float(),
        exp_avg / denom / group["eps"],
    )


def _qkv_dims(model: Module) -> Optional[tuple[int, int, int]]:
    """(n_embd, query width, key/value width) for slicing fused qkv gradients; None for non-transformer models."""
    config = getattr(model, "config", None)
    if config is None or not all(hasattr(config, a) for a in ("n_embd", "head_size", "num_attention_heads")):
        return None
    return config.n_embd, config.n_embd, config.head_size * config.num_attention_heads  # no GQA: kv width == q width


@torch.no_grad()
def track_gradient_metrics(model: Module, optimizer: Optimizer) -> dict[str, torch.Tensor]:
    """Gradient norms, Adam second-moment RMS, effective LRs and parameter norms. Call after `optimizer.step()`
    and before `zero_grad()`."""
    metrics: dict[str, torch.Tensor] = {}
    dims = _qkv_dims(model)
    transformer = getattr(model, "transformer", None)
    wte_module: Optional[Module] = getattr(transformer, "wte", None)
    wte_weight: Optional[torch.Tensor] = getattr(wte_module, "weight", None)

    # Specific gradient norms
    qkv_layer_counter, mlp_layer_counter = 0, 0
    qkv_params: list[torch.Tensor] = []
    proj_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "qkv" in name and "weight" in name:
                qkv_params.append(param)
                if (~torch.isfinite(param.grad)).sum() == 0:
                    if dims is not None and param.grad.numel() % dims[0] == 0:
                        q_grad = param.grad.view(-1, dims[0])[: dims[1], :]
                        metrics[f"query_grad_{qkv_layer_counter}"] = q_grad.norm()
                else:
                    metrics[f"query_grad_{qkv_layer_counter}"] = torch.as_tensor(float("NaN"))
                qkv_layer_counter += 1
            if "mlp" in name and "proj" in name and "weight" in name:
                proj_params.append(param)
                if (~torch.isfinite(param.grad)).sum() == 0:
                    metrics[f"ffn2_grad_{mlp_layer_counter}"] = param.grad.norm()
                else:
                    metrics[f"ffn2_grad_{mlp_layer_counter}"] = torch.as_tensor(float("NaN"))
                mlp_layer_counter += 1

    # 2nd moment quality and effective learning rates
    total_rms: torch.Tensor | float = 0.0
    num_params_with_grad = 0
    qkv_layer_counter, mlp_layer_counter = 0, 0
    params_with_finite_grad = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and (~torch.isfinite(param.grad)).sum() == 0:
                params_with_finite_grad.append(param)
                if param in optimizer.state and "exp_avg_sq" in optimizer.state[param]:
                    exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                    if exp_avg_sq.shape == param.grad.shape:
                        rms = (
                            param.grad.float().pow(2).div_(exp_avg_sq.float().clamp_(min=group["eps"] ** 2)).mean().sqrt()
                        )
                        total_rms += rms
                        num_params_with_grad += 1
                        if wte_weight is not None and param is wte_weight:
                            metrics["embed_RMS"] = rms

                        if any(param is p for p in qkv_params):  # identity check, `in` would compare values
                            qkv_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            if dims is not None and qkv_lr.numel() % dims[0] == 0:
                                H, dim_q, dim_kv = dims
                                qkv_lr = qkv_lr.view(-1, H)
                                metrics[f"q_effective_lr_{qkv_layer_counter}"] = qkv_lr[:dim_q, :].mean()
                                metrics[f"k_effective_lr_{qkv_layer_counter}"] = qkv_lr[dim_q : dim_q + dim_kv, :].mean()
                                metrics[f"v_effective_lr_{qkv_layer_counter}"] = qkv_lr[dim_q + dim_kv :, :].mean()
                            qkv_layer_counter += 1

                        if any(param is p for p in proj_params):
                            proj_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            metrics[f"ffn2_effective_lr_{mlp_layer_counter}"] = proj_lr.mean()
                            mlp_layer_counter += 1

    if num_params_with_grad > 0:
        metrics["avg_RMS"] = torch.as_tensor(total_rms / num_params_with_grad)  # already a Tensor after one add

    if len(params_with_finite_grad) > 0:
        metrics["local_l1_grad_norm"] = torch.mean(
            torch.stack([torch.norm(p.grad.detach(), 1.0) for p in params_with_finite_grad])
        )

    # Parameter norms
    metrics["l2_param_norm"] = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
    metrics["l1_param_norm"] = torch.mean(torch.stack([torch.norm(p.detach(), 1.0) for p in model.parameters()]))

    core_blocks = getattr(transformer, "core_blocks", None)
    if core_blocks is not None:
        for block_idx, core_block in enumerate(core_blocks):
            metrics[f"core_block_{block_idx}_l2_param_norm"] = torch.norm(
                torch.stack([torch.norm(p.detach()) for p in core_block.parameters()])
            )
    if wte_module is not None and wte_weight is not None:
        metrics["word_embed_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(p.detach()) for p in wte_module.parameters()])
        )
        metrics["model_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(p.detach()) for n, p in model.named_parameters() if "wte" not in n])
        )
    return metrics
