# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Logging of a training run: the thin wandb wrapper (`Logger`, offline by default), the gradient / parameter metric
helpers that were logged, and `RunLogger` — every console line, timer and counter of a run in one place, driving the
terminal dashboard and ending in a `TrainingReport`.

`RunLogger` never prints. What a run shows on the terminal goes through two channels, both owned by the dashboard of
`training.ui` for the duration of the run (`RunLogger.open` enters it; the live `TrainingDashboard` on a TTY, the
`NoOpDashboard` console fallback otherwise — one log line per `log_step_interval` steps — and `train.log` under the
run directory in both cases):

* *records* on the `training.logger` logger (the `training` hierarchy the CLI attaches a stream handler to and the
  dashboard takes over for the run): the header lines of `open` and the final line of `close`, marked
  `extra={"keep": True}` so they survive in the scrollback under the live display;
* *dashboard calls*: the bars move at every step (`update_step`; the metric dict only at log steps — never a tensor
  at the other steps, so no device sync is added), validation losses (`update_validation`), the events (checkpoints,
  resume point, transitions, export: `note_event`) and the header status (`set_status`).
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import plain_model
from training.settings import Settings
from training.stage_manager import StageInfo, StageManager
from training.ui.capture import WANDB_QUIET_SETTINGS
from training.ui.common import KEEP, TRAIN_LOG_NAME
from training.ui.dashboard import RunDashboard, training_dashboard
from training.ui.format import TRANSITION_FLAG_KEY, TRANSITION_PROGRESS_KEY

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from training.backend.base import Backend
    from training.data.dataset_resolver import ResolvedDataset
    from training.step import StepResult, TrainingProgress  # `step.py` imports `track_gradient_metrics` from here

CONSOLE_LOGGER_NAME = "training.logger"  # under the `training` hierarchy; named explicitly, not via `__name__`

console = logging.getLogger(CONSOLE_LOGGER_NAME)


class Logger:
    """wandb run wrapper; every method is a no-op when `enabled=False` (wandb is then never imported).

    The run is created quiet (`wandb.Settings(**WANDB_QUIET_SETTINGS)`: `console="off"`, `silent=True`): wandb's
    default `console="wrap"` would replace `sys.stdout` / `sys.stderr` with its own proxies and print its banner
    lines to stderr, both of which fight the terminal dashboard's stream capture. The metrics are unaffected.
    """

    def __init__(
        self, project: str, run_name: str, out_dir: str | Path, offline: bool = True, enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self.run: Optional[Run] = None
        if not enabled:
            return
        import wandb

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # the constant is a `dict[str, object]` shared with the dashboard's environment variables; `wandb.Settings`
        # types every field, so the two values must be passed as `Any` for the checkers
        quiet = wandb.Settings(**cast(dict[str, Any], WANDB_QUIET_SETTINGS))
        self.run = wandb.init(
            project=project, name=run_name, dir=str(out_dir), mode="offline" if offline else "online", settings=quiet
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
    unwrapped = plain_model(model)
    total_parameters = num_parameters(unwrapped)
    core_blocks = cast(Iterable[Module], unwrapped.transformer.core_blocks)
    recurrent_parameters = sum(p.numel() for block in core_blocks for p in block.parameters())
    mean_recurrence = cast(list[int], unwrapped.config.mean_recurrence)  # a list after RecurrentConfig.__post_init__
    mean_of_means = sum(mean_recurrence) / len(mean_recurrence)
    unrolled_parameters = int(total_parameters - recurrent_parameters + recurrent_parameters * mean_of_means)
    return (
        f"Model: {total_parameters:,} parameters, {recurrent_parameters:,} in recurrent blocks, unfolds to "
        f"{unrolled_parameters:,} at mean recurrence."
    )


# --- the run logger --------------------------------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """What `train()` returns: the counts, times, last losses and files of one run (built by `RunLogger.close`,
    re-exported by `training/run.py`)."""

    run_directory: Path
    steps_completed: int  # optimizer steps run by this process (a resumed run counts from its resume step)
    final_step: int  # completed optimizer steps of the run in total (`progress.step` at the end)
    resumed_from: Path | None  # the checkpoint the run resumed from, None for a fresh start
    setup_seconds: float  # from the start of the run to `RunLogger.open` (backend, dataset, loaders, model, resume)
    train_seconds: float  # from `RunLogger.open` to `RunLogger.close`
    last_loss: float | None  # training loss of the last logged step, None if no step was logged
    last_validation: dict[str, float]  # `val_loss*`, `val_ppl*`, `val_time` of the last evaluation, {} if none ran
    checkpoints_written: list[Path]  # every checkpoint saved by this process, in order
    export_dir: Path | None  # the HuggingFace export folder, None without `export_to_hf` (and after a stop)
    stopped: bool = False  # the run stopped on request (`should_stop` of `train()`, the CLI's Ctrl-C) before its last step
    history: dict[int, dict[str, float]] = field(default_factory=dict)  # per logged step: `RunLogger.log_step`'s
    # metrics, only with `train(keep_history=True)` (a test knob); empty otherwise

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


class Dashboard(Protocol):
    """The four calls `RunLogger` makes on the run's terminal dashboard. `training.ui`'s `TrainingDashboard` (the
    live display) and `NoOpDashboard` (the console fallback) satisfy it; tests pass a recording fake."""

    def update_step(self, step: int, stage_index: int, metrics: Mapping[str, object]) -> None: ...

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None: ...

    def note_event(self, text: str) -> None: ...

    def set_status(self, text: str) -> None: ...


def open_dashboard(
    settings: Settings, run_directory: Path, stage_manager: StageManager, *, start_step: int, device: str
) -> AbstractContextManager[RunDashboard]:
    """The run's dashboard (`training.ui.training_dashboard`): the live display when stdout is a terminal and
    `TRAINING_DASHBOARD` is not `0`, the one-line-per-`log_step_interval` console fallback otherwise; one bar per
    stage (named after `stage_manager.stages`, sized by its boundary) plus the overall bar, the header naming the
    run, the model and dataset config (file names without `.yaml`), the device and precision; every record appended
    to `run_directory / train.log`. `start_step` (the resume step) keeps the ETA honest after a resume."""
    return training_dashboard(
        settings.run_name,
        [stage.name for stage in stage_manager.stages],
        [boundary.end_step - boundary.start_step for boundary in stage_manager.boundaries],
        stage_manager.total_steps,
        details={
            "model": Path(settings.model_architecture_config).stem,
            "dataset": Path(settings.dataset_config).stem,
            "device": device,
            "precision": settings.precision,
        },
        start_step=start_step,
        log_step_interval=settings.log_step_interval,
        log_file=run_directory / TRAIN_LOG_NAME,
        fallback_stream=sys.stderr,  # piped runs: step lines join the log handlers' lines on stderr
    )


class RunLogger:
    """Console records, the terminal dashboard, wandb metrics, timers, the data-composition counter and the metric
    history of one run.

    Create it with `open()` once the setup (resume included) is done; use it as a context manager so a failing loop
    still releases what it holds (the dashboard is entered on `resources`, the `ExitStack` kept here, and torn down
    by `__exit__` on a normal end, an exception and a Ctrl-C alike); `close()` returns the `TrainingReport`.
    Nothing here touches the numerics: tensors become floats only at log steps and for validation metrics, exactly
    where the thesis loop called `.item()`; the dashboard gets an empty step dict at every other step.
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
        dashboard: Dashboard | None = None,
        clock: Callable[[], float] = time.time,
        setup_started: float | None = None,
        keep_history: bool = False,
    ) -> None:
        self.settings = settings
        self.run_directory = run_directory
        self.stage_manager = stage_manager
        self.wandb = wandb
        self.start_step = start_step  # `progress.step` when the logger opened (the resume step, 0 for a fresh run)
        self.device = device
        self.resources = ExitStack()  # closed by `close()` / `__exit__`; the dashboard is entered on it
        # a given dashboard (tests: a `TrainingDashboard` on a StringIO console, a recording fake) is driven as it is
        # and not closed here; otherwise `open_dashboard` picks the live display or the fallback for the run
        self.dashboard: Dashboard = (
            dashboard
            if dashboard is not None
            else self.resources.enter_context(
                open_dashboard(settings, run_directory, stage_manager, start_step=start_step, device=device)
            )
        )
        self.keep_history = keep_history  # a test knob: fill `history` (the CLI does not keep every log step)
        self.history: dict[int, dict[str, float]] = {}  # per logged step: the metric dict as floats, if kept
        self.checkpoints_written: list[Path] = []
        self.resumed_from: Path | None = None
        self.tokens_per_step = settings.world_batch_size * settings.block_size
        self._clock = clock
        now = clock()
        self.setup_seconds = now - setup_started if setup_started is not None else 0.0
        self._train_started = now  # the train timer: `total_time` of the metrics, `train_time` of the wandb summary
        self._interval_started = now  # the log-interval timer behind `seconds/step`; reset at every log step
        self._interval_step = start_step  # the step the interval timer started at (the resume step, then each log step)
        self._sample_counter: Counter[str] = Counter()  # data ids of the world batches since the last log step
        self._evaluation_seconds: float | None = None  # duration of the last `evaluating()` block, read by `log_step`
        self._status = "starting"  # the dashboard's header status; `_status_during` restores it after a block
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
        dashboard: Dashboard | None = None,
        clock: Callable[[], float] = time.time,
        setup_started: float | None = None,
        keep_history: bool = False,
    ) -> RunLogger:
        """Open the run's logging once the setup is done: the wandb run with the hyperparameters (the settings plus
        `dataset_config_hash`) and the `num_parameters` summary, then the dashboard (`open_dashboard`, unless a
        `dashboard` is given), then on the console the stage summary, the total-steps line, the parameter line and
        the setup line. The setup timer ends and the train timer starts here.

        `progress.step` is the step training starts at (the resume step), `backend.device` names the device;
        `setup_started` is the clock reading at the start of the run (`setup_seconds` of the report; 0 if not given).
        `clock` is `time.time` unless a test injects a fake; `keep_history` fills `history` (a test knob too).
        """
        wandb = Logger(
            settings.logger_project,
            settings.run_name,
            run_directory,
            offline=settings.wandb_offline,
            enabled=settings.wandb_enabled,
        )
        wandb.log_hyperparams(asdict(settings) | {"dataset_config_hash": dataset.config_hash})
        wandb.log_summary({"num_parameters": num_parameters(plain_model(model))})
        run_logger = cls(
            settings,
            run_directory,
            stage_manager,
            wandb,
            start_step=progress.step,
            device=str(backend.device),
            dashboard=dashboard,
            clock=clock,
            setup_started=setup_started,
            keep_history=keep_history,
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
        """Release what the logger holds — the dashboard included, so the terminal is restored on an exception and
        on a Ctrl-C too; idempotent (`close()` normally ran before).

        Every resource is released even when an earlier release raises: a failing `wandb.finish()` used to leave the
        terminal with the dashboard's redirected streams and a hidden cursor. The first failure is the one raised
        (the later ones would only mask it; the exception that ended the run, if any, stays its `__context__`).
        """
        failures: list[BaseException] = []
        for release in (self.wandb.finish, self.resources.close):
            try:
                release()
            except BaseException as failure:  # released in order, re-raised below: nothing is swallowed
                failures.append(failure)
        if failures:
            raise failures[0]

    # --- status and events -----------------------------------------------------------------------------------------

    def status(self, text: str) -> None:
        """The dashboard's header status: what the run is doing right now (`training`, `stopping after this step,
        saving a checkpoint`, `exporting`; `finished` / `stopped on request` set by `close`)."""
        self._status = text
        self.dashboard.set_status(text)

    @contextmanager
    def _status_during(self, text: str) -> Iterator[None]:
        """Show `text` as the status for the block, then the status from before it."""
        previous = self._status
        self.status(text)
        try:
            yield
        finally:
            self.status(previous)

    @contextmanager
    def evaluating(self) -> Iterator[None]:
        """Around one `evaluate` call: the status reads `evaluating`, and the duration becomes `val_time` (seconds)
        next to the validation metrics of that step in `log_step`, as the thesis loop reported it."""
        started = self._clock()
        with self._status_during("evaluating"):
            try:
                yield
            finally:
                self._evaluation_seconds = self._clock() - started

    def saving_checkpoint(self) -> AbstractContextManager[None]:
        """Around one checkpoint write: the status reads `saving checkpoint`."""
        return self._status_during("saving checkpoint")

    def log_resume(self, path: Path, step: int) -> None:
        """The run continues from checkpoint `path` at optimizer step `step` (the report's `resumed_from`)."""
        self.resumed_from = path
        self.dashboard.note_event(f"resumed from {path} at step {step}")

    def log_fresh_start(self) -> None:
        """No checkpoint was loaded; the run starts at step 0."""
        self.dashboard.note_event("no checkpoint found, starting from scratch")

    def log_checkpoint(self, path: Path) -> None:
        """A checkpoint was written to `path` (the report's `checkpoints_written`)."""
        self.checkpoints_written.append(path)
        self.dashboard.note_event(f"saved checkpoint {path}")

    def log_export(self, path: Path) -> None:
        """The HuggingFace export was written to `path`."""
        self.dashboard.note_event(f"exported HuggingFace model to {path}")

    # --- steps -------------------------------------------------------------------------------------------------------

    def log_step(self, result: StepResult, progress: TrainingProgress) -> None:
        """Account one completed optimizer step (`progress.step`, after `progress.advance()`).

        Every step: the data ids join the composition counter, a stage transition starting or ending with this step
        becomes a dashboard event, a set `result.validation` becomes the dashboard's validation row and the report's
        `last_validation`, and the dashboard's bars move (`update_step` with the stage containing `done` — the bar
        whose steps are counting — plus the transition keys of `done`, and — only at log steps — the metric dict; at
        every other step an empty dict: no tensor is read there, so no device sync is added to the thesis loop). At
        log steps (`done % log_step_interval == 0`) the metric dict goes to wandb and, with `keep_history`, to
        `history[done]` (as floats); the fallback dashboard turns it into its one console line:

        * `loss` (mean micro-batch loss), `ppl` (exp of the mean log-perplexity), `lr` (scheduled LR), `grad_norm`
          (pre-clip), `step` (= done);
        * `seconds/step` (wall time of the last log interval per step), `tokens/second` (`world_batch_size ×
          block_size` per `seconds/step`; 0 if the interval took no measurable time), `total_tokens` (`done × tokens
          per step`, counted from step 0 also after a resume), `total_time` (seconds since `open`), `remaining_time`
          (`seconds/step × steps left`);
        * `stage/current_stage`, `stage/base_lr`, `stage/in_transition` (0/1), `stage/transition_progress`,
          `stage/stage_progress` — the stage info the step trained on (`result.stage`): `current_stage` is the stage
          whose boundary contains the step, the same stage `stage_progress` and `base_lr` describe, also inside the
          transition window at its end (the stage being entered is only visible through `in_transition`);
        * `data_composition/<data id>`: the fraction of world-batch samples since the last log step per data id (they
          sum to 1; the counter resets here);
        * the gradient / parameter metrics of `track_gradient_metrics` (`result.metrics`) and the validation metrics
          (`val_loss*`, `val_ppl*`, `val_time`) when this step evaluated.
        """
        self._sample_counter.update(result.data_ids)
        at_done = self.stage_manager.get_stage_info(progress.step)
        self._note_transition(result.stage, at_done)
        validation = self._log_validation(result, progress)
        transition = {
            TRANSITION_FLAG_KEY: float(at_done.transition_to is not None),
            TRANSITION_PROGRESS_KEY: at_done.transition_progress,
        }
        if progress.step % self.settings.log_step_interval != 0:
            self.dashboard.update_step(progress.step, at_done.stage_idx, {})
            return
        metrics = self._step_metrics(result, progress, validation)
        self.wandb.log(metrics, step=progress.step)
        if self.keep_history:
            self.history[progress.step] = {name: float(value) for name, value in metrics.items()}
        self._last_loss = float(metrics["loss"])
        self.dashboard.update_step(progress.step, at_done.stage_idx, metrics | transition)

    def _note_transition(self, before: StageInfo, after: StageInfo) -> None:
        """The two transition events: after the last plain step of a stage ("starting transition") and after the
        last transition step ("transition complete"). `before` is the info at the step that trained, `after` the
        one at `done`, the step after it."""
        stages = self.stage_manager.stages
        if after.transition_to is not None and before.transition_to is None:
            leaving, entering = stages[after.stage_idx], stages[after.transition_to]
            self.dashboard.note_event(
                f"starting transition {after.stage_idx} -> {after.transition_to} ({leaving.name} -> {entering.name}), "
                f"LR {leaving.base_lr:.2e} -> {entering.base_lr:.2e}"
            )
        elif before.transition_to is not None and after.transition_to is None:
            self.dashboard.note_event(
                f"transition complete, now in stage {after.stage_idx} ({stages[after.stage_idx].name})"
            )

    def _log_validation(self, result: StepResult, progress: TrainingProgress) -> dict[str, float] | None:
        """The validation metrics of this step as floats plus `val_time`, shown on the dashboard (the `val_loss*`
        entries: one per evaluated depth and the mean-recurrence one); None if the step did not evaluate."""
        if result.validation is None:
            return None
        validation = {name: float(_to_scalar(value)) for name, value in result.validation.items()}
        validation["val_time"] = self._evaluation_seconds or 0.0
        self._evaluation_seconds = None
        self._last_validation = validation
        losses = {name: value for name, value in validation.items() if name.startswith("val_loss")}
        self.dashboard.update_validation(progress.step, losses)
        return validation

    def _step_metrics(
        self, result: StepResult, progress: TrainingProgress, validation: dict[str, float] | None
    ) -> dict[str, Any]:
        """The metric dict of a log step (documented in `log_step`); resets the interval timer and the composition
        counter."""
        now = self._clock()
        steps_in_interval = max(progress.step - self._interval_step, 1)  # after an off-grid resume fewer than the interval
        seconds_per_step = (now - self._interval_started) / steps_in_interval
        self._interval_started, self._interval_step = now, progress.step
        total_samples = sum(self._sample_counter.values())
        metrics: dict[str, Any] = {name: _to_scalar(value) for name, value in result.metrics.items()}
        metrics |= validation or {}
        metrics |= {
            "loss": _to_scalar(result.loss),
            "ppl": _to_scalar(result.log_ppl.exp()),
            "lr": result.learning_rate,
            "grad_norm": _to_scalar(result.grad_norm),
            "step": progress.step,
            "seconds/step": seconds_per_step,
            "tokens/second": self.tokens_per_step / seconds_per_step if seconds_per_step > 0 else 0.0,
            "total_tokens": progress.step * self.tokens_per_step,
            "total_time": now - self._train_started,
            "remaining_time": seconds_per_step * (self.stage_manager.total_steps - progress.step),
            "stage/current_stage": result.stage.stage_idx,
            "stage/base_lr": self.stage_manager.stages[result.stage.stage_idx].base_lr,
            "stage/in_transition": int(result.stage.transition_to is not None),
            "stage/transition_progress": result.stage.transition_progress,
            "stage/stage_progress": result.stage.stage_progress,
        }
        metrics |= {f"data_composition/{name}": count / total_samples for name, count in self._sample_counter.items()}
        self._sample_counter.clear()
        return metrics

    def close(self, progress: TrainingProgress, export_dir: Path | None, *, stopped: bool = False) -> TrainingReport:
        """End the run's logging: `train_time` into the wandb summary, `finish()`, the final console line, the
        final status, the resources released (the dashboard closes: erases its frame, prints the kept lines and its
        static summary); returns the report. `stopped` says the run ended on request before its last step.
        `__exit__` afterwards is a no-op (both are idempotent)."""
        train_seconds = self._clock() - self._train_started
        self.wandb.log_summary({"train_time": train_seconds})
        self.wandb.finish()
        ending = "stopped on request" if stopped else "finished"
        console.info(f"Training {ending} after {progress.step} steps in {train_seconds:.1f}s.", extra=KEEP)
        self.status(ending)
        self.resources.close()
        return TrainingReport(
            run_directory=self.run_directory,
            steps_completed=progress.step - self.start_step,
            final_step=progress.step,
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
