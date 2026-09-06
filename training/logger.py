# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Logging of a training run: the wandb wrapper (`Logger`, offline by default), the gradient / parameter metric
helpers, and `RunLogger`: every console line, timer and counter of a run, driving the terminal dashboard and
ending in a `TrainingReport`.

`RunLogger` never prints. Two channels, both owned by the dashboard (`open_dashboard`) for the run:

* records on the `training.logger` logger, marked `extra={"keep": True}` so they survive under the live display;
* dashboard calls: `update_step` at every step (the metric dict only at log steps, so no device sync is added),
  `update_validation`, `note_event` and `set_status`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Protocol, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer

from evaluation.samples import GeneratedSample
from training.backend.base import plain_model
from training.settings import Settings
from training.stage_manager import StageInfo, StageManager
from training.ui.board import TrainingDashboard
from training.ui.capture import WANDB_QUIET_SETTINGS
from training.ui.common import KEEP, TRAIN_LOG_NAME, TRAIN_REPORT_NAME, dashboard_enabled
from training.ui.fallback import ConsoleFallbackDashboard

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

    from training.backend.base import Backend
    from training.data.dataset_resolver import ResolvedDataset
    from training.step import StepResult, TrainingProgress  # `step.py` imports `track_gradient_metrics` from here

CONSOLE_LOGGER_NAME = "training.logger"  # under the `training` hierarchy; named explicitly, not via `__name__`

console = logging.getLogger(CONSOLE_LOGGER_NAME)


class Logger:
    """
    wandb run wrapper; every method is a no-op when `enabled=False` (wandb is then never imported).

    Every process is its own wandb run, grouped under `run_name`; a resumed process is named
    `<run_name>-from-<resume_step>` and tagged `resumed`, with `resume_step` in its config. (Continuing one wandb
    run would drop the re-run steps below its last logged step.) The run is created quiet (`WANDB_QUIET_SETTINGS`):
    wandb's default console wrapping and banner lines would fight the dashboard's stream capture.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        out_dir: str | Path,
        offline: bool = True,
        enabled: bool = True,
        resume_step: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.run: Optional[Run] = None
        if not enabled:
            return
        import wandb

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # the shared constant is a `dict[str, object]`; `wandb.Settings` types every field
        quiet = wandb.Settings(**cast(dict[str, Any], WANDB_QUIET_SETTINGS))
        resumed = resume_step is not None
        self.run = wandb.init(
            project=project,
            name=f"{run_name}-from-{resume_step}" if resumed else run_name,
            group=run_name,
            tags=["resumed"] if resumed else [],
            config={"resume_step": resume_step},
            dir=str(out_dir),
            mode="offline" if offline else "online",
            settings=quiet,
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        self.run.log({key: _to_scalar(value) for key, value in metrics.items()}, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.config.update(params, allow_val_change=True)  # type: ignore[no-untyped-call]  # wandb Config.update is unannotated

    def log_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in values.items():
            self.run.summary[key] = _to_scalar(value)

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
    """
    Total number of parameters (tied weights counted once, as `parameters()` deduplicates them).
    """

    parameters = list(model.parameters())
    if only_trainable:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
    return sum(parameter.numel() for parameter in parameters)


def describe_parameters(model: Module) -> str:
    """
    The parameter-count line printed at the start of a run: total parameters, parameters inside the recurrent core
    blocks and the count of the unrolled model at the mean recurrence (`total - recurrent + recurrent * mean of
    mean_recurrence`). Accepts the compiled wrapper too (it is unwrapped).
    """

    unwrapped = plain_model(model)
    total_parameters = num_parameters(unwrapped)
    core_blocks = cast(Iterable[Module], unwrapped.transformer.core_blocks)
    recurrent_parameters = sum(parameter.numel() for block in core_blocks for parameter in block.parameters())
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
    """
    What `train()` returns: the counts, times, last losses and files of one run (built by `RunLogger.close`,
    re-exported by `training/run.py`); `close` also writes it as `train_report.json` into the run directory.
    """

    run_directory: Path
    steps_this_process: int  # optimizer steps run by this process (a resumed run counts from its resume step)
    completed_steps: int  # completed optimizer steps of the run in total (`progress.step` at the end)
    resumed_from: Path | None  # the checkpoint the run resumed from, None for a fresh start
    setup_seconds: float  # from the start of the run to `RunLogger.open` (backend, dataset, loaders, model, resume)
    train_seconds: float  # from `RunLogger.open` to `RunLogger.close`
    last_loss: float | None  # training loss of the last logged step, None if no step was logged
    last_validation: dict[str, float]  # `val_loss*`, `val_ppl*`, `val_time` of the last evaluation, {} if none ran
    checkpoints_written: list[Path]  # every checkpoint saved by this process, in order
    export_dir: Path | None  # the HuggingFace export folder, None without `export_to_hf` (and after a stop)
    stopped: bool = False  # the run stopped on request (the CLI's Ctrl-C) before its last step
    history: dict[int, dict[str, float]] = field(default_factory=dict)  # per logged step, only with `keep_history`
    samples_written: list[Path] = field(default_factory=list)  # every samples file written by this process, in order
    last_benchmarks: dict[str, float] = field(default_factory=dict)  # `benchmark/<task>/<metric>` of the last run, {} if none

    def to_dict(self) -> dict[str, Any]:
        """
        The report as JSON-ready data: paths as strings, `history` left out (the wandb file holds the metrics),
        plus `written_at` (local time).
        """

        data = {name: value for name, value in asdict(self).items() if name != "history"}
        for name in ("run_directory", "resumed_from", "export_dir"):
            data[name] = None if data[name] is None else str(data[name])
        data["checkpoints_written"] = [str(path) for path in self.checkpoints_written]
        data["samples_written"] = [str(path) for path in self.samples_written]
        data["written_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return data

    def write_json(self, path: Path) -> None:
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=2)
            file.write("\n")

    def summary(self) -> str:
        """
        The lines the CLI prints after `train()` returned (the per-depth validation losses; the per-source ones
        are in the JSON report).
        """

        origin = f"resumed from {self.resumed_from}" if self.resumed_from is not None else "fresh start"
        lines = [
            f"Training run in {self.run_directory}: {self.steps_this_process} optimizer steps completed "
            f"(final step {self.completed_steps}, {origin})",
            f"  setup {self.setup_seconds:.1f}s, training {self.train_seconds:.1f}s",
        ]
        if self.stopped:
            lines.append(f"  stopped on request after step {self.completed_steps}; rerun with resume: true to continue")
        loss = f"last loss {self.last_loss:.4f}" if self.last_loss is not None else "no step logged"
        if self.last_validation:
            losses = ", ".join(
                f"{name} {value:.4f}"
                for name, value in self.last_validation.items()
                if name.startswith("val_loss") and "/" not in name
            )
            lines.append(f"  {loss} | last validation: {losses}")
        else:
            lines.append(f"  {loss} | no validation")
        if self.last_benchmarks:
            scores = ", ".join(f"{name.removeprefix('benchmark/')} {value:.4f}" for name, value in self.last_benchmarks.items())
            lines.append(f"  benchmarks: {scores}")
        if self.samples_written:
            lines.append(f"  {len(self.samples_written)} samples files written, last: {self.samples_written[-1]}")
        if self.checkpoints_written:
            lines.append(f"  {len(self.checkpoints_written)} checkpoints written, last: {self.checkpoints_written[-1]}")
        else:
            lines.append("  no checkpoint written")
        lines.append(f"  HuggingFace export: {self.export_dir}" if self.export_dir else "  no HuggingFace export")
        return "\n".join(lines)


class Dashboard(Protocol):
    """
    The four calls `RunLogger` makes on the run's terminal dashboard. `training.ui`'s `TrainingDashboard` (the
    live display) and `ConsoleFallbackDashboard` satisfy it; tests pass a recording fake.
    """

    def update_step(
        self, step: int, stage_index: int, transition: float | None, metrics: Mapping[str, object]
    ) -> None: ...

    def update_validation(self, step: int, losses: Mapping[str, object]) -> None: ...

    def note_event(self, text: str) -> None: ...

    def set_status(self, text: str) -> None: ...


@contextmanager
def open_dashboard(
    settings: Settings, run_directory: Path, stage_manager: StageManager, *, start_step: int, device: str
) -> Iterator[Dashboard]:
    """
    The run's dashboard, in service for the block: the live `TrainingDashboard` when stdout is a terminal and
    `TRAINING_DASHBOARD` is not `0`, else the `ConsoleFallbackDashboard` with its lines on stderr (where the CLI's
    log handlers write too). One bar per stage plus the overall bar, a header naming run, model, dataset, device and
    precision, the `training` logger routed in and everything appended to `run_directory / train.log`. `start_step`
    (the resume step) keeps the ETA honest.
    """

    stage_names = [stage.name for stage in stage_manager.stages]
    steps_per_stage = [boundary.end_step - boundary.start_step for boundary in stage_manager.boundaries]
    details = {
        "model": Path(settings.model_architecture_config).stem,
        "dataset": Path(settings.dataset_config).stem,
        "device": device,
        "precision": settings.precision,
    }
    board: TrainingDashboard | ConsoleFallbackDashboard
    if dashboard_enabled():
        board = TrainingDashboard(
            settings.run_name,
            stage_names,
            steps_per_stage,
            stage_manager.total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=settings.log_step_interval,
            fallback_stream=sys.stderr,
        )
    else:
        board = ConsoleFallbackDashboard(
            settings.run_name,
            stage_names,
            steps_per_stage,
            stage_manager.total_steps,
            details=details,
            start_step=start_step,
            log_step_interval=settings.log_step_interval,
            stream=sys.stderr,
        )
    with board.running(log_file=run_directory / TRAIN_LOG_NAME):
        yield board


class RunLogger:
    """
    Console records, the terminal dashboard, wandb metrics, timers, the data-composition counter and the metric
    history of one run.

    Create it with `open()` once the setup is done and use it as a context manager, so a failing loop still releases
    the dashboard; `close()` returns the `TrainingReport`. Nothing here touches the numerics: tensors become floats
    only at log steps and for validation metrics, where the thesis loop called `.item()`.
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
        self._exit_stack = ExitStack()  # closed by `close()` / `__exit__`; the dashboard is entered on it
        # a given dashboard (tests) is driven as it is and not closed here; else `open_dashboard` picks one for the run
        self.dashboard: Dashboard = (
            dashboard
            if dashboard is not None
            else self._exit_stack.enter_context(
                open_dashboard(settings, run_directory, stage_manager, start_step=start_step, device=device)
            )
        )
        self.keep_history = keep_history  # a test knob: fill `history`
        self.history: dict[int, dict[str, float]] = {}  # per logged step: the metric dict as floats, if kept
        self.checkpoints_written: list[Path] = []
        self.samples_written: list[Path] = []
        self._last_benchmarks: dict[str, float] = {}
        self.resumed_from: Path | None = None
        self.tokens_per_step = settings.tokens_per_optimizer_step
        self._clock = clock
        now = clock()
        self.setup_seconds = now - setup_started if setup_started is not None else 0.0
        self._train_started = now  # the train timer: `total_time` of the metrics, `train_time` of the wandb summary
        self._interval_started = now  # the log-interval timer behind `seconds/step`; reset at every log step
        self._interval_step = start_step  # the step the interval timer started at
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
        """
        Open the run's logging once the setup is done: the wandb run (hyperparameters, `num_parameters`), the
        dashboard (unless `dashboard` is given), then the console header lines. The setup timer ends and the train
        timer starts here. `progress.step` is the resume step; `setup_started` the clock reading at the start of the
        run; `clock` and `keep_history` are test knobs.
        """

        wandb = Logger(
            settings.logger_project,
            settings.run_name,
            run_directory,
            offline=settings.wandb_offline,
            enabled=settings.wandb_enabled,
            resume_step=progress.resume_step if progress.resume_step >= 0 else None,
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
        """
        Release wandb and the dashboard, so the terminal is restored on an exception and on Ctrl-C too; idempotent.
        Every resource is released even when an earlier release raises; the first failure is the one re-raised.
        """

        failures: list[BaseException] = []
        for release in (self.wandb.finish, self._exit_stack.close):
            try:
                release()
            except BaseException as failure:  # released in order, re-raised below: nothing is swallowed
                failures.append(failure)
        if failures:
            raise failures[0]

    # --- status and events -----------------------------------------------------------------------------------------

    def status(self, text: str) -> None:
        """
        The dashboard's header status: what the run is doing right now (`training`, `stopping after this step,
        saving a checkpoint`, `exporting`; `finished` / `stopped on request` set by `close`).
        """

        self._status = text
        self.dashboard.set_status(text)

    @contextmanager
    def _status_during(self, text: str) -> Iterator[None]:
        """
        Show `text` as the status for the block, then the status from before it.
        """

        previous = self._status
        self.status(text)
        try:
            yield
        finally:
            self.status(previous)

    @contextmanager
    def evaluating(self) -> Iterator[None]:
        """
        Around one `evaluate` call: the status reads `evaluating`, and the duration becomes `val_time` (seconds)
        next to the validation metrics of that step in `log_step`.
        """

        started = self._clock()
        with self._status_during("evaluating"):
            try:
                yield
            finally:
                self._evaluation_seconds = self._clock() - started

    def working(self, text: str) -> AbstractContextManager[None]:
        """
        Around a block that is neither a step nor an evaluation (sampling, benchmarking): the status reads text.
        """

        return self._status_during(text)

    def saving_checkpoint(self) -> AbstractContextManager[None]:
        """
        Around one checkpoint write: the status reads `saving checkpoint`.
        """

        return self._status_during("saving checkpoint")

    def log_resume(self, path: Path, step: int) -> None:
        """
        The run continues from checkpoint `path` at optimizer step `step` (the report's `resumed_from`).
        """

        self.resumed_from = path
        self.dashboard.note_event(f"resumed from {path} at step {step}")

    def log_fresh_start(self) -> None:
        """
        No checkpoint was loaded; the run starts at step 0.
        """

        self.dashboard.note_event("no checkpoint found, starting from scratch")

    def log_checkpoint(self, path: Path) -> None:
        """
        A checkpoint was written to `path` (the report's `checkpoints_written`).
        """

        self.checkpoints_written.append(path)
        self.dashboard.note_event(f"saved checkpoint {path}")

    def log_export(self, path: Path) -> None:
        """
        The HuggingFace export was written to `path`.
        """

        self.dashboard.note_event(f"exported HuggingFace model to {path}")

    def log_triggers(self, label: str, steps: Sequence[int]) -> None:
        """
        One console line naming the steps after which `label` (samples, benchmarks) runs; nothing when none.
        """

        if steps:
            console.info("%s after steps: %s", label, ", ".join(str(step) for step in steps))

    def log_samples(self, path: Path, samples: Sequence[GeneratedSample]) -> None:
        """
        Sample generations were written to `path`: the event names the file, a second one previews the first sample.
        """

        self.samples_written.append(path)
        self.dashboard.note_event(f"wrote {len(samples)} samples to {path}")
        if samples:
            preview = f"sample: {samples[0].prompt!r} -> {samples[0].completion!r}"
            self.dashboard.note_event(preview if len(preview) <= 160 else preview[:157] + "...")

    def log_benchmarks(self, metrics: Mapping[str, float], path: Path, step: int) -> None:
        """
        Benchmark scores (`benchmark/<recurrence>/<task>/<metric>`) of `step`: to wandb at that step, one event
        per recurrence setting and task, and the report's `last_benchmarks`.
        """

        self._last_benchmarks = dict(metrics)
        self.wandb.log(dict(metrics), step=step)
        per_task: dict[tuple[str, str], list[str]] = {}
        for name, value in metrics.items():
            _, recurrence, task, metric = name.split("/", 3)
            per_task.setdefault((recurrence, task), []).append(f"{metric} {value:.4f}")
        for (recurrence, task), scores in per_task.items():
            self.dashboard.note_event(f"benchmark {task} (recurrence {recurrence}) at step {step}: {', '.join(scores)}")
        self.dashboard.note_event(f"wrote benchmark results to {path}")

    def log_benchmark_failure(self, error: BaseException) -> None:
        """
        The benchmark run raised: a kept warning and an event; the run goes on.
        """

        console.warning("benchmark evaluation failed, the run continues: %s", error, extra=KEEP)
        self.dashboard.note_event(f"benchmark evaluation failed: {error}")

    # --- steps -------------------------------------------------------------------------------------------------------

    def log_step(self, result: StepResult, progress: TrainingProgress) -> None:
        """
        Account one completed optimizer step (`progress.step`, after `progress.advance()`).

        Every step: the data ids join the composition counter, a transition starting or ending becomes an event, a
        set `result.validation` becomes the dashboard's validation row, the bars move (with an empty metric dict, so
        no tensor is read). At log steps (`step % log_step_interval == 0`, and the final step whatever the interval)
        the metric dict goes to wandb, to `history` with `keep_history`, and to the dashboard:

        * `loss`, `ppl`, `lr`, `grad_norm` (pre-clip), `step`;
        * `seconds/step`, `tokens/second`, `total_tokens` (from step 0, also after a resume), `total_time`,
          `remaining_time`;
        * `stage/current_stage`, `stage/base_lr`, `stage/in_transition`, `stage/transition_progress`,
          `stage/stage_progress`: the stage info the step trained on (`result.stage`);
        * `data_composition/<data id>`: the fraction of world-batch samples per data id since the last log step;
        * `track_gradient_metrics` (`result.metrics`) and the validation metrics (`val_loss*`, `val_ppl*`,
          `val_loss/<data id>` per validation source, `val_time`).
        """

        self._sample_counter.update(result.data_ids)
        stage_at_done = self.stage_manager.get_stage_info(progress.step)
        self._note_transition(result.stage, stage_at_done)
        validation = self._log_validation(result, progress)
        transition = stage_at_done.transition_progress if stage_at_done.transition_to is not None else None
        final = progress.step >= self.stage_manager.total_steps  # always logged: its metrics and validation close the run
        if progress.step % self.settings.log_step_interval != 0 and not final:
            self.dashboard.update_step(progress.step, stage_at_done.stage_index, transition, {})
            return
        metrics = self._step_metrics(result, progress, validation)
        self.wandb.log(metrics, step=progress.step)
        if self.keep_history:
            self.history[progress.step] = {name: float(value) for name, value in metrics.items()}
        self._last_loss = float(metrics["loss"])
        self.dashboard.update_step(progress.step, stage_at_done.stage_index, transition, metrics)

    def _note_transition(self, before: StageInfo, after: StageInfo) -> None:
        """
        The two transition events: "starting transition" after the last plain step of a stage, "transition
        complete" after the last transition step. `before` is the stage info at the step that trained, `after` the
        one at the completed step.
        """

        stages = self.stage_manager.stages
        if after.transition_to is not None and before.transition_to is None:
            leaving, entering = stages[after.stage_index], stages[after.transition_to]
            self.dashboard.note_event(
                f"starting transition {after.stage_index} -> {after.transition_to} ({leaving.name} -> {entering.name}), "
                f"LR {leaving.base_lr:.2e} -> {entering.base_lr:.2e}"
            )
        elif before.transition_to is not None and after.transition_to is None:
            self.dashboard.note_event(
                f"transition complete, now in stage {after.stage_index} ({stages[after.stage_index].name})"
            )

    def _log_validation(self, result: StepResult, progress: TrainingProgress) -> dict[str, float] | None:
        """
        The validation metrics of this step as floats plus `val_time`, the `val_loss*` entries shown on the
        dashboard; None if the step did not evaluate.
        """

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
        """
        The metric dict of a log step (documented in `log_step`); resets the interval timer and the composition
        counter.
        """

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
            "stage/current_stage": result.stage.stage_index,
            "stage/base_lr": self.stage_manager.stages[result.stage.stage_index].base_lr,
            "stage/in_transition": int(result.stage.transition_to is not None),
            "stage/transition_progress": result.stage.transition_progress,
            "stage/stage_progress": result.stage.stage_progress,
        }
        metrics |= {f"data_composition/{name}": count / total_samples for name, count in self._sample_counter.items()}
        self._sample_counter.clear()
        return metrics

    def close(self, progress: TrainingProgress, export_dir: Path | None, *, stopped: bool = False) -> TrainingReport:
        """
        End the run's logging: `train_time` into the wandb summary, the final console line and status, the
        dashboard closed; returns the report, also written to `train_report.json` in the run directory (the last
        process's report; a resume overwrites it). `stopped` says the run ended on request before its last step.
        """

        train_seconds = self._clock() - self._train_started
        self.wandb.log_summary({"train_time": train_seconds})
        self.wandb.finish()
        ending = "stopped on request" if stopped else "finished"
        console.info(f"Training {ending} after {progress.step} steps in {train_seconds:.1f}s.", extra=KEEP)
        self.status(ending)
        self._exit_stack.close()
        report = TrainingReport(
            run_directory=self.run_directory,
            steps_this_process=progress.step - self.start_step,
            completed_steps=progress.step,
            resumed_from=self.resumed_from,
            setup_seconds=self.setup_seconds,
            train_seconds=train_seconds,
            last_loss=self._last_loss,
            last_validation=dict(self._last_validation),
            checkpoints_written=list(self.checkpoints_written),
            export_dir=export_dir,
            stopped=stopped,
            history=self.history,
            samples_written=list(self.samples_written),
            last_benchmarks=dict(self._last_benchmarks),
        )
        report.write_json(self.run_directory / TRAIN_REPORT_NAME)
        return report


# --- gradient / parameter metrics -----------------------------------------------------------------------------------


def _reverse_engineer_adam_effective_lr(
    param: torch.Tensor, param_state: dict[str, torch.Tensor], group: dict[str, Any]
) -> torch.Tensor:
    """
    Recompute Adam's per-element effective LR (ignoring bias correction and the scheduled LR).
    """

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
    """
    (n_embd, query width, key/value width) for slicing fused qkv gradients; None for non-transformer models.
    """

    config = getattr(model, "config", None)
    if config is None or not all(hasattr(config, name) for name in ("n_embd", "head_size", "num_attention_heads")):
        return None
    return config.n_embd, config.n_embd, config.head_size * config.num_attention_heads  # no GQA: kv width == q width


@torch.no_grad()
def track_gradient_metrics(model: Module, optimizer: Optimizer) -> dict[str, torch.Tensor]:
    """
    Gradient norms, Adam second-moment RMS, effective LRs and parameter norms. Call after `optimizer.step()`
    and before `zero_grad()`.

    One pass over the parameters in optimizer-group order: `query_grad_<i>` / `ffn2_grad_<i>` number the fused-qkv
    and MLP-projection weights with a gradient (NaN when non-finite), `*_effective_lr_<i>` those with a finite
    gradient and Adam state.
    """

    metrics: dict[str, torch.Tensor] = {}
    dims = _qkv_dims(model)
    transformer = getattr(model, "transformer", None)
    wte_module: Optional[Module] = getattr(transformer, "wte", None)
    wte_weight: Optional[torch.Tensor] = getattr(wte_module, "weight", None)
    names = {id(param): name for name, param in model.named_parameters()}

    grad_qkv_layer, grad_mlp_layer = 0, 0  # `query_grad_<i>` / `ffn2_grad_<i>`
    lr_qkv_layer, lr_mlp_layer = 0, 0  # `*_effective_lr_<i>`
    total_rms: torch.Tensor | float = 0.0
    num_params_with_grad = 0
    finite_grads: list[torch.Tensor] = []
    with_grad: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                with_grad.append((group, param, param.grad))
    # one host sync for all finite checks instead of one per parameter (the production run logs every step)
    finite_flags = torch.stack([grad.isfinite().all() for *_, grad in with_grad]).tolist() if with_grad else []
    for (group, param, grad), finite in zip(with_grad, finite_flags):
        name = names.get(id(param), "")
        is_qkv = "qkv" in name and "weight" in name
        is_proj = "mlp" in name and "proj" in name and "weight" in name
        if is_qkv:
            if not finite:
                metrics[f"query_grad_{grad_qkv_layer}"] = torch.as_tensor(float("NaN"))
            elif dims is not None and grad.numel() % dims[0] == 0:
                metrics[f"query_grad_{grad_qkv_layer}"] = grad.view(-1, dims[0])[: dims[1], :].norm()
            grad_qkv_layer += 1
        if is_proj:
            metrics[f"ffn2_grad_{grad_mlp_layer}"] = grad.norm() if finite else torch.as_tensor(float("NaN"))
            grad_mlp_layer += 1
        if not finite:
            continue
        finite_grads.append(grad)
        state = optimizer.state.get(param)
        if state is None:
            continue
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg_sq is None or exp_avg_sq.shape != grad.shape:
            continue
        # out of place: `.float()` aliases an fp32 buffer, an in-place clamp would edit the optimizer state
        rms = grad.float().pow(2).div_(exp_avg_sq.float().clamp(min=group["eps"] ** 2)).mean().sqrt()
        total_rms += rms
        num_params_with_grad += 1
        if wte_weight is not None and param is wte_weight:
            metrics["embed_RMS"] = rms
        if is_qkv:
            qkv_lr = _reverse_engineer_adam_effective_lr(param, state, group)
            if dims is not None and qkv_lr.numel() % dims[0] == 0:
                n_embd, query_width, kv_width = dims
                qkv_lr = qkv_lr.view(-1, n_embd)
                metrics[f"q_effective_lr_{lr_qkv_layer}"] = qkv_lr[:query_width, :].mean()
                metrics[f"k_effective_lr_{lr_qkv_layer}"] = qkv_lr[query_width : query_width + kv_width, :].mean()
                metrics[f"v_effective_lr_{lr_qkv_layer}"] = qkv_lr[query_width + kv_width :, :].mean()
            lr_qkv_layer += 1
        if is_proj:
            metrics[f"ffn2_effective_lr_{lr_mlp_layer}"] = _reverse_engineer_adam_effective_lr(param, state, group).mean()
            lr_mlp_layer += 1

    if num_params_with_grad > 0:
        metrics["avg_RMS"] = torch.as_tensor(total_rms / num_params_with_grad)  # already a Tensor after one add

    if finite_grads:
        metrics["local_l1_grad_norm"] = torch.mean(torch.stack([torch.norm(grad.detach(), 1.0) for grad in finite_grads]))

    # parameter norms
    metrics["l2_param_norm"] = torch.norm(torch.stack([torch.norm(param.detach()) for param in model.parameters()]))
    metrics["l1_param_norm"] = torch.mean(
        torch.stack([torch.norm(param.detach(), 1.0) for param in model.parameters()])
    )

    core_blocks = getattr(transformer, "core_blocks", None)
    if core_blocks is not None:
        for block_idx, core_block in enumerate(core_blocks):
            metrics[f"core_block_{block_idx}_l2_param_norm"] = torch.norm(
                torch.stack([torch.norm(param.detach()) for param in core_block.parameters()])
            )
    if wte_module is not None and wte_weight is not None:
        metrics["word_embed_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(param.detach()) for param in wte_module.parameters()])
        )
        metrics["model_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(param.detach()) for name, param in model.named_parameters() if "wte" not in name])
        )
    return metrics
