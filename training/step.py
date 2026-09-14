# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
One optimizer step of the training loop: the micro-batch stream, the scheduled learning rate and
`run_one_optimizer_step`, coordinating autocast / backward / clipping through `training.steps` helpers.

Everything here affects numerics; the golden tests in `test_step.py` and `test_run.py` pin the current depth schedule
and accumulation arithmetic. The stream is the reference for the data path: ONE continuous reader per source for the whole
run, the source of every document chosen deterministically so that the sources' TOKEN shares follow the stage
weights (`BatchStream._pick_source`), and the documents packed into one row per micro-batch
(`training.data.packing`). Steps are OPTIMIZER steps: `micro_batches_per_step` packed micro-batches over all ranks
(`Settings.micro_batches_per_rank` of them on each), one `optimizer.step()`.

With several ranks the stream lives on the main rank only: `RankBatches` is every rank's view of it. Per micro-batch
index the main rank pulls one pack per rank, scatters each rank its own (`Backend.scatter_packs`) and keeps the data
statistics of all of them, so the logged composition describes the whole world's step; the other ranks own no train
loader at all. At world size 1 it is the stream itself.
"""

from collections.abc import Callable, Iterator

from model import RecurrentGPT
from torch.nn import Module
from torch.optim import Optimizer

from training.backend.base import Backend
from training.data.packing import PackedBatch
from training.optim import set_lr
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps import (
    AccumulatedGradients, StepResult, TrainingProgress, build_model_inputs, collect_step_metrics,
    create_accumulated_gradients, record_batch_composition, run_microbatch_backward, select_gradient_sync_context,
    get_scheduled_learning_rate, normalize_and_clip_gradients, reduce_supervised_loss,
)


def run_one_optimizer_step(
    settings: Settings,
    backend: Backend,
    model: Module,
    optimizer: Optimizer,
    stage_manager: StageManager,
    batches: Iterator[PackedBatch],
    progress: TrainingProgress,
    on_micro_batch: Callable[[int, int], None] | None = None,
) -> StepResult:
    """
    Run optimizer step `progress.step`: this rank's `micro_batches_per_rank` packed micro-batches from `batches`,
    one `optimizer.step()`. Does not advance `progress` (`train()` does, right after). `on_micro_batch(completed,
    total)` is called after every micro-batch's backward was issued (the dashboard's micro-batch bar); on a GPU the
    device may still be working on it, since nothing here synchronises.

    Recurrence depth is sampled per local microbatch; the golden tests in `test_step.py` and `test_run.py` pin the
    numerics. Non-obvious parts: step 0 skips `optimizer.step()`, `grad_norm` is measured before clipping, and the
    loss is all-reduced every step (a no-op on one device) BEFORE it is checked for finiteness, so every rank sees
    the same number and makes the same decision to raise. Backwards accumulate token sums scaled by the fixed
    local physical capacity C. DDP averages these gradients across W ranks; multiplying by W*C/N after all
    backwards yields the global supervised-token mean, before clipping. Counts remain int64 until division.
    """

    # configure this optimizer step and its learning rate
    step = progress.step
    accumulation_steps = settings.micro_batches_per_rank(backend.world_size)
    plain = backend.plain_model(model)
    plain.step = step
    stage = stage_manager.get_stage_info(step)
    learning_rate = get_scheduled_learning_rate(settings, stage_manager, progress)
    set_lr(optimizer, learning_rate)

    # accumulate microbatches and reduce the supervised-token loss
    accumulated = accumulate_microbatch_gradients(settings, backend, model, plain, batches, accumulation_steps, on_micro_batch)
    loss, supervised_count = reduce_supervised_loss(backend, accumulated, step)

    # normalize and clip gradients before applying the update
    grad_norm = normalize_and_clip_gradients(settings, backend, model, accumulated.local_capacity, supervised_count, step)
    if step > 0:  # thesis parity: skip the first update regardless of the LR warmup schedule
        optimizer.step()

    # collect diagnostics before clearing gradients and returning the result
    metrics = collect_step_metrics(settings, backend, model, optimizer, step, accumulated.padding_tokens)
    optimizer.zero_grad(set_to_none=True)

    return StepResult(
        step=step,
        learning_rate=learning_rate,
        loss=loss,
        grad_norm=grad_norm,
        stage=stage,
        data_ids=accumulated.data_ids,
        data_tokens=accumulated.data_tokens,
        metrics=metrics,
    )


def accumulate_microbatch_gradients(
    settings: Settings, backend: Backend, model: Module, plain: RecurrentGPT, batches: Iterator[PackedBatch],
    accumulation_steps: int, on_micro_batch: Callable[[int, int], None] | None,
) -> AccumulatedGradients:
    """Accumulate local token sums, using no_sync until the final backward and preserving callback timing."""

    # initialize the local totals before reading any microbatches
    accumulated = create_accumulated_gradients(settings, backend, accumulation_steps)

    # process each microbatch in the original data and recurrence order
    for micro_batch_index in range(accumulation_steps):
        batch = next(batches)
        record_batch_composition(accumulated, batch)
        inputs = build_model_inputs(batch, backend)
        plain.micro_batch_index = micro_batch_index  # same local index on every rank; latent RNG remains rank-specific

        # run backward, synchronizing gradients only on the final microbatch
        with select_gradient_sync_context(backend, model, micro_batch_index, accumulation_steps):
            outputs = run_microbatch_backward(settings, backend, model, inputs, accumulated.local_capacity)

        # accumulate detached loss statistics, then notify the dashboard
        accumulated.loss_sum += outputs["loss_sum"].detach()
        accumulated.supervised_count += outputs["supervised_count"].detach()
        if on_micro_batch is not None:
            on_micro_batch(micro_batch_index + 1, accumulation_steps)

    return accumulated
