# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for one optimizer step (`training.step`): the progress counter, the micro-batch stream, the LR, the
accumulation arithmetic, the skipped first update, the non-finite checks — and the dataset-independent numerics
reference `training/golden_tiny_steps.json` (five steps of fixed batches through `run_one_optimizer_step`)."""

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from model import RecurrentGPT, build_model
from training.backend.single_device import SingleDeviceBackend
from training.data.collate import IGNORE_INDEX, Batch, Sample, WorkerBatch
from training.data.loader import RunDataloaders, SampleBatch, build_run_dataloaders, dataloader_over, entry_dataset
from training.data.collate import find_multiple
from training.data.dataset_resolver import DataEntry, resolve_dataset
from training.data.datasets import Row
from training.data.tokenizer import Tokenizer
from training.testing.golden import (
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    single_thread_deterministic,
    write_tiny_yaml,
)
from training.run import build_run_optimizer
from training.settings import OptimizerConfig, Settings, parse_settings
from training.stage_manager import StageManager
from training.testing.stages import resolved_stage
from training.step import (
    BatchStream,
    StepResult,
    TrainingProgress,
    run_one_optimizer_step,
    scheduled_learning_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL_ARCHITECTURE = REPO_ROOT / "config" / "model_architecture" / "tiny.yaml"
GOLDEN_STEPS_PATH = Path(__file__).resolve().parent / "golden_tiny_steps.json"

REFERENCE_STEPS = 5
REFERENCE_SEQUENCE_LENGTH = 32


# --------------------------------------------------------------------------------------------------------------
# a run without a dataset: in-memory settings, a one-stage manager, scripted batches


def reference_settings(**overrides: Any) -> Settings:
    """Settings of the step reference: the tiny architecture, `world_batch_size` 4 as 2 micro-batches of 2, the
    thesis optimizer (ELLISAdam with the `crow_300m_final.yaml` options), warmup 2 / cooldown 2, gradient metrics at
    every step. The dataset config is never read (no loader is built)."""
    values: dict[str, Any] = dict(
        dataset_config="config/datasets/tiny.yaml",
        model_architecture_config=str(TINY_MODEL_ARCHITECTURE),
        stage_base_lrs=[3e-4],
        run_name="steps",
        out_dir="unused",
        seed=0,
        block_size=256,
        micro_batch_size=2,
        world_batch_size=4,
        precision="32",
        optimizer="ELLISAdam",
        optim_config=OptimizerConfig(
            lr=1e-4, weight_decay=4e-5, betas=(0.9, 0.95), update_clipping=True, atan_adam=True, running_init=True
        ),
        grad_clip=1.0,
        warmup_steps=2,
        cooldown_steps=2,
        log_step_interval=1,
        log_gradient_metrics=True,
        wandb_enabled=False,
    )
    values.update(overrides)
    return Settings(**values)


def reference_stage_manager(settings: Settings, steps: int = 10) -> StageManager:
    """One stage of `steps` optimizer steps (no transition): LR 0 at step 0, 1.5e-4 at step 1, 3e-4 from step 2."""
    stage = resolved_stage("only", tokens=steps * settings.world_batch_size * settings.block_size, base_lr=3e-4, transition_pct=0.0)
    return StageManager([stage], settings.world_batch_size, settings.block_size, warmup_steps=2, cooldown_steps=2)


def scripted_batches(settings: Settings, seed: int = 0, sequence_length: int = REFERENCE_SEQUENCE_LENGTH) -> Iterator[Batch]:
    """Endless micro-batches of random token ids (vocab 512) from a seeded generator — independent of the global torch
    RNG, the tokenizer and the data pipeline. Labels are ids too (as `collate_fn` yields them, already shifted)."""
    generator = torch.Generator().manual_seed(seed)
    while True:
        input_ids = torch.randint(1, 512, (settings.micro_batch_size, sequence_length), generator=generator)
        labels = torch.randint(1, 512, (settings.micro_batch_size, sequence_length), generator=generator)
        yield input_ids, labels, ["scripted"] * settings.micro_batch_size


def fresh_tiny_model(backend: SingleDeviceBackend, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return backend.setup_model(build_model(TINY_MODEL_ARCHITECTURE))


def fresh_optimizer(settings: Settings, model: torch.nn.Module, backend: SingleDeviceBackend) -> torch.optim.Optimizer:
    return build_run_optimizer(settings, model, backend)


@pytest.fixture
def cpu_backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


@pytest.fixture
def settings() -> Settings:
    return reference_settings()


def run_steps(
    settings: Settings, backend: SingleDeviceBackend, model: torch.nn.Module, optimizer: torch.optim.Optimizer, steps: int
) -> list[StepResult]:
    """`steps` consecutive optimizer steps of scripted batches, advancing a fresh progress counter as `train()` does."""
    stage_manager = reference_stage_manager(settings)
    batches = scripted_batches(settings)
    progress = TrainingProgress()
    results = []
    for _ in range(steps):
        results.append(run_one_optimizer_step(settings, backend, model, optimizer, stage_manager, batches, progress))
        progress.advance()
    return results


# --------------------------------------------------------------------------------------------------------------
# progress and LR


def test_training_progress_counts_steps() -> None:
    progress = TrainingProgress()
    assert (progress.step, progress.resume_step) == (0, -1)
    progress.advance()
    assert progress.step == 1
    resumed = TrainingProgress(step=14, resume_step=14)
    resumed.advance()
    assert (resumed.step, resumed.resume_step) == (15, 14)


def test_scheduled_learning_rate_follows_warmup_and_resume(settings: Settings) -> None:
    stage_manager = reference_stage_manager(settings)  # 10 steps, warmup 2, cooldown 2, base LR 3e-4
    lrs = [scheduled_learning_rate(settings, stage_manager, TrainingProgress(step=s)) for s in range(10)]
    assert lrs[:5] == pytest.approx([0.0, 1.5e-4, 3e-4, 3e-4, 3e-4])
    assert lrs[8] == pytest.approx(3e-4) and lrs[9] == pytest.approx(1.5e-4)  # cooldown over the last 2 steps
    settings.resume_warmup_steps = 2
    resumed = TrainingProgress(step=4, resume_step=4)
    assert scheduled_learning_rate(settings, stage_manager, resumed) == pytest.approx(0.0)  # ramp restarts at min_lr


def test_learning_rate_is_set_on_all_groups(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    assert len(optimizer.param_groups) == 3
    results = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    assert results[1].learning_rate == pytest.approx(1.5e-4)  # warmup step 1 of 2
    for group in optimizer.param_groups:
        assert torch.is_tensor(group["lr"])  # `set_lr` stores a tensor (ELLISAdam clones it)
        assert float(group["lr"]) == pytest.approx(results[1].learning_rate)


# --------------------------------------------------------------------------------------------------------------
# the step body


def _parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_step_zero_performs_no_update_step_one_does(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """The very first update is skipped (as in the thesis); step 1 updates every parameter with a gradient."""
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    stage_manager = reference_stage_manager(settings)
    batches = scripted_batches(settings)
    progress = TrainingProgress()

    before = _parameters(model)
    result = run_one_optimizer_step(settings, cpu_backend, model, optimizer, stage_manager, batches, progress)
    assert result.step == 0 and result.learning_rate == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, _parameters(model)))
    assert all(p.grad is None for p in model.parameters())  # zero_grad(set_to_none=True) ran
    assert not optimizer.state  # ELLISAdam has not initialised its moments

    progress.advance()
    result = run_one_optimizer_step(settings, cpu_backend, model, optimizer, stage_manager, batches, progress)
    assert result.step == 1 and result.learning_rate > 0
    changed = [not torch.equal(a, b) for a, b in zip(before, _parameters(model))]
    assert all(changed)
    assert all("exp_avg" in optimizer.state[p] for group in optimizer.param_groups for p in group["params"])


def _hand_accumulation(
    settings: Settings, model: torch.nn.Module, batches: Iterator[Batch], step: int, seed: int
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The accumulation of one step done by hand on a copy of `model`: per-micro-batch losses and the pre-clip norm of
    the accumulated gradients. Seeds the global RNG like the caller so the recurrence draws match."""
    copy_of_model = copy.deepcopy(model)
    assert isinstance(copy_of_model, RecurrentGPT)
    copy_of_model.step = step
    torch.manual_seed(seed)
    losses = []
    for _ in range(settings.gradient_accumulation_steps):
        input_ids, labels, _ = next(batches)
        loss = copy_of_model(input_ids, labels=labels)["loss"]
        assert loss is not None
        (loss / settings.gradient_accumulation_steps).backward()
        losses.append(loss.detach())
    grads = [p.grad for p in copy_of_model.parameters() if p.grad is not None]
    norm = torch.stack([g.norm() for g in grads]).norm()
    return losses, norm


def test_loss_and_grad_norm_match_a_hand_computation(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """`loss` is the mean of the per-micro-batch losses; `grad_norm` is the pre-clip L2 norm of the accumulated
    gradients (the clip threshold 1.0 is far below it here, so it is not the clipped norm)."""
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    expected_losses, expected_norm = _hand_accumulation(settings, model, scripted_batches(settings), step=0, seed=11)

    torch.manual_seed(11)
    result = run_one_optimizer_step(
        settings, cpu_backend, model, optimizer, reference_stage_manager(settings), scripted_batches(settings), TrainingProgress()
    )
    assert len(expected_losses) == settings.gradient_accumulation_steps == 2
    assert result.loss.item() == pytest.approx(torch.stack(expected_losses).mean().item(), rel=1e-6)
    assert result.log_ppl.item() == pytest.approx(result.loss.item(), rel=1e-6)  # log_ppl is the detached loss
    assert result.grad_norm.item() == pytest.approx(expected_norm.item(), rel=1e-5)
    assert result.grad_norm.item() > settings.grad_clip  # so the clipping actually happened and the norm is pre-clip


def test_data_ids_has_world_batch_size_entries(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    (result,) = run_steps(settings, cpu_backend, model, optimizer, steps=1)
    assert result.data_ids == ["scripted"] * settings.world_batch_size
    assert result.validation is None


def test_stage_infos_and_metrics(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """`stage` is the manager's info at `step`; gradient metrics only at log steps."""
    settings.log_step_interval = 2
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    stage_manager = reference_stage_manager(settings)
    results = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    assert results[0].stage == stage_manager.get_stage_info(0) and results[1].stage == stage_manager.get_stage_info(1)
    assert results[0].metrics == {}  # done = 1, not a log step
    assert "l2_param_norm" in results[1].metrics and "avg_RMS" in results[1].metrics  # done = 2


def test_non_finite_loss_raises_with_the_exact_message(
    settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    forward = RecurrentGPT.forward

    def nan_forward(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        out = forward(self, *args, **kwargs)
        assert out["loss"] is not None
        out["loss"] = out["loss"] * torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(RecurrentGPT, "forward", nan_forward)
    with pytest.raises(RuntimeError) as excinfo:
        run_steps(settings, cpu_backend, model, optimizer, steps=1)
    assert str(excinfo.value) == "Loss is nan at step 0. Terminating."


def test_non_finite_grad_norm_raises_with_the_exact_message(
    settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    monkeypatch.setattr(SingleDeviceBackend, "clip_grad_norm", lambda self, model, max_norm: torch.tensor(float("inf")))
    stage_manager = reference_stage_manager(settings)
    batches = scripted_batches(settings)
    progress = TrainingProgress(step=3)
    with pytest.raises(RuntimeError) as excinfo:
        run_one_optimizer_step(settings, cpu_backend, model, optimizer, stage_manager, batches, progress)
    assert str(excinfo.value) == "Gradient norm is non-finite at step 3. Terminating."


# --------------------------------------------------------------------------------------------------------------
# micro-batch stream (per-SAMPLE source draws over the run-wide readers; fake loaders, hand-made stage managers)


def _fake_samples(tag: str, lengths: list[int]) -> SampleBatch:
    """One worker batch: one unpadded sample per entry of `lengths`, every position supervised."""
    return [(torch.full((n,), 3, dtype=torch.long), torch.full((n,), 3, dtype=torch.long), tag) for n in lengths]


class _Repeat:
    """Endless iterable of one tagged one-sample worker batch with a running counter as the sample length."""

    def __init__(self, tag: str, batch_size: int = 1) -> None:
        self.tag, self.batch_size, self.count = tag, batch_size, 0

    def __iter__(self) -> Iterator[WorkerBatch]:
        while True:
            self.count += 1
            samples = _fake_samples(self.tag, [1 + (self.count + i) % 7 for i in range(self.batch_size)])
            yield WorkerBatch(samples, len(samples))


class _ShortBatches:
    """A loader whose worker batches cycle through `sizes` rows — a short (or empty) batch is what a loader running
    out of rows, or a batch whose rows were all dropped for lack of a supervised label, hands the stream."""

    def __init__(self, tag: str, sizes: list[int]) -> None:
        self.tag, self.sizes, self.count = tag, sizes, 0

    def __iter__(self) -> Iterator[WorkerBatch]:
        while True:
            size = self.sizes[self.count % len(self.sizes)]
            self.count += 1
            samples = _fake_samples(self.tag, [1 + (self.count + i) % 7 for i in range(size)])
            yield WorkerBatch(samples, len(samples))


@pytest.fixture(scope="session")
def stream_tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    """The synthetic tokenizer the stream pads its assembled micro-batches with."""
    return Tokenizer(tiny_tokenizer_dir)


def _abc_stage_manager(settings: Settings) -> StageManager:
    """The tiny stage boundaries ((0,8,6,8), (8,16,14,16), (16,20)) with one fake source per stage: `a` in stage 0,
    `b` in stage 1, `c` in stage 2 — hand-made so no dataset is resolved for the stream tests."""
    stages = [
        resolved_stage("s0", tokens=8192, base_lr=3e-4, transition_pct=0.25, train_weights={"a": 1.0}),
        resolved_stage("s1", tokens=8192, base_lr=1e-4, transition_pct=0.25, train_weights={"b": 1.0}),
        resolved_stage("s2", tokens=4096, base_lr=5e-5, transition_pct=0.0, train_weights={"c": 1.0}),
    ]
    return StageManager(stages, settings.world_batch_size, settings.block_size)


def _stream_setup(
    tmp_path: Path,
    tiny_dataset_dir: Path,
    sort: bool,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    padding_multiple: int = 128,
) -> tuple[Settings, RunDataloaders, StageManager]:
    yaml_path = write_tiny_yaml(
        tmp_path,
        tiny_dataset_dir,
        tmp_path / "out",
        sort_batches_by_length=sort,
        sequence_padding_multiple=padding_multiple,
    )
    settings = parse_settings(
        ["--config", str(yaml_path), "--micro_batch_size", str(batch_size)]  # 4 / batch_size micro-batches per step
    )
    loaders = RunDataloaders({t: _Repeat(t, batch_size) for t in "abc"}, [], tokenizer, {})
    return settings, loaders, _abc_stage_manager(settings)


def _tags(stream: Iterator[Batch], n: int) -> list[str]:
    return [next(stream)[2][0] for _ in range(n)]  # never pull an extra element (zip would)


def test_batch_stream_samples_by_transition_progress(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress(step=3)  # plain stage 0
    stream = BatchStream(settings, loaders, stage_manager, progress)
    assert _tags(stream, 4) == ["a"] * 4
    progress.step = 6  # transition 0 -> 1, progress 0: everything still from stage 0
    assert _tags(stream, 4) == ["a"] * 4
    progress.step = 8  # fully in stage 1
    assert _tags(stream, 4) == ["b"] * 4
    progress.step = 15  # transition 1 -> 2 at progress 0.5: a mix, driven by the rng
    tags = _tags(stream, 40)
    assert set(tags) == {"b", "c"} and 8 < tags.count("c") < 32
    progress.step = 19
    assert _tags(stream, 4) == ["c"] * 4


def test_batch_stream_reads_the_step_lazily(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """The step is read when the first micro-batch of a world batch is requested, not when the stream is created and
    not again inside the world batch."""
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress(step=0)
    stream = BatchStream(settings, loaders, stage_manager, progress)
    progress.step = 8  # changed before the first request: the world batch is stage 1's
    assert _tags(stream, 2) == ["b"] * 2
    progress.step = 19  # changed inside the world batch: the remaining two micro-batches still belong to step 8
    assert _tags(stream, 2) == ["b"] * 2
    assert _tags(stream, 4) == ["c"] * 4  # the next world batch reads step 19


def test_batch_stream_draw_rng_is_seeded_with_the_start_step(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """Two streams created at the same `seed + step` draw the same source sequence; a different start step draws
    another (a resume without a stored data-stream state seeds the draw RNG with `seed + resume step`)."""
    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)

    def mix(start_step: int) -> list[str]:
        loaders = RunDataloaders({tag: _Repeat(tag) for tag in "abc"}, [], stream_tokenizer, {})
        progress = TrainingProgress(step=start_step)
        stream = BatchStream(settings, loaders, stage_manager, progress)
        progress.step = 15
        return _tags(stream, 40)

    assert mix(0) == mix(0)
    assert mix(0) != mix(14)


def test_batch_stream_length_sorting(tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer) -> None:
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, True, stream_tokenizer, padding_multiple=4)
    assert settings.gradient_accumulation_steps == 4
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    for _ in range(3):  # every world batch (4 micro-batches) arrives sorted by length, padded to its own width
        batches = [next(stream) for _ in range(4)]
        lengths = [int((b[1] != IGNORE_INDEX).sum()) for b in batches]  # supervised = sample length - 1
        assert lengths == sorted(lengths)
        assert all(b[0].shape[1] == find_multiple(n + 1, 4) - 1 for b, n in zip(batches, lengths))
        progress.advance()
    unsorted_settings, unsorted_loaders, _ = _stream_setup(
        tmp_path, tiny_dataset_dir, False, stream_tokenizer, padding_multiple=4
    )
    raw = BatchStream(unsorted_settings, unsorted_loaders, stage_manager, TrainingProgress())
    lengths = [int((next(raw)[1] != IGNORE_INDEX).sum()) for _ in range(4)]
    assert lengths == [1, 2, 3, 4]  # loader order, untouched


def test_batch_stream_fills_the_world_batch_from_short_worker_batches(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """Regression (T-M3/T-M4): a short or empty worker batch — a loader reaching its last rows, or a batch whose rows
    were all dropped — used to shrink the world batch and permanently misalign it with the optimizer steps. The
    stream draws exactly `world_batch_size` samples per world batch, pulling a source's loader until its buffer
    holds one and carrying leftover samples over in the buffer."""
    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer, batch_size=2)
    assert (settings.gradient_accumulation_steps, settings.world_batch_size) == (2, 4)
    loaders = RunDataloaders({tag: _ShortBatches(tag, [2, 0, 1, 2, 3]) for tag in "abc"}, [], stream_tokenizer, {})
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    for _ in range(6):
        batches = [next(stream) for _ in range(settings.gradient_accumulation_steps)]
        assert [b[0].shape[0] for b in batches] == [settings.micro_batch_size] * settings.gradient_accumulation_steps
        assert sum(len(b[2]) for b in batches) == settings.world_batch_size
        progress.advance()


# --------------------------------------------------------------------------------------------------------------
# BatchStream state: what a checkpoint stores and what a resume does with it


def test_batch_stream_counts_the_rows_it_consumed(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    assert stream.state_dict()["consumed_rows"] == {}
    for _ in range(3):
        for _ in range(settings.gradient_accumulation_steps):
            next(stream)
        progress.advance()
    # three world batches of `world_batch_size` samples, all drawn from source `a` (stage 0, no transition at steps 0-2)
    assert stream.state_dict()["consumed_rows"] == {"a": 3 * settings.world_batch_size}


def test_batch_stream_state_round_trip(tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer) -> None:
    """`load_state_dict` restores the row counters and the draw RNG, so a stream resumed from the state draws the
    same source sequence as the one it was taken from."""
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress(step=15)  # inside the 1 -> 2 transition: the draws actually mix
    stream = BatchStream(settings, loaders, stage_manager, progress)
    _tags(stream, 8)
    state = stream.state_dict()
    continued = _tags(stream, 12)

    fresh_loaders = RunDataloaders({tag: _Repeat(tag) for tag in "abc"}, [], stream_tokenizer, {})
    resumed = BatchStream(settings, fresh_loaders, stage_manager, TrainingProgress(step=15))
    resumed.load_state_dict(state)
    assert resumed.state_dict()["consumed_rows"] == state["consumed_rows"]
    assert _tags(resumed, 12) == continued


def test_batch_stream_load_state_dict_sets_the_loader_offsets(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """The counters become a per-source row offset (modulo the range), which is what makes a resume skip the rows
    the interrupted run consumed."""
    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    loaders = build_run_dataloaders(settings, dataset, cpu_backend)
    stage_manager = StageManager(dataset.stages, settings.world_batch_size, settings.block_size)
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    parquet = loaders.datasets["synthetic_pretrain"]
    state = {"consumed_rows": {parquet.prefix: parquet.num_rows + 5}, "draw_rng": stream.rng.getstate()}
    stream.load_state_dict(state)
    assert loaders.pending_offsets == {"synthetic_pretrain": parquet.num_rows + 5}
    next(stream)  # stage 0 draws from the pretrain source only: its reader starts now
    assert parquet.resume_offset == 5 and loaders.pending_offsets == {}  # wrapped around one epoch
    assert loaders.datasets["synthetic_instruct"].resume_offset == 0  # untouched sources stay at the start


def test_batch_stream_resume_does_not_repeat_rows(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """The point of checkpointing the stream: a stream restored from a state continues into rows the first one had
    not reached, while a stream that only restarts the loaders serves the very same rows again."""
    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.world_batch_size, settings.block_size)

    def fresh_stream() -> BatchStream:
        loaders = build_run_dataloaders(settings, dataset, cpu_backend)
        return BatchStream(settings, loaders, stage_manager, TrainingProgress())

    def rows(stream: BatchStream, world_batches: int) -> list[tuple[int, ...]]:
        """The first 20 tokens of every sample of `world_batches` world batches — a row identity that survives the
        different padding widths of two runs. Only plain stage-0 steps, so no transition draw is involved."""
        seen: list[tuple[int, ...]] = []
        for _ in range(world_batches):
            for _ in range(settings.gradient_accumulation_steps):
                input_ids, _, _ = next(stream)
                seen += [tuple(row[:20].tolist()) for row in input_ids]
            stream.progress.advance()
        return seen

    stream = fresh_stream()
    before = rows(stream, 3)
    state = stream.state_dict()
    assert len(set(before)) == len(before) == 3 * settings.world_batch_size

    resumed = fresh_stream()
    resumed.load_state_dict(state)
    assert not set(before) & set(rows(resumed, 3))
    assert rows(fresh_stream(), 3) == before  # without the state the rows are read from the top again


def test_stages_sharing_a_source_do_not_re_read_rows(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """The bug the continuous stream fixes: the old per-stage loaders each read `synthetic_pretrain` from the top,
    so stage 1 re-served the very rows stage 0 had trained on. The run-wide reader continues across the boundary:
    every sample up to there and beyond is a distinct row (until the source genuinely wraps around)."""
    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.world_batch_size, settings.block_size)
    loaders = build_run_dataloaders(settings, dataset, cpu_backend)
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    pretrain = loaders.datasets["synthetic_pretrain"]
    steps = 13  # well into stage 1 (the boundary is step 8), before the transition into finetune (step 14)
    assert steps * settings.world_batch_size <= pretrain.num_rows, "fixture too small to distinguish from a wrap"
    seen: list[tuple[int, ...]] = []
    for _ in range(steps):
        for _ in range(settings.gradient_accumulation_steps):
            input_ids, _, data_ids = next(stream)
            assert set(data_ids) == {"synthetic_pretrain"}  # both stages train on the same source
            seen += [tuple(row[:20].tolist()) for row in input_ids]
        stream.progress.advance()
    assert len(set(seen)) == len(seen)  # crossing the stage boundary at step 8 repeated nothing


def test_batch_stream_same_seed_yields_the_same_stream(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """Determinism of the whole stream: two fresh streams over the same dataset and settings yield identical
    micro-batches — same source draws, same rows, same padding (the draw RNG is private and seeded from
    `settings.seed`, and each source's reader walks its range in order)."""
    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.world_batch_size, settings.block_size)

    def batches(world_batches: int) -> list[Batch]:
        loaders = build_run_dataloaders(settings, dataset, cpu_backend)
        stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
        out: list[Batch] = []
        for _ in range(world_batches):
            out += [next(stream) for _ in range(settings.gradient_accumulation_steps)]
            stream.progress.advance()
        return out

    first, second = batches(3), batches(3)
    assert len(first) == len(second) == 3 * settings.gradient_accumulation_steps
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2] for a, b in zip(first, second))


# --------------------------------------------------------------------------------------------------------------
# BatchStream state when the workers DROP rows (H7): the counter is rows READ, the unit the resume skips

DROP_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}
DROP_BLOCK_SIZE = 16  # cap 17 tokens: a 30-word prompt alone fills the window, its row keeps no supervised label
DROP_EVERY = 3  # every third row of the fixture is such a prompt-only row and is dropped in the worker


def _write_drop_parquet(directory: Path, rows: int = 200) -> None:
    """`rows` unique instruct rows of which every `DROP_EVERY`-th tokenizes to nothing at `DROP_BLOCK_SIZE`."""
    long_prompt = " ".join(f"tok_{i}" for i in range(30))
    table = pa.table(
        {
            "instruction": [
                long_prompt if i % DROP_EVERY == 0 else f"tok_{i % 256} tok_{(i // 256) % 256}" for i in range(rows)
            ],
            "input": [""] * rows,
            "output": [f"tok_{i % 256} tok_{(i // 256) % 256} tok_{(i * 11) % 256}" for i in range(rows)],
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "data-00000.parquet")


def _drop_survivors(rows_read: int) -> int:
    """Survivors among the first `rows_read` fixture rows: all but the `i % DROP_EVERY == 0` ones."""
    return rows_read - (rows_read + DROP_EVERY - 1) // DROP_EVERY


class _RecordingLoader:
    """Forwards a real (unpadded) train DataLoader's `WorkerBatch`es while recording every sample and row count that
    passed through."""

    def __init__(self, loader: DataLoader[Row]) -> None:
        self.loader = loader
        self.rows_read = 0
        self.seen: list[Sample] = []

    def __iter__(self) -> Iterator[WorkerBatch]:
        for batch in self.loader:
            self.rows_read += batch.rows_read
            self.seen.extend(batch.samples)
            yield batch


def _drop_stage_manager(settings: Settings) -> StageManager:
    """One long stage drawing every sample from the `drop` source."""
    stage = resolved_stage(
        "only", tokens=100 * settings.world_batch_size * settings.block_size, base_lr=1e-4, transition_pct=0.0,
        train_weights={"drop": 1.0},
    )
    return StageManager([stage], settings.world_batch_size, settings.block_size)


def _drop_stream(settings: Settings, data_dir: Path, tokenizer: Tokenizer) -> tuple[BatchStream, _RecordingLoader]:
    """A stream over one drop-heavy source (single shard, in-process, unsorted): rows are read in range order."""
    parquet = entry_dataset(DataEntry("drop", str(data_dir), data_signature=DROP_SIGNATURE))
    loader = dataloader_over(parquet, tokenizer, DROP_BLOCK_SIZE, settings.micro_batch_size, padded=False)
    recording = _RecordingLoader(loader)
    loaders = RunDataloaders({"drop": recording}, [], tokenizer, {"drop": parquet})
    return BatchStream(settings, loaders, _drop_stage_manager(settings), TrainingProgress()), recording


def _run_world_batches(settings: Settings, stream: BatchStream, world_batches: int) -> None:
    for _ in range(world_batches):
        for _ in range(settings.gradient_accumulation_steps):
            next(stream)
        stream.progress.advance()


def _sample_ids(samples: list[Sample]) -> list[tuple[int, ...]]:
    """A hashable row identity: the unpadded input tokens (unique per fixture row)."""
    return [tuple(input_ids.tolist()) for input_ids, _, _ in samples]


def test_batch_stream_counts_rows_read_not_surviving_samples(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """The stored counter advances by rows READ from disk — dropped rows included — so `state_dict` stores exactly
    what `set_resume_offset` will skip. Counting survivors instead undercounted by one row per drop (H7)."""
    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer, batch_size=2)
    _write_drop_parquet(tmp_path / "drop_data")
    stream, recording = _drop_stream(settings, tmp_path / "drop_data", stream_tokenizer)
    world_batches = 3
    _run_world_batches(settings, stream, world_batches)

    # independent oracle: a draw with an empty buffer pulls worker batches of `micro_batch_size` rows until a
    # sample is there, so after consuming j × world_batch_size samples the rows read are the smallest multiple of
    # the worker batch size whose survivors cover them — the fixture's drop pattern alone decides it
    expected_rows = 0
    for j in range(1, world_batches + 1):
        while _drop_survivors(expected_rows) < j * settings.world_batch_size:
            expected_rows += settings.micro_batch_size
    assert stream.state_dict()["consumed_rows"] == {"drop": expected_rows}
    assert recording.rows_read == expected_rows
    assert len(recording.seen) == _drop_survivors(expected_rows) < expected_rows  # counting survivors would rewind


def test_mid_stage_resume_with_dropped_rows_repeats_and_skips_nothing(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """A resume from a mid-stage checkpoint continues at exactly the next unread row also when the workers dropped
    rows: interrupted + resumed pulls are the very sample sequence of an uninterrupted run over the same data —
    nothing re-read (a repeat), nothing jumped over (a skip). Worker batches of one row (`micro_batch_size` 1)
    keep the stream's buffer empty at the checkpoint, so the counter marks exactly the next unconsumed row."""
    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer, batch_size=1)
    data_dir = tmp_path / "drop_data"
    _write_drop_parquet(data_dir)
    k = 2  # world batches before the checkpoint

    full, full_recording = _drop_stream(settings, data_dir, stream_tokenizer)
    _run_world_batches(settings, full, 2 * k)
    uninterrupted = _sample_ids(full_recording.seen)
    assert len(set(uninterrupted)) == len(uninterrupted)  # unique rows: sequence equality below implies no repeats

    first, first_recording = _drop_stream(settings, data_dir, stream_tokenizer)
    _run_world_batches(settings, first, k)
    state = first.state_dict()
    before = _sample_ids(first_recording.seen)
    assert state["consumed_rows"] == {"drop": first_recording.rows_read}
    assert before == uninterrupted[: len(before)]  # the single-shard stream is deterministic

    resumed, resumed_recording = _drop_stream(settings, data_dir, stream_tokenizer)
    resumed.load_state_dict(state)
    _run_world_batches(settings, resumed, k)
    combined = before + _sample_ids(resumed_recording.seen)

    overlap = min(len(combined), len(uninterrupted))
    assert overlap >= 2 * k * settings.world_batch_size  # covers every trained sample of both runs
    assert combined[:overlap] == uninterrupted[:overlap]


def test_mid_stage_resume_with_buffered_samples_repeats_nothing(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """With worker batches of several rows, samples can sit in the stream's buffer at the checkpoint; they were
    already counted as read, so a resume may skip them — but it never repeats a row the first stream pulled."""
    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer, batch_size=2)
    data_dir = tmp_path / "drop_data"
    _write_drop_parquet(data_dir)

    first, first_recording = _drop_stream(settings, data_dir, stream_tokenizer)
    _run_world_batches(settings, first, 2)
    state = first.state_dict()
    before = _sample_ids(first_recording.seen)
    assert state["consumed_rows"] == {"drop": first_recording.rows_read}  # pull-time accounting, buffers included

    resumed, resumed_recording = _drop_stream(settings, data_dir, stream_tokenizer)
    resumed.load_state_dict(state)
    _run_world_batches(settings, resumed, 2)
    after = _sample_ids(resumed_recording.seen)
    assert not set(before) & set(after)  # no repeats: the resumed range starts after every row read before


# --------------------------------------------------------------------------------------------------------------
# dataset-independent numerics reference: golden_tiny_steps.json


def step_reference_metrics() -> dict[str, Any]:
    """Five optimizer steps of scripted batches through `run_one_optimizer_step` on the tiny model, reduced to their
    numerics: `{"steps": {"<step>": {loss, grad_norm, lr}}, "parameter_norms": {name: L2 norm after step 4}}`.

    fp32 on the CPU, `SingleDeviceBackend("cpu", "32")`, one thread and deterministic algorithms; the global torch
    RNG is seeded with 0 before the model is built (the parameter init and the recurrence draws of every forward
    consume it), the batches come from their own generator (`scripted_batches`). Nothing here reads a dataset, a
    tokenizer or a loader, so a change of the data pipeline never re-records this fixture.
    """
    settings = reference_settings()
    backend = SingleDeviceBackend(device="cpu", precision="32")
    with single_thread_deterministic():
        model = fresh_tiny_model(backend, seed=0)
        optimizer = fresh_optimizer(settings, model, backend)
        results = run_steps(settings, backend, model, optimizer, steps=REFERENCE_STEPS)
    return {
        "steps": {
            str(result.step): {
                "loss": float(result.loss),
                "grad_norm": float(result.grad_norm),
                "lr": float(result.learning_rate),
            }
            for result in results
        },
        "parameter_norms": {
            name: float(torch.linalg.vector_norm(tensor.float())) for name, tensor in model.state_dict().items()
        },
    }


def record_step_reference() -> Path:
    """Re-record `training/golden_tiny_steps.json`. ONLY do this in a commit whose purpose is a numerics change of
    the optimizer step (or of the tiny architecture / the optimizer):

        uv run python -c "from training.test_step import record_step_reference; record_step_reference()"

    Recorded with torch 2.13.0+cu130 on the author's machine (CPU, fp32, one thread, deterministic algorithms), in
    the commit that extracted `run_one_optimizer_step` — the tiny golden run passing unchanged in that same commit is
    what ties this reference to the thesis loop.
    """
    GOLDEN_STEPS_PATH.write_text(golden_run_json(step_reference_metrics()))
    return GOLDEN_STEPS_PATH


def test_golden_tiny_steps() -> None:
    """Numerics regression guard for `run_one_optimizer_step`, independent of the data pipeline: five steps of
    scripted batches reproduce `golden_tiny_steps.json`.

    Same semantics as `test_golden_tiny_run`: floats with `rel=1e-5`, `GOLDEN_EXACT=1` compares with `==`
    (bit-identical on the recording machine), learning rates always exact; fp32 CPU, one thread, deterministic
    algorithms; the bf16 autocast path is not exercised. Re-record only in a numerics commit; loosen rather than
    chase float-order differences on another machine.
    """
    assert GOLDEN_STEPS_PATH.exists(), "step reference missing; record it with record_step_reference() in a numerics commit"
    expected = json.loads(GOLDEN_STEPS_PATH.read_text())
    actual = step_reference_metrics()
    assert sorted(actual["steps"], key=int) == [str(s) for s in range(REFERENCE_STEPS)]
    assert actual["steps"]["0"]["lr"] == 0.0 and actual["steps"]["2"]["lr"] == 3e-4
    exact = golden_exact_requested()
    mismatches = golden_mismatches(expected, json.loads(golden_run_json(actual)), exact=exact)
    assert not mismatches, "step reference changed:\n" + "\n".join(mismatches)
