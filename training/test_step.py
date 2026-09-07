# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for one optimizer step (`training.step`): the progress counter, the micro-batch stream, the LR, the
accumulation arithmetic, the skipped first update, the non-finite checks, and the dataset-independent numerics
reference `training/golden_tiny_steps.json` (five steps of scripted packs through `run_one_optimizer_step`).
"""

import copy
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from torch.nn.attention.flex_attention import BlockMask

from model import RecurrentGPT, build_model
from training.backend.single_device import SingleDeviceBackend
from training.data.collate import Sample, WorkerBatch
from training.data.tokenizer import IGNORE_INDEX
from training.data.packing import POOL_TOKEN_FACTOR, PackedBatch, shifted_length
import training.data.loader as loader_module
from training.data.loader import RunDataloaders, SampleBatch, build_run_dataloaders, dataloader_over, entry_dataset
from training.data.dataset_resolver import DataEntry, ResolvedDataset, resolve_dataset
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
    MAX_CONSECUTIVE_REJECTS,
    StepResult,
    TrainingProgress,
    model_inputs,
    NonFiniteLossError,
    run_one_optimizer_step,
    scheduled_learning_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL_ARCHITECTURE = REPO_ROOT / "config" / "model_architecture" / "tiny.yaml"
GOLDEN_STEPS_PATH = Path(__file__).resolve().parent / "golden_tiny_steps.json"

REFERENCE_STEPS = 5
PACK_LENGTH = 256  # = the tiny training_max_sequence_length, the smallest pack the settings allow
PACKED_MICRO_BATCHES_PER_STEP = 4  # 4 x 256 = 1024, the tiny run's tokens per step, so `_abc_stage_manager` applies unchanged


# --------------------------------------------------------------------------------------------------------------
# a run without a dataset: in-memory settings, a one-stage manager, scripted packed batches


def reference_settings(**overrides: Any) -> Settings:
    """
    Settings of the step reference: the tiny architecture, two packs of `PACK_LENGTH` tokens per optimizer step,
    the thesis optimizer (ELLISAdam with the `crow_300m_final.yaml` options), warmup 2 / cooldown 2, gradient
    metrics at every step. The dataset config is never read (no loader is built).
    """

    values: dict[str, Any] = dict(
        dataset_config="config/datasets/tiny.yaml",
        model_architecture_config=str(TINY_MODEL_ARCHITECTURE),
        stage_base_lrs=[3e-4],
        run_name="steps",
        out_dir="unused",
        seed=0,
        training_max_sequence_length=256,
        tokens_per_micro_batch=PACK_LENGTH,
        micro_batches_per_step=2,
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
    """
    One stage of `steps` optimizer steps (no transition): LR 0 at step 0, 1.5e-4 at step 1, 3e-4 from step 2.
    """

    stage = resolved_stage("only", tokens=steps * settings.tokens_per_optimizer_step, base_lr=3e-4, transition_pct=0.0)
    return StageManager([stage], settings.tokens_per_optimizer_step, warmup_steps=2, cooldown_steps=2)


def scripted_batches(settings: Settings, seed: int = 0) -> Iterator[PackedBatch]:
    """
    Endless packed micro-batches of random ids (vocab 512) from a seeded generator, independent of the global torch
    RNG, the tokenizer and the data pipeline: three documents of L/2, L/4 and L/8 positions and a tail of L/8, laid
    out as `pack_samples` lays them out, labels random too (already shifted), the tail ignored.
    """

    pack_length = settings.tokens_per_micro_batch
    assert pack_length is not None
    lengths = [pack_length // 2, pack_length // 4, pack_length // 8]
    tail = pack_length - sum(lengths)
    generator = torch.Generator().manual_seed(seed)
    while True:
        input_ids = torch.full((1, pack_length), 2, dtype=torch.long)
        labels = torch.full((1, pack_length), IGNORE_INDEX, dtype=torch.long)
        position_ids = torch.zeros((1, pack_length), dtype=torch.long)
        document_ids = torch.full((1, pack_length), len(lengths), dtype=torch.int32)
        offset = 0
        for document, length in enumerate(lengths):
            input_ids[0, offset : offset + length] = torch.randint(1, 512, (length,), generator=generator)
            labels[0, offset : offset + length] = torch.randint(1, 512, (length,), generator=generator)
            position_ids[0, offset : offset + length] = torch.arange(length)
            document_ids[0, offset : offset + length] = document
            offset += length
        position_ids[0, offset:] = torch.arange(tail)
        yield PackedBatch(input_ids, labels, ["scripted"] * len(lengths), position_ids, document_ids, tail, lengths)


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
    settings: Settings,
    backend: SingleDeviceBackend,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: int,
) -> list[StepResult]:
    """
    `steps` consecutive optimizer steps of scripted batches, advancing a fresh progress counter as `train()` does.
    """

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


def test_scheduled_learning_rate_follows_warmup_and_cooldown(settings: Settings) -> None:
    stage_manager = reference_stage_manager(settings)  # 10 steps, warmup 2, cooldown 2, base LR 3e-4
    lrs = [scheduled_learning_rate(settings, stage_manager, TrainingProgress(step=s)) for s in range(10)]
    assert lrs[:5] == pytest.approx([0.0, 1.5e-4, 3e-4, 3e-4, 3e-4])
    assert lrs[8] == pytest.approx(3e-4) and lrs[9] == pytest.approx(1.5e-4)  # cooldown over the last 2 steps
    resumed = TrainingProgress(step=4, resume_step=4)
    assert scheduled_learning_rate(settings, stage_manager, resumed) == pytest.approx(3e-4)  # a resume changes nothing


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
    """
    The very first update is skipped (as in the thesis); step 1 updates every parameter with a gradient.
    """

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
    settings: Settings,
    backend: SingleDeviceBackend,
    model: torch.nn.Module,
    batches: Iterator[PackedBatch],
    step: int,
    seed: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    The accumulation of one step done by hand on a copy of `model`: per-micro-batch losses and the pre-clip norm of
    the accumulated gradients. Seeds the global RNG like the caller so the recurrence draws match.
    """

    copy_of_model = copy.deepcopy(model)
    assert isinstance(copy_of_model, RecurrentGPT)
    copy_of_model.step = step
    torch.manual_seed(seed)
    losses = []
    for _ in range(settings.gradient_accumulation_steps):
        loss = copy_of_model(**model_inputs(next(batches), backend))["loss"]
        assert loss is not None
        (loss / settings.gradient_accumulation_steps).backward()
        losses.append(loss.detach())
    grads = [p.grad for p in copy_of_model.parameters() if p.grad is not None]
    norm = torch.stack([g.norm() for g in grads]).norm()
    return losses, norm


def test_loss_and_grad_norm_match_a_hand_computation(cpu_backend: SingleDeviceBackend) -> None:
    """
    `loss` is the mean of the per-micro-batch losses; `grad_norm` is the pre-clip L2 norm of the accumulated
    gradients (the clip threshold 0.5 is below it here, so it is not the clipped norm).
    """

    settings = reference_settings(grad_clip=0.5)
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    expected_losses, expected_norm = _hand_accumulation(
        settings, cpu_backend, model, scripted_batches(settings), step=0, seed=11
    )

    torch.manual_seed(11)
    result = run_one_optimizer_step(
        settings,
        cpu_backend,
        model,
        optimizer,
        reference_stage_manager(settings),
        scripted_batches(settings),
        TrainingProgress(),
    )
    assert len(expected_losses) == settings.gradient_accumulation_steps == 2
    assert result.loss.item() == pytest.approx(torch.stack(expected_losses).mean().item(), rel=1e-6)
    assert result.log_ppl.item() == pytest.approx(result.loss.item(), rel=1e-6)  # log_ppl is the detached loss
    assert result.grad_norm.item() == pytest.approx(expected_norm.item(), rel=1e-5)
    assert result.grad_norm.item() > settings.grad_clip  # so the clipping actually happened and the norm is pre-clip


def test_stage_infos_and_metrics(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    """
    `stage` is the manager's info at `step`; gradient metrics only at log steps.
    """

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
    assert str(excinfo.value) == "Loss is nan at step 0" and isinstance(excinfo.value, NonFiniteLossError)


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
    assert str(excinfo.value) == "Gradient norm is non-finite at step 3" and isinstance(excinfo.value, NonFiniteLossError)


# --------------------------------------------------------------------------------------------------------------
# micro-batch stream (per-document source picks by token deficit over the run-wide readers; fake loaders, hand-made
# stage managers)


def _fake_samples(tag: str, lengths: list[int]) -> SampleBatch:
    """
    One worker batch: one unpadded sample per entry of `lengths`, every position supervised.
    """

    return [(torch.full((n,), 3, dtype=torch.long), torch.full((n,), 3, dtype=torch.long), tag) for n in lengths]


class _Repeat:
    """
    Endless iterable of one tagged one-sample worker batch with a running counter as the sample length.
    """

    def __init__(self, tag: str, batch_size: int = 1) -> None:
        self.tag, self.batch_size, self.count = tag, batch_size, 0

    def __iter__(self) -> Iterator[WorkerBatch]:
        while True:
            self.count += 1
            samples = _fake_samples(self.tag, [1 + (self.count + i) % 7 for i in range(self.batch_size)])
            yield WorkerBatch(samples, len(samples))


class _Fixed:
    """
    Endless iterable of one-sample worker batches of `tag` whose documents all occupy `slots` positions (`slots + 1`
    tokens): the fake for the token-share tests, where the document lengths must be known.
    """

    def __init__(self, tag: str, slots: int) -> None:
        self.tag, self.slots, self.count = tag, slots, 0

    def __iter__(self) -> Iterator[WorkerBatch]:
        while True:
            self.count += 1
            yield WorkerBatch(_fake_samples(self.tag, [self.slots + 1]), 1)


class _ShortBatches:
    """
    A loader whose worker batches cycle through `sizes` rows; a short (or empty) batch is what a loader running
    out of rows, or a batch whose rows were all dropped for lack of a supervised label, hands the stream.
    """

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
    """
    The synthetic tokenizer the stream pads its assembled micro-batches with.
    """

    return Tokenizer(tiny_tokenizer_dir)


def _abc_stage_manager(settings: Settings) -> StageManager:
    """
    The tiny stage boundaries ((0,8,6,8), (8,16,14,16), (16,20)) with one fake source per stage: `a` in stage 0,
    `b` in stage 1, `c` in stage 2, hand-made so no dataset is resolved for the stream tests.
    """

    stages = [
        resolved_stage("s0", tokens=8192, base_lr=3e-4, transition_pct=0.25, train_weights={"a": 1.0}),
        resolved_stage("s1", tokens=8192, base_lr=1e-4, transition_pct=0.25, train_weights={"b": 1.0}),
        resolved_stage("s2", tokens=4096, base_lr=5e-5, transition_pct=0.0, train_weights={"c": 1.0}),
    ]
    return StageManager(stages, settings.tokens_per_optimizer_step)


def _stream_setup(
    tmp_path: Path, tiny_dataset_dir: Path, tokenizer: Tokenizer, batch_size: int = 1, **overrides: Any
) -> tuple[Settings, RunDataloaders, StageManager]:
    """
    Packed tiny settings (packs of `PACK_LENGTH`, `PACKED_MICRO_BATCHES_PER_STEP` per step, `overrides` on top)
    and fake loaders whose worker batches hold `batch_size` samples.
    """

    packing = {"tokens_per_micro_batch": PACK_LENGTH, "micro_batches_per_step": PACKED_MICRO_BATCHES_PER_STEP}
    yaml_path = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", **(packing | overrides))
    settings = parse_settings(["--config", str(yaml_path)])
    loaders = RunDataloaders({t: _Repeat(t, batch_size) for t in "abc"}, [], tokenizer, {})
    return settings, loaders, _abc_stage_manager(settings)


def _weighted_stage_manager(settings: Settings, *stage_weights: dict[str, float]) -> StageManager:
    """
    One stage of 8 optimizer steps per entry of `stage_weights` (its train weights), no transition windows: the
    weights switch at the stage boundary, the case the no-burst rule is easiest to read on.
    """

    stages = [
        resolved_stage(f"s{i}", tokens=8 * settings.tokens_per_optimizer_step, base_lr=1e-4, transition_pct=0.0, train_weights=weights)
        for i, weights in enumerate(stage_weights)
    ]
    return StageManager(stages, settings.tokens_per_optimizer_step)


def _loaded_tokens(packs: list[PackedBatch]) -> dict[str, int]:
    """
    The document slots per source over `packs` (`data_tokens`, so pack tails are left out).
    """

    out: dict[str, int] = {}
    for pack in packs:
        for data_id, tokens in zip(pack.data_ids, pack.data_tokens):
            out[data_id] = out.get(data_id, 0) + tokens
    return out


def _next_pack(stream: Iterator[PackedBatch]) -> PackedBatch:
    batch = next(stream)
    assert isinstance(batch, PackedBatch)
    return batch


def _packs(stream: Iterator[PackedBatch], n: int, skip: int = 0) -> list[PackedBatch]:
    """
    The next `n` packs, after `skip` packs left to the pool's lag: a document is put into the pool up to
    `POOL_TOKEN_FACTOR` pack lengths ahead of the pack it lands in (`BatchStream`), so the first packs after a step
    change still hold documents picked under the previous step's weights.
    """

    for _ in range(skip):
        next(stream)
    return [_next_pack(stream) for _ in range(n)]  # never pull an extra element (zip would)


def _micro_batches(settings: Settings, stream: BatchStream, world_batches: int) -> list[PackedBatch]:
    """
    What the model would see: every micro-batch of `world_batches` optimizer steps, the step advanced in between.
    """

    out: list[PackedBatch] = []
    for _ in range(world_batches):
        out += _packs(stream, settings.gradient_accumulation_steps)
        stream.progress.advance()
    return out


def _same_batches(a: list[PackedBatch], b: list[PackedBatch]) -> bool:
    return len(a) == len(b) and all(
        x.data_ids == y.data_ids
        and x.padding_tokens == y.padding_tokens
        and all(torch.equal(getattr(x, name), getattr(y, name)) for name in ("input_ids", "labels", "position_ids", "document_ids"))
        for x, y in zip(a, b)
    )


def _sources(packs: list[PackedBatch]) -> list[set[str]]:
    return [set(pack.data_ids) for pack in packs]


def _document_slots(batch: PackedBatch) -> list[int]:
    """
    Positions per document of a pack, in pack order (0 for a one-token document, which occupies no position).
    """

    documents = len(batch.data_ids)
    return torch.bincount(batch.document_ids[0].long(), minlength=documents + 1)[:documents].tolist()


def _documents(batch: PackedBatch) -> list[tuple[int, ...]]:
    """
    A row identity per document of a pack: its first 20 input tokens (unique per tiny fixture row).
    """

    ids, document_ids = batch.input_ids[0], batch.document_ids[0]
    return [tuple(ids[document_ids == d][:20].tolist()) for d in range(len(batch.data_ids))]


def _drawn(state: dict[str, Any], packs: list[PackedBatch], source: str) -> int:
    """
    Every sample a stream pulled from `source` is in a pack, in the pool or in the buffer (the fakes drop no rows),
    so this equals the source's rows read.
    """

    return sum(len(pack.data_ids) for pack in packs) + len(state["pool"]) + len(state["buffers"].get(source, []))


def test_batch_stream_samples_by_transition_progress(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    The step is read at every pull, not when the stream is created, and the documents of a pack come from the
    sources weighted at the step of their pick (up to sixteen packs ahead, `_packs`): a plain step pulls from its
    stage's source only, the start of a transition still from the old stage, its middle from both, at equal TOKEN
    shares (deterministic: the deficit rule alternates the sources within one document length).
    """

    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    progress.step = 3  # plain stage 0, set after the stream was created
    assert _sources(_packs(stream, 4)) == [{"a"}] * 4
    progress.step = 6  # transition 0 -> 1, progress 0: everything still from stage 0
    assert _sources(_packs(stream, 4)) == [{"a"}] * 4
    progress.step = 8  # fully in stage 1
    assert _sources(_packs(stream, 4, skip=POOL_TOKEN_FACTOR)) == [{"b"}] * 4
    progress.step = 15  # transition 1 -> 2 at progress 0.5: half the tokens from each
    mixed = _loaded_tokens(_packs(stream, 8, skip=POOL_TOKEN_FACTOR))
    assert set(mixed) == {"b", "c"} and 0.4 < mixed["c"] / (mixed["b"] + mixed["c"]) < 0.6
    progress.step = 19
    assert _sources(_packs(stream, 4, skip=POOL_TOKEN_FACTOR)) == [{"c"}] * 4


def test_batch_stream_is_deterministic(tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer) -> None:
    """
    Two streams built the same way yield identical packs: the source of every document is a function of the two
    per-source numbers, no RNG is involved (so the start step plays no part either).
    """

    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)

    def packs(start_step: int) -> list[PackedBatch]:
        loaders = RunDataloaders({tag: _Repeat(tag) for tag in "abc"}, [], stream_tokenizer, {})
        progress = TrainingProgress(step=start_step)
        stream = BatchStream(settings, loaders, stage_manager, progress)
        progress.step = 15
        return _packs(stream, 4)

    assert _same_batches(packs(0), packs(0))
    assert _same_batches(packs(0), packs(14))


def test_batch_stream_balances_the_token_shares_within_one_document(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    The rule the weights are token shares by: two sources at 0.5 / 0.5 whose documents are 2000 and 300 slots long
    have loaded the same number of tokens to within one document (the longest) after 50 packs, and the packs
    trained on hold the same share; a per-document draw would give `a` about 6.7 times the tokens of `b`.
    """

    settings, _, _ = _stream_setup(
        tmp_path, tiny_dataset_dir, stream_tokenizer, tokens_per_micro_batch=4096, micro_batches_per_step=1
    )
    stage_manager = _weighted_stage_manager(settings, {"a": 0.5, "b": 0.5})
    loaders = RunDataloaders({"a": _Fixed("a", 2000), "b": _Fixed("b", 300)}, [], stream_tokenizer, {})
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    packs = _packs(stream, 50)
    loaded = stream.state_dict()["pool_loaded"]
    assert loaded["a"] > 20 * 4096 and abs(loaded["a"] - loaded["b"]) <= 2000
    trained = _loaded_tokens(packs)
    assert 0.45 < trained["b"] / (trained["a"] + trained["b"]) < 0.55


def test_batch_stream_gives_a_source_entering_at_a_transition_its_share_without_a_burst(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Targets accumulate with the CURRENT weights: a source whose weight goes from 0 to 0.3 at a stage boundary earns
    30 % of the tokens loaded from then on, not 30 % of everything loaded before (which would fill the pool with it
    alone for many packs). Measured on the loaded numbers over the first 20 packs after the switch, while the
    trained packs still show the old stage (the pool's lag).
    """

    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stage_manager = _weighted_stage_manager(settings, {"a": 1.0}, {"a": 0.7, "b": 0.3})
    loaders = RunDataloaders({"a": _Fixed("a", 60), "b": _Fixed("b", 30)}, [], stream_tokenizer, {})
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    assert _sources(_packs(stream, 30)) == [{"a"}] * 30
    before = stream.state_dict()["pool_loaded"]
    assert before["b"] == 0 and before["a"] > 40 * PACK_LENGTH
    progress.step = 8  # stage 1: b enters at 0.3
    after_switch = _packs(stream, 20)
    after = stream.state_dict()["pool_loaded"]
    loaded_a, loaded_b = after["a"] - before["a"], after["b"] - before["b"]
    assert 0.2 < loaded_b / (loaded_a + loaded_b) < 0.4
    assert _sources(after_switch[: POOL_TOKEN_FACTOR - 1]) == [{"a"}] * (POOL_TOKEN_FACTOR - 1), "the lag"
    assert "b" in _loaded_tokens(after_switch)


def test_batch_stream_ties_go_to_the_alphabetically_first_source(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Equal deficits (the start of a run, and after every pair of equal documents at 0.5 / 0.5) pick the
    alphabetically smallest source name, NOT the first in `train_sources` (config) order: with the loaders
    deliberately ordered `b, a` the pool still alternates a, b, a, b, so the order the dataset config lists its
    sources in cannot change the stream.
    """

    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stage_manager = _weighted_stage_manager(settings, {"a": 0.5, "b": 0.5})
    loaders = RunDataloaders({"b": _Fixed("b", 32), "a": _Fixed("a", 32)}, [], stream_tokenizer, {})
    assert loaders.train_sources == ["b", "a"]
    pack = _next_pack(BatchStream(settings, loaders, stage_manager, TrainingProgress()))
    assert pack.data_ids == ["a", "b"] * 4 and pack.data_tokens == [32] * 8


def test_batch_stream_is_the_same_for_any_order_of_the_sources(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Two configs listing the same sources in a different order yield the same stream. `b` and `c` tie after every
    `a` document here (equal weights, equal lengths), so under the old config-order tie-break the two streams
    differ in the first pack; the resume contract (`check_dataset_unchanged` hashes the config with sorted keys,
    so it cannot see a reordered `sources:` block) needs them not to.
    """

    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}

    def packs(order: str) -> list[PackedBatch]:
        loaders = RunDataloaders({tag: _Fixed(tag, 32) for tag in order}, [], stream_tokenizer, {})
        assert loaders.train_sources == list(order)
        return _packs(BatchStream(settings, loaders, _weighted_stage_manager(settings, weights), TrainingProgress()), 4)

    forward, backward = packs("abc"), packs("cba")
    assert [pack.data_ids for pack in forward] == [pack.data_ids for pack in backward]
    assert _same_batches(forward, backward)
    assert forward[0].data_ids[:3] == ["a", "b", "c"], "the tie between b and c goes to b, the smaller name"


def test_token_shares_of_three_sources_follow_the_weights(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    The rule with more than two sources, where the wrong rules stop coinciding: three sources at 0.5 / 0.3 / 0.2
    whose documents are 2000, 300 and 50 slots long load their configured share to within 2 % over 200 packs.

    Two sources whose weights sum to 1 are the degenerate case - growing only the PICKED source's target balances
    `n_s (1 - w_s) len_s`, which for k = 2 is the right answer and for k = 3 gives .428 / .305 / .267. Picking the
    SMALLEST deficit instead of the largest lets one source take almost everything. Both need k >= 3 and unequal
    document lengths to be visible, which is what this test is.
    """

    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    settings, _, _ = _stream_setup(
        tmp_path, tiny_dataset_dir, stream_tokenizer, tokens_per_micro_batch=4096, micro_batches_per_step=1
    )
    lengths = {"a": 2000, "b": 300, "c": 50}
    loaders = RunDataloaders({tag: _Fixed(tag, n) for tag, n in lengths.items()}, [], stream_tokenizer, {})
    stream = BatchStream(settings, loaders, _weighted_stage_manager(settings, weights), TrainingProgress())
    trained = _loaded_tokens(_packs(stream, 200))

    loaded = stream.state_dict()["pool_loaded"]
    total = sum(loaded.values())
    assert total > 200 * 4000
    for source, weight in weights.items():
        assert loaded[source] / total == pytest.approx(weight, abs=0.02), source
        assert trained[source] / sum(trained.values()) == pytest.approx(weight, abs=0.02), source


@pytest.mark.timeout(60)  # the bug this pins is a hang: without the guard the refill loop never returns
def test_batch_stream_fails_loudly_on_documents_that_never_fit(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A rejected document moves neither number, so the next pick is the same source again: a source whose documents
    are all longer than the pack used to spin the refill loop forever (a million rows in five seconds, no error, no
    step). After `MAX_CONSECUTIVE_REJECTS` in a row the run fails, naming the source and both lengths. The settings
    make this unreachable (`tokens_per_micro_batch >= training_max_sequence_length`); a hang would not be.
    """

    settings, _, _ = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stage_manager = _weighted_stage_manager(settings, {"big": 0.5, "ok": 0.5})
    loaders = RunDataloaders(
        {"big": _Fixed("big", PACK_LENGTH + 1), "ok": _Fixed("ok", 32)}, [], stream_tokenizer, {}
    )
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    with caplog.at_level(logging.WARNING, logger="training.data.packing"), pytest.raises(
        RuntimeError, match=rf"rejected {MAX_CONSECUTIVE_REJECTS} documents in a row"
    ) as info:
        next(stream)
    message = str(info.value)
    assert "'big'" in message and f"{PACK_LENGTH + 1} slots" in message and f"pack holds {PACK_LENGTH}" in message
    assert stream.state_dict()["pool_loaded"] == {"big": 0, "ok": 0}, "a rejected document is never accounted"
    assert caplog.text.count("Dropping a") == MAX_CONSECUTIVE_REJECTS


def test_batch_stream_fills_every_pack_from_short_worker_batches(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Regression (T-M3/T-M4): a short or empty worker batch (a loader reaching its last rows, or a batch whose rows
    were all dropped) used to shrink the world batch and permanently misalign it with the optimizer steps. The
    stream pulls a source's loader until its buffer holds a sample, so every pack is full to within one document
    whatever the worker batches hold.
    """

    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer, batch_size=2)
    loaders = RunDataloaders({tag: _ShortBatches(tag, [2, 0, 1, 2, 3]) for tag in "abc"}, [], stream_tokenizer, {})
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    for _ in range(6):
        for pack in _packs(stream, settings.gradient_accumulation_steps):
            assert sum(_document_slots(pack)) + pack.padding_tokens == PACK_LENGTH and pack.padding_tokens <= 6
        progress.advance()


# --------------------------------------------------------------------------------------------------------------
# BatchStream state: what a checkpoint stores and what a resume does with it


def test_batch_stream_state_carries_the_buffered_samples(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Samples pulled from a loader but not yet drawn into the pool are part of the state: a resumed stream serves
    them first instead of skipping them, so a checkpoint loses no rows. Rows read = documents in packs + pool +
    buffer (`_drawn`) before and after the resume.
    """

    worker_batch = 5
    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    loaders = RunDataloaders({tag: _ShortBatches(tag, [worker_batch]) for tag in "abc"}, [], stream_tokenizer, {})
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    pack = _next_pack(stream)  # stage 0: `a` documents, the last worker batch pulled only partly drawn
    state = stream.state_dict()
    buffered = state["buffers"]["a"]
    assert 0 < len(buffered) < worker_batch and state["consumed_rows"]["a"] % worker_batch == 0
    assert _drawn(state, [pack], "a") == state["consumed_rows"]["a"]

    resumed = BatchStream(
        settings,
        RunDataloaders({tag: _ShortBatches(tag, [worker_batch]) for tag in "abc"}, [], stream_tokenizer, {}),
        stage_manager,
        TrainingProgress(),
    )
    resumed.load_state_dict(state)
    restored = resumed.state_dict()["buffers"]["a"]
    assert [(ids.tolist(), labels.tolist(), tag) for ids, labels, tag in restored] == [
        (ids.tolist(), labels.tolist(), tag) for ids, labels, tag in buffered
    ]
    next_pack = _next_pack(resumed)  # the buffered samples enter the pool before a new worker batch is pulled
    after = resumed.state_dict()
    pulled = after["consumed_rows"]["a"] - state["consumed_rows"]["a"]
    assert pulled % worker_batch == 0
    assert _drawn(after, [next_pack], "a") == len(state["pool"]) + len(buffered) + pulled


@contextmanager
def _run_loaders(settings: Settings, dataset: ResolvedDataset, backend: SingleDeviceBackend) -> Iterator[RunDataloaders]:
    """
    The run's real loaders (one worker process per source), shut down on exit: a leaked iterator keeps its worker
    alive, at hundreds of MiB, until the GC finds the reference cycle, long after the test.
    """

    loaders = build_run_dataloaders(settings, dataset, backend)
    try:
        yield loaders
    finally:
        loaders.close()


def test_batch_stream_load_state_dict_sets_the_loader_offsets(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    The counters become a per-source row offset (modulo the range), which is what makes a resume skip the rows
    the interrupted run consumed.
    """

    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.tokens_per_optimizer_step)
    with _run_loaders(settings, dataset, cpu_backend) as loaders:
        stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
        parquet = loaders.datasets["synthetic_pretrain"]
        state = {
            "consumed_rows": {parquet.prefix: parquet.num_rows + 5},
            "pool_loaded": {},
            "pool_target": {},
            "buffers": {},
            "pool": [],
        }
        stream.load_state_dict(state)
        assert loaders.pending_offsets == {"synthetic_pretrain": parquet.num_rows + 5}
        next(stream)  # stage 0 draws from the pretrain source only: its reader starts now
        assert parquet.resume_offset == 5 and loaders.pending_offsets == {}  # wrapped around one epoch
        assert loaders.datasets["synthetic_instruct"].resume_offset == 0  # untouched sources stay at the start


def test_batch_stream_resume_does_not_repeat_rows(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    The point of checkpointing the stream: a stream restored from a state continues into rows the first one had
    not reached (the pooled documents first), while a stream that only restarts the loaders serves the very same
    rows again.
    """

    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.tokens_per_optimizer_step)

    def documents(stream: BatchStream, world_batches: int) -> list[tuple[int, ...]]:
        """
        The documents of the packs of `world_batches` steps (`_documents`); only plain stage-0 steps, so no
        transition draw is involved.
        """

        return [document for pack in _micro_batches(settings, stream, world_batches) for document in _documents(pack)]

    # one stream (and its worker processes) at a time
    with _run_loaders(settings, dataset, cpu_backend) as loaders:
        stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
        before = documents(stream, 3)
        state = stream.state_dict()
    assert len(set(before)) == len(before) > 3 * settings.gradient_accumulation_steps

    with _run_loaders(settings, dataset, cpu_backend) as loaders:
        resumed = BatchStream(settings, loaders, stage_manager, TrainingProgress())
        resumed.load_state_dict(state)
        assert not set(before) & set(documents(resumed, 3))
    with _run_loaders(settings, dataset, cpu_backend) as loaders:  # without the state the rows are read from the top again
        assert documents(BatchStream(settings, loaders, stage_manager, TrainingProgress()), 3) == before


def test_stages_sharing_a_source_do_not_re_read_rows(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    The bug the continuous stream fixes: the old per-stage loaders each read `synthetic_pretrain` from the top,
    so stage 1 re-served the very rows stage 0 had trained on. The run-wide reader continues across the boundary:
    every document up to there and beyond is a distinct row (until the source genuinely wraps around).
    """

    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    # two stages of two steps each on the pretrain source, no transition: the boundary at step 2 is crossed well
    # before the 82 pretrain rows wrap (the pool alone reads sixteen pack lengths, about 56 rows, ahead)
    stages = [
        resolved_stage(name, tokens=2 * settings.tokens_per_optimizer_step, base_lr=1e-4, transition_pct=0.0, train_weights={"synthetic_pretrain": 1.0})
        for name in ("first", "second")
    ]
    stage_manager = StageManager(stages, settings.tokens_per_optimizer_step)
    steps = 4  # both stages
    seen: list[tuple[int, ...]] = []
    with _run_loaders(settings, dataset, cpu_backend) as loaders:
        stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
        pretrain = loaders.datasets["synthetic_pretrain"]
        for pack in _micro_batches(settings, stream, steps):
            assert set(pack.data_ids) == {"synthetic_pretrain"}  # both stages train on the same source
            seen += _documents(pack)
        assert stream.consumed_rows["synthetic_pretrain"] <= pretrain.num_rows, "fixture too small to distinguish from a wrap"
    assert len(set(seen)) == len(seen)  # crossing the stage boundary at step 2 repeated nothing


def test_batch_stream_same_seed_yields_the_same_stream(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend
) -> None:
    """
    Determinism of the whole stream: two fresh streams over the same dataset and settings yield identical packs:
    same source picks, same rows, same layout (the picks are a function of the loaded and target numbers, and each
    source's reader walks its range in order).
    """

    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.tokens_per_optimizer_step)

    def packs(world_batches: int) -> list[PackedBatch]:
        with _run_loaders(settings, dataset, cpu_backend) as loaders:
            return _micro_batches(settings, BatchStream(settings, loaders, stage_manager, TrainingProgress()), world_batches)

    first, second = packs(3), packs(3)
    assert len(first) == len(second) == 3 * settings.gradient_accumulation_steps
    assert _same_batches(first, second)


# --------------------------------------------------------------------------------------------------------------
# BatchStream state when the workers DROP rows (H7): the counter is rows READ, the unit the resume skips

DROP_SIGNATURE: dict[str, Any] = {
    "keys": ["instruction", "input", "output"],
    "format_fn": "concatenate_instruction_input_output",
}
DROP_BLOCK_SIZE = 16  # cap 17 tokens: a 30-word prompt alone fills the window, its row keeps no supervised label
DROP_EVERY = 3  # every third row of the fixture is such a prompt-only row and is dropped in the worker
DROP_ROWS = 300  # rows of the fixture, 200 of them surviving; the pool reads about 96 rows ahead (`_drop_settings`)
DROP_PACK_LENGTH = 32  # a few such documents per pack, so the fixture lasts many steps (`_drop_settings`)
DROP_WORKER_ROWS = 2  # rows per worker batch of the drop loaders unless a test says otherwise


def _write_drop_parquet(directory: Path, rows: int = DROP_ROWS) -> None:
    """
    `rows` unique instruct rows of which every `DROP_EVERY`-th tokenizes to nothing at `DROP_BLOCK_SIZE`.
    """

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
    """
    Survivors among the first `rows_read` fixture rows: all but the `i % DROP_EVERY == 0` ones.
    """

    return rows_read - (rows_read + DROP_EVERY - 1) // DROP_EVERY


def _survivors_read(rows_read: int) -> int:
    """
    `_drop_survivors` across epoch wraps: the whole fixture per full pass, then its drop pattern again from row 0.
    """

    return (rows_read // DROP_ROWS) * _drop_survivors(DROP_ROWS) + _drop_survivors(rows_read % DROP_ROWS)


class _RecordingLoader:
    """
    Forwards a real (unpadded) train DataLoader's `WorkerBatch`es while recording every sample and row count that
    passed through.
    """

    def __init__(self, loader: DataLoader[Row]) -> None:
        self.loader = loader
        self.rows_read = 0
        self.seen: list[Sample] = []

    def __iter__(self) -> Iterator[WorkerBatch]:
        for batch in self.loader:
            self.rows_read += batch.rows_read
            self.seen.extend(batch.samples)
            yield batch


def _drop_settings(tmp_path: Path, tiny_dataset_dir: Path, tokenizer: Tokenizer) -> Settings:
    """
    `_stream_setup` settings for the drop fixture: documents of up to `DROP_BLOCK_SIZE` positions in packs of
    `DROP_PACK_LENGTH`, four per step, so a step takes about 16 of the 200 surviving rows and the pool of sixteen
    pack lengths holds about 64 more.
    """

    settings, _, _ = _stream_setup(
        tmp_path,
        tiny_dataset_dir,
        tokenizer,
        training_max_sequence_length=DROP_BLOCK_SIZE,
        tokens_per_micro_batch=DROP_PACK_LENGTH,
    )
    return settings


def _drop_stage_manager(settings: Settings) -> StageManager:
    """
    One long stage drawing every sample from the `drop` source.
    """

    stage = resolved_stage(
        "only",
        tokens=100 * settings.tokens_per_optimizer_step,
        base_lr=1e-4,
        transition_pct=0.0,
        train_weights={"drop": 1.0},
    )
    return StageManager([stage], settings.tokens_per_optimizer_step)


def _drop_stream(
    settings: Settings, data_dir: Path, tokenizer: Tokenizer, worker_batch_rows: int = DROP_WORKER_ROWS
) -> tuple[BatchStream, _RecordingLoader]:
    """
    A stream over one drop-heavy source (single shard, in-process): rows are read in range order, in worker
    batches of `worker_batch_rows` rows.
    """

    parquet = entry_dataset(DataEntry("drop", str(data_dir), data_signature=DROP_SIGNATURE))
    loader = dataloader_over(parquet, tokenizer, DROP_BLOCK_SIZE, worker_batch_rows, padded=False)
    recording = _RecordingLoader(loader)
    loaders = RunDataloaders({"drop": recording}, [], tokenizer, {"drop": parquet})
    return BatchStream(settings, loaders, _drop_stage_manager(settings), TrainingProgress()), recording


def _run_world_batches(settings: Settings, stream: BatchStream, world_batches: int) -> None:
    for _ in range(world_batches):
        for _ in range(settings.gradient_accumulation_steps):
            next(stream)
        stream.progress.advance()


def _sample_ids(samples: list[Sample]) -> list[tuple[int, ...]]:
    """
    A hashable row identity: the unpadded input tokens (unique per fixture row).
    """

    return [tuple(input_ids.tolist()) for input_ids, _, _ in samples]


def test_batch_stream_counts_rows_read_not_surviving_samples(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    The stored counter advances by rows READ from disk, dropped rows included, so `state_dict` stores exactly
    what `set_resume_offset` will skip. Counting survivors instead undercounted by one row per drop (H7).
    """

    settings = _drop_settings(tmp_path, tiny_dataset_dir, stream_tokenizer)
    _write_drop_parquet(tmp_path / "drop_data")
    stream, recording = _drop_stream(settings, tmp_path / "drop_data", stream_tokenizer)
    packs = _micro_batches(settings, stream, 3)
    state = stream.state_dict()
    consumed = state["consumed_rows"]["drop"]
    # the loader is pulled in worker batches of `DROP_WORKER_ROWS` rows until the draw finds a sample; the counter
    # is what the loader read, and its survivors (the fixture's drop pattern alone decides how many) are exactly
    # the documents in the packs, the pool and the buffer
    assert consumed == recording.rows_read and consumed % DROP_WORKER_ROWS == 0
    assert len(recording.seen) == _drop_survivors(consumed) < consumed  # counting survivors would rewind
    assert _drawn(state, packs, "drop") == len(recording.seen)


def test_mid_stage_resume_with_dropped_rows_repeats_and_skips_nothing(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    A resume from a mid-stage checkpoint continues at exactly the next unread row also when the workers dropped
    rows: interrupted + resumed pulls are the very sample sequence of an uninterrupted run over the same data:
    nothing re-read (a repeat), nothing jumped over (a skip). Worker batches of one row keep the stream's buffer
    empty at the checkpoint, so the counter marks exactly the next unconsumed row (the pooled documents travel in
    the state).
    """

    settings = _drop_settings(tmp_path, tiny_dataset_dir, stream_tokenizer)
    data_dir = tmp_path / "drop_data"
    _write_drop_parquet(data_dir)
    k = 2  # world batches before the checkpoint

    full, full_recording = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=1)
    _run_world_batches(settings, full, 2 * k)
    uninterrupted = _sample_ids(full_recording.seen)
    assert len(set(uninterrupted)) == len(uninterrupted)  # unique rows: sequence equality below implies no repeats

    first, first_recording = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=1)
    _run_world_batches(settings, first, k)
    state = first.state_dict()
    before = _sample_ids(first_recording.seen)
    assert state["consumed_rows"] == {"drop": first_recording.rows_read} and state["buffers"] == {}
    assert before == uninterrupted[: len(before)]  # the single-shard stream is deterministic

    resumed, resumed_recording = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=1)
    resumed.load_state_dict(state)
    _run_world_batches(settings, resumed, k)
    assert resumed_recording.rows_read > 0
    assert before + _sample_ids(resumed_recording.seen) == uninterrupted  # the same pulls, split at the checkpoint


def test_mid_stage_resume_with_buffered_samples_repeats_nothing(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    With worker batches of several rows, samples can sit in the stream's buffer at the checkpoint; they were
    already counted as read, so a resume may skip them, but it never repeats a row the first stream pulled.
    """

    settings = _drop_settings(tmp_path, tiny_dataset_dir, stream_tokenizer)
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


@pytest.mark.parametrize("worker_batch_rows", [1, 5, 64])
def test_batch_stream_is_the_same_for_any_worker_batch_size(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer, worker_batch_rows: int
) -> None:
    """
    The worker batch size of the train loaders (`TRAIN_LOADER_BATCH_ROWS`, raised so a world batch is tokenized
    ahead of its pull) is invisible to the model: the one reader per source walks its range in order and the stream
    buffers its batches per source, so the micro-batches are identical to those of loaders with worker batches
    of `DROP_WORKER_ROWS` rows, across dropped rows and the wrap into the next epoch (10 steps of 4 packs plus
    the pool's lookahead over 200 surviving rows), with the row counters agreeing at the end of every epoch.
    """

    settings = _drop_settings(tmp_path, tiny_dataset_dir, stream_tokenizer)
    data_dir = tmp_path / "drop_data"
    _write_drop_parquet(data_dir)
    reference, _ = _drop_stream(settings, data_dir, stream_tokenizer)
    stream, recording = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=worker_batch_rows)
    assert recording.loader.batch_size == worker_batch_rows
    packs = _micro_batches(settings, stream, 10)
    assert _same_batches(packs, _micro_batches(settings, reference, 10))
    # the counter is rows READ, for any batch size: the survivors of the rows read (the whole range once, then the
    # fixture's drop pattern again from row 0) are the documents the model saw plus the ones pooled or buffered ahead
    state = stream.state_dict()
    consumed = state["consumed_rows"]["drop"]
    assert DROP_ROWS < consumed <= 2 * DROP_ROWS  # wrapped into the second epoch
    assert _survivors_read(consumed) == _drawn(state, packs, "drop")


def test_batch_stream_with_worker_processes_is_the_same_for_any_worker_batch_size(
    tmp_path: Path, tiny_dataset_dir: Path, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The same on the real path (`build_run_dataloaders`: one worker process per source, prefetching, length-sorted
    micro-batches): the run's loaders at `TRAIN_LOADER_BATCH_ROWS` yield the micro-batches of loaders whose worker
    batch is the micro-batch, over the pretrain stage and into the finetune source.
    """

    settings = parse_settings(["--config", str(write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])
    dataset = resolve_dataset(settings)
    stage_manager = StageManager(dataset.stages, settings.tokens_per_optimizer_step)
    world_batches = stage_manager.total_steps  # 20: both pretrain stages, the transition and the finetune stage

    def batches(worker_batch_rows: int) -> list[PackedBatch]:
        monkeypatch.setattr(loader_module, "TRAIN_LOADER_BATCH_ROWS", worker_batch_rows)
        with _run_loaders(settings, dataset, cpu_backend) as loaders:
            assert all(
                cast(DataLoader[Row], loader).batch_size == worker_batch_rows for loader in loaders.train_loaders.values()
            )
            return _micro_batches(
                settings, BatchStream(settings, loaders, stage_manager, TrainingProgress()), world_batches
            )

    assert _same_batches(batches(loader_module.TRAIN_LOADER_BATCH_ROWS), batches(2))


def test_mid_run_resume_with_wide_worker_batches_reproduces_the_stream(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    With worker batches wider than a pack's worth of documents (64 rows against a pack length of short documents)
    part of a checkpoint's rows sit in the stream's buffer, the rest in the pool: both travel in the state, so a
    stream resumed from a mid-run checkpoint continues with exactly the packs the uninterrupted stream produces,
    across dropped rows and the epoch wrap.
    """

    settings = _drop_settings(tmp_path, tiny_dataset_dir, stream_tokenizer)
    data_dir = tmp_path / "drop_data"
    _write_drop_parquet(data_dir)
    k = 5  # world batches before the checkpoint: the whole first epoch read (four 64-row batches and the 44-row tail), the wrap after

    full, _ = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=64)
    uninterrupted = _micro_batches(settings, full, 2 * k)

    first, _ = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=64)
    before = _micro_batches(settings, first, k)
    state = first.state_dict()
    assert state["consumed_rows"] == {"drop": DROP_ROWS} and len(state["buffers"]["drop"]) > 0 and state["pool"]

    resumed, resumed_recording = _drop_stream(settings, data_dir, stream_tokenizer, worker_batch_rows=64)
    resumed.load_state_dict(state)
    resumed.progress.step = k
    after = _micro_batches(settings, resumed, k)
    assert resumed_recording.rows_read > 0  # the resumed reader started epoch two at row 0
    assert _same_batches(before + after, uninterrupted)


# --------------------------------------------------------------------------------------------------------------
# dataset-independent numerics reference: golden_tiny_steps.json


def step_reference_metrics() -> dict[str, Any]:
    """
    Five optimizer steps of scripted batches through `run_one_optimizer_step` on the tiny model, reduced to their
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
    """
    Re-record `training/golden_tiny_steps.json`. ONLY do this in a commit whose purpose is a numerics change of
    the optimizer step (or of the tiny architecture / the optimizer):

        uv run python -c "from training.test_step import record_step_reference; record_step_reference()"

    Recorded with torch 2.14.0+cu130 on the author's machine (CPU, fp32, one thread, deterministic algorithms), in
    the commit that made packing mandatory. The padded reference before it was recorded in the commit that extracted
    `run_one_optimizer_step`, with the tiny golden run passing unchanged, which tied the step to the thesis loop;
    the packed loop is that step with `model_inputs` adding the per-document positions and mask.
    """

    GOLDEN_STEPS_PATH.write_text(golden_run_json(step_reference_metrics()))
    return GOLDEN_STEPS_PATH


# --------------------------------------------------------------------------------------------------------------
# the packs themselves and the step loop on packed micro-batches


def test_packed_stream_yields_one_full_pack_per_micro_batch(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Every micro-batch is one row of `tokens_per_micro_batch` positions filled with whole documents (first-fit from
    the pool) and a tail shorter than the longest document the loaders hand out; `micro_batches_per_step` micro-batches
    form a step. In stage 0 every document comes from source `a`.
    """

    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    assert settings.gradient_accumulation_steps == 4 and settings.tokens_per_optimizer_step == 1024
    progress = TrainingProgress()
    stream = BatchStream(settings, loaders, stage_manager, progress)
    for _ in range(2):
        for _ in range(settings.gradient_accumulation_steps):
            batch = next(stream)
            assert isinstance(batch, PackedBatch)
            for tensor in (batch.input_ids, batch.labels, batch.position_ids, batch.document_ids):
                assert tensor.shape == (1, PACK_LENGTH)
            assert set(batch.data_ids) == {"a"} and len(batch.data_ids) > 10
            assert 0 <= batch.padding_tokens <= 6, "`_Repeat` documents have at most 6 positions: one always fits"
            assert sum(_document_slots(batch)) + batch.padding_tokens == PACK_LENGTH
            assert int((batch.labels != IGNORE_INDEX).sum()) == PACK_LENGTH - batch.padding_tokens
            assert torch.all(batch.document_ids[0, 1:] >= batch.document_ids[0, :-1]), "documents lie side by side"
        progress.advance()


def test_packed_stream_draws_exactly_the_documents_the_padded_stream_would(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    Packing changes the grouping, not the data: after one step the documents in its packs plus those waiting in
    the pool are exactly the first `consumed_rows` draws of the loader (`_Repeat("a")` hands out sample i with
    `1 + (i + 1) % 7` tokens), each document whole and none dropped or duplicated.
    """

    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress())
    packs = [_next_pack(stream) for _ in range(settings.gradient_accumulation_steps)]
    state = stream.state_dict()
    drawn = sum(len(pack.data_ids) for pack in packs) + len(state["pool"])
    assert state["consumed_rows"] == {"a": drawn} and state["buffers"] == {}
    in_packs = [slots for pack in packs for slots in _document_slots(pack)]
    in_pool = [shifted_length(sample) for sample in state["pool"]]
    assert sorted(in_packs + in_pool) == sorted((1 + (i + 1) % 7) - 1 for i in range(drawn))
    # refilled to sixteen pack lengths before the last pack was taken, which removed at most one pack length
    assert sum(in_pool) >= (POOL_TOKEN_FACTOR - 1) * PACK_LENGTH


def test_packed_stream_state_round_trip_carries_the_pool(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    A resumed packed stream continues with the same packs: the loaded and target slots per source (the next pick
    depends on them: at step 15 `b` and `c` alternate by deficit), the row counters (the fakes are set to the
    consumed rows, what `set_resume_offset` does for a real dataset) and the pooled documents come back.
    """

    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stream = BatchStream(settings, loaders, stage_manager, TrainingProgress(step=15))
    for _ in range(3):
        next(stream)
    state = stream.state_dict()
    assert state["pool"] and set(state["consumed_rows"]) == {"b", "c"}
    assert set(state) == {"consumed_rows", "pool_loaded", "pool_target", "buffers", "pool"}
    assert state["pool_loaded"]["a"] == 0 and state["pool_loaded"]["b"] > 0 and state["pool_loaded"]["c"] > 0
    assert state["pool_target"]["b"] == pytest.approx(state["pool_target"]["c"])  # 0.5 / 0.5 since the start
    assert abs(state["pool_loaded"]["b"] - state["pool_loaded"]["c"]) <= 6  # within the longest `_Repeat` document
    continued = [_next_pack(stream) for _ in range(6)]

    fresh = {t: _Repeat(t) for t in "abc"}
    for tag, loader in fresh.items():
        loader.count = state["consumed_rows"].get(tag, 0)
    resumed = BatchStream(
        settings, RunDataloaders(dict(fresh), [], stream_tokenizer, {}), stage_manager, TrainingProgress(step=15)
    )
    resumed.load_state_dict(state)
    restored = resumed.state_dict()
    assert restored["consumed_rows"] == state["consumed_rows"]
    assert restored["pool_loaded"] == state["pool_loaded"] and restored["pool_target"] == state["pool_target"]
    assert [(s[0].tolist(), s[2]) for s in restored["pool"]] == [(s[0].tolist(), s[2]) for s in state["pool"]]
    for expected in continued:
        got = _next_pack(resumed)
        assert got.data_ids == expected.data_ids and got.padding_tokens == expected.padding_tokens
        assert torch.equal(got.input_ids, expected.input_ids) and torch.equal(got.labels, expected.labels)
        assert torch.equal(got.position_ids, expected.position_ids)
        assert torch.equal(got.document_ids, expected.document_ids)


def test_resumed_three_source_stream_picks_what_the_uninterrupted_one_would(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    The loaded and target slots are restored as state, not as a dict: with three sources at unequal weights and
    unequal document lengths the deficits are asymmetric, so a resume that dropped or zeroed them would pick a
    different source on its very first document. Twenty packs after the resume are compared document for document
    against the uninterrupted stream (more than the pool's `POOL_TOKEN_FACTOR` lag: the first packs come out of the
    restored pool, so a wrong pick only shows up in a pack after it), and the shares still follow the weights.

    (At 0.5 / 0.5 with equal lengths - what the other resume tests use - zeroing both numbers is a no-op: the
    deficits are symmetric, so the pick order survives it and only a dict comparison notices.)
    """

    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    settings, _, _ = _stream_setup(
        tmp_path, tiny_dataset_dir, stream_tokenizer, tokens_per_micro_batch=4096, micro_batches_per_step=1
    )
    lengths = {"a": 700, "b": 130, "c": 41}
    stage_manager = _weighted_stage_manager(settings, weights)

    def stream() -> BatchStream:
        loaders = RunDataloaders({tag: _Fixed(tag, n) for tag, n in lengths.items()}, [], stream_tokenizer, {})
        return BatchStream(settings, loaders, stage_manager, TrainingProgress())

    uninterrupted = stream()
    _packs(uninterrupted, 30)
    state = uninterrupted.state_dict()
    continued = _packs(uninterrupted, 20)

    resumed = stream()
    resumed.load_state_dict(state)
    after_resume = _packs(resumed, 20)
    assert [pack.data_ids for pack in after_resume] == [pack.data_ids for pack in continued]
    assert _same_batches(after_resume, continued)
    assert resumed.state_dict()["pool_loaded"] == uninterrupted.state_dict()["pool_loaded"]

    trained = _loaded_tokens(after_resume)
    for source, weight in weights.items():
        assert trained[source] / sum(trained.values()) == pytest.approx(weight, abs=0.05), source


def test_batch_stream_drops_the_row_counter_of_a_source_that_is_gone(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """
    A checkpoint of a run whose config has since dropped a source must not carry its row counter along: it would be
    written into every later checkpoint and, if the source is ever added back under the same name, silently skip
    that many rows of its first epoch. The buffers and the two numbers are already filtered this way.
    """

    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, stream_tokenizer)
    stream = BatchStream(
        settings, RunDataloaders({"a": _Fixed("a", 32)}, [], stream_tokenizer, {}), stage_manager, TrainingProgress()
    )
    stream.load_state_dict(
        {"consumed_rows": {"a": 7, "gone": 11}, "pool_loaded": {"a": 3}, "pool_target": {"a": 3.0}, "buffers": {}, "pool": []}
    )
    assert stream.state_dict()["consumed_rows"] == {"a": 7}


def test_model_inputs_of_a_packed_batch(cpu_backend: SingleDeviceBackend) -> None:
    """
    A pack reaches the model as `input_ids`, `labels`, `position_ids` and the ready document mask (dense on the
    CPU), built outside the model.
    """

    packed = next(scripted_batches(reference_settings()))
    inputs = model_inputs(packed, cpu_backend)
    assert set(inputs) == {"input_ids", "labels", "position_ids", "attention_mask"}
    assert torch.equal(inputs["position_ids"], packed.position_ids)
    mask = inputs["attention_mask"]
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.shape == (1, 1, 256, 256)
    # documents of 128, 64 and 32 positions: document 1 spans 128..191, the tail 224..255
    assert not bool(mask[0, 0, 128, 0]), "document 1 does not see document 0"
    assert bool(mask[0, 0, 128, 128]) and bool(mask[0, 0, 190, 128]) and not bool(mask[0, 0, 128, 129])
    assert not bool(mask[0, 0, 200, 128]), "document 2 does not see document 1"
    assert bool(mask[0, 0, 230, 230]) and not bool(mask[0, 0, 230, 223]), "the tail is its own document"


def test_optimizer_step_on_packed_batches(cpu_backend: SingleDeviceBackend) -> None:
    """
    The step loop takes packed micro-batches as it takes padded ones: `micro_batches_per_step`
    of them per step, one data id per document, a finite loss and gradient, and the packing-efficiency metric at
    log steps (2 tails of 32 in 512 tokens).
    """

    settings = reference_settings()
    assert settings.gradient_accumulation_steps == 2
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    results = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    for result in results:
        assert torch.isfinite(result.loss) and result.loss > 0
        assert torch.isfinite(result.grad_norm) and result.grad_norm > 0
        assert result.data_ids == ["scripted"] * 6 and result.validation is None
        assert result.data_tokens == {"scripted": 2 * (128 + 64 + 32)}  # the two tails of 32 are not counted
        assert float(result.metrics["packing/padding_fraction"]) == pytest.approx(2 * 32 / 512)
        assert len(result.metrics) > 1, "the gradient metrics of a log step are there too"


def test_padding_metric_only_at_log_steps(cpu_backend: SingleDeviceBackend) -> None:
    settings = reference_settings(log_step_interval=2, log_gradient_metrics=False)
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    first, second = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    assert first.metrics == {} and set(second.metrics) == {"packing/padding_fraction"}


@pytest.mark.gpu
def test_packed_step_on_cuda_runs_through_flex_attention() -> None:
    """
    On CUDA the document mask is a FlexAttention `BlockMask` and the step (bf16 autocast, forward and backward)
    runs through `flex_attention`.
    """

    backend = SingleDeviceBackend(device="cuda:0", precision="bf16-mixed")
    settings = reference_settings(precision="bf16-mixed")
    inputs = model_inputs(next(scripted_batches(settings)), backend)
    assert isinstance(inputs["attention_mask"], BlockMask)
    model = fresh_tiny_model(backend)
    optimizer = fresh_optimizer(settings, model, backend)
    result = run_one_optimizer_step(
        settings, backend, model, optimizer, reference_stage_manager(settings), scripted_batches(settings),
        TrainingProgress(step=1),
    )
    assert torch.isfinite(result.loss) and torch.isfinite(result.grad_norm) and result.grad_norm > 0


def test_golden_tiny_steps() -> None:
    """
    Numerics regression guard for `run_one_optimizer_step`, independent of the data pipeline: five steps of
    scripted packs reproduce `golden_tiny_steps.json`.

    Same semantics as `test_golden_tiny_run`: floats with `rel=1e-5`, `GOLDEN_EXACT=1` compares with `==`
    (bit-identical on the recording machine), learning rates always exact; fp32 CPU, one thread, deterministic
    algorithms; the bf16 autocast path is not exercised. Re-record only in a numerics commit; loosen rather than
    chase float-order differences on another machine.
    """

    assert GOLDEN_STEPS_PATH.exists(), (
        "step reference missing; record it with record_step_reference() in a numerics commit"
    )
    expected = json.loads(GOLDEN_STEPS_PATH.read_text())
    actual = step_reference_metrics()
    assert sorted(actual["steps"], key=int) == [str(s) for s in range(REFERENCE_STEPS)]
    assert actual["steps"]["0"]["lr"] == 0.0 and actual["steps"]["2"]["lr"] == 3e-4
    exact = golden_exact_requested()
    mismatches = golden_mismatches(expected, json.loads(golden_run_json(actual)), exact=exact)
    assert not mismatches, "step reference changed:\n" + "\n".join(mismatches)
