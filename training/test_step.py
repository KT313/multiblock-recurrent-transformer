# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for one optimizer step (`training.step`): the progress counter, the micro-batch stream, the LR, the
accumulation arithmetic, the skipped first update, the non-finite checks — and the dataset-independent numerics
reference `training/golden_tiny_steps.json` (five steps of fixed batches through `run_one_optimizer_step`)."""

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT, build_model
from training.backend import SingleDeviceBackend
from training.data import Batch, IGNORE_INDEX, SampleBatch, StageDataloaders
from training.data.collate import find_multiple
from training.data.dataset_resolver import resolve_dataset
from training.data.tokenizer import Tokenizer
from training.golden import (
    golden_exact_requested,
    golden_mismatches,
    golden_run_json,
    single_thread_deterministic,
    write_tiny_yaml,
)
from training.run import build_run_optimizer
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager, TrainingStage
from training.step import (
    StepResult,
    TrainingProgress,
    micro_batch_stream,
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
        optim_config=dict(
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
    stage = TrainingStage("only", tokens=steps * settings.world_batch_size * settings.block_size, base_lr=3e-4, transition_pct=0.0)
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
    assert (progress.step, progress.resume_step, progress.done) == (0, -1, 0)
    progress.advance()
    assert progress.step == progress.done == 1
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


def test_learning_rate_is_set_on_all_groups_times_base_lr(settings: Settings, cpu_backend: SingleDeviceBackend) -> None:
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    assert len(optimizer.param_groups) == 3
    for group, base_lr in zip(optimizer.param_groups, (1.0, 0.5, 2.0)):
        group["base_lr"] = base_lr
    results = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    assert results[1].learning_rate == pytest.approx(1.5e-4)  # warmup step 1 of 2
    for group, base_lr in zip(optimizer.param_groups, (1.0, 0.5, 2.0)):
        assert torch.is_tensor(group["lr"])  # `set_lr` stores a tensor (ELLISAdam clones it)
        assert float(group["lr"]) == pytest.approx(results[1].learning_rate * base_lr)


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
    """`stage` / `next_stage` are the manager's infos at `step` and `step + 1`; gradient metrics only at log steps."""
    settings.log_step_interval = 2
    model = fresh_tiny_model(cpu_backend)
    optimizer = fresh_optimizer(settings, model, cpu_backend)
    stage_manager = reference_stage_manager(settings)
    results = run_steps(settings, cpu_backend, model, optimizer, steps=2)
    assert results[0].stage == stage_manager.get_stage_info(0) and results[0].next_stage == stage_manager.get_stage_info(1)
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
# micro-batch stream (moved from the former test_train.py; needs the tiny dataset for the stage budgets only)


def _fake_samples(tag: str, lengths: list[int]) -> SampleBatch:
    """One worker batch: one unpadded sample per entry of `lengths`, every position supervised."""
    return [(torch.full((n,), 3, dtype=torch.long), torch.full((n,), 3, dtype=torch.long), tag) for n in lengths]


class _Repeat:
    """Endless iterable of one tagged one-sample worker batch with a running counter as the sample length."""

    def __init__(self, tag: str, batch_size: int = 1) -> None:
        self.tag, self.batch_size, self.count = tag, batch_size, 0

    def __iter__(self) -> Iterator[SampleBatch]:
        while True:
            self.count += 1
            yield _fake_samples(self.tag, [1 + (self.count + i) % 7 for i in range(self.batch_size)])


class _ShortBatches:
    """A loader whose worker batches cycle through `sizes` rows — a short (or empty) batch is what a loader running
    out of rows, or a batch whose rows were all dropped for lack of a supervised label, hands the stream."""

    def __init__(self, tag: str, sizes: list[int]) -> None:
        self.tag, self.sizes, self.count = tag, sizes, 0

    def __iter__(self) -> Iterator[SampleBatch]:
        while True:
            size = self.sizes[self.count % len(self.sizes)]
            self.count += 1
            yield _fake_samples(self.tag, [1 + (self.count + i) % 7 for i in range(size)])


@pytest.fixture(scope="session")
def stream_tokenizer(tiny_tokenizer_dir: Path) -> Tokenizer:
    """The synthetic tokenizer the stream pads its assembled micro-batches with."""
    return Tokenizer(tiny_tokenizer_dir)


def _stream_setup(
    tmp_path: Path,
    tiny_dataset_dir: Path,
    sort: bool,
    tokenizer: Tokenizer,
    batch_size: int = 1,
    padding_multiple: int = 128,
) -> tuple[Settings, StageDataloaders, StageManager]:
    yaml_path = write_tiny_yaml(
        tmp_path,
        tiny_dataset_dir,
        tmp_path / "out",
        sort_batches_by_length=str(sort).lower(),
        sequence_padding_multiple=str(padding_multiple),
    )
    settings = parse_settings(
        ["--config", str(yaml_path), "--micro_batch_size", str(batch_size)]  # 4 / batch_size micro-batches per step
    )
    loaders = StageDataloaders([_Repeat(t, batch_size) for t in "abc"], [], tokenizer)
    stage_manager = StageManager(resolve_dataset(settings).training_stages(), settings.world_batch_size, settings.block_size)
    return settings, loaders, stage_manager


def _tags(stream: Iterator[Batch], n: int) -> list[str]:
    return [next(stream)[2][0] for _ in range(n)]  # never pull an extra element (zip would)


def test_micro_batch_stream_samples_by_transition_progress(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress(step=3)  # plain stage 0
    stream = micro_batch_stream(settings, loaders, stage_manager, progress)
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


def test_micro_batch_stream_reads_the_step_lazily(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """The step is read when the first micro-batch of a world batch is requested, not when the stream is created and
    not again inside the world batch."""
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)
    progress = TrainingProgress(step=0)
    stream = micro_batch_stream(settings, loaders, stage_manager, progress)
    progress.step = 8  # changed before the first request: the world batch is stage 1's
    assert _tags(stream, 2) == ["b"] * 2
    progress.step = 19  # changed inside the world batch: the remaining two micro-batches still belong to step 8
    assert _tags(stream, 2) == ["b"] * 2
    assert _tags(stream, 4) == ["c"] * 4  # the next world batch reads step 19


def test_micro_batch_stream_transition_rng_is_seeded_with_the_start_step(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """Two streams created at the same `seed + step` draw the same mix; a different start step draws another (the
    resume seeds the transition RNG with `seed + resume step`, as the thesis loop did)."""
    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer)

    def mix(start_step: int) -> list[str]:
        loaders = StageDataloaders([_Repeat(tag) for tag in "abc"], [], stream_tokenizer)
        progress = TrainingProgress(step=start_step)
        stream = micro_batch_stream(settings, loaders, stage_manager, progress)
        progress.step = 15
        return _tags(stream, 40)

    assert mix(0) == mix(0)
    assert mix(0) != mix(14)


def test_micro_batch_stream_length_sorting(tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer) -> None:
    settings, loaders, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, True, stream_tokenizer, padding_multiple=4)
    assert settings.gradient_accumulation_steps == 4
    progress = TrainingProgress()
    stream = micro_batch_stream(settings, loaders, stage_manager, progress)
    for _ in range(3):  # every world batch (4 micro-batches) arrives sorted by length, padded to its own width
        batches = [next(stream) for _ in range(4)]
        lengths = [int((b[1] != IGNORE_INDEX).sum()) for b in batches]  # supervised = sample length - 1
        assert lengths == sorted(lengths)
        assert all(b[0].shape[1] == find_multiple(n + 1, 4) - 1 for b, n in zip(batches, lengths))
        progress.advance()
    unsorted_settings, unsorted_loaders, _ = _stream_setup(
        tmp_path, tiny_dataset_dir, False, stream_tokenizer, padding_multiple=4
    )
    raw = micro_batch_stream(unsorted_settings, unsorted_loaders, stage_manager, TrainingProgress())
    lengths = [int((next(raw)[1] != IGNORE_INDEX).sum()) for _ in range(4)]
    assert lengths == [1, 2, 3, 4]  # loader order, untouched


def test_micro_batch_stream_fills_the_world_batch_from_short_worker_batches(
    tmp_path: Path, tiny_dataset_dir: Path, stream_tokenizer: Tokenizer
) -> None:
    """Regression (T-M3/T-M4): a short or empty worker batch — a loader reaching its last rows, or a batch whose rows
    were all dropped — used to shrink the world batch and permanently misalign it with the optimizer steps. The
    stream now pulls until it holds `world_batch_size` samples and carries the surplus over."""
    settings, _, stage_manager = _stream_setup(tmp_path, tiny_dataset_dir, False, stream_tokenizer, batch_size=2)
    assert (settings.gradient_accumulation_steps, settings.world_batch_size) == (2, 4)
    loaders = StageDataloaders([_ShortBatches(tag, [2, 0, 1, 2, 3]) for tag in "abc"], [], stream_tokenizer)
    progress = TrainingProgress()
    stream = micro_batch_stream(settings, loaders, stage_manager, progress)
    for _ in range(6):
        batches = [next(stream) for _ in range(settings.gradient_accumulation_steps)]
        assert [b[0].shape[0] for b in batches] == [settings.micro_batch_size] * settings.gradient_accumulation_steps
        assert sum(len(b[2]) for b in batches) == settings.world_batch_size
        progress.advance()


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
