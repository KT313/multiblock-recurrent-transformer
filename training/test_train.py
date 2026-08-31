# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the training loop helpers (fast) and end-to-end runs of `training.train.train` on the tiny 3-stage
config with synthetic data (marked slow)."""

import random
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from training.data.collate import find_multiple
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import RecurrentGPT
from training import train as train_module
from training.backend import SingleDeviceBackend
from training.checkpoint import checkpoint_dir, find_latest_checkpoint
from training.data import StageDataloaders, Tokenizer
from training.data.dataset_resolver import CHECKPOINT_HASH_KEY, ResolvedDataset, resolve_dataset
from training.data.loader import Batch
from training.logger import Logger
from training.settings import Settings, parse_settings
from training.stage_manager import StageManager
from training.train import IGNORE_INDEX, LoopState, build_stage_dataloaders, micro_batch_stream, unwrap, validate

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"


def _write_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: str) -> Path:
    """`config/tiny.yaml` with dataset_dir/out_dir rewritten and optional `key: value` line replacements."""
    lines = []
    for line in TINY_YAML.read_text().splitlines():
        key = line.split(":")[0].strip() if ":" in line and not line.startswith(" ") else None
        if key == "out_dir":
            line = f"out_dir: {out_dir}"
        elif key == "dataset_dir":
            line = f"dataset_dir: {tiny_dataset_dir}"
        elif key in overrides:
            line = f"{key}: {overrides.pop(key)}"
        lines.append(line)
    lines += [f"{k}: {v}" for k, v in overrides.items()]
    path = tmp_path / "tiny.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


Logged = dict[int, dict[str, Any]]


def _capture_logs(monkeypatch: pytest.MonkeyPatch) -> Logged:
    logged: Logged = {}

    def capture(self: Logger, metrics: dict[str, Any], step: int) -> None:
        logged.setdefault(step, {}).update({k: float(v) if torch.is_tensor(v) else v for k, v in metrics.items()})

    monkeypatch.setattr(Logger, "log", capture)
    return logged


def _run(yaml_path: Path, monkeypatch: pytest.MonkeyPatch) -> Logged:
    """Run training on the yaml and return `{step: metrics}` as handed to `Logger.log`."""
    logged = _capture_logs(monkeypatch)
    train_module.train(parse_settings(["--config", str(yaml_path)]))
    return logged


# --------------------------------------------------------------------------------------------------------------
# fast helper tests


@pytest.fixture
def tiny_settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(_write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out"))])


@pytest.fixture
def tiny_resolved(tiny_settings: Settings) -> ResolvedDataset:
    return resolve_dataset(tiny_settings)


@pytest.fixture
def cpu_backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


def test_unwrap(tiny_model: RecurrentGPT) -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    assert unwrap(tiny_model) is tiny_model
    assert unwrap(Wrapper(tiny_model)) is tiny_model


def test_loop_state_fields() -> None:
    state: LoopState = {"step": 0, "resume_step": -1}
    assert set(LoopState.__annotations__) == set(state)


def test_build_stage_dataloaders(
    tiny_settings: Settings, tiny_resolved: ResolvedDataset, cpu_backend: SingleDeviceBackend
) -> None:
    tokenizer = Tokenizer(tiny_resolved.tokenizer_dir)
    loaders = build_stage_dataloaders(tiny_settings, tiny_resolved, tokenizer, cpu_backend)
    assert isinstance(loaders, StageDataloaders)
    assert len(loaders.train_loaders) == len(loaders.val_loaders) == 3
    input_ids, labels, data_ids = loaders.next_train_batch(0)
    assert input_ids.shape[0] == tiny_settings.micro_batch_size and input_ids.shape == labels.shape
    assert input_ids.shape[1] % 128 == 0 and input_ids.shape[1] <= tiny_settings.block_size
    assert data_ids == ["pretrain_a-synthetic_pretrain"] * tiny_settings.micro_batch_size
    assert (labels == IGNORE_INDEX).any() or (input_ids != tokenizer.pad_id).all()
    _, _, val_ids = next(iter(loaders.val_loaders[2]))
    assert val_ids == ["finetune-synthetic_instruct"] * tiny_settings.micro_batch_size


def _fake_batch(tag: str, length: int, pad_id: int = 0) -> Batch:
    ids = torch.full((1, 8), pad_id)
    ids[0, :length] = 1
    labels = torch.full((1, 8), IGNORE_INDEX)
    labels[0, :length] = 1
    return ids, labels, [tag]


class _Repeat:
    """Endless iterable of one tagged batch with a running counter as the length."""

    def __init__(self, tag: str) -> None:
        self.tag, self.count = tag, 0

    def __iter__(self) -> Iterator[Batch]:
        while True:
            self.count += 1
            yield _fake_batch(self.tag, 1 + self.count % 7)


def _stream_setup(
    tmp_path: Path, tiny_dataset_dir: Path, sort: bool
) -> tuple[Settings, StageDataloaders, StageManager]:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", sort_batches_by_length=str(sort).lower())
    cfg = parse_settings(["--config", str(yaml_path), "--micro_batch_size", "1"])  # 4 micro-batches per step
    loaders = StageDataloaders(train_loaders=[_Repeat("a"), _Repeat("b"), _Repeat("c")], val_loaders=[])
    sm = StageManager(resolve_dataset(cfg).stage_manager_stages(), cfg.world_batch_size, cfg.block_size)
    return cfg, loaders, sm


def _tags(stream: Iterator[Batch], n: int) -> list[str]:
    return [next(stream)[2][0] for _ in range(n)]  # never pull an extra element (zip would)


def test_micro_batch_stream_samples_by_transition_progress(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    cfg, loaders, sm = _stream_setup(tmp_path, tiny_dataset_dir, sort=False)
    state: LoopState = {"step": 3, "resume_step": -1}  # plain stage 0
    stream = micro_batch_stream(cfg, loaders, sm, state, random.Random(0))
    assert _tags(stream, 4) == ["a"] * 4
    state["step"] = 6  # transition 0 -> 1, progress 0: everything still from stage 0
    assert _tags(stream, 4) == ["a"] * 4
    state["step"] = 8  # fully in stage 1
    assert _tags(stream, 4) == ["b"] * 4
    state["step"] = 15  # transition 1 -> 2 at progress 0.5: a mix, driven by the rng
    tags = _tags(stream, 40)
    assert set(tags) == {"b", "c"} and 8 < tags.count("c") < 32
    state["step"] = 19
    assert _tags(stream, 4) == ["c"] * 4


def test_micro_batch_stream_length_sorting(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    cfg, loaders, sm = _stream_setup(tmp_path, tiny_dataset_dir, sort=True)
    assert cfg.gradient_accumulation_steps == 4
    state: LoopState = {"step": 0, "resume_step": -1}
    stream = micro_batch_stream(cfg, loaders, sm, state, random.Random(0))
    for _ in range(3):  # every world batch (4 micro-batches) arrives sorted by supervised length, trimmed
        batches = [next(stream) for _ in range(4)]
        lengths = [int((b[1] != IGNORE_INDEX).sum()) for b in batches]
        assert lengths == sorted(lengths)
        assert all(b[0].shape[1] == min(find_multiple(n, 128), 8) for b, n in zip(batches, lengths))  # trimmed
    cfg_unsorted, loaders2, _ = _stream_setup(tmp_path, tiny_dataset_dir, sort=False)
    raw = micro_batch_stream(cfg_unsorted, loaders2, sm, state, random.Random(0))
    lengths = [int((next(raw)[1] != IGNORE_INDEX).sum()) for _ in range(4)]
    assert lengths == [2, 3, 4, 5]  # loader order, untouched


def test_validate_reports_every_depth(
    tiny_model: RecurrentGPT, tiny_settings: Settings, cpu_backend: SingleDeviceBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny_settings.partial_depth_eval = [1, 3]
    tiny_settings.eval_iters = 2
    torch.manual_seed(0)
    batches = [(torch.randint(1, 512, (2, 16)), torch.randint(1, 512, (2, 16)), ["v", "v"]) for _ in range(5)]
    seen: list[tuple[bool, Any]] = []
    forward = RecurrentGPT.forward

    def spy(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        seen.append((self.training, kwargs.get("num_steps_pair")))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(RecurrentGPT, "forward", spy)
    torch.manual_seed(1)  # the latent state is drawn from the global RNG; depth 1 is evaluated first
    metrics = validate(tiny_settings, cpu_backend, tiny_model, batches)
    expected = {"val_loss", "val_ppl"} | {f"val_{k}_{d}" for k in ("loss", "ppl") for d in (1, 3, "[2, 2]")}
    assert set(metrics) == expected
    assert all(torch.isfinite(v) for v in metrics.values())
    assert metrics["val_loss"] == metrics["val_loss_[2, 2]"]
    assert torch.allclose(metrics["val_ppl_1"], metrics["val_loss_1"].exp())
    assert tiny_model.training  # restored to train mode
    # every depth is evaluated on exactly `eval_iters` batches in eval mode, as one (depth, 0) pair per core block
    assert seen == [(False, [(1, 0), (1, 0)])] * 2 + [(False, [(3, 0), (3, 0)])] * 2 + [(False, [(2, 0), (2, 0)])] * 2
    # the depth actually changes the computation
    assert metrics["val_loss_1"] != metrics["val_loss_3"] != metrics["val_loss_[2, 2]"]
    # each column is the mean over the eval_iters batches (same RNG stream as the depth-1 column above)
    with torch.no_grad():
        torch.manual_seed(1)
        tiny_model.eval()
        per_batch = [tiny_model(x, labels=y, num_steps_pair=[(1, 0), (1, 0)])["loss"] for x, y, _ in batches[:2]]
    assert metrics["val_loss_1"].item() == pytest.approx(torch.stack(per_batch).mean().item(), rel=1e-6)


def test_main_parses_argv_and_trains(monkeypatch: pytest.MonkeyPatch, tiny_dataset_dir: Path, tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    seen: list[Settings] = []
    monkeypatch.setattr(train_module, "train", seen.append)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(yaml_path), "--seed", "5"])
    train_module.main()
    assert len(seen) == 1 and seen[0].seed == 5 and seen[0].out_dir == str(tmp_path / "out")


def test_block_size_mismatch_raises(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out", block_size="128")
    with pytest.raises(ValueError, match="block_size 128 does not match"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


def test_non_finite_loss_terminates(tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    forward = RecurrentGPT.forward

    def nan_forward(self: RecurrentGPT, *args: Any, **kwargs: Any) -> Any:
        out = forward(self, *args, **kwargs)
        assert out["loss"] is not None
        out["loss"] = out["loss"] * torch.tensor(float("nan"))
        return out

    monkeypatch.setattr(RecurrentGPT, "forward", nan_forward)
    with pytest.raises(RuntimeError, match="Loss is nan at step 0"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


def test_non_finite_grad_norm_terminates(tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    monkeypatch.setattr(SingleDeviceBackend, "clip_grad_norm", lambda self, model, max_norm: torch.tensor(float("inf")))
    with pytest.raises(RuntimeError, match="Gradient norm is non-finite at step 0"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))


# --------------------------------------------------------------------------------------------------------------
# end-to-end runs


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory, tiny_dataset_dir: Path) -> dict[str, Any]:
    """One uninterrupted tiny run shared by the assertions below (module-scoped: a few seconds on CPU)."""
    tmp = tmp_path_factory.mktemp("full_run")
    out_dir = tmp / "out"
    yaml_path = _write_yaml(tmp, tiny_dataset_dir, out_dir, export_to_hf="true")
    mp = pytest.MonkeyPatch()
    optimizer_steps: list[int] = []  # `model.step` at every optimizer.step() call (tiny.yaml uses AdamW)
    adamw_step = torch.optim.AdamW.step

    def counting_step(self: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        optimizer_steps.append(len(optimizer_steps))
        return adamw_step(self, *args, **kwargs)

    mp.setattr(torch.optim.AdamW, "step", counting_step)
    try:
        logged = _run(yaml_path, mp)
    finally:
        mp.undo()
    dataset_hash = resolve_dataset(parse_settings(["--config", str(yaml_path)])).config_hash
    return {
        "out_dir": out_dir,
        "yaml": yaml_path,
        "logged": logged,
        "optimizer_steps": len(optimizer_steps),
        "dataset_hash": dataset_hash,
    }


@pytest.mark.slow
def test_tiny_multistage_run_finishes_and_writes_checkpoints(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    assert sorted(logged) == list(range(1, 21))
    names = sorted(p.name for p in checkpoint_dir(full_run["out_dir"]).glob("*.pth"))
    assert names == [
        "step-00000006-tiny-stage-0_end.pth",
        "step-00000014-tiny-stage-1_end.pth",
        "step-00000020-tiny.pth",
    ]
    assert (full_run["out_dir"] / "run_config.json").exists()
    assert (full_run["out_dir"] / "model_config.json").exists()
    for step, m in logged.items():
        assert m["step"] == step and m["total_tokens"] == step * 4 * 256
        assert torch.isfinite(torch.tensor(m["loss"])) and m["grad_norm"] >= 0
    assert full_run["optimizer_steps"] == 19  # the very first update (step 0) is skipped
    # the stage-end checkpoints carry the step they were written at and the stage the run enters next
    for name, step, stage in (("step-00000006-tiny-stage-0_end.pth", 6, 1), ("step-00000014-tiny-stage-1_end.pth", 14, 2)):
        extra = torch.load(checkpoint_dir(full_run["out_dir"]) / name, map_location="cpu", weights_only=False)
        assert (extra["step"], extra["stage"]) == (step, stage)
        assert extra["config"]["run_name"] == "tiny" and set(extra["rng"]) >= {"python", "torch"}
        assert extra[CHECKPOINT_HASH_KEY] == full_run["dataset_hash"]


@pytest.mark.slow
def test_logged_lr_follows_the_multistage_schedule(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    # metrics at `done = step + 1` carry the LR used for optimizer step `step`
    expected = {1: 0.0, 2: 1.5e-4, 3: 3e-4, 7: 3e-4, 8: 2e-4, 9: 1e-4, 15: 1e-4, 16: 7.5e-5, 17: 5e-5, 20: 2.5e-5}
    for done, lr in expected.items():
        assert logged[done]["lr"] == pytest.approx(lr), done
    # inside a transition the stage info already names the next stage (steps 6-7 -> stage 1, 14-15 -> stage 2)
    assert [logged[d]["stage/current_stage"] for d in (1, 6, 7, 8, 9, 14, 15, 17)] == [0, 0, 1, 1, 1, 1, 2, 2]
    assert [logged[d]["stage/in_transition"] for d in (6, 7, 8, 9, 15, 16, 17)] == [0, 1, 1, 0, 1, 1, 0]
    assert logged[8]["stage/transition_progress"] == pytest.approx(0.5)


@pytest.mark.slow
def test_evaluates_at_every_partial_depth(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    eval_steps = [s for s, m in logged.items() if "val_loss" in m]
    assert eval_steps == [8, 16, 20]
    for s in eval_steps:
        m = logged[s]
        for depth in (1, "[2, 2]"):  # partial_depth_eval [1] plus the model's mean recurrence
            assert f"val_loss_{depth}" in m and f"val_ppl_{depth}" in m, (s, depth)
            assert torch.isfinite(torch.tensor(m[f"val_loss_{depth}"]))
        assert m["val_loss"] == pytest.approx(m["val_loss_[2, 2]"])
        assert m["val_ppl"] == pytest.approx(torch.tensor(m["val_loss"]).exp().item(), rel=1e-4)


@pytest.mark.slow
def test_data_composition_follows_the_stages(full_run: dict[str, Any]) -> None:
    logged: Logged = full_run["logged"]
    assert logged[3]["data_composition/pretrain_a-synthetic_pretrain"] == pytest.approx(1.0)
    assert logged[18]["data_composition/finetune-synthetic_instruct"] == pytest.approx(1.0)
    for done in range(15, 17):  # inside the 1 -> 2 transition both stages' sources may appear, weights sum to 1
        total = sum(v for k, v in logged[done].items() if k.startswith("data_composition/"))
        assert total == pytest.approx(1.0)


@pytest.mark.slow
def test_export_to_hf_produces_loadable_folder(full_run: dict[str, Any]) -> None:
    export_dir = full_run["out_dir"] / "hf_export"
    assert (export_dir / "config.json").exists() and (export_dir / "model.safetensors").exists()
    model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    ids = tokenizer("tok_1 tok_2 tok_3", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (1, ids.shape[1], 512)
    assert torch.isfinite(out.logits).all()

    # exported weights are the final checkpoint's weights
    final = torch.load(
        checkpoint_dir(full_run["out_dir"]) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False
    )["model"]
    wte = final["transformer.wte.weight"]
    assert torch.equal(model.model.transformer.wte.weight.detach(), wte)


@pytest.mark.slow
def test_same_seed_is_deterministic(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, tmp_path / "out")
    logged = _run(yaml_path, monkeypatch)
    for done in range(1, 21):
        assert logged[done]["loss"] == pytest.approx(full_run["logged"][done]["loss"], rel=1e-5), done


@pytest.mark.slow
def test_resume_picks_latest_checkpoint_and_restores_the_schedule(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume: true` continues from the latest checkpoint of the run (here the stage-1_end one at step 14).

    Losses cannot be compared exactly here: the resumed run re-seeds the transition sampler with
    `seed + step` and rebuilds the loaders, so the mixed batches of the 1 -> 2 transition (steps 14, 15) differ
    by design. Exact equivalence is asserted in `test_resume_is_bit_exact_without_transitions`."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    latest = find_latest_checkpoint(out_dir, "tiny")
    assert latest is not None and latest.name == "step-00000014-tiny-stage-1_end.pth"
    yaml_path = _write_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    logged = _run(yaml_path, monkeypatch)

    assert sorted(logged) == list(range(15, 21))  # steps 14..19 ran, nothing before
    assert (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").exists()
    full: Logged = full_run["logged"]
    for done in range(15, 21):  # schedule state is restored exactly
        assert logged[done]["lr"] == pytest.approx(full[done]["lr"])
        assert logged[done]["stage/current_stage"] == full[done]["stage/current_stage"]
        assert logged[done]["stage/in_transition"] == full[done]["stage/in_transition"]
        assert logged[done]["total_tokens"] == full[done]["total_tokens"]
    assert [s for s, m in logged.items() if "val_loss" in m] == [16, 20]
    assert logged[16]["data_composition/finetune-synthetic_instruct"] == pytest.approx(0.5, abs=0.5)  # transition mix


def _no_transition_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: str) -> Path:
    """tiny.yaml without transitions; fp32 because bf16 autocast is very slow on the CPU and precision is
    irrelevant for the bit-exactness claim."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset_yaml = tmp_path / "tiny_dataset.yaml"
    dataset_yaml.write_text(TINY_DATASET_YAML.read_text().replace("transition_pct: 0.25", "transition_pct: 0.0"))
    return _write_yaml(
        tmp_path, tiny_dataset_dir, out_dir, precision='"32"', dataset_config=str(dataset_yaml), **overrides
    )


@pytest.mark.slow
def test_resume_is_bit_exact_without_transitions(
    tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without transitions (no rng-driven mixing) resuming from the stage-0_end checkpoint reproduces the
    uninterrupted run exactly: every logged loss, the validation losses and the final model + optimizer state."""
    full_dir = tmp_path / "full" / "out"
    logged_full = _run(_no_transition_yaml(tmp_path / "full", tiny_dataset_dir, full_dir), monkeypatch)
    names = sorted(p.name for p in checkpoint_dir(full_dir).glob("*.pth"))
    assert names == ["step-00000008-tiny-stage-0_end.pth", "step-00000016-tiny-stage-1_end.pth", "step-00000020-tiny.pth"]

    resumed_dir = tmp_path / "resumed" / "out"
    yaml_path = _no_transition_yaml(
        tmp_path / "resumed",
        tiny_dataset_dir,
        resumed_dir,
        resume="true",
        resume_checkpoint_path=str(checkpoint_dir(full_dir) / "step-00000008-tiny-stage-0_end.pth"),
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(9, 21))
    for done in range(9, 21):
        assert logged[done]["loss"] == logged_full[done]["loss"], done  # exact, not approx
        assert logged[done]["lr"] == logged_full[done]["lr"]
        assert logged[done].get("val_loss") == logged_full[done].get("val_loss")
    assert "val_loss" in logged[16] and "val_loss" in logged[20]

    final_full = torch.load(checkpoint_dir(full_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    final_res = torch.load(checkpoint_dir(resumed_dir) / "step-00000020-tiny.pth", map_location="cpu", weights_only=False)
    assert final_full["model"].keys() == final_res["model"].keys()
    assert all(torch.equal(final_full["model"][k], final_res["model"][k]) for k in final_full["model"])
    for sa, sb in zip(final_full["optimizer"]["state"].values(), final_res["optimizer"]["state"].values()):
        assert all(torch.equal(sa[k], sb[k]) for k in sa if torch.is_tensor(sa[k]))
    assert final_res["step"] == 20


@pytest.mark.slow
def test_resume_from_explicit_checkpoint_path_with_resume_warmup(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = checkpoint_dir(full_run["out_dir"]) / "step-00000006-tiny-stage-0_end.pth"
    yaml_path = _write_yaml(
        tmp_path,
        tiny_dataset_dir,
        tmp_path / "fresh_out",
        resume="true",
        resume_checkpoint_path=str(ckpt),
        resume_warmup_steps="2",
        export_to_hf="false",
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(7, 21))
    assert logged[7]["lr"] == pytest.approx(0.0)  # step 6: ramp starts at min_lr
    assert logged[8]["lr"] == pytest.approx(0.5 * 2e-4)  # step 7: halfway to the schedule's 2e-4
    assert logged[9]["lr"] == pytest.approx(1e-4)  # step 8: back on the schedule


@pytest.mark.slow
def test_resume_with_changed_dataset_config_raises_unless_allowed(
    full_run: dict[str, Any], tmp_path: Path, tiny_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dataset config whose hash differs from the checkpoint's (here: transition_pct changed, data unchanged)."""
    out_dir = tmp_path / "out"
    shutil.copytree(full_run["out_dir"], out_dir)
    (checkpoint_dir(out_dir) / "step-00000020-tiny.pth").unlink()
    yaml_path = _no_transition_yaml(tmp_path, tiny_dataset_dir, out_dir, resume="true", export_to_hf="false")
    with pytest.raises(RuntimeError, match="dataset config hash"):
        train_module.train(parse_settings(["--config", str(yaml_path)]))
    yaml_path = _no_transition_yaml(
        tmp_path / "allowed", tiny_dataset_dir, out_dir, resume="true", export_to_hf="false", allow_dataset_change="true"
    )
    logged = _run(yaml_path, monkeypatch)
    assert sorted(logged) == list(range(15, 21))
