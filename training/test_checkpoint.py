# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for checkpoint naming/search, save→load→forward bit-identity, optimizer state and RNG restore."""

import random
from pathlib import Path
from typing import Any

import pytest
import torch

from model import RecurrentGPT, build_model
from training.backend import SingleDeviceBackend
from training.checkpoint import (
    CHECKPOINT_SUBDIR,
    _step_from_name,
    _unwrap,
    checkpoint_dir,
    checkpoint_name,
    collect_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    should_save_checkpoint,
)
from training.optim import ELLISAdam, get_param_groups

TINY_MODEL_ARCHITECTURE = Path(__file__).resolve().parent.parent / "config" / "model_architecture" / "tiny.yaml"


@pytest.fixture
def backend() -> SingleDeviceBackend:
    return SingleDeviceBackend(device="cpu", precision="32")


def test_checkpoint_name_and_dir(tmp_path: Path) -> None:
    assert checkpoint_name(6, "tiny", stage_end=0) == "step-00000006-tiny-stage-0_end.pth"
    assert checkpoint_name(20, "tiny") == "step-00000020-tiny.pth"
    assert checkpoint_name(123456789, "r") == "step-123456789-r.pth"
    assert checkpoint_dir(tmp_path) == tmp_path / CHECKPOINT_SUBDIR
    assert checkpoint_dir(str(tmp_path)) == tmp_path / "checkpoints"


def test_step_from_name() -> None:
    assert _step_from_name(Path("/x/step-00000014-tiny-stage-1_end.pth")) == 14
    assert _step_from_name(Path("step-00000020-my-run.pth")) == 20


def test_find_latest_checkpoint_picks_highest_step_including_stage_end_names(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path, "tiny") is None
    d = checkpoint_dir(tmp_path)
    d.mkdir()
    names = [
        checkpoint_name(6, "tiny", stage_end=0),
        checkpoint_name(10, "tiny"),
        checkpoint_name(14, "tiny", stage_end=1),
        checkpoint_name(99, "other"),
        "step-00000500-tiny.txt",
    ]
    for n in names:
        (d / n).touch()

    def latest(run_name: str) -> str:
        found = find_latest_checkpoint(tmp_path, run_name)
        assert found is not None
        return found.name

    assert latest("tiny") == "step-00000014-tiny-stage-1_end.pth"
    (d / checkpoint_name(9, "tiny")).touch()  # lexically later ("9" > "1") but a lower step
    assert latest("tiny") == "step-00000014-tiny-stage-1_end.pth"
    (d / checkpoint_name(20, "tiny")).touch()
    assert latest("tiny") == "step-00000020-tiny.pth"
    assert latest("other") == "step-00000099-other.pth"
    assert find_latest_checkpoint(tmp_path, "nothing") is None


@pytest.mark.parametrize("foreign", ["step-00000099-tiny-v2.pth", "step-00000099-other-tiny.pth"])
def test_find_latest_checkpoint_ignores_runs_with_a_longer_name(tmp_path: Path, foreign: str) -> None:
    d = checkpoint_dir(tmp_path)
    d.mkdir()
    (d / checkpoint_name(1, "tiny")).touch()
    (d / foreign).touch()
    found = find_latest_checkpoint(tmp_path, "tiny")
    assert found is not None and found.name == "step-00000001-tiny.pth"


def test_should_save_checkpoint() -> None:
    saves = [
        s for s in range(1, 21) if should_save_checkpoint(s, max_steps=20, save_step_interval=8, save_last_step=True)
    ]
    assert saves == [8, 16, 20]
    assert should_save_checkpoint(3, max_steps=20, save_step_interval=8, save_last_step=True, stage_end=True)
    assert not should_save_checkpoint(20, max_steps=20, save_step_interval=0, save_last_step=False)
    assert should_save_checkpoint(20, max_steps=20, save_step_interval=0, save_last_step=True)
    assert should_save_checkpoint(25, max_steps=20, save_step_interval=0, save_last_step=True)


def test_unwrap_strips_compile_wrapper(tiny_model: RecurrentGPT) -> None:
    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    assert _unwrap(tiny_model) is tiny_model
    assert _unwrap(Wrapper(tiny_model)) is tiny_model


def _train_one_step(model: RecurrentGPT) -> tuple[ELLISAdam, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randint(0, 512, (2, 16))
    opt = ELLISAdam(get_param_groups(model, 4e-5), lr=1e-3, betas=(0.9, 0.95))
    loss = model(x, labels=x, num_steps_pair=(0, 2))["loss"]
    assert loss is not None
    loss.backward()
    opt.step()
    opt.zero_grad()
    return opt, x


def test_save_load_forward_bit_identical(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    opt, x = _train_one_step(tiny_model)
    path = checkpoint_dir(tmp_path) / checkpoint_name(1, "tiny")
    rng_state = collect_rng_state()
    extra: dict[str, Any] = {"step": 1, "stage": 0, "rng": rng_state, "config": {"seed": 42}}
    save_checkpoint(backend, path, tiny_model, opt, extra)
    assert path.exists()

    torch.manual_seed(999)
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    fresh_opt = ELLISAdam(get_param_groups(fresh, 4e-5), lr=1e-3, betas=(0.9, 0.95))
    rest = load_checkpoint(backend, path, fresh, fresh_opt)
    assert rest["step"] == 1 and rest["stage"] == 0 and rest["config"] == {"seed": 42}
    assert torch.equal(rest["rng"]["torch"], rng_state["torch"])
    assert "model" not in rest and "optimizer" not in rest

    tiny_model.eval()
    fresh.eval()
    with torch.no_grad():  # the recurrent state is drawn from the global RNG: seed before each forward
        torch.manual_seed(7)
        a = tiny_model(x, return_logits=True, num_steps_pair=(0, 2))["logits"]
        torch.manual_seed(7)
        b = fresh(x, return_logits=True, num_steps_pair=(0, 2))["logits"]
    assert a is not None and b is not None and torch.equal(a, b)
    for (n1, p1), (n2, p2) in zip(tiny_model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)

    # optimizer state restored: the next step on both optimizers is identical
    for p, q in zip(tiny_model.parameters(), fresh.parameters()):
        assert torch.equal(opt.state[p]["exp_avg"], fresh_opt.state[q]["exp_avg"])
        assert torch.equal(opt.state[p]["exp_avg_sq"], fresh_opt.state[q]["exp_avg_sq"])
        assert opt.state[p]["step"].item() == fresh_opt.state[q]["step"].item() == 1
    assert fresh_opt.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    tiny_model.train()
    fresh.train()
    for model, optimizer in ((tiny_model, opt), (fresh, fresh_opt)):
        torch.manual_seed(3)
        loss = model(x, labels=x, num_steps_pair=(0, 2))["loss"]
        assert loss is not None
        loss.backward()
        optimizer.step()
    for p, q in zip(tiny_model.parameters(), fresh.parameters()):
        assert torch.equal(p, q)


def test_load_without_optimizer(tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT) -> None:
    opt, _ = _train_one_step(tiny_model)
    path = tmp_path / "c.pth"
    save_checkpoint(backend, path, tiny_model, opt, {"step": 1})
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    rest = load_checkpoint(backend, path, fresh)
    assert rest == {"step": 1}
    assert all(torch.equal(a, b) for a, b in zip(tiny_model.parameters(), fresh.parameters()))


def test_compiled_wrapper_is_unwrapped_for_state_dict(
    tmp_path: Path, backend: SingleDeviceBackend, tiny_model: RecurrentGPT
) -> None:
    """State-dict keys must not carry an `_orig_mod.` prefix when the model is a torch.compile wrapper."""

    class Wrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self._orig_mod = inner

    opt, _ = _train_one_step(tiny_model)
    path = tmp_path / "w.pth"
    save_checkpoint(backend, path, Wrapper(tiny_model), opt, {"step": 1})
    keys = set(backend.load_checkpoint(path)["model"].keys())
    assert keys == set(tiny_model.state_dict().keys())
    fresh = build_model(TINY_MODEL_ARCHITECTURE)
    load_checkpoint(backend, path, Wrapper(fresh))
    assert all(torch.equal(a, b) for a, b in zip(tiny_model.parameters(), fresh.parameters()))


def test_rng_state_round_trip(tmp_path: Path, backend: SingleDeviceBackend) -> None:
    random.seed(5)
    torch.manual_seed(5)
    state = collect_rng_state()
    assert set(state) >= {"python", "torch"}
    expected = (random.random(), torch.rand(3))
    backend.save_checkpoint(tmp_path / "rng.pth", {"rng": state})
    random.seed(77)
    torch.manual_seed(77)
    restore_rng_state(backend.load_checkpoint(tmp_path / "rng.pth")["rng"])
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])
    assert ("cuda" in state) == torch.cuda.is_available()


def test_restore_rng_state_without_cuda_entry() -> None:
    torch.manual_seed(1)
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    first = torch.rand(2)
    restore_rng_state(state)
    assert torch.equal(torch.rand(2), first)
