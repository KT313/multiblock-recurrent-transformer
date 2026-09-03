# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the single-device backend: device pick, autocast, clipping, checkpoint round trip, no-op collectives,
RNG state round trip, device transfer and pin_memory.
"""

import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from training.backend import BACKENDS, get_backend
from training.backend.base import Backend
from training.backend.single_device import SingleDeviceBackend
from training.backend.single_device import PRECISIONS, _set_torch_flags


def test_registry_and_default_device() -> None:
    backend = get_backend("single_device", precision="32")
    assert isinstance(backend, SingleDeviceBackend)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert backend.device.type == expected
    assert (backend.world_size, backend.rank, backend.is_main) == (1, 0, True)
    assert backend.pin_memory == (expected == "cuda")
    assert set(BACKENDS) == {"single_device"}


def test_single_device_backend_implements_the_protocol() -> None:
    protocol_methods = {n for n in vars(Backend) if not n.startswith("_")}
    assert protocol_methods <= set(dir(SingleDeviceBackend))
    backend: Backend = SingleDeviceBackend(device="cpu", precision="32")  # static check: satisfies the Protocol
    assert backend.device.type == "cpu"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("fabric")


def test_cpu_fallback_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="falling back to CPU"):
        backend = SingleDeviceBackend()
    assert backend.device == torch.device("cpu")


def test_explicit_device_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = SingleDeviceBackend(device="cpu")
    assert backend.device == torch.device("cpu")


def test_invalid_precision_raises() -> None:
    assert PRECISIONS == ("bf16-mixed", "32")
    with pytest.raises(ValueError, match="precision"):
        SingleDeviceBackend(device="cpu", precision="fp16")


def test_set_torch_flags() -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    _set_torch_flags()
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    torch.set_float32_matmul_precision("highest")
    SingleDeviceBackend(device="cpu")  # the constructor applies the flags too
    assert torch.get_float32_matmul_precision() == "high"


def test_autocast_bf16_dtype() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="bf16-mixed")
    with backend.autocast():
        out = torch.nn.functional.linear(torch.ones(2, 4), torch.ones(3, 4))
    assert out.dtype == torch.bfloat16


def test_autocast_fp32_is_nullcontext() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    assert isinstance(backend.autocast(), nullcontext)
    with backend.autocast():
        out = torch.nn.functional.linear(torch.ones(2, 4), torch.ones(3, 4))
    assert out.dtype == torch.float32


def test_setup_model_moves_and_returns_module() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    model = torch.nn.Linear(3, 2)
    assert backend.setup_model(model) is model
    assert next(model.parameters()).device == backend.device
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    assert backend.setup_optimizer(opt) is opt


def test_setup_model_compile_wraps_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    `compile_model=True` goes through `torch.compile(model, dynamic=True)`; stubbed so the test stays fast.
    """

    calls: list[tuple[torch.nn.Module, bool]] = []

    def fake_compile(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
        calls.append((model, dynamic))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    model = torch.nn.Linear(3, 2)
    assert backend.setup_model(model, compile_model=True) is model
    assert calls == [(model, True)]


def test_backward_and_clip_grad_norm_value() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    x = torch.tensor([[3.0, 4.0]])
    backend.backward(model(x).sum())  # grad = x -> norm 5
    assert model.weight.grad is not None and torch.equal(model.weight.grad, x)
    total = backend.clip_grad_norm(model, max_norm=1.0)
    assert total.item() == pytest.approx(5.0)
    assert model.weight.grad.norm().item() == pytest.approx(1.0, rel=1e-5)


def test_clip_grad_norm_tolerates_non_finite_gradients() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("inf"), 1.0]])
    assert torch.isinf(backend.clip_grad_norm(model, max_norm=1.0))  # error_if_nonfinite=False


def test_noop_collectives() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    t = torch.arange(3.0)
    assert backend.all_reduce(t) is t
    assert backend.all_reduce(t, op="sum") is t
    backend.barrier()
    assert isinstance(backend.no_sync(torch.nn.Linear(1, 1)), nullcontext)


def test_checkpoint_round_trip_bit_identical(tmp_path: Path) -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 8)
    rng = torch.get_rng_state()
    state = {"model": model.state_dict(), "step": 7, "rng": rng, "config": {"a": [1, 2]}}
    path = tmp_path / "nested" / "dir" / "ckpt.pth"
    backend.save_checkpoint(path, state)
    assert path.exists()
    loaded = backend.load_checkpoint(str(path))
    assert loaded["step"] == 7
    assert loaded["config"] == {"a": [1, 2]}
    assert torch.equal(loaded["rng"], rng)
    for k, v in model.state_dict().items():
        assert torch.equal(loaded["model"][k], v)


def test_seed_everything_is_reproducible() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    backend.seed_everything(123)
    a = (torch.rand(4), random.random(), np.random.rand())
    backend.seed_everything(123)
    b = (torch.rand(4), random.random(), np.random.rand())
    assert torch.equal(a[0], b[0]) and a[1:] == b[1:]


def test_rng_state_round_trip(tmp_path: Path) -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    random.seed(5)
    torch.manual_seed(5)
    state = backend.rng_state()
    assert set(state) == {"python", "torch"}, "a CPU backend stores no CUDA state, whatever the machine has"
    expected = (random.random(), torch.rand(3))
    backend.save_checkpoint(tmp_path / "rng.pth", {"rng": state})  # survives the checkpoint round trip
    random.seed(77)
    torch.manual_seed(77)
    backend.set_rng_state(backend.load_checkpoint(tmp_path / "rng.pth")["rng"])
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])


def test_set_rng_state_without_cuda_entry() -> None:
    """
    A state written on a CPU-only machine (no "cuda" key) restores the python and torch generators.
    """

    backend = SingleDeviceBackend(device="cpu", precision="32")
    torch.manual_seed(1)
    random.seed(1)
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    first = (torch.rand(2), random.random())
    backend.set_rng_state(state)
    assert torch.equal(torch.rand(2), first[0]) and random.random() == first[1]


def test_rng_state_of_a_cpu_backend_never_touches_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A CPU backend neither stores CUDA generator state nor consults a "cuda" entry it is handed.
    """

    backend = SingleDeviceBackend(device="cpu", precision="32")
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *_: pytest.fail("CUDA generator read by a CPU backend"))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_: pytest.fail("CUDA generator set by a CPU backend"))
    state = backend.rng_state()
    assert set(state) == {"python", "torch"}
    torch.manual_seed(3)
    expected = torch.rand(2)
    torch.manual_seed(3)
    backend.set_rng_state({**backend.rng_state(), "cuda": torch.zeros(1, dtype=torch.uint8)})  # not consulted
    assert torch.equal(torch.rand(2), expected)


@pytest.mark.gpu
def test_rng_state_of_a_cuda_backend_holds_its_own_device_only() -> None:
    """
    The restore is what is measured: capture, draw, let the generator move on, restore, draw again. With no
    reseeding in between, only `set_rng_state` can make the second draw repeat the first (the old version reseeded
    before each draw and passed with the restore stubbed out).
    """

    backend = SingleDeviceBackend(device="cuda:0", precision="32")
    torch.cuda.manual_seed(9)
    state = backend.rng_state()
    assert isinstance(state["cuda"], torch.Tensor), "one generator state, not the per-GPU list"
    first = torch.rand(2, device=backend.device)
    assert not torch.equal(torch.rand(2, device=backend.device), first), "the generator moved on"
    backend.set_rng_state(state)
    assert torch.equal(torch.rand(2, device=backend.device), first)


def test_save_checkpoint_is_atomic_and_leaves_no_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The final name appears only after a complete write; a failing save leaves neither a partial file nor a .tmp.
    """

    backend = SingleDeviceBackend(device="cpu", precision="32")
    path = tmp_path / "ckpt" / "step.pth"
    backend.save_checkpoint(path, {"a": torch.ones(2)})
    assert sorted(p.name for p in path.parent.iterdir()) == ["step.pth"]

    def failing_save(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError):
        backend.save_checkpoint(path, {"a": torch.zeros(2)})
    assert sorted(p.name for p in path.parent.iterdir()) == ["step.pth"], "the previous checkpoint is untouched"
    assert torch.equal(backend.load_checkpoint(path)["a"], torch.ones(2))


def test_to_device_moves_to_the_backend_device() -> None:
    backend = SingleDeviceBackend(device="cpu", precision="32")
    t = torch.arange(3)
    moved = backend.to_device(t)
    assert moved.device == backend.device and torch.equal(moved, t)
    assert moved is t  # `.to` on the same device returns the tensor itself (no copy)


def test_pin_memory_follows_the_device_type() -> None:
    assert SingleDeviceBackend(device="cpu", precision="32").pin_memory is False
    assert SingleDeviceBackend(device="meta", precision="32").pin_memory is False


@pytest.mark.gpu
def test_gpu_device_and_autocast() -> None:
    backend = SingleDeviceBackend()
    assert backend.device == torch.device("cuda:0")
    assert backend.pin_memory is True
    assert backend.to_device(torch.ones(2)).device == backend.device
    assert "cuda" in backend.rng_state()
    model = backend.setup_model(torch.nn.Linear(4, 4))
    with backend.autocast():
        out = model(torch.ones(2, 4, device=backend.device))
    assert out.dtype == torch.bfloat16
