# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for the single-device backend: device pick, autocast, clipping, checkpoint round trip, no-op collectives."""

import random
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from training.backend import BACKENDS, Backend, SingleDeviceBackend, get_backend
from training.backend.single_device import PRECISIONS, _set_torch_flags


def test_registry_and_default_device() -> None:
    backend = get_backend("single_device", precision="32")
    assert isinstance(backend, SingleDeviceBackend)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert backend.device.type == expected
    assert (backend.world_size, backend.rank, backend.is_main) == (1, 0, True)
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
    """`compile=True` goes through `torch.compile(model, dynamic=True)`; stubbed so the test stays fast."""
    calls: list[tuple[torch.nn.Module, bool]] = []

    def fake_compile(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
        calls.append((model, dynamic))
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    backend = SingleDeviceBackend(device="cpu", precision="32")
    model = torch.nn.Linear(3, 2)
    assert backend.setup_model(model, compile=True) is model
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


@pytest.mark.gpu
def test_gpu_device_and_autocast() -> None:
    backend = SingleDeviceBackend()
    assert backend.device == torch.device("cuda:0")
    model = backend.setup_model(torch.nn.Linear(4, 4))
    with backend.autocast():
        out = model(torch.ones(2, 4, device=backend.device))
    assert out.dtype == torch.bfloat16
