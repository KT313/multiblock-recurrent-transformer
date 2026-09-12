# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Precision selection must not change ambient-context or parameter-storage semantics."""
import pytest
import torch

from model.execution import ExecutionPolicy
from training.backend.single_device import SingleDeviceBackend


@pytest.mark.parametrize("precision", [None, "32", "bf16-mixed"])
@pytest.mark.parametrize("ambient", [False, True])
def test_policy_preserves_nested_autocast_and_exception(precision: str | None, ambient: bool) -> None:
    policy = ExecutionPolicy(precision)
    before = torch.is_autocast_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=ambient):
        with pytest.raises(RuntimeError, match="injected"), policy.autocast("cpu"):
            assert torch.is_autocast_enabled("cpu") == (ambient or precision == "bf16-mixed")
            raise RuntimeError("injected")
        assert torch.is_autocast_enabled("cpu") == ambient
    assert torch.is_autocast_enabled("cpu") == before


@pytest.mark.parametrize("precision", ["32", "bf16-mixed"])
def test_backend_forward_and_backward_boundaries(precision: str) -> None:
    backend = SingleDeviceBackend(device="cpu", precision=precision)
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    observed: list[bool] = []
    parameter.register_hook(  # type: ignore[no-untyped-call]  # Tensor.register_hook is unannotated in torch
        lambda grad: observed.append(torch.is_autocast_enabled("cpu"))
    )
    with backend.autocast():
        loss = (parameter @ parameter).sum()
        assert loss.dtype == (torch.bfloat16 if precision == "bf16-mixed" else torch.float32)
    backend.backward(loss)
    assert observed == [False] and parameter.dtype == torch.float32
    assert parameter.grad is not None and parameter.grad.dtype == torch.float32
    assert backend.execution_policy == ExecutionPolicy(precision)


def test_policy_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        ExecutionPolicy("fp16")
