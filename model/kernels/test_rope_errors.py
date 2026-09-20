"""Direct autograd failure guidance without changing the traceable RoPE operations."""
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.library import wrap_triton
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

pytest.importorskip('triton')

from model.layers.attention import precompute_freqs_cis
from . import rope
from .runtime import CustomKernelError, DISABLE_HINT


class _FailingLaunch:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.calls = 0

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> None:
            self.calls += 1
            raise self.failure
        return launch


class _FailingReduction(TorchDispatchMode):  # type: ignore[no-untyped-call]  # torch __init_subclass__ stub gap
    def __init__(self, failure: BaseException) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # torch dispatch-mode stub gap
        self.failure = failure
        self.calls = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func == torch.ops.aten.sum.dim_IntList and args[0].ndim == 4 and args[1] == [0]:
            self.calls += 1
            raise self.failure
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('api', ['backward', 'grad'])
@pytest.mark.parametrize('bias_enabled', [False, True])
@pytest.mark.parametrize('stage', ['launch', 'allocation', 'reduction', 'translated', 'interrupt', 'exit'])
@pytest.mark.parametrize('device', ['fake', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_direct_backward_error_boundary(
    api: str, bias_enabled: bool, stage: str, device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if stage == 'reduction' and not bias_enabled:
        pytest.skip('bias-free backward has no bias reduction')
    mode = FakeTensorMode() if device == 'fake' else None
    # Fake CPU tensors exercise the registered autograd callback and meta body
    # without initializing CUDA; the CUDA cases use the strict public forward.
    with mode if mode is not None else torch.enable_grad():
        tensor_device = 'cpu' if mode is not None else 'cuda'
        qkv = torch.randn(1, 9, 192, device=tensor_device, requires_grad=True)
        bias = torch.randn(2, 1, 2, 32, device=tensor_device, requires_grad=True) if bias_enabled else None
        table = precompute_freqs_cis(32, 9, 50000).to(tensor_device)
        output = rope.packed_forward(qkv, bias, table, 2) if mode is not None else rope.qkv_bias_rope(bias, qkv, table, 2)
        gradients = tuple(torch.ones_like(value) for value in output)
        parameters = (qkv,) if bias is None else (qkv, bias)
        failure: BaseException
        if stage == 'translated':
            failure = CustomKernelError(f'already translated. {DISABLE_HINT}')
        elif stage == 'interrupt':
            failure = KeyboardInterrupt('user interrupt')
        elif stage == 'exit':
            failure = SystemExit('user exit')
        elif stage == 'allocation':
            failure = torch.OutOfMemoryError('injected backward allocation failure')
        else:
            failure = RuntimeError(f'injected backward {stage} failure')
        original_cause = RuntimeError('original launch cause') if stage == 'translated' else None
        failure.__cause__ = original_cause
        launch = _FailingLaunch(failure)
        reduction = _FailingReduction(failure)
        allocation_calls = []
        if stage == 'allocation':
            original_empty = torch.empty

            def failed_empty(*args: Any, **kwargs: Any) -> Tensor:
                if args and tuple(args[0]) == (1, 9, 192):
                    allocation_calls.append(1)
                    raise failure
                return original_empty(*args, **kwargs)

            monkeypatch.setattr(torch, 'empty', failed_empty)
        elif stage != 'reduction':
            def failed_wrap(kernel: Any) -> Any:
                return launch if kernel is rope._backward_kernel else wrap_triton(kernel)

            monkeypatch.setattr(rope, 'wrap_triton', failed_wrap)
        expected_type = type(failure) if stage in ('translated', 'interrupt', 'exit') else CustomKernelError
        with pytest.raises(expected_type) as info, reduction if stage == 'reduction' else torch.enable_grad():
            if api == 'backward':
                torch.autograd.backward(output, gradients)
            else:
                torch.autograd.grad(output, parameters, gradients)
        if stage in ('translated', 'interrupt', 'exit'):
            assert info.value is failure
            assert info.value.__cause__ is original_cause
        else:
            assert info.value.__cause__ is failure
            assert 'RoPE/QKV backward' in str(info.value)
        if stage not in ('interrupt', 'exit'):
            assert str(info.value).count('use_custom_kernels: false') == 1
        assert (reduction.calls if stage == 'reduction' else len(allocation_calls) if stage == 'allocation' else launch.calls) == 1
        assert qkv.grad is None
        assert bias is None or bias.grad is None
