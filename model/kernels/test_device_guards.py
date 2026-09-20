"""Device/stream ownership, exception restoration and fake-safe kernel dispatch."""
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

pytest.importorskip('triton')


@pytest.mark.parametrize('kind', ['mlp', 'lm_head', 'rope'])
def test_fake_implementations_do_not_select_cuda(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from . import lm_head, mlp, rope

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('fake implementation selected a CUDA device')

    mode = FakeTensorMode()
    def fake(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> Tensor:
        return FakeTensor(mode, torch.empty(shape, device='meta', dtype=dtype), torch.device('cuda:0'))

    x, fc, proj = fake((17, 64)), fake((256, 64)), fake((64, 128))
    qkv, table = fake((1, 17, 192)), fake((1, 17, 1, 16, 2), torch.float32)
    labels = fake((17,), torch.int64)
    tangent = fake((1, 17, 2, 32))
    # FakeTensor construction can perform PyTorch's own device discovery. The
    # operation bodies themselves must never exchange/initialize a CUDA device.
    monkeypatch.setattr(torch.cuda, '_exchange_device', forbidden)
    monkeypatch.setattr(torch.cuda, '_maybe_exchange_device', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    with mode:
        if kind == 'mlp':
            out, gu = mlp._forward(x, fc, proj)
            grads = mlp._backward_op(out, x, fc, proj, gu)
        elif kind == 'lm_head':
            out = lm_head.loss_forward(x, fc, labels, 1.0, -100)
            grads = lm_head.loss_backward(x, fc, labels, out, 1.0, -100)
        else:
            out = rope.packed_forward(qkv, None, table, 2)
            grads = rope.packed_backward(tangent, tangent, tangent, table, True, torch.bfloat16)
        assert all(value.device == x.device for value in grads)


class _ObservedLaunch:
    def __init__(self, kernel: Any, device: torch.device, stream: torch.cuda.Stream, fail: bool) -> None:
        self.kernel = kernel
        self.device = device
        self.stream = stream
        self.fail = fail
        self.calls = 0

    def __getitem__(self, grid: Any) -> Callable[..., Any]:
        def launch(*args: Any, **kwargs: Any) -> Any:
            assert torch.cuda.current_device() == self.device.index
            assert torch.cuda.current_stream(self.device) == self.stream
            self.calls += 1
            if self.fail:
                raise RuntimeError('injected synchronous launch failure')
            return self.kernel[grid](*args, **kwargs)
        return launch


@pytest.mark.gpu
@pytest.mark.parametrize('other_device', [False, True], ids=['same-device', 'two-device'])
@pytest.mark.parametrize('kind', ['mlp', 'lm_head', 'rope'])
@pytest.mark.parametrize('backward', [False, True], ids=['forward', 'backward'])
@pytest.mark.parametrize('fail', [False, True], ids=['success', 'exception'])
def test_real_launch_device_stream_and_restoration(
    kind: str, backward: bool, fail: bool, other_device: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import lm_head, mlp, rope

    if other_device and torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices; single-GPU execution cannot establish device switching')
    caller = torch.cuda.current_device()
    device = torch.device('cuda', (caller + 1) % torch.cuda.device_count() if other_device else caller)
    original_stream = torch.cuda.current_stream(device)
    caller_stream = torch.cuda.current_stream(caller)
    stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]  # torch stub gap
    # Leave a nondefault current stream selected on the input device while the
    # caller's current device can remain different. All producers share it.
    with torch.cuda.device(device):
        torch.cuda.set_stream(stream)
    try:
        with torch.cuda.device(device):
            torch.manual_seed(97)
            x = torch.randn(17, 64, device=device, dtype=torch.bfloat16)
            fc = torch.randn(256, 64, device=device, dtype=torch.bfloat16)
            proj = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
            labels = torch.randint(0, 256, (17,), device=device)
            qkv = torch.randn(1, 17, 192, device=device, dtype=torch.bfloat16)
            from model.layers.attention import precompute_freqs_cis
            table = precompute_freqs_cis(32, 17, 50000).to(device)
            tangent = torch.randn(1, 17, 2, 32, device=device, dtype=torch.bfloat16)
            upstream = torch.ones((), device=device)
            gu = torch.mm(x, fc.t())

        fn: Callable[[], Any]
        module: ModuleType
        if kind == 'mlp':
            kernel_name, module = '_activation', mlp
            fn = (lambda: mlp._backward_op(x, x, fc, proj, gu)) if backward else (lambda: mlp._forward(x, fc, proj))
        elif kind == 'lm_head':
            kernel_name, module = '_cross_entropy', lm_head
            fn = (lambda: lm_head.loss_backward(x, fc, labels, upstream, 1.0, -100)) if backward else (
                lambda: lm_head.loss_forward(x, fc, labels, 1.0, -100))
        else:
            kernel_name, module = ('_backward_kernel' if backward else '_forward_kernel'), rope
            fn = (lambda: rope.packed_backward(tangent, tangent, tangent, table, True, torch.bfloat16)) if backward else (
                lambda: rope.packed_forward(qkv, None, table, 2))

        # Known-good current-input-device run is an exact, same-arithmetic oracle.
        with torch.cuda.device(device):
            expected = fn()
        observed = _ObservedLaunch(getattr(module, kernel_name), device, stream, fail)
        if kind == 'rope':
            from torch.library import wrap_triton as original_wrap
            monkeypatch.setattr(rope, 'wrap_triton',
                                lambda kernel: observed if kernel is observed.kernel else original_wrap(kernel))
        else:
            monkeypatch.setattr(module, kernel_name, observed)
        assert torch.cuda.current_device() == caller
        if fail:
            with pytest.raises(RuntimeError, match='injected synchronous launch failure'):
                fn()
        else:
            actual = fn()
            # Consumer remains on the same nondefault stream. Synchronize only
            # its final observation event, never before kernel launch/consumer.
            with torch.cuda.device(device):
                expected_values = expected if isinstance(expected, tuple) else (expected,)
                actual_values = actual if isinstance(actual, tuple) else (actual,)
                equal = [torch.eq(a, b).all() for a, b in zip(actual_values, expected_values, strict=True)]
                done = stream.record_event()
            done.synchronize()
            assert all(value.item() for value in equal)
        assert observed.calls == 1
        assert torch.cuda.current_device() == caller
        assert torch.cuda.current_stream(device) == stream
        if other_device:
            assert torch.cuda.current_stream(caller) == caller_stream
    finally:
        with torch.cuda.device(device):
            torch.cuda.set_stream(original_stream)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('kind', ['mlp', 'lm_head', 'rope'])
@pytest.mark.parametrize('compiled', [False, True])
def test_two_device_nondefault_stream_native_autograd_parity(kind: str, compiled: bool) -> None:
    """Includes RoPE's native bias reduction after its independently guarded op."""
    if torch.cuda.device_count() < 2:
        pytest.skip('requires two CUDA devices, including compiled backward on a nondefault stream')
    from . import lm_head, mlp, rope
    from model.layers.attention import precompute_freqs_cis, qkv_bias_rope
    from model.layers.mlp import mlp_projection
    from model.model import linear_cross_entropy

    caller = torch.cuda.current_device()
    device = torch.device('cuda', (caller + 1) % torch.cuda.device_count())
    stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]  # torch stub gap
    prior = torch.cuda.current_stream(device)
    with torch.cuda.device(device):
        torch.cuda.set_stream(stream)
    try:
        with torch.cuda.device(device):
            x = torch.randn(1, 17, 64, device=device, requires_grad=True)
            fc = torch.nn.Linear(64, 256, bias=False, device=device)
            proj = torch.nn.Linear(128, 64, bias=False, device=device)
            labels = torch.randint(0, 256, (1, 17), device=device)
            qkv = torch.randn(1, 17, 192, device=device, dtype=torch.bfloat16, requires_grad=True)
            bias = torch.randn(2, 1, 2, 32, device=device, requires_grad=True)
            table = precompute_freqs_cis(32, 17, 50000).to(device)
        function: Callable[..., Any]
        reference: Callable[..., Any]
        arguments: tuple[Any, ...]
        parameters: tuple[Tensor, ...]
        if kind == 'mlp':
            function, reference = mlp.mlp_projection, mlp_projection
            arguments, parameters = (x, fc, proj, torch.nn.SiLU()), (x, fc.weight, proj.weight)
        elif kind == 'lm_head':
            function, reference = lm_head.fused_linear_cross_entropy, linear_cross_entropy
            arguments, parameters = (x, fc, labels, 1.0, -100), (x, fc.weight)
        else:
            function, reference = rope.qkv_bias_rope, qkv_bias_rope
            arguments, parameters = (bias, qkv, table, 2), (qkv, bias)
        if compiled:
            function = torch.compile(function, fullgraph=True)
        with torch.cuda.device(device), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = reference(*arguments)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = function(*arguments)
        assert torch.cuda.current_device() == caller
        actual_values = actual if isinstance(actual, tuple) else (actual,)
        expected_values = expected if isinstance(expected, tuple) else (expected,)
        with torch.cuda.device(device):
            tangents = tuple(torch.randn_like(value) for value in expected_values)
            expected_grads = torch.autograd.grad(expected_values, parameters, tangents)
        # The selected device is deliberately wrong specifically at backward.
        actual_grads = torch.autograd.grad(actual_values, parameters, tangents)
        assert torch.cuda.current_device() == caller
        assert torch.cuda.current_stream(device) == stream
        with torch.cuda.device(device):
            done = stream.record_event()
        done.synchronize()
        for actual_value, expected_value in zip(actual_values, expected_values, strict=True):
            torch.testing.assert_close(actual_value, expected_value, atol=2e-6, rtol=2e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            assert actual_grad.device == expected_grad.device == device
            assert (actual_grad - expected_grad).abs().max() <= 2e-6 + 0.01 * expected_grad.abs().max()
    finally:
        with torch.cuda.device(device):
            torch.cuda.set_stream(prior)
