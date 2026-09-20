"""Projection, activation rounding and all three MLP gradients, including compilation."""
from collections.abc import Callable
from typing import cast

import pytest

pytest.importorskip("triton")
import torch
from torch import Tensor

from model.layers.mlp import mlp_projection as reference

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('input_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape', [(2, 17, 64, 128), (1, 257, 512, 1024)])
@pytest.mark.slow
def test_projection_parity(compiled: bool, input_dtype: torch.dtype, shape: tuple[int, int, int, int]) -> None:
    from .mlp import mlp_projection, supports
    torch.manual_seed(17)
    b, t, e, i = shape
    x = torch.randn(b, t, e, device='cuda', dtype=input_dtype, requires_grad=True)
    fc = torch.nn.Linear(e, 2 * i, bias=False, device='cuda')
    proj = torch.nn.Linear(i, e, bias=False, device='cuda')
    nonlin = torch.nn.SiLU()
    fn: Callable[..., Tensor] = mlp_projection
    if compiled:
        fn = torch.compile(fn, dynamic=True, fullgraph=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        assert supports(x, fc, proj, nonlin)
        expected = reference(x, fc, proj, nonlin)
        actual = fn(x, fc, proj, nonlin)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    dy = torch.randn_like(actual)
    variables = (x, fc.weight, proj.weight)
    expected_grads = torch.autograd.grad(expected, variables, dy)
    actual_grads = torch.autograd.grad(actual, variables, dy)
    for got, wanted in zip(actual_grads, expected_grads):
        # Same tolerance rule as the bf16 harness, plus exact dtype/finite checks.
        assert got.dtype == wanted.dtype and torch.isfinite(got).all()
        assert (got - wanted).abs().max() <= 0.01 + 0.01 * wanted.abs().max()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        repeated = fn(x, fc, proj, nonlin)
    repeated_grads = torch.autograd.grad(repeated, variables, dy)
    for got, prior in zip(repeated_grads, actual_grads):
        torch.testing.assert_close(got, prior, atol=0, rtol=0)


def test_unsupported_layout_and_nonlinearity_raise() -> None:
    from .mlp import mlp_projection, supports
    x = torch.randn(1, 9, 128, device='cuda')[..., ::2]
    fc = torch.nn.Linear(64, 256, bias=False, device='cuda')
    proj = torch.nn.Linear(128, 64, bias=False, device='cuda')
    class AlteredLinear(torch.nn.Linear):
        def forward(self, input: Tensor) -> Tensor:
            return super().forward(input) * 2

    altered_fc = AlteredLinear(64, 256, bias=False, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        assert not supports(x.contiguous(), altered_fc, proj, torch.nn.SiLU())
        assert not supports(x, fc, proj, torch.nn.SiLU())
        assert not supports(x.contiguous(), fc, proj, torch.nn.ReLU())
        for value, linear, activation in [(x, fc, torch.nn.SiLU()), (x.contiguous(), altered_fc, torch.nn.SiLU()), (x.contiguous(), fc, torch.nn.ReLU())]:
            with pytest.raises(RuntimeError, match='use_custom_kernels: false'):
                mlp_projection(value, linear, proj, activation)


def test_only_gate_up_activation_is_saved() -> None:
    from .mlp import mlp_projection
    x = torch.randn(1, 17, 64, device='cuda', requires_grad=True)
    fc = torch.nn.Linear(64, 256, bias=False, device='cuda')
    proj = torch.nn.Linear(128, 64, bias=False, device='cuda')
    saved: list[tuple[int, ...]] = []

    def pack(tensor: Tensor) -> Tensor:
        saved.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.autograd.grad(mlp_projection(x, fc, proj, torch.nn.SiLU()).sum(), (x, fc.weight, proj.weight))
    assert (17, 256) in saved
    assert (17, 128) not in saved




@pytest.mark.parametrize('hook_kind', ['forward', 'forward_pre', 'backward', 'backward_pre'])
def test_module_hooks_raise_instead_of_being_ignored(hook_kind: str) -> None:
    from .mlp import mlp_projection, supports
    x = torch.randn(1, 9, 64, device='cuda', requires_grad=True)
    fc = torch.nn.Linear(64, 256, bias=False, device='cuda')
    proj = torch.nn.Linear(128, 64, bias=False, device='cuda')
    nonlin = torch.nn.SiLU()
    if hook_kind == 'forward':
        handle = fc.register_forward_hook(lambda module, inputs, output: cast(Tensor, output) * 0.5)
    elif hook_kind == 'forward_pre':
        handle = fc.register_forward_pre_hook(lambda module, inputs: (cast(tuple[Tensor, ...], inputs)[0] * 0.5,))
    elif hook_kind == 'backward':
        handle = fc.register_full_backward_hook(lambda module, inputs, outputs: (cast(tuple[Tensor, ...], inputs)[0] * 0.5,))
    else:
        handle = fc.register_full_backward_pre_hook(lambda module, outputs: (cast(tuple[Tensor, ...], outputs)[0] * 0.5,))
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            assert not supports(x, fc, proj, nonlin)
            with pytest.raises(RuntimeError, match='use_custom_kernels: false'):
                mlp_projection(x, fc, proj, nonlin)
    finally:
        handle.remove()
