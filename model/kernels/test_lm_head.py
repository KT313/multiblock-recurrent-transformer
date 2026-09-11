"""Behavioral contracts for chunked linear CE and its bounded custom backward."""
from collections.abc import Callable
from typing import cast

import pytest

pytest.importorskip("triton")
import torch
from torch import Tensor

from model.model import linear_cross_entropy
from .lm_head import fused_linear_cross_entropy, supports


def test_cpu_inputs_raise_with_disable_guidance() -> None:
    x = torch.randn(2, 3, 16, requires_grad=True)
    head = torch.nn.Linear(16, 43, bias=False)
    labels = torch.randint(0, 40, (2, 3))
    assert not supports(x, head, labels, 0.7, -100)
    with pytest.raises(RuntimeError, match='use_custom_kernels: false'):
        fused_linear_cross_entropy(x, head, labels, 0.7, -100)


@pytest.mark.gpu
@pytest.mark.parametrize('tokens,vocab,scale', [(31, 127, 1.0), (2065, 1024, 0.25), (2049, 32768, 1.0)])
@pytest.mark.parametrize('compiled', [False, True])
def test_bf16_forward_backward(tokens: int, vocab: int, scale: float, compiled: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(123)
    x = torch.randn(1, tokens, 64, device='cuda', requires_grad=True)
    head = torch.nn.Linear(64, vocab, bias=False, device='cuda')
    labels = torch.randint(0, vocab - 3, (1, tokens), device='cuda')
    labels[:, ::11] = -100
    function: Callable[..., Tensor] = fused_linear_cross_entropy
    if compiled:
        function = torch.compile(function, fullgraph=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        assert supports(x, head, labels, scale, -100)
        expected = linear_cross_entropy(x, head, labels, scale, -100)
        actual = function(x, head, labels, scale, -100)
    eg = torch.autograd.grad(expected * 0.37, (x, head.weight))
    ag = torch.autograd.grad(actual * 0.37, (x, head.weight))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    for candidate, reference in zip(ag, eg):
        # Same scale-aware rule as the benchmark, substantially tighter absolute
        # floor so tiny gradients cannot make a zero/incorrect backward pass.
        assert (candidate - reference).abs().max() <= 2e-6 + 0.01 * reference.abs().max()
        assert candidate.dtype == reference.dtype
    assert actual.dtype == torch.float32
    assert torch.count_nonzero(ag[0][:, ::11]) == 0


@pytest.mark.gpu
def test_all_ignored_and_tied_weight() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(67)
    embedding = torch.nn.Embedding(1024, 64, device='cuda')
    head = torch.nn.Linear(64, 1024, bias=False, device='cuda')
    head.weight = embedding.weight
    ids = torch.randint(0, 1000, (1, 37), device='cuda')
    labels = torch.full_like(ids, -100)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = fused_linear_cross_entropy(embedding(ids), head, labels, 1.0, -100)
    assert torch.isnan(loss)
    torch.autograd.backward(loss)
    assert embedding.weight.grad is not None
    assert torch.count_nonzero(embedding.weight.grad) == 0
    assert head.weight is embedding.weight
    labels = torch.randint(0, 1000, ids.shape, device='cuda')
    gradients = []
    for function in (linear_cross_entropy, fused_linear_cross_entropy):
        embedding.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = function(embedding(ids), head, labels, 0.1, -100)
        torch.autograd.backward(loss)
        assert embedding.weight.grad is not None
        gradients.append(embedding.weight.grad.clone())
    assert (gradients[0] - gradients[1]).abs().max() <= 2e-6 + 0.01 * gradients[0].abs().max()


@pytest.mark.gpu
def test_native_bf16_and_support_boundary() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(1, 17, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    head = torch.nn.Linear(64, 127, bias=False, device='cuda', dtype=torch.bfloat16)
    labels = torch.randint(0, 120, (1, 17), device='cuda')
    assert supports(x, head, labels, 1.0, -100)
    assert not supports(x.transpose(1, 2).contiguous().transpose(1, 2), head, labels, 1.0, -100)
    assert not supports(x.float(), head.float(), labels, 1.0, -100)
    assert not supports(x, head, labels.int(), 1.0, -100)
    head = head.bfloat16()
    with torch.autocast('cuda', dtype=torch.float16):
        assert not supports(x, head, labels, 1.0, -100)
        with pytest.raises(RuntimeError, match='use_custom_kernels: false'):
            fused_linear_cross_entropy(x, head, labels, 1.0, -100)
    hook = head.register_forward_hook(lambda module, args, output: cast(Tensor, output) + 0.25)
    try:
        assert not supports(x, head, labels, 1.0, -100)
        with pytest.raises(RuntimeError, match='use_custom_kernels: false'):
            fused_linear_cross_entropy(x, head, labels, 1.0, -100)
    finally:
        hook.remove()
    losses = [function(x, head, labels, 1.0, -100) for function in (linear_cross_entropy, fused_linear_cross_entropy)]
    gradients = [torch.autograd.grad(loss, (x, head.weight)) for loss in losses]
    torch.testing.assert_close(losses[0], losses[1], rtol=2e-6, atol=2e-6)
    for reference, actual in zip(*gradients):
        assert actual.dtype == torch.bfloat16
        assert (actual - reference).abs().max() <= 2e-6 + 0.01 * reference.abs().max()


@pytest.mark.gpu
@pytest.mark.parametrize('compiled', [False, True])
def test_strided_labels_forward_backward(compiled: bool) -> None:
    """The CE kernel receives stride-one labels even when native CE accepts views."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(918)
    x = torch.randn(1, 2065, 64, device='cuda', requires_grad=True)
    head = torch.nn.Linear(64, 1024, bias=False, device='cuda')
    storage = torch.randint(0, 1024, (1, 4130), device='cuda')
    labels = storage[:, ::2]
    labels[:, ::11] = -100
    assert labels.stride(-1) == 2
    function: Callable[..., Tensor] = fused_linear_cross_entropy
    if compiled:
        function = torch.compile(function, fullgraph=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = linear_cross_entropy(x, head, labels, 0.25, -100)
        actual = function(x, head, labels, 0.25, -100)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    eg = torch.autograd.grad(expected * 0.37, (x, head.weight))
    ag = torch.autograd.grad(actual * 0.37, (x, head.weight))
    for candidate, reference in zip(ag, eg):
        assert (candidate - reference).abs().max() <= 2e-6 + 0.01 * reference.abs().max()
        assert candidate.dtype == reference.dtype
    assert torch.count_nonzero(ag[0][:, ::11]) == 0
