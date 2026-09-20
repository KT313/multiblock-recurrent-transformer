"""Exercise masked heads, multi-block programs, and unused/frozen gradient routes."""
import pytest

pytest.importorskip("triton")
import torch

from model.layers.attention import precompute_freqs_cis, qkv_bias_rope as reference
from .rope import qkv_bias_rope


@pytest.mark.gpu
@pytest.mark.slow
def test_multiblock_padded_heads_and_partial_gradients() -> None:
    torch.manual_seed(99)
    # E96 uses 16 positions/program: 4103 positions requires >256 blocks.
    qkv = torch.randn(1, 4103, 288, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(2, 1, 3, 32, device="cuda", requires_grad=True)
    base = precompute_freqs_cis(32, 117, 50000).cuda()
    table = base[:, torch.arange(4103, device="cuda") % 117]
    candidate = torch.compile(qkv_bias_rope, fullgraph=True)
    for freeze_bias in (False, True):
        active_bias = bias.detach() if freeze_bias else bias
        actual = candidate(active_bias, qkv, table, 3)[0]
        expected = reference(active_bias, qkv, table, 3)[0]
        parameters = (qkv,) if freeze_bias else (qkv, bias)
        actual_grads = torch.autograd.grad(actual.float().square().sum(), parameters)
        expected_grads = torch.autograd.grad(expected.float().square().sum(), parameters)
        assert actual_grads[0].dtype == qkv.dtype
        assert actual_grads[0].is_contiguous()
        for value, ref in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(value, ref, atol=1e-4, rtol=1e-5)
    # QKV itself is frozen; only the bias needs gradients, and K/V are unused.
    actual = candidate(bias, qkv.detach(), table, 3)[0]
    expected = reference(bias, qkv.detach(), table, 3)[0]
    torch.testing.assert_close(torch.autograd.grad(actual.float().sum(), bias)[0],
                               torch.autograd.grad(expected.float().sum(), bias)[0], atol=1e-4, rtol=1e-5)
