# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.norms.RMSNorm`.
"""

import pytest
import torch

from model.layers.norms import RMSNorm


def test_shape_and_unit_rms() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 5, 16) * 7 + 3
    y = norm(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.pow(2).mean(-1), torch.ones(2, 5), atol=1e-5, rtol=0)


def test_matches_hand_formula() -> None:
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[3.0, -4.0, 0.0, 0.0]])
    rms = torch.sqrt(torch.tensor((9 + 16) / 4) + 1e-6)
    torch.testing.assert_close(norm(x), x / rms * norm.weight)


def test_scale_invariance() -> None:
    norm = RMSNorm(8)
    x = torch.randn(3, 8)
    torch.testing.assert_close(norm(x), norm(x * 100), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_input_uses_fp32_statistics(dtype: torch.dtype) -> None:
    """
    Statistics are computed in float32; the output dtype follows the usual promotion with the weight (float32 weight
    -> float32 output, half weight -> half output).
    """

    norm = RMSNorm(64)
    x = (torch.randn(4, 64) * 1e-3).to(dtype)  # squares underflow in half precision but not in float32
    y = norm(x)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, norm(x.float()), atol=1e-2, rtol=1e-2)
    assert torch.isfinite(y).all()
    y_half = norm.to(dtype)(x)
    assert y_half.dtype == dtype
    torch.testing.assert_close(y_half.float(), y, atol=1e-2, rtol=1e-2)


def test_norm_helper_is_the_unweighted_normalization() -> None:
    norm = RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        norm.weight.fill_(5.0)
    x = torch.randn(3, 4)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm._norm(x), expected)
    torch.testing.assert_close(norm(x), norm._norm(x) * 5.0)


def test_reset_parameters_restores_ones() -> None:
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(3.0)
    norm.reset_parameters()
    assert torch.equal(norm.weight, torch.ones(8))


def test_gradient_flows_to_weight_and_input() -> None:
    norm = RMSNorm(8)
    x = torch.randn(2, 8, requires_grad=True)
    norm(x).sum().backward()
    assert norm.weight.grad is not None and x.grad is not None
    assert norm.weight.grad.abs().sum() > 0


# --- the bf16 residual stream switch (`autocast_output`) --------------------------------------------------------------


def test_autocast_output_rounds_to_the_autocast_dtype_once() -> None:
    """
    Under autocast the switched norm returns the autocast dtype, and its value is the fp32 result (normalization
    times weight, as the plain norm computes it) rounded once.
    """

    plain, switched = RMSNorm(16), RMSNorm(16, autocast_output=True)
    with torch.no_grad():
        plain.weight.uniform_(0.5, 1.5)
        switched.weight.copy_(plain.weight)
    x = torch.randn(3, 16) * 4
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = switched(x)
        reference = plain(x)
    assert out.dtype == torch.bfloat16 and reference.dtype == torch.float32
    assert torch.equal(out, reference.to(torch.bfloat16))


def test_autocast_output_is_a_no_op_without_autocast() -> None:
    plain, switched = RMSNorm(16), RMSNorm(16, autocast_output=True)
    x = torch.randn(3, 16)
    assert switched(x).dtype == torch.float32
    assert torch.equal(switched(x), plain(x))
    x_half = x.to(torch.bfloat16)
    assert switched(x_half).dtype == torch.float32  # bf16 input times the fp32 weight promotes, as in the plain norm
    assert torch.equal(switched(x_half), plain(x_half))


def test_autocast_output_gradient_reaches_weight_and_input() -> None:
    norm = RMSNorm(8, autocast_output=True)
    x = torch.randn(2, 8, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        norm(x).float().sum().backward()
    assert norm.weight.grad is not None and norm.weight.grad.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
