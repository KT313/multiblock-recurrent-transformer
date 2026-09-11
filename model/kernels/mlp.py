# (c) 2026 Tobias Kerner. Apache-2.0.
"""Complete biasless SwiGLU MLP with explicit backward activation recomputation.

Projection GEMMs use torch.mm/cuBLAS; custom Triton kernels preserve each bf16
activation/gradient rounding boundary. Opaque custom-op boundaries ensure AOT
Autograd cannot retain the gated activation across forward/backward instead of
recomputing it. Round two keeps that lifetime while reducing activation program count and
reusing the dead DH allocation for H; all GEMMs remain vendor implementations.
"""
from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError as error:
    raise ImportError("CUDA custom kernels require Triton") from error

from ..layers.init import Linear


from .runtime import CustomKernelError, DISABLE_HINT, kernel_errors, kernel_namespace, require_supported

_NAMESPACE = kernel_namespace(__name__, "mlp")


@triton.jit  # type: ignore[untyped-decorator]
def _activation(GU, H, DH, DGU, N, I: tl.constexpr, BACKWARD: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    row = offsets // I
    col = offsets % I
    g = tl.load(GU + row * 2 * I + col, valid, 0).to(tl.float32)
    u = tl.load(GU + row * 2 * I + I + col, valid, 0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    silu = (g * sigmoid).to(GU.dtype.element_ty).to(tl.float32)
    if BACKWARD:
        dh = tl.load(DH + offsets, valid, 0).to(tl.float32)
        dsilu = (dh * u).to(GU.dtype.element_ty).to(tl.float32)
        dg = dsilu * (sigmoid * (1.0 + g * (1.0 - sigmoid)))
        du = dh * silu
        tl.store(DGU + row * 2 * I + col, dg, valid)
        tl.store(DGU + row * 2 * I + I + col, du, valid)
    # DH may alias H: load each incoming gradient before overwriting its location.
    tl.store(H + offsets, silu * u, valid)


@torch.library.custom_op(f"{_NAMESPACE}::forward", mutates_args=())
@kernel_errors("MLP")
def _forward(x: Tensor, fc: Tensor, proj: Tensor) -> tuple[Tensor, Tensor]:
    gu = torch.mm(x, fc.t())
    h = torch.empty((x.shape[0], proj.shape[1]), device=x.device, dtype=x.dtype)
    _activation[(triton.cdiv(h.numel(), 4096),)](
        gu, h, h, gu, h.numel(), proj.shape[1], False, 4096, enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]  # Triton launch option
    )
    return torch.mm(h, proj.t()), gu


@_forward.register_fake
def _forward_fake(x: Tensor, fc: Tensor, proj: Tensor) -> tuple[Tensor, Tensor]:
    return x.new_empty((x.shape[0], proj.shape[0])), x.new_empty((x.shape[0], fc.shape[0]))


@torch.library.custom_op(f"{_NAMESPACE}::backward", mutates_args=())
@kernel_errors("MLP")
def _backward_op(dy: Tensor, x: Tensor, fc: Tensor, proj: Tensor, gu: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    dy = dy.contiguous()
    dh = torch.mm(dy, proj)
    h = dh  # Recompute H in the dead DH allocation; neither is retained for backward.
    dgu = torch.empty_like(gu)
    _activation[(triton.cdiv(h.numel(), 4096),)](
        gu, h, dh, dgu, h.numel(), proj.shape[1], True, 4096, enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]  # Triton launch option
    )
    dproj = torch.mm(dy.t(), h)
    del h, dh
    dx = torch.mm(dgu, fc)
    dfc = torch.mm(dgu.t(), x)
    return dx, dfc, dproj


@_backward_op.register_fake
def _backward_fake(dy: Tensor, x: Tensor, fc: Tensor, proj: Tensor, gu: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(fc), torch.empty_like(proj)


def _setup_context(ctx: Any, inputs: tuple[Tensor, Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    x, fc, proj = inputs
    _, gu = output
    ctx.save_for_backward(x, fc, proj, gu)
    ctx.mark_non_differentiable(gu)


def _backward(ctx: Any, dy: Tensor, _dgu: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    x, fc, proj, gu = ctx.saved_tensors
    return cast(tuple[Tensor, Tensor, Tensor], _backward_op(dy, x, fc, proj, gu))


_forward.register_autograd(_backward, setup_context=_setup_context)


def supports(x: Tensor, fc: torch.nn.Module, proj: torch.nn.Module, nonlin: torch.nn.Module) -> bool:
    if type(fc) not in (torch.nn.Linear, Linear) or type(proj) not in (torch.nn.Linear, Linear):
        return False
    for module in (fc, proj, nonlin):
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:
            return False
    fc, proj = cast(torch.nn.Linear, fc), cast(torch.nn.Linear, proj)
    if type(nonlin) is not torch.nn.SiLU:
        return False
    # PyTorch annotates Linear.bias as Tensor even for bias=False.
    if nonlin.inplace or cast(Tensor | None, fc.bias) is not None or cast(Tensor | None, proj.bias) is not None:
        return False
    if not x.is_cuda or not x.is_contiguous() or x.ndim < 2 or x.numel() == 0:
        return False
    if x.dtype not in (torch.float32, torch.bfloat16):
        return False
    if not (torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16):
        return False
    return (
        fc.weight.device == x.device and proj.weight.device == x.device
        and fc.weight.dtype in (torch.float32, torch.bfloat16)
        and proj.weight.dtype in (torch.float32, torch.bfloat16)
        and fc.weight.is_contiguous() and proj.weight.is_contiguous()
        and fc.in_features == x.shape[-1] and fc.out_features == 2 * proj.in_features
        and proj.out_features == x.shape[-1]
    )


def mlp_projection(x: Tensor, fc: torch.nn.Module, proj: torch.nn.Module, nonlin: torch.nn.Module) -> Tensor:
    try:
        require_supported(supports(x, fc, proj, nonlin), "MLP",
                          "Requires contiguous CUDA tensors under BF16 autocast and supported biasless Linear/SiLU modules without hooks.")
        fc_linear, proj_linear = cast(torch.nn.Linear, fc), cast(torch.nn.Linear, proj)
        shape = x.shape
        y, _ = _forward(
            x.reshape(-1, shape[-1]).to(torch.bfloat16),
            fc_linear.weight.to(torch.bfloat16), proj_linear.weight.to(torch.bfloat16),
        )
        return cast(Tensor, y.view(shape))
    except CustomKernelError:
        raise
    except Exception as error:
        raise CustomKernelError(f"mlp custom kernel failed: {error}. {DISABLE_HINT}") from error
