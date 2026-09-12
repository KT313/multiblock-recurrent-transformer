# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Larger bounded CE chunks with vendor FP32 GEMM accumulation for dW."""
from __future__ import annotations

from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.nn import Module
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError as error:
    raise ImportError("CUDA custom kernels require Triton") from error

from ..layers.init import Linear
from .runtime import CustomKernelError, DISABLE_HINT, kernel_errors, kernel_namespace, require_supported

_NAMESPACE = kernel_namespace(__name__, "lm_head_stats_v1")


__all__ = ['fused_linear_cross_entropy', 'supports']

CHUNK_TOKENS = 2048


@triton.jit  # type: ignore[untyped-decorator]
def _cross_entropy(  # type: ignore[no-untyped-def]
    LOGITS, LABELS, LOSSES, GRAD, COUNT, UPSTREAM,
    VOCAB: tl.constexpr, SCALE: tl.constexpr, IGNORE: tl.constexpr,
    BACKWARD: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    z = tl.load(LOGITS + row * VOCAB + cols, cols < VOCAB, other=-float('inf')).to(tl.float32)
    z = z * SCALE
    target = tl.load(LABELS + row)
    valid = target != IGNORE
    maximum = tl.max(z, 0)
    shifted = z - maximum
    denominator = tl.sum(tl.exp(shifted), 0)
    log_denom = tl.log(denominator)
    if BACKWARD:
        count = tl.load(COUNT).to(tl.float32)
        upstream = tl.load(UPSTREAM)
        # Match fp32 log_softmax followed by NLL backward, then the cast through
        # the bf16 projection output. Ignored rows always contribute zero.
        probability = tl.exp(shifted - log_denom)
        dz = (probability - (cols == target).to(tl.float32)) * (upstream / count)
        dz = tl.where(valid, dz * SCALE, 0.0)
        tl.store(GRAD + row * VOCAB + cols, dz, cols < VOCAB)
    else:
        # Stay in shifted coordinates: adding the maximum back can round away
        # log_denom before a similarly large target score is subtracted.
        target_shifted = tl.sum(tl.where(cols == target, shifted, 0.0), 0)
        loss = tl.where(valid, log_denom - target_shifted, 0.0)
        tl.store(LOSSES + row, loss)


def supports(
    x: Tensor, head: Module, labels: Tensor, logit_scale: float, ignore_index: int,
    reduction: Literal["mean", "sum"] = "mean",
) -> bool:
    if reduction not in ("mean", "sum"):
        return False
    if not isinstance(head, torch.nn.Linear):
        return False
    if type(head) not in (torch.nn.Linear, Linear) or cast(Tensor | None, head.bias) is not None:
        return False
    if head._forward_hooks or head._forward_pre_hooks or head._backward_hooks or head._backward_pre_hooks:
        return False
    weight = head.weight
    if not (x.is_cuda and x.device == weight.device == labels.device):
        return False
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1] or x.numel() == 0:
        return False
    if not x.is_contiguous() or not weight.is_contiguous() or labels.numel() != x.numel() // x.shape[-1]:
        return False
    if labels.dtype != torch.int64 or not (0 < weight.shape[0] <= 65536) or logit_scale <= 0:
        return False
    autocast_enabled = torch.is_autocast_enabled('cuda')
    autocast_bf16 = autocast_enabled and torch.get_autocast_dtype('cuda') == torch.bfloat16
    return ((autocast_bf16 and x.dtype in (torch.float32, torch.bfloat16) and weight.dtype in (torch.float32, torch.bfloat16))
            or (not autocast_enabled and x.dtype == weight.dtype == torch.bfloat16))


@torch.library.custom_op(f"{_NAMESPACE}::loss_forward", mutates_args=())
@kernel_errors("LM head")
def loss_forward(x: Tensor, weight: Tensor, labels: Tensor, scale: float, ignore: int, sum_loss: bool = False) -> Tensor:
    rows, width = x.shape
    vocab = weight.shape[0]
    losses = torch.empty(rows, device=x.device, dtype=torch.float32)
    count = (labels != ignore).sum()
    with torch.autocast('cuda', enabled=False):
        for start in range(0, rows, CHUNK_TOKENS):
            logits = torch.mm(x[start:start + CHUNK_TOKENS], weight.t())
            _cross_entropy[(logits.shape[0],)](
                logits, labels[start:], losses[start:], logits, count, count,
                vocab, scale, ignore, False, triton.next_power_of_2(vocab),
                num_warps=32 if vocab >= 16384 else 8, enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
            )
            del logits
    return losses.sum() if sum_loss else losses.sum() / count


@loss_forward.register_fake
def _forward_fake(x: Tensor, weight: Tensor, labels: Tensor, scale: float, ignore: int, sum_loss: bool = False) -> Tensor:
    return torch.empty((), device=x.device, dtype=torch.float32)


@torch.library.custom_op(f"{_NAMESPACE}::loss_backward", mutates_args=())
@kernel_errors("LM head")
def loss_backward(
    x: Tensor, weight: Tensor, labels: Tensor, upstream: Tensor, scale: float, ignore: int, sum_loss: bool = False,
) -> tuple[Tensor, Tensor]:
    rows, width = x.shape
    vocab = weight.shape[0]
    dx = torch.empty_like(x)
    dw = torch.empty(weight.shape, device=weight.device, dtype=torch.float32)
    count = torch.ones((), device=labels.device, dtype=torch.int64) if sum_loss else (labels != ignore).sum()
    with torch.autocast('cuda', enabled=False):
        for start in range(0, rows, CHUNK_TOKENS):
            chunk = x[start:start + CHUNK_TOKENS]
            logits = torch.mm(chunk, weight.t())
            # Safe in-place overwrite: one program owns the entire vocabulary row.
            _cross_entropy[(chunk.shape[0],)](
                logits, labels[start:], upstream, logits, count, upstream,
                vocab, scale, ignore, True, triton.next_power_of_2(vocab),
                num_warps=32 if vocab >= 16384 else 8, enable_fp_fusion=False,  # pyright: ignore[reportCallIssue]
            )
            torch.mm(logits, weight, out=dx[start:start + CHUNK_TOKENS])
            # FP32 beta accumulation keeps chunk partials unrounded without a
            # separate partial buffer/add kernel. The final BF16 cast is native.
            torch.addmm(dw, logits.t(), chunk, out_dtype=torch.float32,
                        beta=0 if start == 0 else 1, out=dw)
            del logits
    return dx, dw.to(weight.dtype)


@loss_backward.register_fake
def _backward_fake(
    x: Tensor, weight: Tensor, labels: Tensor, upstream: Tensor, scale: float, ignore: int, sum_loss: bool = False,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
    x, weight, labels, scale, ignore, sum_loss = inputs
    ctx.save_for_backward(x, weight, labels)
    ctx.scale = scale
    ctx.ignore = ignore
    ctx.sum_loss = sum_loss


def _backward(ctx: Any, upstream: Tensor) -> tuple[Tensor, Tensor, None, None, None, None]:
    x, weight, labels = ctx.saved_tensors
    dx, dw = loss_backward(x, weight, labels, upstream, ctx.scale, ctx.ignore, ctx.sum_loss)
    return dx, dw, None, None, None, None


torch.library.register_autograd(loss_forward, _backward, setup_context=_setup_context)


def fused_linear_cross_entropy(
    x: Tensor, head: Module, labels: Tensor, logit_scale: float, ignore_index: int,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"unsupported loss reduction: {reduction!r}")
    try:
        require_supported(supports(x, head, labels, logit_scale, ignore_index), "LM head",
                          "Requires contiguous CUDA BF16 projection inputs (or BF16 autocast), int64 labels and a supported biasless head without hooks.")
        weight = cast(torch.nn.Linear, head).weight
        return cast(Tensor, loss_forward(
            x.reshape(-1, x.shape[-1]).to(torch.bfloat16), weight.to(torch.bfloat16),
            labels.reshape(-1).contiguous(), logit_scale, ignore_index, reduction == "sum",
        ))
    except CustomKernelError:
        raise
    except Exception as error:
        raise CustomKernelError(f"lm_head custom kernel failed: {error}. {DISABLE_HINT}") from error
