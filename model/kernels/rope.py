# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fresh Q/K/V storage and direct packed projection gradients.

Rotation and deterministic bias reduction preserve the native rounding boundaries.
Direct eager autograd translates synchronous backward failures with disable guidance.
Compiled runtime errors and later asynchronous CUDA failures can bypass this Python
boundary; the training-step error protection remains in place.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import is_fake
from torch.library import register_autograd, triton_op, wrap_triton
try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except ImportError as error:
    raise ImportError("CUDA custom kernels require Triton") from error

from .runtime import CustomKernelError, DISABLE_HINT, kernel_namespace, require_supported

_NAMESPACE = kernel_namespace(__name__, "rope")
MAX_HEAD_DIM = 256
BACKWARD_PROGRAMS = 256


def _tile(n_embd: int) -> tuple[int, int]:
    """
    Positions per program (a power of two) and warps: about 8K elements of q plus k per program (E = 512: 8
    positions, E = 1024: 4). Both kernels run at the DRAM roofline at 2048x8 with this; at 2048x2 and below they
    are launch-bound, and the tile does not matter.
    """

    block_positions = max(1, min(16, 4096 // n_embd))
    return 1 << (block_positions.bit_length() - 1), 4


@triton.jit
def _mul_rn(a, b):  # type: ignore[no-untyped-def]  # Triton JIT does not accept typing.Any annotations
    """
    `a * b` rounded to float32 as its own instruction: the reference computes each product in a separate kernel, and
    a fused multiply-add would round differently.
    """

    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit  # type: ignore[untyped-decorator]
def _rotate_tile(  # type: ignore[no-untyped-def]
    x,
    cos,
    sin,
    BLOCK_S: tl.constexpr,
    NH_PAD: tl.constexpr,
    HD: tl.constexpr,
):
    """
    Rotate every pair `(x[..., 2j], x[..., 2j+1])` of a float32 `[BLOCK_S, NH_PAD, HD]` tile by the `[BLOCK_S, HD // 2]`
    angles: `(re * cos - im * sin, im * cos + re * sin)` with each product rounded on its own, as the reference.
    """

    pairs = tl.reshape(x, [BLOCK_S, NH_PAD, HD // 2, 2])
    re, im = tl.split(pairs)
    cos3 = tl.expand_dims(cos, 1)
    sin3 = tl.expand_dims(sin, 1)
    out_re = _mul_rn(re, cos3) - _mul_rn(im, sin3)  # pyright: ignore[reportOperatorIssue]  # inline asm returns one tensor
    out_im = _mul_rn(im, cos3) + _mul_rn(re, sin3)
    return tl.reshape(tl.join(out_re, out_im), [BLOCK_S, NH_PAD, HD])


def _row_strides(x: Tensor) -> tuple[int, int, int]:
    return x.stride(0), x.stride(1), x.stride(2)


def _kernel_applies(qk_bias: Tensor | None, q: Tensor, k: Tensor, freqs_cis: Tensor) -> bool:
    if not (q.is_cuda and k.is_cuda and freqs_cis.is_cuda) or q.dim() != 4 or q.shape != k.shape or q.device != k.device or q.device != freqs_cis.device:
        return False
    if q.dtype != k.dtype or q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    B, S, nh, hd = q.shape
    if B * S == 0:
        return False
    if hd < 2 or hd > MAX_HEAD_DIM or hd & (hd - 1):
        return False
    if tuple(freqs_cis.shape) not in ((1, S, 1, hd // 2, 2), (B, S, 1, hd // 2, 2)) or freqs_cis.stride(4) != 1 or freqs_cis.stride(3) != 2:
        return False
    if freqs_cis.dtype != torch.float32:
        return False
    return qk_bias is None or (tuple(qk_bias.shape) == (2, 1, nh, hd) and qk_bias.device == q.device and qk_bias.dtype in (torch.float32, torch.bfloat16, torch.float16))


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b  # works on symbolic sizes under torch.compile, unlike triton.cdiv


@triton.jit  # type: ignore[untyped-decorator]
def _forward_kernel(  # type: ignore[no-untyped-def]
    Q,
    K,
    V,
    VO,
    BIAS,
    FREQS,
    QO,
    KO,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    freqs_stride_b,
    freqs_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    seq_len,
    n_positions,
    n_heads,
    HAS_BIAS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NH_PAD: tl.constexpr,
    HD: tl.constexpr,
):
    pid = tl.program_id(0)
    positions = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = positions < n_positions
    safe_positions = tl.minimum(positions, n_positions - 1)
    b = safe_positions // seq_len
    s = safe_positions % seq_len
    heads = tl.arange(0, NH_PAD)
    dims = tl.arange(0, HD)
    head_mask = heads < n_heads
    mask = valid[:, None, None] & head_mask[None, :, None] & (dims < HD)[None, None, :]

    # The table row of every position of the block: `(cos, sin)` interleaved, split into two [BLOCK_S, HD // 2].
    pair_index = tl.arange(0, HD // 2)
    components = tl.arange(0, 2)
    table = tl.load(
        FREQS + b[:, None, None] * freqs_stride_b + s[:, None, None] * freqs_stride_s + pair_index[None, :, None] * 2 + components[None, None, :],
        mask=valid[:, None, None] & (pair_index < HD // 2)[None, :, None] & (components < 2)[None, None, :],
        other=0.0,
    )
    cos, sin = tl.split(table)

    q_offsets = (
        b[:, None, None] * q_stride_b
        + s[:, None, None] * q_stride_s
        + heads[None, :, None] * q_stride_h
        + dims[None, None, :]
    )
    k_offsets = (
        b[:, None, None] * k_stride_b
        + s[:, None, None] * k_stride_s
        + heads[None, :, None] * k_stride_h
        + dims[None, None, :]
    )
    out_offsets = (
        b[:, None, None] * out_stride_b
        + s[:, None, None] * out_stride_s
        + heads[None, :, None] * out_stride_h
        + dims[None, None, :]
    )
    v_offsets = (b[:, None, None] * v_stride_b + s[:, None, None] * v_stride_s
                 + heads[None, :, None] * v_stride_h + dims[None, None, :])
    v = tl.load(V + v_offsets, mask=mask, other=0.0)
    tl.store(VO + out_offsets, v, mask=mask)
    q = tl.load(Q + q_offsets, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(K + k_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        bias_offsets = heads[:, None] * HD + dims[None, :]
        bias_mask = head_mask[:, None] & (dims < HD)[None, :]
        q_bias = tl.load(BIAS + bias_offsets, mask=bias_mask, other=0.0)
        k_bias = tl.load(BIAS + n_heads * HD + bias_offsets, mask=bias_mask, other=0.0)
        # The reference rounds the float32 sum to the activation dtype before the rotation.
        q = (q + q_bias[None, :, :]).to(Q.dtype.element_ty).to(tl.float32)
        k = (k + k_bias[None, :, :]).to(K.dtype.element_ty).to(tl.float32)
    q_out = _rotate_tile(q, cos, sin, BLOCK_S, NH_PAD, HD)
    k_out = _rotate_tile(k, cos, sin, BLOCK_S, NH_PAD, HD)
    tl.store(QO + out_offsets, q_out.to(QO.dtype.element_ty), mask=mask)
    tl.store(KO + out_offsets, k_out.to(KO.dtype.element_ty), mask=mask)


@triton.jit  # type: ignore[untyped-decorator]
def _backward_kernel(  # type: ignore[no-untyped-def]
    DQO,
    DKO,
    DVO,
    FREQS,
    DQKV,
    PARTIAL,
    dqo_stride_b,
    dqo_stride_s,
    dqo_stride_h,
    dko_stride_b,
    dko_stride_s,
    dko_stride_h,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    freqs_stride_b,
    freqs_stride_s,
    out_stride_b,
    out_stride_s,
    out_stride_h,
    seq_len,
    n_positions,
    n_heads,
    n_blocks,
    BIAS_GRAD: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NH_PAD: tl.constexpr,
    HD: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    heads = tl.arange(0, NH_PAD)
    dims = tl.arange(0, HD)
    head_mask = heads < n_heads
    pair_index = tl.arange(0, HD // 2)
    components = tl.arange(0, 2)
    dq_bias = tl.zeros([NH_PAD, HD], dtype=tl.float32)
    dk_bias = tl.zeros([NH_PAD, HD], dtype=tl.float32)
    for i in range(BLOCKS_PER_PROGRAM):
        block = pid + i * num_programs
        positions = block * BLOCK_S + tl.arange(0, BLOCK_S)
        # Positions past the end (and whole blocks past `n_blocks`) are clamped for the address arithmetic and
        # masked out of every load and store; their masked gradient of zero adds nothing to the bias gradient.
        valid = (positions < n_positions) & (block < n_blocks)
        safe_positions = tl.minimum(positions, n_positions - 1)
        b = safe_positions // seq_len
        s = safe_positions % seq_len
        mask = valid[:, None, None] & head_mask[None, :, None] & (dims < HD)[None, None, :]
        table = tl.load(
            FREQS + b[:, None, None] * freqs_stride_b + s[:, None, None] * freqs_stride_s + pair_index[None, :, None] * 2 + components[None, None, :],
            mask=valid[:, None, None] & (pair_index < HD // 2)[None, :, None] & (components < 2)[None, None, :],
            other=0.0,
        )
        cos, sin = tl.split(table)
        neg_sin = -sin
        dqo_offsets = (
            b[:, None, None] * dqo_stride_b
            + s[:, None, None] * dqo_stride_s
            + heads[None, :, None] * dqo_stride_h
            + dims[None, None, :]
        )
        dko_offsets = (
            b[:, None, None] * dko_stride_b
            + s[:, None, None] * dko_stride_s
            + heads[None, :, None] * dko_stride_h
            + dims[None, None, :]
        )
        out_offsets = (
            b[:, None, None] * out_stride_b
            + s[:, None, None] * out_stride_s
            + heads[None, :, None] * out_stride_h
            + dims[None, None, :]
        )
        dv_offsets = (b[:, None, None] * v_stride_b + s[:, None, None] * v_stride_s
                      + heads[None, :, None] * v_stride_h + dims[None, None, :])
        dv = tl.load(DVO + dv_offsets, mask=mask, other=0.0)
        tl.store(DQKV + out_offsets + 2 * n_heads * HD, dv, mask=mask)
        dq_out = tl.load(DQO + dqo_offsets, mask=mask, other=0.0).to(tl.float32)
        dk_out = tl.load(DKO + dko_offsets, mask=mask, other=0.0).to(tl.float32)
        # The inverse rotation, rounded to the input dtype where the reference's `.float()` cast rounds; the bias
        # gradient sums those rounded values, as autograd reduces the broadcast add after the cast.
        dq = _rotate_tile(dq_out, cos, neg_sin, BLOCK_S, NH_PAD, HD).to(DQKV.dtype.element_ty)
        dk = _rotate_tile(dk_out, cos, neg_sin, BLOCK_S, NH_PAD, HD).to(DQKV.dtype.element_ty)
        tl.store(DQKV + out_offsets, dq, mask=mask)
        tl.store(DQKV + out_offsets + n_heads * HD, dk, mask=mask)
        if BIAS_GRAD:
            dq_bias += tl.sum(dq.to(tl.float32), axis=0)
            dk_bias += tl.sum(dk.to(tl.float32), axis=0)
    if BIAS_GRAD:
        slot = PARTIAL + pid * (2 * n_heads * HD)
        bias_offsets = heads[:, None] * HD + dims[None, :]
        bias_mask = head_mask[:, None] & (dims < HD)[None, :]
        tl.store(slot + bias_offsets, dq_bias, mask=bias_mask)
        tl.store(slot + n_heads * HD + bias_offsets, dk_bias, mask=bias_mask)


def _split(qkv: Tensor, n_head: int) -> tuple[Tensor, Tensor, Tensor]:
    batch, length, width = qkv.shape
    embd = width // 3
    q, k, v = qkv.split(embd, dim=-1)  # type: ignore[no-untyped-call]  # torch stub gap
    shape = (batch, length, n_head, embd // n_head)
    return q.view(shape), k.view(shape), v.view(shape)


# Version both opaque op identities when changing registered backward structure:
# persisted AOT graphs can otherwise reuse the previous backward decomposition.
# triton_op also runs these bodies for fake tensors and AOT decomposition. Only
# real launches need a device scope; compiled kernels receive Inductor guards.
@triton_op(f"{_NAMESPACE}::forward", mutates_args=())
def packed_forward(qkv: Tensor, bias: Tensor | None, freqs: Tensor, n_head: int) -> tuple[Tensor, Tensor, Tensor]:
    with torch.cuda.device(qkv.device) if qkv.device.type != "meta" and not is_fake(qkv) else nullcontext():
        q, k, v = _split(qkv, n_head)
        batch, length, heads, dim = q.shape
        outputs = [torch.empty(q.shape, dtype=q.dtype, device=q.device) for _ in range(3)]
        qo, ko, vo = outputs
        block, warps = _tile(heads * dim)
        wrap_triton(_forward_kernel)[(_cdiv(batch * length, block),)](
            q, k, v, vo, bias.contiguous() if bias is not None else q, freqs, qo, ko,
            *_row_strides(q), *_row_strides(k), *_row_strides(v),
            freqs.stride(0) if freqs.shape[0] != 1 else 0, freqs.stride(1),
            *_row_strides(qo), length, batch * length, heads,
            HAS_BIAS=bias is not None, BLOCK_S=block, NH_PAD=triton.next_power_of_2(heads), HD=dim,
            num_warps=warps,  # pyright: ignore[reportCallIssue]  # Triton launch option
        )
        return qo, ko, vo


@triton_op(f"{_NAMESPACE}::backward", mutates_args=())
def packed_backward(
    dqo: Tensor, dko: Tensor, dvo: Tensor, freqs: Tensor, bias_grad: bool, in_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    with torch.cuda.device(dqo.device) if dqo.device.type != "meta" and not is_fake(dqo) else nullcontext():
        batch, length, heads, dim = dqo.shape
        # FlexAttention can return heads-first gradients; read their real strides.
        dqo = dqo.contiguous() if dqo.stride(-1) != 1 else dqo
        dko = dko.contiguous() if dko.stride(-1) != 1 else dko
        dvo = dvo.contiguous() if dvo.stride(-1) != 1 else dvo
        # Compiled consumers can promote tangents. Rotation must still round to the
        # original projection dtype before both dQKV storage and the bias reduction.
        dqkv = torch.empty((batch, length, 3 * heads * dim), dtype=in_dtype, device=dqo.device)
        block, warps = _tile(heads * dim)
        n_blocks = _cdiv(batch * length, block)
        programs = min(n_blocks, BACKWARD_PROGRAMS)
        partial = torch.empty((programs, 2, heads, dim) if bias_grad else (0,), dtype=torch.float32, device=dqo.device)
        wrap_triton(_backward_kernel)[(programs,)](
            # One output pointer is essential: separate aliased view arguments cause
            # functionalization to clone each strided storage span and reassemble it.
            dqo, dko, dvo, freqs, dqkv, partial,
            *_row_strides(dqo), *_row_strides(dko), *_row_strides(dvo),
            freqs.stride(0) if freqs.shape[0] != 1 else 0, freqs.stride(1),
            dqkv.stride(0), dqkv.stride(1), dim, length, batch * length, heads, n_blocks,
            BIAS_GRAD=bias_grad, BLOCKS_PER_PROGRAM=_cdiv(n_blocks, programs),
            BLOCK_S=block, NH_PAD=triton.next_power_of_2(heads), HD=dim,
            num_warps=warps,  # pyright: ignore[reportCallIssue]  # Triton launch option
        )
        return dqkv, partial


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    qkv, bias, freqs, _n_head = inputs
    ctx.save_for_backward(freqs)
    ctx.set_materialize_grads(True)
    ctx.in_dtype = qkv.dtype
    ctx.bias_shape = None if bias is None else tuple(bias.shape)
    ctx.bias_dtype = None if bias is None else bias.dtype


def _backward(ctx: Any, dq: Tensor, dk: Tensor, dv: Tensor) -> tuple[Tensor, Tensor | None, None, None]:
    # Keep the Triton operation traceable and include the separate bias reduction
    # in the eager error boundary. AOT traces the successful tensor operations;
    # generated runtime code and later asynchronous CUDA failures can bypass this
    # Python handler, so the training-step error boundary remains necessary.
    try:
        (freqs,) = ctx.saved_tensors
        bias_grad = ctx.bias_shape is not None and ctx.needs_input_grad[1]
        dqkv, partial = packed_backward(dq, dk, dv, freqs, bias_grad, ctx.in_dtype)
        dbias = partial.sum(0).view(cast(tuple[int, ...], ctx.bias_shape)).to(ctx.bias_dtype) if bias_grad else None
        return dqkv, dbias, None, None
    except CustomKernelError:
        raise
    except Exception as error:
        raise CustomKernelError(f"RoPE/QKV backward custom kernel failed: {error}. {DISABLE_HINT}") from error


register_autograd(f"{_NAMESPACE}::forward", _backward, setup_context=_setup_context)


def supports(qk_bias: Tensor | None, qkv: Tensor, freqs_cis: Tensor, n_head: int) -> bool:
    if qkv.ndim != 3 or n_head <= 0 or qkv.shape[-1] == 0 or qkv.shape[-1] % (3 * n_head) != 0:
        return False
    if not qkv.is_contiguous() or freqs_cis.requires_grad:
        return False
    q, k, _v = _split(qkv, n_head)
    return _kernel_applies(qk_bias, q, k, freqs_cis)


def qkv_bias_rope(qk_bias: Tensor | None, qkv: Tensor, freqs_cis: Tensor, n_head: int) -> tuple[Tensor, Tensor, Tensor]:
    try:
        require_supported(supports(qk_bias, qkv, freqs_cis, n_head), "RoPE/QKV",
                          "Requires contiguous CUDA QKV, supported bias and FP32 frequency-table shapes, no frequency gradients, and a power-of-two head dimension up to 256.")
        q, k, v = packed_forward(qkv, qk_bias, freqs_cis, n_head)
        return q, k, v
    except CustomKernelError:
        raise
    except Exception as error:
        raise CustomKernelError(f"rope custom kernel failed: {error}. {DISABLE_HINT}") from error
