# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from torchao/optim/subclass_8bit.py (pytorch/ao 6792133): imports point at this package; the DTensor /
# distributed-checkpoint ops (view, c10d all-gather and wait, is_pinned, slice) are left out.

import torch
from torch import Tensor
from torch.serialization import add_safe_globals
from torch.utils._python_dispatch import return_and_correct_aliasing

from .base import TorchAOBaseTensor
from .quant_utils import (
    create_dynamic_map,
    dequant_with_qmap,
    quantize_8bit_with_qmap,
    scale_tensor,
)

aten = torch.ops.aten

# Lazy initialization to avoid meta device issues during import
from functools import lru_cache


@lru_cache(maxsize=1)
def get_qmap_signed():
    return tuple(create_dynamic_map(signed=True))


@lru_cache(maxsize=1)
def get_qmap_unsigned():
    return tuple(create_dynamic_map(signed=False))


class OptimState8bit(TorchAOBaseTensor):
    tensor_attrs = ["codes", "scale", "qmap"]

    # dtype only acts as an appearance dtype to work with the rest of PyTorch
    @staticmethod
    def __new__(
        cls,
        codes: Tensor,
        scale: Tensor,
        qmap: Tensor,
        signed: bool,
        dtype: torch.dtype | None = None,
    ):
        return Tensor._make_wrapper_subclass(
            cls, codes.shape, device=codes.device, dtype=dtype
        )

    def __init__(
        self,
        codes: Tensor,
        scale: Tensor,
        qmap: Tensor,
        signed: bool,
        dtype: torch.dtype | None = None,
    ):
        """Create quantized 8-bit optimizer state as proposed in https://arxiv.org/abs/2110.02861

        Args
            codes: quantized 8-bit data stored as uint8. Has the same shape as the original float tensor.
            scale: scale data for block-wise quantization.
            qmap: lookup table that maps between quantized value (code) and float value.
            signed: whether the tensor is signed or unsigned.

        NOTE: To get block-wise scale, the original float tensor is first reshape to (-1, block_size).
        Thus, the last dimension of the original float tensor is not necessarily divisible by block size.
        Given `codes` and `scale`, `block_size` is calculated as `codes.numel() // scale.numel()`.
        """
        assert codes.dtype is torch.uint8
        assert scale.ndim == 1
        assert qmap.dtype is torch.float32
        self.codes = codes
        self.scale = scale
        self.qmap = qmap
        self.signed = signed
        self.block_size = codes.numel() // scale.numel()

    def __tensor_flatten__(self):
        return self.tensor_attrs, [self.signed, self.dtype]

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None
    ):
        return cls(
            *[tensor_data_dict[name] for name in cls.tensor_attrs], *tensor_attributes
        )

    def dequantize(self, output_dtype=None):
        float_data = dequant_with_qmap(self.codes, self.qmap, self.scale)
        if output_dtype is not None:
            float_data = float_data.to(output_dtype)
        return float_data

    @classmethod
    def zeros(
        cls,
        shape,
        signed: bool = True,
        block_size: int = 256,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        codes = torch.zeros(shape, dtype=torch.uint8, device=device)
        scale = torch.zeros(codes.numel() // block_size, device=device)
        qmap_list = get_qmap_signed() if signed else get_qmap_unsigned()
        qmap = torch.tensor(qmap_list, dtype=torch.float32, device=device)
        return cls(codes, scale, qmap, signed, dtype=dtype)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(signed={self.signed}, block_size={self.block_size}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"requires_grad={self.requires_grad})"
        )


@OptimState8bit.implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst = args[0]
    src = args[1]

    if isinstance(dst, OptimState8bit) and isinstance(src, OptimState8bit):
        assert dst.signed == src.signed and dst.block_size == src.block_size
        dst.codes.copy_(src.codes)
        dst.scale.copy_(src.scale)
        # qmap should be the same, don't need to copy

    elif isinstance(dst, OptimState8bit):
        scaled_src, scale = scale_tensor(src, dst.block_size)
        codes = quantize_8bit_with_qmap(scaled_src, dst.qmap)
        dst.codes.copy_(codes)
        dst.scale.copy_(scale)

    else:
        dst.copy_(src.dequantize())

    return dst


@OptimState8bit.implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    # only change the appearance dtype
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", None)
    out = OptimState8bit(
        args[0].codes.to(device=device),
        args[0].scale.to(device=device),
        args[0].qmap.to(device=device),
        args[0].signed,
        dtype=dtype,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@OptimState8bit.implements(aten.lerp.Scalar)
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, OptimState8bit) else x for x in args]
    return func(*args, **kwargs)


add_safe_globals([OptimState8bit])
