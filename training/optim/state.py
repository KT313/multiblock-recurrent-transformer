# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Create and inspect block-quantized optimizer moments."""

from typing import cast

from torch import Tensor

from training.optim.torchao import OptimState8bit

# ELLISAdam8bit quantises a moment only when its tensor has at least `STATE_8BIT_MIN_NUMEL` elements and a size that
# is a multiple of `STATE_8BIT_BLOCK_SIZE` (one absmax scale per block); smaller or odd-sized tensors (norm weights,
# biases) stay fp32. torchao's rules and defaults. Module constants, not settings: the block size is recoverable
# from the state itself and the threshold decides nothing a checkpoint has to agree on.
STATE_8BIT_BLOCK_SIZE = 256
STATE_8BIT_MIN_NUMEL = 4096


def _quantizable(param: Tensor) -> bool:
    return param.numel() >= STATE_8BIT_MIN_NUMEL and param.numel() % STATE_8BIT_BLOCK_SIZE == 0


def _quantized_zeros(param: Tensor, signed: bool) -> Tensor:
    # the appearance dtype is the parameter's: `Optimizer.load_state_dict` casts state to it, a no-op then
    zeros = OptimState8bit.zeros(param.shape, signed, STATE_8BIT_BLOCK_SIZE, param.device, dtype=param.dtype)
    return cast(Tensor, zeros)


def is_quantized_state(tensor: Tensor) -> bool:
    """
    Whether `tensor` is an 8-bit moment of ELLISAdam8bit (a plain tensor otherwise).
    """

    return isinstance(tensor, OptimState8bit)


def dequantized_state(tensor: Tensor) -> Tensor:
    """
    `tensor` as plain fp32 values: dequantised when it is an 8-bit moment, itself otherwise. For reading the state
    (the gradient metrics of the logger); the optimizer step uses the same on its way into the update maths.
    """

    if is_quantized_state(tensor):
        return tensor.dequantize()
    return tensor
