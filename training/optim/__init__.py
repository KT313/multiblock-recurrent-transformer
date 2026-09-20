# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
The optimizer of the thesis runs, ELLISAdam (a port of the upstream implementation with its four options:
`update_clipping`, `atan_adam`, `running_init`, `decouple_wd`), its memory-saving twin ELLISAdam8bit, plus the
parameter-group split.

ELLISAdam8bit runs the same update but stores the two Adam moments block-wise quantised to 8 bits (torchao's
`OptimState8bit`, vendored under `training/optim/torchao/`): 2 bytes per parameter of optimizer state instead of 8.
The embedding group keeps fp32 moments (`build_optimizer` pins it: the 8-bit optimizer paper found embeddings the
one layer that destabilises under quantised state), as do tensors too small for the block quantisation. The update
maths dequantises into fp32 temporaries, so the ELLIS options apply unchanged; a checkpoint written by one of the
two optimizers cannot resume the other (`state_bits` is a parameter-group hyperparameter, compared on resume).
"""

from training.optim.ellis import ELLISAdam, ELLISAdam8bit, _validate_lr
from training.optim.factory import ELLIS_ONLY_OPTIONS, ELLIS_OPTIMIZERS, OPTIMIZERS, build_optimizer, set_lr
from training.optim.groups import EMBEDDING_GROUP, _pin_embedding_group_to_fp32, get_param_groups
from training.optim.state import (
    STATE_8BIT_BLOCK_SIZE,
    STATE_8BIT_MIN_NUMEL,
    _quantizable,
    _quantized_zeros,
    dequantized_state,
    is_quantized_state,
)
from training.optim.torchao import OptimState8bit
from training.optim.update import _adamw_group_update, _compiled_adamw_group_update, _single_tensor_modded_adamw

__all__ = [
    "ELLIS_ONLY_OPTIONS", "ELLIS_OPTIMIZERS", "OPTIMIZERS", "EMBEDDING_GROUP", "STATE_8BIT_BLOCK_SIZE",
    "STATE_8BIT_MIN_NUMEL", "ELLISAdam", "ELLISAdam8bit", "OptimState8bit", "build_optimizer",
    "dequantized_state", "get_param_groups", "is_quantized_state", "set_lr", "_validate_lr",
    "_pin_embedding_group_to_fp32", "_quantizable", "_quantized_zeros", "_adamw_group_update",
    "_compiled_adamw_group_update", "_single_tensor_modded_adamw",
]
