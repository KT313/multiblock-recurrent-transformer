# (c) 2026 Tobias Kerner. Apache-2.0.
"""PyTorch ZeRO-1 with validated restore and no redundant outer moment buffers.

Imported only when sharding is requested. PyTorch owns assignment, communication and global state indexing.
The two lifecycle adaptations here (outer-load hook and consolidation release) need version-specific tests.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Optimizer

from training.optim.ellis import ELLISAdam, ELLISAdam8bit
from training.optim.state import _quantizable, is_quantized_state


class ShardedOptimizer(ZeroRedundancyOptimizer):
    """Whole-parameter sharding, retaining the existing ELLIS per-tensor update semantics."""

    def __init__(
        self, params: Iterable[Tensor] | list[dict[str, Any]], *, optimizer_class: type[Optimizer], **options: Any
    ) -> None:
        if not dist.is_initialized():
            raise ValueError("optimizer_sharding='zero1' requires an initialized DDP process group")
        if optimizer_class not in (ELLISAdam, ELLISAdam8bit, torch.optim.AdamW):
            raise ValueError("zero1 supports ELLISAdam, ELLISAdam8bit and AdamW only")
        materialized: list[Any] = list(params)
        for index, item in enumerate(materialized):
            if isinstance(item, dict):
                materialized[index] = {**item, "params": list(item["params"])}
        tensors: list[Tensor] = [
            p for item in materialized for p in (item["params"] if isinstance(item, dict) else [item])
        ]
        if any(p.dtype != torch.float32 for p in tensors):
            raise ValueError("zero1 requires FP32 parameter storage; BF16 forward autocast is allowed")
        super().__init__(
            materialized, optimizer_class=optimizer_class, overlap_with_ddp=False,
            parameters_as_bucket_view=False, **options,
        )
        # ZRO loads owner-local state before calling Optimizer.load_state_dict on its outer wrapper.
        # Loading those CPU moments again would retain a second CUDA copy, unused by optimizer.step().
        self.register_load_state_dict_pre_hook(_omit_outer_moments)

    def validate_state_dict(self, state: dict[str, Any]) -> None:
        """Check saved evidence before ZRO mutates it or transfers tensors to the device."""
        saved_groups = state["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("sharded optimizer checkpoint has a different number of parameter groups")
        known: set[int] = set()
        for current, saved in zip(self.param_groups, saved_groups, strict=True):
            if len(current["params"]) != len(saved["params"]):
                raise ValueError("sharded optimizer checkpoint parameter group size differs")
            if isinstance(self.optim, ELLISAdam):
                self.optim._validate_group(saved, restored=True)
            for param, index in zip(current["params"], saved["params"], strict=True):
                if index in known:
                    raise ValueError("duplicate optimizer checkpoint parameter index")
                known.add(index)
                values = state["state"].get(index, {})
                if not isinstance(values, dict):
                    raise ValueError("optimizer parameter state must be a mapping")
                if not values:
                    continue  # lazy state, including the deliberately skipped first update
                quantized = (
                    isinstance(self.optim, ELLISAdam8bit) and current.get("state_bits") == 8 and _quantizable(param)
                )
                for key in ("exp_avg", "exp_avg_sq"):
                    moment = values.get(key)
                    if not isinstance(moment, Tensor) or moment.shape != param.shape or moment.dtype != torch.float32:
                        raise ValueError(f"invalid optimizer {key}: expected FP32 appearance dtype and parameter shape")
                    if is_quantized_state(moment) != quantized:
                        raise ValueError(f"optimizer {key} precision differs from the configured optimizer")
                step = values.get("step")
                if not isinstance(step, Tensor) or step.ndim != 0:
                    raise ValueError("optimizer step must be a scalar tensor")
                if isinstance(self.optim, ELLISAdam) and step.dtype != torch.int64:
                    raise ValueError("ELLIS optimizer step must be an int64 scalar")
        if set(state["state"]) - known:
            raise ValueError("optimizer checkpoint contains unknown parameter indices")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.validate_state_dict(state_dict)
        if self.optim.state:
            raise ValueError("restore zero1 into a freshly constructed optimizer, not an already-used optimizer")
        # ZRO replaces non-owned entries with None; preserve the caller's checkpoint/provenance dictionary.
        super().load_state_dict({**state_dict, "state": dict(state_dict["state"])})
        self.release_consolidated_state()

    def release_consolidated_state(self) -> None:
        """Release ZRO's retained CPU snapshot after publication, and prevent stale state_dict reads."""
        self._all_state_dicts: list[dict[str, Any]] = []


def _omit_outer_moments(optimizer: Optimizer, state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "state": {}}
