# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Parameter initialization: the `takase` ("spike no more", Takase et al.) scheme with truncated-orthogonal weights.

All weights are drawn with `trunc_orthogonal_`; the standard deviations come from the takase table below. Biases are
always zero. `Linear` is the thin `torch.nn.Linear` subclass whose `reset_parameters` applies such an init function.
"""

import math
from collections.abc import Callable
from math import sqrt

import torch

# `torch.nn.init.*` return the tensor, the fused inits return None: the result is never used.
InitFn = Callable[[torch.Tensor], object]


@torch.no_grad()
def trunc_orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    Orthogonal init from a truncated-normal random matrix (simplified, no guarantees).

    The tensor is treated as a (rows, cols) matrix (all trailing dims flattened into cols); the result has orthonormal
    rows if rows <= cols, orthonormal columns otherwise, times `gain`.
    """

    rows = tensor.size(0)
    cols = tensor.numel() // rows
    flattened = tensor.new_empty(rows, cols)
    torch.nn.init.trunc_normal_(flattened, mean=0.0, std=1.0)

    # QR needs a tall matrix to produce orthonormal columns; transpose a wide one and undo it afterwards.
    if rows < cols:
        flattened.t_()

    q_factor, r_factor = torch.linalg.qr(flattened)
    # Make Q uniformly distributed over orthogonal matrices (https://arxiv.org/pdf/math-ph/0609050.pdf).
    signs = torch.diag(r_factor, 0).sign()
    q_factor *= signs

    if rows < cols:
        q_factor.t_()

    tensor.view_as(q_factor).copy_(q_factor)
    tensor.mul_(gain)
    return tensor


def wrapped_trunc_ortho(tensor: torch.Tensor, std: float) -> None:
    """
    `trunc_orthogonal_` with the gain chosen so that the entries have standard deviation `std`.
    """

    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    trunc_orthogonal_(tensor, gain=std * math.sqrt(max(rows, cols)))


@torch.no_grad()
def init_qkv(qkv_tensor: torch.Tensor, qk_std: float, v_std: float, dim: int, head_dim: int) -> None:
    """
    Initialize the fused (q, k, v) projection weight, shape (dim + 2 * kv_dim, dim), one orthogonal block per
    component (q and k with `qk_std`, v with `v_std`). Without grouped-query attention kv_dim == dim.
    """

    total_rows = qkv_tensor.shape[0]
    n_kv_heads = (total_rows - dim) // (2 * head_dim)
    kv_dim = n_kv_heads * head_dim

    # Drawn in the order q, k, v (fixes the RNG consumption).
    q_weight = qkv_tensor.new_empty([dim, dim])
    k_weight = qkv_tensor.new_empty([kv_dim, dim])
    v_weight = qkv_tensor.new_empty([kv_dim, dim])
    wrapped_trunc_ortho(q_weight, qk_std)
    wrapped_trunc_ortho(k_weight, qk_std)
    wrapped_trunc_ortho(v_weight, v_std)
    qkv_tensor.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous())


@torch.no_grad()
def init_glu(glu_tensor: torch.Tensor, w1_std: float, w2_std: float) -> None:
    """
    Initialize the fused (gate, up) projection weight of the gated MLP, shape (2 * intermediate, dim), one
    orthogonal block per half (gate rows first with `w1_std`, then up rows with `w2_std`).
    """

    out_features, in_features = glu_tensor.shape
    rows_per_half = out_features // 2
    gate_weight = glu_tensor.new_empty([rows_per_half, in_features])
    up_weight = glu_tensor.new_empty([rows_per_half, in_features])
    wrapped_trunc_ortho(gate_weight, w1_std)
    wrapped_trunc_ortho(up_weight, w2_std)
    glu_tensor.data.copy_(torch.cat([gate_weight, up_weight], dim=0).contiguous())


class Init:
    """
    Dispatches the takase init by layer name.

    `num_layers` is the expected unrolled depth of the recurrent model (prelude + coda + sum of layers x recurrence);
    only the output projections (`out_attn`, `out_proj`) are scaled down by it.
    Layer names: "embedding", "head", "normalization", "qkv", "out_attn", "glu", "out_proj", "in_proj".
    """

    def __init__(self, dim: int, head_dim: int, num_layers: int) -> None:
        self.dim = dim
        self.head_dim = head_dim
        self.num_layers = num_layers
        std = sqrt(2 / (5 * dim))
        self.table = {
            "std": std,  # every weight not listed below
            "out_proj": std / sqrt(2 * num_layers),  # residual-branch outputs, also used for "out_attn"
            "embedding": std,
            "embed_scale": sqrt(dim),
        }

    def _std(self, name_of_layer: str) -> float:
        if name_of_layer in self.table:
            return self.table[name_of_layer]
        if name_of_layer in ("out_attn", "out_proj"):
            return self.table["out_proj"]
        return self.table["std"]

    def fn(self, name_of_layer: str) -> InitFn:
        """
        Return the init function for a weight tensor, to be stored for `reset_parameters()`.
        """

        if name_of_layer == "normalization":
            return torch.nn.init.ones_
        if name_of_layer == "qkv":
            std = self._std("std")
            return lambda tensor: init_qkv(tensor, std, std, self.dim, self.head_dim)
        if name_of_layer == "glu":
            # Upstream passes the w1 std for both halves; kept for numerical identity.
            std = self._std("std")
            return lambda tensor: init_glu(tensor, std, std)
        std = self._std(name_of_layer)
        return lambda tensor: wrapped_trunc_ortho(tensor, std=std)

    def apply(self, module: torch.nn.Module, name_of_layer: str) -> None:
        """
        Directly apply the init to an already constructed module (weight by name, bias to zero).
        """

        weight = getattr(module, "weight", None)
        if weight is not None:
            self.fn(name_of_layer)(weight)
        bias = getattr(module, "bias", None)
        if bias is not None:
            torch.nn.init.zeros_(bias)

    @property
    def logit_scale(self) -> float:
        return 1.0

    @property
    def embedding_scale(self) -> float:
        return float(self.table["embed_scale"])

    def __repr__(self) -> str:
        return f"takase Initializer {self.dim}x{self.head_dim}-{self.num_layers}"


class Linear(torch.nn.Linear):
    """
    `torch.nn.Linear` whose weight init is given explicitly; the bias is always zeroed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, init_method: InitFn) -> None:
        self.init_method = init_method  # set before super().__init__, which calls reset_parameters()
        super().__init__(in_features, out_features, bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()
