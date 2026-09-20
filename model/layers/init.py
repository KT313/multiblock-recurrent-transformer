# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""
Parameter initialization: Takase scales with orthogonal weights or independent truncated-normal entries.

Matrix weights default to scaled orthogonal initialization. With `init_orthogonal=False`, the same scale table
specifies the pre-truncation normal std, with bounds +/-3 std. Norm scales are one and biases zero.
`Linear` is the thin `torch.nn.Linear` subclass whose `reset_parameters` applies such an init function.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from math import sqrt

import torch

# `torch.nn.init.*` return the tensor, the fused inits return None: the result is never used.
InitFn = Callable[[torch.Tensor], object]
MatrixInitFn = Callable[[torch.Tensor, float], None]

_CHECKPOINT_INITIALIZATION: ContextVar[bool] = ContextVar("checkpoint_initialization", default=False)


@contextmanager
def checkpoint_initialization() -> Iterator[None]:
    """Use cheap placeholders for weights about to be restored, without changing later reset_parameters calls.

    Parameters remain allocated normally, preserving aliases and nonpersistent buffers. Embedding's own normal
    initialization and small normalization/bias fills remain; both custom matrix initialization paths are skipped.
    The caller must load a complete checkpoint before using the model.
    """
    token = _CHECKPOINT_INITIALIZATION.set(True)
    try:
        yield
    finally:
        _CHECKPOINT_INITIALIZATION.reset(token)


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


def wrapped_trunc_normal(tensor: torch.Tensor, std: float) -> None:
    """Independent normal entries, truncated at +/-3 std; std is the scale BEFORE truncation.

    Restores the non-orthogonal path from the repository's former recpre/init.py. Do not use the torch default
    bounds (-2, 2): those are absolute values, not multiples of this layer's standard deviation.
    """
    torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std)


@torch.no_grad()
def init_qkv(
    qkv_tensor: torch.Tensor, qk_std: float, v_std: float, dim: int, head_dim: int,
    *, init_fn: MatrixInitFn = wrapped_trunc_ortho,
) -> None:
    """
    Initialize the fused (q, k, v) projection weight, shape (dim + 2 * kv_dim, dim), one orthogonal block per
    component by default (q and k with `qk_std`, v with `v_std`); init_fn can select truncated-normal entries.
    Without grouped-query attention kv_dim == dim.
    """

    total_rows = qkv_tensor.shape[0]
    n_kv_heads = (total_rows - dim) // (2 * head_dim)
    kv_dim = n_kv_heads * head_dim

    # Drawn in the order q, k, v (fixes the RNG consumption).
    q_weight = qkv_tensor.new_empty([dim, dim])
    k_weight = qkv_tensor.new_empty([kv_dim, dim])
    v_weight = qkv_tensor.new_empty([kv_dim, dim])
    init_fn(q_weight, qk_std)
    init_fn(k_weight, qk_std)
    init_fn(v_weight, v_std)
    qkv_tensor.data.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0).contiguous())


@torch.no_grad()
def init_glu(
    glu_tensor: torch.Tensor, w1_std: float, w2_std: float, *, init_fn: MatrixInitFn = wrapped_trunc_ortho,
) -> None:
    """
    Initialize the fused (gate, up) projection weight of the gated MLP, shape (2 * intermediate, dim), one
    orthogonal block per half by default (gate rows first with `w1_std`, then up rows with `w2_std`).
    init_fn can select truncated-normal entries with the same component scales and draw order.
    """

    out_features, in_features = glu_tensor.shape
    rows_per_half = out_features // 2
    gate_weight = glu_tensor.new_empty([rows_per_half, in_features])
    up_weight = glu_tensor.new_empty([rows_per_half, in_features])
    init_fn(gate_weight, w1_std)
    init_fn(up_weight, w2_std)
    glu_tensor.data.copy_(torch.cat([gate_weight, up_weight], dim=0).contiguous())


class Init:
    """
    Dispatches the takase init by layer name.

    `num_layers` is the expected unrolled depth of the recurrent model (prelude + coda + sum of layers x recurrence);
    only the output projections (`out_attn`, `out_proj`) are scaled down by it.
    Layer names: "embedding", "head", "normalization", "qkv", "out_attn", "glu", "out_proj", "in_proj".
    """

    def __init__(self, dim: int, head_dim: int, num_layers: int, *, orthogonal: bool = True) -> None:
        self.dim = dim
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.orthogonal = orthogonal
        self.normal_: MatrixInitFn = wrapped_trunc_ortho if orthogonal else wrapped_trunc_normal
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
            return lambda tensor: init_qkv(tensor, std, std, self.dim, self.head_dim, init_fn=self.normal_)
        if name_of_layer == "glu":
            # Upstream passes the w1 std for both halves; kept for numerical identity.
            std = self._std("std")
            return lambda tensor: init_glu(tensor, std, std, init_fn=self.normal_)
        std = self._std(name_of_layer)
        return lambda tensor: self.normal_(tensor, std)

    def apply(self, module: torch.nn.Module, name_of_layer: str) -> None:
        """
        Directly apply the init to an already constructed module (weight by name, bias to zero).
        """

        weight = getattr(module, "weight", None)
        if weight is not None:
            if _CHECKPOINT_INITIALIZATION.get() and name_of_layer != "normalization":
                torch.nn.init.zeros_(weight)
            else:
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
        suffix = "" if self.orthogonal else " (truncated normal, +/-3 std)"
        return f"takase Initializer {self.dim}x{self.head_dim}-{self.num_layers}{suffix}"


class Linear(torch.nn.Linear):
    """
    `torch.nn.Linear` whose weight init is given explicitly; the bias is always zeroed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool, init_method: InitFn) -> None:
        self.init_method = init_method  # set before super().__init__, which calls reset_parameters()
        super().__init__(in_features, out_features, bias)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if _CHECKPOINT_INITIALIZATION.get():
            self.weight.zero_()
        else:
            self.init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()
