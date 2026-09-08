# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
How a core block is iterated: the `(n no-grad, k backprop)` depth sampler, the random latent state, one recurrence
iteration (adapter over `[latent, input]`, then the block's layers) and the iteration loop with optional activation
checkpointing. `RecurrentGPT` (`model/model.py`) binds these to a model's `step`, mode, config and modules.

The adapter is one `Linear` (2E -> E) over `[latent, input]`; its input half does not change over the iterations of a
block, so `adapter_base_projection` computes it once and the iteration only runs the latent half.

Activation checkpointing (`CheckpointMode`): every backprop iteration of a block is one checkpointed region. `full`
recomputes the whole iteration in the backward pass (v2_small, 8192-token packs: about 22 percent slower than `none`,
35 percent of the peak memory); `selective` keeps the outputs of the GEMMs and of FlexAttention and recomputes only
the cheap ops, RMSNorms, RoPE, the SiLU gate and the residual adds (about 6 percent slower, 75 percent of the memory).
The checkpoint call lives in `checkpointed_iteration`, a frame of its own that dynamo compiles when the model is
compiled: torch.compile then traces the checkpoint and its partitioner fuses the recompute into the backward graph.
Calling `torch.utils.checkpoint` from the eager loop *around* the compiled iteration instead (the code before
2026-09-07) launched the compiled forward a second time per iteration under eager saved-tensor hooks and ran
FlexAttention's forward again: 2.1 times the step time of `none` for the same memory as `full`.
"""

import math
from functools import partial
from typing import Any, Callable, Literal, cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from ..layers.attention import AttentionMask

# (num_steps_no_grad, num_steps_with_grad) for one core block.
StepsPair = tuple[int, int]
# What callers may pass for one block: a pair, a 1- or 2-element tensor, or a scalar n (meaning (n, 0)).
StepsSpec = StepsPair | Tensor | int
# What `RecurrentGPT.forward` accepts: nothing (sample), one spec for all blocks, or one spec per block.
NumSteps = StepsSpec | list[StepsSpec] | None

# Activation checkpointing of the backprop iterations of a block, see the module docstring.
CheckpointMode = Literal["none", "selective", "full"]
CHECKPOINT_MODES: tuple[str, ...] = ("none", "selective", "full")

# Ops whose outputs `selective` keeps: the GEMMs (adapter, attention projections, MLP) and FlexAttention; everything
# else in an iteration is recomputed. The list form of the policy saves exactly these and recomputes the rest.
_SAVED_OPS = [torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.higher_order.flex_attention]
_selective_contexts = partial(create_selective_checkpoint_contexts, _SAVED_OPS)
# The AOT autograd cache cannot hash an arbitrary context_fn; without this attribute it is bypassed (a recompile per
# process start).
_selective_contexts.cache_hash = "recurrence-save-mm-addmm-flex"  # type: ignore[attr-defined]

# Non-reentrant checkpoints. The RNG state is not saved/restored per checkpoint (the iteration draws no random
# numbers; under torch.compile the flag has no effect anyway) and the recomputation is not checked for determinism.
_full_checkpoint = partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")
_selective_checkpoint = partial(_full_checkpoint, context_fn=_selective_contexts)
CheckpointFn = Callable[..., Any]


def check_checkpoint_mode(mode: str) -> CheckpointMode:
    """
    `mode` if it is one of `CHECKPOINT_MODES`, else a ValueError naming them.
    """

    if mode not in CHECKPOINT_MODES:
        raise ValueError(f"gradient_checkpointing must be one of {', '.join(CHECKPOINT_MODES)}, not {mode!r}")
    return cast(CheckpointMode, mode)


def canon_steps(steps: StepsSpec) -> StepsPair:
    """
    Turn one `StepsSpec` into an (n, k) pair of plain ints; a missing k means 0.
    """

    if isinstance(steps, torch.Tensor):
        values = steps.detach().reshape(-1)
        pair = (int(values[0].item()), int(values[1].item()) if values.numel() > 1 else 0)
    elif isinstance(steps, (list, tuple)):
        pair = (int(steps[0]), int(steps[1]) if len(steps) > 1 else 0)
    else:
        pair = (int(steps), 0)
    if min(pair) < 0 or sum(pair) < 1:
        raise ValueError(f"num_steps {steps!r}: a block runs at least one recurrent step (n + k >= 1, both >= 0), else its output is the random initial state")
    return pair


def normalize_num_steps(num_steps: NumSteps, num_blocks: int) -> list[StepsPair | None]:
    """
    One entry per core block: None (sample per block), one (n_no_grad, k_with_grad) pair broadcast to all blocks,
    or a list of pairs with one entry per block.
    """

    if num_steps is None:
        return [None] * num_blocks
    if isinstance(num_steps, list):
        if len(num_steps) != num_blocks:
            raise ValueError(f"num_steps has {len(num_steps)} entries but there are {num_blocks} blocks")
        per_block: list[StepsPair | None] = []
        for steps in num_steps:
            per_block.append(canon_steps(steps))
        return per_block
    return [canon_steps(num_steps)] * num_blocks


def initialize_state(x: Tensor) -> Tensor:
    """
    `state_init=normal`: a standard-normal latent state of `x`'s shape, drawn from the global RNG.
    """

    return torch.randn_like(x)


@torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
def sample_recurrence_steps(
    mean_recurrence: int, mean_backprop_depth: int, *, step: int, block_idx: int, training: bool
) -> tuple[Tensor, Tensor]:
    """
    Sample (n no-grad steps, k backprop steps) with the poisson-lognormal-filling scheme, seeded by `step` and
    `block_idx` (blocks draw independently); in eval mode return (`mean_recurrence`, 0).

    Outputs are long tensors so that they can be passed through compiled functions.
    """

    # One draw, two jobs: it detects meta-tensor tracing (flop counting) and advances the global RNG by one number.
    # The reference forward pass depends on that RNG consumption (right after `initialize_state`); never skip it.
    if torch.rand((1,)).is_meta:
        return mean_recurrence - mean_backprop_depth, mean_backprop_depth  # type: ignore[return-value]  # plain ints are fine for the tracer

    if not training:
        num_steps_no_grad = torch.as_tensor(mean_recurrence)
        num_steps_with_grad = torch.as_tensor(0)
        return num_steps_no_grad.to(dtype=torch.long), num_steps_with_grad.to(dtype=torch.long)

    # A private generator seeded by the optimizer step and the block, not the global RNG: a forward re-run under
    # activation checkpointing draws the same depth again. The block stride is far above any step count, so blocks
    # never share a seed. Distributed training must multiply the seed by (rank + 1) so ranks draw different depths.
    generator = torch.Generator(device="cpu")
    generator.manual_seed((514229 + step + 2**24 * block_idx) % (2**31 - 1))

    # "poisson-lognormal-filling": total depth = Poisson(rate) + 1, the +1 being the guaranteed pass, with a
    # log-normal rate of mean `mean_recurrence - 1` so the total has mean `mean_recurrence`; the last
    # min(total, mean_backprop_depth) iterations get gradient (they "fill" the backprop budget). A mean of 1 leaves
    # no rate to draw (and no `log(0)`): the total is always 1.
    max_steps_with_grad = mean_backprop_depth
    sigma = 0.5
    if mean_recurrence > 1:
        mu = math.log(mean_recurrence - 1) - (sigma**2 / 2)
        rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=generator)
    else:
        rate = torch.zeros((1,))
    total_steps = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=generator) + 1
    num_steps_no_grad = torch.clamp(total_steps - max_steps_with_grad, min=0)
    num_steps_with_grad = torch.minimum(torch.as_tensor(max_steps_with_grad), total_steps)
    return num_steps_no_grad.to(dtype=torch.long), num_steps_with_grad.to(dtype=torch.long)


def adapter_base_projection(x_base: Tensor, adapter: torch.nn.Module) -> Tensor:
    """
    The block-input half of the adapter, `x_base @ W[:, E:].T` (plus the bias, if any), for `adapter` a `Linear` with
    the single (E, 2E) weight over `[latent, input]`. It is constant over the iterations of a block, so the callers
    compute it once per block and pass it to `core_block_forward` as `base_proj`. The column slice is a strided view
    of the weight: no copy, and the parameter (and its state-dict key) stays as it is.
    """

    weight = cast(Tensor, adapter.weight)
    n_embd = weight.shape[0]
    return torch.nn.functional.linear(x_base, weight[:, n_embd:], cast(Tensor | None, adapter.bias))


def core_block_forward(
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: AttentionMask,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
    base_proj: Tensor | None = None,
) -> Tensor:
    """
    One recurrence iteration: inject the (normalised) block input into the latent state, then run the layers.

    The adapter `W @ [latent, input]` runs as its two halves: `base_proj` is the constant input half
    (`adapter_base_projection`, hoisted out of the loop by `iterate_core_block`; computed here when not given) and only
    the latent half `W[:, :E] @ latent` is a GEMM per iteration. Under bf16 autocast both halves are bf16 GEMM outputs
    (fp32 accumulation) added in bf16: rounding-level different from the single K = 2E GEMM over the concatenation.
    """

    if base_proj is None:
        base_proj = adapter_base_projection(x_base, adapter)
    weight = cast(Tensor, adapter.weight)
    n_embd = weight.shape[0]
    x_latent = torch.nn.functional.linear(x_latent, weight[:, :n_embd]) + base_proj  # (B, S, E)
    for layer in layers:
        x_latent = layer(x_latent, freqs_cis, mask)
    return x_latent


def checkpointed_iteration(
    checkpoint_fn: CheckpointFn,
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: AttentionMask,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
    base_proj: Tensor,
) -> Tensor:
    """
    One checkpointed recurrence iteration (`checkpoint_fn` is `_full_checkpoint` or `_selective_checkpoint`) as a
    frame of its own: called from the dynamo-disabled loop, this frame compiles, so the checkpoint is traced as part of
    the compiled graph and the recompute is fused into the compiled backward (module docstring). `mask` may be a
    FlexAttention `BlockMask` (packed sequences); the non-reentrant checkpoint passes it through as a plain positional
    argument.
    """

    return cast(Tensor, checkpoint_fn(core_block_forward, x_latent, x_base, freqs_cis, mask, adapter, layers, base_proj))


@torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
def iterate_core_block(
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: AttentionMask,
    num_steps_no_grad: int | Tensor,
    num_steps_with_grad: int | Tensor,
    *,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
    gradient_checkpointing: CheckpointMode,
    base_proj: Tensor | None = None,
) -> Tensor:
    """
    Iterate `core_block_forward` first `num_steps_no_grad` times under `torch.no_grad`, then `num_steps_with_grad`
    times with gradient, each of those activation-checkpointed in the `gradient_checkpointing` mode (`none`,
    `selective`, `full`; module docstring).

    `base_proj` is the adapter's input half (`adapter_base_projection(x_base, adapter)`), shared by all iterations;
    computed here once when not given. It is an input of every checkpointed iteration, so the recomputation reuses it.
    """

    mode = check_checkpoint_mode(gradient_checkpointing)
    if base_proj is None:
        base_proj = adapter_base_projection(x_base, adapter)

    with torch.no_grad():
        for _ in range(num_steps_no_grad):
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers, base_proj)

    # resolved here, not at import: tests swap the module-level wrappers to count the calls
    checkpoint_fn = _selective_checkpoint if mode == "selective" else _full_checkpoint
    for _ in range(num_steps_with_grad):
        if mode == "none":
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers, base_proj)
        else:
            x_latent = checkpointed_iteration(checkpoint_fn, x_latent, x_base, freqs_cis, mask, adapter, layers, base_proj)
    return x_latent
