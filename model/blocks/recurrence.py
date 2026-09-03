# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
How a core block is iterated: the `(n no-grad, k backprop)` depth sampler, the random latent state, one recurrence
iteration (adapter over `[latent, input]`, then the block's layers) and the iteration loop with optional activation
checkpointing. `RecurrentGPT` (`model/model.py`) binds these to a model's `step`, mode, config and modules.
"""

import math
from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# (num_steps_no_grad, num_steps_with_grad) for one core block.
StepsPair = tuple[int, int]
# What callers may pass for one block: a pair, a 1- or 2-element tensor, or a scalar n (meaning (n, 0)).
StepsSpec = StepsPair | Tensor | int
# What `RecurrentGPT.forward` accepts: nothing (sample), one spec for all blocks, or one spec per block.
NumSteps = StepsSpec | list[StepsSpec] | None

# Activation checkpointing of one recurrence iteration. The RNG state is not saved/restored per checkpoint (the
# iteration draws no random numbers) and the recomputation is not checked for determinism.
_checkpoint = partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")


def canon_steps(steps: StepsSpec) -> StepsPair:
    """
    Turn one `StepsSpec` into an (n, k) pair of plain ints; a missing k means 0.
    """

    if isinstance(steps, torch.Tensor):
        values = steps.detach().reshape(-1)
        num_steps_no_grad = int(values[0].item())
        if values.numel() == 1:
            return num_steps_no_grad, 0
        return num_steps_no_grad, int(values[1].item())
    if isinstance(steps, (list, tuple)):
        num_steps_with_grad = 0
        if len(steps) > 1:
            num_steps_with_grad = int(steps[1])
        return int(steps[0]), num_steps_with_grad
    return int(steps), 0


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
    mean_recurrence: int, mean_backprop_depth: int, *, step: int, training: bool
) -> tuple[Tensor, Tensor]:
    """
    Sample (n no-grad steps, k backprop steps) with the poisson-lognormal-filling scheme, seeded by `step`; in eval
    mode return (`mean_recurrence`, 0).

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

    # A private generator seeded by the optimizer step, not the global RNG: a forward re-run under activation
    # checkpointing draws the same depth again. Distributed training must multiply the seed by (rank + 1) so ranks
    # draw different depths.
    generator = torch.Generator(device="cpu")
    generator.manual_seed((514229 + step) % (2**31 - 1))

    # "poisson-lognormal-filling": total depth = Poisson(rate) + 1 with a log-normal rate of mean `mean_recurrence`;
    # the last min(total, mean_backprop_depth) iterations get gradient (they "fill" the backprop budget).
    max_steps_with_grad = mean_backprop_depth
    mean_steps_no_grad = max(mean_recurrence - mean_backprop_depth, 0)
    sigma = 0.5
    mu = math.log(mean_steps_no_grad + max_steps_with_grad) - (sigma**2 / 2)
    rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=generator)
    total_steps = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=generator) + 1
    num_steps_no_grad = torch.clamp(total_steps - max_steps_with_grad, min=0)
    num_steps_with_grad = torch.minimum(torch.as_tensor(max_steps_with_grad), total_steps)
    return num_steps_no_grad.to(dtype=torch.long), num_steps_with_grad.to(dtype=torch.long)


def core_block_forward(
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: Tensor | None,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
) -> Tensor:
    """
    One recurrence iteration: inject the (normalised) block input into the latent state, then run the layers.
    """

    x_latent = adapter(torch.cat([x_latent, x_base], dim=-1))  # (B, S, 2 * E) -> (B, S, E)
    for layer in layers:
        x_latent = layer(x_latent, freqs_cis, mask)
    return x_latent


@torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
def iterate_core_block(
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: Tensor | None,
    num_steps_no_grad: int | Tensor,
    num_steps_with_grad: int | Tensor,
    *,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
    gradient_checkpointing: bool,
) -> Tensor:
    """
    Iterate `core_block_forward` first `num_steps_no_grad` times under `torch.no_grad`, then `num_steps_with_grad`
    times with gradient (each of those activation-checkpointed when `gradient_checkpointing`).
    """

    with torch.no_grad():
        for _ in range(num_steps_no_grad):
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers)

    for _ in range(num_steps_with_grad):
        if gradient_checkpointing:
            x_latent = _checkpoint(core_block_forward, x_latent, x_base, freqs_cis, mask, adapter, layers)
        else:
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers)
    return x_latent
