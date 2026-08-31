# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""How a core block is iterated: the `(n no-grad, k backprop)` depth sampler, the random latent state, one recurrence
iteration (adapter over `[latent, input]`, then the block's layers) and the iteration loop with optional activation
checkpointing. Binding these to a model (its `step`, train/eval mode, config and modules) is `RecurrentGPT`'s job
(`model/model.py`)."""

import math
from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

StepsPair = tuple[int, int]
StepsSpec = StepsPair | Tensor | int
NumSteps = StepsSpec | list[StepsSpec] | None

# Same kwargs the old `Config.checkpoint` property used for the non-SAC path.
_checkpoint = partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")


def canon_steps(steps: StepsSpec) -> StepsPair:
    """Accept a (n, k) pair, a 1- or 2-element tensor, or a scalar n (k = 0)."""
    if isinstance(steps, torch.Tensor):
        v = steps.detach().reshape(-1)
        if v.numel() == 1:
            return int(v[0].item()), 0
        return int(v[0].item()), int(v[1].item())
    if isinstance(steps, (list, tuple)):
        return int(steps[0]), int(steps[1] if len(steps) > 1 else 0)
    return int(steps), 0


def normalize_num_steps(num_steps_pair: NumSteps, num_blocks: int) -> list[StepsPair | None]:
    """One entry per core block: None (sample per block), one (n_no_grad, k_with_grad) pair broadcast to all blocks,
    or a list of pairs with one entry per block."""
    if num_steps_pair is None:
        return [None] * num_blocks
    if isinstance(num_steps_pair, list):
        if len(num_steps_pair) != num_blocks:
            raise ValueError(f"num_steps_pair has {len(num_steps_pair)} entries but there are {num_blocks} blocks")
        return [canon_steps(s) for s in num_steps_pair]
    return [canon_steps(num_steps_pair)] * num_blocks


def initialize_state(x: Tensor) -> Tensor:
    """`state_init=normal`: a standard-normal latent state of `x`'s shape, drawn from the global RNG."""
    return torch.randn_like(x)


@torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
def sample_recurrence_steps(
    mean_recurrence: int, mean_backprop_depth: int, *, step: int, training: bool
) -> tuple[Tensor, Tensor]:
    """Sample (n no-grad steps, k backprop steps) with the poisson-lognormal-filling scheme, seeded by `step`; in eval
    mode return (`mean_recurrence`, 0).

    Outputs are long tensors so that they can be passed through compiled functions."""
    # Meta-tensor tracing (flop counting) gets the expected values. NB: this draw also advances the global RNG (after
    # `initialize_state`); it is kept so the forward pass stays bit-identical to the original code.
    if torch.rand((1,)).is_meta:
        return mean_recurrence - mean_backprop_depth, mean_backprop_depth  # type: ignore[return-value]  # ints, see above

    # Seeded by the optimizer step so the sampler is re-runnable under activation checkpointing.
    # With distributed training the seed must be multiplied by (rank + 1) again so ranks draw different depths.
    seed_n = 514229 + step
    n_generator = torch.Generator(device="cpu")
    n_generator.manual_seed(seed_n % (2**31 - 1))

    t = max(mean_recurrence - mean_backprop_depth, 0)
    s = mean_backprop_depth

    if training:
        sigma = 0.5
        mu = math.log(t + s) - (sigma**2 / 2)
        rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
        p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
        n = torch.clamp(p - s, min=0)
        k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
    else:
        n, k = torch.as_tensor(mean_recurrence), torch.as_tensor(0)

    return n.to(dtype=torch.long), k.to(dtype=torch.long)


def core_block_forward(
    x_latent: Tensor,
    x_base: Tensor,
    freqs_cis: Tensor,
    mask: Tensor | None,
    adapter: torch.nn.Module,
    layers: torch.nn.ModuleList,
) -> Tensor:
    """One recurrence iteration: inject the (normalised) block input into the latent state, then run the layers."""
    x_latent = adapter(torch.cat([x_latent, x_base], dim=-1))
    for block in layers:
        x_latent = block(x_latent, freqs_cis, mask)
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
    """Iterate `core_block_forward` first `num_steps_no_grad` times under `torch.no_grad`, then `num_steps_with_grad`
    times with gradient (each of those activation-checkpointed when `gradient_checkpointing`)."""
    with torch.no_grad():
        for _ in range(num_steps_no_grad):
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers)

    for _ in range(num_steps_with_grad):
        if gradient_checkpointing:
            x_latent = _checkpoint(core_block_forward, x_latent, x_base, freqs_cis, mask, adapter, layers)
        else:
            x_latent = core_block_forward(x_latent, x_base, freqs_cis, mask, adapter, layers)
    return x_latent
