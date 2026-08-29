# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Multi-block recurrent transformer: prelude -> N recurrent core blocks (each with its own adapter, input norm and
recurrence depth, residual around each block) -> coda -> final norm -> tied LM head."""

import math
from functools import partial
from typing import cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import precompute_freqs_cis
from .blocks import SandwichBlock
from .config import RecurrentConfig
from .init import Linear

StepsPair = tuple[int, int]
StepsSpec = StepsPair | Tensor | int
NumSteps = StepsSpec | list[StepsSpec] | None

# Same kwargs the old `Config.checkpoint` property used for the non-SAC path.
_checkpoint = partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")


class TransformerModules(torch.nn.ModuleDict):
    """`ModuleDict` of the model parts; the annotations only give `model.transformer.<name>` a precise static type."""

    wte: torch.nn.Embedding
    prelude: torch.nn.ModuleList
    adapters: torch.nn.ModuleList
    core_blocks: torch.nn.ModuleList
    coda: torch.nn.ModuleList
    ln_fs: torch.nn.ModuleList
    ln_final: torch.nn.LayerNorm


class RecurrentGPT(torch.nn.Module):
    freqs_cis: Tensor  # registered buffer (declared here for the type checkers only)

    def __init__(
        self, config: RecurrentConfig, *, ignore_index: int = -100, gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.ignore_index = ignore_index
        self.gradient_checkpointing = gradient_checkpointing

        # Normalized to lists / filled in by RecurrentConfig.__post_init__ (asserts only narrow the static types).
        n_layers_per_block = config.n_layers_in_recurrent_block
        assert isinstance(n_layers_per_block, list)
        padded_vocab_size = config.padded_vocab_size
        assert padded_vocab_size is not None

        # Construction order matters: it fixes the RNG consumption of the parameter init.
        prelude = torch.nn.ModuleList(SandwichBlock(config) for _ in range(config.n_layers_in_prelude))
        core_blocks = torch.nn.ModuleList(
            torch.nn.ModuleList(SandwichBlock(config) for _ in range(n_layers)) for n_layers in n_layers_per_block
        )
        adapters = torch.nn.ModuleList(
            Linear(config.n_embd * 2, config.n_embd, bias=False, init_method=config.init.fn("in_proj"))
            for _ in n_layers_per_block
        )
        coda = torch.nn.ModuleList(SandwichBlock(config) for _ in range(config.n_layers_in_coda))
        ln_fs = torch.nn.ModuleList(torch.nn.LayerNorm(config.n_embd, eps=config.norm_eps) for _ in n_layers_per_block)
        ln_final = torch.nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.transformer = TransformerModules(
            dict(
                wte=torch.nn.Embedding(padded_vocab_size, config.n_embd),
                prelude=prelude,
                adapters=adapters,
                core_blocks=core_blocks,
                coda=coda,
                ln_fs=ln_fs,
                ln_final=ln_final,
            )
        )
        self.emb_scale = config.init.embedding_scale
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False, init_method=config.init.fn("head"))
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Set externally each optimizer step; seeds the recurrence sampler.
        self.step: int = 0
        self.reset_parameters()

    def _precompute_freqs_cis(self) -> Tensor:
        return precompute_freqs_cis(
            self.config.n_embd // self.config.num_attention_heads,
            self.config.block_size,
            self.config.rope_settings.rope_base,
        )

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        for ln_f in self.transformer.ln_fs:
            self.config.init.apply(ln_f, "normalization")
        self.config.init.apply(self.transformer.ln_final, "normalization")

    @staticmethod
    def _canon_steps(steps: StepsSpec) -> StepsPair:
        """Accept a (n, k) pair, a 1- or 2-element tensor, or a scalar n (k = 0)."""
        if isinstance(steps, torch.Tensor):
            v = steps.detach().reshape(-1)
            if v.numel() == 1:
                return int(v[0].item()), 0
            return int(v[0].item()), int(v[1].item())
        if isinstance(steps, (list, tuple)):
            return int(steps[0]), int(steps[1] if len(steps) > 1 else 0)
        return int(steps), 0

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        return_logits: bool = False,
        num_steps_pair: NumSteps = None,
    ) -> dict[str, Tensor | None]:
        """`num_steps_pair`: None (sample per block), one (n_no_grad, k_with_grad) pair for all blocks, or a list of
        pairs with one entry per core block."""
        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale

        latent_tensor_merker = input_embeds
        for block in self.transformer.prelude:
            latent_tensor_merker = block(latent_tensor_merker, freqs_cis, attention_mask)

        num_blocks = len(self.transformer.core_blocks)
        if num_steps_pair is None:
            normalized_steps: list[StepsPair | None] = [None] * num_blocks
        elif isinstance(num_steps_pair, list):
            if len(num_steps_pair) != num_blocks:
                raise ValueError(f"num_steps_pair has {len(num_steps_pair)} entries but there are {num_blocks} blocks")
            normalized_steps = [self._canon_steps(s) for s in num_steps_pair]
        else:
            normalized_steps = [self._canon_steps(num_steps_pair)] * num_blocks

        x = latent_tensor_merker
        for block_idx, core_block in enumerate(self.transformer.core_blocks):
            x = self.iterate_forward(
                x,
                freqs_cis,
                attention_mask,
                normalized_steps[block_idx],
                core_block=cast(torch.nn.ModuleList, core_block),  # ModuleList is not generic in the torch stubs
                core_block_number=block_idx,
            )
            x = x + latent_tensor_merker
            latent_tensor_merker = x

        for block in self.transformer.coda:
            x = block(x, freqs_cis, attention_mask)
        x = self.transformer.ln_final(x)

        logits = self.lm_head(x).float() * self.config.init.logit_scale
        if labels is not None:
            n_classes = logits.shape[-1]
            labels = labels.to(torch.long)
            invalid = (labels < 0) | (labels >= n_classes)
            labels = labels.masked_fill(invalid, self.ignore_index)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, n_classes), labels.view(-1), ignore_index=self.ignore_index
            )
            log_ppl = loss.clone().detach()
        else:
            loss, log_ppl = torch.as_tensor(0.0), torch.as_tensor(0.0)

        return {"loss": loss, "logits": logits if return_logits else None, "log_ppl": log_ppl}

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def iterate_forward(
        self,
        input_tensor: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None,
        num_steps_pair: StepsPair | None,
        *,
        core_block: torch.nn.ModuleList,
        core_block_number: int,
    ) -> Tensor:
        x_base = self.transformer.ln_fs[core_block_number](input_tensor)
        x_latent = self.initialize_state(input_tensor)

        num_steps_no_grad: int | Tensor
        num_steps_with_grad: int | Tensor
        if num_steps_pair is None:
            num_steps_no_grad, num_steps_with_grad = self.randomized_iteration_sampler(core_block_number)
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair

        with torch.no_grad():
            for _ in range(num_steps_no_grad):
                x_latent = self.core_block_forward(x_latent, x_base, freqs_cis, mask, core_block, core_block_number)

        for _ in range(num_steps_with_grad):
            if self.gradient_checkpointing:
                x_latent = _checkpoint(
                    self.core_block_forward, x_latent, x_base, freqs_cis, mask, core_block, core_block_number
                )
            else:
                x_latent = self.core_block_forward(x_latent, x_base, freqs_cis, mask, core_block, core_block_number)
        return x_latent

    def core_block_forward(
        self,
        x_latent: Tensor,
        x_base: Tensor,
        freqs_cis: Tensor,
        mask: Tensor | None,
        core_block: torch.nn.ModuleList,
        core_block_number: int,
    ) -> Tensor:
        x_latent = self.transformer.adapters[core_block_number](torch.cat([x_latent, x_base], dim=-1))
        for block in core_block:
            x_latent = block(x_latent, freqs_cis, mask)
        return x_latent

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def randomized_iteration_sampler(self, core_block_number: int = 0) -> tuple[Tensor, Tensor]:
        """Sample (n no-grad steps, k backprop steps) with the poisson-lognormal-filling scheme.

        Outputs are long tensors so that they can be passed through compiled functions."""
        assert isinstance(self.config.mean_recurrence, list)  # normalized by RecurrentConfig.__post_init__
        assert isinstance(self.config.mean_backprop_depth, list)
        mean_recurrence = self.config.mean_recurrence[core_block_number]
        mean_backprop_depth = self.config.mean_backprop_depth[core_block_number]

        # Meta-tensor tracing (flop counting) gets the expected values. NB: this draw also advances the global RNG
        # before `initialize_state`; it is kept so the forward pass stays bit-identical to the original code.
        if torch.rand((1,)).is_meta:
            return mean_recurrence - mean_backprop_depth, mean_backprop_depth  # type: ignore[return-value]  # ints, see above

        # Seeded by the optimizer step so the sampler is re-runnable under activation checkpointing.
        # With distributed training the seed must be multiplied by (rank + 1) again so ranks draw different depths.
        seed_n = 514229 + self.step
        n_generator = torch.Generator(device="cpu")
        n_generator.manual_seed(seed_n % (2**31 - 1))

        t = max(mean_recurrence - mean_backprop_depth, 0)
        s = mean_backprop_depth

        if self.training:
            sigma = 0.5
            mu = math.log(t + s) - (sigma**2 / 2)
            rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=n_generator)
            p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=n_generator) + 1
            n = torch.clamp(p - s, min=0)
            k = torch.as_tensor(torch.minimum(torch.as_tensor(s), p))
        else:
            n, k = torch.as_tensor(mean_recurrence), torch.as_tensor(0)

        return n.to(dtype=torch.long), k.to(dtype=torch.long)

    def initialize_state(self, latent_tensor_merker: Tensor) -> Tensor:
        """`state_init=normal`: a standard-normal draw from the global RNG."""
        return torch.randn_like(latent_tensor_merker)
