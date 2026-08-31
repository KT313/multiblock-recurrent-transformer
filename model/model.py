# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Multi-block recurrent transformer: prelude -> N recurrent core blocks (each with its own adapter, input norm and
recurrence depth, residual around each block) -> coda -> final norm -> tied LM head.

The parts live in the topic packages: `layers/` (norms, attention, MLP, init), `blocks/sandwich.py` (the transformer
block) and `blocks/recurrence.py` (depth sampler, latent state, one recurrence iteration, the iteration loop). This
module assembles them into `RecurrentGPT` and binds the recurrence to the model's `step`, mode, config and modules."""

from typing import cast

import torch
from torch import Tensor

from .blocks.recurrence import (
    NumSteps,
    StepsPair,
    initialize_state,
    iterate_core_block,
    normalize_num_steps,
    sample_recurrence_steps,
)
from .blocks.sandwich import SandwichBlock
from .config import RecurrentConfig
from .layers.attention import precompute_freqs_cis
from .layers.init import Linear


class TransformerModules(torch.nn.ModuleDict):
    """`ModuleDict` of the model parts; the annotations only give `model.transformer.<name>` a precise static type."""

    wte: torch.nn.Embedding  # token embedding
    prelude: torch.nn.ModuleList  # SandwichBlocks run once before the recurrence
    adapters: torch.nn.ModuleList  # one Linear per core block: [latent, block input] -> latent
    core_blocks: torch.nn.ModuleList  # one ModuleList of SandwichBlocks per core block
    coda: torch.nn.ModuleList  # SandwichBlocks run once after the recurrence
    ln_fs: torch.nn.ModuleList  # one LayerNorm per core block, applied to the block input
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
        prelude = torch.nn.ModuleList()
        for _ in range(config.n_layers_in_prelude):
            prelude.append(SandwichBlock(config))

        core_blocks = torch.nn.ModuleList()
        for n_layers in n_layers_per_block:
            layers = torch.nn.ModuleList()
            for _ in range(n_layers):
                layers.append(SandwichBlock(config))
            core_blocks.append(layers)

        adapters = torch.nn.ModuleList()
        for _ in n_layers_per_block:
            adapters.append(Linear(config.n_embd * 2, config.n_embd, bias=False, init_method=config.init.fn("in_proj")))

        coda = torch.nn.ModuleList()
        for _ in range(config.n_layers_in_coda):
            coda.append(SandwichBlock(config))

        ln_fs = torch.nn.ModuleList()
        for _ in n_layers_per_block:
            ln_fs.append(torch.nn.LayerNorm(config.n_embd, eps=config.norm_eps))
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
        """The RoPE table for every position up to `block_size`."""
        return precompute_freqs_cis(self.config.head_size, self.config.block_size, self.config.rope_settings.rope_base)

    def reset_parameters(self) -> None:
        """Re-initialize the modules that are not `Linear` (those init themselves): embedding and LayerNorms."""
        self.config.init.apply(self.transformer.wte, "embedding")
        for ln_f in self.transformer.ln_fs:
            self.config.init.apply(ln_f, "normalization")
        self.config.init.apply(self.transformer.ln_final, "normalization")

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
        # RoPE rows for the positions of this batch: the first S rows, or the rows selected by `position_ids`.
        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        x = self.transformer.wte(input_ids)  # (B, S, E)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        for block in self.transformer.prelude:
            x = block(x, freqs_cis, attention_mask)

        # Each core block is iterated on its input and added back onto it (residual around the whole block).
        num_steps = normalize_num_steps(num_steps_pair, len(self.transformer.core_blocks))
        for block_idx, block_steps in enumerate(num_steps):
            block_out = self.iterate_forward(x, freqs_cis, attention_mask, block_steps, block_idx)
            x = block_out + x

        for block in self.transformer.coda:
            x = block(x, freqs_cis, attention_mask)
        x = self.transformer.ln_final(x)

        logits = self.lm_head(x).float() * self.config.init.logit_scale  # (B, S, padded_vocab), float32
        if labels is not None:
            loss = self.loss(logits, labels)
        else:
            loss = torch.as_tensor(0.0)
        returned_logits: Tensor | None = None
        if return_logits:
            returned_logits = logits
        return {"loss": loss, "logits": returned_logits, "log_ppl": loss.clone().detach()}

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy over the vocabulary; labels outside `[0, vocab)` count as `ignore_index`."""
        n_classes = logits.shape[-1]
        labels = labels.to(torch.long)
        invalid = (labels < 0) | (labels >= n_classes)
        labels = labels.masked_fill(invalid, self.ignore_index)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, n_classes), labels.view(-1), ignore_index=self.ignore_index
        )

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def iterate_forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None, num_steps: StepsPair | None, block_idx: int
    ) -> Tensor:
        """Core block `block_idx` on `x`: normalise the input (`ln_fs`), draw the random latent state, then iterate
        the block n times without and k times with gradient — `num_steps`, or the sampler's draw when None."""
        transformer = self.transformer
        x_base = transformer.ln_fs[block_idx](x)
        x_latent = initialize_state(x)  # consumes the global RNG first, then (if sampling) the sampler's draw

        steps: tuple[int, int] | tuple[Tensor, Tensor]
        if num_steps is None:
            steps = self.randomized_iteration_sampler(block_idx)
        else:
            steps = num_steps
        num_steps_no_grad, num_steps_with_grad = steps

        # ModuleList is not generic in the torch stubs, so indexing `core_blocks` needs the cast.
        layers = cast(torch.nn.ModuleList, transformer.core_blocks[block_idx])
        # `iterate_core_block` is dynamo-disabled, which makes it untyped for mypy; hence the explicit annotation.
        x_out: Tensor = iterate_core_block(
            x_latent,
            x_base,
            freqs_cis,
            mask,
            num_steps_no_grad,
            num_steps_with_grad,
            adapter=transformer.adapters[block_idx],
            layers=layers,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        return x_out

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def randomized_iteration_sampler(self, block_idx: int = 0) -> tuple[Tensor, Tensor]:
        """(n no-grad, k backprop) iterations for core block `block_idx`: the poisson-lognormal-filling draw seeded by
        `self.step` in training, (`mean_recurrence`, 0) in eval mode."""
        assert isinstance(self.config.mean_recurrence, list)  # normalized by RecurrentConfig.__post_init__
        assert isinstance(self.config.mean_backprop_depth, list)
        # `sample_recurrence_steps` is dynamo-disabled, which makes it untyped for mypy; hence the explicit annotation.
        steps: tuple[Tensor, Tensor] = sample_recurrence_steps(
            self.config.mean_recurrence[block_idx],
            self.config.mean_backprop_depth[block_idx],
            step=self.step,
            training=self.training,
        )
        return steps
