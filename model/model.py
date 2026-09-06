# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Multi-block recurrent transformer: prelude -> N recurrent core blocks (each with its own adapter, input norm and
recurrence depth, residual around each block) -> coda -> final norm -> tied LM head.

The parts live in the topic packages: `layers/` (norms, attention, MLP, init), `blocks/sandwich.py` (the transformer
block) and `blocks/recurrence.py` (depth sampler, latent state, one recurrence iteration, the iteration loop). This
module assembles them into `RecurrentGPT` and binds the recurrence to the model's `step`, mode, config and modules.
"""

from functools import partial
from typing import cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .blocks.recurrence import (
    NumSteps,
    StepsPair,
    adapter_base_projection,
    initialize_state,
    iterate_core_block,
    normalize_num_steps,
    sample_recurrence_steps,
)
from .blocks.sandwich import SandwichBlock
from .config import RecurrentConfig
from .layers.attention import precompute_freqs_cis
from .layers.init import Linear

# The chunked loss (validation) splits the tokens into this many pieces: a fixed count, so the loop is static under
# `torch.compile(dynamic=True)` while the chunk lengths stay dynamic (see `RecurrentGPT.chunked_loss`).
LOSS_CHUNKS = 8

# A chunk of the loss is recomputed in the backward instead of saving its logits (no RNG inside); a no-op without
# gradients, which is how validation calls it.
_checkpoint = partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")


def prepare_attention_inputs(
    freqs_cis: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """
    The two per-batch inputs every attention layer needs, as `(rotary, mask)`: the RoPE rows and the sdpa mask.

    `rotary`: the rows of `freqs_cis` for this batch's positions. Without `position_ids` the first S rows, shape
    `(1, S, 1, hd // 2, 2)` (the training path, an exact no-op); with 1-D positions those rows in that order; with the
    `(B, S)` positions transformers passes for a left-padded batch, one row per sequence, shape `(B, S, 1, hd // 2, 2)`.

    `mask`: None without an `attention_mask` (the caller then uses sdpa's `is_causal=True`). Otherwise the `(B, S)`
    padding mask (1 = keep) becomes a `(B, 1, S, S)` bool mask that already contains the causal triangle, because
    some sdpa backends reject an explicit mask together with `is_causal=True`. True means attend. Every query keeps
    its own position (the diagonal): a row allowed to attend to nothing, a pad token at the start of a left-padded
    sequence, would give a NaN softmax row that spreads through the next layer's value matmul.
    """

    sequence_length = input_ids.shape[1]
    if position_ids is None:
        rotary = freqs_cis[:, :sequence_length]
    elif position_ids.dim() == 1:
        rotary = freqs_cis.index_select(1, position_ids.to(torch.long))
    elif position_ids.dim() == 2:
        rotary_rows = freqs_cis[0].index_select(0, position_ids.to(torch.long).reshape(-1))  # (B * S, 1, hd // 2, 2)
        rotary = rotary_rows.view(position_ids.shape[0], position_ids.shape[1], *rotary_rows.shape[1:])
    else:
        raise ValueError(f"position_ids must be 1-D (S,) or 2-D (B, S), got shape {tuple(position_ids.shape)}")

    if attention_mask is None:
        return rotary, None
    if attention_mask.dim() != 2 or attention_mask.shape[1] != sequence_length:
        raise ValueError(
            f"attention_mask must be (B, S) with S={sequence_length}, got shape {tuple(attention_mask.shape)}"
        )
    device = attention_mask.device
    keys_allowed = attention_mask.to(torch.bool)[:, None, None, :]  # (B, 1, 1, S): the keys each query may attend to
    causal = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device).tril()
    own_position = torch.eye(sequence_length, dtype=torch.bool, device=device)  # never leave a row fully masked
    return rotary, (keys_allowed & causal) | own_position


class TransformerModules(torch.nn.ModuleDict):
    """
    `ModuleDict` of the model parts; the annotations only give `model.transformer.<name>` a precise static type.
    """

    wte: torch.nn.Embedding  # token embedding
    prelude: torch.nn.ModuleList  # SandwichBlocks run once before the recurrence
    adapters: torch.nn.ModuleList  # one Linear per core block: [latent, block input] -> latent
    core_blocks: torch.nn.ModuleList  # one ModuleList of SandwichBlocks per core block
    coda: torch.nn.ModuleList  # SandwichBlocks run once after the recurrence
    ln_fs: torch.nn.ModuleList  # one LayerNorm per core block, applied to the block input
    ln_final: torch.nn.LayerNorm


class RecurrentGPT(torch.nn.Module):
    """
    Prelude, recurrent core blocks, coda, final norm and tied LM head; `step` seeds the recurrence sampler.
    """

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

        # The bf16 residual stream (`bf16_residual_stream`): "core" rounds the core blocks' norm outputs to the
        # autocast dtype, "all" the prelude's and coda's as well; see `RMSNorm`.
        core_bf16_stream = config.bf16_residual_stream != "none"
        outer_bf16_stream = config.bf16_residual_stream == "all"
        self.core_bf16_stream = core_bf16_stream

        # Construction order matters: it fixes the RNG consumption of the parameter init.
        prelude = torch.nn.ModuleList()
        for _ in range(config.n_layers_in_prelude):
            prelude.append(SandwichBlock(config, bf16_stream=outer_bf16_stream))

        core_blocks = torch.nn.ModuleList()
        for n_layers in n_layers_per_block:
            layers = torch.nn.ModuleList()
            for _ in range(n_layers):
                layers.append(SandwichBlock(config, bf16_stream=core_bf16_stream))
            core_blocks.append(layers)

        adapters = torch.nn.ModuleList()
        for _ in n_layers_per_block:
            adapters.append(Linear(config.n_embd * 2, config.n_embd, bias=False, init_method=config.init.fn("in_proj")))

        coda = torch.nn.ModuleList()
        for _ in range(config.n_layers_in_coda):
            coda.append(SandwichBlock(config, bf16_stream=outer_bf16_stream))

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
        """
        The RoPE table for every position up to `block_size`.
        """

        return precompute_freqs_cis(self.config.head_size, self.config.block_size, self.config.rope_settings.rope_base)

    def reset_parameters(self) -> None:
        """
        Re-initialize the modules that are not `Linear` (those init themselves): embedding and LayerNorms.
        """

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
        num_steps: NumSteps = None,
        return_token_losses_chunked_nograd: bool = False,
    ) -> dict[str, Tensor | None]:
        """
        One forward pass: embedding, prelude, the recurrent core blocks, coda, final norm, LM head and, given
        `labels`, the loss.

        Inputs. `labels` must be pre-shifted (the trainer's collate shifts; the HuggingFace wrapper shifts
        internally instead): the loss is `CE(logits[t], labels[t])`. `attention_mask` is a `(B, S)` padding mask
        (1 = keep), `position_ids` 1-D or `(B, S)`; both are None on the training path. `num_steps`: None (sample
        the depth per block), one (n_no_grad, k_with_grad) pair for all blocks, or one pair per core block.

        Outputs. `loss`: the mean cross-entropy over the valid labels (0 without labels). `log_ppl`: its detached
        copy. `logits`: the full fp32 `(B, S, padded_vocab)` logits when `return_logits`, else None.
        `token_losses`: the `(B, S)` fp32 per-token losses (zero at ignored positions) when
        `return_token_losses_chunked_nograd`, else None.

        Two loss paths. Training and every `return_logits` caller build the full logits and take the loss from
        them. `return_token_losses_chunked_nograd` (validation only, under `no_grad`) takes `chunked_loss` instead,
        which never holds the full logits (1 GiB at batch 4): validation used to keep them alive for the
        per-source token losses. Not for training: there the compiled step gained no memory from it and lost
        about 3 percent of speed.

        Compilation. Under `torch.compile` this frame becomes two graphs around one graph break at the
        `run_core_blocks` call (embedding and prelude; coda, final norm, LM head and loss). The core-block loop
        itself is eager by design, see `run_core_blocks`.
        """

        freqs_cis, mask = prepare_attention_inputs(self.freqs_cis, input_ids, attention_mask, position_ids)

        x = self.transformer.wte(input_ids)  # (B, S, E)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        for block in self.transformer.prelude:
            x = block(x, freqs_cis, mask)

        per_block_steps = normalize_num_steps(num_steps, len(self.transformer.core_blocks))
        x = self.run_core_blocks(x, freqs_cis, mask, per_block_steps)

        for block in self.transformer.coda:
            x = block(x, freqs_cis, mask)
        x = self.transformer.ln_final(x)

        loss = torch.as_tensor(0.0)
        logits: Tensor | None = None
        token_losses: Tensor | None = None
        if return_token_losses_chunked_nograd and labels is not None and not return_logits:
            loss, token_losses = self.chunked_loss(x, labels)
        else:
            logits = self.full_logits(x)  # (B, S, padded_vocab), float32
            if labels is not None:
                loss = self.loss(logits, labels)
                if return_token_losses_chunked_nograd:
                    token_losses = self.token_losses(logits, labels)
            if not return_logits:
                logits = None
        return {"loss": loss, "logits": logits, "token_losses": token_losses, "log_ppl": loss.clone().detach()}

    def full_logits(self, x: Tensor) -> Tensor:
        """
        The fp32 logits `(B, S, padded_vocab)` of the final hidden states `x`: LM head, cast, logit scale (if not 1).
        """

        logits: Tensor = self.lm_head(x).float()
        if self.config.init.logit_scale != 1:
            logits = logits * self.config.init.logit_scale
        return logits

    def mask_labels(self, labels: Tensor, n_classes: int | None = None) -> Tensor:
        """
        `labels` as the loss sees them: long, with every label outside `[0, n_classes)` (default: the padded
        vocabulary) replaced by `ignore_index`.
        """

        if n_classes is None:
            n_classes = self.lm_head.weight.shape[0]
        labels = labels.to(torch.long)
        invalid = (labels < 0) | (labels >= n_classes)
        return labels.masked_fill(invalid, self.ignore_index)

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Mean cross-entropy over the vocabulary; labels outside `[0, vocab)` count as `ignore_index`.
        """

        n_classes = logits.shape[-1]
        labels = self.mask_labels(labels, n_classes)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, n_classes), labels.view(-1), ignore_index=self.ignore_index
        )

    def token_losses(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Per-token cross-entropy `(B, S)`, fp32, zero at ignored positions (the masking of `loss`).
        """

        n_classes = logits.shape[-1]
        labels = self.mask_labels(labels, n_classes)
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, n_classes), labels.view(-1), ignore_index=self.ignore_index, reduction="none"
        )
        return losses.view(labels.shape)

    def chunked_loss(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        The loss without the full logits: `(mean loss, per-token losses)` of the final hidden states `x` against
        `labels`.

        The sequence is cut into `LOSS_CHUNKS` pieces; each piece runs the LM head, the fp32 cast, the logit scale
        and the per-token cross-entropy on its own, so at most one piece's `(B, S / LOSS_CHUNKS, vocab)` logits
        exist at a time. The mean is the sum of all token losses over the number of valid labels in the whole
        batch, the same as `loss` on the full logits up to summation order; the per-token losses match
        `token_losses`.

        Meant for validation under `no_grad` (the `return_token_losses_chunked_nograd` flag). With gradients enabled
        each piece runs under an activation checkpoint (recomputed in the backward instead of saved), which eager
        autograd handles piece by piece but a compiled backward does not: it recomputes every piece before
        consuming any, so training keeps the full-logits path.
        """

        labels = self.mask_labels(labels)
        sequence_length = x.shape[1]
        chunk_length = sequence_length // LOSS_CHUNKS
        losses = []
        for chunk_idx in range(LOSS_CHUNKS):
            start = chunk_idx * chunk_length
            length = chunk_length if chunk_idx < LOSS_CHUNKS - 1 else sequence_length - start
            losses.append(
                _checkpoint(self._chunk_token_losses, x.narrow(1, start, length), labels.narrow(1, start, length))
            )
        token_losses = torch.cat(losses, dim=1)
        valid_count = (labels != self.ignore_index).sum()
        return token_losses.sum() / valid_count, token_losses

    def _chunk_token_losses(self, x: Tensor, labels: Tensor) -> Tensor:
        """
        One piece of `chunked_loss`: the per-token losses of already masked `labels` from the hidden states `x`.
        """

        logits = self.full_logits(x)
        n_classes = logits.shape[-1]
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, n_classes), labels.reshape(-1), ignore_index=self.ignore_index, reduction="none"
        )
        return losses.view(labels.shape)

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def run_core_blocks(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None, per_block_steps: list[StepsPair | None]
    ) -> Tensor:
        """
        All core blocks in order: each is iterated on its input and added back onto it (residual around the whole
        block).

        Dynamo-disabled (not recursively, the callees still compile) because `run_core_block` is disabled too and
        dynamo cannot resume after a graph break inside a `for` loop: with the loop in `forward`, the whole `forward`
        frame would be skipped and run eagerly. Here the loop is eager and `forward` compiles around one plain call.
        """

        for block_idx, block_steps in enumerate(per_block_steps):
            block_out = self.run_core_block(x, freqs_cis, mask, block_steps, block_idx)
            x = block_out + x
        return x

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def run_core_block(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None, num_steps: StepsPair | None, block_idx: int
    ) -> Tensor:
        """
        Core block `block_idx` on `x`: normalise the input (`ln_fs`), draw the random latent state, project the
        normalised input through its half of the adapter once, then iterate the block n times without and k times
        with gradient (`num_steps`, or the sampler's draw when None).
        """

        transformer = self.transformer
        x_base = transformer.ln_fs[block_idx](x)
        x_latent = initialize_state(x)  # consumes the global RNG first, then (if sampling) the sampler's draw
        if self.core_bf16_stream and torch.is_autocast_enabled(x.device.type):
            # The bf16 stream: the latent enters the first iteration in the dtype every later iteration has. The
            # adapter GEMM would cast it to this dtype anyway, so the values are the same; what it avoids is a second
            # dtype variant of the compiled iteration, which pushed the recompile count past dynamo's limit.
            x_latent = x_latent.to(torch.get_autocast_dtype(x.device.type))

        steps: tuple[int, int] | tuple[Tensor, Tensor]
        if num_steps is None:
            steps = self.sample_block_depths(block_idx)
        else:
            steps = num_steps
        num_steps_no_grad, num_steps_with_grad = steps

        # ModuleList is not generic in the torch stubs, so indexing `core_blocks` needs the cast.
        layers = cast(torch.nn.ModuleList, transformer.core_blocks[block_idx])
        adapter = transformer.adapters[block_idx]
        # The adapter's input half is the same in every iteration: one GEMM per block instead of one per iteration.
        base_proj = adapter_base_projection(x_base, adapter)
        # `iterate_core_block` is dynamo-disabled, which makes it untyped for mypy; hence the explicit annotation.
        x_out: Tensor = iterate_core_block(
            x_latent,
            x_base,
            freqs_cis,
            mask,
            num_steps_no_grad,
            num_steps_with_grad,
            adapter=adapter,
            layers=layers,
            gradient_checkpointing=self.gradient_checkpointing,
            base_proj=base_proj,
        )
        return x_out

    @torch._dynamo.disable(recursive=False)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
    def sample_block_depths(self, block_idx: int = 0) -> tuple[Tensor, Tensor]:
        """
        (n no-grad, k backprop) iterations for core block `block_idx`: the poisson-lognormal-filling draw seeded by
        `self.step` in training, (`mean_recurrence`, 0) in eval mode.

        The seed is `self.step` alone, as in the reference implementation (the golden test in `test_model.py` fails on
        any change): `block_idx` only selects the block's means, so blocks with equal `(mean_recurrence,
        mean_backprop_depth)` draw the same `(n, k)` every step.
        """

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
