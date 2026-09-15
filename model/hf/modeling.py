# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
HuggingFace `transformers` wrapper and export for `RecurrentGPT`.

`export_to_hf` writes safetensors + config.json and copies this package's modules flat next to them, so the folder
loads with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`. Flat because transformers' dynamic
module loader only follows `from .name import` lines within one directory and an `auto_map` entry is `module.Class`
with exactly one dot: every module `a/b.py` is written as `a_b.py` and `flatten_relative_imports` rewrites its import
lines. The `auto_map` names this module's flat name.
"""

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import RecurrentConfig, RoPESettings
from ..blocks.recurrence import NumSteps
from ..model import RecurrentGPT
from ..generation import GenerationState
from ..execution import ExecutionPolicy
from .forwarding import parse_recurrence_steps
from .forwarding import prepare_forward_state, resolve_evaluation_steps
from .exporting import (
    export_sources,
    flat_module_name,
    flatten_relative_imports,
    resolve_special_token_ids,
    validate_special_token_id,
    validate_single_token_id,
    prepare_export_metadata,
    transfer_export_weights,
)

__all__ = [
    "RecurrentGPTConfig", "RecurrentGPTForCausalLM", "export_to_hf", "parse_recurrence_steps",
    "mask_padded_vocabulary", "export_sources", "flat_module_name", "flatten_relative_imports",
    "resolve_special_token_ids", "validate_special_token_id", "validate_single_token_id",
]

# The `RecurrentConfig` fields stored in config.json (all of them except `name` and the nested `rope_settings`).
_MODEL_FIELDS = (
    "use_custom_kernels",
    "model_max_sequence_length",
    "n_embd",
    "intermediate_size",
    "num_attention_heads",
    "vocab_size",
    "padding_multiple",
    "padded_vocab_size",
    "tie_embeddings",
    "attn_impl",
    "norm_eps",
    "qk_bias",
    "bf16_residual_stream",
    "init_strategy",
    "init_orthogonal",
    "activation_checkpoint_impl",
    "injection_type",
    "n_layers_in_prelude",
    "n_layers_in_coda",
    "n_layers_in_recurrent_block",
    "state_init",
    "sampling_scheme",
    "mean_recurrence",
    "mean_backprop_depth",
)


def mask_padded_vocabulary(logits: torch.Tensor, vocab_size: int, padded_vocab_size: int) -> torch.Tensor:
    """
    `logits` with the embedding table's padding columns (`vocab_size:`) set to -inf; unchanged when the table is
    not padded. Those columns are trained on no target, and `generate(do_sample=True)` could otherwise draw an id the
    tokenizer cannot decode. Only the HF wrapper masks: the training loss must see the model's own logits (the golden
    tests check those numbers).
    """

    if padded_vocab_size <= vocab_size:
        return logits
    masked = logits.clone()  # a clone, not an in-place write: `logits` is part of the autograd graph
    masked[..., vocab_size:] = float("-inf")
    return masked


class RecurrentGPTConfig(PretrainedConfig):  # type: ignore[no-untyped-call]  # transformers' __init_subclass__ is untyped
    """
    `RecurrentConfig` fields as a `PretrainedConfig` (RoPE settings flattened to `rope_base`).

    The nested `rope_settings` of `RecurrentConfig.to_dict()` is accepted as well, so
    `RecurrentGPTConfig(**recurrent_config.to_dict())` keeps the RoPE base.
    """

    model_type = "recurrent_gpt"

    def __init__(
        self,
        rope_base: float | None = None,
        rope_settings: dict[str, Any] | RoPESettings | None = None,
        *,
        execution_precision: str | None = None,
        **kwargs: Any,
    ) -> None:
        # `rope_settings` is the nested form of the same field: `RecurrentGPTConfig(**recurrent_config.to_dict())`
        # passes it instead of `rope_base`, and dropping it silently would leave the RoPE base at its default.
        if rope_base is not None:
            RoPESettings(rope_base=rope_base)
        if rope_settings is not None:
            if isinstance(rope_settings, RoPESettings):
                nested_rope_base = rope_settings.rope_base
            else:
                nested_rope_base = RoPESettings(**rope_settings).rope_base
            if rope_base is not None and rope_base != nested_rope_base:
                raise ValueError(f"rope_base ({rope_base}) and rope_settings ({nested_rope_base}) disagree")
            rope_base = nested_rope_base
        if rope_base is None:
            rope_base = RoPESettings().rope_base

        # Defaults fill keys missing from a config.json (export_to_hf writes every field); the exported folder has no
        # access to config/model_architecture/, so the dataclass defaults apply.
        defaults = RecurrentConfig()
        values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            values[name] = kwargs.pop(name, getattr(defaults, name))
        # Validate raw API/serialized values before HF or numeric coercion can hide malformed settings.
        validated = RecurrentConfig(rope_settings=RoPESettings(rope_base=rope_base), **values)
        for name in ("n_layers_in_recurrent_block", "mean_recurrence", "mean_backprop_depth"):
            values[name] = getattr(validated, name)
        for name, value in values.items():
            setattr(self, name, value)
        self.rope_base = rope_base
        self.execution_precision = ExecutionPolicy(execution_precision).precision

        # Standard HF attribute names, derived from ours (`num_hidden_layers` = the expected unrolled depth).
        self.hidden_size = self.n_embd
        recurrent_depth = 0
        for n_layers, mean_recurrence in zip(self.n_layers_in_recurrent_block, self.mean_recurrence):
            recurrent_depth += n_layers * mean_recurrence
        self.num_hidden_layers = self.n_layers_in_prelude + self.n_layers_in_coda + recurrent_depth

        kwargs.setdefault("tie_word_embeddings", self.tie_embeddings)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if name in kwargs:
                kwargs[name] = validate_special_token_id(name, kwargs[name], int(self.vocab_size))
        super().__init__(**kwargs)

    @classmethod
    def from_recurrent_config(cls, config: RecurrentConfig, **kwargs: Any) -> "RecurrentGPTConfig":
        field_values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            field_values[name] = getattr(config, name)
        return cls(rope_base=config.rope_settings.rope_base, **field_values, **kwargs)

    def to_recurrent_config(self) -> RecurrentConfig:
        field_values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            field_values[name] = getattr(self, name)
        return RecurrentConfig(rope_settings=RoPESettings(rope_base=self.rope_base), **field_values)


class RecurrentGPTForCausalLM(PreTrainedModel, GenerationMixin):  # type: ignore[no-untyped-call]  # see above
    """
    Wraps `RecurrentGPT` for lm-eval / generation. Cached generation retains independent per-token/core latents;
    use_cache=False keeps legacy full-prefix latent resampling. Direct forward/scoring defaults remain uncached.
    """

    config_class = RecurrentGPTConfig
    base_model_prefix = "model"
    _no_split_modules = ["SandwichBlock"]
    _tied_weights_keys = {"model.lm_head.weight": "model.transformer.wte.weight"}

    def __init__(self, config: RecurrentGPTConfig) -> None:
        super().__init__(config)
        self.model: RecurrentGPT = RecurrentGPT(config.to_recurrent_config())
        # persistent here: `from_pretrained` materialises only the tensors of the saved state dict, a non-persistent
        # buffer would stay uninitialised (in the training model it is not persistent, so the config always wins)
        self.model.register_buffer("freqs_cis", self.model.freqs_cis, persistent=True)
        self.num_recurrent_blocks = len(self.model.transformer.core_blocks)
        self._training_forwards = 0  # the inner model's sampler step, see `forward`
        self.post_init()  # type: ignore[no-untyped-call]  # untyped in transformers

    def execution_policy(self) -> ExecutionPolicy:
        """Optional export precision, explicitly entered by the caller; forward/Trainer semantics stay unchanged."""
        return ExecutionPolicy(self.config.execution_precision)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """
        Weights are initialized by `RecurrentGPT` itself.
        """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        num_steps: NumSteps = None,
        logits_to_keep: int = 0,
        use_cache: bool = False,
        past_key_values: GenerationState | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...] | CausalLMOutputWithPast:
        """
        `num_steps` as in `RecurrentGPT.forward`. When None: in eval mode `EVAL_RECURRENCE_STEPS` ("12" or
        "4,12,4") if set, else the config's `mean_recurrence` per block; in training mode always the sampler (the env
        var sets zero backprop iterations, which would silently train the recurrence without gradient).

        Logits over the padding columns of the embedding table are -inf, so sampling only produces decodable ids. The
        inner model is untouched: masking there would change the training numerics.

        `labels` follow the HuggingFace contract and are shifted here: `model(x, labels=x).loss` is the next-token
        loss `CE(logits[t], x[t + 1])`, positions labelled -100 ignored. The inner `RecurrentGPT` expects pre-shifted
        labels, so it gets `labels=None`. `attention_mask` `(B, S)` (1 = keep) and `position_ids` (1-D or `(B, S)`)
        are forwarded to the inner model.

        Every sampled training forward counts as one sampler step: the inner model's `step` (which seeds the
        recurrence depths, `model/blocks/recurrence.py`) is set to `self._training_forwards` here, so a HF Trainer
        or PEFT run draws a new depth per forward instead of the fixed depth of step 0, and the micro-batches of
        one accumulated optimizer step draw independently. The value stays until the next forward, so an
        activation-checkpoint recompute in the backward draws the same depth. The counter is a plain attribute,
        not a buffer: `save_pretrained` does not store it and a resumed Trainer restarts the depth sequence at 0
        (the depths are a distribution, not a schedule). The native training loop writes `step` itself
        (`training/step.py`) and never goes through this wrapper.
        """

        # validate cache usage and select the output format
        state = prepare_forward_state(past_key_values, use_cache, self.training, labels, logits_to_keep)

        if return_dict is None:
            return_dict = self.config.return_dict

        # select recurrence depth and preserve the sampler context for backward replay
        sampler_step = num_steps is None and self.training
        if sampler_step:
            self.model.step = self._training_forwards

        if num_steps is None and not self.training:
            num_steps = resolve_evaluation_steps(self.num_recurrent_blocks, self.config.mean_recurrence)

        # run the native model and advance the successful training-forward counter
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,  # the inner loss is unshifted; the HF contract shifts, below
            return_logits=True,
            num_steps=num_steps,
            logits_to_keep=logits_to_keep,
            generation_state=state,
            use_cache=use_cache,
        )
        if sampler_step:
            self._training_forwards += 1

        # adapt native logits to the HF vocabulary, label shift and return contract
        return self._assemble_forward_output(outputs, labels, return_dict, state)

    def _assemble_forward_output(
        self, outputs: dict[str, torch.Tensor | None], labels: torch.Tensor | None,
        return_dict: bool | None, state: GenerationState | None,
    ) -> tuple[Any, ...] | CausalLMOutputWithPast:
        logits = outputs["logits"]
        assert logits is not None  # `return_logits=True`
        logits = mask_padded_vocabulary(logits, int(self.config.vocab_size), int(self.config.padded_vocab_size))
        loss: torch.Tensor | None = None
        if labels is not None:
            # `contiguous()`: both slices are views, and the inner `loss` flattens them with `view`.
            loss = self.model.loss(logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous())

        if return_dict:
            # transformers annotates the field as FloatTensor; ours is a plain float32 Tensor (there is no such subclass)
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=state)  # type: ignore[arg-type]
        if loss is None:
            return (logits, state) if state is not None else (logits,)
        return (loss, logits)

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # Shared layers need one slot per recurrence occurrence, and standard caches contain no fixed latents.
        return False

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """HF generation; cache sessions support ordinary greedy/multinomial decoding only.

        Every call creates its own session. To resume an explicit cached prefix use forward, not generate; training
        sample events must never accidentally inherit a cache from an earlier optimizer update.
        """
        generation = kwargs.get("generation_config") or (args[1] if len(args) > 1 else self.generation_config)

        def setting(name: str, default: Any = None) -> Any:
            if name in kwargs:
                return kwargs[name]
            value = getattr(generation, name, None)
            if value is None:
                value = getattr(self.generation_config, name, None)
            return default if value is None else value

        use_cache = setting("use_cache", True)
        if use_cache:
            if setting("num_beams", 1) != 1:
                raise ValueError("cached generation does not support beams; use_cache=False selects legacy generation")
            assistant = kwargs.get("assistant_model", args[6] if len(args) > 6 else None)
            if assistant is not None or setting("prompt_lookup_num_tokens"):
                raise ValueError("cached generation does not support assisted/speculative decoding")
            if setting("guidance_scale", 1) != 1:
                raise ValueError("cached generation does not support classifier-free guidance")
            if setting("prefill_chunk_size") is not None:
                raise ValueError("cached generation does not support HF chunked prefill")
            if setting("cache_implementation") is not None:
                raise ValueError("cached generation uses the model's own cache; omit cache_implementation")
            if kwargs.get("inputs_embeds") is not None:
                raise ValueError("cached generation requires input_ids, not inputs_embeds")
            if kwargs.get("past_key_values") is not None:
                raise ValueError("generate starts a fresh session; resume an explicit cache through forward")
        kwargs.setdefault("logits_to_keep", 0 if use_cache is False else 1)
        return super().generate(*args, **kwargs)  # type: ignore[misc]  # transformers dynamic model protocol

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Derive left-padded row positions; pass only new tokens once the per-occurrence cache exists.

        Legacy use_cache=False passes the entire sequence and continues resampling prefix latents. The full mask
        always accompanies cached queries, because storage columns and each row's real RoPE positions differ.
        """
        attention_mask: torch.Tensor | None = kwargs.get("attention_mask")
        position_ids: torch.Tensor | None = kwargs.get("position_ids")
        state: GenerationState | None = kwargs.get("past_key_values")
        use_cache: bool = kwargs.get("use_cache", False)
        if attention_mask is not None and position_ids is None:
            position_ids = (attention_mask.to(torch.long).cumsum(-1) - 1).clamp(min=0)
        if state is not None:
            if not isinstance(state, GenerationState) or not use_cache:
                raise ValueError("generation requires its own cache with use_cache=True")
            try:
                state.validate_prefix(input_ids, position_ids)
            except Exception:
                state.invalidate()
                raise
            start = state.get_seq_length()
            input_ids = input_ids[:, start:]
            if position_ids is not None:
                position_ids = position_ids[..., start:]
        model_inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
        for key in ("logits_to_keep", "num_steps", "use_cache"):
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        if state is not None:
            model_inputs["past_key_values"] = state
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> torch.nn.Embedding:
        """Preserve HF's lookup-only call; reject resize requests before any weights or config can change."""
        if new_num_tokens is None and pad_to_multiple_of is None:
            return self.model.transformer.wte
        raise NotImplementedError(
            "Vocabulary resizing is not supported by RecurrentGPT. Finalize the tokenizer and configure "
            "vocab_size/padded_vocab_size before constructing the model; resizing an existing model would "
            "leave native loss and saved vocabulary settings inconsistent."
        )

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        # transformers types the argument as Module; the contract (resize_token_embeddings) passes an Embedding.
        self.model.transformer.wte = value  # type: ignore[assignment]

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:
        self.model.lm_head = new_embeddings  # type: ignore[assignment]  # RecurrentGPT declares its own Linear subclass


AutoConfig.register("recurrent_gpt", RecurrentGPTConfig)
AutoModelForCausalLM.register(RecurrentGPTConfig, RecurrentGPTForCausalLM)


# --- export ----------------------------------------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # the `model/` package (this file is `model/hf/modeling.py`)

def export_to_hf(
    model: RecurrentGPT, config: RecurrentConfig, out_dir: str | Path, tokenizer_dir: str | Path | None = None,
    *, execution_policy: ExecutionPolicy | None = None, tokenizer: PreTrainedTokenizerBase | None = None,
    bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, pad_token_id: int | None = None,
    allow_missing_generation_metadata: bool = False,
) -> Path:
    """
    Write a self-contained HF folder with model and generation special-token metadata.

    Supply tokenizer_dir, a loaded tokenizer, or explicit IDs. None means unspecified, never an override of
    tokenizer metadata. Explicit IDs must agree with any tokenizer IDs. EOS is required unless
    allow_missing_generation_metadata=True intentionally permits a model-only export without EOS stopping.
    BOS/PAD remain optional; no IDs are guessed and no vocabulary resizing occurs. Metadata errors precede writes.
    Without a tokenizer, existing tokenizer artifacts require a fresh output directory or an explicit tokenizer.
    """

    # validate metadata before constructing or writing exported artifacts
    out_dir, tokenizer, metadata = prepare_export_metadata(
        config.vocab_size, out_dir, tokenizer_dir, tokenizer, bos_token_id, eos_token_id, pad_token_id,
        allow_missing_generation_metadata,
    )

    # configure the self-contained wrapper and its generation metadata
    this_module = flat_module_name(Path(__file__).resolve().relative_to(_PACKAGE_DIR))
    hf_config = RecurrentGPTConfig.from_recurrent_config(
        config, execution_precision=execution_policy.precision if execution_policy is not None else None,
        **metadata,
    )
    hf_config.auto_map = {
        "AutoConfig": f"{this_module}.RecurrentGPTConfig",
        "AutoModelForCausalLM": f"{this_module}.RecurrentGPTForCausalLM",
    }

    # Build the wrapper without allocating weights, then hand it `model`'s tensors under the wrapper's `model.` prefix.
    with torch.device("meta"):
        hf_model = RecurrentGPTForCausalLM(hf_config)
    for name, value in metadata.items():
        setattr(hf_model.generation_config, name, value)
    transfer_export_weights(model, hf_model)

    # publish weights, source modules and optional tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(out_dir, safe_serialization=True)
    export_sources(_PACKAGE_DIR, out_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return out_dir
