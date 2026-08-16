# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.
"""HuggingFace-compatible wrapper for RecurrentGPT model.

This module provides HF transformers model compatibility for RecurrentGPT while preserving
the exact training architecture. Use for lm-eval-harness and HF model hub.
"""

import os
import torch
from typing import Optional, Tuple, Union, List

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin

from .configuration_recurrent_gpt import RecurrentGPTConfig


def parse_recurrence_steps(steps_str: str, num_blocks: int) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """Parse recurrence steps from string format.

    Args:
        steps_str: String like "12" or "4,4,4" or "4,12,4"
        num_blocks: Number of recurrent blocks in the model

    Returns:
        Either a single tuple (n, 0) or list of tuples [(n1, 0), (n2, 0), ...]
        Format is (num_steps_no_grad, num_steps_with_grad)
    """
    steps_str = steps_str.strip()
    if not steps_str:
        return None

    if "," in steps_str:
        # Per-block specification: "4,4,4" or "4,12,4"
        steps = [int(s.strip()) for s in steps_str.split(",")]
        if len(steps) != num_blocks:
            raise ValueError(
                f"EVAL_RECURRENCE_STEPS has {len(steps)} values but model has {num_blocks} recurrent blocks. "
                f"Expected format: '{','.join(['N']*num_blocks)}' or single value 'N'"
            )
        return [(s, 0) for s in steps]
    else:
        # Single value: "12" - will be replicated for all blocks
        return (int(steps_str), 0)


class RecurrentGPTForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible wrapper for RecurrentGPT.

    This wraps the RecurrentGPT model from model_dynamic.py to make it compatible
    with transformers library for use with lm-eval-harness and HF model hub.
    """

    config_class = RecurrentGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SandwichBlock", "TransformerPreNormBlock"]
    _supports_cache_class = False  # RecurrentGPT doesn't use HF cache classes yet

    # Tell HF about tied weights (embeddings and lm_head share weights)
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: RecurrentGPTConfig):
        super().__init__(config)

        # Convert HF config to RecurrentConfig and instantiate model
        from recpre.config_dynamic import RecurrentConfig, RoPESettings
        from recpre.model_dynamic import RecurrentGPT

        # Create RoPESettings
        rope_settings = RoPESettings(
            use_rope=True,
            rope_condense_ratio=1,
            rope_base=int(config.rope_base),
        )

        # Create RecurrentConfig from HF config
        recurrent_config = RecurrentConfig(
            vocab_size=config.vocab_size,
            padded_vocab_size=config.padded_vocab_size,
            n_embd=config.n_embd,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            block_size=config.block_size,
            n_layers_in_prelude=config.n_layers_in_prelude,
            n_layers_in_coda=config.n_layers_in_coda,
            n_layers_in_recurrent_block=config.n_layers_in_recurrent_block,
            mean_recurrence=config.mean_recurrence,
            mean_backprop_depth=config.mean_backprop_depth,
            injection_type=config.injection_type,
            sampling_scheme=config.sampling_scheme,
            block_class_name=config.block_class_name,
            norm_class_name=config.norm_class_name,
            mlp_class_name=config.mlp_class_name,
            nonlin_name=config.nonlin_name,
            init_strategy=config.init_strategy,
            init_orthogonal=config.init_orthogonal,
            state_init=config.state_init,
            bias=config.bias,
            norm_eps=config.norm_eps,
            tie_embeddings=config.tie_embeddings,
            qk_bias=config.qk_bias,
            rope_settings=rope_settings,
            activation_checkpoint_impl=config.activation_checkpoint_impl,
            skip_initialization=True,  # We'll load weights from checkpoint
        )

        # Instantiate the actual RecurrentGPT model
        self.model = RecurrentGPT(
            config=recurrent_config,
            objective={"ignore_index": -100},
            gradient_checkpointing=False,
        )

        # Store number of recurrent blocks for step configuration
        self.num_recurrent_blocks = len(config.n_layers_in_recurrent_block)

    def tie_weights(self):
        """Tie weights between embedding and lm_head if configured.

        This is called by HuggingFace during save/load to properly handle
        weight sharing between the embedding layer and language model head.
        """
        if self.config.tie_embeddings:
            # Tie lm_head weight to embedding weight
            self._tie_or_clone_weights(
                self.model.lm_head,
                self.model.transformer.wte
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        num_steps_pair: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass compatible with HF transformers.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for language modeling loss
            return_dict: Whether to return a dict or tuple
            num_steps_pair: Recurrence steps configuration. Can be:
                - None: Auto-determined based on mode and environment
                - (n, 0): Single tuple for all blocks (n no-grad steps, 0 with-grad)
                - [(n1, 0), (n2, 0), ...]: Per-block specification
                Set via EVAL_RECURRENCE_STEPS env var or pass directly.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine recurrence steps if not provided
        if num_steps_pair is None:
            # Check environment variable first (allows runtime configuration)
            env_steps = os.environ.get("EVAL_RECURRENCE_STEPS", "").strip()
            if env_steps:
                try:
                    num_steps_pair = parse_recurrence_steps(env_steps, self.num_recurrent_blocks)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Failed to parse EVAL_RECURRENCE_STEPS='{env_steps}': {e}\n"
                        f"Expected format: 'N' (single value) or 'N1,N2,N3' (per-block, {self.num_recurrent_blocks} values)"
                    )
            # If in eval mode and no env var, use mean_recurrence from config (deterministic)
            elif not self.training:
                mean_rec = self.config.mean_recurrence
                if isinstance(mean_rec, list):
                    # Multi-block model with per-block mean_recurrence
                    num_steps_pair = [(n, 0) for n in mean_rec]
                else:
                    # Single value, will be replicated for all blocks
                    num_steps_pair = (mean_rec, 0)
            # else: training mode with no config → leave as None (triggers random sampling)

        # Call the underlying RecurrentGPT model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_logits=True,
            num_steps_pair=num_steps_pair,
        )

        loss = outputs.get("loss")
        logits = outputs.get("logits")

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Prepare inputs for generation."""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def get_input_embeddings(self):
        """Get input embeddings (required by PreTrainedModel)."""
        return self.model.transformer.wte

    def set_input_embeddings(self, value):
        """Set input embeddings (required by PreTrainedModel)."""
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        """Get output embeddings (required by PreTrainedModel)."""
        return self.model.lm_head

    def set_output_embeddings(self, value):
        """Set output embeddings (required by PreTrainedModel)."""
        self.model.lm_head = value

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """Resize token embeddings (required by PreTrainedModel)."""
        # RecurrentGPT uses fixed vocab size, so this is not supported
        raise NotImplementedError(
            "RecurrentGPT uses fixed vocabulary size. "
            "Resizing token embeddings is not supported."
        )


# Auto-register the model with transformers
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.register(RecurrentGPTConfig, RecurrentGPTForCausalLM)
except ImportError:
    # transformers not installed, skip registration
    pass
