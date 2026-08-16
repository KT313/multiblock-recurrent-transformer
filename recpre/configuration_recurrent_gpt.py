# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.
"""HuggingFace configuration for RecurrentGPT model.

This module provides HF transformers configuration compatibility for RecurrentGPT.
"""

import torch
from transformers import PretrainedConfig


class RecurrentGPTConfig(PretrainedConfig):
    """HuggingFace configuration for RecurrentGPT.

    This mirrors RecurrentConfig from config_dynamic.py but inherits from
    PretrainedConfig for HF compatibility.
    """

    model_type = "recurrent_gpt"

    def __init__(
        self,
        # Core architecture
        vocab_size: int = 32000,
        padded_vocab_size: int = None,  # If None, will be calculated from vocab_size
        n_embd: int = 1024,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        intermediate_size: int = 4096,
        block_size: int = 2048,

        # Recurrent structure
        n_layers_in_prelude: int = 2,
        n_layers_in_coda: int = 2,
        n_layers_in_recurrent_block: list = None,  # e.g., [4, 4, 4]
        mean_recurrence: list = None,              # e.g., [12, 12, 12]
        mean_backprop_depth: list = None,         # e.g., [8, 8, 8]

        # Injection and sampling
        injection_type: str = "add",
        sampling_scheme: str = "poisson-lognormal-filling",

        # Layer components
        block_class_name: str = "SandwichBlock",
        norm_class_name: str = "RMSNorm_llama",
        mlp_class_name: str = "GatedMLP",
        nonlin_name: str = "SiLU",

        # Initialization
        init_strategy: str = "takase",
        init_orthogonal: bool = True,
        state_init: str = "normal-small",

        # Other settings
        bias: bool = False,
        norm_eps: float = 1e-6,
        tie_embeddings: bool = True,
        qk_bias: bool = True,
        rope_base: float = 50000.0,
        activation_checkpoint_impl: str = "per-iteration",

        # HF standard fields
        torch_dtype = None,  # Will default to bfloat16
        **kwargs,
    ):
        # Set torch_dtype default
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Set defaults for lists
        if n_layers_in_recurrent_block is None:
            n_layers_in_recurrent_block = [4]
        if mean_recurrence is None:
            mean_recurrence = [12]
        if mean_backprop_depth is None:
            mean_backprop_depth = [8]

        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.n_embd = n_embd
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.block_size = block_size

        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_coda = n_layers_in_coda
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth

        self.injection_type = injection_type
        self.sampling_scheme = sampling_scheme

        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name

        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.state_init = state_init

        self.bias = bias
        self.norm_eps = norm_eps
        self.tie_embeddings = tie_embeddings
        self.qk_bias = qk_bias
        self.rope_base = rope_base
        self.activation_checkpoint_impl = activation_checkpoint_impl

        # Derived properties for HF compatibility
        self.hidden_size = n_embd
        self.num_hidden_layers = (
            n_layers_in_prelude +
            n_layers_in_coda +
            sum(n * m for n, m in zip(n_layers_in_recurrent_block, mean_recurrence))
        )

        # Initialize parent class first
        super().__init__(**kwargs)

        # Set torch_dtype after parent init (PretrainedConfig has special handling for this)
        self.torch_dtype = torch_dtype


# Auto-register the config with transformers
try:
    from transformers import AutoConfig
    AutoConfig.register("recurrent_gpt", RecurrentGPTConfig)
except ImportError:
    # transformers not installed, skip registration
    pass
