"""Standalone HuggingFace-compatible implementation of RecurrentGPT.

This file is completely self-contained and requires only:
- torch
- transformers
- Standard library (math, typing, dataclasses, os)

NO dependencies on the `recpre` package - can be used anywhere!

Copyright: Lightning AI. Licensed under the Apache License 2.0.
Modified for standalone HuggingFace compatibility.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Tuple, List, Literal
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers import PretrainedConfig


# ==================== Configuration Classes ====================

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


@dataclass
class RoPESettings:
    """Settings for Rotary Position Embeddings."""
    use_rope: bool = True
    rope_condense_ratio: int = 1
    rope_base: int = 50_000


@dataclass
class MinimalRecurrentConfig:
    """Minimal config class for internal use (not exported to HF)."""
    vocab_size: int
    padded_vocab_size: int
    n_embd: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    block_size: int
    n_layers_in_prelude: int
    n_layers_in_coda: int
    n_layers_in_recurrent_block: List[int]
    mean_recurrence: List[int]
    mean_backprop_depth: List[int]
    injection_type: str
    sampling_scheme: str
    init_strategy: str
    init_orthogonal: bool
    state_init: str
    bias: bool
    norm_eps: float
    tie_embeddings: bool
    qk_bias: bool
    rope_settings: RoPESettings
    activation_checkpoint_impl: str
    skip_initialization: bool = False

    # Derived properties
    head_size: int = field(init=False)

    def __post_init__(self):
        self.head_size = self.n_embd // self.num_attention_heads

        # Normalize recurrence lists
        if isinstance(self.n_layers_in_recurrent_block, int):
            self.n_layers_in_recurrent_block = [self.n_layers_in_recurrent_block]
        if isinstance(self.mean_recurrence, int):
            self.mean_recurrence = [self.mean_recurrence]
        if isinstance(self.mean_backprop_depth, int):
            self.mean_backprop_depth = [self.mean_backprop_depth]

        # Replicate single values across blocks
        num_blocks = len(self.n_layers_in_recurrent_block)
        if len(self.mean_recurrence) == 1 and num_blocks > 1:
            self.mean_recurrence = self.mean_recurrence * num_blocks
        if len(self.mean_backprop_depth) == 1 and num_blocks > 1:
            self.mean_backprop_depth = self.mean_backprop_depth * num_blocks


# ==================== Utility Functions ====================

def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is >= n."""
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1):
    """Precompute rotary position embedding frequencies.

    Args:
        dim: Dimension of the embedding (should be head_dim)
        end: Maximum sequence length
        theta: Base for the frequency computation
        condense_ratio: Ratio to condense position indices

    Returns:
        Tensor of shape [1, end, 1, dim//2, 2] containing cos/sin frequencies
    """
    with torch.autocast("cuda", enabled=False):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(end, dtype=torch.float32, device=inv_freqs.device) / condense_ratio
        freqs = torch.outer(t, inv_freqs).float()
        return torch.stack([torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]], dim=4)


def apply_rotary_emb_complex_like(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [B, S, n_heads, head_dim]
        k: Key tensor of shape [B, S, n_heads, head_dim]
        freqs_cis: Precomputed frequencies of shape [1, S, 1, head_dim//2, 2]

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    with torch.autocast("cuda", enabled=False):
        qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
        rotated_qk_r2 = torch.stack(
            [
                qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
                qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        ).flatten(3)
        rotated_qk = rotated_qk_r2
        return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)


def get_init_std(init_strategy: str, dim: int, intermediate_dim: int, num_layers: int, layer_id: int = 0) -> dict:
    """Get initialization standard deviations for different components.

    Args:
        init_strategy: Name of initialization strategy (e.g., "takase", "scaled")
        dim: Model embedding dimension
        intermediate_dim: MLP intermediate dimension
        num_layers: Total number of effective layers
        layer_id: Current layer index

    Returns:
        Dict with initialization stds for different components
    """
    if init_strategy == "takase":
        return {
            "std": math.sqrt(2 / (5 * dim)),
            "out_proj": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
            "embedding": math.sqrt(2 / (5 * dim)),
            "embed_scale": math.sqrt(dim),
            "in_proj": math.sqrt(2 / (5 * dim)),
            "glu": math.sqrt(2 / (5 * dim)),
            "qkv": math.sqrt(2 / (5 * dim)),
            "out_attn": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
            "w3": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
        }
    elif init_strategy == "scaled":
        return {
            "std": math.sqrt(2 / (5 * dim)),
            "out_proj": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
            "embedding": math.sqrt(2 / (5 * dim)),
            "embed_scale": 1.0,
            "in_proj": math.sqrt(2 / (5 * dim)),
            "glu": math.sqrt(2 / (5 * dim)),
            "qkv": math.sqrt(2 / (5 * dim)),
            "out_attn": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
            "w3": math.sqrt(2 / (5 * dim)) / math.sqrt(2 * num_layers),
        }
    else:
        # Default fallback
        return {
            "std": 1.0 / math.sqrt(dim),
            "out_proj": 1.0 / math.sqrt(dim),
            "embedding": 1.0 / math.sqrt(dim),
            "embed_scale": 1.0,
            "in_proj": 1.0 / math.sqrt(dim),
            "glu": 1.0 / math.sqrt(dim),
            "qkv": 1.0 / math.sqrt(dim),
            "out_attn": 1.0 / math.sqrt(dim),
            "w3": 1.0 / math.sqrt(dim),
        }


# ==================== Model Components ====================

class RMSNorm_llama(torch.nn.Module):
    """Root Mean Square Layer Normalization (Llama-style).

    Saner dtype handling and slightly better for fusion.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class CausalSelfAttention(torch.nn.Module):
    """Causal self-attention with optional grouped-query attention (GQA)."""

    __constants__ = ("n_head", "n_kv_heads", "head_dim", "n_rep", "chunks")

    def __init__(self, config: MinimalRecurrentConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_heads
        shape = (self.n_head + 2 * self.n_kv_heads) * self.head_dim
        self.chunks = (
            config.n_embd,
            self.n_kv_heads * self.head_dim,
            self.n_kv_heads * self.head_dim,
        )

        # Projections
        self.Wqkv = torch.nn.Linear(config.n_embd, shape, bias=config.bias)
        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.proj = torch.nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, S, E = x.shape  # batch size, sequence length, embedding dimensionality

        qkv = self.Wqkv(x)
        q, k, v = qkv.split(self.chunks, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, S, self.n_head, self.head_dim)
        k = k.view(B, S, self.n_kv_heads, self.head_dim)
        v = v.view(B, S, self.n_kv_heads, self.head_dim)

        # Repeat k/v heads if n_kv_heads < n_heads (GQA)
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        # QK bias
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)

        # Apply rotary position embeddings
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)

        # Attention computation using PyTorch SDPA
        y = self.scaled_dot_product_attention(q, k, v, mask)
        y = y.reshape(B, S, E).contiguous()

        return self.proj(y)

    @staticmethod
    def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """Repeat key/value heads for grouped-query attention."""
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            torch.unsqueeze(x, dim=3)
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Compute scaled dot-product attention using PyTorch's native implementation."""
        # Transpose to [B, n_heads, S, head_dim] for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's efficient SDPA implementation
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # Causal masking handled by is_causal
            dropout_p=0.0,
            is_causal=True,  # Enable causal masking
        )

        # Transpose back to [B, S, n_heads, head_dim]
        return y.transpose(1, 2)


class GatedMLP(torch.nn.Module):
    """Gated MLP (SwiGLU-style) used in modern transformers."""

    def __init__(self, config: MinimalRecurrentConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = torch.nn.Linear(in_features, config.intermediate_size * 2, bias=config.bias)
        self.proj = torch.nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.nonlin = torch.nn.SiLU()  # SiLU activation

    def forward(self, x: Tensor) -> Tensor:
        # Split into gate and value
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2  # Gated activation
        return self.proj(x)


class SandwichBlock(torch.nn.Module):
    """Transformer block with sandwich normalization (4 norms: pre+post for attention and MLP)."""

    expanded = False

    def __init__(self, config: MinimalRecurrentConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = RMSNorm_llama(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = RMSNorm_llama(config.n_embd, eps=config.norm_eps)
        self.norm_3 = RMSNorm_llama(config.n_embd, eps=config.norm_eps)
        self.mlp = GatedMLP(config, layer_id=layer_id)
        self.norm_4 = RMSNorm_llama(config.n_embd, eps=config.norm_eps)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Attention with pre+post normalization
        x = self.norm_2(self.attn(self.norm_1(x), freqs_cis, mask) + x)
        # MLP with pre+post normalization
        x = self.norm_4(self.mlp(self.norm_3(x)) + x)
        return x

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.norm_1.weight)
        torch.nn.init.ones_(self.norm_2.weight)
        torch.nn.init.ones_(self.norm_3.weight)
        torch.nn.init.ones_(self.norm_4.weight)


# ==================== Main RecurrentGPT Model ====================

class RecurrentGPT(torch.nn.Module):
    """Recurrent GPT model with depth recurrence.

    Architecture:
        Embedding → Prelude (fixed layers) → Core Blocks (recurrent) → Coda (fixed layers) → LM Head

    Core blocks are recurrent: each block contains N layers and is executed for K iterations,
    with state injection between iterations.
    """

    def __init__(
        self,
        config: MinimalRecurrentConfig,
        objective: dict,
        gradient_checkpointing: bool = False,
        **_extras,
    ) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        # Calculate effective depth for initialization
        effective_expected_depth = (
            config.n_layers_in_prelude + config.n_layers_in_coda +
            sum(n_layers * mean_rec for n_layers, mean_rec in zip(
                config.n_layers_in_recurrent_block, config.mean_recurrence
            ))
        )

        # Get initialization parameters
        init_params = get_init_std(
            config.init_strategy,
            config.n_embd,
            config.intermediate_size,
            effective_expected_depth
        )
        self.emb_scale = init_params["embed_scale"]

        # Build model layers
        # 1. Prelude (non-recurrent entry layers)
        prelude = torch.nn.ModuleList([
            SandwichBlock(config, layer_id=i)
            for i in range(config.n_layers_in_prelude)
        ])

        # 2. Core recurrent blocks
        core_blocks = torch.nn.ModuleList()
        layer_offset = config.n_layers_in_prelude
        for block_idx, n_layers in enumerate(config.n_layers_in_recurrent_block):
            core_block = torch.nn.ModuleList([
                SandwichBlock(config, layer_id=i + layer_offset)
                for i in range(n_layers)
            ])
            core_blocks.append(core_block)
            layer_offset += n_layers * config.mean_recurrence[block_idx]

        # 3. Adapters for state injection
        adapters = torch.nn.ModuleList()
        for block_idx in range(len(config.n_layers_in_recurrent_block)):
            if config.injection_type == "linear":
                adapter = torch.nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias)
            elif config.injection_type == "ffn":
                adapter = GatedMLP(config, layer_id=block_idx, in_features=config.n_embd * 2)
            else:
                adapter = torch.nn.Identity()
            adapters.append(adapter)

        # 4. Coda (non-recurrent exit layers)
        o = layer_offset
        coda = torch.nn.ModuleList([
            SandwichBlock(config, layer_id=i + o)
            for i in range(config.n_layers_in_coda)
        ])

        # 5. Per-block normalizations (applied after each core block completes iterations)
        ln_fs = torch.nn.ModuleList([
            torch.nn.LayerNorm(config.n_embd, eps=config.norm_eps)
            for _ in range(len(config.n_layers_in_recurrent_block))
        ])

        # 6. Final normalization (before LM head)
        ln_final = torch.nn.LayerNorm(config.n_embd, eps=config.norm_eps)

        # Assemble transformer
        self.transformer = torch.nn.ModuleDict(dict(
            wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
            prelude=prelude,
            adapters=adapters,
            core_blocks=core_blocks,
            coda=coda,
            ln_fs=ln_fs,
            ln_final=ln_final,
        ))

        # LM head
        self.lm_head = torch.nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.objective = objective
        self.max_seq_length = config.block_size
        self.gradient_checkpointing = gradient_checkpointing

        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize weights (unless skip_initialization is True)
        if not config.skip_initialization:
            self.reset_parameters(init_params)

    def _precompute_freqs_cis(self):
        """Precompute RoPE frequency matrix."""
        dim = self.config.n_embd // self.config.num_attention_heads
        max_length = self.config.block_size
        freqs_cis = precompute_freqs_cis(
            dim,
            max_length,
            self.config.rope_settings.rope_base,
            self.config.rope_settings.rope_condense_ratio,
        )
        return freqs_cis

    def reset_parameters(self, init_params: dict) -> None:
        """Initialize model parameters."""
        # Initialize embedding
        std = init_params["embedding"]
        torch.nn.init.trunc_normal_(
            self.transformer.wte.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std
        )

        # Initialize normalizations
        for ln_f in self.transformer.ln_fs:
            torch.nn.init.ones_(ln_f.weight)
            torch.nn.init.zeros_(ln_f.bias)
        torch.nn.init.ones_(self.transformer.ln_final.weight)
        torch.nn.init.zeros_(self.transformer.ln_final.bias)

        # Initialize blocks (norms already initialized in block constructors)
        for block in list(self.transformer.prelude) + list(self.transformer.coda):
            block.reset_parameters()
        for core_block in self.transformer.core_blocks:
            for block in core_block:
                block.reset_parameters()

    def _canon_steps(self, steps) -> Optional[Tuple[int, int]]:
        """Canonicalize recurrence steps to (no_grad_steps, with_grad_steps) format."""
        if steps is None:
            return None
        if isinstance(steps, torch.Tensor):
            v = steps.detach().reshape(-1)
            if v.numel() == 2:
                return int(v[0].item()), int(v[1].item())
            elif v.numel() == 1:
                return int(v[0].item()), 0
            else:
                return int(v[0].item()), int(v[1].item())
        if isinstance(steps, (list, tuple)):
            return int(steps[0]), int(steps[1] if len(steps) > 1 else 0)
        return int(steps), 0

    def initialize_state(self, latent_tensor_merker: Tensor) -> Tensor:
        """Initialize recurrent state based on config.state_init."""
        if self.config.state_init == "none":
            return latent_tensor_merker
        if self.config.state_init == "none-detach":
            return latent_tensor_merker.detach()
        if self.config.state_init == "normal":
            return torch.randn_like(latent_tensor_merker)
        if self.config.state_init == "normal-small":
            return torch.randn_like(latent_tensor_merker).mul(0.2)
        elif self.config.state_init == "zero":
            return torch.zeros_like(latent_tensor_merker)
        # Default fallback
        return torch.randn_like(latent_tensor_merker).mul(0.2)

    def core_block_forward(
        self,
        x_latent: Tensor,
        x_base: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor],
        step: int,
        core_block: torch.nn.ModuleList,
        core_block_number: int
    ) -> Tensor:
        """Forward pass through a core block for one iteration.

        Args:
            x_latent: Current latent state
            x_base: Normalized input (fixed across iterations)
            freqs_cis: RoPE frequencies
            mask: Attention mask
            step: Current iteration number (unused in most injection types)
            core_block: The recurrent block layers
            core_block_number: Index of the current core block

        Returns:
            Updated latent state
        """
        # State injection
        if self.config.injection_type == "none":
            x_latent = x_latent
        elif self.config.injection_type == "add":
            x_latent = x_latent + x_base
        elif self.config.injection_type in ["linear", "ffn"]:
            # Concatenate current state with base input and project
            x_latent = self.transformer.adapters[core_block_number](
                torch.cat([x_latent, x_base], dim=-1)
            )

        # Apply all layers in the core block
        for layer in core_block:
            x_latent = layer(x_latent, freqs_cis, mask)

        return x_latent

    def iterate_forward(
        self,
        input_tensor: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor],
        num_steps_pair: Optional[Tuple[int, int]] = None,
        *,
        core_block: torch.nn.ModuleList,
        core_block_number: int,
    ) -> Tuple[Tensor, int, int, Tensor]:
        """Execute recurrence iterations for a core block.

        Args:
            input_tensor: Input to the recurrent block
            freqs_cis: RoPE frequencies
            mask: Attention mask
            num_steps_pair: (no_grad_steps, with_grad_steps) or None for random sampling
            core_block: The recurrent block layers
            core_block_number: Index of the current core block

        Returns:
            Tuple of (output, num_no_grad_steps, num_with_grad_steps, final_state)
        """
        # Normalize input
        x_base = self.transformer.ln_fs[core_block_number](input_tensor)

        # Initialize latent state
        x_latent = self.initialize_state(input_tensor)

        # Determine number of iterations
        if num_steps_pair is None:
            # Training mode: use mean_recurrence
            num_steps_no_grad = 0
            num_steps_with_grad = self.config.mean_recurrence[core_block_number]
        else:
            num_steps_no_grad, num_steps_with_grad = num_steps_pair

        # No-grad iterations (for efficiency during inference)
        with torch.no_grad():
            for step in range(num_steps_no_grad):
                x_latent = self.core_block_forward(
                    x_latent, x_base, freqs_cis, mask, step,
                    core_block=core_block,
                    core_block_number=core_block_number
                )

        # With-grad iterations (for training or when gradients needed)
        for step in range(num_steps_with_grad):
            x_latent = self.core_block_forward(
                x_latent, x_base, freqs_cis, mask, num_steps_no_grad + step,
                core_block=core_block,
                core_block_number=core_block_number
            )

        return x_latent, num_steps_no_grad, num_steps_with_grad, x_latent.detach()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        num_steps_pair: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None,
    ) -> dict:
        """Forward pass of RecurrentGPT.

        Args:
            input_ids: Input token IDs [B, S]
            attention_mask: Attention mask (unused, causal masking is automatic)
            position_ids: Position IDs (unused, RoPE handles positions)
            labels: Target labels for language modeling loss [B, S]
            return_logits: Whether to return logits (always True for HF compatibility)
            num_steps_pair: Recurrence configuration:
                - None: Use mean_recurrence from config
                - (n, m): Single tuple for all blocks (n no-grad steps, m with-grad steps)
                - [(n1, m1), (n2, m2), ...]: Per-block specification

        Returns:
            Dict with 'loss' (if labels provided) and 'logits'
        """
        # Get RoPE frequencies for this sequence length
        if position_ids is None:
            freqs_cis = self.freqs_cis[:, :input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        # 1. Embedding
        input_embeds = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            input_embeds = input_embeds * self.emb_scale

        # 2. Prelude (non-recurrent)
        x = input_embeds
        for block in self.transformer.prelude:
            x = block(x, freqs_cis, attention_mask)
        latent_tensor_merker = x

        # 3. Normalize num_steps_pair to per-block format
        if len(self.transformer.core_blocks) == 0:
            normalized_steps = []
        elif num_steps_pair is not None:
            if isinstance(num_steps_pair, list):
                assert len(num_steps_pair) == len(self.transformer.core_blocks), \
                    f"num_steps_pair list length ({len(num_steps_pair)}) must match number of core_blocks ({len(self.transformer.core_blocks)})"
                normalized_steps = [self._canon_steps(s) for s in num_steps_pair]
            else:
                # Single tuple, replicate for all blocks
                canonized = self._canon_steps(num_steps_pair)
                normalized_steps = [canonized] * len(self.transformer.core_blocks)
        else:
            normalized_steps = [None] * len(self.transformer.core_blocks)

        # 4. Core recurrent blocks
        x = latent_tensor_merker
        for block_idx, core_block in enumerate(self.transformer.core_blocks):
            steps = normalized_steps[block_idx]
            x, _, _, _ = self.iterate_forward(
                x,
                freqs_cis,
                attention_mask,
                steps,
                core_block=core_block,
                core_block_number=block_idx,
            )
            # Residual connection from block input
            x = x + latent_tensor_merker
            latent_tensor_merker = x

        # 5. Coda (non-recurrent)
        for block in self.transformer.coda:
            x = block(x, freqs_cis, attention_mask)

        # 6. Final normalization
        x = self.transformer.ln_final(x)

        # 7. LM head
        logits = self.lm_head(x).float()

        # 8. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Use ignore_index from objective
            ignore_index = self.objective.get("ignore_index", -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=ignore_index,
            )

        return {"loss": loss, "logits": logits}


# ==================== HuggingFace Wrapper ====================

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

    This wraps the RecurrentGPT model to make it compatible with transformers library
    for use with lm-eval-harness and HF model hub.

    Example:
        ```python
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "path/to/model",
            trust_remote_code=True
        )

        # Control recurrence at runtime
        import os
        os.environ["EVAL_RECURRENCE_STEPS"] = "12"  # All blocks use 12 iterations
        # or
        os.environ["EVAL_RECURRENCE_STEPS"] = "8,12,16"  # Per-block control
        ```
    """

    config_class = RecurrentGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SandwichBlock", "TransformerPreNormBlock"]
    _supports_cache_class = False

    # Tell HF about tied weights
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(self, config: RecurrentGPTConfig):
        super().__init__(config)

        # Create RoPESettings
        rope_settings = RoPESettings(
            use_rope=True,
            rope_condense_ratio=1,
            rope_base=int(config.rope_base),
        )

        # Calculate padded vocab size
        if config.padded_vocab_size is None:
            padded_vocab_size = find_multiple(config.vocab_size, 512)
        else:
            padded_vocab_size = config.padded_vocab_size

        # Create internal MinimalRecurrentConfig
        recurrent_config = MinimalRecurrentConfig(
            vocab_size=config.vocab_size,
            padded_vocab_size=padded_vocab_size,
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
            init_strategy=config.init_strategy,
            init_orthogonal=config.init_orthogonal,
            state_init=config.state_init,
            bias=config.bias,
            norm_eps=config.norm_eps,
            tie_embeddings=config.tie_embeddings,
            qk_bias=config.qk_bias,
            rope_settings=rope_settings,
            activation_checkpoint_impl=config.activation_checkpoint_impl,
            skip_initialization=True,  # Weights will be loaded from checkpoint
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
        """Tie weights between embedding and lm_head if configured."""
        if self.config.tie_embeddings:
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
            attention_mask: Attention mask (unused, causal masking is automatic)
            position_ids: Position IDs (unused, RoPE handles positions)
            labels: Labels for language modeling loss
            return_dict: Whether to return a dict or tuple
            num_steps_pair: Recurrence steps configuration:
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
            # else: training mode with no config → leave as None (triggers default behavior)

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
        raise NotImplementedError(
            "RecurrentGPT uses fixed vocabulary size. "
            "Resizing token embeddings is not supported."
        )


# Auto-register with transformers
try:
    from transformers import AutoModelForCausalLM, AutoConfig
    AutoConfig.register("recurrent_gpt", RecurrentGPTConfig)
    AutoModelForCausalLM.register(RecurrentGPTConfig, RecurrentGPTForCausalLM)
except ImportError:
    # transformers not installed, skip registration
    pass
