# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Architecture configuration of the multi-block recurrent transformer (the `crow-300m-final` code path only)."""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

from .layers.init import Init


def find_multiple(n: int, k: int) -> int:
    """Smallest multiple of `k` that is >= `n`."""
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class RoPESettings:
    rope_base: int = 50_000


# Fields that only ever had one value in the thesis run. They stay in the config (and in exported config.json files)
# so that a different value is rejected loudly instead of silently running a different architecture.
_FIXED_FIELD_VALUES: tuple[tuple[str, object], ...] = (
    ("attn_impl", "sdpa"),
    ("init_strategy", "takase"),
    ("init_orthogonal", True),
    ("activation_checkpoint_impl", "per-iteration"),
    ("injection_type", "linear"),
    ("state_init", "normal"),
    ("sampling_scheme", "poisson-lognormal-filling"),
)


@dataclass
class RecurrentConfig:
    """Hyper-parameters of `RecurrentGPT`. Per-block fields accept an int (broadcast) or one entry per core block."""

    name: str = ""
    # Core
    block_size: int = 2048
    n_embd: int = 1024
    intermediate_size: int | None = None
    num_attention_heads: int = 16
    # Vocabulary
    vocab_size: int = 32000
    padding_multiple: int = 2048
    padded_vocab_size: int | None = None
    tie_embeddings: bool = True
    # Block details
    rope_settings: RoPESettings = field(default_factory=RoPESettings)
    attn_impl: Literal["sdpa"] = "sdpa"
    norm_eps: float = 1e-6
    qk_bias: bool = True
    init_strategy: Literal["takase"] = "takase"
    init_orthogonal: Literal[True] = True
    activation_checkpoint_impl: Literal["per-iteration"] = "per-iteration"
    # Recurrent structure
    injection_type: Literal["linear"] = "linear"
    n_layers_in_prelude: int = 2
    n_layers_in_coda: int = 2
    n_layers_in_recurrent_block: int | list[int] = 4
    state_init: Literal["normal"] = "normal"
    sampling_scheme: Literal["poisson-lognormal-filling"] = "poisson-lognormal-filling"
    mean_recurrence: int | list[int] = 12
    mean_backprop_depth: int | list[int] = 8

    def __post_init__(self) -> None:
        # Nested settings arrive as plain dicts from YAML / JSON.
        if isinstance(self.rope_settings, dict):
            self.rope_settings = RoPESettings(**self.rope_settings)

        for field_name, allowed_value in _FIXED_FIELD_VALUES:
            actual_value = getattr(self, field_name)
            if actual_value != allowed_value:
                raise ValueError(f"{field_name}={actual_value!r} is not supported, only {allowed_value!r}")

        # Vocabulary: pad the embedding table up to a multiple of `padding_multiple`, unless the padded size is given
        # explicitly, in which case the vocabulary must fit into it.
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # Derived sizes.
        if self.n_embd % self.num_attention_heads != 0:
            raise ValueError("n_embd must be divisible by num_attention_heads")
        self.head_size = self.n_embd // self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

        # Per-block fields become lists with one entry per core block; the number of blocks is the length of
        # `n_layers_in_recurrent_block`.
        if isinstance(self.n_layers_in_recurrent_block, int):
            self.n_layers_in_recurrent_block = [self.n_layers_in_recurrent_block]
        num_blocks = len(self.n_layers_in_recurrent_block)
        self.mean_recurrence = self._broadcast("mean_recurrence", self.mean_recurrence, num_blocks)
        self.mean_backprop_depth = self._broadcast("mean_backprop_depth", self.mean_backprop_depth, num_blocks)

        # Expected unrolled depth of the model: every core block contributes its layers times its mean recurrence.
        # It scales the output-projection init (see `Init`).
        recurrent_depth = 0
        for n_layers, mean_recurrence in zip(self.n_layers_in_recurrent_block, self.mean_recurrence):
            recurrent_depth += n_layers * mean_recurrence
        self.effective_expected_depth = self.n_layers_in_prelude + self.n_layers_in_coda + recurrent_depth

        # Mean number of core-block layers the gradient flows through (layers times backprop depth, summed).
        self.n_layer = 0
        for n_layers, mean_backprop_depth in zip(self.n_layers_in_recurrent_block, self.mean_backprop_depth):
            self.n_layer += n_layers * mean_backprop_depth

        self.init = Init(self.n_embd, self.head_size, self.effective_expected_depth)

    @staticmethod
    def _broadcast(name: str, value: int | list[int], num_blocks: int) -> list[int]:
        """An int (or a one-element list) is repeated for every block; a longer list must have one entry per block."""
        if isinstance(value, int):
            values = [value]
        else:
            values = list(value)
        if len(values) == 1 and num_blocks > 1:
            values = values * num_blocks
        if len(values) != num_blocks:
            raise ValueError(f"{name} has {len(values)} entries but there are {num_blocks} recurrent blocks")
        return values

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "RecurrentConfig":
        """Build a config from a model architecture YAML (`config/model_architecture/<name>.yaml`: a mapping of the
        dataclass fields, nested settings as nested mappings), with keyword overrides applied on top."""
        with open(path, encoding="utf-8") as yaml_file:
            loaded = yaml.safe_load(yaml_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: expected a mapping of RecurrentConfig fields, got {type(loaded).__name__}")

        known_field_names: set[str] = set()
        for dataclass_field in fields(cls):
            known_field_names.add(dataclass_field.name)
        unknown_keys: list[str] = []
        for key in loaded:
            if key not in known_field_names:
                unknown_keys.append(key)
        if unknown_keys:
            raise ValueError(f"{path}: unknown RecurrentConfig key(s) {sorted(unknown_keys)}")

        kwargs: dict[str, Any] = dict(loaded)
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "RecurrentConfig":
        with open(path, encoding="utf-8") as json_file:
            kwargs = json.load(json_file)
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Dataclass fields only (derived attributes and the `Init` object are recomputed on load)."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(self.to_dict(), json_file, indent=2)
