# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Model package: configuration and the multi-block recurrent transformer."""

from pathlib import Path
from typing import Any

from .config import RecurrentConfig, RoPESettings
from .model import RecurrentGPT

# `build_model` keyword arguments that belong to `RecurrentGPT`, not to the config.
_MODEL_KWARGS = ("ignore_index", "gradient_checkpointing")


def build_model(config: RecurrentConfig | str | Path, **overrides: Any) -> RecurrentGPT:
    """A `RecurrentGPT` from a model architecture YAML (`config/model_architecture/<name>.yaml`) plus config
    overrides, or from an existing `RecurrentConfig`. `ignore_index` and `gradient_checkpointing` go to the model,
    every other keyword to the config."""
    model_kwargs: dict[str, Any] = {}
    for name in _MODEL_KWARGS:
        if name in overrides:
            model_kwargs[name] = overrides.pop(name)

    if isinstance(config, (str, Path)):
        config = RecurrentConfig.from_yaml(config, **overrides)
    elif overrides:
        raise ValueError("config overrides are only accepted together with a model architecture YAML path")
    return RecurrentGPT(config, **model_kwargs)


__all__ = ["RecurrentConfig", "RecurrentGPT", "RoPESettings", "build_model"]
