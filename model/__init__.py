# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Model package: configuration and the multi-block recurrent transformer."""

from pathlib import Path
from typing import Any

from .config import RecurrentConfig, RoPESettings
from .model import RecurrentGPT


def build_model(config: RecurrentConfig | str | Path, **overrides: Any) -> RecurrentGPT:
    """Instantiate `RecurrentGPT` from a model architecture YAML (`config/model_architecture/<name>.yaml`, with
    config overrides applied on top) or from an existing `RecurrentConfig`.

    Keyword arguments `ignore_index` and `gradient_checkpointing` go to the model, everything else to the config."""
    model_kwargs = {k: overrides.pop(k) for k in ("ignore_index", "gradient_checkpointing") if k in overrides}
    if isinstance(config, (str, Path)):
        config = RecurrentConfig.from_yaml(config, **overrides)
    elif overrides:
        raise ValueError("config overrides are only accepted together with a model architecture YAML path")
    return RecurrentGPT(config, **model_kwargs)


__all__ = ["RecurrentConfig", "RecurrentGPT", "RoPESettings", "build_model"]
