# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Model package: configuration, presets and the multi-block recurrent transformer."""

from typing import Any

from .config import RecurrentConfig, RoPESettings
from .presets import PRESETS
from .recurrent_gpt import RecurrentGPT


def build_model(name_or_config: str | RecurrentConfig, **overrides: Any) -> RecurrentGPT:
    """Instantiate `RecurrentGPT` from a preset name (with config overrides) or an existing config.

    Keyword arguments `ignore_index` and `gradient_checkpointing` go to the model, everything else to the config."""
    model_kwargs = {k: overrides.pop(k) for k in ("ignore_index", "gradient_checkpointing") if k in overrides}
    if isinstance(name_or_config, str):
        config = RecurrentConfig.from_name(name_or_config, **overrides)
    elif overrides:
        raise ValueError("config overrides are only accepted together with a preset name")
    else:
        config = name_or_config
    return RecurrentGPT(config, **model_kwargs)


__all__ = ["PRESETS", "RecurrentConfig", "RecurrentGPT", "RoPESettings", "build_model"]
