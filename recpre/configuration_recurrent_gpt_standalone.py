"""Minimal configuration file that imports from standalone modeling file.

This is a thin wrapper that allows HuggingFace's auto_map to find the config class
while keeping everything in a single standalone file.

NOTE: This file will be exported as configuration_recurrent_gpt.py,
and the modeling file will be exported as modeling_recurrent_gpt.py.
"""

# Import the config from the modeling file
# (which is renamed from modeling_recurrent_gpt_standalone.py during export)
from .modeling_recurrent_gpt import RecurrentGPTConfig

__all__ = ["RecurrentGPTConfig"]

# Re-export for HuggingFace's trust_remote_code mechanism
try:
    from transformers import AutoConfig
    AutoConfig.register("recurrent_gpt", RecurrentGPTConfig)
except ImportError:
    pass
