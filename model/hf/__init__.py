# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
HuggingFace `transformers` wrapper and `trust_remote_code` export.
"""

from .modeling import RecurrentGPTConfig, RecurrentGPTForCausalLM, export_to_hf, parse_recurrence_steps

__all__ = ["RecurrentGPTConfig", "RecurrentGPTForCausalLM", "export_to_hf", "parse_recurrence_steps"]
