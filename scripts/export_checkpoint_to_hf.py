#!/usr/bin/env python3
# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.
"""
Standalone script to export training checkpoints to HuggingFace format.

Usage:
    python scripts/export_checkpoint_to_hf.py \\
        --checkpoint path/to/checkpoint.pth \\
        --output_dir outputs/hf_models/my-model \\
        --tokenizer_path path/to/tokenizer \\
        --description "Model finetuned on GLUE"

Example:
    python scripts/export_checkpoint_to_hf.py \\
        --checkpoint outputs/3313/checkpoints-SingleDeviceStrategy/step-00184000-recur1b-mig-3313.pth \\
        --output_dir outputs/hf_models/recur1b-mig-3313-finetuned \\
        --tokenizer_path /path/to/shared_storage/recpre/artifacts/tokenizer_llama32k \\
        --description "RecurrentGPT model finetuned on GLUE tasks" \\
        --base_checkpoint outputs/3313/checkpoints-SingleDeviceStrategy/step-00184000-recur1b-mig-3313.pth
"""

import sys
from pathlib import Path

# Add parent directory to path to import recpre
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from jsonargparse import CLI
from recpre.hf_export import export_checkpoint_to_hf


def main(
    checkpoint: str = None,
    output_dir: str = None,
    tokenizer_path: str = None,
    description: str = "",
    base_checkpoint: str = None,
    config_path: str = None,
) -> None:
    """
    Export a training checkpoint to HuggingFace format with safetensors.

    Args:
        checkpoint: Path to the .pth checkpoint file to export
        output_dir: Directory where the HF model will be saved
        tokenizer_path: Path to the tokenizer directory or HF model ID
        description: Optional description of the model for metadata
        base_checkpoint: Optional path to the base checkpoint (if this is a finetuned model)
        config_path: Optional path to model config YAML (if model config not in checkpoint)
    """
    # Validate required arguments
    if not checkpoint:
        raise ValueError("--checkpoint is required")
    if not output_dir:
        raise ValueError("--output_dir is required")
    if not tokenizer_path:
        raise ValueError("--tokenizer_path is required")

    import torch

    print("=" * 80)
    print("HuggingFace Model Export")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    if description:
        print(f"Description: {description}")
    if base_checkpoint:
        print(f"Base checkpoint: {base_checkpoint}")
    if config_path:
        print(f"Config: {config_path}")
    print("=" * 80)

    # Try to extract model config
    model_config = None

    # Option 1: Load from checkpoint if it contains the model object
    print("\nAttempting to extract model config from checkpoint...")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = state.get("model")

        # Unwrap Fabric/compiled wrappers
        if hasattr(model, "_forward_module"):
            model = model._forward_module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "_original_module"):
            model = model._original_module

        if hasattr(model, "config"):
            model_config = model.config
            print(f"✓ Extracted config from checkpoint (architecture: {model_config.architecture_class_name})")
        else:
            print("✗ Checkpoint model doesn't have config attribute")
    except Exception as e:
        print(f"✗ Could not extract config from checkpoint: {e}")

    # Option 2: Load from config file if provided
    if model_config is None and config_path:
        print(f"\nLoading config from {config_path}...")
        try:
            import yaml
            from recpre.config_dynamic import Config as DynamicConfig

            # Simple YAML load (no validation)
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Extract model_name and model_overwrite
            model_name = yaml_config.get('model_name')
            model_overwrite = yaml_config.get('model_overwrite', {})

            if not model_name:
                raise ValueError(f"Config file missing 'model_name' field")

            # Construct model config (same as CLISettings does)
            model_config = DynamicConfig.from_name(model_name, **model_overwrite)
            print(f"✓ Loaded config for model '{model_name}' (architecture: {model_config.architecture_class_name})")
        except Exception as e:
            print(f"✗ Could not load config from file: {e}")
            import traceback
            traceback.print_exc()

    if model_config is None:
        raise ValueError(
            "Could not extract model config. Either:\n"
            "  1. Ensure the checkpoint contains the full model object with config, or\n"
            "  2. Provide --config_path pointing to the model's YAML config"
        )

    export_checkpoint_to_hf(
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        model_description=description,
        training_config=None,  # Not available in standalone mode
        base_checkpoint=base_checkpoint,
        model_config=model_config,
        verbose=True,
    )

    print("=" * 80)
    print("Export complete!")
    print("=" * 80)
    print(f"\nYour model is ready at: {output_dir}")
    print("\nTo load with transformers:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    CLI(main)
