# (c) 2025-2026 Tobias Kerner, part of the multi-block recurrent thesis work.
# Released under Apache-2.0 alongside seal-rg/recurrent-pretraining code. See LICENSE.
"""Export checkpoints to HuggingFace format with safetensors."""

import json
import shutil
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict, is_dataclass

# Try to import from standalone file first (no recpre dependencies)
# Falls back to regular imports if standalone not available
try:
    from recpre.modeling_recurrent_gpt_standalone import RecurrentGPTConfig, RecurrentGPTForCausalLM
except ImportError:
    from recpre.configuration_recurrent_gpt import RecurrentGPTConfig
    from recpre.modeling_recurrent_gpt import RecurrentGPTForCausalLM

from transformers import AutoTokenizer


def copy_modeling_files(output_dir: Path, verbose: bool = True) -> None:
    """
    Copy modeling files to export directory for trust_remote_code loading.

    This enables loading the model with:
        AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    """
    if verbose:
        print(f"Copying standalone modeling files for trust_remote_code support...")

    # Get the recpre package directory
    import recpre
    recpre_dir = Path(recpre.__file__).parent

    # Copy the standalone modeling file
    src_modeling = recpre_dir / "modeling_recurrent_gpt_standalone.py"
    dst_modeling = output_dir / "modeling_recurrent_gpt.py"

    # Copy the standalone configuration file (thin wrapper)
    src_config = recpre_dir / "configuration_recurrent_gpt_standalone.py"
    dst_config = output_dir / "configuration_recurrent_gpt.py"

    if src_modeling.exists() and src_config.exists():
        shutil.copy2(src_modeling, dst_modeling)
        shutil.copy2(src_config, dst_config)
        if verbose:
            print(f"  ✓ Copied modeling_recurrent_gpt_standalone.py → modeling_recurrent_gpt.py")
            print(f"  ✓ Copied configuration_recurrent_gpt_standalone.py → configuration_recurrent_gpt.py")
            print(f"  ✓ Standalone files are fully self-contained (no recpre dependencies)")
    else:
        if verbose:
            if not src_modeling.exists():
                print(f"  Error: Standalone modeling file not found at {src_modeling}")
            if not src_config.exists():
                print(f"  Error: Standalone config file not found at {src_config}")
            print(f"  Falling back to legacy files...")

        # Fallback to old behavior
        files_to_copy = [
            "configuration_recurrent_gpt.py",
            "modeling_recurrent_gpt.py",
        ]

        for filename in files_to_copy:
            src = recpre_dir / filename
            dst = output_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                if verbose:
                    print(f"  ✓ Copied {filename}")
            else:
                if verbose:
                    print(f"  Warning: {filename} not found at {src}")


def export_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_path: str,
    model_description: str = "",
    training_config: Optional[Dict[str, Any]] = None,
    base_checkpoint: Optional[str] = None,
    model_config: Optional[Any] = None,  # RecurrentConfig object
    verbose: bool = True,
) -> None:
    """
    Export a training checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_dir: Directory to save the HF model
        tokenizer_path: Path to the tokenizer
        model_description: User-provided description of the model
        training_config: Complete training configuration dict
        base_checkpoint: Path to the base checkpoint used (if finetuning)
        model_config: RecurrentConfig object (extracted from model if available)
        verbose: Whether to print progress messages
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)

    if verbose:
        print(f"Extracting model configuration...")

    # Get source config (either provided or from training_config)
    if model_config is not None:
        source_config = model_config
    elif training_config is not None and "model_config" in training_config:
        source_config = training_config["model_config"]
    else:
        raise ValueError(
            "Model config not provided. Either pass model_config parameter "
            "or include model_config in training_config."
        )

    # Extract actual padded_vocab_size from checkpoint weights
    # The config may have wrong value if padding changed between registry and training
    if verbose:
        print(f"Checking vocab size in checkpoint...")
    try:
        # Find embedding weight key
        wte_key = None
        for k in state["model"].keys():
            if "wte.weight" in k or "transformer.wte.weight" in k:
                wte_key = k
                break

        if wte_key:
            actual_padded_vocab = state["model"][wte_key].shape[0]
            if actual_padded_vocab != source_config.padded_vocab_size:
                if verbose:
                    print(f"  Adjusting padded_vocab_size: {source_config.padded_vocab_size} → {actual_padded_vocab} (from checkpoint)")
                source_config.padded_vocab_size = actual_padded_vocab
                # Also update vocab_size to match
                source_config.vocab_size = actual_padded_vocab
            else:
                if verbose:
                    print(f"  Vocab size matches: {actual_padded_vocab}")
        else:
            if verbose:
                print(f"  Warning: Could not find embedding weight in checkpoint")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not extract vocab size from checkpoint: {e}")

    if verbose:
        print(f"Converting to HuggingFace config format...")

    # Convert RecurrentConfig to RecurrentGPTConfig
    config = RecurrentGPTConfig(
        vocab_size=source_config.vocab_size,
        padded_vocab_size=source_config.padded_vocab_size,
        n_embd=source_config.n_embd,
        num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads,
        intermediate_size=source_config.intermediate_size,
        block_size=source_config.block_size,
        n_layers_in_prelude=source_config.n_layers_in_prelude,
        n_layers_in_coda=source_config.n_layers_in_coda,
        n_layers_in_recurrent_block=source_config.n_layers_in_recurrent_block,
        mean_recurrence=source_config.mean_recurrence,
        mean_backprop_depth=source_config.mean_backprop_depth,
        injection_type=source_config.injection_type,
        sampling_scheme=source_config.sampling_scheme,
        block_class_name=source_config.block_class_name,
        norm_class_name=source_config.norm_class_name,
        mlp_class_name=source_config.mlp_class_name,
        nonlin_name=source_config.nonlin_name,
        init_strategy=source_config.init_strategy,
        init_orthogonal=source_config.init_orthogonal,
        state_init=source_config.state_init,
        bias=source_config.bias,
        norm_eps=source_config.norm_eps,
        tie_embeddings=source_config.tie_embeddings,
        qk_bias=source_config.qk_bias,
        activation_checkpoint_impl=source_config.activation_checkpoint_impl,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for optimal performance
    )

    # Add auto_map for trust_remote_code loading
    config.auto_map = {
        "AutoConfig": "configuration_recurrent_gpt.RecurrentGPTConfig",
        "AutoModelForCausalLM": "modeling_recurrent_gpt.RecurrentGPTForCausalLM",
    }

    if verbose:
        print(f"Creating HuggingFace model...")

    # Create HF model (fast initialization with meta device)
    with torch.device("meta"):
        model = RecurrentGPTForCausalLM(config)

    # Load weights (handle compiled model keys and Fabric wrappers)
    model_state_dict = {}
    for k, v in state["model"].items():
        # Remove Fabric/compilation prefixes
        clean_key = k.replace("_orig_mod._original_module.", "")
        clean_key = clean_key.replace("_orig_mod.", "")
        clean_key = clean_key.replace("_forward_module.", "")

        # The wrapper adds "model." prefix, so weights should be under "model.*"
        if not clean_key.startswith("model."):
            clean_key = "model." + clean_key

        # Skip forward module keys that aren't part of the model
        if "forward_module" not in k:
            model_state_dict[clean_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, assign=True)

    if verbose:
        if missing_keys:
            print(f"Warning: Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    # Move to CPU for saving
    model.to(device="cpu")

    if verbose:
        print(f"Saving model to {output_dir}...")

    # Save model in safetensors format
    model.save_pretrained(
        output_dir,
        safe_serialization=True,  # Use safetensors format
        max_shard_size="5GB",  # Split into shards if needed
    )

    # Copy modeling files for trust_remote_code support
    copy_modeling_files(output_dir, verbose=verbose)

    # Copy tokenizer
    if verbose:
        print(f"Copying tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        print(f"You may need to manually copy tokenizer files to {output_dir}")

    # Save metadata
    metadata = {
        "model_description": model_description,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
        "source_checkpoint": str(checkpoint_path),
        "training_step": state.get("microbatch_step", None),
        "tokens_seen": state.get("total_tokens", None),
    }

    # Add training hyperparameters if config provided
    if training_config:
        # Convert dataclass to dict if needed
        if is_dataclass(training_config):
            training_config = asdict(training_config)

        metadata["training_hyperparameters"] = {
            "optimizer": training_config.get("optimizer", None),
            "learning_rate": training_config.get("optim_config", {}).get("lr", None),
            "batch_size": training_config.get("world_batch_size", None),
            "max_steps": training_config.get("max_steps", None),
            "max_tokens": training_config.get("max_tokens", None),
            "warmup_steps": training_config.get("warmup_steps", None),
            "lr_schedule": training_config.get("lr_schedule", None),
            "grad_clip": training_config.get("grad_clip", None),
        }

        # Save complete config as well
        metadata_dir = output_dir / "training_metadata"
        metadata_dir.mkdir(exist_ok=True)
        with open(metadata_dir / "complete_training_config.json", "w") as f:
            json.dump(training_config, f, indent=2, default=str)

    # Save metadata JSON
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if verbose:
        print(f"✓ Successfully exported model to {output_dir}")
        print(f"  - Model weights (safetensors): model.safetensors*")
        print(f"  - Model config: config.json")
        print(f"  - Modeling files: configuration_recurrent_gpt.py, modeling_recurrent_gpt.py")
        print(f"  - Tokenizer files: tokenizer*")
        print(f"  - Metadata: export_metadata.json")
        if training_config:
            print(f"  - Full config: training_metadata/complete_training_config.json")
        print(f"\nTo load this model, use:")
        print(f"  AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")


def export_from_fabric_state(
    fabric,
    state: Dict[str, Any],
    cfg: Any,
    checkpoint_path: Optional[Path] = None,
) -> None:
    """
    Export directly from training state (called at end of training).

    Args:
        fabric: Lightning Fabric instance
        state: Training state dict
        cfg: Training config
        checkpoint_path: Optional specific checkpoint to export (otherwise uses state["model"])
    """
    if not cfg.export_to_hf:
        return

    # Determine output path
    if cfg.export_hf_path:
        output_dir = Path(cfg.export_hf_path)
    else:
        output_dir = Path(cfg.out_dir) / "hf_export" / cfg.run_name

    fabric.print(f"Exporting model to HuggingFace format at {output_dir}...")

    # If no checkpoint_path provided, save current state first
    if checkpoint_path is None:
        temp_checkpoint = Path(cfg.out_dir) / "temp_export_checkpoint.pth"
        fabric.save(temp_checkpoint, state)
        checkpoint_path = temp_checkpoint
        cleanup_temp = True
    else:
        cleanup_temp = False

    try:
        # Extract model config from the actual model object
        model = state["model"]
        # Unwrap Fabric/compiled wrappers to get the actual model
        if hasattr(model, "_forward_module"):
            model = model._forward_module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if hasattr(model, "_original_module"):
            model = model._original_module

        model_config = model.config if hasattr(model, "config") else cfg.model_config

        export_checkpoint_to_hf(
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            tokenizer_path=cfg.tokenizer_path,
            model_description=cfg.model_description,
            training_config=asdict(cfg) if is_dataclass(cfg) else cfg.__dict__,
            base_checkpoint=cfg.model_checkpoint,
            model_config=model_config,
            verbose=fabric.is_global_zero,
        )
    finally:
        if cleanup_temp and checkpoint_path.exists():
            checkpoint_path.unlink()

    fabric.print(f"✓ Model exported successfully!")
