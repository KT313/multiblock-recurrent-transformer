#!/usr/bin/env python3
"""
Standalone evaluation script for converted HuggingFace model.

This script verifies that the converted HuggingFace model produces the same
validation losses as during training by evaluating on the same validation datasets
at multiple recurrence depths.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from functools import partial
from typing import Dict, List, Tuple, Optional

# Add recurrent-pretraining to path for importing utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "recurrent-pretraining"))

from recpre.tokenizer import Tokenizer
from recpre.huggingface_dataset import HuggingfaceDataset
from recpre.data_loading_utils import generic_collate_fn


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/path/to/shared_storage/recpre/outputs/hf_models/final-model-step-00141906-recur1b-mig-234" # "/path/to/shared_storage/recpre/outputs/hf_models/final-model-step-00151060-standalone"
TOKENIZER_PATH = "/path/to/shared_storage/recpre/artifacts/tokenizer_llama32k"

DATASETS = [
    {
        "name": "fineweb-edu-val",
        "path": "/path/to/shared_storage/recpre/datasets/fineweb-edu/validation",
        "data_signature": {"keys": ["text"], "format_fn": "pass_text"},
    },
    {
        "name": "flan-mixture-val",
        "path": "/path/to/shared_storage/recpre/datasets/flan_mixture_no_chat_template/validation",
        "data_signature": {
            "keys": ["instruction", "input", "output"],
            "format_fn": "concatenate_instruction_input_output",
        },
    },
]

PARTIAL_DEPTH_EVAL = [1, 2, 4, 8, 16]
MEAN_RECURRENCE = [12, 12, 12]  # Full recurrence from config

BLOCK_SIZE = 2048
BATCH_SIZE = 4
EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model and Tokenizer Loading
# ============================================================================

def load_model_and_tokenizer():
    """Load the converted HuggingFace model and tokenizer."""
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer(TOKENIZER_PATH)

    # Set pad_id to -100 (same as in train.py) - this is the ignore_index for loss
    tokenizer.pad_id = -100
    print(f"  Set pad_id to -100 (ignore_index for loss)")

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for CUDA scaled_dot_product_attention
    )
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


# ============================================================================
# Dataset Loading
# ============================================================================

def create_dataloader(
    dataset_config: Dict,
    tokenizer: Tokenizer,
    batch_size: int = BATCH_SIZE,
    block_size: int = BLOCK_SIZE,
) -> DataLoader:
    """Create a DataLoader for the specified dataset."""

    print(f"\nCreating dataloader for {dataset_config['name']}...")
    print(f"  Loading from: {dataset_config['path']}")

    data_signature = dataset_config['data_signature']
    print(f"  Format function: {data_signature['format_fn']}")
    print(f"  Keys: {data_signature['keys']}")

    # Create HuggingfaceDataset wrapper (handles data_signature formatting)
    dataset = HuggingfaceDataset(
        ds_name_or_path=dataset_config['path'],
        seed=12345,
        shuffle=False,
        num_processes=1,
        process_rank=0,
        data_id=dataset_config['name'],
        data_signature=data_signature,
        return_data_id=False,
    )

    # Create collate function
    collate_fn = partial(
        generic_collate_fn,
        tokenizer=tokenizer,
        block_size=block_size,
        pad_to_block_size=False,
        sequence_padding_multiple=None,
        add_bos=True,
        add_eos=True,
        collate_checks_enabled=True,
        all_block_size_tensors=False,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Single process for simplicity
        pin_memory=True if DEVICE == "cuda" else False,
    )

    print(f"  DataLoader created successfully")

    return dataloader


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_at_depth(
    model,
    dataloader: DataLoader,
    depth_steps: List[int] or int,
    eval_iters: int = EVAL_ITERS,
    device: str = DEVICE,
) -> Tuple[float, float]:
    """
    Evaluate model at a specific recurrence depth.

    Args:
        model: The HuggingFace model
        dataloader: DataLoader for validation data
        depth_steps: Recurrence steps (int or list of ints for per-block control)
        eval_iters: Number of batches to evaluate
        device: Device to run on

    Returns:
        (mean_loss, perplexity)
    """

    # Set environment variable to control recurrence
    if isinstance(depth_steps, list):
        os.environ["EVAL_RECURRENCE_STEPS"] = ",".join(map(str, depth_steps))
    else:
        os.environ["EVAL_RECURRENCE_STEPS"] = str(depth_steps)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (input_ids, labels, _) in enumerate(dataloader):
            if batch_idx >= eval_iters:
                break

            # Move to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    # Calculate mean loss and perplexity
    mean_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(mean_loss)).item()

    return mean_loss, perplexity


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def main():
    """Main evaluation loop."""

    print("=" * 80)
    print("HuggingFace Model Evaluation Script")
    print("=" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Store all results
    all_results = {}

    # Evaluate each dataset
    for dataset_config in DATASETS:
        dataset_name = dataset_config["name"]
        print("\n" + "=" * 80)
        print(f"Evaluating on: {dataset_name}")
        print("=" * 80)

        # Create dataloader
        dataloader = create_dataloader(dataset_config, tokenizer)

        # Store results for this dataset
        results = {}

        # Evaluate at each partial depth
        print(f"\nEvaluating at partial depths: {PARTIAL_DEPTH_EVAL}")
        for depth in PARTIAL_DEPTH_EVAL:
            print(f"\n  Depth {depth}...", end=" ", flush=True)
            loss, ppl = evaluate_at_depth(model, dataloader, depth)
            results[f"depth_{depth}"] = {"loss": loss, "ppl": ppl}
            print(f"loss={loss:.4f}, ppl={ppl:.2f}")

        # Evaluate at full mean_recurrence
        print(f"\n  Depth {MEAN_RECURRENCE} (full)...", end=" ", flush=True)
        loss, ppl = evaluate_at_depth(model, dataloader, MEAN_RECURRENCE)
        results["depth_full"] = {"loss": loss, "ppl": ppl, "depth": MEAN_RECURRENCE}
        print(f"loss={loss:.4f}, ppl={ppl:.2f}")

        all_results[dataset_name] = results

    # Print summary
    print("\n\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 60)

        # Print partial depths
        for depth in PARTIAL_DEPTH_EVAL:
            key = f"depth_{depth}"
            if key in results:
                loss = results[key]["loss"]
                ppl = results[key]["ppl"]
                print(f"  Depth {depth:2d}:        loss={loss:.4f}, ppl={ppl:7.2f}")

        # Print full depth
        if "depth_full" in results:
            loss = results["depth_full"]["loss"]
            ppl = results["depth_full"]["ppl"]
            depth_str = str(results["depth_full"]["depth"])
            print(f"  Depth {depth_str}: loss={loss:.4f}, ppl={ppl:7.2f}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
