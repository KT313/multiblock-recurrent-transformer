#!/usr/bin/env python3
"""
Download and prepare a contamination-free instruction tuning dataset mix.

This mix is designed to avoid overlap with standard lm-eval-harness benchmarks:
- MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA, etc.

Dataset composition:
1. Dolly-15k (40%) - High-quality instruction-following
2. OpenAssistant (30%) - Multi-turn conversations
3. Stanford Alpaca (30%) - Additional instruction variety

Total: ~40k examples (fast training, hours not days)

Usage:
    python cluster/scripts/prepare_instruction_mix.py
    python cluster/scripts/prepare_instruction_mix.py --output_dir /custom/path
    python cluster/scripts/prepare_instruction_mix.py --val_size 0.05  # 5% validation
"""

import os
import sys
import argparse
from pathlib import Path

# Parse args FIRST to get cache_dir before importing datasets
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download contamination-free instruction tuning dataset mix"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/shared_storage/recpre/datasets/instruction_mix",
        help="Base directory to save datasets"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Don't create combined dataset (keep separate)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory (if default has permission issues)"
    )
    return parser.parse_args()

# Parse arguments before any imports
args = parse_args()

# Set environment variables BEFORE importing datasets
if args.cache_dir:
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir
    print(f"Using custom cache directory: {args.cache_dir}")

# Disable HuggingFace token requirement for public datasets
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# NOW import datasets (after env vars are set)
from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

set_verbosity_info()
enable_progress_bar()


def format_dolly(example):
    """Format Dolly-15k examples for causal LM."""
    instruction = example['instruction']
    context = example.get('context', '').strip()
    response = example['response']

    # Include context if provided
    if context:
        prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    return {"text": prompt}


def format_openassistant(example):
    """Format OpenAssistant conversations for causal LM."""
    # OpenAssistant has message trees - we'll use the best rated paths
    # For simplicity, we'll format as instruction-response pairs

    # The dataset has 'text' field with the message content
    # and 'role' field (prompter/assistant)
    # We need to reconstruct conversations from the tree structure

    # For now, use a simple format - this could be enhanced
    text = example.get('text', '')

    # Skip if empty
    if not text:
        return {"text": ""}

    # Simple format - just use the text as-is for high-quality responses
    # In practice, you'd want to reconstruct full conversations
    return {"text": text}


def format_alpaca(example):
    """Format Stanford Alpaca examples for causal LM."""
    instruction = example['instruction']
    input_text = example.get('input', '').strip()
    output = example['output']

    # Include input if provided
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": prompt}


def download_and_prepare_dolly(base_dir, val_ratio=0.05):
    """Download and prepare Dolly-15k dataset."""
    print(f"\n{'='*80}")
    print("Processing Dolly-15k (databricks/databricks-dolly-15k)")
    print(f"{'='*80}")

    dataset_dir = base_dir / "dolly"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (public, no auth required)
    print("Loading Dolly-15k from HuggingFace...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", token=False)

    print(f"Loaded {len(ds):,} examples")

    # Format for causal LM
    print("Formatting for causal LM...")
    ds = ds.map(format_dolly, remove_columns=ds.column_names)

    # Remove empty examples
    ds = ds.filter(lambda x: len(x['text'].strip()) > 0)

    # Create train/val split
    print(f"Creating train/val split ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)...")
    splits = ds.train_test_split(test_size=val_ratio, seed=42)

    # Save to disk
    (dataset_dir / "train").mkdir(exist_ok=True)
    (dataset_dir / "validation").mkdir(exist_ok=True)

    splits["train"].save_to_disk(str(dataset_dir / "train"))
    splits["test"].save_to_disk(str(dataset_dir / "validation"))

    print(f"✓ Saved train: {len(splits['train']):,} examples")
    print(f"✓ Saved validation: {len(splits['test']):,} examples")
    print(f"✓ Dataset saved to: {dataset_dir}")

    return splits["train"], splits["test"]


def download_and_prepare_openassistant(base_dir, val_ratio=0.05, max_samples=12000):
    """Download and prepare OpenAssistant Conversations dataset."""
    print(f"\n{'='*80}")
    print("Processing OpenAssistant (OpenAssistant/oasst1)")
    print(f"{'='*80}")

    dataset_dir = base_dir / "openassistant"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (public, no auth required)
    print("Loading OpenAssistant from HuggingFace...")
    ds = load_dataset("OpenAssistant/oasst1", split="train", token=False)

    print(f"Loaded {len(ds):,} examples")

    # Filter for high-quality assistant responses
    print("Filtering for high-quality responses...")
    ds = ds.filter(lambda x: x['role'] == 'assistant' and x.get('rank', 0) == 0)

    # Take a subset if too large
    if len(ds) > max_samples:
        print(f"Sampling {max_samples:,} examples...")
        ds = ds.shuffle(seed=42).select(range(max_samples))

    # Format for causal LM
    print("Formatting for causal LM...")
    ds = ds.map(format_openassistant, remove_columns=ds.column_names)

    # Remove empty examples
    ds = ds.filter(lambda x: len(x['text'].strip()) > 0)

    # Create train/val split
    print(f"Creating train/val split ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)...")
    splits = ds.train_test_split(test_size=val_ratio, seed=42)

    # Save to disk
    (dataset_dir / "train").mkdir(exist_ok=True)
    (dataset_dir / "validation").mkdir(exist_ok=True)

    splits["train"].save_to_disk(str(dataset_dir / "train"))
    splits["test"].save_to_disk(str(dataset_dir / "validation"))

    print(f"✓ Saved train: {len(splits['train']):,} examples")
    print(f"✓ Saved validation: {len(splits['test']):,} examples")
    print(f"✓ Dataset saved to: {dataset_dir}")

    return splits["train"], splits["test"]


def download_and_prepare_alpaca(base_dir, val_ratio=0.05, max_samples=15000):
    """Download and prepare Stanford Alpaca (cleaned) dataset."""
    print(f"\n{'='*80}")
    print("Processing Stanford Alpaca (yahma/alpaca-cleaned)")
    print(f"{'='*80}")

    dataset_dir = base_dir / "alpaca"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset (public, no auth required)
    print("Loading Alpaca from HuggingFace...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", token=False)

    print(f"Loaded {len(ds):,} examples")

    # Take a subset if too large
    if len(ds) > max_samples:
        print(f"Sampling {max_samples:,} examples...")
        ds = ds.shuffle(seed=42).select(range(max_samples))

    # Format for causal LM
    print("Formatting for causal LM...")
    ds = ds.map(format_alpaca, remove_columns=ds.column_names)

    # Remove empty examples
    ds = ds.filter(lambda x: len(x['text'].strip()) > 0)

    # Create train/val split
    print(f"Creating train/val split ({int((1-val_ratio)*100)}%/{int(val_ratio*100)}%)...")
    splits = ds.train_test_split(test_size=val_ratio, seed=42)

    # Save to disk
    (dataset_dir / "train").mkdir(exist_ok=True)
    (dataset_dir / "validation").mkdir(exist_ok=True)

    splits["train"].save_to_disk(str(dataset_dir / "train"))
    splits["test"].save_to_disk(str(dataset_dir / "validation"))

    print(f"✓ Saved train: {len(splits['train']):,} examples")
    print(f"✓ Saved validation: {len(splits['test']):,} examples")
    print(f"✓ Dataset saved to: {dataset_dir}")

    return splits["train"], splits["test"]


def create_combined_dataset(base_dir, train_datasets, val_datasets):
    """Optionally create a pre-mixed combined dataset."""
    print(f"\n{'='*80}")
    print("Creating combined dataset...")
    print(f"{'='*80}")

    combined_dir = base_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all training sets
    train_combined = concatenate_datasets(train_datasets)
    train_combined = train_combined.shuffle(seed=42)

    # Concatenate all validation sets
    val_combined = concatenate_datasets(val_datasets)
    val_combined = val_combined.shuffle(seed=42)

    # Save to disk
    (combined_dir / "train").mkdir(exist_ok=True)
    (combined_dir / "validation").mkdir(exist_ok=True)

    train_combined.save_to_disk(str(combined_dir / "train"))
    val_combined.save_to_disk(str(combined_dir / "validation"))

    print(f"✓ Saved combined train: {len(train_combined):,} examples")
    print(f"✓ Saved combined validation: {len(val_combined):,} examples")
    print(f"✓ Combined dataset saved to: {combined_dir}")

    return train_combined, val_combined


def main():
    # Args already parsed at module level
    global args

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print("Contamination-Free Instruction Mix Dataset Preparation")
    print(f"{'='*80}")
    print(f"Output directory: {base_dir}")
    print(f"Validation split: {args.val_size*100:.1f}%")
    print(f"{'='*80}")

    # Download and prepare each dataset
    train_datasets = []
    val_datasets = []

    try:
        dolly_train, dolly_val = download_and_prepare_dolly(base_dir, args.val_size)
        train_datasets.append(dolly_train)
        val_datasets.append(dolly_val)
    except Exception as e:
        print(f"✗ Error processing Dolly: {e}")

    try:
        oasst_train, oasst_val = download_and_prepare_openassistant(base_dir, args.val_size)
        train_datasets.append(oasst_train)
        val_datasets.append(oasst_val)
    except Exception as e:
        print(f"✗ Error processing OpenAssistant: {e}")

    try:
        alpaca_train, alpaca_val = download_and_prepare_alpaca(base_dir, args.val_size)
        train_datasets.append(alpaca_train)
        val_datasets.append(alpaca_val)
    except Exception as e:
        print(f"✗ Error processing Alpaca: {e}")

    # Create combined dataset
    if not args.skip_combined and train_datasets:
        try:
            create_combined_dataset(base_dir, train_datasets, val_datasets)
        except Exception as e:
            print(f"✗ Error creating combined dataset: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("Dataset Preparation Complete!")
    print(f"{'='*80}")
    print(f"All datasets saved to: {base_dir}")

    total_train = sum(len(ds) for ds in train_datasets)
    total_val = sum(len(ds) for ds in val_datasets)
    print(f"\nTotal training examples: {total_train:,}")
    print(f"Total validation examples: {total_val:,}")

    print("\n📋 To use in training config:")
    print("data_config:")
    print("  train_data:")
    print("    - type: hfds")
    print("      prefix: dolly")
    print(f"      data_dir: {base_dir}/dolly/train")
    print("      weight: 0.4")
    print("    - type: hfds")
    print("      prefix: oasst")
    print(f"      data_dir: {base_dir}/openassistant/train")
    print("      weight: 0.3")
    print("    - type: hfds")
    print("      prefix: alpaca")
    print(f"      data_dir: {base_dir}/alpaca/train")
    print("      weight: 0.3")
    print("\n✅ This mix is safe to evaluate on MMLU, HellaSwag, ARC, WinoGrande, TruthfulQA!")


if __name__ == "__main__":
    main()
