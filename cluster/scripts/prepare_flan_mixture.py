#!/usr/bin/env python3
"""
Prepare FLAN-Focused Balanced Mixture Dataset for Instruction Finetuning

This script downloads and processes 8 datasets to create a 400K example mixture
optimized for benchmark performance on ARC, HellaSwag, MMLU, GSM8K, HumanEval, etc.

Dataset Composition (Option 1 - Recommended):
- 40% FLAN Collection (160K): Reasoning & commonsense
- 25% Math (100K): MetaMathQA (60K) + Orca-Math (40K)
- 15% Code (60K): Evol-Instruct-Code (50K) + Code Alpaca (10K)
- 10% Explanation (40K): SlimOrca-Dedup (GPT-4, cleaned & deduplicated)
- 5% Conversational (20K): ShareGPT filtered
- 5% Complex Instructions (20K): WizardLM Evol-Instruct V2

Features:
- Chain-of-Thought (CoT) variants for reasoning tasks
- Few-shot prompt variants (1-3 examples)
- Input inversions (30% of examples)
- Quality control and deduplication
- Train/val split (95%/5%)

Usage:
    python prepare_flan_mixture.py --output_dir /path/to/output [options]

Author: Claude Code
Date: 2025-10-22
"""

import os
import sys
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

# Parse args FIRST before importing datasets (to set cache dir)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare FLAN-Focused Balanced Mixture for instruction finetuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed datasets"
    )

    # Cache directory argument (IMPORTANT for permission issues)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory (if default /home/user/.cache has permission issues)"
    )

    # Dataset size arguments
    parser.add_argument(
        "--total_examples",
        type=int,
        default=400000,
        help="Total number of examples to generate (default: 400,000)"
    )

    parser.add_argument(
        "--flan_count",
        type=int,
        default=None,
        help="FLAN Collection examples (default: 40%% of total)"
    )

    parser.add_argument(
        "--metamath_count",
        type=int,
        default=None,
        help="MetaMathQA examples (default: 15%% of total)"
    )

    parser.add_argument(
        "--orca_math_count",
        type=int,
        default=None,
        help="Orca-Math examples (default: 10%% of total)"
    )

    parser.add_argument(
        "--evol_code_count",
        type=int,
        default=None,
        help="Evol-Instruct-Code examples (default: 12.5%% of total)"
    )

    parser.add_argument(
        "--code_alpaca_count",
        type=int,
        default=None,
        help="Code Alpaca examples (default: 2.5%% of total)"
    )

    parser.add_argument(
        "--openorca_count",
        type=int,
        default=None,
        help="SlimOrca-Dedup examples (default: 10%% of total)"
    )

    parser.add_argument(
        "--sharegpt_count",
        type=int,
        default=None,
        help="ShareGPT examples (default: 5%% of total)"
    )

    parser.add_argument(
        "--wizardlm_count",
        type=int,
        default=None,
        help="WizardLM Evol-Instruct V2 examples (default: 5%% of total)"
    )

    # Augmentation arguments
    parser.add_argument(
        "--add_cot_variants",
        action="store_true",
        help="Add Chain-of-Thought variants for reasoning tasks"
    )

    parser.add_argument(
        "--add_fewshot_variants",
        action="store_true",
        help="Add few-shot (1-3 examples) variants"
    )

    parser.add_argument(
        "--fewshot_ratio",
        type=float,
        default=0.3,
        help="Ratio of examples to convert to few-shot (default: 0.3)"
    )

    parser.add_argument(
        "--add_input_inversions",
        action="store_true",
        help="Add input inversion variants (swap input/output)"
    )

    parser.add_argument(
        "--inversion_ratio",
        type=float,
        default=0.3,
        help="Ratio of examples to invert (default: 0.3)"
    )

    # Other arguments
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05 = 5%%)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show plan without downloading"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data processing"
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/path/to/shared_storage/recpre/artifacts/tokenizer_llama32k",
        help="Path to tokenizer for verification samples (default: cluster path)"
    )

    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply full chat template formatting (default: False, only add generation tags for masking)"
    )

    return parser.parse_args()


# Parse arguments BEFORE importing datasets
args = parse_arguments()

# Set environment variables BEFORE importing datasets (to avoid permission issues)
if args.cache_dir:
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = args.cache_dir
    print(f"Using custom cache directory: {args.cache_dir}")
else:
    # Use a sensible default that's likely writable
    default_cache = "/path/to/fast_storage/.cache"
    if os.path.exists("/path/to/fast_storage"):
        os.environ["HF_HOME"] = default_cache
        os.environ["HF_DATASETS_CACHE"] = default_cache
        os.environ["TRANSFORMERS_CACHE"] = default_cache
        os.environ["HF_HUB_CACHE"] = default_cache
        print(f"Using default writable cache: {default_cache}")

# Disable HuggingFace token requirement for public datasets
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# NOW import datasets (after env vars are set)
try:
    from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
    from datasets.utils.logging import set_verbosity_info, enable_progress_bar
    from tqdm.auto import tqdm
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import gc
    import shutil

    set_verbosity_info()
    enable_progress_bar()
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install datasets tqdm numpy pyarrow")
    raise

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# Data Format Conversion
# =============================================================================

def standardize_format(example: Dict) -> Dict:
    """Convert various formats to standard instruction format.

    Standard format:
    {
        "instruction": str,  # The task description or question
        "input": str,        # Optional context (can be "")
        "output": str        # The expected response
    }
    """
    # If already in standard format (or close to it)
    if "instruction" in example and "output" in example:
        return {
            "instruction": str(example["instruction"]),
            "input": str(example.get("input", "")),
            "output": str(example["output"])
        }

    # FLAN format: inputs → instruction, targets → output
    if "inputs" in example and "targets" in example:
        return {
            "instruction": str(example["inputs"]),
            "input": "",
            "output": str(example["targets"])
        }

    # Conversation formats (ShareGPT, SlimOrca, etc.)
    if "conversations" in example and isinstance(example["conversations"], list):
        convs = example["conversations"]

        # SlimOrca format: [{"from": "system/human/gpt", "value": "..."}]
        if convs and "from" in convs[0]:
            system_msg = ""
            human_msg = ""
            gpt_msg = ""

            for turn in convs:
                turn_from = turn.get("from", "")
                turn_value = turn.get("value", "")

                if turn_from == "system":
                    system_msg = turn_value
                elif turn_from == "human":
                    human_msg = turn_value
                elif turn_from == "gpt":
                    gpt_msg = turn_value

            # Use system prompt as input context if present
            return {
                "instruction": human_msg,
                "input": system_msg if system_msg else "",
                "output": gpt_msg
            }

        # ShareGPT format: [{"value": "..."}, {"value": "..."}]
        elif len(convs) >= 2:
            instruction = convs[0].get("value", "")
            output = convs[1].get("value", "")
            return {
                "instruction": instruction,
                "input": "",
                "output": output
            }

    # OpenOrca format: question → instruction, response → output
    if "question" in example and "response" in example:
        return {
            "instruction": str(example["question"]),
            "input": str(example.get("system_prompt", "")),
            "output": str(example["response"])
        }

    # Code format: instruction/problem → instruction, solution/output → output
    if "problem" in example and "solution" in example:
        return {
            "instruction": str(example["problem"]),
            "input": "",
            "output": str(example["solution"])
        }

    # Math format: query/problem → instruction, response/answer → output
    if "query" in example and "response" in example:
        return {
            "instruction": str(example["query"]),
            "input": "",
            "output": str(example["response"])
        }

    # Orca-Math format: question → instruction, answer → output
    if "question" in example and "answer" in example:
        return {
            "instruction": str(example["question"]),
            "input": "",
            "output": str(example["answer"])
        }

    # Fallback: Try to infer
    raise ValueError(f"Cannot convert example to standard format: {list(example.keys())}")


def convert_to_chat_format(example: Dict) -> Dict:
    """Convert instruction/input/output format to chat messages format.

    This prepares the data for use with chat templates during training.
    The training config YAML will specify the data_signature that tells
    the dataloader to use apply_chat_template_supervise_assistant, which will:
    1. Apply the tokenizer's chat template (e.g., Llama-2 style)
    2. Mask user tokens so only assistant tokens are trained

    Args:
        example: Dict with "instruction", "input", "output" keys

    Returns:
        Dict with "messages" list (OpenAI-style chat format)
    """
    # Combine instruction with optional input context
    user_content = example["instruction"]
    if example.get("input") and example["input"].strip():
        user_content += "\n\n" + example["input"]

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]
    }


# =============================================================================
# Dataset Downloaders & Processors
# =============================================================================

def streaming_download_and_save(
    dataset_name: str,
    hf_dataset: str,
    target_count: int,
    output_path: Path,
    max_tokens: int = 2048,
    split: str = "train",
    **load_kwargs
) -> int:
    """Generic streaming download and save function.

    Args:
        dataset_name: Display name for logging
        hf_dataset: HuggingFace dataset identifier
        target_count: Number of examples to download
        output_path: Where to save Parquet shards
        max_tokens: Maximum sequence length
        split: Dataset split to use
        **load_kwargs: Additional arguments for load_dataset

    Returns:
        Number of examples saved
    """
    print(f"\n📦 Downloading {dataset_name} (target: {target_count:,} examples)...")

    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load dataset with streaming
        dataset_stream = load_dataset(hf_dataset, split=split, streaming=True, **load_kwargs)

        # Process and save in batches
        batch_size = 10000
        batch = []
        total_saved = 0
        shard_num = 0

        for i, example in enumerate(tqdm(dataset_stream, total=target_count, desc=f"  {dataset_name}")):
            if total_saved >= target_count:
                break

            try:
                # Standardize format
                formatted = standardize_format(example)

                # Length filtering inline
                if not check_length(formatted, max_tokens):
                    continue

                # Keep in instruction/input/output format (chat conversion happens later)
                batch.append(formatted)

                # Write batch to disk when full
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    pq.write_table(table, output_path / f"data-{shard_num:05d}.parquet")
                    total_saved += len(batch)
                    batch = []
                    shard_num += 1

            except Exception:
                continue

        # Write remaining batch
        if batch:
            table = pa.Table.from_pylist(batch)
            pq.write_table(table, output_path / f"data-{shard_num:05d}.parquet")
            total_saved += len(batch)

        print(f"  ✓ Saved {total_saved:,} examples")
        return total_saved

    except Exception as e:
        print(f"  ✗ Error downloading {dataset_name}: {e}")
        return 0


def download_flan_collection(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download FLAN Collection with streaming - save directly to disk.

    FLAN has 378M rows (~300GB), so we use streaming and save incrementally.
    """
    return streaming_download_and_save(
        "FLAN Collection",
        "Open-Orca/FLAN",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_metamathqa(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample MetaMathQA using streaming."""
    return streaming_download_and_save(
        "MetaMathQA",
        "meta-math/MetaMathQA",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_orca_math(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample Orca-Math using streaming."""
    return streaming_download_and_save(
        "Orca-Math",
        "microsoft/orca-math-word-problems-200k",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_evol_instruct_code(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample Evol-Instruct-Code using streaming."""
    return streaming_download_and_save(
        "Evol-Instruct-Code",
        "nickrosh/Evol-Instruct-Code-80k-v1",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_code_alpaca(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample Code Alpaca using streaming."""
    return streaming_download_and_save(
        "Code Alpaca",
        "sahil2801/CodeAlpaca-20k",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_slim_orca(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample SlimOrca-Dedup using streaming.

    SlimOrca is a cleaned and deduplicated version of OpenOrca,
    all generated by GPT-4. Uses conversation format with system/human/gpt turns.
    """
    return streaming_download_and_save(
        "SlimOrca-Dedup",
        "Open-Orca/SlimOrca-Dedup",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


def download_sharegpt_filtered(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample ShareGPT (filtered for quality) using streaming."""
    print(f"\n💬 Downloading ShareGPT (target: {target_count:,} examples)...")
    print(f"  Using streaming mode with quality filtering")

    try:
        output_path.mkdir(parents=True, exist_ok=True)

        dataset_stream = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True, token=False)

        def is_quality(example):
            """Quality filter for ShareGPT conversations."""
            if "conversations" not in example:
                return False
            convs = example["conversations"]
            if len(convs) < 2:
                return False
            if convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
                return False

            human_text = convs[0].get("value", "")
            gpt_text = convs[1].get("value", "")

            # Length check
            if len(human_text) < 50 or len(human_text) > 2000:
                return False
            if len(gpt_text) < 50 or len(gpt_text) > 2000:
                return False

            # No code execution patterns
            code_patterns = ["```python", "```java", "```cpp", "```javascript"]
            if any(p in gpt_text.lower() for p in code_patterns):
                return False

            return True

        # Process and save in batches
        batch_size = 10000
        batch = []
        total_saved = 0
        shard_num = 0
        checked_count = 0

        for example in tqdm(dataset_stream, desc="  ShareGPT"):
            checked_count += 1

            if total_saved >= target_count:
                break

            # Apply quality filter
            if not is_quality(example):
                continue

            try:
                # Standardize format
                formatted = standardize_format(example)

                # Length filtering inline
                if not check_length(formatted, max_tokens):
                    continue

                # Keep in instruction/input/output format (chat conversion happens later)
                batch.append(formatted)

                # Write batch to disk when full
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    pq.write_table(table, output_path / f"data-{shard_num:05d}.parquet")
                    total_saved += len(batch)
                    batch = []
                    shard_num += 1

            except Exception:
                continue

            # Safety limit
            if checked_count >= 100000:
                print(f"  Reached check limit of 100k examples")
                break

        # Write remaining batch
        if batch:
            table = pa.Table.from_pylist(batch)
            pq.write_table(table, output_path / f"data-{shard_num:05d}.parquet")
            total_saved += len(batch)

        print(f"  ✓ Saved {total_saved:,} quality examples (checked {checked_count:,} total)")
        return total_saved

    except Exception as e:
        print(f"  ✗ Error downloading ShareGPT: {e}")
        return 0


def download_wizardlm_evol_v2(target_count: int, output_path: Path, max_tokens: int = 2048) -> int:
    """Download and sample WizardLM Evol-Instruct V2 using streaming."""
    return streaming_download_and_save(
        "WizardLM Evol V2",
        "WizardLM/WizardLM_evol_instruct_V2_196k",
        target_count,
        output_path,
        max_tokens,
        token=False
    )


# =============================================================================
# Data Augmentation Functions
# =============================================================================

def add_cot_variant(example: Dict) -> Dict:
    """Add Chain-of-Thought step-by-step reasoning to math/reasoning examples.

    Identifies if the output already has CoT (contains "step", "first", "then", etc.),
    if not, wraps the output with reasoning structure.
    """
    # Safety check: ensure required fields exist
    if "instruction" not in example or "output" not in example:
        return example

    output = example["output"]
    instruction = example["instruction"].lower()

    # Check if this is a math/reasoning task
    is_math = any(kw in instruction for kw in [
        "solve", "calculate", "compute", "math", "equation", "problem",
        "how many", "what is", "find"
    ])

    # Check if already has CoT
    has_cot = any(kw in output.lower() for kw in [
        "step 1", "step 2", "first,", "then,", "finally,", "therefore"
    ])

    if is_math and not has_cot and len(output) > 20:
        # Wrap output with CoT structure
        example["output"] = f"Let's solve this step by step:\n\n{output}\n\nTherefore, the answer is: {output.split()[-1]}"

    return example


def create_fewshot_variant(example: Dict, example_pool: List[Dict], num_shots: int = 2) -> Dict:
    """Create few-shot version by adding 1-3 example demonstrations.

    Args:
        example: The target example to augment
        example_pool: Pool of examples to draw demonstrations from
        num_shots: Number of demonstration examples (1-3)
    """
    # Safety check: ensure required fields exist
    if "instruction" not in example or "output" not in example:
        return example

    if not example_pool or len(example_pool) < num_shots:
        return example

    # Sample demonstration examples (different from target)
    demos = random.sample(example_pool, num_shots)

    # Build few-shot prompt
    fewshot_instruction = f"{example['instruction']}\n\nHere are some examples:\n\n"

    for i, demo in enumerate(demos, 1):
        # Safety check: ensure demo has required fields
        if "output" not in demo:
            continue

        demo_input = demo.get("input", "")
        if demo_input:
            fewshot_instruction += f"Example {i}:\nInput: {demo_input}\n"
        fewshot_instruction += f"Output: {demo['output']}\n\n"

    fewshot_instruction += "Now solve:\n"

    return {
        "instruction": fewshot_instruction,
        "input": example.get("input", ""),
        "output": example["output"]
    }


def create_input_inversion(example: Dict) -> Dict:
    """Create input inversion: swap input and output for robustness.

    Example:
        Original: "Translate to French: Hello" → "Bonjour"
        Inverted: "What English text translates to: Bonjour" → "Hello"
    """
    # Safety check: ensure required fields exist
    if "instruction" not in example or "output" not in example:
        return example

    # Skip if both input and output are empty
    if not example.get("input", "") and not example.get("output", ""):
        return example

    # Simple inversion: ask model to generate the input given the output
    inverted_instruction = f"Given this output, what was the likely instruction or input?\n\nOutput: {example['output']}"
    inverted_output = example["instruction"]
    if example.get("input"):
        inverted_output += f"\nInput: {example['input']}"

    return {
        "instruction": inverted_instruction,
        "input": "",
        "output": inverted_output
    }


# =============================================================================
# Quality Control
# =============================================================================

def check_length(example: Dict, max_tokens: int) -> bool:
    """Quick length check using whitespace tokenization approximation.

    Uses ~1.3 tokens per word heuristic to avoid loading tokenizer.
    This is fast enough for streaming and close enough for filtering.
    """
    text = f"{example['instruction']} {example.get('input', '')} {example['output']}"
    word_count = len(text.split())
    estimated_tokens = word_count * 1.3
    return estimated_tokens <= max_tokens


def compute_example_hash(example: Dict) -> str:
    """Compute hash of example for deduplication."""
    text = f"{example['instruction']}\n{example['input']}\n{example['output']}"
    return hashlib.md5(text.encode()).hexdigest()


def deduplicate_dataset(dataset: Dataset) -> Dataset:
    """Remove exact duplicates based on content hash."""
    print("  Deduplicating...")

    seen_hashes = set()
    unique_indices = []

    for idx, example in enumerate(tqdm(dataset, desc="  Computing hashes")):
        h = compute_example_hash(example)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)

    dataset = dataset.select(unique_indices)
    removed = len(seen_hashes) - len(unique_indices)
    print(f"  ✓ Removed {removed:,} duplicates, kept {len(dataset):,} unique")

    return dataset


def filter_by_length(dataset: Dataset, max_tokens: int = 2048, tokenizer=None) -> Dataset:
    """Filter examples that are too long."""
    print(f"  Filtering by length (max {max_tokens} tokens)...")

    def is_valid_length(example):
        # Rough estimate: 4 chars per token
        text = f"{example['instruction']} {example['input']} {example['output']}"
        approx_tokens = len(text) // 4
        return approx_tokens <= max_tokens

    original_len = len(dataset)
    dataset = dataset.filter(is_valid_length, desc="  Checking lengths")
    removed = original_len - len(dataset)

    print(f"  ✓ Removed {removed:,} too-long examples, kept {len(dataset):,}")

    return dataset


def verify_fields(dataset: Dataset) -> Dataset:
    """Verify all required fields are non-empty."""
    print("  Verifying fields...")

    def is_valid(example):
        return (
            example["instruction"] and len(example["instruction"].strip()) > 0 and
            example["output"] and len(example["output"].strip()) > 0
        )

    original_len = len(dataset)
    dataset = dataset.filter(is_valid, desc="  Checking fields")
    removed = original_len - len(dataset)

    print(f"  ✓ Removed {removed:,} invalid examples, kept {len(dataset):,}")

    return dataset


def fix_chat_template_for_masking(tokenizer, apply_chat_template: bool = True):
    """Fix chat template to support assistant token masking.

    Llama-2 chat templates don't have {% generation %} tags by default,
    which are required for return_assistant_tokens_mask to work.
    This function either adds those tags to existing templates or creates
    a minimal template that only adds generation tags without other formatting.

    Args:
        tokenizer: HuggingFace tokenizer with chat_template attribute
        apply_chat_template: If True, use full chat template formatting.
                           If False, use minimal template (only generation tags, no formatting).
    """
    if not apply_chat_template:
        # Use minimal template: only add generation tags, preserve exact text
        minimal_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = minimal_template
        print("  ✓ Set minimal chat template (only generation tags for masking)")
        return

    # Full chat template mode: add generation tags to existing template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Check if already has generation tags
        if '{% generation %}' in tokenizer.chat_template:
            return  # Already fixed

        # Fix Llama-2 style template by adding generation tags around assistant content
        # Original: {% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}
        # Fixed:    {% elif message['role'] == 'assistant' %}{% generation %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endgeneration %}

        original_template = tokenizer.chat_template

        # Replace assistant response section with generation-tagged version
        if "message['role'] == 'assistant'" in original_template:
            # Pattern: finds the assistant block and wraps the output in generation tags
            import re
            pattern = r"(\{%\s*elif\s+message\['role'\]\s*==\s*'assistant'\s*%\})(.*?)(\{%\s*endif\s*%\})"

            def add_generation_tags(match):
                prefix = match.group(1)  # {% elif message['role'] == 'assistant' %}
                content = match.group(2)  # {{ ' '  + content.strip() + ' ' + eos_token }}
                suffix = match.group(3)  # {% endif %}
                return f"{prefix}{{% generation %}}{content}{{% endgeneration %}}{suffix}"

            fixed_template = re.sub(pattern, add_generation_tags, original_template, flags=re.DOTALL)

            # Apply the fix
            tokenizer.chat_template = fixed_template
            print("  ✓ Fixed chat template to support assistant token masking")
        else:
            print("  ⚠️  Warning: Could not automatically fix chat template (unknown format)")


def save_verification_samples(
    datasets: Dict[str, Dataset],
    output_path: Path,
    tokenizer_path: str,
    apply_chat_template: bool = True
):
    """Save 3 samples per dataset showing chat template format and label masking.

    This helps verify that:
    1. Chat template is applied correctly
    2. User tokens are masked (not trained)
    3. Assistant tokens are trained
    4. Token IDs and masking align properly

    Args:
        datasets: Dict mapping dataset name to Dataset object
        output_path: Path where verification_samples.txt will be saved
        tokenizer_path: Path to tokenizer directory
        apply_chat_template: Whether to apply full chat template or minimal template
    """
    print(f"\n📋 Generating verification samples...")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ⚠️  Warning: transformers not available, skipping verification samples")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Fix chat template to support assistant token masking
        fix_chat_template_for_masking(tokenizer, apply_chat_template)

    except Exception as e:
        print(f"  ⚠️  Warning: Could not load tokenizer from {tokenizer_path}: {e}")
        print("  Skipping verification samples")
        return

    verification_file = output_path / "verification_samples.txt"
    print(f"  Saving to: {verification_file}")

    with open(verification_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FLAN MIXTURE VERIFICATION SAMPLES\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nTokenizer: {tokenizer_path}\n")
        f.write(f"Chat Template: {tokenizer.chat_template[:100] if tokenizer.chat_template else 'None'}...\n")
        f.write("\n")

        for dataset_name, dataset in datasets.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'='*80}\n\n")

            num_samples = min(3, len(dataset))
            for i in range(num_samples):
                example = dataset[i]
                f.write(f"--- Sample {i+1}/{num_samples} ---\n\n")

                # Show messages format
                f.write("Messages Format:\n")
                if "messages" in example:
                    for msg in example["messages"]:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        content_preview = content[:100] + "..." if len(content) > 100 else content
                        f.write(f"  [{role}]: {content_preview}\n")
                else:
                    f.write("  (No messages field - may not be converted yet)\n")
                f.write("\n")

                # Apply chat template (non-tokenized)
                try:
                    if "messages" in example:
                        chat_str = tokenizer.apply_chat_template(
                            example["messages"],
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        f.write(f"Chat Template Applied (raw text):\n")
                        f.write(f"  {chat_str}\n\n")

                        # Tokenized with masking
                        tokenized = tokenizer.apply_chat_template(
                            example["messages"],
                            tokenize=True,
                            return_assistant_tokens_mask=True,
                            return_dict=True,
                            add_generation_prompt=False
                        )

                        input_ids = tokenized["input_ids"]
                        assistant_masks = tokenized["assistant_masks"]

                        f.write(f"Tokenized ({len(input_ids)} tokens total):\n")
                        f.write(f"  Input IDs (first 20): {input_ids[:20]}...\n\n")

                        # Show complete token-by-token breakdown (ALL tokens)
                        f.write(f"Token-by-Token Breakdown (all {len(input_ids)} tokens):\n")
                        f.write(f"{'Idx':<5} {'Token ID':<10} {'Token Text':<30} {'Status':<10}\n")
                        f.write("-" * 60 + "\n")

                        for idx in range(len(input_ids)):
                            token_id = input_ids[idx]
                            is_assistant = assistant_masks[idx]
                            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                            token_str = token_str.replace("\n", "\\n").replace("\t", "\\t")
                            status = "TRAIN" if is_assistant else "MASK"

                            f.write(f"{idx:<5} {token_id:<10} {token_str[:28]:<30} {status:<10}\n")

                        f.write("\n")

                        # Summary statistics
                        num_trained = sum(assistant_masks)
                        num_masked = len(assistant_masks) - num_trained
                        pct_trained = 100 * num_trained / len(input_ids) if len(input_ids) > 0 else 0
                        pct_masked = 100 * num_masked / len(input_ids) if len(input_ids) > 0 else 0

                        f.write("Summary:\n")
                        f.write(f"  Total tokens:   {len(input_ids)}\n")
                        f.write(f"  Trained tokens: {num_trained} ({pct_trained:.1f}%)\n")
                        f.write(f"  Masked tokens:  {num_masked} ({pct_masked:.1f}%)\n")

                        # Show what the model actually trains on
                        trained_ids = [tid for tid, mask in zip(input_ids, assistant_masks) if mask]
                        if trained_ids:
                            trained_text = tokenizer.decode(trained_ids, skip_special_tokens=False)
                            f.write(f"\nModel trains on (assistant response only):\n")
                            f.write(f"  {trained_text}\n")

                    else:
                        f.write("  (Skipping tokenization - no messages field)\n")

                except Exception as e:
                    f.write(f"  ⚠️  Error applying chat template: {e}\n")

                f.write("\n" + "-" * 80 + "\n\n")

    print(f"  ✓ Saved {len(datasets)} dataset samples to {verification_file}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    # Args already parsed at module level
    global args

    # Calculate dataset counts if not specified
    total = args.total_examples
    counts = {
        "flan": args.flan_count or int(total * 0.40),
        "metamath": args.metamath_count or int(total * 0.15),
        "orca_math": args.orca_math_count or int(total * 0.10),
        "evol_code": args.evol_code_count or int(total * 0.125),
        "code_alpaca": args.code_alpaca_count or int(total * 0.025),
        "openorca": args.openorca_count or int(total * 0.10),
        "sharegpt": args.sharegpt_count or int(total * 0.05),
        "wizardlm": args.wizardlm_count or int(total * 0.05),
    }

    # Display plan
    print("=" * 80)
    print("FLAN-Focused Balanced Mixture Dataset Preparation")
    print("=" * 80)
    print(f"\nTarget: {total:,} total examples")
    print(f"Output directory: {args.output_dir}")
    print(f"\nDataset composition:")
    print(f"  - FLAN Collection:        {counts['flan']:7,}  (40.0%)")
    print(f"  - MetaMathQA:             {counts['metamath']:7,}  (15.0%)")
    print(f"  - Orca-Math:              {counts['orca_math']:7,}  (10.0%)")
    print(f"  - Evol-Instruct-Code:     {counts['evol_code']:7,}  (12.5%)")
    print(f"  - Code Alpaca:            {counts['code_alpaca']:7,}  ( 2.5%)")
    print(f"  - SlimOrca-Dedup:         {counts['openorca']:7,}  (10.0%)")
    print(f"  - ShareGPT (filtered):    {counts['sharegpt']:7,}  ( 5.0%)")
    print(f"  - WizardLM Evol V2:       {counts['wizardlm']:7,}  ( 5.0%)")
    print(f"  {'─' * 40}")
    print(f"  Total:                    {sum(counts.values()):7,}  (100.0%)")

    print(f"\nAugmentation:")
    print(f"  - CoT variants:           {'✓ Enabled' if args.add_cot_variants else '✗ Disabled'}")
    print(f"  - Few-shot variants:      {'✓ Enabled' if args.add_fewshot_variants else '✗ Disabled'} ({args.fewshot_ratio:.0%})")
    print(f"  - Input inversions:       {'✓ Enabled' if args.add_input_inversions else '✗ Disabled'} ({args.inversion_ratio:.0%})")

    print(f"\nQuality control:")
    print(f"  - Deduplication:          ✓ Enabled")
    print(f"  - Length filtering:       ✓ Enabled (max 2048 tokens)")
    print(f"  - Field verification:     ✓ Enabled")

    print(f"\nTrain/val split: {1-args.val_split:.0%} / {args.val_split:.0%}")

    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without downloading.")
        return

    print("\n" + "=" * 80)
    print("Starting dataset preparation...")
    print("=" * 80)

    # Create temporary directory for individual datasets
    temp_dir = Path(args.output_dir) / "temp_datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download datasets - now saves to disk incrementally
    dataset_counts = {}
    max_tokens = 2048

    dataset_counts["flan"] = download_flan_collection(counts["flan"], temp_dir / "flan", max_tokens)
    dataset_counts["metamath"] = download_metamathqa(counts["metamath"], temp_dir / "metamath", max_tokens)
    dataset_counts["orca_math"] = download_orca_math(counts["orca_math"], temp_dir / "orca_math", max_tokens)
    dataset_counts["evol_code"] = download_evol_instruct_code(counts["evol_code"], temp_dir / "evol_code", max_tokens)
    dataset_counts["code_alpaca"] = download_code_alpaca(counts["code_alpaca"], temp_dir / "code_alpaca", max_tokens)
    dataset_counts["slimorca"] = download_slim_orca(counts["openorca"], temp_dir / "slimorca", max_tokens)
    dataset_counts["sharegpt"] = download_sharegpt_filtered(counts["sharegpt"], temp_dir / "sharegpt", max_tokens)
    dataset_counts["wizardlm"] = download_wizardlm_evol_v2(counts["wizardlm"], temp_dir / "wizardlm", max_tokens)

    # Filter out failed downloads
    successful_datasets = {k: v for k, v in dataset_counts.items() if v > 0}

    if not successful_datasets:
        print("\n✗ Error: No datasets successfully downloaded. Exiting.")
        return

    print(f"\n✓ Successfully downloaded {len(successful_datasets)} / 8 datasets")
    print(f"  Total examples saved: {sum(successful_datasets.values()):,}")

    # Combine all datasets by loading from disk
    print("\n🔗 Combining datasets...")
    dataset_list = []
    for name in successful_datasets.keys():
        ds = load_dataset("parquet", data_dir=str(temp_dir / name), split="train")
        dataset_list.append(ds)
        # Free memory after loading
        gc.collect()

    combined = concatenate_datasets(dataset_list)
    print(f"  ✓ Combined: {len(combined):,} examples")

    # Free memory
    del dataset_list
    gc.collect()

    # Apply augmentations
    if args.add_cot_variants:
        print("\n🧠 Adding Chain-of-Thought variants...")
        combined = combined.map(
            add_cot_variant,
            desc="  Adding CoT reasoning",
            num_proc=args.num_workers
        )

    if args.add_fewshot_variants:
        print(f"\n🎯 Adding few-shot variants ({args.fewshot_ratio:.0%} of examples)...")
        # Sample examples to convert
        fewshot_count = int(len(combined) * args.fewshot_ratio)
        fewshot_indices = random.sample(range(len(combined)), fewshot_count)

        # Get example pool for demonstrations
        example_pool = [combined[i] for i in range(min(1000, len(combined)))]

        for idx in tqdm(fewshot_indices, desc="  Creating few-shot examples"):
            combined = combined.map(
                lambda x, i=idx: create_fewshot_variant(x, example_pool) if i == idx else x,
                with_indices=True
            )

        print(f"  ✓ Added {fewshot_count:,} few-shot variants")

    if args.add_input_inversions:
        print(f"\n🔄 Adding input inversions ({args.inversion_ratio:.0%} of examples)...")
        inversion_count = int(len(combined) * args.inversion_ratio)
        inversion_indices = set(random.sample(range(len(combined)), inversion_count))

        def maybe_invert(example, idx):
            if idx in inversion_indices:
                return create_input_inversion(example)
            return example

        combined = combined.map(
            maybe_invert,
            with_indices=True,
            desc="  Creating inversions",
            num_proc=args.num_workers
        )

        print(f"  ✓ Added {inversion_count:,} inverted examples")

    # Quality control
    print("\n🔍 Applying quality control...")

    # Deduplication (skip if dataset too large for memory)
    if len(combined) < 5_000_000:
        combined = deduplicate_dataset(combined)
    else:
        print("  ⚠️  Skipping deduplication (dataset > 5M examples)")
        print("     Length filtering already applied during download")

    # Note: Length filtering already done during streaming download
    combined = verify_fields(combined)

    # Convert to chat format (after all augmentations)
    print("\n💬 Converting to chat format...")
    combined = combined.map(
        convert_to_chat_format,
        desc="  Converting to messages format",
        num_proc=args.num_workers
    )

    print(f"\n✓ Final dataset: {len(combined):,} examples")

    # Shuffle
    print("\n🔀 Shuffling...")
    combined = combined.shuffle(seed=RANDOM_SEED)

    # Train/val split
    print(f"\n✂️  Creating train/val split ({1-args.val_split:.0%}/{args.val_split:.0%})...")
    split_idx = int(len(combined) * (1 - args.val_split))
    train_dataset = combined.select(range(split_idx))
    val_dataset = combined.select(range(split_idx, len(combined)))

    print(f"  ✓ Train: {len(train_dataset):,} examples")
    print(f"  ✓ Val:   {len(val_dataset):,} examples")

    # Generate verification samples (before saving, to validate format)
    print("\n📋 Generating verification samples...")
    print("  This helps verify chat template application and label masking")

    # Sample 3 examples from combined dataset for verification
    # Note: These samples now have chat format applied after augmentations
    verification_datasets = {
        "FLAN_Collection": combined.select(range(0, min(3, len(combined)))),
        "MetaMathQA": combined.select(range(3, min(6, len(combined)))),
        "Orca_Math": combined.select(range(6, min(9, len(combined)))),
        "Evol_Instruct_Code": combined.select(range(9, min(12, len(combined)))),
        "Code_Alpaca": combined.select(range(12, min(15, len(combined)))),
        "SlimOrca": combined.select(range(15, min(18, len(combined)))),
        "ShareGPT": combined.select(range(18, min(21, len(combined)))),
        "WizardLM_V2": combined.select(range(21, min(24, len(combined)))),
    }

    # Save datasets
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate verification samples with tokenizer path
    save_verification_samples(verification_datasets, output_path, args.tokenizer_path, args.apply_chat_template)

    print(f"\n💾 Saving datasets to {output_path}...")

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    dataset_dict.save_to_disk(str(output_path))

    # Clean up temporary files
    print("\n🧹 Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("  ✓ Temporary dataset files removed")
    except Exception as e:
        print(f"  ⚠️  Could not remove temp files: {e}")

    # Save metadata
    metadata = {
        "total_examples": len(combined),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "dataset_counts": dataset_counts,  # Now using the counts dict
        "augmentations": {
            "cot_variants": args.add_cot_variants,
            "fewshot_variants": args.add_fewshot_variants,
            "fewshot_ratio": args.fewshot_ratio,
            "input_inversions": args.add_input_inversions,
            "inversion_ratio": args.inversion_ratio,
        },
        "quality_control": {
            "deduplication": len(combined) < 5_000_000,
            "max_tokens": 2048,
            "field_verification": True,
            "length_filtering": "applied_during_download",
            "chat_format_conversion": "applied_after_augmentations",
            "apply_chat_template": args.apply_chat_template,
        },
        "random_seed": RANDOM_SEED,
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ Dataset preparation complete!")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")
    print(f"  - Train dataset:      {len(train_dataset):,} examples")
    print(f"  - Validation dataset: {len(val_dataset):,} examples")
    print(f"  - Metadata:           metadata.json")
    print(f"  - Verification:       verification_samples.txt")
    print(f"\nTo use in training, update your config YAML:")
    print(f"  data_config:")
    print(f"    train_data:")
    print(f"      - type: hfds")
    print(f"        prefix: flan-mixture-train")
    print(f"        data_dir: {output_path}/train")
    print(f"    val_data:")
    print(f"      - type: hfds")
    print(f"        prefix: flan-mixture-val")
    print(f"        data_dir: {output_path}/validation")
    print()


if __name__ == "__main__":
    main()
