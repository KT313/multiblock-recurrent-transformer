#!/usr/bin/env python3
"""
Download Pretraining Datasets for Depth-Recurrent Language Models

This script downloads raw pretraining datasets for 300M-1B parameter models
using a two-stage training strategy (Stage 1: broad pretraining, Stage 2: domain upsampling).

Target: 300M model with 30B tokens (24B Stage 1 + 6B Stage 2)

Stage 1 Dataset Composition (24B tokens):
- 65% FineWeb-Edu: 15.6B tokens (high-quality web)
- 9% Wikipedia: 2.16B tokens (encyclopedia, oversampled 2x)
- 6% Books: 1.44B tokens (Project Gutenberg)
- 12% The Stack v2: 2.88B tokens (code)
- 3% peS2o: 0.72B tokens (academic papers)
- 2% arXiv: 0.48B tokens (STEM papers)
- 3% OpenWebMath: 0.72B tokens (math)

Stage 2 Dataset Composition (6B tokens - domain upsampling):
- 35% FineWeb-Edu (≥3 score): 2.1B tokens (ultra-filtered web)
- 28% The Stack v2 (≥2 stars): 1.68B tokens (quality code)
- 22% Math mixture: 1.32B tokens
  - 40% OpenWebMath: 0.528B
  - 30% TinyGSM-MIND: 0.396B
  - 20% Algebraic Stack: 0.264B
  - 10% GSM8K train: 0.132B
- 15% peS2o + arXiv: 0.9B tokens (STEM)

Features:
- Streaming downloads (memory efficient)
- Intelligent sampling to download only what's needed (with 15% buffer)
- Basic quality filtering during download (language, length)
- Resumable downloads with metadata
- Saves to parquet shards

Usage:
    # Dry run to see plan
    python download_pretraining_data.py --dry_run --model_size 300M

    # Download all datasets
    python download_pretraining_data.py \
        --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
        --cache_dir /path/to/fast_storage/.cache \
        --model_size 300M \
        --stages both

    # Download only Stage 1
    python download_pretraining_data.py \
        --output_dir /path/to/output \
        --stages stage1 \
        --model_size 300M

Author: Claude Code
Date: 2025-01-25
"""

import os
import sys
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading


# Parse args FIRST before importing datasets (to set cache dir)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download pretraining datasets for depth-recurrent LMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for raw datasets (will create raw_datasets/ subdirectory)"
    )

    # Model size determines token budgets
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["300M", "1B"],
        default="300M",
        help="Model size (determines total token budget): 300M=30B tokens, 1B=100B tokens"
    )

    # Which stages to download
    parser.add_argument(
        "--stages",
        type=str,
        choices=["stage1", "stage2", "both"],
        default="both",
        help="Which training stages to download datasets for"
    )

    # Cache directory argument (IMPORTANT for permission issues)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory"
    )

    # Token buffer for preprocessing losses
    parser.add_argument(
        "--token_buffer",
        type=float,
        default=0.15,
        help="Extra tokens to download as buffer for preprocessing losses (default: 0.15 = 15%%)"
    )

    # Download options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for filtering (default: 8)"
    )

    parser.add_argument(
        "--parallel_downloads",
        type=int,
        default=3,
        help="Number of datasets to download in parallel (default: 3)"
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Number of examples per parquet shard (default: 10,000)"
    )

    parser.add_argument(
        "--keep_raw",
        action="store_true",
        default=True,
        help="Keep raw unfiltered data in raw_unfiltered/ directory (default: True)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show download plan without actually downloading"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted download (skip already downloaded datasets)"
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
    from datasets import load_dataset, Dataset
    from tqdm.auto import tqdm
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install datasets tqdm numpy pyarrow")
    raise

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# Token Budget Calculation
# =============================================================================

def calculate_token_budgets(model_size: str, buffer: float = 0.15) -> Dict[str, Dict[str, int]]:
    """Calculate token budgets for each dataset based on model size.

    Args:
        model_size: "300M" (30B tokens) or "1B" (100B tokens)
        buffer: Extra tokens to download as buffer for preprocessing losses

    Returns:
        Dict with 'stage1' and 'stage2' keys, each containing dataset -> token count
    """
    # Total token budgets by model size
    TOTAL_TOKENS = {
        "300M": 30_000_000_000,  # 30B tokens
        "1B": 100_000_000_000,   # 100B tokens
    }

    total_tokens = TOTAL_TOKENS[model_size]
    stage1_tokens = int(total_tokens * 0.80)  # 80% for Stage 1
    stage2_tokens = int(total_tokens * 0.20)  # 20% for Stage 2

    # Apply buffer (we download extra to account for preprocessing losses)
    stage1_tokens = int(stage1_tokens * (1 + buffer))
    stage2_tokens = int(stage2_tokens * (1 + buffer))

    # Stage 1 proportions
    stage1_budgets = {
        "fineweb_edu": int(stage1_tokens * 0.65),
        "wikipedia": int(stage1_tokens * 0.09),
        "books_gutenberg": int(stage1_tokens * 0.06),
        "stack_v2": int(stage1_tokens * 0.12),
        "peso": int(stage1_tokens * 0.03),
        "arxiv": int(stage1_tokens * 0.02),
        "openwebmath": int(stage1_tokens * 0.03),
    }

    # Stage 2 proportions (with upsampling)
    stage2_budgets = {
        "fineweb_edu_filtered": int(stage2_tokens * 0.35),
        "stack_v2_filtered": int(stage2_tokens * 0.28),
        "openwebmath": int(stage2_tokens * 0.22 * 0.40),  # 40% of math mixture
        "tinygsm": int(stage2_tokens * 0.22 * 0.30),      # 30% of math mixture
        "algebraic_stack": int(stage2_tokens * 0.22 * 0.20),  # 20% of math mixture
        "gsm8k_train": int(stage2_tokens * 0.22 * 0.10),  # 10% of math mixture
        "peso": int(stage2_tokens * 0.15 * 0.50),        # 50% of STEM
        "arxiv": int(stage2_tokens * 0.15 * 0.50),       # 50% of STEM
    }

    return {
        "stage1": stage1_budgets,
        "stage2": stage2_budgets
    }


def tokens_to_gigabytes(tokens: int) -> float:
    """Estimate disk space in GB from token count.

    Heuristic: ~4 tokens per byte → ~4B tokens per GB
    """
    return tokens / 4_000_000_000


# =============================================================================
# Shared Dataset Configuration (Optimization to avoid duplicate downloads)
# =============================================================================

# Datasets that appear in both Stage 1 and Stage 2
# These will be downloaded ONCE and filtered with different criteria
SHARED_DATASETS_CONFIG = {
    'fineweb_edu': {
        'hf_dataset': 'HuggingFaceFW/fineweb-edu',
        'text_field': 'text',
        'split': 'train',
        'load_kwargs': {'token': False},
        'stage1': {
            'filter_type': 'basic',  # Basic quality filters only
            'filter_fn': None,       # No custom filter
            'pass_rate': 0.70,       # ~70% pass basic quality
        },
        'stage2': {
            'filter_type': 'score_gte_3',  # Score ≥3 filter
            'filter_fn': lambda ex: ex.get('score', 0) >= 3,
            'pass_rate': 0.10,       # ~10% have score ≥3
        },
    },
    'peso': {
        'hf_dataset': 'allenai/peS2o',
        'text_field': 'text',
        'split': 'train',
        'load_kwargs': {'token': False, 'trust_remote_code': True},  # Custom code
        'stage1': {
            'filter_type': 'basic',
            'filter_fn': None,
            'pass_rate': 0.70,
        },
        'stage2': {
            'filter_type': 'basic',  # Same filter for both stages
            'filter_fn': None,
            'pass_rate': 0.70,
        },
    },
    'arxiv': {
        'hf_dataset': 'common-pile/arxiv_papers_filtered',
        'text_field': 'text',
        'split': 'train',
        'load_kwargs': {'token': False},  # Simple loading, no special config needed
        'stage1': {
            'filter_type': 'basic',
            'filter_fn': None,
            'pass_rate': 0.70,
        },
        'stage2': {
            'filter_type': 'basic',
            'filter_fn': None,
            'pass_rate': 0.70,
        },
    },
    'openwebmath': {
        'hf_dataset': 'open-web-math/open-web-math',
        'text_field': 'text',
        'split': 'train',
        'load_kwargs': {'token': False},
        'stage1': {
            'filter_type': 'basic',
            'filter_fn': None,
            'pass_rate': 0.70,
        },
        'stage2': {
            'filter_type': 'basic',  # Same filter for both stages
            'filter_fn': None,
            'pass_rate': 0.70,
        },
    },
}


def calculate_shared_dataset_quotas(stage1_budgets: Dict, stage2_budgets: Dict) -> Dict:
    """Calculate combined download quotas for shared datasets.

    For datasets appearing in both stages, we download once with enough
    raw data to satisfy both stage requirements after different filtering.

    Args:
        stage1_budgets: Token quotas for Stage 1 datasets
        stage2_budgets: Token quotas for Stage 2 datasets

    Returns:
        Dict mapping dataset name to combined raw download quota
    """
    shared_quotas = {}

    for dataset_name, config in SHARED_DATASETS_CONFIG.items():
        # Get token requirements for each stage
        stage1_key = dataset_name
        stage2_key = dataset_name if dataset_name != 'fineweb_edu' else f"{dataset_name}_filtered"

        stage1_need = stage1_budgets.get(stage1_key, 0)
        stage2_need = stage2_budgets.get(stage2_key, 0)

        # Calculate raw download needed for each stage
        stage1_pass_rate = config['stage1']['pass_rate']
        stage2_pass_rate = config['stage2']['pass_rate']

        raw_for_stage1 = int(stage1_need / stage1_pass_rate) if stage1_pass_rate > 0 else 0
        raw_for_stage2 = int(stage2_need / stage2_pass_rate) if stage2_pass_rate > 0 else 0

        # Take maximum (download enough for both)
        combined_quota = max(raw_for_stage1, raw_for_stage2)

        shared_quotas[dataset_name] = {
            'raw_quota': combined_quota,
            'stage1_target': stage1_need,
            'stage2_target': stage2_need,
            'stage1_pass_rate': stage1_pass_rate,
            'stage2_pass_rate': stage2_pass_rate,
        }

    return shared_quotas


def calculate_stack_v2_language_budgets(stack_v2_total_tokens: int) -> Dict[str, int]:
    """Calculate per-language token budgets for Stack v2 dedup.

    Distributes the total Stack v2 budget across top programming languages
    based on industry usage, versatility, and LLM utility.

    Args:
        stack_v2_total_tokens: Total tokens allocated for Stack v2

    Returns:
        Dict mapping language name (capitalized) to token budget

    Distribution:
        Python (30%), JavaScript (20%), TypeScript (10%), Java (10%),
        C++ (8%), Go (7%), Rust (5%), Shell (4%), SQL (3%), HTML (3%)
    """
    # Language distribution percentages
    distribution = {
        'Python': 0.30,
        'JavaScript': 0.20,
        'TypeScript': 0.10,
        'Java': 0.10,
        'C++': 0.08,
        'Go': 0.07,
        'Rust': 0.05,
        'Shell': 0.04,
        'SQL': 0.03,
        'HTML': 0.03,
    }

    # Calculate per-language budgets
    language_budgets = {}
    for lang, pct in distribution.items():
        language_budgets[lang] = int(stack_v2_total_tokens * pct)

    return language_budgets


# =============================================================================
# Basic Quality Filters (Applied During Download)
# =============================================================================

def is_english(text: str, min_alpha_ratio: float = 0.7) -> bool:
    """Quick English check: at least 70% of characters are ASCII letters/spaces.

    This is a fast heuristic to avoid loading FastText during download.
    More sophisticated language detection happens in processing script.
    """
    if not text or len(text) < 100:
        return False

    alpha_count = sum(1 for c in text if c.isalpha() or c.isspace())
    return (alpha_count / len(text)) >= min_alpha_ratio


def passes_basic_quality(text: str, min_length: int = 100, max_length: int = 100000) -> bool:
    """Basic quality checks: length, not too repetitive, has some structure."""
    if not text or len(text) < min_length or len(text) > max_length:
        return False

    # Check for extreme repetition (>50% of text is repeating 3-grams)
    words = text.split()
    if len(words) < 10:
        return False

    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    if len(trigrams) > 0:
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.5:  # More than 50% repetition
            return False

    return True


def estimate_tokens(text: str) -> int:
    """Estimate token count using characters/4 heuristic.

    This is approximate but good enough for download quotas.
    Real tokenization happens during training.
    """
    return len(text) // 4


# =============================================================================
# PHASE 1: Fast Raw Download (No Filtering)
# =============================================================================

def count_existing_progress(output_path: Path, file_prefix: str = "raw") -> Tuple[int, int, int]:
    """Count samples and tokens in existing parquet files for resume functionality.

    Args:
        output_path: Directory containing parquet shards
        file_prefix: Prefix of parquet files (e.g., "raw", "data")

    Returns:
        Tuple of (total_samples, total_tokens, next_shard_num)
        - total_samples: Number of examples already downloaded/filtered
        - total_tokens: Sum of estimated_tokens from all examples
        - next_shard_num: Next shard number to write (e.g., if raw-00011.parquet exists, returns 12)
    """
    if not output_path.exists():
        return 0, 0, 0

    # Find all parquet files with the given prefix
    parquet_files = sorted(output_path.glob(f"{file_prefix}-*.parquet"))

    if not parquet_files:
        return 0, 0, 0

    total_samples = 0
    total_tokens = 0

    try:
        # Read each parquet file and count samples + tokens
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file, columns=['estimated_tokens'])
            total_samples += len(table)
            total_tokens += sum(table['estimated_tokens'].to_pylist())

        # Extract next shard number from last file
        # Example: "raw-00011.parquet" -> extract "00011" -> 11 -> next is 12
        last_file_stem = parquet_files[-1].stem  # "raw-00011"
        last_shard_num = int(last_file_stem.split('-')[1])  # 11
        next_shard_num = last_shard_num + 1  # 12

        return total_samples, total_tokens, next_shard_num

    except Exception as e:
        print(f"    ⚠️  Warning: Could not count existing progress: {e}")
        return 0, 0, 0


def download_raw_fast(
    dataset_name: str,
    hf_dataset: str,
    target_tokens: int,
    output_path: Path,
    text_field: str = "text",
    split: str = "train",
    custom_filter_fn=None,
    **load_kwargs
) -> Tuple[int, int, str]:
    """Fast streaming download with NO quality filtering.

    Only stops when token quota is reached. This is 5-10x faster than
    downloading with filtering because:
    1. No CPU-bound quality checks per sample
    2. Larger batches can be written at once
    3. Network becomes the bottleneck, not CPU

    Filtering happens later with multiprocessing in filter_dataset_parallel().

    Args:
        dataset_name: Display name for logging
        hf_dataset: HuggingFace dataset identifier
        target_tokens: Stop after downloading this many tokens
        output_path: Where to save raw parquet shards
        text_field: Name of text field in dataset
        split: Dataset split to use
        custom_filter_fn: Optional dataset-specific filter (e.g., star count for code)
        **load_kwargs: Additional arguments for load_dataset

    Returns:
        (examples_saved, tokens_saved, output_path) tuple
    """
    # Initialize counters
    existing_samples = 0
    existing_tokens = 0
    shard_num = 0

    # Check if already exists (for resume)
    if output_path.exists() and args.resume:
        metadata_file = output_path / "metadata.json"

        # First check if fully completed
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if metadata.get("phase") == "download":
                    print(f"  ✓ {dataset_name} already completed ({metadata['tokens_saved']/1e9:.2f}B tokens)")
                    return metadata['examples_saved'], metadata['tokens_saved'], str(output_path)

        # Check for partial progress
        existing_samples, existing_tokens, shard_num = count_existing_progress(output_path, file_prefix="raw")

        if existing_tokens >= target_tokens:
            # Quota already met! Save metadata and return
            print(f"  ✓ {dataset_name} quota already met ({existing_tokens/1e9:.2f}B tokens)")
            metadata = {
                "phase": "download",
                "dataset_name": dataset_name,
                "hf_dataset": hf_dataset,
                "examples_saved": existing_samples,
                "tokens_saved": existing_tokens,
                "target_tokens": target_tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return existing_samples, existing_tokens, str(output_path)

        if existing_samples > 0:
            print(f"  🔄 Resuming {dataset_name} from {existing_samples:,} samples ({existing_tokens/1e9:.2f}B tokens)")
            print(f"     Remaining: {(target_tokens - existing_tokens)/1e9:.2f}B tokens to download")

    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load dataset with streaming
        dataset_stream = load_dataset(hf_dataset, split=split, streaming=True, **load_kwargs)

        # Skip already downloaded samples (for resume)
        if existing_samples > 0:
            print(f"     Skipping first {existing_samples:,} samples in stream...")
            dataset_stream = dataset_stream.skip(existing_samples)

        # Fast batch writing - larger batches for better I/O performance
        batch_size = args.shard_size * 2  # 2x larger batches since no filtering
        batch = []
        total_saved = existing_samples  # Start from existing count
        total_tokens = existing_tokens  # Start from existing tokens
        # shard_num already set above from existing files

        start_time = time.time()

        # Create progress bar with token tracking
        pbar = tqdm(
            dataset_stream,
            desc=f"  ⬇️  Downloading {dataset_name}",
            unit=" docs",
            miniters=1000,      # Only update display every 1000 iterations
            mininterval=5.0     # Or every 5 seconds, whichever comes first
        )
        update_interval = 1000  # Update postfix every 1000 examples

        for example in pbar:
            # Check if we've reached quota
            if total_tokens >= target_tokens:
                break

            try:
                # Extract text field
                if text_field not in example:
                    continue

                text = example[text_field]
                if not isinstance(text, str) or len(text) < 50:  # Only skip very short
                    continue

                # Apply custom filter if provided (e.g., star count for code datasets)
                if custom_filter_fn and not custom_filter_fn(example):
                    continue

                # Estimate tokens
                tokens = estimate_tokens(text)

                # Save example with NO quality filtering
                batch.append({
                    "text": text,
                    "source": dataset_name,
                    "estimated_tokens": tokens
                })
                total_tokens += tokens

                # Update progress bar with token info (every N examples to avoid slowdown)
                if total_saved % update_interval == 0 or total_tokens >= target_tokens:
                    pbar.set_postfix({
                        'tokens': f'{total_tokens/1e9:.2f}B / {target_tokens/1e9:.2f}B',
                        'progress': f'{100*total_tokens/target_tokens:.1f}%'
                    }, refresh=False)

                # Write batch to disk when full
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    pq.write_table(table, output_path / f"raw-{shard_num:05d}.parquet")
                    total_saved += len(batch)
                    batch = []
                    shard_num += 1

            except Exception:
                # Skip problematic examples
                continue

        pbar.close()

        # Write remaining batch
        if batch:
            table = pa.Table.from_pylist(batch)
            pq.write_table(table, output_path / f"raw-{shard_num:05d}.parquet")
            total_saved += len(batch)

        elapsed_time = time.time() - start_time

        # Save metadata
        metadata = {
            "phase": "download",
            "dataset_name": dataset_name,
            "hf_dataset": hf_dataset,
            "split": split,
            "examples_saved": total_saved,
            "tokens_saved": total_tokens,
            "target_tokens": target_tokens,
            "download_time_seconds": elapsed_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return total_saved, total_tokens, str(output_path)

    except Exception as e:
        print(f"  ✗ Error downloading {dataset_name}: {e}")
        return 0, 0, str(output_path)


# =============================================================================
# PHASE 2: Parallel Filtering (Multiprocessing)
# =============================================================================

def filter_dataset_parallel(
    raw_path: Path,
    filtered_path: Path,
    dataset_name: str,
    custom_filter_fn=None
) -> Tuple[int, int]:
    """Apply quality filters to raw dataset using multiprocessing.

    This runs after download completes and uses all CPU cores for filtering.
    Much faster than filtering during download because:
    1. Uses dataset.map() with num_proc for parallelism
    2. All CPU cores work simultaneously
    3. Data is already local (no network latency)

    Args:
        raw_path: Path to raw unfiltered parquet files
        filtered_path: Path to save filtered parquet files
        dataset_name: Display name for logging
        custom_filter_fn: Optional custom filter function

    Returns:
        (examples_saved, tokens_saved) tuple
    """
    # Check if already filtered (for resume)
    if filtered_path.exists() and args.resume:
        metadata_file = filtered_path / "metadata.json"

        # First check if fully completed
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if metadata.get("phase") == "filter":
                    print(f"  ✓ {dataset_name} filtering already completed ({metadata['examples_saved']:,} examples)")
                    return metadata['examples_saved'], metadata['tokens_saved']

        # Check for existing filtered files (filtering interrupted before metadata was saved)
        existing_samples, existing_tokens, _ = count_existing_progress(filtered_path, file_prefix="data")
        if existing_samples > 0:
            print(f"  ✓ {dataset_name} already has {existing_samples:,} filtered examples ({existing_tokens/1e9:.2f}B tokens)")
            print(f"     Using existing filtered data (metadata will be created)")
            # Create metadata for existing files
            metadata = {
                "phase": "filter",
                "dataset_name": dataset_name,
                "examples_saved": existing_samples,
                "tokens_saved": existing_tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            return existing_samples, existing_tokens

    try:
        # Load raw dataset
        raw_files = list(raw_path.glob("raw-*.parquet"))
        if not raw_files:
            print(f"  ✗ No raw files found in {raw_path}")
            return 0, 0

        dataset = load_dataset("parquet", data_files=[str(f) for f in raw_files], split="train")

        print(f"  Loaded {len(dataset):,} raw examples")

        start_time = time.time()

        # Define filter function
        def quality_filter(example):
            """Check if example passes quality filters."""
            text = example["text"]

            # English check
            if not is_english(text):
                return False

            # Quality check
            if not passes_basic_quality(text):
                return False

            # Custom filter if provided
            if custom_filter_fn and not custom_filter_fn(example):
                return False

            # Token count check
            if example["estimated_tokens"] < 100:
                return False

            return True

        # Apply filters with multiprocessing
        print(f"  🔍 Filtering {dataset_name} with {args.num_workers} workers...")
        dataset_filtered = dataset.filter(
            quality_filter,
            num_proc=args.num_workers,
            desc=f"  Filtering {dataset_name}"
        )

        elapsed_time = time.time() - start_time

        # Calculate token counts
        raw_tokens = sum(example["estimated_tokens"] for example in dataset)
        filtered_tokens = sum(example["estimated_tokens"] for example in dataset_filtered)

        print(f"  📊 Tokens: {raw_tokens/1e9:.2f}B → {filtered_tokens/1e9:.2f}B ({100*filtered_tokens/raw_tokens:.1f}% kept)")

        # Save filtered dataset
        filtered_path.mkdir(parents=True, exist_ok=True)

        # Save in shards
        batch_size = args.shard_size
        shard_num = 0
        for i in range(0, len(dataset_filtered), batch_size):
            batch = dataset_filtered.select(range(i, min(i + batch_size, len(dataset_filtered))))
            table = pa.Table.from_pylist([{
                "text": ex["text"],
                "source": ex["source"],
                "estimated_tokens": ex["estimated_tokens"]
            } for ex in batch])
            pq.write_table(table, filtered_path / f"data-{shard_num:05d}.parquet")
            shard_num += 1

        # Save metadata
        pass_rate = len(dataset_filtered) / len(dataset) if len(dataset) > 0 else 0
        token_pass_rate = filtered_tokens / raw_tokens if raw_tokens > 0 else 0
        metadata = {
            "phase": "filter",
            "dataset_name": dataset_name,
            "raw_examples": len(dataset),
            "raw_tokens": raw_tokens,
            "examples_saved": len(dataset_filtered),
            "tokens_saved": filtered_tokens,
            "pass_rate": pass_rate,
            "token_pass_rate": token_pass_rate,
            "filter_time_seconds": elapsed_time,
            "num_workers": args.num_workers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filtered_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return len(dataset_filtered), filtered_tokens

    except Exception as e:
        print(f"  ✗ Error filtering {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def dual_filter_shared_dataset(
    raw_path: Path,
    stage1_path: Path,
    stage2_path: Path,
    dataset_name: str,
    config: Dict,
    filter_stages: str = "both"
) -> Tuple[Dict, Dict]:
    """Apply different filters to shared dataset for Stage 1 and/or Stage 2.

    This is the KEY optimization: download once, filter once or twice with different criteria.

    Args:
        raw_path: Path to raw shared parquet files
        stage1_path: Where to save Stage 1 filtered data
        stage2_path: Where to save Stage 2 filtered data
        dataset_name: Display name
        config: SHARED_DATASETS_CONFIG entry for this dataset
        filter_stages: Which stages to filter - "stage1", "stage2", or "both" (default)

    Returns:
        (stage1_result, stage2_result) tuple of dicts with statistics
        Returns None for stages not requested
    """
    print(f"\n{'='*80}")
    stages_str = {"stage1": "Stage 1 only", "stage2": "Stage 2 only", "both": "Both stages"}[filter_stages]
    print(f"🔄 Filtering {dataset_name} (SHARED - {stages_str})")
    print(f"{'='*80}")
    print(f"  Raw data: {raw_path}")
    if filter_stages in ["stage1", "both"]:
        print(f"  → Stage 1: {stage1_path} ({config['stage1']['filter_type']})")
    if filter_stages in ["stage2", "both"]:
        print(f"  → Stage 2: {stage2_path} ({config['stage2']['filter_type']})")

    # Determine which stages to process
    should_process_stage1 = filter_stages in ["stage1", "both"]
    should_process_stage2 = filter_stages in ["stage2", "both"]

    # Check if filtering already done (for resume)
    stage1_done = False
    stage2_done = False
    stage1_result_cached = None
    stage2_result_cached = None

    if args.resume:
        # Check Stage 1 (only if requested)
        if should_process_stage1:
            stage1_samples, stage1_tokens_existing, _ = count_existing_progress(stage1_path, file_prefix="data")
            if stage1_samples > 0:
                print(f"  ✓ Stage 1 already filtered: {stage1_samples:,} examples ({stage1_tokens_existing/1e9:.2f}B tokens)")
                stage1_done = True
                stage1_result_cached = {
                    "dataset_name": f"{dataset_name} (Stage 1)",
                    "status": "success",
                    "filtered_examples": stage1_samples,
                    "filtered_tokens": stage1_tokens_existing,
                    "filter_type": config['stage1']['filter_type'],
                }
        else:
            # Stage 1 not requested, mark as done (will return None)
            stage1_done = True

        # Check Stage 2 (only if requested)
        if should_process_stage2:
            stage2_samples, stage2_tokens_existing, _ = count_existing_progress(stage2_path, file_prefix="data")
            if stage2_samples > 0:
                print(f"  ✓ Stage 2 already filtered: {stage2_samples:,} examples ({stage2_tokens_existing/1e9:.2f}B tokens)")
                stage2_done = True
                stage2_result_cached = {
                    "dataset_name": f"{dataset_name} (Stage 2)",
                    "status": "success",
                    "filtered_examples": stage2_samples,
                    "filtered_tokens": stage2_tokens_existing,
                    "filter_type": config['stage2']['filter_type'],
                }
        else:
            # Stage 2 not requested, mark as done (will return None)
            stage2_done = True

        # If all requested stages are done, return early
        if stage1_done and stage2_done:
            done_msg = []
            if should_process_stage1 and stage1_result_cached:
                done_msg.append("Stage 1")
            if should_process_stage2 and stage2_result_cached:
                done_msg.append("Stage 2")
            if done_msg:
                print(f"  ✓ {' and '.join(done_msg)} already filtered, skipping")
            return stage1_result_cached, stage2_result_cached

    try:
        # Initialize results (will be set to None for unrequested stages)
        stage1_result = stage1_result_cached  # Will be None if stage not requested
        stage2_result = stage2_result_cached  # Will be None if stage not requested

        # Load raw dataset (only if we need to filter at least one requested stage)
        need_to_filter = (should_process_stage1 and not stage1_done) or (should_process_stage2 and not stage2_done)
        if need_to_filter:
            raw_files = list(raw_path.glob("raw-*.parquet"))
            if not raw_files:
                print(f"  ✗ No raw files found")
                return None, None

            dataset = load_dataset("parquet", data_files=[str(f) for f in raw_files], split="train")
            print(f"  Loaded {len(dataset):,} raw examples")

            raw_tokens = sum(ex["estimated_tokens"] for ex in dataset)
            print(f"  Raw tokens: {raw_tokens/1e9:.2f}B")

        # =============================================================================
        # STAGE 1 FILTERING
        # =============================================================================
        if should_process_stage1 and not stage1_done:
            print(f"\n  🔍 Stage 1 Filter ({config['stage1']['filter_type']}) with {args.num_workers} workers...")
            start_time = time.time()

            def stage1_quality_filter(example):
                """Stage 1 quality filter."""
                text = example["text"]

                # English check
                if not is_english(text):
                    return False

                # Quality check
                if not passes_basic_quality(text):
                    return False

                # Custom Stage 1 filter if provided
                custom_filter = config['stage1']['filter_fn']
                if custom_filter and not custom_filter(example):
                    return False

                # Token count check
                if example["estimated_tokens"] < 100:
                    return False

                return True

            dataset_stage1 = dataset.filter(
                stage1_quality_filter,
                num_proc=args.num_workers,
                desc=f"  Stage 1 {dataset_name}"
            )

            stage1_elapsed = time.time() - start_time
            stage1_tokens = sum(ex["estimated_tokens"] for ex in dataset_stage1)

            print(f"  📊 Stage 1: {raw_tokens/1e9:.2f}B → {stage1_tokens/1e9:.2f}B ({100*stage1_tokens/raw_tokens:.1f}% kept)")

            # Save Stage 1 filtered data
            stage1_path.mkdir(parents=True, exist_ok=True)
            shard_num = 0
            for i in range(0, len(dataset_stage1), args.shard_size):
                batch = dataset_stage1.select(range(i, min(i + args.shard_size, len(dataset_stage1))))
                table = pa.Table.from_pylist([{
                    "text": ex["text"],
                    "source": ex["source"],
                    "estimated_tokens": ex["estimated_tokens"]
                } for ex in batch])
                pq.write_table(table, stage1_path / f"data-{shard_num:05d}.parquet")
                shard_num += 1

            stage1_result = {
                "dataset_name": f"{dataset_name} (Stage 1)",
                "status": "success",
                "raw_examples": len(dataset),
                "raw_tokens": raw_tokens,
                "filtered_examples": len(dataset_stage1),
                "filtered_tokens": stage1_tokens,
                "filter_type": config['stage1']['filter_type'],
                "filter_time": stage1_elapsed,
            }
        else:
            stage1_result = stage1_result_cached

        # =============================================================================
        # STAGE 2 FILTERING
        # =============================================================================
        if should_process_stage2 and not stage2_done:
            print(f"\n  🔍 Stage 2 Filter ({config['stage2']['filter_type']}) with {args.num_workers} workers...")
            start_time = time.time()

            def stage2_quality_filter(example):
                """Stage 2 quality filter (stricter)."""
                text = example["text"]

                # English check
                if not is_english(text):
                    return False

                # Quality check
                if not passes_basic_quality(text):
                    return False

                # Custom Stage 2 filter if provided (e.g., score ≥3, stars ≥2)
                custom_filter = config['stage2']['filter_fn']
                if custom_filter and not custom_filter(example):
                    return False

                # Token count check
                if example["estimated_tokens"] < 100:
                    return False

                return True

            dataset_stage2 = dataset.filter(
                stage2_quality_filter,
                num_proc=args.num_workers,
                desc=f"  Stage 2 {dataset_name}"
            )

            stage2_elapsed = time.time() - start_time
            stage2_tokens = sum(ex["estimated_tokens"] for ex in dataset_stage2)

            print(f"  📊 Stage 2: {raw_tokens/1e9:.2f}B → {stage2_tokens/1e9:.2f}B ({100*stage2_tokens/raw_tokens:.1f}% kept)")

            # Save Stage 2 filtered data
            stage2_path.mkdir(parents=True, exist_ok=True)
            shard_num = 0
            for i in range(0, len(dataset_stage2), args.shard_size):
                batch = dataset_stage2.select(range(i, min(i + args.shard_size, len(dataset_stage2))))
                table = pa.Table.from_pylist([{
                    "text": ex["text"],
                    "source": ex["source"],
                    "estimated_tokens": ex["estimated_tokens"]
                } for ex in batch])
                pq.write_table(table, stage2_path / f"data-{shard_num:05d}.parquet")
                shard_num += 1

            stage2_result = {
                "dataset_name": f"{dataset_name} (Stage 2)",
                "status": "success",
                "raw_examples": len(dataset),
                "raw_tokens": raw_tokens,
                "filtered_examples": len(dataset_stage2),
                "filtered_tokens": stage2_tokens,
                "filter_type": config['stage2']['filter_type'],
                "filter_time": stage2_elapsed,
            }
        else:
            stage2_result = stage2_result_cached

        print(f"  ✓ Filtering complete!")
        if stage1_result:
            print(f"    Stage 1: {stage1_result['filtered_examples']:,} docs, {stage1_result['filtered_tokens']/1e9:.2f}B tokens")
        if stage2_result:
            print(f"    Stage 2: {stage2_result['filtered_examples']:,} docs, {stage2_result['filtered_tokens']/1e9:.2f}B tokens")

        return stage1_result, stage2_result

    except Exception as e:
        print(f"  ✗ Error dual-filtering {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# Pipeline Orchestration (Parallel Downloads + Serial Filtering)
# =============================================================================

def download_and_filter_one_dataset(
    dataset_name: str,
    hf_dataset: str,
    target_tokens: int,
    raw_base_dir: Path,
    filtered_base_dir: Path,
    text_field: str = "text",
    split: str = "train",
    custom_filter_fn=None,
    **load_kwargs
) -> Dict:
    """Download and filter a single dataset (two-phase approach).

    This function:
    1. Downloads raw data fast (no filtering)
    2. Filters with multiprocessing (uses all cores)

    Args:
        dataset_name: Display name
        hf_dataset: HuggingFace dataset ID
        target_tokens: Token quota
        raw_base_dir: Base directory for raw data
        filtered_base_dir: Base directory for filtered data
        text_field: Text field name in dataset
        split: Dataset split
        custom_filter_fn: Optional custom filter
        **load_kwargs: Args for load_dataset

    Returns:
        Dict with statistics
    """
    raw_path = raw_base_dir / dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    filtered_path = filtered_base_dir / dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    result = {
        "dataset_name": dataset_name,
        "status": "pending",
        "raw_examples": 0,
        "raw_tokens": 0,
        "filtered_examples": 0,
        "filtered_tokens": 0,
    }

    try:
        # Phase 1: Fast download
        print(f"\n{'='*80}")
        print(f"📦 {dataset_name}")
        print(f"{'='*80}")
        print(f"PHASE 1: Fast Download (target: {target_tokens/1e9:.2f}B tokens)")

        raw_examples, raw_tokens, _ = download_raw_fast(
            dataset_name,
            hf_dataset,
            target_tokens,
            raw_path,
            text_field,
            split,
            custom_filter_fn,
            **load_kwargs
        )

        result["raw_examples"] = raw_examples
        result["raw_tokens"] = raw_tokens

        if raw_examples == 0:
            result["status"] = "download_failed"
            return result

        print(f"  ✓ Downloaded {raw_examples:,} raw examples ({raw_tokens/1e9:.2f}B tokens)")

        # Phase 2: Multiprocessing filter
        print(f"\nPHASE 2: Parallel Filtering ({args.num_workers} workers)")

        filtered_examples, filtered_tokens = filter_dataset_parallel(
            raw_path,
            filtered_path,
            dataset_name,
            custom_filter_fn
        )

        result["filtered_examples"] = filtered_examples
        result["filtered_tokens"] = filtered_tokens

        if filtered_examples == 0:
            result["status"] = "filter_failed"
            return result

        pass_rate = filtered_examples / raw_examples if raw_examples > 0 else 0
        print(f"  ✓ Filtered to {filtered_examples:,} examples ({filtered_tokens/1e9:.2f}B tokens)")
        print(f"    Pass rate: {100*pass_rate:.1f}%")

        result["status"] = "success"
        return result

    except Exception as e:
        print(f"  ✗ Error: {e}")
        result["status"] = "error"
        result["error"] = str(e)
        return result


# =============================================================================
# Stage 1 & Stage 2 Dataset Downloaders (Using New Pipeline)
# =============================================================================

# All downloaders now use download_and_filter_one_dataset() which:
# 1. Downloads raw data fast (no filtering)
# 2. Filters with multiprocessing
# These can be run in parallel with ThreadPoolExecutor


# GSM8K special handler (small dataset, needs special handling)
def prepare_gsm8k_train(target_tokens: int, raw_base_dir: Path, filtered_base_dir: Path) -> Dict:
    """Download GSM8K training set (special case - small dataset)."""
    dataset_name = "GSM8K (train)"
    raw_path = raw_base_dir / "gsm8k_train"
    filtered_path = filtered_base_dir / "gsm8k_train"

    result = {
        "dataset_name": dataset_name,
        "status": "pending",
        "raw_examples": 0,
        "raw_tokens": 0,
        "filtered_examples": 0,
        "filtered_tokens": 0,
    }

    # Check if already exists
    if filtered_path.exists() and args.resume:
        metadata_file = filtered_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                result.update({
                    "status": "success",
                    "filtered_examples": metadata.get("examples_saved", 0),
                    "filtered_tokens": metadata.get("tokens_saved", 0),
                })
                return result

    try:
        print(f"\n{'='*80}")
        print(f"📦 {dataset_name}")
        print(f"{'='*80}")
        print("  ⚠️  Using training split ONLY (not test!)")

        filtered_path.mkdir(parents=True, exist_ok=True)

        # Load GSM8K train split
        dataset = load_dataset("gsm8k", "main", split="train", token=False)

        batch = []
        total_tokens = 0

        for example in dataset:
            text = f"Question: {example['question']}\n\nAnswer: {example['answer']}"
            tokens = estimate_tokens(text)

            batch.append({
                "text": text,
                "source": dataset_name,
                "estimated_tokens": tokens
            })
            total_tokens += tokens

        # Repeat if needed (GSM8K is small: 7,473 examples)
        original_size = len(batch)
        while total_tokens < target_tokens:
            for item in batch[:original_size]:
                batch.append(item)
                total_tokens += item["estimated_tokens"]
                if total_tokens >= target_tokens:
                    break

        # Save to parquet
        table = pa.Table.from_pylist(batch)
        pq.write_table(table, filtered_path / "data-00000.parquet")

        # Save metadata
        metadata = {
            "phase": "filter",
            "dataset_name": dataset_name,
            "examples_saved": len(batch),
            "tokens_saved": total_tokens,
            "target_tokens": target_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filtered_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved {len(batch):,} examples ({total_tokens/1e9:.2f}B tokens)")

        result.update({
            "status": "success",
            "filtered_examples": len(batch),
            "filtered_tokens": total_tokens,
        })
        return result

    except Exception as e:
        print(f"  ✗ Error: {e}")
        result["status"] = "error"
        return result


# =============================================================================
# Main Pipeline with Parallel Downloads
# =============================================================================

def main():
    global args

    # Calculate token budgets
    budgets = calculate_token_budgets(args.model_size, args.token_buffer)

    stage1_budgets = budgets["stage1"]
    stage2_budgets = budgets["stage2"]

    # Calculate totals
    stage1_total = sum(stage1_budgets.values())
    stage2_total = sum(stage2_budgets.values())
    grand_total = stage1_total + stage2_total

    # Estimate disk space
    stage1_gb = tokens_to_gigabytes(stage1_total)
    stage2_gb = tokens_to_gigabytes(stage2_total)
    total_gb = stage1_gb + stage2_gb

    # Display plan
    print("=" * 80)
    print("Pretraining Dataset Download Plan (PARALLEL OPTIMIZED)")
    print("=" * 80)
    print(f"\nModel size: {args.model_size}")
    print(f"Stages: {args.stages}")
    print(f"Token buffer: {args.token_buffer:.0%} (for preprocessing losses)")
    print(f"Output directory: {args.output_dir}")
    print(f"Parallel downloads: {args.parallel_downloads}")
    print(f"Filter workers: {args.num_workers}")
    print(f"Resume mode: {'✓ Enabled' if args.resume else '✗ Disabled'}")

    if args.stages in ["stage1", "both"]:
        print(f"\n{'='*80}")
        print(f"Stage 1: Broad Pretraining ({stage1_total/1e9:.2f}B tokens, ~{stage1_gb:.1f}GB)")
        print(f"{'='*80}")
        for name, tokens in stage1_budgets.items():
            pct = 100 * tokens / stage1_total
            gb = tokens_to_gigabytes(tokens)
            print(f"  {name:25} {tokens/1e9:6.2f}B tokens ({pct:5.1f}%)  ~{gb:5.1f}GB")

    if args.stages in ["stage2", "both"]:
        print(f"\n{'='*80}")
        print(f"Stage 2: Domain Upsampling ({stage2_total/1e9:.2f}B tokens, ~{stage2_gb:.1f}GB)")
        print(f"{'='*80}")
        for name, tokens in stage2_budgets.items():
            pct = 100 * tokens / stage2_total
            gb = tokens_to_gigabytes(tokens)
            print(f"  {name:25} {tokens/1e9:6.2f}B tokens ({pct:5.1f}%)  ~{gb:5.1f}GB")

    print(f"\n{'─'*80}")
    print(f"Total: {grand_total/1e9:.2f}B tokens (~{total_gb:.1f}GB disk space)")
    print(f"Estimated time: 3-5 hours (with {args.parallel_downloads} parallel downloads)")
    print(f"{'─'*80}")

    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without downloading.")
        return

    print("\n" + "=" * 80)
    print(f"Starting PARALLEL downloads ({args.parallel_downloads} concurrent)...")
    print("=" * 80)

    # Create output directories
    output_base = Path(args.output_dir)
    raw_datasets_dir = output_base / "raw_datasets"
    raw_unfiltered_dir = raw_datasets_dir / "raw_unfiltered"
    shared_raw_dir = raw_unfiltered_dir / "shared"
    raw_datasets_dir.mkdir(parents=True, exist_ok=True)
    raw_unfiltered_dir.mkdir(parents=True, exist_ok=True)
    shared_raw_dir.mkdir(parents=True, exist_ok=True)

    # Calculate shared dataset quotas (OPTIMIZATION: download once, filter twice)
    shared_quotas = calculate_shared_dataset_quotas(stage1_budgets, stage2_budgets)

    print(f"\n💡 OPTIMIZATION: Shared datasets (download once, filter twice)")
    print(f"   Shared: {', '.join(shared_quotas.keys())}")
    print(f"   Savings: ~40% less download time and bandwidth\n")

    results = []
    total_start_time = time.time()

    # =============================================================================
    # PHASE 1: Download shared datasets (in parallel)
    # =============================================================================
    # NOTE: Shared datasets are downloaded regardless of --stages flag
    # They will be filtered appropriately for the requested stage(s)
    print("=" * 80)
    print("PHASE 1: Download SHARED Datasets")
    print("=" * 80)
    print(f"⚡ Downloading {len(shared_quotas)} shared datasets in parallel...")

    shared_download_jobs = []
    for dataset_name, quota_info in shared_quotas.items():
        config = SHARED_DATASETS_CONFIG[dataset_name]
        shared_download_jobs.append((
            dataset_name,
            config['hf_dataset'],
            quota_info['raw_quota'],
            shared_raw_dir / dataset_name,
            config['text_field'],
            config['split'],
            None,  # No custom filter during download
            config['load_kwargs']
        ))

    # Download in parallel
    shared_raw_paths = {}
    with ThreadPoolExecutor(max_workers=args.parallel_downloads) as executor:
        future_to_name = {}
        for job in shared_download_jobs:
            name, hf_ds, quota, output_path, text_field, split, custom_filter, load_kwargs = job
            future = executor.submit(
                download_raw_fast,
                name, hf_ds, quota, output_path,
                text_field, split, custom_filter, **load_kwargs
            )
            future_to_name[future] = (name, output_path)

        # Collect download results
        for future in as_completed(future_to_name):
            name, output_path = future_to_name[future]
            try:
                examples, tokens, path = future.result()
                shared_raw_paths[name] = (output_path, examples, tokens)
                print(f"  ✓ {name}: {examples:,} examples, {tokens/1e9:.2f}B tokens downloaded")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

    # =============================================================================
    # PHASE 2: Filter shared datasets (for requested stage(s))
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Filter SHARED Datasets")
    print("=" * 80)
    stages_msg = {
        "stage1": "Stage 1 only",
        "stage2": "Stage 2 only",
        "both": "both Stage 1 and Stage 2"
    }
    print(f"🔄 Filtering shared datasets for {stages_msg[args.stages]}...")

    for dataset_name in shared_quotas.keys():
        if dataset_name not in shared_raw_paths:
            continue

        raw_path, raw_examples, raw_tokens = shared_raw_paths[dataset_name]
        config = SHARED_DATASETS_CONFIG[dataset_name]

        stage1_result, stage2_result = dual_filter_shared_dataset(
            raw_path,
            raw_datasets_dir / "stage1" / dataset_name,
            raw_datasets_dir / "stage2" / dataset_name,
            dataset_name,
            config,
            filter_stages=args.stages  # Pass which stages to filter
        )

        if stage1_result:
            results.append(stage1_result)
        if stage2_result:
            results.append(stage2_result)

    # =============================================================================
    # PHASE 3: Download unique datasets (in parallel)
    # =============================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: Download UNIQUE Datasets")
    print("=" * 80)
    print(f"⚡ Downloading unique datasets in parallel...")

    unique_jobs = []

    # Unique to Stage 1
    if args.stages in ["stage1", "both"]:
        raw_stage1 = raw_unfiltered_dir / "stage1_only"
        filt_stage1 = raw_datasets_dir / "stage1"

        unique_jobs.extend([
            ("Wikipedia", "wikipedia", stage1_budgets["wikipedia"],
             raw_stage1, filt_stage1, "text", "train", None, {"name": "20231101.en", "token": False}),

            ("Books (Gutenberg)", "sedthh/gutenberg_english", stage1_budgets["books_gutenberg"],
             raw_stage1, filt_stage1, "text", "train", None, {"token": False}),
        ])

        # Add Stack v2 language subsets for Stage 1
        stack_v2_languages = calculate_stack_v2_language_budgets(stage1_budgets["stack_v2"])
        for lang, tokens in stack_v2_languages.items():
            unique_jobs.append((
                f"Stack-v2 ({lang})",
                "bigcode/the-stack-v2-dedup",
                tokens,
                raw_stage1,
                filt_stage1,
                "content",
                "train",
                None,  # No custom filter for Stage 1
                {"data_dir": f"data/{lang}", "token": True}
            ))

    # Unique to Stage 2
    if args.stages in ["stage2", "both"]:
        raw_stage2 = raw_unfiltered_dir / "stage2_only"
        filt_stage2 = raw_datasets_dir / "stage2"

        unique_jobs.extend([
            ("TinyGSM-MIND", "Aeala/TinyGSM-MIND", stage2_budgets["tinygsm"],
             raw_stage2, filt_stage2, "text", "train", None, {"token": False}),

            ("Algebraic Stack", "EleutherAI/proof-pile-2", stage2_budgets["algebraic_stack"],
             raw_stage2, filt_stage2, "text", "train", None, {"token": False}),
        ])

        # Add Stack v2 language subsets for Stage 2 (with stars ≥2 filter)
        stack_v2_languages = calculate_stack_v2_language_budgets(stage2_budgets["stack_v2_filtered"])
        for lang, tokens in stack_v2_languages.items():
            # Custom filter for Stage 2: stars ≥2
            stars_filter = lambda ex: ex.get('max_stars_count', 0) >= 2
            unique_jobs.append((
                f"Stack-v2 ({lang} ≥2★)",
                "bigcode/the-stack-v2-dedup",
                tokens,
                raw_stage2,
                filt_stage2,
                "content",
                "train",
                stars_filter,  # Filter for repos with ≥2 stars
                {"data_dir": f"data/{lang}", "token": True}
            ))

    # Download and filter unique datasets in parallel
    with ThreadPoolExecutor(max_workers=args.parallel_downloads) as executor:
        future_to_name = {}
        for job in unique_jobs:
            name, hf_ds, tokens, raw_base, filt_base, text_field, split, custom_filter, load_kwargs = job
            future = executor.submit(
                download_and_filter_one_dataset,
                name, hf_ds, tokens, raw_base, filt_base,
                text_field, split, custom_filter, **load_kwargs
            )
            future_to_name[future] = name

        # Collect results
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\n✗ {name} failed with exception: {e}")
                results.append({
                    "dataset_name": name,
                    "status": "exception",
                    "error": str(e)
                })

    # Handle GSM8K separately (special case)
    if args.stages in ["stage2", "both"]:
        gsm8k_result = prepare_gsm8k_train(
            stage2_budgets["gsm8k_train"],
            raw_unfiltered_dir / "stage2_only",
            raw_datasets_dir / "stage2"
        )
        results.append(gsm8k_result)

    total_elapsed_time = time.time() - total_start_time

    # Print summary
    print("\n" + "=" * 80)
    print("Download & Filter Summary")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"\n✓ Successful: {len(successful)} / {len(results)}")
    for result in successful:
        name = result["dataset_name"]
        filt_ex = result["filtered_examples"]
        filt_tok = result["filtered_tokens"]
        raw_ex = result.get("raw_examples", 0)
        pass_rate = (filt_ex / raw_ex * 100) if raw_ex > 0 else 0
        print(f"  ✓ {name:30} {filt_ex:8,} docs  {filt_tok/1e9:6.2f}B tokens  (pass: {pass_rate:.1f}%)")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for result in failed:
            print(f"  ✗ {result['dataset_name']:30} {result['status']}")

    total_docs = sum(r.get("filtered_examples", 0) for r in successful)
    total_tokens = sum(r.get("filtered_tokens", 0) for r in successful)

    print(f"\n{'─'*80}")
    print(f"Total: {total_docs:,} documents, {total_tokens/1e9:.2f}B tokens")
    print(f"Time: {total_elapsed_time/3600:.2f} hours")
    print(f"{'─'*80}")

    # Save master metadata
    master_metadata = {
        "model_size": args.model_size,
        "stages": args.stages,
        "token_buffer": args.token_buffer,
        "parallel_downloads": args.parallel_downloads,
        "num_workers": args.num_workers,
        "budgets": budgets,
        "results": results,
        "total_time_seconds": total_elapsed_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED
    }

    with open(raw_datasets_dir / "download_metadata.json", 'w') as f:
        json.dump(master_metadata, f, indent=2)

    print(f"\n✅ Download complete!")
    print(f"   Filtered data: {raw_datasets_dir}/")
    if args.keep_raw:
        print(f"   Raw data (kept): {raw_unfiltered_dir}/")
    print(f"   Metadata: {raw_datasets_dir}/download_metadata.json")
    print(f"\nNext step: Run process_pretraining_data.py to create final mixtures")


if __name__ == "__main__":
    main()
