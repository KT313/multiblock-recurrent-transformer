#!/usr/bin/env python3
"""
Dataset Downloader with Optional Parallelism

Downloads pretraining datasets with incremental saves and optional parallel downloading.
No filtering - just robust downloads (filtering happens in separate preprocessing step).

Usage:
    # Sequential (default)
    python download_simple.py --output_dir /path/to/output --cache_dir /path/to/cache

    # Parallel (2 datasets at once)
    python download_simple.py --output_dir /path/to/output --cache_dir /path/to/cache --parallel 2

    # Parallel (4 datasets at once)
    python download_simple.py --output_dir /path/to/output --cache_dir /path/to/cache --parallel 4

Author: Claude Code
Date: 2025-01-26
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set cache directory from args before importing datasets
parser = argparse.ArgumentParser(description="Simple sequential dataset downloader")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
parser.add_argument("--parallel", type=int, default=1, help="Number of datasets to download in parallel (default: 1)")
args = parser.parse_args()

if args.cache_dir:
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_DATASETS_CACHE'] = args.cache_dir
    print(f"Using cache directory: {args.cache_dir}")

# Now import datasets
from datasets import load_dataset
import datasets
import pyarrow as pa
import pyarrow.parquet as pq

scale = 1.0


# =============================================================================
# Dataset Configuration (Hardcoded for 300M model)
# =============================================================================

DATASETS = [
    # Web/General Text
    # {
    #     "name": "fineweb_edu",
    #     "hf_dataset": "HuggingFaceFW/fineweb-edu",
    #     "text_field": "text",
    #     "target_samples": int(9_000_000 * scale),  # ~18B tokens @ 2000 tok/sample
    #     "load_kwargs": {"name": "CC-MAIN-2013-20"},
    # },
    {
        "name": "wikipedia",
        "hf_dataset": "wikipedia",
        "text_field": "text",
        "target_samples": int(1_700_000 * scale),  # ~2.5B tokens @ 1500 tok/sample
        "load_kwargs": {"name": "20220301.en", "trust_remote_code": True},
    },
    {
        "name": "books_gutenberg",
        "hf_dataset": "sedthh/gutenberg_english",
        "text_field": "text",
        "target_samples": int(600_000 * scale),  # ~1.8B tokens @ 3000 tok/sample
        "load_kwargs": {},
    },

    # Academic/STEM
    {
        "name": "peso",
        "hf_dataset": "nampdn-ai/mini-peS2o",
        "text_field": "text",
        "target_samples": int(600_000 * scale),  # ~0.9B tokens @ 1500 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "arxiv",
        "hf_dataset": "common-pile/arxiv_papers_filtered",
        "text_field": "text",
        "target_samples": int(400_000 * scale),  # ~0.6B tokens @ 1500 tok/sample
        "load_kwargs": {},
    },

    # Math
    {
        "name": "openwebmath",
        "hf_dataset": "open-web-math/open-web-math",
        "text_field": "text",
        "target_samples": int(2_000_000 * scale),  # ~1.5B tokens @ 750 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "tinygsm",
        "hf_dataset": "ostapeno/tinygsm-mind",
        "text_field": "text",
        "target_samples": int(1_600_000 * scale),  # ~0.5B tokens @ 300 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "algebraic_stack",
        "hf_dataset": "EleutherAI/proof-pile-2",
        "text_field": "text",
        "target_samples": int(450_000 * scale),  # ~0.34B tokens @ 750 tok/sample
        "load_kwargs": {"name": "algebraic-stack", "trust_remote_code": True},
    },
    {
        "name": "gsm8k",
        "hf_dataset": "gsm8k",
        "text_field": "question",  # Will combine question + answer
        "target_samples": int(550_000 * scale),  # ~0.17B tokens @ 300 tok/sample (with repetition)
        "load_kwargs": {"name": "main"},
        "special_handler": "gsm8k",  # Needs special processing
    },

    # Code - Stack v2 (10 languages)
    {
        "name": "github_code_clean_python",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(3_200_000 * scale),  # ~1.6B tokens @ 500 tok/sample
        "load_kwargs": {"name": "Python-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(3_200_000 * scale)},
    },
    {
        "name": "github_code_clean_javascript",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(2_200_000 * scale),  # ~1.1B tokens
        "load_kwargs": {"name": "JavaScript-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(2_200_000 * scale)},
    },
    {
        "name": "github_code_clean_typescript",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(1_100_000 * scale),  # ~0.55B tokens
        "load_kwargs": {"name": "TypeScript-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(1_100_000 * scale)},
    },
    {
        "name": "github_code_clean_java",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(1_100_000 * scale),  # ~0.55B tokens
        "load_kwargs": {"name": "Java-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(1_100_000 * scale)},
    },
    {
        "name": "github_code_clean_cpp",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(900_000 * scale),  # ~0.45B tokens
        "load_kwargs": {"name": "C++-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(900_000 * scale)},
    },
    {
        "name": "github_code_clean_go",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(800_000 * scale),  # ~0.4B tokens
        "load_kwargs": {"name": "GO-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(800_000 * scale)},
    },
    {
        "name": "github_code_clean_rust",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(550_000 * scale),  # ~0.28B tokens
        "load_kwargs": {"name": "Rust-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(550_000 * scale)},
    },
    {
        "name": "github_code_clean_shell",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(450_000 * scale),  # ~0.23B tokens
        "load_kwargs": {"name": "Shell-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(450_000 * scale)},
    },
    {
        "name": "github_code_clean_sql",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(350_000 * scale),  # ~0.18B tokens
        "load_kwargs": {"name": "SQL-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(350_000 * scale)},
    },
    {
        "name": "github_code_clean_html",
        "hf_dataset": "/path/to/shared_storage/recpre/datasets/github-code-clean-modified/github-code-clean.py",
        "text_field": "code",
        "target_samples": int(350_000 * scale),  # ~0.18B tokens
        "load_kwargs": {"name": "HTML-all", "trust_remote_code": True, "token": os.environ.get("HF_TOKEN"), "max_samples": int(350_000 * scale)},
    },
]


# =============================================================================
# Download Function
# =============================================================================

def download_dataset(config: dict, output_dir: Path) -> bool:
    """Download a single dataset with incremental saves.

    Args:
        config: Dataset configuration dict
        output_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    dataset_name = config["name"]
    hf_dataset = config["hf_dataset"]
    text_field = config["text_field"]
    target_samples = config["target_samples"]
    load_kwargs = config.get("load_kwargs", {})
    special_handler = config.get("special_handler", None)

    # Create output directory
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📦 {dataset_name}")
    print(f"{'='*80}")
    print(f"  Dataset: {hf_dataset}")
    print(f"  Target: {target_samples:,} samples")
    print(f"  Output: {dataset_dir}")

    try:
        # Special handler for GSM8K
        if special_handler == "gsm8k":
            return download_gsm8k(config, dataset_dir)

        # Load dataset with split slicing (much faster than streaming!)
        print(f"  Loading {target_samples:,} samples...")
        dataset = load_dataset(
            hf_dataset,
            split=f"train[:{target_samples}]",  # Download exact number needed
            **load_kwargs
        )

        print(f"  Loaded {len(dataset):,} samples")
        print(f"  Saving to parquet shards...")

        # Save in batches
        batch_size = 100_000  # Save every 10k samples
        shard_num = 0

        print(f"  Length dataset: {len(dataset)}")
        

        for i in tqdm(range(0, len(dataset), batch_size), desc="  💾 Saving", unit=" shards"):
            # Get batch slice (vectorized - no Python iteration)
            end_idx = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, end_idx))

            # Save directly to parquet (Arrow → Parquet)
            filename = f"shard-{shard_num:05d}.parquet"
            batch.to_parquet(dataset_dir / filename)
            # print(f"  Saved {filename}, currenty downloaded: {i} / {target_samples}")

            shard_num += 1

        print(f"  ✓ Downloaded and saved {len(dataset):,} samples in {shard_num} shards")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_gsm8k(config: dict, dataset_dir: Path) -> bool:
    """Special handler for GSM8K (small dataset, needs repetition)."""
    try:
        print(f"  ⚠️  Special handling for GSM8K (will repeat examples to reach target)")

        # Load full dataset
        dataset = load_dataset("gsm8k", "main", split="train")

        # Transform: combine question + answer (vectorized)
        def combine_qa(example):
            return {
                "text": f"Question: {example['question']}\n\nAnswer: {example['answer']}",
                "source": "gsm8k"
            }

        dataset = dataset.map(combine_qa, remove_columns=dataset.column_names)

        original_count = len(dataset)
        target_samples = config["target_samples"]

        print(f"  Original: {original_count:,} samples, Target: {target_samples:,}")

        # Create repeated indices for dataset.select()
        num_full_copies = target_samples // original_count
        remainder = target_samples % original_count
        indices = list(range(original_count)) * num_full_copies + list(range(remainder))

        # Select with repeated indices (vectorized)
        repeated_dataset = dataset.select(indices)

        print(f"  Created {len(repeated_dataset):,} samples (repeated {num_full_copies}x + {remainder})")
        print(f"  Saving to parquet shards...")

        # Save in batches
        batch_size = 100_000
        shard_num = 0

        for i in tqdm(range(0, len(repeated_dataset), batch_size), desc="  💾 Saving", unit=" shards"):
            end_idx = min(i + batch_size, len(repeated_dataset))
            batch = repeated_dataset.select(range(i, end_idx))
            
            filename = f"shard-{shard_num:05d}.parquet"
            batch.to_parquet(dataset_dir / filename)
            # print(f"  Saved {filename}, currenty downloaded: {i} / {target_samples}")

            shard_num += 1

        print(f"  ✓ Saved {len(repeated_dataset):,} samples in {shard_num} shards")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    output_dir = Path(args.output_dir) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    parallel = args.parallel

    print("="*80)
    if parallel > 1:
        print(f"Parallel Dataset Downloader ({parallel} concurrent)")
    else:
        print("Sequential Dataset Downloader")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Total samples: {sum(d['target_samples'] for d in DATASETS):,}")
    print(f"Parallelism: {parallel} dataset(s) at a time")
    print("="*80)

    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        print("\n⚠️  Warning: HF_TOKEN not set. Stack v2 downloads will fail.")
        print("   Set it with: export HF_TOKEN=your_token_here")

    successful = []
    failed = []

    start_time = time.time()

    if parallel == 1:
        # Sequential download (original behavior)
        for config in DATASETS:
            success = download_dataset(config, output_dir)
            if success:
                successful.append(config["name"])
            else:
                failed.append(config["name"])
    else:
        # Parallel download using ThreadPoolExecutor
        print(f"\n🚀 Launching {parallel} parallel downloads...\n")

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            # Submit all download jobs
            future_to_config = {
                executor.submit(download_dataset, config, output_dir): config
                for config in DATASETS
            }

            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                dataset_name = config["name"]
                completed += 1

                try:
                    success = future.result()
                    if success:
                        successful.append(dataset_name)
                        print(f"\n✓ [{completed}/{len(DATASETS)}] {dataset_name} completed successfully")
                    else:
                        failed.append(dataset_name)
                        print(f"\n✗ [{completed}/{len(DATASETS)}] {dataset_name} failed")
                except Exception as e:
                    failed.append(dataset_name)
                    print(f"\n✗ [{completed}/{len(DATASETS)}] {dataset_name} failed with exception: {e}")

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    print(f"✓ Successful: {len(successful)} / {len(DATASETS)}")
    for name in successful:
        print(f"    - {name}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for name in failed:
            print(f"    - {name}")

    print(f"\nTotal time: {elapsed/3600:.2f} hours")
    print(f"Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
