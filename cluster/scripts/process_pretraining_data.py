#!/usr/bin/env python3
"""
Process Pretraining Datasets with Comprehensive Preprocessing Pipeline

This script loads raw downloaded datasets and applies a comprehensive preprocessing
pipeline following modern best practices for LLM pretraining data preparation.

Preprocessing Pipeline (in order):
1. Exact Deduplication (MD5 hash-based) - removes ~10-15% duplicates
2. Fuzzy Deduplication (MinHash + LSH) - removes ~30-40% near-duplicates
3. Quality Filtering (heuristic-based) - removes ~5-10% low-quality
4. PII Removal (regex-based) - masks emails, IPs, phone numbers, API keys
5. Benchmark Decontamination (n-gram overlap) - removes test set contamination

After preprocessing, creates Stage 1 and Stage 2 mixtures with correct token budgets.

Usage:
    # Process all downloaded data
    python process_pretraining_data.py \
        --input_dir /path/to/raw_datasets \
        --output_dir /path/to/processed \
        --model_size 300M \
        --stages both \
        --num_workers 8

    # Process only Stage 1 (faster for testing)
    python process_pretraining_data.py \
        --input_dir /path/to/raw_datasets \
        --output_dir /path/to/processed \
        --stages stage1 \
        --skip_fuzzy_dedup  # Skip slow fuzzy dedup for quick testing

Author: Claude Code
Date: 2025-01-25
"""

import os
import sys
import argparse
import json
import random
import re
import hashlib
import pickle
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import time

# Parse args first
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process pretraining datasets with comprehensive preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing raw_datasets/ subdirectory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed datasets"
    )

    # Model size determines token budgets
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["300M", "1B"],
        default="300M",
        help="Model size (determines token budget): 300M=30B tokens, 1B=100B tokens"
    )

    # Which stages to process
    parser.add_argument(
        "--stages",
        type=str,
        choices=["stage1", "stage2", "both", "merged"],
        default="both",
        help="Which training stages to process"
    )

    # Cache directory
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Custom HuggingFace cache directory (for benchmark datasets)"
    )

    # Tokenizer for accurate token counting
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional: Path to tokenizer for accurate token counting (if not provided, uses char/4 estimation)"
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional: Maximum sequence length in tokens. If provided with --tokenizer_path, texts will be truncated to this length."
    )

    # Processing options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing"
    )

    parser.add_argument(
        "--skip_exact_dedup",
        action="store_true",
        help="Skip exact deduplication (MD5-based)"
    )

    parser.add_argument(
        "--skip_fuzzy_dedup",
        action="store_true",
        help="Skip fuzzy deduplication (MinHash+LSH, slow but very effective)"
    )

    parser.add_argument(
        "--skip_quality_filter",
        action="store_true",
        help="Skip quality filtering"
    )

    parser.add_argument(
        "--skip_pii_removal",
        action="store_true",
        help="Skip PII removal"
    )

    parser.add_argument(
        "--skip_decontamination",
        action="store_true",
        help="Skip benchmark decontamination"
    )

    # Deduplication parameters
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for fuzzy dedup (default: 0.8)"
    )

    parser.add_argument(
        "--minhash_num_perm",
        type=int,
        default=256,
        help="Number of permutations for MinHash (default: 256)"
    )

    # Output options
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Number of examples per output parquet shard (default: 10,000)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show processing plan without actually processing"
    )

    return parser.parse_args()


args = parse_arguments()

# Set environment variables
if args.cache_dir:
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir
    print(f"Using custom cache directory: {args.cache_dir}")
else:
    default_cache = "/path/to/fast_storage/.cache"
    if os.path.exists("/path/to/fast_storage"):
        os.environ["HF_HOME"] = default_cache
        os.environ["HF_DATASETS_CACHE"] = default_cache
        print(f"Using default cache: {default_cache}")

# Now import datasets
try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    from tqdm.auto import tqdm
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Please install: pip install datasets tqdm numpy pyarrow datasketch")
    raise

# Import MinHash for fuzzy deduplication
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("Warning: datasketch not installed. Fuzzy deduplication will be skipped.")
    print("Install with: pip install datasketch")
    MinHash = None
    MinHashLSH = None

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# Token Budget Calculation (must match download script)
# =============================================================================

def calculate_target_tokens(model_size: str) -> Dict[str, int]:
    """Calculate target token counts for Stage 1 and Stage 2.

    These are the FINAL token counts after all preprocessing.
    No buffer applied here (buffer was in download script).
    """
    TOTAL_TOKENS = {
        "300M": 30_000_000_000,  # 30B tokens
        "1B": 100_000_000_000,   # 100B tokens
    }

    total = TOTAL_TOKENS[model_size]
    return {
        "stage1": int(total * 0.80),  # 80% for Stage 1
        "stage2": int(total * 0.20),  # 20% for Stage 2
    }


def estimate_tokens(text: str) -> int:
    """Estimate token count using characters/4 heuristic."""
    return len(text) // 4


# =============================================================================
# Exact Deduplication (MD5-based)
# =============================================================================

def exact_deduplicate(dataset: Dataset, desc: str = "Exact dedup") -> Tuple[Dataset, Dict]:
    """Remove exact duplicates using MD5 hashing.

    Fast and removes ~10-15% of documents.
    """
    print(f"  Running exact deduplication...")

    # Stage 1: Compute MD5 hashes in parallel
    def compute_hash(example):
        hash_val = hashlib.md5(
            example["text"].encode('utf-8', errors='ignore')
        ).hexdigest()
        return {"md5_hash": hash_val}

    dataset_with_hash = dataset.map(
        compute_hash,
        num_proc=args.num_workers,
        desc=f"  {desc} - computing hashes"
    )

    # Stage 2: Deduplicate sequentially (fast - just set lookups)
    print(f"  {desc} - deduplicating...")
    seen_hashes = set()
    unique_indices = []

    for idx, example in enumerate(dataset_with_hash):
        if example["md5_hash"] not in seen_hashes:
            seen_hashes.add(example["md5_hash"])
            unique_indices.append(idx)

    # Select unique examples and remove temporary hash column
    dataset_deduped = dataset.select(unique_indices)

    stats = {
        "original_count": len(dataset),
        "unique_count": len(dataset_deduped),
        "duplicates_removed": len(dataset) - len(dataset_deduped),
        "duplicate_rate": 1 - (len(dataset_deduped) / len(dataset)) if len(dataset) > 0 else 0
    }

    print(f"  ✓ Removed {stats['duplicates_removed']:,} duplicates ({100*stats['duplicate_rate']:.1f}%)")
    print(f"    Kept {stats['unique_count']:,} unique documents")

    return dataset_deduped, stats


# =============================================================================
# Fuzzy Deduplication (MinHash + LSH)
# =============================================================================

def get_ngrams(text: str, n: int = 5) -> List[str]:
    """Extract n-grams from text for MinHash."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def fuzzy_deduplicate(
    dataset: Dataset,
    threshold: float = 0.8,
    num_perm: int = 256,
    desc: str = "Fuzzy dedup"
) -> Tuple[Dataset, Dict]:
    """Remove near-duplicates using MinHash + LSH.

    This is the most impactful preprocessing step, removing ~30-40% of documents.
    Uses Jaccard similarity of 5-gram sets.

    Args:
        threshold: Jaccard similarity threshold (0.8 = 80% similar)
        num_perm: Number of hash permutations for MinHash
    """
    if MinHash is None or MinHashLSH is None:
        print("  ⚠️  Skipping fuzzy deduplication (datasketch not installed)")
        return dataset, {"skipped": True}

    print(f"  Running fuzzy deduplication (threshold={threshold}, num_perm={num_perm})...")
    print(f"  This may take a while for large datasets...")

    start_time = time.time()

    # Stage 1: Compute MinHashes in parallel
    def compute_minhash(example):
        m = MinHash(num_perm=num_perm)
        for ngram in get_ngrams(example["text"], n=5):
            m.update(ngram.encode('utf-8'))
        # Serialize MinHash for storage
        return {"minhash_bytes": pickle.dumps(m)}

    print(f"  {desc} - computing MinHash signatures...")
    dataset_with_minhash = dataset.map(
        compute_minhash,
        num_proc=args.num_workers,
        desc=f"  {desc} - MinHash"
    )

    # Stage 2: LSH querying sequentially (must be sequential due to stateful index)
    print(f"  {desc} - LSH deduplication (sequential)...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = []

    for idx, example in enumerate(tqdm(dataset_with_minhash, desc=f"  {desc} - LSH")):
        # Deserialize MinHash
        m = pickle.loads(example["minhash_bytes"])

        # Check if similar document exists
        result = lsh.query(m)

        if not result:
            # No similar document found, this is unique
            lsh.insert(f"doc_{idx}", m)
            unique_indices.append(idx)

    elapsed_time = time.time() - start_time

    # Select unique documents
    dataset_deduped = dataset.select(unique_indices)

    stats = {
        "original_count": len(dataset),
        "unique_count": len(dataset_deduped),
        "near_duplicates_removed": len(dataset) - len(dataset_deduped),
        "near_duplicate_rate": 1 - (len(dataset_deduped) / len(dataset)) if len(dataset) > 0 else 0,
        "threshold": threshold,
        "num_perm": num_perm,
        "time_seconds": elapsed_time
    }

    print(f"  ✓ Removed {stats['near_duplicates_removed']:,} near-duplicates ({100*stats['near_duplicate_rate']:.1f}%)")
    print(f"    Kept {stats['unique_count']:,} unique documents")
    print(f"    Processing time: {elapsed_time/60:.1f} minutes")

    return dataset_deduped, stats


# =============================================================================
# Quality Filtering (Heuristic-based)
# =============================================================================

def check_quality(text: str) -> Tuple[bool, str]:
    """Check if document passes quality filters.

    Returns:
        (passes, reason) tuple
    """
    # Minimum length check
    if len(text) < 10:
        return False, "too_short"

    # Sentence check
    sentences = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return False, "too_few_sentences"

    # ALL CAPS check
    words = text.split()
    if len(words) == 0:
        return False, "no_words"

    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    caps_ratio = len(caps_words) / len(words)
    if caps_ratio > 0.3:
        return False, "too_many_caps"

    # Alphanumeric ratio check
    alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
    alnum_ratio = alphanumeric / len(text) if len(text) > 0 else 0
    if alnum_ratio < 0.25:
        return False, "too_few_alphanumeric"

    # Repetition check (2-grams)
    if len(words) > 10:
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        unique_ratio = len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0
        if unique_ratio < 0.7:  # More than 30% repetition
            return False, "too_repetitive_bigrams"

    # Repetition check (3-grams)
    if len(words) > 20:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if len(trigrams) > 0 else 0
        if unique_ratio < 0.8:  # More than 20% repetition
            return False, "too_repetitive_trigrams"

    return True, "passed"


def quality_filter(dataset: Dataset, desc: str = "Quality filter") -> Tuple[Dataset, Dict]:
    """Filter low-quality documents using heuristics."""
    print(f"  Running quality filtering...")

    # Add quality check results in parallel
    def add_quality_check(example):
        passes, reason = check_quality(example["text"])
        return {
            "passes_quality": passes,
            "rejection_reason": reason if not passes else ""
        }

    dataset_with_quality = dataset.map(
        add_quality_check,
        num_proc=args.num_workers,
        desc=f"  {desc} - checking quality"
    )

    # Collect rejection statistics
    rejection_reasons = Counter()
    for example in dataset_with_quality:
        if not example["passes_quality"] and example["rejection_reason"]:
            rejection_reasons[example["rejection_reason"]] += 1

    # Filter to keep only quality documents
    dataset_filtered = dataset_with_quality.filter(
        lambda ex: ex["passes_quality"],
        desc=f"  {desc} - filtering"
    )

    # Remove temporary columns
    dataset_filtered = dataset_filtered.remove_columns(["passes_quality", "rejection_reason"])

    stats = {
        "original_count": len(dataset),
        "passed_count": len(dataset_filtered),
        "filtered_count": len(dataset) - len(dataset_filtered),
        "filter_rate": 1 - (len(dataset_filtered) / len(dataset)) if len(dataset) > 0 else 0,
        "rejection_reasons": dict(rejection_reasons)
    }

    print(f"  ✓ Filtered {stats['filtered_count']:,} low-quality documents ({100*stats['filter_rate']:.1f}%)")
    print(f"    Kept {stats['passed_count']:,} quality documents")
    if rejection_reasons:
        print(f"    Top rejection reasons:")
        for reason, count in rejection_reasons.most_common(5):
            print(f"      - {reason}: {count:,}")

    return dataset_filtered, stats


# =============================================================================
# PII Removal (Regex-based)
# =============================================================================

# PII patterns
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b')
# API key patterns (common formats)
API_KEY_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9]{32,}\b'),  # Generic long alphanumeric
    re.compile(r'api[_\-]?key[_\-:=\s]+[A-Za-z0-9]+', re.IGNORECASE),
    re.compile(r'token[_\-:=\s]+[A-Za-z0-9]+', re.IGNORECASE),
]


def remove_pii(text: str) -> Tuple[str, int]:
    """Remove personally identifiable information from text.

    Returns:
        (cleaned_text, num_removed) tuple
    """
    num_removed = 0

    # Remove emails
    text, n = EMAIL_PATTERN.subn('[EMAIL]', text)
    num_removed += n

    # Remove IPs
    text, n = IP_PATTERN.subn('[IP]', text)
    num_removed += n

    # Remove phone numbers
    text, n = PHONE_PATTERN.subn('[PHONE]', text)
    num_removed += n

    # Remove potential API keys (be conservative to avoid false positives)
    for pattern in API_KEY_PATTERNS:
        # Only remove if it looks like it's in a key/token context
        if 'api' in text.lower() or 'key' in text.lower() or 'token' in text.lower():
            text, n = pattern.subn('[KEY]', text)
            num_removed += n

    return text, num_removed


def apply_pii_removal(dataset: Dataset, desc: str = "PII removal") -> Tuple[Dataset, Dict]:
    """Apply PII removal to all documents."""
    print(f"  Running PII removal...")

    def remove_pii_from_example(example):
        text, num_removed = remove_pii(example["text"])
        return {
            "text": text,
            "source": example["source"],
            "estimated_tokens": example["estimated_tokens"],
            "pii_count": num_removed  # Track per-document for proper multiprocessing
        }

    dataset_with_counts = dataset.map(
        remove_pii_from_example,
        desc=f"  {desc}",
        num_proc=args.num_workers
    )

    # Aggregate counts after parallel processing
    total_removed = sum(example["pii_count"] for example in dataset_with_counts)

    # Remove temporary column
    dataset_clean = dataset_with_counts.remove_columns(["pii_count"])

    stats = {
        "documents_processed": len(dataset),
        "pii_instances_removed": total_removed,
        "avg_pii_per_document": total_removed / len(dataset) if len(dataset) > 0 else 0
    }

    print(f"  ✓ Removed {total_removed:,} PII instances")
    print(f"    Avg: {stats['avg_pii_per_document']:.2f} per document")

    return dataset_clean, stats


# =============================================================================
# Benchmark Decontamination (N-gram overlap detection)
# =============================================================================

# Benchmark test sets to check for contamination
BENCHMARK_DATASETS = {
    "gsm8k_test": ("gsm8k", "main", "test"),
    "math_test": ("hendrycks_math", "all", "test"),
    "humaneval": ("openai_humaneval", None, "test"),
    "mbpp_test": ("mbpp", None, "test"),
    "arc_challenge_test": ("ai2_arc", "ARC-Challenge", "test"),
    "hellaswag_test": ("hellaswag", None, "test"),
    "mmlu_test": ("cais/mmlu", "all", "test"),
    "winogrande_test": ("winogrande", "winogrande_xl", "test"),
}


def normalize_text(text: str) -> str:
    """Normalize text for contamination checking: lowercase, remove extra whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_ngrams_list(text: str, n: int = 13) -> Set[str]:
    """Get n-grams for contamination detection.

    Using 13-grams as recommended in the literature.
    """
    text = normalize_text(text)
    words = text.split()
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))


def load_benchmark_ngrams(n: int = 13) -> Dict[str, Set[str]]:
    """Load all benchmark test sets and extract n-grams.

    This is slow but only done once at start of processing.
    """
    print(f"\n📚 Loading benchmark test sets for decontamination...")

    all_ngrams = {}

    for name, (dataset_name, config, split) in BENCHMARK_DATASETS.items():
        try:
            print(f"  Loading {name}...")

            # Load dataset
            if config:
                ds = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

            # Extract all text fields and compute n-grams
            ngrams = set()
            for example in ds:
                # Concatenate all text fields in the example
                text_parts = []
                for value in example.values():
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, list):
                        text_parts.extend([str(v) for v in value if isinstance(v, str)])

                text = ' '.join(text_parts)
                ngrams.update(get_ngrams_list(text, n))

            all_ngrams[name] = ngrams
            print(f"    ✓ {len(ngrams):,} {n}-grams from {len(ds):,} examples")

        except Exception as e:
            print(f"    ✗ Error loading {name}: {e}")
            all_ngrams[name] = set()

    total_ngrams = sum(len(ngrams) for ngrams in all_ngrams.values())
    print(f"\n  Total: {total_ngrams:,} unique {n}-grams from {len(all_ngrams)} benchmarks")

    return all_ngrams


def check_contamination(text: str, benchmark_ngrams: Dict[str, Set[str]], n: int = 13, threshold: float = 0.1) -> Tuple[bool, List[str]]:
    """Check if document is contaminated with benchmark data.

    A document is considered contaminated if >10% of its n-grams appear in any test set.

    Returns:
        (is_contaminated, contaminated_benchmarks) tuple
    """
    doc_ngrams = get_ngrams_list(text, n)

    if len(doc_ngrams) == 0:
        return False, []

    contaminated = []

    for name, test_ngrams in benchmark_ngrams.items():
        if len(test_ngrams) == 0:
            continue

        # Count overlapping n-grams
        overlap = len(doc_ngrams & test_ngrams)
        overlap_ratio = overlap / len(doc_ngrams)

        if overlap_ratio > threshold:
            contaminated.append(name)

    return len(contaminated) > 0, contaminated


def decontaminate(dataset: Dataset, desc: str = "Decontamination") -> Tuple[Dataset, Dict]:
    """Remove documents contaminated with benchmark test sets."""
    print(f"  Running benchmark decontamination...")

    # Load benchmark n-grams
    benchmark_ngrams = load_benchmark_ngrams(n=13)

    # Check contamination in parallel
    def check_contamination_detailed(example):
        is_contaminated, benchmarks = check_contamination(
            example["text"],
            benchmark_ngrams
        )
        return {
            "is_clean": not is_contaminated,
            "contaminated_benchmarks": benchmarks
        }

    dataset_with_check = dataset.map(
        check_contamination_detailed,
        num_proc=args.num_workers,
        desc=f"  {desc} - checking contamination"
    )

    # Collect contamination statistics
    contaminated_counts = Counter()
    for example in dataset_with_check:
        if not example["is_clean"]:
            for bench in example["contaminated_benchmarks"]:
                contaminated_counts[bench] += 1

    # Filter to keep only clean documents
    dataset_clean = dataset_with_check.filter(
        lambda ex: ex["is_clean"],
        desc=f"  {desc} - filtering"
    )

    # Remove temporary columns
    dataset_clean = dataset_clean.remove_columns(["is_clean", "contaminated_benchmarks"])

    stats = {
        "original_count": len(dataset),
        "clean_count": len(dataset_clean),
        "contaminated_count": len(dataset) - len(dataset_clean),
        "contamination_rate": 1 - (len(dataset_clean) / len(dataset)) if len(dataset) > 0 else 0,
        "contaminated_by_benchmark": dict(contaminated_counts)
    }

    print(f"  ✓ Removed {stats['contaminated_count']:,} contaminated documents ({100*stats['contamination_rate']:.2f}%)")
    print(f"    Kept {stats['clean_count']:,} clean documents")
    if contaminated_counts:
        print(f"    Contamination by benchmark:")
        for bench, count in contaminated_counts.most_common():
            print(f"      - {bench}: {count:,}")

    return dataset_clean, stats


# =============================================================================
# Dataset Loading
# =============================================================================

def load_raw_dataset(dataset_dir: Path) -> Optional[Dataset]:
    """Load a raw dataset from parquet files."""
    if not dataset_dir.exists():
        return None

    try:
        parquet_files = sorted(dataset_dir.glob("data-*.parquet"))
        if not parquet_files:
            return None

        # Load directly from parquet (faster than load_dataset)
        if len(parquet_files) == 1:
            dataset = Dataset.from_parquet(str(parquet_files[0]))
        else:
            # For multiple files, load and concatenate
            datasets_list = [Dataset.from_parquet(str(f)) for f in parquet_files]
            dataset = concatenate_datasets(datasets_list)

        return dataset

    except Exception as e:
        print(f"  ✗ Error loading {dataset_dir.name}: {e}")
        return None


def prepare_dataset(dataset: Dataset, dataset_name: str) -> Dataset:
    """Normalize raw dataset and add required columns (estimated_tokens, source).

    Args:
        dataset: Raw dataset from load_raw_dataset()
        dataset_name: Name of the dataset (for source column)

    Returns:
        Dataset with estimated_tokens and source columns
    """
    print(f"  Preparing dataset (adding token estimates)...")

    # Ensure source column exists
    if "source" not in dataset.column_names:
        dataset = dataset.add_column("source", [dataset_name] * len(dataset))

    # Add estimated_tokens column
    if "estimated_tokens" not in dataset.column_names:
        if args.tokenizer_path:
            # Use actual tokenizer for accurate counts
            print(f"    Using tokenizer: {args.tokenizer_path}")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

            if args.max_seq_length:
                # Truncate texts to max_seq_length tokens
                print(f"    Truncating texts to {args.max_seq_length} tokens")

                def tokenize_and_truncate(example):
                    # Tokenize
                    tokens = tokenizer.encode(example["text"], add_special_tokens=False)

                    # Truncate if needed
                    if len(tokens) > args.max_seq_length:
                        tokens = tokens[:args.max_seq_length]
                        # Decode back to text
                        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                    else:
                        truncated_text = example["text"]

                    return {
                        "text": truncated_text,
                        "estimated_tokens": len(tokens)
                    }

                dataset = dataset.map(
                    tokenize_and_truncate,
                    num_proc=args.num_workers,
                    desc="    Tokenizing & truncating"
                )
            else:
                # Just count tokens without truncation
                def count_tokens_accurate(example):
                    tokens = tokenizer.encode(example["text"], add_special_tokens=False)
                    return {"estimated_tokens": len(tokens)}

                dataset = dataset.map(
                    count_tokens_accurate,
                    num_proc=args.num_workers,  # Parallelize tokenization!
                    desc="    Tokenizing"
                )
        else:
            # Use char/4 estimation (fast)
            print(f"    Using char/4 estimation (use --tokenizer_path for accurate counts)")

            def estimate_tokens(example):
                return {"estimated_tokens": len(example.get("text", "")) // 4}

            dataset = dataset.map(
                estimate_tokens,
                num_proc=args.num_workers,  # Still parallelize even for estimation
                desc="    Estimating tokens"
            )

    return dataset


# =============================================================================
# Mixture Creation
# =============================================================================

def save_dataset_separately(
    dataset: Dataset,
    dataset_name: str,
    output_path: Path,
    stage_name: str
) -> Dict:
    """Save a single dataset to its own directory.

    Args:
        dataset: Dataset to save
        dataset_name: Name of the dataset
        output_path: Base output path (e.g., output_dir/stage1)
        stage_name: "stage1" or "stage2"

    Returns:
        Stats dict with token counts and document counts
    """
    dataset_dir = output_path / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  💾 Saving {dataset_name}...")

    # Calculate tokens
    total_tokens = sum(example["estimated_tokens"] for example in dataset)
    print(f"    {len(dataset):,} documents, {total_tokens/1e9:.2f}B tokens")

    # Save to parquet shards
    batch_size = args.shard_size
    shard_num = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        table = pa.Table.from_pylist([{
            "text": ex["text"],
            "source": ex["source"],
            "estimated_tokens": ex["estimated_tokens"]
        } for ex in batch])
        pq.write_table(table, dataset_dir / f"data-{shard_num:05d}.parquet")
        shard_num += 1

    print(f"    ✓ Saved to {dataset_dir} ({shard_num} shards)")

    return {
        "stage": stage_name,
        "dataset_name": dataset_name,
        "tokens": total_tokens,
        "documents": len(dataset),
        "num_shards": shard_num
    }


def create_mixture(
    datasets: Dict[str, Dataset],
    target_tokens: int,
    output_path: Path,
    stage_name: str
) -> Dict:
    """Create mixture from multiple datasets with target token budget.

    Args:
        datasets: Dict mapping dataset name to Dataset
        target_tokens: Target total tokens for mixture
        output_path: Where to save mixture
        stage_name: "stage1" or "stage2"
    """
    print(f"\n🔗 Creating {stage_name} mixture (target: {target_tokens/1e9:.2f}B tokens)...")

    # Calculate current token counts
    dataset_tokens = {}
    for name, ds in datasets.items():
        tokens = sum(example["estimated_tokens"] for example in ds)
        dataset_tokens[name] = tokens
        print(f"  {name}: {len(ds):,} docs, {tokens/1e9:.2f}B tokens")

    total_tokens = sum(dataset_tokens.values())
    print(f"  Total available: {total_tokens/1e9:.2f}B tokens")

    if total_tokens < target_tokens:
        print(f"  ⚠️  Warning: Available tokens ({total_tokens/1e9:.2f}B) < target ({target_tokens/1e9:.2f}B)")
        print(f"     This may be due to preprocessing losses. Using all available data.")

    # Combine all datasets
    print(f"\n  Combining datasets...")
    combined = concatenate_datasets(list(datasets.values()))

    # Shuffle
    print(f"  Shuffling...")
    combined = combined.shuffle(seed=RANDOM_SEED)

    # Trim to target token count if we have excess
    if total_tokens > target_tokens:
        print(f"  Trimming to target token count...")
        current_tokens = 0
        trim_index = 0

        for i, example in enumerate(combined):
            current_tokens += example["estimated_tokens"]
            if current_tokens >= target_tokens:
                trim_index = i + 1
                break

        combined = combined.select(range(trim_index))
        final_tokens = sum(example["estimated_tokens"] for example in combined)
        print(f"    Trimmed to {len(combined):,} documents ({final_tokens/1e9:.2f}B tokens)")
    else:
        final_tokens = total_tokens

    # Save to parquet shards
    print(f"\n  Saving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = args.shard_size
    shard_num = 0
    for i in range(0, len(combined), batch_size):
        batch = combined.select(range(i, min(i + batch_size, len(combined))))
        table = pa.Table.from_pylist([{
            "text": ex["text"],
            "source": ex["source"],
            "estimated_tokens": ex["estimated_tokens"]
        } for ex in batch])
        pq.write_table(table, output_path / f"data-{shard_num:05d}.parquet")
        shard_num += 1

    stats = {
        "stage": stage_name,
        "target_tokens": target_tokens,
        "final_tokens": final_tokens,
        "final_documents": len(combined),
        "num_shards": shard_num,
        "dataset_contributions": {name: {"tokens": dataset_tokens[name], "documents": len(ds)} for name, ds in datasets.items()}
    }

    print(f"  ✓ Saved {len(combined):,} documents in {shard_num} shards")
    print(f"    Final tokens: {final_tokens/1e9:.2f}B")

    return stats


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    global args

    # Validate arguments
    if args.max_seq_length and not args.tokenizer_path:
        print("ERROR: --max_seq_length requires --tokenizer_path")
        print("       You must provide a tokenizer to truncate texts to a specific token length.")
        sys.exit(1)

    # Calculate target token budgets
    target_tokens = calculate_target_tokens(args.model_size)

    # Display plan
    print("=" * 80)
    print("Pretraining Dataset Processing Plan")
    print("=" * 80)
    print(f"\nModel size: {args.model_size}")
    print(f"Stages: {args.stages}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    print(f"\nTarget token counts (after preprocessing):")
    if args.stages in ["stage1", "both"]:
        print(f"  Stage 1: {target_tokens['stage1']/1e9:.2f}B tokens")
    if args.stages in ["stage2", "both"]:
        print(f"  Stage 2: {target_tokens['stage2']/1e9:.2f}B tokens")

    print(f"\nPreprocessing pipeline:")
    print(f"  1. Exact dedup:          {'✓ Enabled' if not args.skip_exact_dedup else '✗ Skipped'}")
    print(f"  2. Fuzzy dedup:          {'✓ Enabled' if not args.skip_fuzzy_dedup else '✗ Skipped'}")
    print(f"  3. Quality filter:       {'✓ Enabled' if not args.skip_quality_filter else '✗ Skipped'}")
    print(f"  4. PII removal:          {'✓ Enabled' if not args.skip_pii_removal else '✗ Skipped'}")
    print(f"  5. Decontamination:      {'✓ Enabled' if not args.skip_decontamination else '✗ Skipped'}")

    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without processing.")
        return

    print("\n" + "=" * 80)
    print("Starting processing...")
    print("=" * 80)

    # Track statistics
    all_stats = {
        "stage1": {},
        "stage2": {},
        "merged": {}
    }

    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)

    # Process Stage all
    if args.stages in ["merged"]:
        print("\n" + "="*80)
        print("STAGE merged: Broad Pretraining + Finer Pretraining")
        print("="*80)

        merged_dir = input_base  # Look directly in input_dir
        merged_datasets = {}

        # List of Stage 1 datasets
        dataset_names = [
            "algebraic_stack",
            "arxiv",
            "books_gutenberg",
            "fineweb_edu",
            "github_code_clean_cpp",
            "github_code_clean_go",
            "github_code_clean_html",
            "github_code_clean_java",
            "github_code_clean_javascript",
            "github_code_clean_python",
            "github_code_clean_rust",
            "github_code_clean_shell",
            "github_code_clean_sql",
            "github_code_clean_typescript",
            "gsm8k",
            "openwebmath",
            "peso",
            "tinygsm",
            "wikipedia",
        ]

        for name in dataset_names:
            print(f"\n{'─'*80}")
            print(f"Processing {name}...")
            print(f"{'─'*80}")

            dataset = load_raw_dataset(merged_dir / name)
            if dataset is None:
                print(f"  ✗ Dataset not found, skipping")
                continue

            print(f"  Loaded {len(dataset):,} documents")

            # Prepare dataset (add estimated_tokens and source columns)
            dataset = prepare_dataset(dataset, name)

            dataset_stats = {}

            # Apply preprocessing pipeline
            if not args.skip_exact_dedup:
                dataset, stats = exact_deduplicate(dataset, f"{name} exact dedup")
                dataset_stats["exact_dedup"] = stats

            if not args.skip_fuzzy_dedup:
                dataset, stats = fuzzy_deduplicate(dataset, args.fuzzy_threshold, args.minhash_num_perm, f"{name} fuzzy dedup")
                dataset_stats["fuzzy_dedup"] = stats

            if not args.skip_quality_filter:
                dataset, stats = quality_filter(dataset, f"{name} quality")
                dataset_stats["quality_filter"] = stats

            if not args.skip_pii_removal:
                dataset, stats = apply_pii_removal(dataset, f"{name} PII")
                dataset_stats["pii_removal"] = stats

            if not args.skip_decontamination:
                dataset, stats = decontaminate(dataset, f"{name} decontam")
                dataset_stats["decontamination"] = stats

            print(f"  ✓ Final: {len(dataset):,} documents")

            # Save dataset separately
            save_stats = save_dataset_separately(
                dataset,
                name,
                output_base / "merged",
                "merged"
            )

            # Combine preprocessing stats with save stats
            dataset_stats["save"] = save_stats
            all_stats["merged"][name] = dataset_stats

    # Process Stage 1
    if args.stages in ["stage1", "both"]:
        print("\n" + "="*80)
        print("STAGE 1: Broad Pretraining")
        print("="*80)

        stage1_dir = input_base  # Look directly in input_dir
        stage1_datasets = {}

        # List of Stage 1 datasets
        dataset_names = [
            "fineweb_edu",
            "wikipedia",
            "books_gutenberg",
            "github_code_clean_python",
            "github_code_clean_javascript",
            "github_code_clean_typescript",
            "github_code_clean_java",
            "github_code_clean_cpp",
            "github_code_clean_go",
            "github_code_clean_rust",
            "github_code_clean_shell",
            "github_code_clean_sql",
            "github_code_clean_html",
            "peso",
            "arxiv",
            "openwebmath"
        ]

        for name in dataset_names:
            print(f"\n{'─'*80}")
            print(f"Processing {name}...")
            print(f"{'─'*80}")

            dataset = load_raw_dataset(stage1_dir / name)
            if dataset is None:
                print(f"  ✗ Dataset not found, skipping")
                continue

            print(f"  Loaded {len(dataset):,} documents")

            # Prepare dataset (add estimated_tokens and source columns)
            dataset = prepare_dataset(dataset, name)

            dataset_stats = {}

            # Apply preprocessing pipeline
            if not args.skip_exact_dedup:
                dataset, stats = exact_deduplicate(dataset, f"{name} exact dedup")
                dataset_stats["exact_dedup"] = stats

            if not args.skip_fuzzy_dedup:
                dataset, stats = fuzzy_deduplicate(dataset, args.fuzzy_threshold, args.minhash_num_perm, f"{name} fuzzy dedup")
                dataset_stats["fuzzy_dedup"] = stats

            if not args.skip_quality_filter:
                dataset, stats = quality_filter(dataset, f"{name} quality")
                dataset_stats["quality_filter"] = stats

            if not args.skip_pii_removal:
                dataset, stats = apply_pii_removal(dataset, f"{name} PII")
                dataset_stats["pii_removal"] = stats

            if not args.skip_decontamination:
                dataset, stats = decontaminate(dataset, f"{name} decontam")
                dataset_stats["decontamination"] = stats

            print(f"  ✓ Final: {len(dataset):,} documents")

            # Save dataset separately
            save_stats = save_dataset_separately(
                dataset,
                name,
                output_base / "stage1",
                "stage1"
            )

            # Combine preprocessing stats with save stats
            dataset_stats["save"] = save_stats
            all_stats["stage1"][name] = dataset_stats

    # Process Stage 2
    if args.stages in ["stage2", "both"]:
        print("\n" + "="*80)
        print("STAGE 2: Domain Upsampling")
        print("="*80)

        stage2_dir = input_base  # Look directly in input_dir
        stage2_datasets = {}

        # List of Stage 2 datasets
        dataset_names = [
            "fineweb_edu",  # Will use different samples than Stage 1
            "github_code_clean_python",
            "github_code_clean_javascript",
            "github_code_clean_typescript",
            "github_code_clean_java",
            "github_code_clean_cpp",
            "github_code_clean_go",
            "github_code_clean_rust",
            "github_code_clean_shell",
            "github_code_clean_sql",
            "github_code_clean_html",
            "openwebmath",
            "tinygsm",
            "algebraic_stack",
            "gsm8k",  # Changed from gsm8k_train
            "peso",
            "arxiv"
        ]

        for name in dataset_names:
            print(f"\n{'─'*80}")
            print(f"Processing {name}...")
            print(f"{'─'*80}")

            dataset = load_raw_dataset(stage2_dir / name)
            if dataset is None:
                print(f"  ✗ Dataset not found, skipping")
                continue

            print(f"  Loaded {len(dataset):,} documents")

            # Prepare dataset (add estimated_tokens and source columns)
            dataset = prepare_dataset(dataset, name)

            dataset_stats = {}

            # Apply preprocessing pipeline
            if not args.skip_exact_dedup:
                dataset, stats = exact_deduplicate(dataset, f"{name} exact dedup")
                dataset_stats["exact_dedup"] = stats

            if not args.skip_fuzzy_dedup:
                dataset, stats = fuzzy_deduplicate(dataset, args.fuzzy_threshold, args.minhash_num_perm, f"{name} fuzzy dedup")
                dataset_stats["fuzzy_dedup"] = stats

            if not args.skip_quality_filter:
                dataset, stats = quality_filter(dataset, f"{name} quality")
                dataset_stats["quality_filter"] = stats

            if not args.skip_pii_removal:
                dataset, stats = apply_pii_removal(dataset, f"{name} PII")
                dataset_stats["pii_removal"] = stats

            if not args.skip_decontamination:
                dataset, stats = decontaminate(dataset, f"{name} decontam")
                dataset_stats["decontamination"] = stats

            print(f"  ✓ Final: {len(dataset):,} documents")

            # Save dataset separately
            save_stats = save_dataset_separately(
                dataset,
                name,
                output_base / "stage2",
                "stage2"
            )

            # Combine preprocessing stats with save stats
            dataset_stats["save"] = save_stats
            all_stats["stage2"][name] = dataset_stats

    # Save comprehensive statistics
    print("\n" + "=" * 80)
    print("Saving statistics...")
    print("=" * 80)

    # Ensure output directory exists
    output_base.mkdir(parents=True, exist_ok=True)

    stats_file = output_base / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "model_size": args.model_size,
            "stages": args.stages,
            "target_tokens": target_tokens,
            "preprocessing_config": {
                "exact_dedup": not args.skip_exact_dedup,
                "fuzzy_dedup": not args.skip_fuzzy_dedup,
                "fuzzy_threshold": args.fuzzy_threshold,
                "minhash_num_perm": args.minhash_num_perm,
                "quality_filter": not args.skip_quality_filter,
                "pii_removal": not args.skip_pii_removal,
                "decontamination": not args.skip_decontamination,
            },
            "statistics": all_stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "random_seed": RANDOM_SEED
        }, f, indent=2)

    print(f"  ✓ Saved to: {stats_file}")

    # Create verification samples
    print("\n" + "=" * 80)
    print("Creating verification samples...")
    print("=" * 80)

    verification_file = output_base / "verification_samples.txt"
    with open(verification_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Pretraining Dataset Verification Samples\n")
        f.write("=" * 80 + "\n\n")

        for stage in ["stage1", "stage2", "merged"]:
            stage_dir = output_base / stage
            if not stage_dir.exists():
                continue

            f.write(f"\n{'='*80}\n")
            f.write(f"{stage.upper()}\n")
            f.write(f"{'='*80}\n\n")

            # Load first shard
            first_shard = stage_dir / "data-00000.parquet"
            if first_shard.exists():
                ds = load_dataset("parquet", data_files=str(first_shard), split="train")

                # Show 3 samples
                for i in range(min(3, len(ds))):
                    example = ds[i]
                    f.write(f"--- Sample {i+1} ---\n\n")
                    f.write(f"Source: {example['source']}\n")
                    f.write(f"Estimated tokens: {example['estimated_tokens']:,}\n\n")
                    f.write(f"Text (first 500 chars):\n")
                    f.write(example['text'][:500] + "...\n\n")
                    f.write("-" * 80 + "\n\n")

    print(f"  ✓ Saved to: {verification_file}")

    print("\n" + "=" * 80)
    print("✅ Processing complete!")
    print("=" * 80)
    print(f"\nOutput saved to: {output_base}")
    print(f"  - Stage 1: {output_base}/stage1/")
    print(f"  - Stage 2: {output_base}/stage2/")
    print(f"  - Stage 2: {output_base}/merged/")
    print(f"  - Statistics: {stats_file}")
    print(f"  - Verification: {verification_file}")
    print(f"\nTo use in training, update your config YAML:")
    print(f"  data_config:")
    print(f"    train_data:")
    print(f"      - type: hfds")
    print(f"        prefix: pretrain-stage1")
    print(f"        data_dir: {output_base}/stage1")
    print(f"      - type: hfds")
    print(f"        prefix: pretrain-stage2")
    print(f"        data_dir: {output_base}/stage2")
    print()


if __name__ == "__main__":
    main()
