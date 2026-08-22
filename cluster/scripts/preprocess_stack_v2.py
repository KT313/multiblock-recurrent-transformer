#!/usr/bin/env python3
"""
Stack v2 Dataset Content Downloader and Preprocessor

The Stack v2 datasets on HuggingFace contain only metadata, not file content.
This script downloads the actual code content from GitHub using the repository
metadata and preprocesses it in the same format as preprocess_datasets.py.

Output schema:
- text: str (downloaded code content)
- source: str (dataset name, e.g., 'stack_v2_python')
- original_length: int (original character count before truncation)
- url: str (GitHub raw URL for verification)

Usage:
    # Full download
    python preprocess_stack_v2.py \
        --input_dir /path/to/raw \
        --output_dir /path/to/filtered \
        --num_workers 8

    # Test mode (1000 samples per dataset)
    python preprocess_stack_v2.py \
        --input_dir /path/to/raw \
        --output_dir /path/to/filtered_test \
        --max_samples 1000 \
        --num_workers 4

Author: Claude Code
Date: 2025-01-27
"""

import argparse
import json
import time
import threading
import requests
from multiprocessing import Pool, Manager, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

# Stack v2 dataset names
STACK_V2_DATASETS = [
    "stack_v2_python",
    "stack_v2_javascript",
    "stack_v2_typescript",
    "stack_v2_java",
    "stack_v2_cpp",
    "stack_v2_go",
    "stack_v2_rust",
    "stack_v2_shell",
    "stack_v2_sql",
    "stack_v2_html",
]


# =============================================================================
# Content Download
# =============================================================================

def download_content_from_github(
    repo_name: str,
    revision_id: str,
    path: str,
    max_retries: int = 3
) -> Tuple[Optional[str], str, str]:
    """Download file content from GitHub.

    Args:
        repo_name: Repository name (e.g., 'user/repo')
        revision_id: Git commit hash
        path: File path (starts with '/')
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (file_content, error_type, url)
        error_type can be: 'success', '404', 'timeout', 'encoding', 'network'
        url is the GitHub raw URL
    """
    # Construct GitHub raw URL
    # https://raw.githubusercontent.com/user/repo/commit_hash/path/to/file
    url = f"{GITHUB_RAW_BASE}/{repo_name}/{revision_id}{path}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Try to decode as UTF-8
                try:
                    return response.content.decode('utf-8'), 'success', url
                except UnicodeDecodeError:
                    # Try latin-1 as fallback
                    try:
                        return response.content.decode('latin-1'), 'success', url
                    except:
                        return None, 'encoding', url

            elif response.status_code == 404:
                # File/repo not found (deleted, private, or moved)
                return None, '404', url

            elif response.status_code == 429:
                # Rate limited, wait and retry
                time.sleep(2 ** attempt)
                continue

            else:
                # Other HTTP error
                return None, 'network', url

        except requests.exceptions.Timeout:
            # Timeout
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, 'timeout', url

        except requests.exceptions.RequestException:
            # Other network error, retry
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, 'network', url

    return None, 'network', url


# =============================================================================
# Batch Processing Worker
# =============================================================================

def process_batch_worker(args: Tuple) -> Tuple[List[Dict], Dict]:
    """Worker function to download and process a batch of samples.

    Args:
        args: Tuple of (batch_metadata, source_name, min_chars, max_chars, shared_stats)

    Returns:
        Tuple of (processed_samples_list, statistics_dict)
    """
    batch_metadata, source_name, min_chars, max_chars, shared_stats = args

    stats = {
        'input_samples': len(batch_metadata),
        'output_samples': 0,
        'error_404': 0,
        'error_encoding': 0,
        'error_timeout': 0,
        'error_network': 0,
        'too_short': 0,
        'truncated': 0
    }

    processed_samples = []

    for sample in batch_metadata:
        repo_name = sample['repo_name']
        revision_id = sample['revision_id']
        path = sample['path']

        # Increment "in progress" counter
        shared_stats['in_progress'] += 1

        # Download content from GitHub
        content, error_type, url = download_content_from_github(repo_name, revision_id, path)

        # Decrement "in progress" counter
        shared_stats['in_progress'] -= 1

        if content is None:
            # Track error type
            if error_type == '404':
                stats['error_404'] += 1
                shared_stats['failed'] += 1
            elif error_type == 'encoding':
                stats['error_encoding'] += 1
                shared_stats['failed'] += 1
            elif error_type == 'timeout':
                stats['error_timeout'] += 1
                shared_stats['failed'] += 1
            else:  # 'network'
                stats['error_network'] += 1
                shared_stats['failed'] += 1
            continue

        # Filter: minimum length
        if len(content) < min_chars:
            stats['too_short'] += 1
            shared_stats['failed'] += 1
            continue

        # Record original length
        original_length = len(content)

        # Truncate if needed
        if len(content) > max_chars:
            content = content[:max_chars]
            stats['truncated'] += 1

        # Add to output
        processed_samples.append({
            'text': content,
            'source': source_name,
            'original_length': original_length,
            'url': url
        })
        stats['output_samples'] += 1
        shared_stats['completed'] += 1

    return processed_samples, stats


# =============================================================================
# Dataset Processing
# =============================================================================

def process_stack_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    min_chars: int,
    max_chars: int,
    batch_size: int,
    shard_size: int,
    num_workers: int,
    max_samples: Optional[int] = None
) -> Dict[str, any]:
    """Process a single Stack v2 dataset.

    Returns:
        Statistics dictionary
    """
    dataset_input_dir = input_dir / dataset_name
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📦 Processing: {dataset_name}")
    print(f"{'='*80}")
    print(f"  Input:  {dataset_input_dir}")
    print(f"  Output: {dataset_output_dir}")

    # Find all parquet shards
    parquet_files = sorted(dataset_input_dir.glob("shard-*.parquet"))

    if not parquet_files:
        print(f"  ⚠️  No parquet files found, skipping")
        return None

    print(f"  Found {len(parquet_files)} input shards")

    # Aggregate statistics
    total_stats = {
        'input_samples': 0,
        'output_samples': 0,
        'error_404': 0,
        'error_encoding': 0,
        'error_timeout': 0,
        'error_network': 0,
        'too_short': 0,
        'truncated': 0
    }

    # Output tracking
    output_buffer = []
    output_shard_num = 0

    # Create shared statistics (for real-time updates)
    manager = Manager()
    shared_stats = manager.dict({
        'completed': 0,
        'failed': 0,
        'in_progress': 0
    })

    # Collect all samples from all parquet files
    print(f"  Loading metadata for content download...")
    if max_samples:
        print(f"  ⚠️  Test mode: Limiting to {max_samples:,} samples")

    all_work_items = []
    samples_collected = 0

    for parquet_file in tqdm(parquet_files, desc="  📖 Reading metadata", unit=" files"):
        pf = pq.ParquetFile(parquet_file)

        for batch in pf.iter_batches(batch_size=batch_size):
            # Check if we've hit the limit
            if max_samples and samples_collected >= max_samples:
                break

            # Extract metadata we need (repo_name, revision_id, path)
            batch_metadata = []
            for i in range(len(batch)):
                # Stop if we hit max_samples
                if max_samples and samples_collected >= max_samples:
                    break

                batch_metadata.append({
                    'repo_name': batch['repo_name'][i].as_py(),
                    'revision_id': batch['revision_id'][i].as_py(),
                    'path': batch['path'][i].as_py()
                })
                samples_collected += 1

            if batch_metadata:  # Only add if not empty
                work_item = (batch_metadata, dataset_name, min_chars, max_chars, shared_stats)
                all_work_items.append(work_item)

        # Break outer loop if limit reached
        if max_samples and samples_collected >= max_samples:
            break

    print(f"  Total batches to process: {len(all_work_items)}")
    print(f"  Total samples to download: {sum(len(item[0]) for item in all_work_items):,}")
    print(f"  Downloading content from GitHub with {num_workers} workers...")

    # Process batches in parallel
    pbar = tqdm(
        total=len(all_work_items),
        desc=f"  ⬇️  Downloading",
        unit=" batches",
        miniters=1,
        mininterval=1.0  # Update every second
    )

    with Pool(processes=num_workers) as pool:
        # Use imap to maintain order
        for processed_samples, batch_stats in pool.imap(process_batch_worker, all_work_items, chunksize=1):
            # Update statistics
            total_stats['input_samples'] += batch_stats['input_samples']
            total_stats['output_samples'] += batch_stats['output_samples']
            total_stats['error_404'] += batch_stats['error_404']
            total_stats['error_encoding'] += batch_stats['error_encoding']
            total_stats['error_timeout'] += batch_stats['error_timeout']
            total_stats['error_network'] += batch_stats['error_network']
            total_stats['too_short'] += batch_stats['too_short']
            total_stats['truncated'] += batch_stats['truncated']

            # Add to output buffer
            if processed_samples:
                # Convert to RecordBatch
                processed_batch = pa.RecordBatch.from_pylist(processed_samples)
                output_buffer.append(processed_batch)

            # Save output shard if buffer is large enough
            while sum(len(b) for b in output_buffer) >= shard_size:
                # Combine batches until we have shard_size samples
                shard_batches = []
                shard_samples = 0

                while output_buffer and shard_samples < shard_size:
                    next_batch = output_buffer.pop(0)
                    shard_batches.append(next_batch)
                    shard_samples += len(next_batch)

                # If we exceeded shard_size, split the last batch
                if shard_samples > shard_size:
                    last_batch = shard_batches.pop()
                    overflow = shard_samples - shard_size

                    # Split batch
                    keep_batch = last_batch.slice(0, len(last_batch) - overflow)
                    overflow_batch = last_batch.slice(len(last_batch) - overflow)

                    shard_batches.append(keep_batch)
                    output_buffer.insert(0, overflow_batch)

                # Combine and save
                shard_table = pa.Table.from_batches(shard_batches)
                output_file = dataset_output_dir / f"data-{output_shard_num:05d}.parquet"
                pq.write_table(shard_table, output_file)
                output_shard_num += 1

            # Update progress bar with live stats
            pbar.update(1)
            pbar.set_postfix({
                'completed': f"{shared_stats['completed']:,}",
                'failed': f"{shared_stats['failed']:,}",
                'in_progress': f"{shared_stats['in_progress']}"
            })

    pbar.close()

    # Save remaining samples in buffer
    if output_buffer:
        shard_table = pa.Table.from_batches(output_buffer)
        output_file = dataset_output_dir / f"data-{output_shard_num:05d}.parquet"
        pq.write_table(shard_table, output_file)
        output_shard_num += 1

    # Calculate final statistics
    if total_stats['input_samples'] > 0:
        total_stats['success_rate'] = total_stats['output_samples'] / total_stats['input_samples']
    else:
        total_stats['success_rate'] = 0.0

    print(f"  ✓ Complete:")
    print(f"      Input:  {total_stats['input_samples']:,} samples")
    print(f"      Output: {total_stats['output_samples']:,} samples ({total_stats['success_rate']:.1%} success rate)")
    print(f"      Errors:")
    print(f"        - 404 (not found):  {total_stats['error_404']:,}")
    print(f"        - Encoding:         {total_stats['error_encoding']:,}")
    print(f"        - Timeout:          {total_stats['error_timeout']:,}")
    print(f"        - Network:          {total_stats['error_network']:,}")
    print(f"      Too short: {total_stats['too_short']:,}")
    print(f"      Truncated: {total_stats['truncated']:,}")
    print(f"      Output shards: {output_shard_num}")

    return total_stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Stack v2 datasets")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with raw Stack v2 datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for filtered datasets")
    parser.add_argument("--min_chars", type=int, default=50,
                       help="Minimum text length in characters (default: 50)")
    parser.add_argument("--max_chars", type=int, default=20000,
                       help="Maximum text length before truncation (default: 20000 = ~5000 tokens)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Samples to process at once (default: 100, lower for network-bound tasks)")
    parser.add_argument("--shard_size", type=int, default=100000,
                       help="Samples per output parquet file (default: 100000)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of worker processes (default: CPU count)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per dataset (for testing, default: all)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                       help="Specific Stack v2 datasets to process (default: all)")

    args = parser.parse_args()

    # Set default num_workers to CPU count
    if args.num_workers is None:
        args.num_workers = min(8, cpu_count())  # Cap at 8 to avoid overwhelming SWH
        print(f"Using {args.num_workers} worker processes")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = [d for d in args.datasets if d in STACK_V2_DATASETS]
    else:
        # Find all Stack v2 datasets in input directory
        datasets_to_process = [d for d in STACK_V2_DATASETS if (input_dir / d).exists()]

    if not datasets_to_process:
        print(f"❌ Error: No Stack v2 datasets found in {input_dir}")
        return

    print("="*80)
    print("Stack v2 Content Download and Preprocessing (PARALLEL)")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.num_workers} processes")
    if args.max_samples:
        print(f"⚠️  TEST MODE: Max {args.max_samples:,} samples per dataset")
    print(f"Filters:")
    print(f"  - Minimum length: {args.min_chars} chars")
    print(f"  - Maximum length: {args.max_chars} chars (~{args.max_chars // 4} tokens)")
    print(f"Datasets: {len(datasets_to_process)}")
    print(f"  {', '.join(datasets_to_process)}")
    print("="*80)
    print("ℹ️  Downloading from GitHub (raw.githubusercontent.com)")
    print("ℹ️  Live stats update every second during download")
    print("="*80)

    # Process each dataset
    all_stats = {}
    successful = []
    failed = []

    start_time = time.time()

    for dataset_name in datasets_to_process:
        try:
            stats = process_stack_dataset(
                dataset_name,
                input_dir,
                output_dir,
                args.min_chars,
                args.max_chars,
                args.batch_size,
                args.shard_size,
                args.num_workers,
                args.max_samples
            )

            if stats:
                all_stats[dataset_name] = stats
                successful.append(dataset_name)
            else:
                failed.append(dataset_name)

        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(dataset_name)

    elapsed = time.time() - start_time

    # Save statistics
    stats_file = output_dir / "stack_v2_preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("Stack v2 Preprocessing Summary")
    print("="*80)
    print(f"✓ Successful: {len(successful)} / {len(datasets_to_process)}")

    total_input = sum(s['input_samples'] for s in all_stats.values())
    total_output = sum(s['output_samples'] for s in all_stats.values())
    total_error_404 = sum(s['error_404'] for s in all_stats.values())
    total_error_encoding = sum(s['error_encoding'] for s in all_stats.values())
    total_error_timeout = sum(s['error_timeout'] for s in all_stats.values())
    total_error_network = sum(s['error_network'] for s in all_stats.values())
    total_too_short = sum(s['too_short'] for s in all_stats.values())
    total_truncated = sum(s['truncated'] for s in all_stats.values())

    print(f"\nOverall Statistics:")
    print(f"  Input samples:    {total_input:,}")
    print(f"  Output samples:   {total_output:,}")
    print(f"  Errors:")
    print(f"    - 404 (not found): {total_error_404:,} ({100*total_error_404/total_input:.2f}%)")
    print(f"    - Encoding:        {total_error_encoding:,} ({100*total_error_encoding/total_input:.2f}%)")
    print(f"    - Timeout:         {total_error_timeout:,} ({100*total_error_timeout/total_input:.2f}%)")
    print(f"    - Network:         {total_error_network:,} ({100*total_error_network/total_input:.2f}%)")
    print(f"  Too short:        {total_too_short:,} ({100*total_too_short/total_input:.2f}%)")
    print(f"  Truncated:        {total_truncated:,} ({100*total_truncated/total_input:.2f}%)")
    print(f"  Success rate:     {100*total_output/total_input:.1f}%")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for name in failed:
            print(f"    - {name}")

    print(f"\nTime: {elapsed/3600:.2f} hours")
    print(f"Stats saved to: {stats_file}")
    print("="*80)


if __name__ == "__main__":
    main()
