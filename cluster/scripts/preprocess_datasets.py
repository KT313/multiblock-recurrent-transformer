#!/usr/bin/env python3
"""
Dataset Preprocessing and Filtering

Filters and restructures raw datasets with unified schema.

Filters:
- Remove text with length < min_chars (default: 50)
- Truncate text with length > max_chars (default: 20,000 = ~5000 tokens)

Output schema:
- text: str (cleaned/truncated text)
- source: str (dataset name)
- original_length: int (character count before truncation)

Usage:
    python preprocess_datasets.py \
        --input_dir /path/to/raw \
        --output_dir /path/to/filtered \
        --min_chars 50 \
        --max_chars 20000

Author: Claude Code
Date: 2025-01-26
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


# =============================================================================
# Text Field Detection
# =============================================================================

def detect_text_field(schema: pa.Schema, dataset_name: str) -> str:
    """Detect the text field in the schema.

    Tries in order: 'text', 'TEXT', 'content', 'code'
    """
    column_names = schema.names

    if 'text' in column_names:
        return 'text'
    elif 'TEXT' in column_names:
        return 'TEXT'
    elif 'content' in column_names:
        return 'content'
    elif 'code' in column_names:
        return 'code'
    else:
        raise ValueError(
            f"No text field found in {dataset_name}. "
            f"Available columns: {column_names}"
        )


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_batch(
    batch: pa.RecordBatch,
    text_field: str,
    source_name: str,
    min_chars: int,
    max_chars: int
) -> Tuple[pa.RecordBatch, Dict[str, int]]:
    """Preprocess a batch of samples.

    Returns:
        Processed batch and statistics dict
    """
    stats = {
        'input_samples': len(batch),
        'removed_too_short': 0,
        'removed_invalid': 0,
        'truncated': 0,
        'output_samples': 0
    }

    # Extract text column
    text_array = batch[text_field]

    # Filter 1: Remove null/empty
    is_valid = pc.is_valid(text_array)
    stats['removed_invalid'] = len(batch) - pc.sum(is_valid).as_py()

    if stats['removed_invalid'] > 0:
        batch = batch.filter(is_valid)
        text_array = batch[text_field]

    if len(batch) == 0:
        # Return empty batch with correct schema
        empty_batch = pa.RecordBatch.from_pydict({
            'text': pa.array([], type=pa.string()),
            'source': pa.array([], type=pa.string()),
            'original_length': pa.array([], type=pa.int64())
        })
        return empty_batch, stats

    # Compute original lengths
    original_lengths = pc.utf8_length(text_array)

    # Filter 2: Remove too short (< min_chars)
    length_mask = pc.greater_equal(original_lengths, min_chars)
    stats['removed_too_short'] = len(batch) - pc.sum(length_mask).as_py()

    if stats['removed_too_short'] > 0:
        batch = batch.filter(length_mask)
        text_array = batch[text_field]
        original_lengths = pc.utf8_length(text_array)

    if len(batch) == 0:
        # Return empty batch with correct schema
        empty_batch = pa.RecordBatch.from_pydict({
            'text': pa.array([], type=pa.string()),
            'source': pa.array([], type=pa.string()),
            'original_length': pa.array([], type=pa.int64())
        })
        return empty_batch, stats

    # Truncate long texts
    text_list = text_array.to_pylist()
    original_lengths_list = original_lengths.to_pylist()

    truncated_texts = []
    for text, orig_len in zip(text_list, original_lengths_list):
        if orig_len > max_chars:
            truncated_texts.append(text[:max_chars])
            stats['truncated'] += 1
        else:
            truncated_texts.append(text)

    # Create output batch with unified schema
    output_batch = pa.RecordBatch.from_pydict({
        'text': pa.array(truncated_texts, type=pa.string()),
        'source': pa.array([source_name] * len(truncated_texts), type=pa.string()),
        'original_length': original_lengths
    })

    stats['output_samples'] = len(output_batch)

    return output_batch, stats


# =============================================================================
# Dataset Processing
# =============================================================================

def process_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    min_chars: int,
    max_chars: int,
    batch_size: int,
    shard_size: int
) -> Dict[str, any]:
    """Process a single dataset.

    Returns:
        Statistics dictionary
    """
    dataset_input_dir = input_dir / dataset_name
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📊 Processing: {dataset_name}")
    print(f"{'='*80}")
    print(f"  Input:  {dataset_input_dir}")
    print(f"  Output: {dataset_output_dir}")

    # Find all parquet shards
    parquet_files = sorted(dataset_input_dir.glob("shard-*.parquet"))

    if not parquet_files:
        print(f"  ⚠️  No parquet files found, skipping")
        return None

    print(f"  Found {len(parquet_files)} input shards")

    # Detect text field from first file
    first_file = pq.ParquetFile(parquet_files[0])
    text_field = detect_text_field(first_file.schema, dataset_name)
    print(f"  Text field: '{text_field}'")

    # Aggregate statistics
    total_stats = {
        'input_samples': 0,
        'output_samples': 0,
        'removed_too_short': 0,
        'removed_invalid': 0,
        'truncated': 0,
        'total_original_chars': 0,
        'total_output_chars': 0
    }

    # Output tracking
    output_buffer = []
    output_shard_num = 0

    # Process all parquet files
    pbar = tqdm(
        parquet_files,
        desc=f"  🔧 Processing",
        unit=" shards",
        miniters=1,
        mininterval=2.0
    )

    for parquet_file in pbar:
        # Open file
        pf = pq.ParquetFile(parquet_file)

        # Process in batches
        for batch in pf.iter_batches(batch_size=batch_size):
            # Preprocess batch
            processed_batch, batch_stats = preprocess_batch(
                batch, text_field, dataset_name, min_chars, max_chars
            )

            # Update statistics
            total_stats['input_samples'] += batch_stats['input_samples']
            total_stats['output_samples'] += batch_stats['output_samples']
            total_stats['removed_too_short'] += batch_stats['removed_too_short']
            total_stats['removed_invalid'] += batch_stats['removed_invalid']
            total_stats['truncated'] += batch_stats['truncated']

            # Add to output buffer
            if len(processed_batch) > 0:
                output_buffer.append(processed_batch)

                # Calculate total chars
                total_stats['total_output_chars'] += pc.sum(
                    pc.utf8_length(processed_batch['text'])
                ).as_py()
                total_stats['total_original_chars'] += pc.sum(
                    processed_batch['original_length']
                ).as_py()

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

            # Update progress bar
            pbar.set_postfix({
                'out_samples': f"{total_stats['output_samples']:,}",
                'filtered': f"{total_stats['removed_too_short']:,}"
            })

    # Save remaining samples in buffer
    if output_buffer:
        shard_table = pa.Table.from_batches(output_buffer)
        output_file = dataset_output_dir / f"data-{output_shard_num:05d}.parquet"
        pq.write_table(shard_table, output_file)
        output_shard_num += 1

    # Calculate final statistics
    if total_stats['output_samples'] > 0:
        total_stats['pass_rate'] = total_stats['output_samples'] / total_stats['input_samples']
        total_stats['avg_original_length'] = int(total_stats['total_original_chars'] / total_stats['output_samples'])
        total_stats['avg_output_length'] = int(total_stats['total_output_chars'] / total_stats['output_samples'])
    else:
        total_stats['pass_rate'] = 0.0
        total_stats['avg_original_length'] = 0
        total_stats['avg_output_length'] = 0

    # Remove intermediate counters
    del total_stats['total_original_chars']
    del total_stats['total_output_chars']

    print(f"  ✓ Complete:")
    print(f"      Input:  {total_stats['input_samples']:,} samples")
    print(f"      Output: {total_stats['output_samples']:,} samples ({total_stats['pass_rate']:.1%} pass rate)")
    print(f"      Filtered: {total_stats['removed_too_short']:,} too short, {total_stats['removed_invalid']:,} invalid")
    print(f"      Truncated: {total_stats['truncated']:,} samples")
    print(f"      Avg length: {total_stats['avg_original_length']:,} → {total_stats['avg_output_length']:,} chars")
    print(f"      Output shards: {output_shard_num}")

    return total_stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess and filter datasets")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory with raw datasets")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for filtered datasets")
    parser.add_argument("--min_chars", type=int, default=50,
                       help="Minimum text length in characters (default: 50)")
    parser.add_argument("--max_chars", type=int, default=20000,
                       help="Maximum text length before truncation (default: 20000 = ~5000 tokens)")
    parser.add_argument("--batch_size", type=int, default=100000,
                       help="Samples to process at once (default: 100000)")
    parser.add_argument("--shard_size", type=int, default=100000,
                       help="Samples per output parquet file (default: 100000)")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                       help="Specific datasets to process (default: all)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all dataset directories
    if args.datasets:
        dataset_dirs = []
        for pattern in args.datasets:
            # Use glob to match patterns
            matched = list(input_dir.glob(pattern))
            # Filter to only directories
            dataset_dirs.extend([d for d in matched if d.is_dir()])
        # Remove duplicates while preserving order
        seen = set()
        dataset_dirs = [d for d in dataset_dirs if not (d in seen or seen.add(d))]
    else:
        dataset_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not dataset_dirs:
        print(f"❌ Error: No dataset directories found in {input_dir}")
        return

    print("="*80)
    print("Dataset Preprocessing and Filtering")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Filters:")
    print(f"  - Minimum length: {args.min_chars} chars")
    print(f"  - Maximum length: {args.max_chars} chars (~{args.max_chars // 4} tokens)")
    print(f"Datasets: {len(dataset_dirs)}")
    print("="*80)

    # Process each dataset
    all_stats = {}
    successful = []
    failed = []

    start_time = time.time()

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        try:
            stats = process_dataset(
                dataset_name,
                input_dir,
                output_dir,
                args.min_chars,
                args.max_chars,
                args.batch_size,
                args.shard_size
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
    stats_file = output_dir / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("Preprocessing Summary")
    print("="*80)
    print(f"✓ Successful: {len(successful)} / {len(dataset_dirs)}")

    total_input = sum(s['input_samples'] for s in all_stats.values())
    total_output = sum(s['output_samples'] for s in all_stats.values())
    total_removed = sum(s['removed_too_short'] + s['removed_invalid'] for s in all_stats.values())
    total_truncated = sum(s['truncated'] for s in all_stats.values())

    print(f"\nOverall Statistics:")
    print(f"  Input samples:  {total_input:,}")
    print(f"  Output samples: {total_output:,}")
    print(f"  Removed:        {total_removed:,} ({100*total_removed/total_input:.2f}%)")
    print(f"  Truncated:      {total_truncated:,} ({100*total_truncated/total_input:.2f}%)")
    print(f"  Pass rate:      {100*total_output/total_input:.1f}%")

    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for name in failed:
            print(f"    - {name}")

    print(f"\nTime: {elapsed/3600:.2f} hours")
    print(f"Stats saved to: {stats_file}")
    print("="*80)


if __name__ == "__main__":
    main()
