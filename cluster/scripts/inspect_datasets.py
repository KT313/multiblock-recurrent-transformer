#!/usr/bin/env python3
"""
Simple Dataset Structure Inspector

Shows the column structure and first sample of each downloaded dataset.

Usage:
    python inspect_datasets.py --input_dir /path/to/raw/datasets

Author: Claude Code
Date: 2025-01-26
"""

import argparse
from pathlib import Path
import pyarrow.parquet as pq


def inspect_dataset(dataset_dir: Path) -> None:
    """Inspect a single dataset and print its structure."""
    dataset_name = dataset_dir.name

    # Find first parquet shard
    parquet_files = sorted(dataset_dir.glob("shard-*.parquet"))

    if not parquet_files:
        print(f"⚠️  No parquet files found in {dataset_name}")
        return

    first_shard = parquet_files[0]

    parquet_file = pq.ParquetFile(first_shard)

    # Arrow schema (this has real pyarrow.Field objects with .type)
    arrow_schema = parquet_file.schema_arrow  # ← key change

    # Check if file has rows
    if parquet_file.metadata.num_rows == 0:
        print(f"⚠️  {dataset_name}: Empty dataset")
        return

    # Read only the first row
    first_batch = next(parquet_file.iter_batches(batch_size=1))
    first_row = first_batch.to_pydict()

    # Print results
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print("=" * 80)
    print(f"First shard: {first_shard.name}")
    print(f"Total rows in first shard: {parquet_file.metadata.num_rows:,}")
    print()

    print("Columns:")
    for field in arrow_schema:  # iterate Arrow fields
        print(f"  - {field.name}: {field.type}")
    print()

    print("First Sample:")
    for key, values in first_row.items():
        value = values[0]

        if isinstance(value, str) and len(value) > 500:
            display_value = value[:500] + f"... (truncated, total length: {len(value)} chars)"
        else:
            display_value = value

        print(f"  {key}: {repr(display_value)}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset structure")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing raw datasets")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        return

    # Find all dataset directories
    dataset_dirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]

    if not dataset_dirs:
        print(f"❌ Error: No dataset directories found in {input_dir}")
        return

    print(f"\nFound {len(dataset_dirs)} datasets in {input_dir}\n")

    # Inspect each dataset
    for dataset_dir in dataset_dirs:
        inspect_dataset(dataset_dir)

    print("=" * 80)
    print(f"Inspection complete: {len(dataset_dirs)} datasets")
    print("=" * 80)


if __name__ == "__main__":
    main()
