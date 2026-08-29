# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Length-filter the raw downloads into a unified schema at ``dataset/pretraining/filtered/<source>/data-*.parquet``.

Per source: detect the text column (``text`` / ``TEXT`` / ``content`` / ``code``), drop null and too-short texts
(< ``--min_chars``), truncate texts longer than ``--max_chars`` characters. Output columns: ``text``, ``source``,
``original_length``. Writes ``preprocessing_stats.json`` next to the source directories.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from data_preparation.lib.common import (
    add_common_args,
    list_parquet_files,
    print_header,
    select_dataset_dirs,
    write_parquet_shards,
)

TEXT_FIELD_CANDIDATES = ("text", "TEXT", "content", "code")
_OUTPUT_FIELDS: list[tuple[str, pa.DataType]] = [
    ("text", pa.string()),
    ("source", pa.string()),
    ("original_length", pa.int64()),
]
OUTPUT_SCHEMA = pa.schema(_OUTPUT_FIELDS)


def detect_text_field(schema: pa.Schema, dataset_name: str) -> str:
    """Return the first of ``text``/``TEXT``/``content``/``code`` present in ``schema``."""
    for candidate in TEXT_FIELD_CANDIDATES:
        if candidate in schema.names:
            return candidate
    raise ValueError(f"No text field found in {dataset_name}. Available columns: {schema.names}")


def preprocess_batch(
    batch: pa.RecordBatch, text_field: str, source_name: str, min_chars: int, max_chars: int
) -> tuple[pa.RecordBatch, dict[str, int]]:
    """Filter and truncate one batch; returns the unified-schema batch and per-batch statistics."""
    stats = {"input_samples": len(batch), "removed_too_short": 0, "removed_invalid": 0, "truncated": 0}
    # the raw shards store text as (large_)string; the cast only narrows the stub type, nothing happens at runtime
    text_array = cast(pa.StringArray, batch[text_field])

    is_valid = pc.is_valid(text_array)
    # pyarrow-stubs restricts `pc.sum` to numeric arrays, but summing a boolean array is valid pyarrow
    stats["removed_invalid"] = len(batch) - pc.sum(is_valid).as_py()  # type: ignore[type-var]
    if stats["removed_invalid"] > 0:
        batch = batch.filter(is_valid)
        text_array = cast(pa.StringArray, batch[text_field])
    if len(batch) == 0:
        return pa.RecordBatch.from_pylist([], schema=OUTPUT_SCHEMA), stats | {"output_samples": 0}

    original_lengths = pc.utf8_length(text_array)
    # pyarrow-stubs does not accept a Python int as the second operand, pyarrow does
    length_mask = pc.greater_equal(original_lengths, min_chars)  # type: ignore[call-overload]
    stats["removed_too_short"] = len(batch) - pc.sum(length_mask).as_py()
    if stats["removed_too_short"] > 0:
        batch = batch.filter(length_mask)
        text_array = cast(pa.StringArray, batch[text_field])
        original_lengths = pc.utf8_length(text_array)
    if len(batch) == 0:
        return pa.RecordBatch.from_pylist([], schema=OUTPUT_SCHEMA), stats | {"output_samples": 0}

    truncated_texts: list[str] = []
    texts = cast(list[str], text_array.to_pylist())  # nulls were filtered above
    lengths = cast(list[int], original_lengths.to_pylist())
    for text, orig_len in zip(texts, lengths):
        if orig_len > max_chars:
            truncated_texts.append(text[:max_chars])
            stats["truncated"] += 1
        else:
            truncated_texts.append(text)

    output = pa.RecordBatch.from_arrays(
        [
            pa.array(truncated_texts, type=pa.string()),
            pa.array([source_name] * len(truncated_texts), type=pa.string()),
            pc.cast(original_lengths, pa.int64()),
        ],
        schema=OUTPUT_SCHEMA,
    )
    stats["output_samples"] = len(output)
    return output, stats


def process_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    min_chars: int,
    max_chars: int,
    batch_size: int,
    shard_size: int,
) -> dict[str, float] | None:
    """Filter every ``*.parquet`` of one source into ``output_dir/<name>/data-*.parquet``."""
    dataset_input_dir = input_dir / dataset_name
    dataset_output_dir = output_dir / dataset_name
    print_header(f"Processing: {dataset_name}")
    print(f"  Input:  {dataset_input_dir}\n  Output: {dataset_output_dir}")

    parquet_files = list_parquet_files(dataset_input_dir)
    if not parquet_files:
        print("  No parquet files found, skipping")
        return None
    text_field = detect_text_field(pq.ParquetFile(parquet_files[0]).schema_arrow, dataset_name)
    print(f"  Found {len(parquet_files)} input shards, text field '{text_field}'")

    totals: dict[str, float] = {
        "input_samples": 0,
        "output_samples": 0,
        "removed_too_short": 0,
        "removed_invalid": 0,
        "truncated": 0,
        "total_original_chars": 0,
        "total_output_chars": 0,
    }

    def filtered_batches() -> Iterator[pa.RecordBatch]:
        pbar = tqdm(parquet_files, desc="  Filtering", unit=" shards", miniters=1, mininterval=2.0)
        for parquet_file in pbar:
            for batch in pq.ParquetFile(parquet_file).iter_batches(batch_size=batch_size):
                processed, stats = preprocess_batch(batch, text_field, dataset_name, min_chars, max_chars)
                for key, value in stats.items():
                    totals[key] += value
                if len(processed) > 0:
                    texts = cast(pa.StringArray, processed["text"])  # OUTPUT_SCHEMA declares it a string column
                    totals["total_output_chars"] += pc.sum(pc.utf8_length(texts)).as_py()
                    totals["total_original_chars"] += pc.sum(processed["original_length"]).as_py()
                    yield processed
                pbar.set_postfix(
                    out_samples=f"{totals['output_samples']:,}", filtered=f"{totals['removed_too_short']:,}"
                )

    num_shards = write_parquet_shards(filtered_batches(), dataset_output_dir, shard_size)

    if totals["output_samples"] > 0:
        totals["pass_rate"] = totals["output_samples"] / totals["input_samples"]
        totals["avg_original_length"] = int(totals["total_original_chars"] / totals["output_samples"])
        totals["avg_output_length"] = int(totals["total_output_chars"] / totals["output_samples"])
    else:
        totals["pass_rate"], totals["avg_original_length"], totals["avg_output_length"] = 0.0, 0, 0
    del totals["total_original_chars"], totals["total_output_chars"]

    print(f"  Input {totals['input_samples']:,} -> output {totals['output_samples']:,} ({totals['pass_rate']:.1%})")
    print(f"  Filtered: {totals['removed_too_short']:,} too short, {totals['removed_invalid']:,} invalid")
    print(f"  Truncated: {totals['truncated']:,}; output shards: {num_shards}")
    return totals


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this command's options on ``parser`` (used by ``prepare.py`` and ``build_parser``)."""
    add_common_args(parser)
    parser.add_argument("--min_chars", type=int, default=50, help="Minimum text length in characters (default: 50)")
    parser.add_argument(
        "--max_chars", type=int, default=20000, help="Truncate texts longer than this many characters (default: 20000)"
    )
    parser.add_argument("--batch_size", type=int, default=100000, help="Rows processed at once (default: 100000)")
    parser.add_argument("--shard_size", type=int, default=100000, help="Rows per output parquet file (default: 100000)")
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None, help="Source names or glob patterns to process (default: all)"
    )


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for this command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the command with parsed ``args``."""
    input_dir = args.dataset_dir / "pretraining" / "raw"
    output_dir = args.dataset_dir / "pretraining" / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = select_dataset_dirs(input_dir, args.datasets)
    if not dataset_dirs:
        raise SystemExit(f"No dataset directories found in {input_dir}")

    print_header("Dataset length filtering")
    print(f"Input:  {input_dir}\nOutput: {output_dir}")
    print(f"Min length: {args.min_chars} chars, max length: {args.max_chars} chars; {len(dataset_dirs)} datasets")

    all_stats: dict[str, dict[str, float]] = {}
    failed: list[str] = []
    start = time.time()
    for dataset_dir in dataset_dirs:
        name = dataset_dir.name
        try:
            stats = process_dataset(
                name, input_dir, output_dir, args.min_chars, args.max_chars, args.batch_size, args.shard_size
            )
        except Exception as exc:  # keep going with the remaining sources
            print(f"  Error processing {name}: {exc}")
            traceback.print_exc()
            stats = None
        if stats:
            all_stats[name] = stats
        else:
            failed.append(name)

    stats_file = output_dir / "preprocessing_stats.json"
    stats_file.write_text(json.dumps(all_stats, indent=2))

    print_header("Filtering summary")
    total_in = sum(s["input_samples"] for s in all_stats.values())
    total_out = sum(s["output_samples"] for s in all_stats.values())
    print(f"Successful: {len(all_stats)} / {len(dataset_dirs)}; input {total_in:,} -> output {total_out:,} rows")
    if failed:
        print(f"Failed: {failed}")
    print(f"Time: {(time.time() - start) / 3600:.2f} hours; stats saved to {stats_file}")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and run."""
    run(build_parser().parse_args(argv))
