# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build_instruct_mixture``: the per-config instruct mixture (``dataset/instruct_mixtures/<config>/<mixture>/{train,validation}/``)
built from the standardized raw shards of its ``instruct`` sources.

Per source: measure tokens per row with the tokenizer, take ``ceil(budget_tokens × share ÷ tokens_per_row)`` rows,
drop rows longer than ``mixture.max_tokens``; then input inversions on a seeded sample, normalized exact dedup,
empty-field removal, seeded shuffle, train/validation split. Rebuilt whenever the mixture hash, ``budget_tokens`` or
any input source's raw shard list changed; otherwise a no-op returning the stored manifests.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from itertools import islice
from math import ceil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data_preparation.lib.storage.parquet import normalized_hash, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig, InstructMixtureConfig
from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import progress
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.stages.row_pipeline import check_length, create_input_inversion, has_required_fields, instruct_text
from data_preparation.lib.stages.shared import (
    DEFAULT_SHARD_SIZE,
    TokenCounter,
    current_manifest,
    new_manifest,
    record_new_shards,
    require_manifest,
    shard_list,
)

log = get_logger(__name__)

Row = dict[str, Any]
INSTRUCT_COLUMNS = ("instruction", "input", "output")


def build_instruct_mixture(
    cfg: DatasetConfig,
    instruct_mixture_name: str,
    layout: DatasetLayout,
    *,
    budget_tokens: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Manifest]:
    """Build (or return) the ``train`` / ``validation`` splits of a mixture; returns ``{split: Manifest}``.

    Columns: ``instruction``, ``input``, ``output``, ``tokens``. ``extra`` records per-source counts, the measured
    ``tokens_per_row``, ``short_sources`` (sources with fewer raw rows than needed — the planner tops them up),
    ``input_shards`` (raw shard lists) and the build ``metadata``.
    """
    mixture = cfg.instruct_mixtures[instruct_mixture_name]
    mixture_hash = cfg.instruct_mixture_hash(instruct_mixture_name)
    split_dirs = {split: layout.instruct_mixture_dir(cfg.name, instruct_mixture_name, split) for split in INSTRUCT_MIXTURE_SPLITS}

    # The inputs: one raw manifest per instruct source (an error if a source has not been downloaded yet).
    raw_manifests = {
        src: require_manifest(layout.source_dir(src, "raw"), cfg.source_hash(src), "raw", src) for src in mixture.sources
    }
    input_shards = {src: shard_list(manifest) for src, manifest in raw_manifests.items()}

    existing = _existing_manifests(split_dirs, mixture_hash, input_shards=input_shards, budget_tokens=budget_tokens)
    if existing is not None:
        return existing

    log.info("building mixture %s (%d tokens) -> %s", instruct_mixture_name, budget_tokens, split_dirs["train"].parent)
    counter = TokenCounter(cfg, layout)

    # Step 0: take the budgeted rows of every source.
    rows: list[Row] = []
    counts: dict[str, dict[str, Any]] = {}
    for src, share in mixture.sources.items():
        taken, info = _take_source_rows(layout, src, raw_manifests[src], share * budget_tokens, mixture.max_tokens, counter)
        rows.extend(taken)
        counts[src] = info
        log.info(
            "  %s: %d of %d needed rows (%.1f tokens/row), %d kept after length check",
            src, info["available_rows"], info["needed_rows"], info["tokens_per_row"], info["kept_rows"],
        )
    tokens_per_row = {src: info["tokens_per_row"] for src, info in counts.items()}
    short_sources = {
        src: {"available_rows": info["available_rows"], "needed_rows": info["needed_rows"]}
        for src, info in counts.items()
        if info["available_rows"] < info["needed_rows"]
    }

    # Steps 1-4: inversions -> dedup + empty removal -> shuffle + split -> write. One seeded RNG drives steps 1 and 3.
    steps = progress(total=4, desc=f"{instruct_mixture_name}: build_instruct_mixture", unit="step", leave=False)
    rng = random.Random(mixture.seed)

    steps.set_postfix({"step": "input_inversions"}, refresh=False)
    inverted = _apply_input_inversions(rows, mixture, rng, counter)
    steps.update(1)

    steps.set_postfix({"step": "dedup"}, refresh=False)
    rows_before_dedup = len(rows)
    rows = _dedup(rows)
    duplicates = rows_before_dedup - len(rows)
    rows_before_empty_removal = len(rows)
    rows = [row for row in rows if has_required_fields(row)]
    empty = rows_before_empty_removal - len(rows)
    steps.update(1)

    steps.set_postfix({"step": "shuffle_split"}, refresh=False)
    rng.shuffle(rows)
    split_at = int(len(rows) * (1 - mixture.val_split))
    splits = {"train": rows[:split_at], "validation": rows[split_at:]}
    steps.update(1)

    steps.set_postfix({"step": "write"}, refresh=False)
    metadata = {
        "total_examples": len(rows),
        "train_examples": len(splits["train"]),
        "val_examples": len(splits["validation"]),
        "inverted": inverted,
        "duplicates_removed": duplicates,
        "empty_removed": empty,
        "max_tokens": mixture.max_tokens,
        "seed": mixture.seed,
    }
    manifests: dict[str, Manifest] = {}
    for split, split_rows in splits.items():
        out_dir = split_dirs[split]
        write_dict_rows(_ordered(split_rows), out_dir, shard_size, start_shard=0)
        manifest = new_manifest(cfg, instruct_mixture_name, mixture_hash, "instruct_mixture", tokens=True)
        record_new_shards(manifest, out_dir, 0, tokens=_tokens_per_shard(split_rows, shard_size))
        manifest.extra = {
            "split": split,
            "budget_tokens": budget_tokens,
            "input_shards": input_shards,
            "counts": counts,
            "tokens_per_row": tokens_per_row,
            "short_sources": short_sources,
            "metadata": metadata,
        }
        manifest.save(out_dir)
        manifests[split] = manifest
    steps.update(1)
    steps.close()
    return manifests


def _existing_manifests(
    split_dirs: dict[str, Path], mixture_hash: str, *, input_shards: dict[str, list[list[Any]]], budget_tokens: int
) -> dict[str, Manifest] | None:
    """The stored manifests of every split if they were built from exactly these inputs and budget, else None."""
    manifests: dict[str, Manifest] = {}
    for split, directory in split_dirs.items():
        manifest = current_manifest(directory, mixture_hash, "instruct_mixture")
        if manifest is None:
            return None
        if manifest.extra.get("input_shards") != input_shards or manifest.extra.get("budget_tokens") != budget_tokens:
            return None
        manifests[split] = manifest
    return manifests


def _apply_input_inversions(rows: list[Row], mixture: InstructMixtureConfig, rng: random.Random, counter: TokenCounter) -> int:
    """Replace a seeded sample of ``rows`` (fraction ``mixture.input_inversions``) by their input inversion, in place.

    Rows the inversion leaves unchanged (empty instruction or output) are not counted. Returns the number inverted.
    """
    if mixture.input_inversions <= 0 or not rows:
        return 0
    inverted = 0
    sample_size = int(len(rows) * mixture.input_inversions)
    for index in rng.sample(range(len(rows)), sample_size):
        new_row = create_input_inversion(rows[index])
        if new_row is rows[index]:
            continue
        new_row["tokens"] = counter.count(instruct_text(new_row))
        rows[index] = new_row
        inverted += 1
    return inverted


def _iter_raw_batches(raw_dir: Path, raw: Manifest) -> Iterator[list[Row]]:
    """The standardized rows of every raw shard, batch by batch in shard order (only the instruct columns are read)."""
    for shard in raw.shards:
        parquet_file = pq.ParquetFile(raw_dir / shard.name)
        for batch in parquet_file.iter_batches(columns=list(INSTRUCT_COLUMNS)):
            yield batch.to_pylist()


def _iter_raw_rows(raw_dir: Path, raw: Manifest) -> Iterator[Row]:
    """The standardized rows of every raw shard, one by one in shard order."""
    for batch in _iter_raw_batches(raw_dir, raw):
        yield from batch


def _take_source_rows(
    layout: DatasetLayout, src: str, raw: Manifest, target_tokens: float, max_tokens: int, counter: TokenCounter
) -> tuple[list[Row], dict[str, Any]]:
    """The first ``ceil(target_tokens / tokens_per_row)`` standardized rows of a source (token-counted, length-checked).

    Pass 1 measures the token count of every raw row (ints only), pass 2 re-reads just the rows that are taken.
    """
    raw_dir = layout.source_dir(src, "raw")

    # Pass 1: token count of every row -> how many rows this source needs to contribute.
    tokens: list[int] = []
    with progress(total=raw.rows(), desc=f"{src}: count_tokens", unit="row", leave=False) as bar:
        for batch in _iter_raw_batches(raw_dir, raw):
            tokens.extend(counter.count_many([instruct_text(row) for row in batch]))
            bar.update(len(batch))
    available = len(tokens)
    tokens_per_row = sum(tokens) / available if available else 0.0
    needed = ceil(target_tokens / tokens_per_row) if tokens_per_row > 0 else 0
    take = min(needed, available)

    # Pass 2: re-read the first `take` rows and keep those within the length limit.
    taken: list[Row] = []
    dropped_long = 0
    for row, n_tokens in zip(islice(_iter_raw_rows(raw_dir, raw), take), tokens):
        if not check_length(n_tokens, max_tokens):
            dropped_long += 1
            continue
        taken.append({"instruction": row["instruction"], "input": row["input"] or "", "output": row["output"], "tokens": n_tokens})

    info = {
        "available_rows": available,
        "needed_rows": needed,
        "taken_rows": take,
        "dropped_too_long": dropped_long,
        "kept_rows": len(taken),
        "tokens_per_row": tokens_per_row,
        "target_tokens": target_tokens,
    }
    return taken, info


def _dedup(rows: list[Row]) -> list[Row]:
    """Normalized exact dedup over instruction + input + output; the first occurrence wins."""
    seen: set[str] = set()
    kept: list[Row] = []
    for row in rows:
        key = normalized_hash(f"{row['instruction']}\n{row['input']}\n{row['output']}")
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _ordered(rows: list[Row]) -> list[Row]:
    """The rows with their columns in the fixed output order."""
    return [{"instruction": r["instruction"], "input": r["input"], "output": r["output"], "tokens": r["tokens"]} for r in rows]


def _tokens_per_shard(rows: list[Row], shard_size: int) -> dict[str, int]:
    """Token sum per output shard (``data-00000.parquet``, ...) for rows written in chunks of ``shard_size``."""
    sums: dict[str, int] = {}
    for start in range(0, len(rows), shard_size):
        shard_name = f"data-{start // shard_size:05d}.parquet"
        sums[shard_name] = sum(int(r["tokens"]) for r in rows[start : start + shard_size])
    return sums
