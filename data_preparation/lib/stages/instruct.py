# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build_mixture``: the per-config instruct mixture (``dataset/mixtures/<config>/<mixture>/{train,validation}/``)
built from the standardized raw shards of its ``instruct`` sources.

Per source: measure tokens per row with the tokenizer, take ``ceil(budget_tokens × share ÷ tokens_per_row)`` rows,
drop rows longer than ``mixture.max_tokens``; then input inversions on a seeded sample, normalized exact dedup,
empty-field removal, seeded shuffle, train/validation split. Rebuilt whenever the mixture hash, ``budget_tokens`` or
any input source's raw shard list changed; otherwise a no-op returning the stored manifests.
"""

from __future__ import annotations

import random
from math import ceil
from typing import Any

import pyarrow.parquet as pq

from data_preparation.lib.storage.parquet import normalized_hash, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig
from data_preparation.lib.schema.layout import MIXTURE_SPLITS, DatasetLayout
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


def build_mixture(
    cfg: DatasetConfig,
    mixture_name: str,
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
    mixture = cfg.mixtures[mixture_name]
    mixture_hash = cfg.mixture_hash(mixture_name)
    dirs = {split: layout.mixture_dir(cfg.name, mixture_name, split) for split in MIXTURE_SPLITS}
    raw_manifests = {
        src: require_manifest(layout.source_dir(src, "raw"), cfg.source_hash(src), "raw", src) for src in mixture.sources
    }
    input_shards = {src: shard_list(m) for src, m in raw_manifests.items()}
    existing = {split: current_manifest(path, mixture_hash, "mixture") for split, path in dirs.items()}
    if all(
        m is not None and m.extra.get("input_shards") == input_shards and m.extra.get("budget_tokens") == budget_tokens
        for m in existing.values()
    ):
        return {split: m for split, m in existing.items() if m is not None}

    log.info("building mixture %s (%d tokens) -> %s", mixture_name, budget_tokens, dirs["train"].parent)
    counter = TokenCounter(cfg, layout)
    rows: list[Row] = []
    counts: dict[str, dict[str, Any]] = {}
    tokens_per_row: dict[str, float] = {}
    short_sources: dict[str, dict[str, int]] = {}
    for src, share in mixture.sources.items():
        taken, info = _take_source_rows(layout, src, raw_manifests[src], share * budget_tokens, mixture.max_tokens, counter)
        rows.extend(taken)
        counts[src] = info
        tokens_per_row[src] = info["tokens_per_row"]
        if info["available_rows"] < info["needed_rows"]:
            short_sources[src] = {"available_rows": info["available_rows"], "needed_rows": info["needed_rows"]}
        log.info("  %s: %d of %d needed rows (%.1f tokens/row), %d kept after length check", src, info["available_rows"], info["needed_rows"], info["tokens_per_row"], info["kept_rows"])

    steps = progress(total=4, desc=f"{mixture_name}: build_mixture", unit="step", leave=False)
    rng = random.Random(mixture.seed)
    inverted = 0
    steps.set_postfix({"step": "input_inversions"}, refresh=False)
    if mixture.input_inversions > 0 and rows:
        for index in rng.sample(range(len(rows)), int(len(rows) * mixture.input_inversions)):
            new_row = create_input_inversion(rows[index])
            if new_row is not rows[index]:
                new_row["tokens"] = counter.count(instruct_text(new_row))
                rows[index] = new_row
                inverted += 1
    steps.update(1)
    steps.set_postfix({"step": "dedup"}, refresh=False)
    before = len(rows)
    rows = _dedup(rows)
    duplicates = before - len(rows)
    before = len(rows)
    rows = [r for r in rows if has_required_fields(r)]
    empty = before - len(rows)
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
        out = dirs[split]
        write_dict_rows(_ordered(split_rows), out, shard_size, start_shard=0)
        manifest = new_manifest(cfg, mixture_name, mixture_hash, "mixture", tokens=True)
        record_new_shards(manifest, out, 0, tokens=_tokens_per_shard(split_rows, shard_size))
        manifest.extra = {
            "split": split,
            "budget_tokens": budget_tokens,
            "input_shards": input_shards,
            "counts": counts,
            "tokens_per_row": tokens_per_row,
            "short_sources": short_sources,
            "metadata": metadata,
        }
        manifest.save(out)
        manifests[split] = manifest
    steps.update(1)
    steps.close()
    return manifests


def _take_source_rows(
    layout: DatasetLayout, src: str, raw: Manifest, target_tokens: float, max_tokens: int, counter: TokenCounter
) -> tuple[list[Row], dict[str, Any]]:
    """The first ``ceil(target_tokens / tokens_per_row)`` standardized rows of a source (token-counted, length-checked).

    Pass 1 measures the token count of every raw row (ints only), pass 2 re-reads just the rows that are taken.
    """
    raw_dir = layout.source_dir(src, "raw")
    tokens: list[int] = []
    with progress(total=raw.rows(), desc=f"{src}: count_tokens", unit="row", leave=False) as bar:
        for shard in raw.shards:
            for batch in pq.ParquetFile(raw_dir / shard.name).iter_batches(columns=list(INSTRUCT_COLUMNS)):
                tokens.extend(counter.count_many([instruct_text(r) for r in batch.to_pylist()]))
                bar.update(batch.num_rows)
    available = len(tokens)
    per_row = sum(tokens) / available if available else 0.0
    needed = ceil(target_tokens / per_row) if per_row > 0 else 0
    take = min(needed, available)
    taken: list[Row] = []
    dropped_long = 0
    position = 0
    for shard in raw.shards:
        if position >= take:
            break
        for batch in pq.ParquetFile(raw_dir / shard.name).iter_batches(columns=list(INSTRUCT_COLUMNS)):
            for row in batch.to_pylist():
                if position >= take:
                    break
                n = tokens[position]
                position += 1
                if not check_length(n, max_tokens):
                    dropped_long += 1
                    continue
                taken.append({"instruction": row["instruction"], "input": row["input"] or "", "output": row["output"], "tokens": n})
    info = {
        "available_rows": available,
        "needed_rows": needed,
        "taken_rows": take,
        "dropped_too_long": dropped_long,
        "kept_rows": len(taken),
        "tokens_per_row": per_row,
        "target_tokens": target_tokens,
    }
    return taken, info


def _dedup(rows: list[Row]) -> list[Row]:
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
    return [{"instruction": r["instruction"], "input": r["input"], "output": r["output"], "tokens": r["tokens"]} for r in rows]


def _tokens_per_shard(rows: list[Row], shard_size: int) -> dict[str, int]:
    sums: dict[str, int] = {}
    for start in range(0, len(rows), shard_size):
        sums[f"data-{start // shard_size:05d}.parquet"] = sum(int(r["tokens"]) for r in rows[start : start + shard_size])
    return sums
