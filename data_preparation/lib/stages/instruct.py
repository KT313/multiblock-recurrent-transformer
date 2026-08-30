# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``build_instruct_mixture``: the per-config instruct mixture (``dataset/instruct_mixtures/<config>/<mixture>/{train,validation}/``)
built from the standardized raw shards of its ``instruct`` sources.

Per source: read the raw rows in order with their ``tokens`` column (counted at download time), skip rows longer
than ``mixture.max_tokens`` and stop as soon as the kept rows hold ``budget_tokens × share`` tokens — a source is
read only as far as needed; then input inversions on a seeded sample, normalized exact dedup, empty-field removal,
seeded shuffle, train/validation split. Rebuilt whenever the mixture hash, ``budget_tokens`` or any input source's
raw shard list changed; otherwise a no-op returning the stored manifests.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.storage.parquet import normalized_hash, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig, InstructMixtureConfig
from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.ui.dashboard import progress
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.stages.row_pipeline import check_length, create_input_inversion, has_required_fields, instruct_text
from data_preparation.lib.stages.shared import (
    DEFAULT_SHARD_SIZE,
    TokenCounter,
    current_manifest,
    ensure_raw_tokens,
    new_manifest,
    record_new_shards,
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
    should_stop: StopCheck | None = None,
) -> dict[str, Manifest]:
    """Build (or return) the ``train`` / ``validation`` splits of a mixture; returns ``{split: Manifest}``.

    Columns: ``instruction``, ``input``, ``output``, ``tokens``. ``extra`` records per-source counts, the measured
    ``tokens_per_row``, ``short_sources`` (sources whose raw rows ran out before their share — the planner tops them up),
    ``input_shards`` (raw shard lists) and the build ``metadata``. The build is all-or-nothing (the splits are
    rewritten as a whole); ``should_stop`` is checked between raw shards while reading and before writing.
    """
    mixture = cfg.instruct_mixtures[instruct_mixture_name]
    mixture_hash = cfg.instruct_mixture_hash(instruct_mixture_name)
    split_dirs = {split: layout.instruct_mixture_dir(cfg.name, instruct_mixture_name, split) for split in INSTRUCT_MIXTURE_SPLITS}

    # The inputs: one raw manifest per instruct source (an error if a source has not been downloaded yet); a raw
    # directory from before the tokens column is upgraded in place.
    raw_manifests: dict[str, Manifest] = {}
    for src in mixture.sources:
        raw = ensure_raw_tokens(cfg, src, layout)
        if raw is None:
            raise FileNotFoundError(f"{src}: no current raw manifest in {layout.source_dir(src, 'raw')}; run the download stage first")
        raw_manifests[src] = raw
    input_shards = {src: shard_list(manifest) for src, manifest in raw_manifests.items()}

    existing = _existing_manifests(split_dirs, mixture_hash, input_shards=input_shards, budget_tokens=budget_tokens)
    if existing is not None:
        return existing

    log.info("building mixture %s (%d tokens) -> %s", instruct_mixture_name, budget_tokens, split_dirs["train"].parent)
    counter = TokenCounter(cfg, layout)  # instruct examples: uncapped, like their raw `tokens`

    # Step 0: take the budgeted rows of every source.
    rows: list[Row] = []
    counts: dict[str, dict[str, Any]] = {}
    for src, share in mixture.sources.items():
        taken, info = _take_source_rows(layout, src, raw_manifests[src], share * budget_tokens, mixture.max_tokens, should_stop)
        rows.extend(taken)
        counts[src] = info
        log.info(
            "  %s: read %d of %d rows (%.1f tokens/row), %d kept after length check",
            src, info["taken_rows"], info["available_rows"], info["tokens_per_row"], info["kept_rows"],
        )
    tokens_per_row = {src: info["tokens_per_row"] for src, info in counts.items()}
    short_sources = {
        src: {"available_rows": info["available_rows"], "needed_rows": info["needed_rows"]}
        for src, info in counts.items()
        if info["available_rows"] < info["needed_rows"]
    }

    # Steps 1-4: inversions -> dedup + empty removal -> shuffle + split -> write. One seeded RNG drives steps 1 and 3.
    check_stop(should_stop)
    with progress(total=4, desc=f"{instruct_mixture_name}: build_instruct_mixture", unit="step", leave=False) as steps:
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


def _iter_raw_rows(raw_dir: Path, raw: Manifest, should_stop: StopCheck | None = None) -> Iterator[Row]:
    """The standardized rows of every raw shard with their ``tokens``, one by one in shard order; a consumer that
    stops early never opens the remaining shards. ``should_stop`` is checked before every shard."""
    for shard in raw.shards:
        check_stop(should_stop)
        parquet_file = pq.ParquetFile(raw_dir / shard.name)
        for batch in parquet_file.iter_batches(columns=[*INSTRUCT_COLUMNS, "tokens"]):
            yield from batch.to_pylist()


def _take_source_rows(
    layout: DatasetLayout, src: str, raw: Manifest, target_tokens: float, max_tokens: int, should_stop: StopCheck | None = None
) -> tuple[list[Row], dict[str, Any]]:
    """The standardized rows of a source read in order until the kept rows (those within ``max_tokens``) hold
    ``target_tokens`` tokens; rows beyond that are not read.

    ``info``: ``available_rows`` (raw rows on disk), ``taken_rows`` (rows read), ``needed_rows`` (rows read when the
    target was reached, else the estimate ``ceil(target ÷ tokens/row)`` — larger than ``available_rows`` marks the
    source short), ``dropped_too_long``, ``kept_rows``, ``tokens_per_row`` (measured over the rows read),
    ``target_tokens``.
    """
    taken: list[Row] = []
    rows_read = 0
    tokens_read = 0
    kept_tokens = 0
    dropped_long = 0
    with progress(total=raw.rows(), desc=f"{src}: take_rows", unit="row", leave=False) as bar:
        for row in _iter_raw_rows(layout.source_dir(src, "raw"), raw, should_stop):
            rows_read += 1
            bar.update(1)
            n_tokens = int(row["tokens"])
            tokens_read += n_tokens
            if not check_length(n_tokens, max_tokens):
                dropped_long += 1
                continue
            taken.append({"instruction": row["instruction"], "input": row["input"] or "", "output": row["output"], "tokens": n_tokens})
            kept_tokens += n_tokens
            if kept_tokens >= target_tokens:
                break

    tokens_per_row = tokens_read / rows_read if rows_read else 0.0
    if kept_tokens >= target_tokens:
        needed = rows_read
    else:
        needed = ceil(target_tokens / tokens_per_row) if tokens_per_row > 0 else 0
    info = {
        "available_rows": raw.rows(),
        "needed_rows": needed,
        "taken_rows": rows_read,
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
