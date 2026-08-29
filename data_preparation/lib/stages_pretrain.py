# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pipeline stages of ``pretrain`` sources: ``length_filter`` (raw -> filtered, shard by shard) and ``process``
(filtered -> processed: exact dedup, quality filter, decontamination, token counting, fuzzy dedup (see the
``fuzzy_dedup`` module), repetition).

Both are idempotent and incremental via the manifests (see ``stages_shared``). ``length_filter`` mirrors raw shards
1:1, so only new raw shards are filtered. ``process`` rewrites the whole processed directory whenever the filtered
shard list changed, but by construction the rows it keeps from old shards are identical (first occurrence wins in a
fresh streaming pass over old + new shards).
"""

from __future__ import annotations

import multiprocessing
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.lib.benchmarks import load_benchmark_ngrams
from data_preparation.lib.common import (
    list_parquet_files,
    md5_hex,
    normalized_hash,
    shard_index,
    write_dict_rows,
    write_parquet_shards,
)
from data_preparation.lib.dataset_config import DatasetConfig, DecontaminationConfig
from data_preparation.lib.fuzzy_dedup import fuzzy_dedup
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.manifest import Manifest, shard_rows
from data_preparation.lib.row_pipeline import (
    FILTERED_SCHEMA,
    check_contamination,
    check_quality,
    preprocess_batch,
)
from data_preparation.lib.sources import repeat_indices
from data_preparation.lib.stages_shared import (
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


# --- length filter -----------------------------------------------------------------------------------------------------


def length_filter(cfg: DatasetConfig, name: str, layout: DatasetLayout, *, batch_size: int = DEFAULT_SHARD_SIZE) -> Manifest:
    """Filter every raw shard not yet in the filtered manifest (``preprocess_batch``: drop null / short, truncate).

    Filtered shard ``data-N`` corresponds to raw shard ``data-N`` (same numbering, possibly fewer rows); columns
    ``text``, ``source``, ``original_length``. Per-shard statistics are recorded in ``extra["shard_stats"]``.
    """
    source = cfg.sources[name]
    if source.kind != "pretrain":
        raise ValueError(f"{name}: length_filter() applies to pretrain sources only (kind={source.kind})")
    processing = cfg.source_processing(name)
    source_hash = cfg.source_hash(name)
    raw_dir = layout.source_dir(name, "raw")
    out = layout.source_dir(name, "filtered")
    raw = require_manifest(raw_dir, source_hash, "raw", name)
    manifest = current_manifest(out, source_hash, "filtered")
    done: list[list[Any]] = manifest.extra.get("input_shards", []) if manifest is not None else []
    raw_shards = shard_list(raw)
    if manifest is not None and raw_shards[: len(done)] != done:
        log.warning("%s: raw shards changed under the filtered manifest, refiltering everything", name)
        manifest = None
    if manifest is None:
        manifest = new_manifest(cfg, name, source_hash, "filtered")
        manifest.extra = {"input_shards": [], "shard_stats": {}}
        done = []
    pending = raw.shards[len(done) :]
    if not pending:
        return manifest

    log.info("%s: length-filtering %d raw shard(s) -> %s", name, len(pending), out)
    for shard in pending:
        index = int(shard.name[len("data-") : -len(".parquet")])
        stats: Counter[str] = Counter()

        def batches(path: Path = raw_dir / shard.name, stats: Counter[str] = stats) -> Iterator[pa.RecordBatch]:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                out_batch, batch_stats = preprocess_batch(
                    batch, source.text_field, name, processing.min_chars, processing.max_chars
                )
                stats.update(batch_stats)
                if len(out_batch) > 0:
                    yield out_batch

        written = write_parquet_shards(batches(), out, shard_size=max(shard.rows, 1), start_shard=index)
        if written == 0:  # every row was dropped: keep the 1:1 numbering with an empty shard
            _write_empty_shard(out / shard.name)
        manifest.add_shard(shard.name, shard_rows(out / shard.name))
        manifest.extra["shard_stats"][shard.name] = dict(stats)
        manifest.extra["input_shards"].append([shard.name, shard.rows])
        manifest.rows_fetched = raw.rows_fetched
        manifest.save(out)
    return manifest


def _write_empty_shard(path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist([], schema=FILTERED_SCHEMA), tmp)
    tmp.replace(path)


# --- process -----------------------------------------------------------------------------------------------------------


def process(
    cfg: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    num_workers: int = 1,
    target_tokens: int | None = None,
) -> Manifest:
    """Stream all filtered shards in order through exact dedup -> quality filter -> decontamination -> token count
    -> fuzzy dedup -> ``repeat_to_budget`` and write processed shards (``text``, ``source``, ``tokens``).

    Skipped entirely when the filtered shard list (and, for ``repeat_to_budget`` sources, ``target_tokens``) matches
    what the processed manifest recorded; otherwise the processed directory is rewritten from shard 0.
    """
    source = cfg.sources[name]
    if source.kind != "pretrain":
        raise ValueError(f"{name}: process() applies to pretrain sources only (kind={source.kind})")
    processing = cfg.source_processing(name)
    source_hash = cfg.source_hash(name)
    filtered_dir = layout.source_dir(name, "filtered")
    out = layout.source_dir(name, "processed")
    filtered = require_manifest(filtered_dir, source_hash, "filtered", name)
    input_shards = shard_list(filtered)
    existing = current_manifest(out, source_hash, "processed")
    if (
        existing is not None
        and existing.extra.get("input_shards") == input_shards
        and (not source.repeat_to_budget or existing.extra.get("target_tokens") == target_tokens)
    ):
        return existing

    log.info("%s: processing %d filtered shard(s) -> %s", name, len(filtered.shards), out)
    counter = TokenCounter(cfg, layout)
    stats: dict[str, Any] = {
        "input_rows": filtered.rows(),
        "dedup": {"mode": processing.dedup.mode, "duplicates_removed": 0},
        "quality_filter": {"enabled": processing.quality_filter, "filtered_count": 0, "rejection_reasons": {}},
        "decontamination": {"enabled": processing.decontamination.enabled, "contaminated_count": 0, "contaminated_by_benchmark": {}},
    }
    rows: Iterator[Row] = _iter_filtered_rows(filtered_dir, filtered, shard_size)
    if processing.dedup.mode == "exact":
        rows = _exact_dedup(rows, processing.dedup.normalize, stats["dedup"])
    if processing.quality_filter:
        rows = _quality_filter(rows, stats["quality_filter"])
    if processing.decontamination.enabled:
        rows = _decontaminate(rows, processing.decontamination, num_workers, layout, stats["decontamination"])
    rows = _count_tokens(rows, counter, name, shard_size)
    if processing.dedup.mode == "minhash":
        rows = fuzzy_dedup(rows, processing.dedup, stats["dedup"], num_workers)
    token_sums: list[int] = []
    if source.repeat_to_budget:
        rows = _repeat_to_budget(list(rows), target_tokens, stats)
    rows = _accumulate_tokens(rows, token_sums, shard_size)
    write_dict_rows(rows, out, shard_size, start_shard=0)

    manifest = new_manifest(cfg, name, source_hash, "processed", tokens=True)
    manifest.rows_fetched = filtered.rows_fetched
    record_new_shards(manifest, out, 0, tokens=dict(zip((s.name for s in _sorted_shards(out)), token_sums)))
    manifest.extra = {"input_shards": input_shards, "target_tokens": target_tokens, "stats": stats}
    manifest.save(out)
    log.info("%s: %d rows, %s tokens", name, manifest.rows(), manifest.tokens())
    return manifest


def _sorted_shards(directory: Path) -> list[Path]:
    return [p for p in list_parquet_files(directory) if shard_index(p) is not None]


def _iter_filtered_rows(directory: Path, manifest: Manifest, batch_size: int) -> Iterator[Row]:
    for shard in manifest.shards:
        parquet = pq.ParquetFile(directory / shard.name)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text", "source"]):
            yield from batch.to_pylist()


def _exact_dedup(rows: Iterator[Row], normalize: bool, stats: dict[str, Any]) -> Iterator[Row]:
    """First occurrence wins; the hash set is built fresh on every run, so rerunning over old + new shards keeps
    exactly the rows the previous run kept from the old shards."""
    seen: set[str] = set()
    hasher = normalized_hash if normalize else md5_hex
    for row in rows:
        key = hasher(row["text"])
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        yield row


def _quality_filter(rows: Iterator[Row], stats: dict[str, Any]) -> Iterator[Row]:
    reasons: Counter[str] = Counter()
    for row in rows:
        passes, reason = check_quality(row["text"])
        if not passes:
            reasons[reason] += 1
            stats["filtered_count"] += 1
            stats["rejection_reasons"] = dict(reasons)
            continue
        yield row


# decontamination: the benchmark n-grams are loaded once per process (pool initializer for num_workers > 1)
_BENCHMARK_NGRAMS: dict[str, set[str]] = {}
_DECONTAM: dict[str, Any] = {}


def _init_decontamination(names: list[str], n: int, threshold: float, cache_dir: str) -> None:
    global _BENCHMARK_NGRAMS
    _BENCHMARK_NGRAMS = load_benchmark_ngrams(names, n, cache_dir)
    _DECONTAM.update({"n": n, "threshold": threshold})


def _contaminated_by(text: str) -> list[str]:
    return check_contamination(text, _BENCHMARK_NGRAMS, _DECONTAM["n"], _DECONTAM["threshold"])[1]


def _decontaminate(
    rows: Iterator[Row], config: DecontaminationConfig, num_workers: int, layout: DatasetLayout, stats: dict[str, Any]
) -> Iterator[Row]:
    cache_dir = str(layout.benchmark_cache_dir())
    args = (list(config.benchmarks), config.ngram, config.threshold, cache_dir)
    hits: Counter[str] = Counter()

    def account(row: Row, contaminated: list[str]) -> bool:
        if contaminated:
            hits.update(contaminated)
            stats["contaminated_count"] += 1
            stats["contaminated_by_benchmark"] = dict(hits)
            return False
        return True

    if num_workers <= 1:
        _init_decontamination(*args)
        for row in rows:
            if account(row, _contaminated_by(row["text"])):
                yield row
        return
    with multiprocessing.Pool(num_workers, initializer=_init_decontamination, initargs=args) as pool:
        for chunk in _chunks(rows, 1024):
            texts = [row["text"] for row in chunk]
            for row, contaminated in zip(chunk, pool.map(_contaminated_by, texts, chunksize=64)):
                if account(row, contaminated):
                    yield row


def _chunks(rows: Iterator[Row], size: int) -> Iterator[list[Row]]:
    chunk: list[Row] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _count_tokens(rows: Iterator[Row], counter: TokenCounter, name: str, batch_size: int) -> Iterator[Row]:
    for chunk in _chunks(rows, batch_size):
        tokens = counter.count_many([row["text"] for row in chunk])
        for row, n in zip(chunk, tokens):
            yield {"text": row["text"], "source": name, "tokens": n}


def _repeat_to_budget(rows: list[Row], target_tokens: int | None, stats: dict[str, Any]) -> Iterator[Row]:
    """Cycle through the kept rows until their token sum reaches ``target_tokens`` (``repeat_indices``)."""
    total = sum(int(r["tokens"]) for r in rows)
    stats["repeat_to_budget"] = {"unique_rows": len(rows), "unique_tokens": total, "target_tokens": target_tokens}
    if target_tokens is None or not rows or total <= 0 or total >= target_tokens:
        yield from rows
        return
    copies, remainder = divmod(target_tokens, total)
    extra_rows = 0
    running = 0
    while running < remainder:
        running += int(rows[extra_rows]["tokens"])
        extra_rows += 1
    indices = repeat_indices(len(rows), copies * len(rows) + extra_rows)
    stats["repeat_to_budget"]["repeated_rows"] = len(indices)
    for i in indices:
        yield rows[i]


def _accumulate_tokens(rows: Iterator[Row], sums: list[int], shard_size: int) -> Iterator[Row]:
    """Track the token sum per output shard (``write_dict_rows`` cuts exactly every ``shard_size`` rows)."""
    for count, row in enumerate(rows):
        if count % shard_size == 0:
            sums.append(0)
        sums[-1] += int(row["tokens"])
        yield row
