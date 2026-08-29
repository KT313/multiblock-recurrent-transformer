# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The ``process`` stage of ``pretrain`` sources (raw -> processed): length filter, exact dedup, quality filter,
decontamination, token counting, fuzzy dedup (see the ``fuzzy_dedup`` module).

Idempotent and incremental via the manifests (see ``stages/shared.py``): the processed manifest records the raw
shards it covers and every processed row carries its exact-dedup key (``hash`` column), so new raw shards are
deduplicated against the rows already on disk and only they are filtered and tokenized.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from data_preparation.lib.stages.benchmarks import load_benchmark_ngrams
from data_preparation.lib.storage.parquet import list_parquet_files, shard_index, text_hash64, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig, DecontaminationConfig, ProcessingConfig
from data_preparation.lib.stages.fuzzy_dedup import fuzzy_dedup
from data_preparation.lib.schema.layout import PROCESSED_COLUMNS, DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.ui.dashboard import progress
from data_preparation.lib.storage.manifest import Manifest, ShardInfo
from data_preparation.lib.stages.row_pipeline import check_contamination, check_quality, preprocess_batch
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


# --- process -----------------------------------------------------------------------------------------------------------

def process(
    cfg: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    num_workers: int = 1,
) -> Manifest:
    """Stream the raw shards not yet covered by the processed manifest, in order, through the length filter
    (``preprocess_batch``: drop null / short texts, truncate to ``max_chars``) -> quality filter -> decontamination
    -> exact dedup -> token count and **append** processed shards (``text``, ``source``, ``tokens``, ``hash``).
    ``hash`` is the row's 64-bit exact-dedup key (:func:`text_hash64`); the keys of every processed row already on
    disk are loaded first, so new rows are deduplicated against old ones and first occurrence still wins — the
    rows kept are exactly those of a full pass over every shard. Every kept row is written once; a source smaller
    than its budget is cycled by the training sampler, not repeated on disk.

    A no-op when the processed manifest already covers every raw shard (``extra["input_shards"]``). The directory
    is rebuilt from scratch when its manifest is stale, when the covered shards are no longer a prefix of the raw
    shards, when it predates the ``hash`` column, and — always — with ``dedup.mode: minhash`` (fuzzy dedup needs
    every signature in one LSH index, so it is a full pass over everything).
    """
    source = cfg.sources[name]
    if source.kind != "pretrain":
        raise ValueError(f"{name}: process() applies to pretrain sources only (kind={source.kind})")
    processing = cfg.source_processing(name)
    source_hash = cfg.source_hash(name)
    raw_dir = layout.source_dir(name, "raw")
    out = layout.source_dir(name, "processed")
    raw = ensure_raw_tokens(cfg, name, layout)  # upgrades a raw dir from before the tokens column in place
    if raw is None:
        raise FileNotFoundError(f"{name}: no current raw manifest in {raw_dir}; run the download stage first")

    full_pass = processing.dedup.mode == "minhash"
    manifest = _resumable_processed_manifest(cfg, name, source_hash, out, raw, full_pass)
    covered = len(manifest.extra["input_shards"])
    pending = raw.shards[covered:]
    if not pending:
        return manifest

    log.info("%s: processing %d raw shard(s) (%d already covered) -> %s", name, len(pending), covered, out)
    stats: dict[str, Any] = manifest.extra["stats"]
    stats["input_rows"] += sum(shard.rows for shard in pending)
    seen = _stored_hashes(out, manifest)

    # build the lazy pipeline: nothing runs until the shard writer pulls rows through it
    rows: Iterator[Row] = _length_filtered_rows(raw_dir, pending, source.text_field, name, processing, shard_size, stats["length_filter"])
    stats["tokens_recounted"] = 0
    # the row filters run before the dedup, so the hashes stored on disk are exactly the dedup's "seen" set and an
    # incremental run keeps the same rows as a full pass (a filtered-out row never claims a hash)
    if processing.quality_filter:
        rows = _quality_filter(rows, stats["quality_filter"])
    if processing.decontamination.enabled:
        rows = _decontaminate(rows, processing.decontamination, num_workers, layout, stats["decontamination"])
    rows = _with_hashes(rows, processing.dedup.normalize)
    if processing.dedup.mode == "exact":
        rows = _exact_dedup(rows, seen, stats["dedup"])
    pending_rows = sum(shard.rows for shard in pending)
    start_shard = len(manifest.shards)
    tokens_per_new_shard: list[int] = []
    with progress(total=pending_rows, desc=f"{name}: process", unit="row", leave=False) as bar:
        rows = _count_tokens(rows, TokenCounter(cfg, layout), name, processing.max_chars, shard_size, bar, stats)
        if full_pass:
            rows = fuzzy_dedup(rows, processing.dedup, stats["dedup"], num_workers)
        rows = _accumulate_tokens(rows, tokens_per_new_shard, shard_size)
        write_dict_rows(rows, out, shard_size, start_shard=start_shard)

    manifest.rows_fetched = raw.rows_fetched
    new_names = _shard_names(out)[start_shard:]
    record_new_shards(manifest, out, start_shard, tokens=dict(zip(new_names, tokens_per_new_shard)))
    manifest.extra["input_shards"] = shard_list(raw)
    manifest.save(out)
    log.info("%s: %d rows, %s tokens", name, manifest.rows(), manifest.tokens())
    return manifest


def _resumable_processed_manifest(
    cfg: DatasetConfig, name: str, source_hash: str, out: Path, raw: Manifest, full_pass: bool
) -> Manifest:
    """The stored processed manifest if new raw shards can be appended to it (current hash, ``hash`` column
    present, covered shards a prefix of the raw shards, not a full-pass mode); otherwise a fresh one, which makes
    the run rewrite the directory from shard 0."""
    manifest = current_manifest(out, source_hash, "processed")
    if manifest is not None and not full_pass:
        covered: list[list[Any]] = manifest.extra.get("input_shards", [])
        has_hashes = manifest.extra.get("columns") == list(PROCESSED_COLUMNS)
        if has_hashes and shard_list(raw)[: len(covered)] == covered:
            return manifest
        why = "raw shards changed under the processed manifest" if has_hashes else "processed shards predate the hash column"
        log.warning("%s: %s, reprocessing everything", name, why)
    processing = cfg.source_processing(name)
    manifest = new_manifest(cfg, name, source_hash, "processed", tokens=True)
    manifest.extra = {
        "input_shards": [],
        "columns": list(PROCESSED_COLUMNS),
        "stats": {
            "input_rows": 0,
            "tokens_recounted": 0,
            "length_filter": {"input_samples": 0, "removed_too_short": 0, "removed_invalid": 0, "truncated": 0, "output_samples": 0},
            "dedup": {"mode": processing.dedup.mode, "duplicates_removed": 0},
            "quality_filter": {"enabled": processing.quality_filter, "filtered_count": 0, "rejection_reasons": {}},
            "decontamination": {"enabled": processing.decontamination.enabled, "contaminated_count": 0, "contaminated_by_benchmark": {}},
        },
    }
    return manifest


def _stored_hashes(directory: Path, manifest: Manifest) -> set[int]:
    """The ``hash`` column of every processed shard the manifest lists (the exact-dedup keys already on disk)."""
    seen: set[int] = set()
    for shard in manifest.shards:
        column = pq.read_table(directory / shard.name, columns=["hash"]).column("hash")
        seen.update(cast(list[int], column.to_pylist()))  # int64 column without nulls
    return seen


def _shard_names(directory: Path) -> list[str]:
    """Names of the ``data-NNNNN.parquet`` shards in ``directory``, in index order."""
    return [p.name for p in list_parquet_files(directory) if shard_index(p) is not None]


def _length_filtered_rows(
    directory: Path,
    shards: list[ShardInfo],
    text_field: str,
    name: str,
    processing: ProcessingConfig,
    batch_size: int,
    stats: dict[str, int],
) -> Iterator[Row]:
    """``{text, source, original_length, tokens}`` rows of the given raw shards, in order, through the length filter
    (``preprocess_batch`` per Arrow batch: null / shorter than ``min_chars`` dropped, truncated to ``max_chars``);
    ``tokens`` is the raw count (of the untruncated text) and the per-batch statistics are summed into ``stats``."""
    for shard in shards:
        parquet = pq.ParquetFile(directory / shard.name)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_field, "tokens"]):
            kept, batch_stats = preprocess_batch(batch, text_field, name, processing.min_chars, processing.max_chars)
            for key, value in batch_stats.items():
                stats[key] += value
            yield from kept.to_pylist()


def _with_hashes(rows: Iterator[Row], normalize: bool) -> Iterator[Row]:
    """Add the 64-bit exact-dedup key of every row's text as ``hash``."""
    for row in rows:
        row["hash"] = text_hash64(row["text"], normalize)
        yield row


def _exact_dedup(rows: Iterator[Row], seen: set[int], stats: dict[str, Any]) -> Iterator[Row]:
    """First occurrence wins: drop rows whose ``hash`` is in ``seen`` (the keys of every processed row on disk plus
    the rows kept earlier in this pass), so appending shards keeps exactly the rows a full pass would keep."""
    for row in rows:
        key = row["hash"]
        if key in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(key)
        yield row


def _quality_filter(rows: Iterator[Row], stats: dict[str, Any]) -> Iterator[Row]:
    """Drop rows failing ``check_quality``; counts the rejections per reason in ``stats``."""
    for row in rows:
        passes, reason = check_quality(row["text"])
        if not passes:
            stats["filtered_count"] += 1
            _increment(stats["rejection_reasons"], reason)
            continue
        yield row


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


# decontamination: the benchmark n-grams are loaded once per process (pool initializer for num_workers > 1)
_BENCHMARK_NGRAMS: dict[str, set[str]] = {}
_DECONTAM: dict[str, Any] = {}


def _init_decontamination(names: list[str], n: int, threshold: float, cache_dir: str) -> None:
    global _BENCHMARK_NGRAMS
    _BENCHMARK_NGRAMS = load_benchmark_ngrams(names, n, cache_dir)
    _DECONTAM.update({"n": n, "threshold": threshold})


def _contaminated_by(text: str) -> list[str]:
    """Benchmarks ``text`` is contaminated by, using the process-global n-grams of ``_init_decontamination``."""
    return check_contamination(text, _BENCHMARK_NGRAMS, _DECONTAM["n"], _DECONTAM["threshold"])[1]


def _decontaminate(
    rows: Iterator[Row], config: DecontaminationConfig, num_workers: int, layout: DatasetLayout, stats: dict[str, Any]
) -> Iterator[Row]:
    """Drop rows contaminated by a benchmark; counts hits per benchmark in ``stats``."""
    init_args = (list(config.benchmarks), config.ngram, config.threshold, str(layout.benchmark_cache_dir()))
    for row, contaminated in _contamination_checks(rows, init_args, num_workers):
        if contaminated:
            stats["contaminated_count"] += 1
            for benchmark in contaminated:
                _increment(stats["contaminated_by_benchmark"], benchmark)
            continue
        yield row


def _contamination_checks(
    rows: Iterator[Row], init_args: tuple[list[str], int, float, str], num_workers: int
) -> Iterator[tuple[Row, list[str]]]:
    """``(row, contaminating benchmarks)`` for every row, checked in this process or in a pool of ``num_workers``."""
    if num_workers <= 1:
        _init_decontamination(*init_args)
        for row in rows:
            yield row, _contaminated_by(row["text"])
        return
    with multiprocessing.Pool(num_workers, initializer=_init_decontamination, initargs=init_args) as pool:
        for chunk in _chunks(rows, 1024):
            texts = [row["text"] for row in chunk]
            yield from zip(chunk, pool.map(_contaminated_by, texts, chunksize=64))


def _chunks(rows: Iterator[Row], size: int) -> Iterator[list[Row]]:
    """``rows`` grouped into lists of ``size`` (the last one may be shorter)."""
    chunk: list[Row] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _count_tokens(
    rows: Iterator[Row], counter: TokenCounter, name: str, max_chars: int, batch_size: int, bar: Progress, stats: dict[str, Any]
) -> Iterator[Row]:
    """The processed ``{text, source, tokens, hash}`` rows: the raw ``tokens`` count is reused, only rows the length
    filter truncated (``original_length > max_chars``) are recounted (``stats["tokens_recounted"]``). ``bar``
    advances per input row (input rows, not survivors, since the dedup / quality / decontamination generators
    upstream drop rows silently) with the running token sum as postfix."""
    total = 0
    for chunk in _chunks(rows, batch_size):
        truncated = [row for row in chunk if row["original_length"] > max_chars]
        for row, n in zip(truncated, counter.count_many([row["text"] for row in truncated])):
            row["tokens"] = n
        stats["tokens_recounted"] += len(truncated)
        total += sum(int(row["tokens"]) for row in chunk)
        bar.update(len(chunk))
        bar.set_postfix({"tokens": total}, refresh=False)
        for row in chunk:
            yield {"text": row["text"], "source": name, "tokens": int(row["tokens"]), "hash": row["hash"]}


def _accumulate_tokens(rows: Iterator[Row], sums: list[int], shard_size: int) -> Iterator[Row]:
    """Track the token sum per output shard (``write_dict_rows`` cuts exactly every ``shard_size`` rows)."""
    for count, row in enumerate(rows):
        if count % shard_size == 0:
            sums.append(0)
        sums[-1] += int(row["tokens"])
        yield row
