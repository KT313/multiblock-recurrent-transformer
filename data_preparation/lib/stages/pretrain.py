# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pipeline stages of ``pretrain`` sources: ``length_filter`` (raw -> filtered, shard by shard) and ``process``
(filtered -> processed: exact dedup, quality filter, decontamination, token counting, fuzzy dedup (see the
``fuzzy_dedup`` module), repetition).

Both are idempotent and incremental via the manifests (see ``stages/shared.py``). ``length_filter`` mirrors raw
shards 1:1, so only new raw shards are filtered. ``process`` rewrites the whole processed directory whenever the
filtered shard list changed, but by construction the rows it keeps from old shards are identical (first occurrence
wins in a fresh streaming pass over old + new shards).
"""

from __future__ import annotations

import multiprocessing
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.lib.stages.benchmarks import load_benchmark_ngrams
from data_preparation.lib.storage.parquet import (
    list_parquet_files,
    md5_hex,
    normalized_hash,
    shard_index,
    write_dict_rows,
    write_parquet_shards,
)
from data_preparation.lib.schema.dataset_config import DatasetConfig, DecontaminationConfig, ProcessingConfig
from data_preparation.lib.stages.fuzzy_dedup import fuzzy_dedup
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress, progress
from data_preparation.lib.storage.manifest import Manifest, ShardInfo, shard_rows
from data_preparation.lib.stages.row_pipeline import (
    FILTERED_SCHEMA,
    check_contamination,
    check_quality,
    preprocess_batch,
)
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

    manifest = _resumable_filtered_manifest(cfg, name, source_hash, out, raw)
    already_filtered: list[list[Any]] = manifest.extra["input_shards"]
    pending = raw.shards[len(already_filtered) :]
    if not pending:
        return manifest

    log.info("%s: length-filtering %d raw shard(s) -> %s", name, len(pending), out)
    for shard in progress(pending, desc=f"{name}: length_filter", unit="shard", leave=False):
        stats = _filter_one_shard(raw_dir / shard.name, out, shard, source.text_field, name, processing, batch_size)
        manifest.add_shard(shard.name, shard_rows(out / shard.name))
        manifest.extra["shard_stats"][shard.name] = stats
        manifest.extra["input_shards"].append([shard.name, shard.rows])
        manifest.rows_fetched = raw.rows_fetched
        manifest.save(out)
    return manifest


def _resumable_filtered_manifest(
    cfg: DatasetConfig, name: str, source_hash: str, out: Path, raw: Manifest
) -> Manifest:
    """The stored filtered manifest if its already-filtered raw shards are still a prefix of the raw manifest's
    shards (so only the remaining raw shards need filtering); otherwise a fresh one and everything is refiltered."""
    manifest = current_manifest(out, source_hash, "filtered")
    if manifest is not None:
        already_filtered: list[list[Any]] = manifest.extra.get("input_shards", [])
        if shard_list(raw)[: len(already_filtered)] == already_filtered:
            return manifest
        log.warning("%s: raw shards changed under the filtered manifest, refiltering everything", name)
    manifest = new_manifest(cfg, name, source_hash, "filtered")
    manifest.extra = {"input_shards": [], "shard_stats": {}}
    return manifest


def _filter_one_shard(
    raw_path: Path,
    out: Path,
    shard: ShardInfo,
    text_field: str,
    name: str,
    processing: ProcessingConfig,
    batch_size: int,
) -> dict[str, int]:
    """Write the filtered counterpart of one raw shard (same index; an empty shard if every row was dropped) and
    return the summed ``preprocess_batch`` statistics."""
    index = shard_index(Path(shard.name))
    if index is None:
        raise ValueError(f"{name}: raw manifest lists a non-shard file {shard.name!r}")
    stats: Counter[str] = Counter()

    def filtered_batches() -> Iterator[pa.RecordBatch]:
        for batch in pq.ParquetFile(raw_path).iter_batches(batch_size=batch_size):
            out_batch, batch_stats = preprocess_batch(batch, text_field, name, processing.min_chars, processing.max_chars)
            stats.update(batch_stats)
            if len(out_batch) > 0:
                yield out_batch

    written = write_parquet_shards(filtered_batches(), out, shard_size=max(shard.rows, 1), start_shard=index)
    if written == 0:  # every row was dropped: keep the 1:1 numbering with an empty shard
        _write_empty_shard(out / shard.name)
    return dict(stats)


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
) -> Manifest:
    """Stream all filtered shards in order through exact dedup -> quality filter -> decontamination -> token count
    -> fuzzy dedup and write processed shards (``text``, ``source``, ``tokens``). Every kept row is written once;
    a source smaller than its budget is cycled by the training sampler, not repeated on disk.

    Skipped entirely when the filtered shard list matches what the processed manifest recorded; otherwise the
    processed directory is rewritten from shard 0.
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
    if existing is not None and existing.extra.get("input_shards") == input_shards:
        return existing

    log.info("%s: processing %d filtered shard(s) -> %s", name, len(filtered.shards), out)
    counter = TokenCounter(cfg, layout)
    # `stats` is stored in the manifest; the generators below update their sub-dict as rows stream through
    stats: dict[str, Any] = {
        "input_rows": filtered.rows(),
        "dedup": {"mode": processing.dedup.mode, "duplicates_removed": 0},
        "quality_filter": {"enabled": processing.quality_filter, "filtered_count": 0, "rejection_reasons": {}},
        "decontamination": {"enabled": processing.decontamination.enabled, "contaminated_count": 0, "contaminated_by_benchmark": {}},
    }

    # build the lazy pipeline: nothing runs until `write_dict_rows` pulls rows through it
    rows: Iterator[Row] = _iter_filtered_rows(filtered_dir, filtered, shard_size)
    if processing.dedup.mode == "exact":
        rows = _exact_dedup(rows, processing.dedup.normalize, stats["dedup"])
    if processing.quality_filter:
        rows = _quality_filter(rows, stats["quality_filter"])
    if processing.decontamination.enabled:
        rows = _decontaminate(rows, processing.decontamination, num_workers, layout, stats["decontamination"])
    tokens_per_output_shard: list[int] = []
    with progress(total=filtered.rows(), desc=f"{name}: process", unit="row", leave=False) as bar:
        rows = _count_tokens(rows, counter, name, shard_size, bar)
        if processing.dedup.mode == "minhash":
            rows = fuzzy_dedup(rows, processing.dedup, stats["dedup"], num_workers)
        rows = _accumulate_tokens(rows, tokens_per_output_shard, shard_size)
        write_dict_rows(rows, out, shard_size, start_shard=0)

    manifest = new_manifest(cfg, name, source_hash, "processed", tokens=True)
    manifest.rows_fetched = filtered.rows_fetched
    record_new_shards(manifest, out, 0, tokens=dict(zip(_shard_names(out), tokens_per_output_shard)))
    manifest.extra = {"input_shards": input_shards, "stats": stats}
    manifest.save(out)
    log.info("%s: %d rows, %s tokens", name, manifest.rows(), manifest.tokens())
    return manifest


def _shard_names(directory: Path) -> list[str]:
    """Names of the ``data-NNNNN.parquet`` shards in ``directory``, in index order."""
    return [p.name for p in list_parquet_files(directory) if shard_index(p) is not None]


def _iter_filtered_rows(directory: Path, manifest: Manifest, batch_size: int) -> Iterator[Row]:
    """``{text, source}`` rows of every filtered shard, in manifest order."""
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


def _count_tokens(rows: Iterator[Row], counter: TokenCounter, name: str, batch_size: int, bar: Progress) -> Iterator[Row]:
    """Token-count ``rows`` in batches; ``bar`` advances per input row (input rows, not survivors, since the dedup /
    quality / decontamination generators upstream drop rows silently) with the running token sum as postfix."""
    total = 0
    for chunk in _chunks(rows, batch_size):
        tokens = counter.count_many([row["text"] for row in chunk])
        total += sum(tokens)
        bar.update(len(chunk))
        bar.set_postfix({"tokens": total}, refresh=False)
        for row, n in zip(chunk, tokens):
            yield {"text": row["text"], "source": name, "tokens": n}


def _accumulate_tokens(rows: Iterator[Row], sums: list[int], shard_size: int) -> Iterator[Row]:
    """Track the token sum per output shard (``write_dict_rows`` cuts exactly every ``shard_size`` rows)."""
    for count, row in enumerate(rows):
        if count % shard_size == 0:
            sums.append(0)
        sums[-1] += int(row["tokens"])
        yield row
