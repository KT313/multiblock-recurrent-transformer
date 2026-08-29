# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pipeline stages shared by every source kind: tokenizer, raw download, held-out sets; plus the token counter and
manifest helpers used by ``stages/pretrain.py`` / ``stages/instruct.py``.

Every stage is a function ``(cfg, name, layout, *options) -> Manifest`` that is **idempotent via the manifest**
(a second call with nothing new returns the stored manifest without touching the shards) and **incremental** where
the data allow it. A stored manifest whose ``source_hash`` differs from the current config is stale: the stage logs
a warning and rebuilds the directory from scratch.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from data_preparation.lib.storage.parquet import estimate_tokens, list_parquet_files, shard_index, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress, progress
from data_preparation.lib.storage.manifest import Manifest, library_versions, shard_rows
from data_preparation.lib.sources.loaders import MAX_CACHED_FILE_KEY
from data_preparation.lib.sources import (
    FetchStats,
    Row,
    get_converter,
    get_filter,
    get_loader,
    list_local_files,
    write_synthetic_tokenizer,
)

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000

# --- token counting ----------------------------------------------------------------------------------------------------


class TokenCounter:
    """``min(tokens(text), max_seq_length)`` with the config's tokenizer (``token_count: tokenizer``) or chars/4.

    The text itself is never rewritten; training's own truncation cuts at ``block_size``.
    """

    def __init__(self, cfg: DatasetConfig, layout: DatasetLayout) -> None:
        self.mode = cfg.token_count
        self.cap = cfg.max_seq_length
        self.tokenizer_name = cfg.tokenizer.name
        self._tokenizer: Any = None
        if self.mode == "tokenizer":
            self._tokenizer = _load_tokenizer(layout.tokenizer_dir(cfg.tokenizer.name), cfg.tokenizer.name)

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return min(estimate_tokens(text), self.cap)
        return min(len(self._tokenizer.encode(text, add_special_tokens=False)), self.cap)

    def count_many(self, texts: list[str]) -> list[int]:
        if self._tokenizer is None:
            return [min(estimate_tokens(t), self.cap) for t in texts]
        encoded = self._tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [min(len(ids), self.cap) for ids in encoded]


def _load_tokenizer(tokenizer_dir: Path, name: str) -> Any:
    """The saved HF tokenizer in ``tokenizer_dir``; fails if the tokenizer stage has not run yet."""
    has_tokenizer_files = (tokenizer_dir / "tokenizer.json").is_file() or (tokenizer_dir / "tokenizer_config.json").is_file()
    if not has_tokenizer_files:
        raise FileNotFoundError(f"tokenizer {name!r} not found at {tokenizer_dir}; run the tokenizer stage first")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


# --- manifest helpers --------------------------------------------------------------------------------------------------


def current_manifest(directory: Path, source_hash: str, stage: str) -> Manifest | None:
    """The stored manifest if it matches ``source_hash`` and ``stage``; None (with a warning) if stale or absent."""
    manifest = Manifest.load(directory)
    if manifest is None:
        return None
    if manifest.stage != stage:
        log.warning("%s: manifest stage %r != %r, rebuilding", directory, manifest.stage, stage)
        return None
    if not manifest.is_current(source_hash):
        log.warning("%s: stored hash %s != current %s, rebuilding from scratch", directory, manifest.source_hash, source_hash)
        return None
    return manifest


def new_manifest(cfg: DatasetConfig, source: str, source_hash: str, stage: str, *, tokens: bool = False) -> Manifest:
    """An empty manifest for ``stage``; with ``tokens`` it records how token counts are measured."""
    return Manifest(
        source=source,
        source_hash=source_hash,
        stage=stage,
        token_count=cfg.token_count if tokens else None,
        tokenizer=cfg.tokenizer.name if tokens and cfg.token_count == "tokenizer" else None,
        versions=library_versions(),
    )


def record_new_shards(manifest: Manifest, directory: Path, start_shard: int, tokens: dict[str, int] | None = None) -> None:
    """Add every ``data-NNNNN.parquet`` with index >= ``start_shard`` to ``manifest`` (row counts from the footer)."""
    for path in list_parquet_files(directory):
        index = shard_index(path)
        if index is None or index < start_shard:
            continue
        shard_tokens = tokens.get(path.name) if tokens else None
        manifest.add_shard(path.name, shard_rows(path), shard_tokens)


def shard_list(manifest: Manifest) -> list[list[Any]]:
    """``[[name, rows], ...]`` — the JSON-friendly identity of a manifest's shards (stored as ``extra["input_shards"]``)."""
    return [[s.name, s.rows] for s in manifest.shards]


def require_manifest(directory: Path, source_hash: str, stage: str, what: str) -> Manifest:
    """Like ``current_manifest`` but a missing or stale manifest is an error (the previous stage has to run first)."""
    manifest = current_manifest(directory, source_hash, stage)
    if manifest is None:
        raise FileNotFoundError(f"{what}: no current {stage} manifest in {directory}; run the {stage} stage first")
    return manifest


def text_row(source: SourceConfig, row: Row, name: str) -> Row:
    """Apply a pretrain/validation source's converter (if any) and check that ``text_field`` is present."""
    converter = get_converter(source)
    if converter is not None:
        row = converter(row)
    if source.text_field not in row:
        raise ValueError(f"{name}: row has no {source.text_field!r} column; columns: {sorted(row)}")
    return row


# --- tokenizer ---------------------------------------------------------------------------------------------------------


def prepare_tokenizer(cfg: DatasetConfig, layout: DatasetLayout) -> Manifest:
    """Save the config's tokenizer to ``layout.tokenizer_dir(name)`` (Hub download or the synthetic WordLevel one)."""
    tok = cfg.tokenizer
    out = layout.tokenizer_dir(tok.name)
    source_hash = cfg.tokenizer_hash()
    manifest = current_manifest(out, source_hash, "tokenizer")
    if manifest is not None and (out / "tokenizer_config.json").is_file():
        return manifest

    log.info("preparing tokenizer %s (%s) -> %s", tok.name, tok.kind, out)
    out.mkdir(parents=True, exist_ok=True)
    if tok.kind == "synthetic":
        write_synthetic_tokenizer(out)
    else:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tok.hf_id, revision=tok.revision).save_pretrained(str(out))
    manifest = new_manifest(cfg, tok.name, source_hash, "tokenizer")
    manifest.extra = {"kind": tok.kind, "hf_id": tok.hf_id, "revision": tok.revision}
    manifest.save(out)
    return manifest


# --- download ----------------------------------------------------------------------------------------------------------


def fetch_source(cfg: DatasetConfig, source: SourceConfig) -> SourceConfig:
    """The source as handed to its loader: with ``cfg.always_range_requests`` every Hub file is read remotely by
    piece (``max_cached_file_mb`` forced to 0), otherwise the source's own threshold applies."""
    if cfg.always_range_requests and source.loader in ("hf_files", "github_code"):
        return replace(source, load_kwargs={**source.load_kwargs, MAX_CACHED_FILE_KEY: 0})
    return source


@dataclass
class _FetchCounters:
    """What one download increment did so far (updated while ``_fetch_rows`` runs, read back by ``download``)."""

    consumed: int = 0  # source rows the loader yielded (the loader offset advances by this much)
    kept: int = 0  # rows written to disk
    skipped_malformed: int = 0  # instruct rows whose converter raised ValueError
    exhausted: bool = False  # the loader ran dry, or ``check_limit`` was reached


def download(
    cfg: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    *,
    rows_needed: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    hf_token: str | None = None,
) -> Manifest:
    """Append raw shards until ``rows_needed`` rows are on disk (no-op if they already are).

    ``manifest.rows_fetched`` is the loader offset reached (source rows consumed); for pretrain/validation sources
    every row is kept (converter applied, ``text_field`` guaranteed), for instruct sources the converter and filter
    run at download time and only standardized ``{instruction, input, output}`` rows are stored — malformed rows
    (converter raises ``ValueError``) are skipped and counted in ``extra["skipped_malformed"]``; ``check_limit``
    bounds the number of source rows inspected in total. A loader that yields fewer rows than requested sets
    ``extra["exhausted"]`` (a source smaller than its budget is cycled by the training sampler).

    ``rows_needed`` is a minimum: a loader reading a large parquet file remotely finishes the row group it is in
    (see ``sources/loaders.py``), **every** row it yields is written and ``rows_fetched`` advances to that row-group
    boundary, so a later call with a ``rows_needed`` at or below the rows on disk is a no-op and a top-up beyond it
    starts at the boundary — the same bytes are never downloaded twice. Sources without a converter are read with
    only ``text_field`` projected (``columns``); converters and ``fields`` mappings get every column.
    """
    source = fetch_source(cfg, cfg.sources[name])
    source_hash = cfg.source_hash(name)
    out = layout.source_dir(name, "raw")
    manifest = current_manifest(out, source_hash, "raw") or new_manifest(cfg, name, source_hash, "raw")

    # nothing to do?
    if manifest.extra.get("exhausted"):
        log.info("%s: source exhausted after %d rows, nothing more to fetch", name, manifest.rows_fetched)
        return manifest
    wanted = rows_needed - manifest.rows()
    if wanted <= 0:
        return manifest
    max_consume = None if source.check_limit is None else source.check_limit - manifest.rows_fetched
    if max_consume is not None and max_consume <= 0:
        manifest.extra["exhausted"] = True
        manifest.save(out)
        return manifest

    # fetch one increment and append it as new shards
    log.info("%s: fetching %d rows from offset %d -> %s", name, wanted, manifest.rows_fetched, out)
    counters = _FetchCounters()
    start_shard = len(manifest.shards)
    # the bar's total is the minimum; it overshoots (e.g. 1000/11) when the loader finishes a remote row group
    with progress(total=wanted, desc=f"{name}: download", unit="row") as bar:
        rows = _fetch_rows(source, name, manifest.rows_fetched, wanted, max_consume, hf_token, counters, layout, bar)
        write_dict_rows(rows, out, shard_size, start_shard=start_shard)

    # record the increment
    record_new_shards(manifest, out, start_shard)
    manifest.rows_fetched += counters.consumed  # source rows consumed: a row-group boundary after an over-read
    manifest.extra["skipped_malformed"] = manifest.extra.get("skipped_malformed", 0) + counters.skipped_malformed
    if counters.exhausted:
        manifest.extra["exhausted"] = True
    manifest.save(out)
    log.info("%s: kept %d of %d fetched rows (%d rows on disk)", name, counters.kept, counters.consumed, manifest.rows())
    return manifest


def _fetch_rows(
    source: SourceConfig,
    name: str,
    offset: int,
    wanted: int,
    max_consume: int | None,
    hf_token: str | None,
    counters: _FetchCounters,
    layout: DatasetLayout,
    bar: Progress,
) -> Iterator[Row]:
    """Rows to store for one download increment; keeps calling the loader until at least ``wanted`` rows are kept,
    the source is exhausted or ``max_consume`` source rows were inspected (instruct filters may drop rows, so one
    loader call may not be enough). Everything a loader yields is kept — it may finish a remote row group beyond
    ``count`` — except that consumption stops at ``max_consume``. ``bar`` tracks kept rows (postfix: source rows
    consumed, current repo file, MB read remotely)."""
    loader = get_loader(source.loader)
    is_instruct = source.kind == "instruct"
    converter = get_converter(source) if is_instruct else None
    row_filter = get_filter(source.filter) if source.filter is not None else None
    if is_instruct and converter is None and source.loader != "synthetic":
        raise ValueError(f"{name}: instruct source needs `fields` or `converter`")
    columns = loader_columns(source)
    fetch_stats = FetchStats()
    postfix = _DownloadPostfix(bar, fetch_stats)

    while counters.kept < wanted:
        # how many rows to ask the loader for in this round
        count = wanted - counters.kept
        if max_consume is not None:
            count = min(count, max_consume - counters.consumed)
            if count <= 0:
                counters.exhausted = True
                return
        consume_budget = None if max_consume is None else max_consume - counters.consumed

        rows = loader(
            source, offset + counters.consumed, count, token=hf_token, index_dir=layout.hub_index_dir(), columns=columns,
            on_file=postfix.on_file, stats=fetch_stats, align_to_row_group=True,
        )
        yielded = 0
        for raw in _bounded(rows, consume_budget):
            yielded += 1
            counters.consumed += 1
            postfix.consumed(counters.consumed)

            if not is_instruct:
                yield text_row(source, raw, name)
                counters.kept += 1
                bar.update(1)
                continue

            if row_filter is not None and not row_filter(raw):
                continue
            try:
                row = _instruct_row(raw, converter)
            except ValueError as err:
                counters.skipped_malformed += 1
                log.debug("%s: skipping malformed row: %s", name, err)
                continue
            yield row
            counters.kept += 1
            bar.update(1)

        if yielded < count:
            counters.exhausted = True
            return


def _instruct_row(raw: Row, converter: Callable[[Row], Row] | None) -> Row:
    """The standardized ``{instruction, input, output}`` row for ``raw``.

    A ``ValueError`` raised by the converter (malformed row) propagates to the caller, which skips the row.
    """
    row = converter(raw) if converter is not None else dict(raw)
    return {"instruction": row["instruction"], "input": row.get("input", ""), "output": row["output"]}


class _DownloadPostfix:
    """The download bar's postfix: source rows consumed, current repo file, MB read remotely (refreshed sparsely)."""

    def __init__(self, bar: Progress, fetch_stats: FetchStats) -> None:
        self._bar = bar
        self._fetch_stats = fetch_stats
        self._values: dict[str, Any] = {"consumed": 0}

    def on_file(self, file: str) -> None:
        """Loader callback: a new repo file is being read."""
        self._values["file"] = file.rsplit("/", 1)[-1]
        self._refresh()

    def consumed(self, total: int) -> None:
        """Record the running count of consumed source rows; the bar is refreshed every 100 rows."""
        self._values["consumed"] = total
        if total % 100 == 0:
            self._refresh()

    def _refresh(self) -> None:
        if self._fetch_stats.bytes_read:
            self._values["MB"] = f"{self._fetch_stats.bytes_read / 2**20:.0f}"
        self._bar.set_postfix(self._values, refresh=False)


def loader_columns(source: SourceConfig) -> list[str] | None:
    """Parquet column projection for a source's loader: ``[text_field]`` for pretrain/validation sources read as-is,
    None (every column) when a converter or ``fields`` mapping may need others or the rows are instruct rows."""
    if source.kind == "instruct" or get_converter(source) is not None:
        return None
    return [source.text_field]


def _bounded(rows: Iterator[Row], limit: int | None) -> Iterator[Row]:
    """``rows`` up to ``limit`` (None: all), closing the loader's generator when stopping early."""
    if limit is None:
        yield from rows
        return
    taken = 0
    try:
        for row in rows:
            if taken >= limit:
                return
            yield row
            taken += 1
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()


# --- validation -----------------------------------------------------------------------------------------------------------


def validation(
    cfg: DatasetConfig, name: str, layout: DatasetLayout, *, shard_size: int = DEFAULT_SHARD_SIZE
) -> Manifest:
    """Write the held-out validation rows of a ``validation`` source: ``source.rows`` rows, shuffled with
    ``random.Random(source.seed)``, token-counted like ``process`` — no dedup and no filters.

    Disjointness from the training data is the config author's job and depends on the loader:

    * ``hf_split``: the config picks a split / ``load_kwargs`` disjoint from every training source (the crow config
      uses fineweb-edu's ``sample-10BT`` subset while training reads the ``CC-MAIN-*`` dumps); rows ``[0:rows]``.
    * ``synthetic``: the source's own ``seed`` (different from the training source) generates different rows.
    * ``local``: the **last** ``rows`` rows of the directory, so a training source reading the first rows of the
      same directory stays disjoint as long as it needs fewer than ``total - rows`` rows.
    * ``hf_files`` / ``hf_stream``: the **first** ``rows`` rows of the configured files / stream are taken, so the
      config must point them at files disjoint from every training source (a different ``data_files`` glob).
    """
    source = fetch_source(cfg, cfg.sources[name])
    if source.kind != "validation" or source.rows is None:
        raise ValueError(f"{name}: validation() needs a source of kind validation with rows > 0")
    source_hash = cfg.source_hash(name)
    out = layout.validation_dir(name)
    existing = current_manifest(out, source_hash, "validation")
    if existing is not None:
        return existing

    # fetch the rows
    offset = _validation_offset(source, source.rows)
    log.info("%s: holding out %d rows from offset %d -> %s", name, source.rows, offset, out)
    loader = get_loader(source.loader)
    with progress(total=source.rows, desc=f"{name}: validation", unit="row", leave=False) as bar:
        fetched = loader(  # exact: a validation is fetched once, its row count is part of its identity
            source, offset, source.rows, index_dir=layout.hub_index_dir(), columns=loader_columns(source),
            align_to_row_group=False,
        )
        rows = [text_row(source, r, name) for r in bar_rows(bar, fetched)]
    if len(rows) < source.rows:
        log.warning("%s: only %d of %d requested validation rows available", name, len(rows), source.rows)

    # shuffle, count tokens, write
    random.Random(source.seed).shuffle(rows)
    texts = [str(r[source.text_field]) for r in rows]
    tokens = TokenCounter(cfg, layout).count_many(texts)
    out_rows = ({"text": t, "source": name, "tokens": n} for t, n in zip(texts, tokens))
    write_dict_rows(out_rows, out, shard_size, start_shard=0)

    manifest = new_manifest(cfg, name, source_hash, "validation", tokens=True)
    manifest.rows_fetched = offset + len(rows)
    record_new_shards(manifest, out, 0, tokens=_tokens_per_shard(out, tokens))
    manifest.extra = {"offset": offset, "requested_rows": source.rows, "seed": source.seed}
    manifest.save(out)
    return manifest


def _validation_offset(source: SourceConfig, rows: int) -> int:
    """Where the ``rows`` held-out rows start: the tail of a ``local`` directory, the beginning of everything else."""
    if source.loader != "local":
        return 0
    total = _local_row_count(Path(str(source.path)))
    return max(total - rows, 0)


def bar_rows(bar: Progress, rows: Iterator[Row]) -> Iterator[Row]:
    """Pass ``rows`` through, advancing ``bar`` by one per row."""
    for row in rows:
        bar.update(1)
        yield row


def _local_row_count(directory: Path) -> int:
    """Rows in a ``local`` source directory: parquet footers plus non-blank lines of every other (jsonl) file."""
    total = 0
    for file in list_local_files(directory):
        if file.suffix == ".parquet":
            total += shard_rows(file)
        else:
            with file.open(encoding="utf-8") as fh:
                total += sum(1 for line in fh if line.strip())
    return total


def _tokens_per_shard(directory: Path, tokens: list[int]) -> dict[str, int]:
    """Split a per-row token list into per-shard sums following the shard row counts on disk."""
    per_shard: dict[str, int] = {}
    position = 0
    for path in list_parquet_files(directory):
        if shard_index(path) is None:
            continue
        rows = shard_rows(path)
        per_shard[path.name] = sum(tokens[position : position + rows])
        position += rows
    return per_shard
