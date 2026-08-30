# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pipeline stages shared by every source kind: tokenizer, raw download, held-out sets; plus the token counter and
manifest helpers used by ``stages/pretrain.py`` / ``stages/instruct.py``.

Every stage is a function ``(cfg, name, layout, *options) -> Manifest`` that is **idempotent via the manifest**
(a second call with nothing new returns the stored manifest without touching the shards) and **incremental** where
the data allow it. A stored manifest whose hash differs from the stage's current key (``DatasetConfig.raw_hash`` for
``raw/``: loader identity only; ``processed_hash`` / ``validation_hash`` for the derived directories) is stale: the
stage logs a warning and rebuilds the directory from scratch.
"""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.lib.storage.parquet import (
    SHARD_COMPRESSION,
    ShardWriter,
    estimate_tokens,
    list_parquet_files,
    shard_index,
    write_dict_rows,
)
from data_preparation.lib.schema.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.ui.dashboard import progress
from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.storage.manifest import Manifest, has_shards, library_versions, shard_problem, shard_rows, shard_tokens
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.sources.loaders import MAX_CACHED_FILE_KEY
from data_preparation.lib.sources import (
    FetchStats,
    GithubCodeRequest,
    Row,
    get_converter,
    get_filter,
    get_loader,
    github_code_repo_key,
    list_local_files,
    read_github_code_group,
    write_synthetic_tokenizer,
)

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000

# --- token counting ----------------------------------------------------------------------------------------------------


class TokenCounter:
    """``min(tokens(text), cap)`` with the config's tokenizer (``token_count: tokenizer``) or chars/4.

    ``cap`` defaults to ``max_seq_length`` (pretrain / validation documents: training truncates them there; the text
    itself is never rewritten) and is None for instruct examples (:meth:`DatasetConfig.token_cap`), whose full
    length decides whether ``mixture.max_tokens`` drops them. Use :meth:`for_source` to pick the source's cap.
    """

    def __init__(self, cfg: DatasetConfig, layout: DatasetLayout, *, cap: int | None = None) -> None:
        self.mode = cfg.token_count
        self.cap = cap
        self.tokenizer_name = cfg.tokenizer.name
        self._tokenizer: Any = None
        if self.mode == "tokenizer":
            self._tokenizer = _load_tokenizer(layout.tokenizer_dir(cfg.tokenizer.name), cfg.tokenizer.name)

    @classmethod
    def for_source(cls, cfg: DatasetConfig, layout: DatasetLayout, source_name: str) -> TokenCounter:
        return cls(cfg, layout, cap=cfg.token_cap(source_name))

    def _capped(self, n: int) -> int:
        return n if self.cap is None else min(n, self.cap)

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return self._capped(estimate_tokens(text))
        return self._capped(len(self._tokenizer.encode(text, add_special_tokens=False)))

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []  # HF fast tokenizers choke on an empty batch
        if self._tokenizer is None:
            return [self._capped(estimate_tokens(t)) for t in texts]
        encoded = self._tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [self._capped(len(ids)) for ids in encoded]


def _load_tokenizer(tokenizer_dir: Path, name: str) -> Any:
    """The saved HF tokenizer in ``tokenizer_dir``; fails if the tokenizer stage has not run yet."""
    has_tokenizer_files = (tokenizer_dir / "tokenizer.json").is_file() or (tokenizer_dir / "tokenizer_config.json").is_file()
    if not has_tokenizer_files:
        raise FileNotFoundError(f"tokenizer {name!r} not found at {tokenizer_dir}; run the tokenizer stage first")
    return _auto_tokenizer().from_pretrained(str(tokenizer_dir))


_IMPORT_LOCK = threading.Lock()


def _auto_tokenizer() -> Any:
    """``transformers.AutoTokenizer``, imported lazily (the HF cache env must be configurable before the import)
    and under a lock: ``transformers`` initialises its lazy modules on first import, which is not thread-safe and
    the build runs items in threads."""
    with _IMPORT_LOCK:
        from transformers import AutoTokenizer

    return AutoTokenizer


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
    """An empty manifest for ``stage``; with ``tokens`` it records how token counts are measured (the cap is the
    source's, see :meth:`DatasetConfig.token_cap`; mixtures are named like no source and carry no cap)."""
    return Manifest(
        source=source,
        source_hash=source_hash,
        stage=stage,
        token_count=cfg.token_count if tokens else None,
        tokenizer=cfg.tokenizer.name if tokens and cfg.token_count == "tokenizer" else None,
        token_cap=cfg.token_cap(source) if tokens and source in cfg.sources else None,
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
        _auto_tokenizer().from_pretrained(tok.hf_id, revision=tok.revision).save_pretrained(str(out))
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
    should_stop: StopCheck | None = None,
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

    Every stored row gets a ``tokens`` column (:class:`TokenCounter` over ``text_field``, or instruction + input +
    output for instruct rows), counted once here and reused by ``process`` and the instruct mixtures; the manifest
    records the mode / tokenizer and per-shard sums. A raw directory from before this column is upgraded in place
    (:func:`ensure_raw_tokens`), never re-downloaded.

    Every shard is published and recorded in the manifest (with the loader offset after its last row,
    ``ShardInfo.offset``) as soon as it is written, so a failure or a stop request (``should_stop``, checked after
    every shard) keeps everything fetched so far and the next call resumes from the last complete shard.
    """
    source = fetch_source(cfg, cfg.sources[name])
    source_hash = cfg.raw_hash(name)
    out = layout.source_dir(name, "raw")
    manifest = ensure_raw_tokens(cfg, name, layout) or _fresh_raw_manifest(cfg, name, source_hash, out)

    # nothing to do?
    _reset_check_limit_exhaustion(manifest, source, name)
    if manifest.extra.get("exhausted"):
        log.info("%s: source exhausted after %d rows, nothing more to fetch", name, manifest.rows_fetched)
        return manifest
    wanted = rows_needed - manifest.rows()
    if wanted <= 0:
        return manifest
    max_consume = None if source.check_limit is None else source.check_limit - manifest.rows_fetched
    if max_consume is not None and max_consume <= 0:
        manifest.extra["exhausted"] = True
        manifest.extra["check_limit"] = source.check_limit
        manifest.save(out)
        return manifest

    # fetch one increment and append it shard by shard
    log.info("%s: fetching %d rows from offset %d -> %s", name, wanted, manifest.rows_fetched, out)
    counters = _FetchCounters()
    increment = _Increment(manifest, out, counters, should_stop)
    counter = TokenCounter.for_source(cfg, layout, name)
    # the bar's total is the minimum; it overshoots (e.g. 1000/11) when the loader finishes a remote row group
    with progress(total=wanted, desc=f"{name}: download", unit="row") as bar:
        rows = _fetch_rows(source, name, manifest.rows_fetched, wanted, max_consume, hf_token, counters, layout, bar)
        rows = _with_tokens(rows, counter, raw_text_of(source))
        with ShardWriter(out, shard_size, start_shard=len(manifest.shards), on_shard=increment.record_shard) as writer:
            for row in rows:
                increment.add(writer, row)

    increment.finish(source)
    log.info("%s: kept %d of %d fetched rows (%d rows on disk)", name, counters.kept, counters.consumed, manifest.rows())
    return manifest


def _fresh_raw_manifest(cfg: DatasetConfig, name: str, source_hash: str, out: Path) -> Manifest:
    """An empty raw manifest to start the directory from shard 0 — refused when ``out`` holds shards without any
    manifest: nothing would say where those rows came from, and starting over would delete them. A *stale*
    manifest (the source itself changed) is a rebuild, see ``current_manifest``."""
    if Manifest.load(out) is None and has_shards(out):
        raise RuntimeError(f"{name}: {out} holds shards but no manifest; delete the directory to download the source again")
    return new_manifest(cfg, name, source_hash, "raw", tokens=True)


def _reset_check_limit_exhaustion(manifest: Manifest, source: SourceConfig, name: str) -> None:
    """A source marked exhausted because its ``check_limit`` was reached may be read further when the limit grew
    (or was removed): ``check_limit`` is not part of the raw hash, so the manifest is not stale, only its flag."""
    reached = manifest.extra.get("check_limit")
    if not manifest.extra.get("exhausted") or reached is None:
        return
    if source.check_limit is None or source.check_limit > int(reached):
        log.info("%s: check_limit grew from %s to %s, source no longer exhausted", name, reached, source.check_limit)
        manifest.extra["exhausted"] = False
        del manifest.extra["check_limit"]


CONSUMED_KEY = "_consumed"  # private row key: loader offset after this row (stripped before the row is written)


class _Increment:
    """Bookkeeping of one download increment of a raw directory: every published shard is recorded in the manifest
    with the loader offset after its last row and the manifest is saved, so the increment is resumable at shard
    granularity; then the stop request is checked."""

    def __init__(self, manifest: Manifest, out: Path, counters: _FetchCounters, should_stop: StopCheck | None) -> None:
        self.manifest = manifest
        self.out = out
        self.counters = counters
        self.should_stop = should_stop
        self.start_offset = manifest.rows_fetched
        self.skipped_before = int(manifest.extra.get("skipped_malformed", 0))
        self.last_consumed = 0  # consumed count of the row most recently handed to the writer

    def add(self, writer: ShardWriter, row: Row) -> None:
        """Hand ``row`` (tagged with :data:`CONSUMED_KEY` by ``_fetch_rows``) to ``writer``."""
        self.last_consumed = int(row.pop(CONSUMED_KEY))
        writer.add(row)

    def record_shard(self, path: Path) -> None:
        self.manifest.add_shard(path.name, shard_rows(path), shard_tokens(path), offset=self.start_offset + self.last_consumed)
        self._save()
        check_stop(self.should_stop)

    def finish(self, source: SourceConfig) -> None:
        """After the loader ran dry / the target was reached: the final offset and the exhaustion flag."""
        if self.counters.exhausted:
            self.manifest.extra["exhausted"] = True
            if source.check_limit is not None and self.start_offset + self.counters.consumed >= source.check_limit:
                self.manifest.extra["check_limit"] = source.check_limit  # exhausted by the limit, not by the loader
        self.last_consumed = self.counters.consumed
        self._save()

    def _save(self) -> None:
        self.manifest.rows_fetched = self.start_offset + self.last_consumed
        self.manifest.extra["skipped_malformed"] = self.skipped_before + self.counters.skipped_malformed
        self.manifest.save(self.out)


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
                row = text_row(source, raw, name)
                row[CONSUMED_KEY] = counters.consumed
                yield row
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
            row[CONSUMED_KEY] = counters.consumed
            yield row
            counters.kept += 1
            bar.update(1)

        if yielded < count:
            counters.exhausted = True
            return


@dataclass
class _GroupMember:
    """One source of a :func:`download_github_code_group` pass that still has rows to fetch."""

    name: str
    source: SourceConfig
    out: Path
    manifest: Manifest
    wanted: int
    counters: _FetchCounters
    increment: _Increment


def download_github_code_group(
    cfg: DatasetConfig,
    names: list[str],
    layout: DatasetLayout,
    *,
    rows_needed: dict[str, int],
    shard_size: int = DEFAULT_SHARD_SIZE,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
) -> dict[str, Manifest]:
    """:func:`download` for several `github_code` sources of one repo in a **single pass** over its files: every
    row group is fetched once and its rows are dispatched to the language source that wants them (a source that
    has its ``rows_needed[name]`` stops taking rows, the others read on). The raw shards, ``rows_fetched`` and
    ``extra["exhausted"]`` of every source are exactly what separate ``download`` calls would produce; shards are
    published and recorded per member as they fill (see :func:`download`). Returns the raw manifest of every
    source in ``names``.
    """
    results: dict[str, Manifest] = {}
    members: list[_GroupMember] = []
    for name in names:
        source = fetch_source(cfg, cfg.sources[name])
        if source.loader != "github_code" or source.check_limit is not None:
            raise ValueError(f"{name}: download_github_code_group needs github_code sources without check_limit")
        if members and github_code_repo_key(source) != github_code_repo_key(members[0].source):
            raise ValueError(f"{name}: github_code group members must share hf_id, revision and data_files")
        source_hash = cfg.raw_hash(name)
        out = layout.source_dir(name, "raw")
        manifest = ensure_raw_tokens(cfg, name, layout) or _fresh_raw_manifest(cfg, name, source_hash, out)
        results[name] = manifest
        if manifest.extra.get("exhausted"):
            log.info("%s: source exhausted after %d rows, nothing more to fetch", name, manifest.rows_fetched)
            continue
        wanted = rows_needed[name] - manifest.rows()
        if wanted > 0:
            counters = _FetchCounters()
            members.append(_GroupMember(name, source, out, manifest, wanted, counters, _Increment(manifest, out, counters, should_stop)))
    if not members:
        return results

    _fetch_group(members, layout, TokenCounter(cfg, layout, cap=cfg.max_seq_length), shard_size, hf_token)  # all pretrain
    for member in members:
        member.increment.finish(member.source)
        counters = member.counters
        log.info("%s: kept %d of %d fetched rows (%d rows on disk)", member.name, counters.kept, counters.consumed, member.manifest.rows())
    return results


def _fetch_group(
    members: list[_GroupMember], layout: DatasetLayout, counter: TokenCounter, shard_size: int, hf_token: str | None
) -> None:
    """Run the group pass and append every member's rows (token-counted) to its raw directory, one shard writer
    per member."""
    for member in members:
        log.info("%s: fetching %d rows from offset %d -> %s", member.name, member.wanted, member.manifest.rows_fetched, member.out)
    requests = [GithubCodeRequest(m.name, m.source, m.manifest.rows_fetched, m.wanted) for m in members]
    columns = _union_columns([loader_columns(m.source) for m in members])
    fetch_stats = FetchStats()
    by_name = {m.name: m for m in members}
    repo = members[0].source.hf_id
    total = sum(m.wanted for m in members)

    with ExitStack() as stack:
        bar = stack.enter_context(progress(total=total, desc=f"{repo}: download ({len(members)} languages)", unit="row"))
        postfix = _DownloadPostfix(bar, fetch_stats)
        sinks = {
            m.name: _TokenizingSink(
                stack.enter_context(
                    ShardWriter(m.out, shard_size, start_shard=len(m.manifest.shards), on_shard=m.increment.record_shard)
                ),
                m.increment,
                counter,
                raw_text_of(m.source),
            )
            for m in members
        }
        rows = read_github_code_group(
            requests, token=hf_token, index_dir=layout.hub_index_dir(), columns=columns, on_file=postfix.on_file,
            stats=fetch_stats, align_to_row_group=True,
        )
        for consumed_total, (name, raw) in enumerate(rows, start=1):
            member = by_name[name]
            member.counters.consumed += 1
            row = text_row(member.source, raw, name)
            row[CONSUMED_KEY] = member.counters.consumed
            sinks[name].add(row)
            member.counters.kept += 1
            postfix.consumed(consumed_total)
            bar.update(1)
        for sink in sinks.values():
            sink.flush()
    for member in members:
        if member.counters.kept < member.wanted:
            member.counters.exhausted = True


def _union_columns(projections: list[list[str] | None]) -> list[str] | None:
    """The column projection covering every member's (None as soon as one member needs every column)."""
    union: list[str] = []
    for columns in projections:
        if columns is None:
            return None
        union.extend(c for c in columns if c not in union)
    return union


TOKEN_BATCH = 256  # rows tokenized per `count_many` call while downloading


def raw_text_of(source: SourceConfig) -> Callable[[Row], str]:
    """What the ``tokens`` column of a raw row counts: ``text_field`` for pretrain / validation sources, instruction +
    input + output for instruct rows."""
    if source.kind == "instruct":
        return instruct_text
    text_field = source.text_field
    return lambda row: str(row[text_field])


def _with_tokens(rows: Iterator[Row], counter: TokenCounter, text_of: Callable[[Row], str]) -> Iterator[Row]:
    """Add ``tokens`` to every row, counting ``TOKEN_BATCH`` rows per tokenizer call."""
    batch: list[Row] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= TOKEN_BATCH:
            yield from _tokenized(batch, counter, text_of)
            batch = []
    if batch:
        yield from _tokenized(batch, counter, text_of)


def _tokenized(batch: list[Row], counter: TokenCounter, text_of: Callable[[Row], str]) -> list[Row]:
    for row, tokens in zip(batch, counter.count_many([text_of(row) for row in batch])):
        row["tokens"] = tokens
    return batch


class _TokenizingSink:
    """``add(row)`` for a :class:`ShardWriter`: rows are token-counted in batches of ``TOKEN_BATCH`` before they
    reach the writer; ``flush()`` counts and hands over the rest."""

    def __init__(self, writer: ShardWriter, increment: _Increment, counter: TokenCounter, text_of: Callable[[Row], str]) -> None:
        self._writer = writer
        self._increment = increment
        self._counter = counter
        self._text_of = text_of
        self._batch: list[Row] = []

    def add(self, row: Row) -> None:
        self._batch.append(row)
        if len(self._batch) >= TOKEN_BATCH:
            self.flush()

    def flush(self) -> None:
        for row in _tokenized(self._batch, self._counter, self._text_of):
            self._increment.add(self._writer, row)
        self._batch = []


def truncate_raw_to_good_prefix(directory: Path, manifest: Manifest) -> bool:
    """Repair a raw directory with a missing / unreadable / mismatching shard by dropping that shard and everything
    after it: the manifest keeps the good prefix, ``rows_fetched`` becomes the offset after its last shard (so the
    next download resumes there) and the exhaustion flag is cleared. Returns False — nothing changed — when no
    prefix can be kept (the first shard is bad, or a kept shard has no recorded offset).
    """
    good = 0
    for shard in manifest.shards:
        if shard_problem(directory, shard) is not None:
            break
        good += 1
    if good == len(manifest.shards):
        return True  # nothing wrong
    if good == 0 or manifest.shards[good - 1].offset is None:
        return False
    dropped = manifest.shards[good:]
    log.warning("%s: dropping %d shard(s) from %s (%s and after)", manifest.source, len(dropped), directory, dropped[0].name)
    manifest.shards = manifest.shards[:good]
    manifest.rows_fetched = int(manifest.shards[-1].offset or 0)
    manifest.extra.pop("exhausted", None)
    manifest.extra.pop("check_limit", None)
    for path in list_parquet_files(directory):
        index = shard_index(path)
        if index is not None and index >= good:
            path.unlink()
    manifest.save(directory)
    return True


def raw_has_tokens(cfg: DatasetConfig, manifest: Manifest) -> bool:
    """Whether a raw manifest's ``tokens`` column was counted the way ``cfg`` counts (mode, tokenizer and cap)."""
    tokenizer = cfg.tokenizer.name if cfg.token_count == "tokenizer" else None
    return (
        manifest.token_count == cfg.token_count
        and manifest.tokenizer == tokenizer
        and manifest.token_cap == cfg.token_cap(manifest.source)
        and manifest.tokens() is not None
    )


def ensure_raw_tokens(cfg: DatasetConfig, name: str, layout: DatasetLayout) -> Manifest | None:
    """The current raw manifest of ``name`` with a ``tokens`` column in every shard, or None if there is no current
    raw manifest. A raw directory from before the column (or counted differently) is upgraded **in place**: every
    shard is rewritten with the same rows and name plus ``tokens``; ``rows_fetched`` and the shard numbering do not
    change and nothing is downloaded."""
    source = cfg.sources[name]
    out = layout.source_dir(name, "raw")
    manifest = current_manifest(out, cfg.raw_hash(name), "raw")
    if manifest is None or raw_has_tokens(cfg, manifest):
        return manifest

    log.info("%s: adding the tokens column to %d raw shard(s) in place -> %s", name, len(manifest.shards), out)
    counter = TokenCounter.for_source(cfg, layout, name)
    text_of = raw_text_of(source)
    for shard in progress(list(manifest.shards), desc=f"{name}: count_tokens", unit="shard", leave=False):
        path = out / shard.name
        table = pq.read_table(path)
        tokens = counter.count_many([text_of(row) for row in table.to_pylist()])  # a shard is one batch
        if "tokens" in table.column_names:
            table = table.drop_columns(["tokens"])
        table = table.append_column("tokens", pa.array(tokens, type=pa.int64()))  # original schema kept
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression=SHARD_COMPRESSION)
        tmp.replace(path)
        shard.tokens = sum(tokens)
    manifest.token_count = cfg.token_count
    manifest.tokenizer = cfg.tokenizer.name if cfg.token_count == "tokenizer" else None
    manifest.token_cap = cfg.token_cap(name)
    manifest.save(out)
    return manifest


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
    source_hash = cfg.validation_hash(name)
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
    tokens = TokenCounter.for_source(cfg, layout, name).count_many(texts)
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
