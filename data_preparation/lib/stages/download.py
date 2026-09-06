# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The download step (sources/<source>/raw/) and the tokenizer step, plus the token counter and manifest helpers
that stages/build.py shares.

Every step is a function (config, name, layout, *options) -> Manifest that is idempotent via the manifest (a
second call with nothing new returns the stored manifest without touching the shards) and incremental where the
data allow it. Raw folders are append-only and precious (bandwidth): :func:`download` appends to a *current* raw
manifest, starts a fresh folder when there is none, and never deletes one. A folder whose manifest is *stale*
(DatasetConfig.raw_hash: loader identity, token_count or tokenizer changed) or *outdated* (stored with a
smaller max_seq_length than the config asks for, :meth:`Manifest.is_outdated`) raises :class:`RawFolderError`;
the repair step (lib/build/repair.py) deletes such folders after the user confirmed, nothing else does.

What a raw row is: pretrain rows carry text_field truncated at the token boundary max_seq_length (the
stored tokens is the true count of the stored text, see truncation.py); instruct rows carry instruction /
input / output with tokens = the count of their concatenation, uncapped. An instruct row longer than
max_seq_length is not stored at all (dropped_too_long; cutting an answer would be worse than losing the
row). The raw manifest records truncated_at_tokens (the cap used, both kinds), token_count and the
tokenizer name.

A download pass (:func:`_fetch`) is a two-stage pipeline: the job's own thread pulls rows from the loader, converts
them and buffers :data:`TOKEN_BATCH` rows per source, and a token worker thread (:class:`_TokenWorker`) tokenizes
the batches and writes the shards, in order. Fetching the next row group (network, parquet decode) so overlaps
tokenizing the previous batches, which took as long as the fetch itself in one thread. The tokenizer's own thread
pool is a separate matter (`TOKENIZERS_PARALLELISM`, see :func:`_auto_tokenizer`).
"""

from __future__ import annotations

import functools
import os
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

from data_preparation.dataset_config import DatasetConfig, SourceConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import StopCheck
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.converters import Filter, get_converter, get_filter, text_or_empty
from data_preparation.lib.sources.hub_files import FetchStats
from data_preparation.lib.sources.loaders import (
    MAX_CACHED_FILE_KEY,
    GithubCodeRequest,
    Row,
    SharedLoaderParameters,
    get_loader,
    github_code_repo_key,
    read_github_code_group,
)
from data_preparation.lib.sources.synthetic import write_synthetic_tokenizer
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.stages.truncation import estimate_tokens, truncate_many
from data_preparation.lib.storage.manifest import Manifest, has_shards, library_versions
from data_preparation.lib.storage.parquet import ShardWriter
from data_preparation.lib.storage.raw_folder import RawFolder, RowProgress
from data_preparation.lib.ui.dashboard import progress

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000

# --- token counting ----------------------------------------------------------------------------------------------------


class TokenCounter:
    """
    Token counts with the config's tokenizer (token_count: tokenizer, add_special_tokens=False) or
    len(text) // 4 (estimate). Counts are never capped here: the download truncates pretrain *text* at the
    cap (:meth:`truncate_many`) and drops long instruct rows, so every stored count is a true count.
    """

    def __init__(self, config: DatasetConfig, layout: DatasetLayout) -> None:
        self.mode = config.token_count
        self.tokenizer_name = config.tokenizer.name
        self._tokenizer: Any = None
        if self.mode == "tokenizer":
            self._tokenizer = _load_tokenizer(layout.tokenizer_dir(config.tokenizer.name), config.tokenizer.name)

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return estimate_tokens(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_many(self, texts: list[str]) -> list[int]:
        if not texts:
            return []  # HF fast tokenizers choke on an empty batch
        if self._tokenizer is None:
            return [estimate_tokens(text) for text in texts]
        encoded = self._tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        """
        (prefix, count) per text with count <= max_tokens: truncation.truncate_many with this
        counter's token definition (the cut text re-counts to exactly count).
        """

        return truncate_many(texts, max_tokens, self._tokenizer)


def _load_tokenizer(tokenizer_dir: Path, name: str) -> Any:
    """
    The saved HF tokenizer in tokenizer_dir; fails if the tokenizer stage has not run yet.
    """

    has_tokenizer_files = (tokenizer_dir / "tokenizer.json").is_file() or (tokenizer_dir / "tokenizer_config.json").is_file()
    if not has_tokenizer_files:
        raise FileNotFoundError(f"tokenizer {name!r} not found at {tokenizer_dir}; run the tokenizer stage first")
    return _auto_tokenizer().from_pretrained(str(tokenizer_dir))


_IMPORT_LOCK = threading.Lock()


def _auto_tokenizer() -> Any:
    """
    transformers.AutoTokenizer, imported lazily (the HF cache env must be configurable before the import)
    and under a lock: transformers initialises its lazy modules on first import, which is not thread-safe and
    the build runs items in threads.
    """

    with _IMPORT_LOCK:
        # the tokenizer's Rust thread pool + a later fork (torch DataLoader workers; the decontamination /
        # minhash pools are spawn and immune) is the well-known tokenizers deadlock; the library's own
        # mitigation, set before the first load (spawn children inherit it through the environment). The
        # prepare CLI, which never forks after this point, sets "true" (and the pool size) before it gets here
        # (prepare.py): a batch then tokenizes on several cores instead of one, the biggest lever on the download rate.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer

    return AutoTokenizer


# --- manifest helpers --------------------------------------------------------------------------------------------------


def current_manifest(directory: Path, source_hash: str, stage: str) -> Manifest | None:
    """
    The stored manifest if it matches source_hash and stage; None (with a warning) if stale or absent.
    For derived folders (processed/, tokenizers) that the caller rebuilds; raw folders go through
    :func:`inspect_raw`, which never treats a stale folder as absent.
    """

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


def new_manifest(
    config: DatasetConfig, source: str, source_hash: str, stage: str, *, tokens: bool = False, truncated_at_tokens: int | None = None
) -> Manifest:
    """
    An empty manifest for stage; with tokens it records how token counts are measured, raw manifests
    record truncated_at_tokens (the max_seq_length their rows were cut / dropped at).
    """

    return Manifest(
        source=source,
        source_hash=source_hash,
        stage=stage,
        token_count=config.token_count if tokens else None,
        tokenizer=config.tokenizer.name if tokens and config.token_count == "tokenizer" else None,
        truncated_at_tokens=truncated_at_tokens,
        versions=library_versions(),
    )


def text_row(source: SourceConfig, row: Row, name: str) -> Row:
    """
    Apply a pretrain source's converter (if any) and check that text_field is present.
    """

    converter = get_converter(source)
    if converter is not None:
        row = converter(row)
    if source.text_field not in row:
        raise ValueError(f"{name}: row has no {source.text_field!r} column; columns: {sorted(row)}")
    return row


# --- raw manifest state ------------------------------------------------------------------------------------------------

RawManifestState = Literal["missing", "current", "stale", "outdated"]


class RawInspection(NamedTuple):
    """
    The state of sources/<name>/raw against the config, its manifest (None when missing) and the reason line
    the download's refusal, the repair step's plan and the status table all phrase from.
    """

    state: RawManifestState
    manifest: Manifest | None
    reason: str  # "missing" | "current" | "stale: source identity or tokenizer changed" | "outdated: max_seq_length 2048 -> 4096"

    @property
    def current_manifest(self) -> Manifest | None:
        """
        The manifest when the folder is current (the download appends to it, the build reads it), else None.
        """

        return self.manifest if self.state == "current" else None


def inspect_raw(config: DatasetConfig, name: str, layout: DatasetLayout) -> RawInspection:
    """
    missing (no manifest; the folder may still hold shards, which :func:`download` refuses to start over),
    stale (the manifest's hash differs from config.raw_hash(name): loader identity, token_count or
    tokenizer changed, or it is another stage's manifest), outdated (config.max_seq_length was raised above
    the cap the rows were truncated / dropped at) or current.
    """

    manifest = Manifest.load(layout.raw_dir(name))
    if manifest is None:
        return RawInspection("missing", None, "missing")
    if manifest.stage != "raw" or not manifest.is_current(config.raw_hash(name)):
        return RawInspection("stale", manifest, "stale: source identity or tokenizer changed")
    if manifest.is_outdated(config.max_seq_length):
        return RawInspection("outdated", manifest, f"outdated: max_seq_length {manifest.truncated_at_tokens} -> {config.max_seq_length}")
    return RawInspection("current", manifest, "current")


class RawFolderError(RuntimeError):
    """
    A raw folder that :func:`download` may not append to (stale or outdated; problem is the
    :func:`inspect_raw` reason). The download never deletes raw data; the repair step does, after the user
    confirmed (lib/build/repair.py).
    """

    def __init__(self, name: str, directory: Path, problem: str) -> None:
        super().__init__(
            f"{name}: raw folder {directory} is {problem}; it must be deleted and downloaded again; "
            "the download never deletes raw data, run the repair step (it asks for confirmation)"
        )
        self.name = name
        self.directory = directory


# --- tokenizer ---------------------------------------------------------------------------------------------------------


def prepare_tokenizer(config: DatasetConfig, layout: DatasetLayout) -> Manifest:
    """
    Save the config's tokenizer to layout.tokenizer_dir(name) (Hub download or the synthetic WordLevel one).
    """

    tokenizer = config.tokenizer
    tokenizer_dir = layout.tokenizer_dir(tokenizer.name)
    source_hash = config.tokenizer_hash()
    manifest = current_manifest(tokenizer_dir, source_hash, "tokenizer")
    if manifest is not None and (tokenizer_dir / "tokenizer_config.json").is_file():
        return manifest

    log.info("preparing tokenizer %s (%s) -> %s", tokenizer.name, tokenizer.kind, tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer.kind == "synthetic":
        write_synthetic_tokenizer(tokenizer_dir)
    else:
        _auto_tokenizer().from_pretrained(tokenizer.hf_id, revision=tokenizer.revision).save_pretrained(str(tokenizer_dir))
    manifest = new_manifest(config, tokenizer.name, source_hash, "tokenizer")
    manifest.extra = {"kind": tokenizer.kind, "hf_id": tokenizer.hf_id, "revision": tokenizer.revision}
    manifest.save(tokenizer_dir)
    return manifest


# --- download ----------------------------------------------------------------------------------------------------------


def fetch_source(config: DatasetConfig, source: SourceConfig) -> SourceConfig:
    """
    The source as handed to its loader: with config.always_range_requests every Hub file is read remotely by
    piece (max_cached_file_mb forced to 0), otherwise the source's own threshold applies.
    """

    if config.always_range_requests and source.loader in ("hf_files", "github_code"):
        return replace(source, load_kwargs={**source.load_kwargs, MAX_CACHED_FILE_KEY: 0})
    return source


@dataclass
class _IncrementCounters:
    """
    What one download increment did so far (updated while :func:`_fetch` runs, read back at the end).

    Two threads write here, each its own fields: the fetch thread `consumed` and `skipped_malformed`, the token
    worker `kept` and `dropped_too_long` (they follow the tokenizer); either may read the other's. `exhausted` is
    set after the worker joined.
    """

    consumed: int = 0  # source rows the loader yielded (the loader offset advances by this much)
    kept: int = 0  # rows written to disk
    skipped_malformed: int = 0  # instruct rows whose converter raised ValueError
    dropped_too_long: int = 0  # instruct rows with more than `max_seq_length` tokens
    exhausted: bool = False  # the loader ran dry, or check_limit was reached


def download(
    config: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    *,
    rows_needed: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
) -> Manifest:
    """
    Append raw shards until rows_needed rows are on disk (no-op if they already are).

    The folder's manifest must be current (:func:`inspect_raw`): a stale or outdated one raises
    :class:`RawFolderError` (nothing is deleted here), a missing one starts the folder from shard 0 (refused when
    shards without a manifest are present). manifest.rows_fetched is the loader offset reached (source rows
    consumed). Pretrain sources keep every row (converter applied, text_field guaranteed, the text truncated to
    max_seq_length tokens with its true count in tokens). Instruct sources run the converter and filter at
    download time and store only standardized {instruction, input, output} rows of at most max_seq_length
    tokens; malformed rows (converter raises ValueError) are counted in skipped_malformed, longer rows in
    dropped_too_long. check_limit bounds the source rows inspected in total. A loader that yields fewer rows
    than requested sets exhausted (the training sampler cycles a source smaller than its budget).

    rows_needed is a minimum: a loader reading a large parquet file remotely finishes the row group it is in
    (see sources/loaders.py), every row it yields is written and rows_fetched advances to that row-group
    boundary, so the same bytes are never downloaded twice. Sources without a converter are read with only
    text_field projected (columns); converters and fields mappings get every column.

    Every shard is published and recorded in the manifest (with the loader offset after its last row and the
    skipped / dropped totals up to that row, :class:`RawFolder`) as soon as it is written, so a failure or a stop
    request (checked after every shard) keeps everything fetched so far and the next call resumes from the last
    complete shard without counting anything twice.
    """

    folder, increment = _plan_increment(config, name, layout, rows_needed, token_counter=lambda: TokenCounter(config, layout), should_stop=should_stop)
    if increment is None:
        return folder.manifest
    log.info("%s: fetching %d rows from offset %d -> %s", name, increment.rows_to_keep, folder.rows_fetched, folder.directory)
    loader = get_loader(increment.source.loader)
    fetch_stats = FetchStats()
    # the bar's total is the minimum; it overshoots (e.g. 1000/11) when the loader finishes a remote row group
    with progress(total=increment.rows_to_keep, desc=name, unit="row", panel="downloads", bytes_fetched=lambda: fetch_stats.bytes_fetched) as bar:
        postfix = _DownloadPostfix(bar)
        shared_parameters = SharedLoaderParameters(
            token=hf_token, index_dir=layout.hub_index_dir(), on_file=postfix.on_file, stats=fetch_stats,
            columns=loader_columns(increment.source),
        )
        rows = loader(increment.source, folder.rows_fetched, increment.loader_count, shared_parameters)
        _fetch([increment], _tagged(name, rows), bar, postfix, shard_size)
    _finish_increment(folder, increment.source, increment.counters)
    _log_increment(name, increment.counters, folder.manifest)
    return folder.manifest


def _finish_increment(folder: RawFolder, source: SourceConfig, counters: _IncrementCounters) -> None:
    """
    Hand the increment's totals to the folder; a source stopped by its own check_limit records that limit
    (so a later, larger one reopens it) instead of counting as a loader that ran dry.
    """

    limit = source.check_limit
    by_limit = limit if limit is not None and folder.start_offset + counters.consumed >= limit else None
    progress_now = RowProgress(counters.consumed, counters.skipped_malformed, counters.dropped_too_long)
    folder.finish(progress_now, exhausted=counters.exhausted, check_limit=by_limit)


def _log_increment(name: str, counters: _IncrementCounters, manifest: Manifest) -> None:
    log.info(
        "%s: kept %d of %d fetched rows (%d rows on disk; %d malformed skipped, %d too long dropped)",
        name, counters.kept, counters.consumed, manifest.rows(), counters.skipped_malformed, counters.dropped_too_long,
    )


def _raw_folder_to_append_to(config: DatasetConfig, name: str, layout: DatasetLayout, *, should_stop: StopCheck | None = None) -> RawFolder:
    """
    The raw folder of name around its current manifest, or a fresh one (truncated_at_tokens =
    max_seq_length) when the directory has none. Refused when the folder is stale or outdated
    (:class:`RawFolderError`; deleting it is the repair step's decision) or holds shards without any manifest:
    nothing would say where those rows came from, and starting over would delete them.
    """

    raw_dir = layout.raw_dir(name)
    inspection = inspect_raw(config, name, layout)
    if inspection.manifest is not None and inspection.state != "current":
        raise RawFolderError(name, raw_dir, inspection.reason)
    manifest = inspection.manifest
    if manifest is None:
        if has_shards(raw_dir):
            raise RuntimeError(f"{name}: {raw_dir} holds shards but no manifest; delete the directory to download the source again")
        manifest = new_manifest(config, name, config.raw_hash(name), "raw", tokens=True, truncated_at_tokens=config.max_seq_length)
    return RawFolder(raw_dir, manifest, config_cap=config.max_seq_length, should_stop=should_stop)


UNBOUNDED_COUNT = 2**62  # "as many rows as there are": instruct downloads stop consuming once `rows_to_keep` rows are kept
StoredRow = tuple[Row, RowProgress]  # a row ready to store, with where the fetch stood right after it

TOKEN_BATCH = 256  # rows tokenized per tokenizer call while downloading
TOKEN_QUEUE_DEPTH = 3  # batches the fetch thread may run ahead of the token worker (bounds the raw text alive per job)


class _TokenStep:
    """
    The token step of a download, in two halves used from two threads. The fetch thread feeds it row by row:
    add(row, progress) returns a full batch of :data:`TOKEN_BATCH` rows (else []), take() whatever is
    buffered; every row comes with the :class:`RowProgress` right after it. The token worker calls
    tokenize(batch) on those batches, in order, and gets the rows ready to store.

    Pretrain rows: text_field is truncated at token max_tokens (truncation.py) and tokens is the
    true count of the stored text. Instruct rows: tokens counts instruction + input + output uncapped; a row over
    max_tokens is dropped (counters.dropped_too_long), never cut, and every stored row's progress carries the
    drop count of the rows before it (exact per row, so a resume never double counts).
    """

    def __init__(self, source: SourceConfig, counter: TokenCounter, max_tokens: int, counters: _IncrementCounters) -> None:
        self._counter = counter
        self._max_tokens = max_tokens
        self._counters = counters
        self._is_instruct = source.kind == "instruct"
        self._text_field = source.text_field
        self._batch: list[StoredRow] = []

    @property
    def pending(self) -> int:
        """
        Rows buffered for the next tokenizer call.
        """

        return len(self._batch)

    def add(self, row: Row, progress: RowProgress) -> list[StoredRow]:
        """
        Buffer row; a full batch (:data:`TOKEN_BATCH` rows) is released, untokenized.
        """

        self._batch.append((row, progress))
        return self.take() if len(self._batch) >= TOKEN_BATCH else []

    def take(self) -> list[StoredRow]:
        """
        The buffered rows (possibly none), untokenized; the buffer is empty afterwards.
        """

        batch, self._batch = self._batch, []
        return batch

    def tokenize(self, batch: list[StoredRow]) -> list[StoredRow]:
        """
        The rows of batch ready to store: pretrain rows truncated and counted, instruct rows counted or dropped.
        """

        if not batch:
            return []
        return self._drop_long_instruct_rows(batch) if self._is_instruct else self._truncate_pretrain_rows(batch)

    def _truncate_pretrain_rows(self, batch: list[StoredRow]) -> list[StoredRow]:
        texts = [text_or_empty(row.get(self._text_field)) for row, _ in batch]
        for (row, _), text, (cut, tokens) in zip(batch, texts, self._counter.truncate_many(texts, self._max_tokens), strict=True):
            if cut != text:
                row[self._text_field] = cut
            row["tokens"] = tokens
        return batch

    def _drop_long_instruct_rows(self, batch: list[StoredRow]) -> list[StoredRow]:
        stored: list[StoredRow] = []
        for (row, before), tokens in zip(batch, self._counter.count_many([instruct_text(row) for row, _ in batch]), strict=True):
            if tokens > self._max_tokens:
                self._counters.dropped_too_long += 1
                continue
            row["tokens"] = tokens
            stored.append((row, RowProgress(before.consumed, before.skipped_malformed, self._counters.dropped_too_long)))
        return stored


@dataclass
class _Increment:
    """
    One source's part of a download pass: what it still wants, how a source row becomes a stored row, and what
    the pass did for it so far (:attr:`counters`).
    """

    name: str
    source: SourceConfig
    folder: RawFolder
    rows_to_keep: int  # rows to keep in this pass
    max_consume: int | None  # source rows this pass may consume (`check_limit` less the offset reached); None = no bound
    token_step: _TokenStep
    counters: _IncrementCounters
    converter: Callable[[Row], Row] | None  # instruct sources: the standardizing converter (None: rows are standard already)
    row_filter: Filter | None
    submitted: int = 0  # rows handed to the token worker (fetch thread)
    settled: int = 0  # rows the token worker stored or dropped (worker thread)

    @property
    def is_instruct(self) -> bool:
        return self.source.kind == "instruct"

    @property
    def in_flight(self) -> int:
        """
        Rows handed to the token worker whose fate (stored or dropped) is not settled yet.
        """

        return self.submitted - self.settled

    @property
    def loader_count(self) -> int:
        """
        What the loader is asked for. Pretrain rows are all kept: rows_to_keep (or the consume budget when that is
        smaller); a remote loader may still finish its row group beyond it. Instruct rows may be dropped by the
        filter, the converter or the token step: everything within the budget, or without bound.
        """

        if self.is_instruct:
            return UNBOUNDED_COUNT if self.max_consume is None else self.max_consume
        return self.rows_to_keep if self.max_consume is None else min(self.rows_to_keep, self.max_consume)

    @property
    def done(self) -> bool:
        """
        No more source rows are taken: the consume budget is spent (check_limit bounds the source rows
        consumed, whatever the loader yields beyond its count), or an instruct source kept its rows_to_keep rows.
        """

        if self.max_consume is not None and self.counters.consumed >= self.max_consume:
            return True
        return self.is_instruct and self.counters.kept >= self.rows_to_keep

    def convert(self, name: str, raw: Row) -> Row | None:
        """
        The row to store for source row raw, or None when the filter rejects it or the converter finds it
        malformed (ValueError, counted in skipped_malformed).
        """

        if not self.is_instruct:
            return text_row(self.source, raw, name)
        if self.row_filter is not None and not self.row_filter(raw):
            return None
        try:
            return _instruct_row(raw, self.converter)
        except ValueError as err:
            self.counters.skipped_malformed += 1
            log.debug("%s: skipping malformed row: %s", name, err)
            return None


def _plan_increment(
    config: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    rows_needed: int,
    *,
    token_counter: Callable[[], TokenCounter],
    should_stop: StopCheck | None,
) -> tuple[RawFolder, _Increment | None]:
    """
    Open the raw folder of name and decide what this pass fetches for it: None when there is nothing to do
    (the source is exhausted, the rows are on disk, or its check_limit is spent, recorded as the exhaustion).
    """

    source = fetch_source(config, config.sources[name])
    folder = _raw_folder_to_append_to(config, name, layout, should_stop=should_stop)
    folder.reopen_if_check_limit_grew(source.check_limit)
    if folder.exhausted:
        log.info("%s: source exhausted after %d rows, nothing more to fetch", name, folder.rows_fetched)
        return folder, None
    wanted = rows_needed - folder.rows
    if wanted <= 0:
        return folder, None
    max_consume = None if source.check_limit is None else source.check_limit - folder.rows_fetched
    if max_consume is not None and max_consume <= 0:
        folder.mark_exhausted(check_limit=source.check_limit)
        return folder, None
    converter = get_converter(source) if source.kind == "instruct" else None
    if source.kind == "instruct" and converter is None and source.loader != "synthetic":
        raise ValueError(f"{name}: instruct source needs `fields` or `converter`")
    row_filter = get_filter(source.filter) if source.filter is not None else None
    counters = _IncrementCounters()
    token_step = _TokenStep(source, token_counter(), folder.cap, counters)
    return folder, _Increment(name, source, folder, wanted, max_consume, token_step, counters, converter, row_filter)


def _tagged(name: str, rows: Iterable[Row]) -> Iterator[tuple[str, Row]]:
    """
    rows as (name, row) pairs; closing this generator closes the loader's.
    """

    try:
        for raw in rows:
            yield name, raw
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()


class _TokenWorker:
    """
    The tokenizing half of a download pass on its own thread: batches submitted by the fetch thread are tokenized
    (:meth:`_TokenStep.tokenize`) and stored (:func:`_store`) in submission order, so row order, the per-row
    progress and the shard boundaries are exactly those of the same pass done in one thread. The queue holds
    :data:`TOKEN_QUEUE_DEPTH` batches: submit blocks the fetch thread when the worker is that far behind.

    A failure on the worker (a tokenizer error, a write error, :class:`BuildAborted` from the stop check after a
    published shard) is kept and re-raised on the fetch thread by the next :meth:`submit`, :meth:`drain` or
    :meth:`close` (:attr:`failed` tells earlier); from then on the worker only settles what is queued without
    storing it. close is what leaving the with block does: it joins the thread whatever happened, so the shard
    writers are closed after the worker is done with them.
    """

    def __init__(self, name: str, writers: dict[str, ShardWriter], bar: Progress) -> None:
        self._queue: queue.Queue[tuple[_Increment, list[StoredRow]] | None] = queue.Queue(maxsize=TOKEN_QUEUE_DEPTH)
        self._writers = writers
        self._bar = bar
        self._failure: BaseException | None = None
        self._raised = False
        self._thread = threading.Thread(target=self._run, name=f"tokenize:{name}")
        self._thread.start()

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def submit(self, increment: _Increment, batch: list[StoredRow]) -> None:
        """
        Queue batch (nothing for an empty one) for increment; raises the worker's failure instead if it has one.
        """

        self._raise_failure()
        if not batch:
            return
        increment.submitted += len(batch)
        self._queue.put((increment, batch))

    def drain(self) -> None:
        """
        Wait until every submitted batch is stored (or dropped), then raise the worker's failure if it has one.
        """

        self._queue.join()
        self._raise_failure()

    def close(self) -> None:
        """
        End the worker after the queued batches (or after settling them, once failed) and join it; raises the
        worker's failure if it was not raised before.
        """

        self._queue.put(None)
        self._thread.join()
        self._raise_failure()

    def __enter__(self) -> _TokenWorker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _raise_failure(self) -> None:
        if self._failure is not None and not self._raised:
            self._raised = True
            raise self._failure

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                increment, batch = item
                if self._failure is None:
                    try:
                        _store(increment, self._writers[increment.name], increment.token_step.tokenize(batch), self._bar)
                    except BaseException as error:  # noqa: BLE001  # whatever it is, the fetch thread re-raises it
                        self._failure = error
                increment.settled += len(batch)  # after the store: `kept` is up to date before the rows leave `in_flight`
            finally:
                self._queue.task_done()


def _fetch(increments: list[_Increment], rows: Iterator[tuple[str, Row]], bar: Progress, postfix: _DownloadPostfix, shard_size: int) -> None:
    """
    One download pass: every (name, row) of rows goes to its increment (counted as consumed, converted,
    batched for the token worker, which tokenizes the batches and appends them shard by shard to the raw
    directory, one shard writer per increment) until every increment is :attr:`~_Increment.done` or the stream
    ends (the stream is closed either way); the last batches are submitted, the worker joined, and an increment
    that kept fewer rows than it wanted is exhausted. An instruct increment submits early and waits for the
    worker when the rows in flight and buffered would meet the target, so it stops exactly there and never
    reads on while the outcome is open (a second pass would re-stream the file prefix). bar tracks kept rows
    (postfix: source rows consumed, current repo file; the bytes fetched are the counter the bar was created
    with).
    """

    increments_by_name = {increment.name: increment for increment in increments}
    consumed_total = 0
    with ExitStack() as stack:
        writers = {
            increment.name: stack.enter_context(
                ShardWriter(increment.folder.directory, shard_size, start_shard=increment.folder.shard_count, on_shard=increment.folder.record_shard)
            )
            for increment in increments
        }
        worker = stack.enter_context(_TokenWorker(",".join(increments_by_name), writers, bar))  # closed before the writers
        try:
            for name, raw in rows:
                if worker.failed:
                    worker.drain()  # raises: stop pulling rows for a worker that stores nothing anymore
                increment = increments_by_name[name]
                if increment.done:
                    if all(increment.done for increment in increments):
                        break
                    continue  # this source is done, the others read on
                counters = increment.counters
                counters.consumed += 1
                consumed_total += 1
                postfix.consumed(consumed_total)
                row = increment.convert(name, raw)
                if row is None:
                    continue
                worker.submit(increment, increment.token_step.add(row, RowProgress(counters.consumed, counters.skipped_malformed, 0)))
                if increment.is_instruct and counters.kept + increment.in_flight + increment.token_step.pending >= increment.rows_to_keep:
                    # the rows in flight and buffered would meet the target: settle them before reading on
                    worker.submit(increment, increment.token_step.take())
                    worker.drain()
                if all(increment.done for increment in increments):
                    break  # enough: stop pulling (the finally closes the stream)
            for increment in increments:
                worker.submit(increment, increment.token_step.take())
        finally:
            close = getattr(rows, "close", None)
            if close is not None:
                close()
    for increment in increments:
        if increment.counters.kept < increment.rows_to_keep:
            increment.counters.exhausted = True  # the loader ran dry (or the budget was spent) before `rows_to_keep` rows were kept


def _store(increment: _Increment, writer: ShardWriter, stored: list[StoredRow], bar: Progress) -> None:
    """
    The rows the token step released, appended and counted as kept.
    """

    for row, row_progress in stored:
        increment.folder.add(writer, row, row_progress)
        increment.counters.kept += 1
        bar.update(1)


def download_github_code_group(
    config: DatasetConfig,
    names: list[str],
    layout: DatasetLayout,
    *,
    rows_needed: dict[str, int],
    shard_size: int = DEFAULT_SHARD_SIZE,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
) -> dict[str, Manifest]:
    """
    :func:`download` for several `github_code` sources of one repo in a single pass over its files: every
    row group is fetched once and its rows are dispatched to the language source that wants them (a source that
    has its rows_needed[name] or spent its check_limit stops taking rows, the others read on). The raw
    shards (texts truncated by the same token step), rows_fetched and exhausted of every source are exactly
    what separate download calls would produce: the same :func:`_fetch` pass over the repo reader instead of
    one loader. A stale or outdated member raises :class:`RawFolderError` before anything is fetched. Returns the
    raw manifest of every source in names.
    """

    for name in names:
        source = config.sources[name]
        if source.loader != "github_code":
            raise ValueError(f"{name}: download_github_code_group needs github_code sources")
        if github_code_repo_key(source) != github_code_repo_key(config.sources[names[0]]):
            raise ValueError(f"{name}: github_code group members must share hf_id, revision and data_files")
    token_counter = functools.cache(lambda: TokenCounter(config, layout))
    results: dict[str, Manifest] = {}
    increments: list[_Increment] = []
    for name in names:
        folder, increment = _plan_increment(config, name, layout, rows_needed[name], token_counter=token_counter, should_stop=should_stop)
        results[name] = folder.manifest
        if increment is not None:
            increments.append(increment)
    if not increments:
        return results

    for increment in increments:
        log.info(
            "%s: fetching %d rows from offset %d -> %s",
            increment.name, increment.rows_to_keep, increment.folder.rows_fetched, increment.folder.directory,
        )
    requests = [
        GithubCodeRequest(increment.name, increment.source, increment.folder.rows_fetched, increment.loader_count) for increment in increments
    ]
    columns = _union_columns([loader_columns(increment.source) for increment in increments])
    fetch_stats = FetchStats()
    repo = increments[0].source.hf_id
    with progress(
        total=sum(increment.rows_to_keep for increment in increments), desc=f"{repo} ({len(increments)} languages)", unit="row", panel="downloads",
        bytes_fetched=lambda: fetch_stats.bytes_fetched,
    ) as bar:
        postfix = _DownloadPostfix(bar)
        shared_parameters = SharedLoaderParameters(
            token=hf_token, index_dir=layout.hub_index_dir(), on_file=postfix.on_file, stats=fetch_stats, columns=columns,
        )
        rows = read_github_code_group(requests, shared_parameters)
        _fetch(increments, rows, bar, postfix, shard_size)
    for increment in increments:
        _finish_increment(increment.folder, increment.source, increment.counters)
        _log_increment(increment.name, increment.counters, increment.folder.manifest)
    return results


def _union_columns(projections: list[list[str] | None]) -> list[str] | None:
    """
    The column projection covering every member's (None as soon as one member needs every column).
    """

    union: list[str] = []
    for columns in projections:
        if columns is None:
            return None
        union.extend(column for column in columns if column not in union)
    return union


def _instruct_row(raw: Row, converter: Callable[[Row], Row] | None) -> Row:
    """
    The standardized {instruction, input, output} row for raw.

    A ValueError raised by the converter (malformed row) propagates to the caller, which skips the row.
    """

    row = converter(raw) if converter is not None else dict(raw)
    return {"instruction": row["instruction"], "input": row.get("input", ""), "output": row["output"]}


class _DownloadPostfix:
    """
    The download bar's postfix: source rows consumed, current repo file (refreshed sparsely).
    """

    def __init__(self, bar: Progress) -> None:
        self._bar = bar
        self._values: dict[str, Any] = {"consumed": 0}

    def on_file(self, file: str) -> None:
        """
        Loader callback: a new repo file is being read.
        """

        self._values["file"] = file.rsplit("/", 1)[-1]
        self._refresh()

    def consumed(self, total: int) -> None:
        """
        Record the running count of consumed source rows; the bar is refreshed every 100 rows.
        """

        self._values["consumed"] = total
        if total % 100 == 0:
            self._refresh()

    def _refresh(self) -> None:
        self._bar.set_postfix(self._values, refresh=False)


def loader_columns(source: SourceConfig) -> list[str] | None:
    """
    Column projection for a source's loader (applied whatever the file format): [text_field] for pretrain
    sources read as-is, None (every column) when a converter or fields mapping may need others or the rows are
    instruct rows.
    """

    if source.kind == "instruct" or get_converter(source) is not None:
        return None
    return [source.text_field]
