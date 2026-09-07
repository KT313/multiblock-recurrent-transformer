# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The download step (sources/<source>/raw/) and the tokenizer step, plus the token counter and manifest helpers
that stages/build.py shares.

Every step is a function (config, name, layout, *options) -> Manifest that is idempotent via the manifest (a
second call with nothing new returns the stored manifest without touching the shards) and incremental where the
data allow it. Raw folders are append-only and precious (bandwidth): :func:`download` appends to a *current* raw
manifest, starts a fresh folder when there is none, and never deletes one. A folder whose manifest is *stale*
(DatasetConfig.raw_hash: the loader identity changed), *outdated* (stored with a smaller dataset_max_sequence_length
than the config asks for, :meth:`Manifest.is_outdated`) or *tokenizer_changed* (its token counts were made with
another tokenizer or token_count than the config's) raises :class:`RawFolderError`; the repair step
(lib/build/repair.py) deletes stale and outdated folders after the user confirmed, and asks whether to keep a
tokenizer_changed one under the new tokenizer; nothing else touches a raw folder.

What a raw row is: pretrain rows carry text_field only (a string, whatever the loader delivered) truncated at
a token boundary so that tokens, the true count of the stored text plus the BOS and EOS the trainer adds
(truncation.NUMBER_OF_SPECIAL_TOKENS), is at most dataset_max_sequence_length; instruct rows carry instruction / input / output with
tokens = the count of the text the trainer formats from them (row_pipeline.instruct_text) plus the same two
specials, uncapped. An instruct row whose tokens exceeds dataset_max_sequence_length is not stored at all (dropped_too_long;
cutting an answer would be worse than losing the row). So a stored tokens is the length the trainer sees and
never exceeds dataset_max_sequence_length. The raw manifest records truncated_at_tokens (the cap used, both kinds),
token_count and the tokenizer name.

A download pass (:func:`_fetch`) is a two-stage pipeline: the job's own thread pulls rows from the loader, converts
them and buffers :data:`TOKEN_BATCH` rows per source, and a token worker thread (:class:`_TokenWorker`) tokenizes
the batches and writes the shards, in order. Fetching the next row group (network, parquet decode) so overlaps
tokenizing the previous batches, which took as long as the fetch itself in one thread. The tokenizer's own thread
pool is a separate matter (`TOKENIZERS_PARALLELISM`, see :func:`_guard_tokenizers_parallelism`).

The github_code group pass (:func:`download_github_code_group`) keeps everything it decodes: a member that has its
rows stays in the pass as a *passive* increment and stores every further row of its language the pass reads for
the others, and a language no source names gets a raw folder of its own (:func:`_extra_increment`; named by
`loaders.github_code_extra_name`, so a later config entry of that name adopts it). Passive rows are truncated and
counted like any other pretrain row and never bound the pass (`rows_to_keep` 0, never exhausted). A stop or a
failure first stores everything the pass already consumed (the stop check is suspended, :class:`_StopGate`; the
token worker drains its queue, the fetch thread's buffers are submitted, every writer publishes its buffered rows
as a short final shard, :func:`_flush_partial_shards`), so every folder's offset is the frontier the pass reached:
the file index only records row groups the pass consumed whole, and a passive folder resumes aligned from there.
"""

from __future__ import annotations

import functools
import os
import queue
import shutil
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

from data_preparation.dataset_config import DatasetConfig, SourceConfig, describe_hash_change
from data_preparation.layout import DatasetLayout
from data_preparation.lib.abort import BuildAborted, StopCheck
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.converters import Filter, get_converter, get_filter, text_or_empty
from data_preparation.lib.sources.hub_files import FetchStats, ReadRequest
from data_preparation.lib.sources.loaders import (
    MAX_CACHED_FILE_KEY,
    GithubCodeRequest,
    Row,
    SharedLoaderParameters,
    get_loader,
    github_code_extra_name,
    github_code_extra_source,
    github_code_repo_key,
    language_request,
    read_github_code_group,
)
from data_preparation.lib.sources.synthetic import write_synthetic_tokenizer
from data_preparation.lib.storage.atomic import write_atomically
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
from data_preparation.lib.stages.truncation import NUMBER_OF_SPECIAL_TOKENS, estimate_tokens, truncate_many
from data_preparation.lib.storage.manifest import Manifest, has_shards, library_versions
from data_preparation.lib.storage.parquet import ShardWriter
from data_preparation.lib.storage.raw_folder import RawFolder, RowProgress
from data_preparation.lib.ui.dashboard import progress

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000

# --- token counting ----------------------------------------------------------------------------------------------------


class TokenCounter:
    """
    Token counts with the config's tokenizer (token_count: tokenizer, no special tokens) or
    len(text) // 4 (estimate): the text's own tokens, without the BOS and EOS the trainer adds (the token step adds
    truncation.NUMBER_OF_SPECIAL_TOKENS to what it stores). Counts are never capped here: the download truncates pretrain
    *text* at the cap (:meth:`truncate_many`) and drops long instruct rows, so every stored count is a true count.
    """

    def __init__(self, config: DatasetConfig, layout: DatasetLayout) -> None:
        self.mode = config.token_count
        self.tokenizer_name = config.tokenizer.name
        self._tokenizer: SavedTokenizer | None = None
        if self.mode == "tokenizer":
            self._tokenizer = _load_tokenizer(layout.tokenizer_dir(config.tokenizer.name), config.tokenizer.name)

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return estimate_tokens(text)
        return len(self._tokenizer.encode(text))

    def count_many(self, texts: list[str]) -> list[int]:
        if self._tokenizer is None:
            return [estimate_tokens(text) for text in texts]
        return [len(encoding.ids) for encoding in self._tokenizer.encode_batch(texts)]  # `encode_batch([])` is `[]`

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        """
        (prefix, count) per text with count <= max_tokens: truncation.truncate_many with this
        counter's token definition (the cut text re-counts to exactly count).
        """

        return truncate_many(texts, max_tokens, self._tokenizer)


def _load_tokenizer(tokenizer_dir: Path, name: str) -> SavedTokenizer:
    """
    The saved tokenizer in tokenizer_dir (:class:`SavedTokenizer`: the `tokenizers` library alone, transformers
    would cost every download job seconds and hundreds of MB); fails if the tokenizer stage has not run yet.
    """

    has_tokenizer_files = (tokenizer_dir / "tokenizer.json").is_file() or (tokenizer_dir / "tokenizer_config.json").is_file()
    if not has_tokenizer_files:
        raise FileNotFoundError(f"tokenizer {name!r} not found at {tokenizer_dir}; run the tokenizer stage first")
    _guard_tokenizers_parallelism()
    return SavedTokenizer(tokenizer_dir)


_IMPORT_LOCK = threading.Lock()


def _guard_tokenizers_parallelism() -> None:
    """
    The tokenizer's Rust thread pool + a later fork (torch DataLoader workers; the decontamination / minhash pools
    are spawn and immune) is the well-known tokenizers deadlock; the library's own mitigation, set before the first
    load (spawn children inherit it through the environment). The prepare CLI, which never forks after this point,
    sets "true" (and the pool size) before it gets here (prepare.py): a batch then tokenizes on several cores
    instead of one, the biggest lever on the download rate.
    """

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _auto_tokenizer() -> Any:
    """
    transformers.AutoTokenizer for the Hub download of :func:`prepare_tokenizer`, imported lazily (the HF cache env
    must be configurable before the import, and transformers costs seconds no other step needs) and under a lock:
    transformers initialises its lazy modules on first import, which is not thread-safe and the build runs items
    in threads.
    """

    with _IMPORT_LOCK:
        _guard_tokenizers_parallelism()
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


def token_measure(config: DatasetConfig) -> dict[str, Any]:
    """
    The manifest fields that say how a folder's token counts are made: token_count, and with token_count
    tokenizer the tokenizer's name and definition hash (None otherwise: an estimate needs no tokenizer). What
    :func:`new_manifest` records and what the repair step's adopt action rewrites (lib/build/repair.py).
    """

    with_tokenizer = config.token_count == "tokenizer"
    return {
        "token_count": config.token_count,
        "tokenizer": config.tokenizer.name if with_tokenizer else None,
        "tokenizer_hash": config.tokenizer_hash() if with_tokenizer else None,
    }


def new_manifest(
    config: DatasetConfig,
    source: str,
    source_hash: str,
    stage: str,
    *,
    tokens: bool = False,
    truncated_at_tokens: int | None = None,
    dataset_config: str | None = None,
    hash_payload: dict[str, Any] | None = None,
) -> Manifest:
    """
    An empty manifest for stage; with tokens it records how token counts are measured (:func:`token_measure`;
    the tokenizer's hash on raw manifests only, where it decides the tokenizer_changed state). hash_payload is
    the dict source_hash was computed from (raw and processed manifests record it, so a later mismatch is
    explained field by field). Raw manifests record truncated_at_tokens (the dataset_max_sequence_length their
    rows were cut / dropped at) and dataset_config (the file name of the config the folder is downloaded under,
    when the caller knows it).
    """

    measure = token_measure(config) if tokens else {"token_count": None, "tokenizer": None, "tokenizer_hash": None}
    return Manifest(
        source=source,
        source_hash=source_hash,
        stage=stage,
        token_count=measure["token_count"],
        tokenizer=measure["tokenizer"],
        tokenizer_hash=measure["tokenizer_hash"] if stage == "raw" else None,
        hash_payload=hash_payload,
        truncated_at_tokens=truncated_at_tokens,
        versions=library_versions(),
        dataset_config=dataset_config,
    )


def text_row(source: SourceConfig, row: Row, name: str) -> Row:
    """
    The pretrain row to store for a source row: the converter (if any) applied, then text_field alone, as a
    string (None becomes "", which the build's min_chars filter drops). Only the Hub file reader projects
    columns; hf_split and hf_stream deliver every source column, and a surplus column of varying type would
    fail the shard write, a nested one bloat it.
    """

    converter = get_converter(source)
    if converter is not None:
        row = converter(row)
    if source.text_field not in row:
        raise ValueError(f"{name}: row has no {source.text_field!r} column; columns: {sorted(row)}")
    return {source.text_field: text_or_empty(row[source.text_field])}


# --- raw manifest state ------------------------------------------------------------------------------------------------

RawManifestState = Literal["missing", "current", "stale", "outdated", "tokenizer_changed", "unreadable"]


class RawInspection(NamedTuple):
    """
    The state of sources/<name>/raw against the config, its manifest (None when missing) and the reason line
    the download's refusal, the repair step's plan and the status table all phrase from.
    """

    state: RawManifestState
    manifest: Manifest | None
    reason: str  # "missing" | "current" | "stale: source.revision: "a" -> "b"" | "outdated: dataset_max_sequence_length 2048 -> 4096" | "tokenizer changed: ..." | "unreadable manifest ..."

    @property
    def current_manifest(self) -> Manifest | None:
        """
        The manifest when the folder is current (the download appends to it, the build reads it), else None.
        """

        return self.manifest if self.state == "current" else None


def inspect_raw(config: DatasetConfig, name: str, layout: DatasetLayout) -> RawInspection:
    """
    missing (no manifest; the folder may still hold shards, which :func:`download` refuses to start over),
    stale (the manifest's hash differs from config.raw_hash(name): the loader identity changed, the reason lists
    the changed fields; or it is another stage's manifest), outdated (config.dataset_max_sequence_length was
    raised above the cap the rows were truncated / dropped at), tokenizer_changed (the same rows, but their token
    counts were made with another tokenizer or token_count than the config's: the download refuses to append rows
    counted differently, the repair step asks whether to keep the folder under the new tokenizer,
    :func:`token_measure_change`), unreadable (a manifest next to shards that does not parse: a state every caller
    reports and nobody repairs, the rows may have been expensive) or current.
    """

    return inspect_raw_folder(config, config.sources[name], layout.raw_dir(name), layout)


def inspect_raw_folder(config: DatasetConfig, source: SourceConfig, raw_dir: Path, layout: DatasetLayout) -> RawInspection:
    """
    :func:`inspect_raw` for a folder of any source config (a github_code language stored without a source of its
    own has no config entry to look it up by); layout locates the tokenizer manifests the reason line names.
    Stale wins over outdated (the folder holds other rows altogether), outdated over tokenizer_changed (the
    folder is re-downloaded with the new tokenizer anyway).
    """

    try:
        manifest = Manifest.load(raw_dir)
    except RuntimeError:
        return RawInspection("unreadable", None, "unreadable manifest next to shards; fix or delete the directory by hand")
    if manifest is None:
        return RawInspection("missing", None, "missing")
    if manifest.stage != "raw":
        return RawInspection("stale", manifest, f"stale: a {manifest.stage} manifest where a raw one belongs")
    if not manifest.is_current(config.raw_hash_of(source)):
        changes = describe_hash_change(manifest.hash_payload, config.raw_hash_payload_of(source))
        return RawInspection("stale", manifest, "stale: " + "; ".join(changes))
    cap = config.dataset_max_sequence_length
    if manifest.is_outdated(cap):
        return RawInspection("outdated", manifest, f"outdated: dataset_max_sequence_length {manifest.truncated_at_tokens} -> {cap}")
    change = token_measure_change(config, manifest, layout)
    if change is not None:
        return RawInspection("tokenizer_changed", manifest, change)
    return RawInspection("current", manifest, "current")


def token_measure_change(config: DatasetConfig, manifest: Manifest, layout: DatasetLayout) -> str | None:
    """
    Why the config would count tokens differently from how the rows of a raw manifest were counted, or None:
    token_count changed, or (with token_count tokenizer) the tokenizer definition did. A manifest without
    tokenizer_hash (written before it was recorded) counts as made with the current tokenizer: unknown is not a
    change. The line names the old and the new tokenizer and the rows measured under the old one; the stored
    token counts of those rows (and the truncation of pretrain texts) will not match the new tokenizer, which is
    the cost of keeping them (the repair step's adopt action, lib/build/repair.py).
    """

    rows = f"{manifest.rows():,} rows were counted"
    if manifest.token_count is not None and manifest.token_count != config.token_count:
        return f"token_count changed: {manifest.token_count} -> {config.token_count}; {rows} the old way, their stored token counts will not match the new one"
    if config.token_count != "tokenizer" or manifest.tokenizer_hash in (None, config.tokenizer_hash()):
        return None
    old = _stored_tokenizer_label(manifest.tokenizer, manifest.tokenizer_hash, layout)
    new = _tokenizer_label(config.tokenizer.name, config.tokenizer.kind, config.tokenizer.hf_id, config.tokenizer.revision)
    return (
        f"tokenizer changed: {old} -> {new}; {rows} and truncated under the old one, "
        "their token counts and truncation will not match the new tokenizer"
    )


def _tokenizer_label(name: str, kind: str | None, hf_id: str | None, revision: str | None) -> str:
    """
    A tokenizer as the reason lines name it: the directory name, and for a Hub tokenizer its repo and revision.
    """

    if kind == "hf":
        return f"{name} ({hf_id} @ {revision or 'unpinned'})"
    return f"{name} ({kind})" if kind else name


def _stored_tokenizer_label(name: str | None, tokenizer_hash: str | None, layout: DatasetLayout) -> str:
    """
    The tokenizer a raw folder was counted with, as the manifest under tokenizers/<name> describes it, when that
    manifest still is the one the rows were counted with: a same-named definition overwrites the folder in the
    tokenizer step, which runs before the raw folders are inspected, so the hash decides.
    """

    if name is None:
        return "unknown tokenizer"
    stored = Manifest.load(layout.tokenizer_dir(name))
    if stored is None or stored.source_hash != tokenizer_hash:
        return f"{name} (definition {tokenizer_hash}, no longer under tokenizers/)"
    return _tokenizer_label(name, stored.extra.get("kind"), stored.extra.get("hf_id"), stored.extra.get("revision"))


class RawFolderError(RuntimeError):
    """
    A raw folder that :func:`download` may not append to (stale, outdated, tokenizer_changed or unreadable;
    problem is the :func:`inspect_raw` reason). The download never deletes raw data; the repair step does, after
    the user confirmed (lib/build/repair.py), and it asks whether to keep a tokenizer_changed folder; an
    unreadable manifest is the user's to fix or delete.
    """

    def __init__(self, name: str, directory: Path, problem: str) -> None:
        remedy = "the download never deletes raw data"
        if problem.startswith(("tokenizer changed", "token_count changed")):
            remedy = f"{remedy}; the repair step asks whether to keep the folder and go on with the new tokenizer"
        elif not problem.startswith("unreadable"):
            remedy = f"it must be deleted and downloaded again; {remedy}, run the repair step (it asks for confirmation)"
        super().__init__(f"{name}: raw folder {directory} is {problem}; {remedy}")
        self.name = name
        self.directory = directory


def reopen_raw(config: DatasetConfig, name: str, layout: DatasetLayout) -> bool:
    """
    prepare --reopen: clear the exhausted flag of sources/<name>/raw so the next download reads on from its
    offset. A loader that yielded fewer rows than asked is latched exhausted whatever the reason, and only the
    user knows whether the source has more rows now (a grown check_limit reopens by itself,
    :meth:`RawFolder.reopen_if_check_limit_grew`). Returns whether there was a flag to clear; a folder that is
    not current has nothing to reopen.
    """

    manifest = inspect_raw(config, name, layout).current_manifest
    if manifest is None or not manifest.exhausted:
        return False
    RawFolder(layout.raw_dir(name), manifest).reopen()
    return True


# --- tokenizer ---------------------------------------------------------------------------------------------------------


def prepare_tokenizer(config: DatasetConfig, layout: DatasetLayout, *, hf_token: str | None = None) -> Manifest:
    """
    Save the config's tokenizer to layout.tokenizer_dir(name) (Hub download, with hf_token for a gated
    repo, or the synthetic WordLevel one). The files are written to a sibling directory and swapped into place, so
    a stale tokenizer's files never linger next to the new ones.
    """

    tokenizer = config.tokenizer
    tokenizer_dir = layout.tokenizer_dir(tokenizer.name)
    source_hash = config.tokenizer_hash()
    manifest = current_manifest(tokenizer_dir, source_hash, "tokenizer")
    if manifest is not None and (tokenizer_dir / "tokenizer_config.json").is_file():
        return manifest

    log.info("preparing tokenizer %s (%s) -> %s", tokenizer.name, tokenizer.kind, tokenizer_dir)
    with write_atomically(tokenizer_dir) as temporary:
        if tokenizer.kind == "synthetic":
            write_synthetic_tokenizer(temporary)
        else:
            _auto_tokenizer().from_pretrained(tokenizer.hf_id, revision=tokenizer.revision, token=hf_token).save_pretrained(str(temporary))
        shutil.rmtree(tokenizer_dir, ignore_errors=True)  # a non-empty directory cannot be replaced
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
    skipped_malformed: int = 0  # instruct rows whose converter raised ValueError or left out instruction / output
    dropped_too_long: int = 0  # instruct rows with more than `dataset_max_sequence_length` tokens
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
    config_name: str | None = None,
) -> Manifest:
    """
    Append raw shards until rows_needed rows are on disk (no-op if they already are). config_name (the
    dataset config's file name) is recorded in a manifest this call creates, for the repair step.

    The folder's manifest must be current (:func:`inspect_raw`): a stale or outdated one raises
    :class:`RawFolderError` (nothing is deleted here), a missing one starts the folder from shard 0 (refused when
    shards without a manifest are present). manifest.rows_fetched is the loader offset reached (source rows
    consumed). Pretrain sources keep every row (converter applied, text_field alone and a string, the text
    truncated so that tokens, its count with the trainer's specials, is at most dataset_max_sequence_length). Instruct sources
    run the converter and filter at download time and store only standardized {instruction, input, output} rows
    of at most dataset_max_sequence_length tokens; malformed rows (the converter raises ValueError or yields no instruction /
    output) are counted in skipped_malformed, longer rows in dropped_too_long. check_limit bounds the source rows
    inspected in total. A loader that yields fewer rows than requested sets exhausted (the training sampler cycles
    a source smaller than its budget).

    rows_needed is a minimum: a loader reading a large parquet file remotely finishes the row group it is in
    (see sources/loaders.py), every row it yields is written and rows_fetched advances to that row-group
    boundary, so the same bytes are never downloaded twice. Sources without a converter are read with only
    text_field projected (columns); converters and fields mappings get every column.

    Every shard is published and recorded in the manifest (with the loader offset after its last row and the
    skipped / dropped totals up to that row, :class:`RawFolder`) as soon as it is written, so a failure or a stop
    request (checked after every shard) keeps everything fetched so far and the next call resumes from the last
    complete shard without counting anything twice.
    """

    gate = _StopGate(should_stop)
    folder, increment = _plan_increment(
        config, name, layout, rows_needed, token_counter=lambda: TokenCounter(config, layout), should_stop=gate, config_name=config_name
    )
    if increment is None or increment.passive:  # passive: the rows are on disk; only a group pass reads on
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
        _fetch([increment], _tagged(name, rows), bar, postfix, shard_size, gate)
    _finish_increment(folder, increment.source, increment.counters)
    _log_increment(name, increment.counters, folder.manifest)
    return folder.manifest


def _finish_increment(folder: RawFolder, source: SourceConfig, counters: _IncrementCounters) -> None:
    """
    Hand the increment's totals to the folder; a source stopped by its own check_limit records that limit
    (so a later, larger one reopens it) instead of counting as a loader that ran dry.
    """

    if counters.consumed == 0 and folder.shard_count == 0 and folder.rows_fetched == 0 and not counters.exhausted:
        return  # a passive folder no row reached (a language discovered out of alignment): not created at all
    limit = source.check_limit
    by_limit = limit if limit is not None and folder.start_offset + counters.consumed >= limit else None
    progress_now = RowProgress(counters.consumed, counters.skipped_malformed, counters.dropped_too_long)
    folder.finish(progress_now, exhausted=counters.exhausted, check_limit=by_limit)


def _log_increment(name: str, counters: _IncrementCounters, manifest: Manifest) -> None:
    log.info(
        "%s: kept %d of %d fetched rows (%d rows on disk; %d malformed skipped, %d too long dropped)",
        name, counters.kept, counters.consumed, manifest.rows(), counters.skipped_malformed, counters.dropped_too_long,
    )


def _raw_folder_to_append_to(
    config: DatasetConfig, name: str, layout: DatasetLayout, *, should_stop: StopCheck | None = None, config_name: str | None = None
) -> RawFolder:
    """
    The raw folder of name around its current manifest, or a fresh one (truncated_at_tokens =
    dataset_max_sequence_length, dataset_config = config_name) when the directory has none. Refused when the folder is stale, outdated or unreadable
    (:class:`RawFolderError`; deleting it is the repair step's or the user's decision) or holds shards without any manifest:
    nothing would say where those rows came from, and starting over would delete them.
    """

    return _raw_folder_for(config, name, config.sources[name], layout, should_stop=should_stop, config_name=config_name)


def _raw_folder_for(
    config: DatasetConfig, name: str, source: SourceConfig, layout: DatasetLayout, *, should_stop: StopCheck | None, config_name: str | None
) -> RawFolder:
    """
    :func:`_raw_folder_to_append_to` for a folder of any source config (a language stored without a source of
    its own): keyed by that source's raw hash.
    """

    raw_dir = layout.raw_dir(name)
    inspection = inspect_raw_folder(config, source, raw_dir, layout)
    if inspection.state not in ("missing", "current"):
        raise RawFolderError(name, raw_dir, inspection.reason)
    manifest = inspection.manifest
    if manifest is None:
        if has_shards(raw_dir):
            raise RuntimeError(f"{name}: {raw_dir} holds shards but no manifest; delete the directory to download the source again")
        manifest = new_manifest(
            config, name, config.raw_hash_of(source), "raw", tokens=True, truncated_at_tokens=config.dataset_max_sequence_length,
            dataset_config=config_name, hash_payload=config.raw_hash_payload_of(source),
        )
    return RawFolder(raw_dir, manifest, config_cap=config.dataset_max_sequence_length, should_stop=should_stop)


class _StopGate:
    """
    The stop check of one download pass (handed to every raw folder as its should_stop). A stop request trips it
    once, and from then on it is suspended: the rows the pass already consumed are stored without another abort
    on the way (each shard publish checks the stop again), and the pass ends after that.
    """

    def __init__(self, should_stop: StopCheck | None) -> None:
        self._should_stop = should_stop
        self.tripped = False
        self.suspended = False

    def __call__(self) -> bool:
        if self.suspended or self._should_stop is None:
            return False
        if self._should_stop():
            self.tripped = True
            return True
        return False

    def suspend(self) -> None:
        self.suspended = True


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

    Every stored tokens counts the text plus :data:`NUMBER_OF_SPECIAL_TOKENS` (the BOS and EOS the trainer adds), the one
    place the specials enter a count. Pretrain rows: text_field is truncated (truncation.py) so that this sum is at
    most max_tokens. Instruct rows: tokens counts the trainer's text (row_pipeline.instruct_text) uncapped; a row
    over max_tokens is dropped (counters.dropped_too_long), never cut, and every stored row's progress carries the
    drop count of the rows before it (exact per row, so a resume never double counts).
    """

    def __init__(self, source: SourceConfig, counter: TokenCounter, max_tokens: int, counters: _IncrementCounters) -> None:
        if max_tokens < NUMBER_OF_SPECIAL_TOKENS:
            raise ValueError(f"dataset_max_sequence_length {max_tokens} leaves no room for the {NUMBER_OF_SPECIAL_TOKENS} special tokens of a row")
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
        texts = [row[self._text_field] for row, _ in batch]
        for (row, _), (cut, tokens) in zip(batch, self._counter.truncate_many(texts, self._max_tokens - NUMBER_OF_SPECIAL_TOKENS), strict=True):
            row[self._text_field] = cut
            row["tokens"] = tokens + NUMBER_OF_SPECIAL_TOKENS
        return batch

    def _drop_long_instruct_rows(self, batch: list[StoredRow]) -> list[StoredRow]:
        stored: list[StoredRow] = []
        for (row, before), count in zip(batch, self._counter.count_many([instruct_text(row) for row, _ in batch]), strict=True):
            tokens = count + NUMBER_OF_SPECIAL_TOKENS
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
    passive: bool = False  # stores whatever rows the pass hands it (a group member past its target, or a language without a source); never bounds the pass
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

        if self.passive:
            return 0
        if self.is_instruct:
            return UNBOUNDED_COUNT if self.max_consume is None else self.max_consume
        return self.rows_to_keep if self.max_consume is None else min(self.rows_to_keep, self.max_consume)

    @property
    def done(self) -> bool:
        """
        No more source rows are taken: the consume budget is spent (check_limit bounds the source rows
        consumed, whatever the loader yields beyond its count), or an instruct source kept its rows_to_keep rows.
        """

        if self.passive:
            return False
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
    config_name: str | None = None,
) -> tuple[RawFolder, _Increment | None]:
    """
    Open the raw folder of name and decide what this pass fetches for it: None when there is nothing to do
    (the source is exhausted, or its check_limit is spent, recorded as the exhaustion), a passive increment when
    the rows are on disk (a group pass stores what it reads on for the others; a single download does nothing).
    """

    source = fetch_source(config, config.sources[name])
    folder = _raw_folder_to_append_to(config, name, layout, should_stop=should_stop, config_name=config_name)
    folder.reopen_if_check_limit_grew(source.check_limit)
    if folder.exhausted:
        log.info("%s: source exhausted after %d rows, nothing more to fetch", name, folder.rows_fetched)
        return folder, None
    wanted = max(rows_needed - folder.rows, 0)
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
    return folder, _Increment(name, source, folder, wanted, max_consume, token_step, counters, converter, row_filter, passive=wanted == 0)


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

    A failure on the worker is kept and re-raised on the fetch thread by the next :meth:`submit`, :meth:`drain`
    or :meth:`close` (:attr:`failed` tells earlier). After a tokenizer or write error the worker only settles what
    is queued without storing it; after :class:`BuildAborted` (the stop check, raised by a shard publish) it
    stores on: the gate is suspended by then (:func:`_store`), and every row the pass consumed belongs on disk so
    the folders' offsets stay the pass's frontier. close is what leaving the with block does: it joins the thread
    whatever happened, so the shard writers are closed after the worker is done with them.
    """

    def __init__(self, name: str, writers: dict[str, ShardWriter], bar: Progress, gate: _StopGate) -> None:
        self._queue: queue.Queue[tuple[_Increment, list[StoredRow]] | None] = queue.Queue(maxsize=TOKEN_QUEUE_DEPTH)
        self._writers = writers
        self._bar = bar
        self._gate = gate
        self._failure: BaseException | None = None
        self._storing = True  # False after an error other than the stop: the queued batches are settled unstored
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
                if self._storing:
                    try:
                        _store(increment, self._writers[increment.name], increment.token_step.tokenize(batch), self._bar, self._gate)
                    except BuildAborted as stop:  # the gate is suspended now: keep storing what is queued
                        self._failure = self._failure or stop
                    except BaseException as error:  # noqa: BLE001  # whatever it is, the fetch thread re-raises it
                        self._failure = self._failure or error
                        self._storing = False
                increment.settled += len(batch)  # after the store: `kept` is up to date before the rows leave `in_flight`
            finally:
                self._queue.task_done()


PASSIVE_SHARD_DIVISOR = 4  # passive increments publish shards of shard_size / this: one buffer per language, bounded together


def _fetch(
    increments: list[_Increment], rows: Iterator[tuple[str, Row]], bar: Progress, postfix: _DownloadPostfix, shard_size: int, gate: _StopGate
) -> None:
    """
    One download pass: every (name, row) of rows goes to its increment (counted as consumed, converted,
    batched for the token worker, which tokenizes the batches and appends them shard by shard to the raw
    directory, one shard writer per increment) until every increment is :attr:`~_Increment.done` or the stream
    ends (the stream is closed either way); the last batches are submitted, the worker joined, and an active
    increment that kept fewer rows than it wanted is exhausted. An instruct increment submits early and waits for
    the worker when the rows in flight and buffered would meet the target, so it stops exactly there and never
    reads on while the outcome is open (a second pass would re-stream the file prefix). bar tracks kept rows
    (postfix: source rows consumed, rows of passive increments, current repo file; the bytes fetched are the
    counter the bar was created with).

    increments may grow while the pass runs (a group pass discovers languages): a row of a name not seen before
    finds its increment in the list and gets a shard writer then. On a stop or a failure everything consumed so far
    is stored before the error propagates: the stop check is suspended (gate), the rows still buffered here are
    submitted, the worker drains its queue, and every writer publishes its buffered rows as a short final shard
    (:func:`_flush_partial_shards`).
    """

    increments_by_name: dict[str, _Increment] = {}
    writers: dict[str, ShardWriter] = {}
    consumed_total = surplus_total = 0

    def increment_named(name: str) -> _Increment:
        if name not in increments_by_name:
            increments_by_name.update((increment.name, increment) for increment in increments)
        increment = increments_by_name[name]
        if name not in writers:
            size = max(shard_size // PASSIVE_SHARD_DIVISOR, 1) if increment.passive else shard_size
            writers[name] = ShardWriter(increment.folder.directory, size, start_shard=increment.folder.shard_count, on_shard=increment.folder.record_shard).__enter__()
        return increment

    for increment in increments:
        increment_named(increment.name)
    worker = _TokenWorker(",".join(writers), writers, bar, gate)
    failure: BaseException | None = None
    try:
        for name, raw in rows:
            if worker.failed:
                worker.drain()  # raises: stop pulling rows for a worker that stores nothing anymore
            increment = increment_named(name)
            if increment.done:
                if all(increment.done for increment in increments):
                    break
                continue  # this source is done, the others read on
            counters = increment.counters
            counters.consumed += 1
            if increment.passive:
                surplus_total += 1
                postfix.surplus(surplus_total)
            else:
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
    except BaseException as error:  # noqa: BLE001  # re-raised below, after the buffered rows were saved
        failure = error
        gate.suspend()
        for increment in increments:  # the rows consumed but not handed over yet (submit raises nothing: the failure was raised)
            worker.submit(increment, increment.token_step.take())
    finally:
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    try:
        worker.close()  # joins the thread; raises the worker's own failure if none was raised yet
    except BaseException as error:  # noqa: BLE001
        failure = failure or error
    if failure is None:
        for writer in writers.values():
            writer.__exit__(None, None, None)  # the last partial shards
        for increment in increments:
            if not increment.passive and increment.counters.kept < increment.rows_to_keep:
                increment.counters.exhausted = True  # the loader ran dry (or the budget was spent) before `rows_to_keep` rows were kept
        return
    _flush_partial_shards(writers)
    for writer in writers.values():
        writer.__exit__(type(failure), failure, failure.__traceback__)
    raise failure


def _flush_partial_shards(writers: dict[str, ShardWriter]) -> None:
    """
    After a stop or a failure: publish every writer's buffered rows as a short shard, so each folder's offset is
    where its increment really stood (a rare language's shard fills over millions of scanned rows, and every
    passive increment has to resume at the same frontier as the active ones). Best effort: a writer whose publish
    fails (or raises the stop again, which `RawFolder.record_shard` does after recording) is logged and skipped.
    """

    for name, writer in writers.items():
        try:
            writer.flush()
        except BuildAborted:
            pass  # the shard was published and recorded before the stop check raised again
        except Exception as error:  # noqa: BLE001
            log.warning("%s: could not publish the buffered rows after the failure: %s", name, error)


def _store(increment: _Increment, writer: ShardWriter, stored: list[StoredRow], bar: Progress, gate: _StopGate) -> None:
    """
    The rows the token step released, appended and counted as kept. A stop raised by a shard publish on the way
    (the shard is published and recorded before the check) suspends the gate, the rest of the batch is stored, and
    the stop is raised at the end.
    """

    stop: BuildAborted | None = None
    for row, row_progress in stored:
        try:
            increment.folder.add(writer, row, row_progress)
        except BuildAborted as error:
            stop = error
            gate.suspend()
        increment.counters.kept += 1
    bar.update(len(stored))  # once per batch: the dashboard bar takes a lock per update
    if stop is not None:
        raise stop


def download_github_code_group(
    config: DatasetConfig,
    names: list[str],
    layout: DatasetLayout,
    *,
    rows_needed: dict[str, int],
    shard_size: int = DEFAULT_SHARD_SIZE,
    hf_token: str | None = None,
    should_stop: StopCheck | None = None,
    config_name: str | None = None,
) -> dict[str, Manifest]:
    """
    :func:`download` for several `github_code` sources of one repo in a single pass over its files: every
    row group is fetched once and its rows are dispatched to the language source that wants them. A member that
    has its rows_needed[name] (from the start, or once it reached it) stays in the pass passively and stores every
    further row of its language the pass reads for the others; a member that spent its check_limit or is
    exhausted takes nothing. The rows of a language no member names go to a raw folder of their own
    (:func:`_extra_increment`). Nothing is read when no member has rows to fetch. The raw shards (texts truncated
    by the same token step), rows_fetched and exhausted of every member are, up to its target, exactly what a
    separate download call would produce: the same :func:`_fetch` pass over the repo reader instead of one loader.
    A stale or outdated member raises :class:`RawFolderError` before anything is fetched. Returns the raw manifest
    of every source in names and of every extra language the pass stored.
    """

    for name in names:
        source = config.sources[name]
        if source.loader != "github_code":
            raise ValueError(f"{name}: download_github_code_group needs github_code sources")
        if github_code_repo_key(source) != github_code_repo_key(config.sources[names[0]]):
            raise ValueError(f"{name}: github_code group members must share hf_id, revision and data_files")
    token_counter = functools.cache(lambda: TokenCounter(config, layout))
    gate = _StopGate(should_stop)
    results: dict[str, Manifest] = {}
    increments: list[_Increment] = []
    for name in names:
        folder, increment = _plan_increment(
            config, name, layout, rows_needed[name], token_counter=token_counter, should_stop=gate, config_name=config_name
        )
        results[name] = folder.manifest
        if increment is not None:
            increments.append(increment)
    if not any(not increment.passive for increment in increments):
        return results  # nothing to fetch: nothing is read, so nothing could be collected

    for increment in increments:
        if increment.passive:
            log.info("%s: has its rows; storing what the pass reads on from offset %d -> %s", increment.name, increment.folder.rows_fetched, increment.folder.directory)
        else:
            log.info(
                "%s: fetching %d rows from offset %d -> %s",
                increment.name, increment.rows_to_keep, increment.folder.rows_fetched, increment.folder.directory,
            )
    requests = [
        GithubCodeRequest(increment.name, increment.source, increment.folder.rows_fetched, increment.loader_count, passive=increment.passive)
        for increment in increments
    ]
    template = increments[0].source
    repo = str(template.hf_id)
    columns = _union_columns([loader_columns(increment.source) for increment in increments])
    fetch_stats = FetchStats()

    def discover(language: str) -> ReadRequest | None:
        extra = _extra_increment(config, names, template, language, layout, token_counter, should_stop=gate, config_name=config_name)
        if extra is None:
            return None
        increments.append(extra)
        return language_request(extra.name, language, extra.folder.rows_fetched, 0, passive=True)

    with progress(
        total=sum(increment.rows_to_keep for increment in increments), desc=f"{repo} ({len(increments)} languages)", unit="row", panel="downloads",
        bytes_fetched=lambda: fetch_stats.bytes_fetched,
    ) as bar:
        postfix = _DownloadPostfix(bar)
        shared_parameters = SharedLoaderParameters(
            token=hf_token, index_dir=layout.hub_index_dir(), on_file=postfix.on_file, stats=fetch_stats, columns=columns,
        )
        rows = read_github_code_group(requests, shared_parameters, discover=discover)
        try:
            _fetch(increments, rows, bar, postfix, shard_size, gate)
        finally:
            for increment in increments:  # after a stop or failure too: the flushed shards moved the offsets
                results[increment.name] = increment.folder.manifest
    for increment in increments:
        _finish_increment(increment.folder, increment.source, increment.counters)
        _log_increment(increment.name, increment.counters, increment.folder.manifest)
    _log_surplus(repo, names, increments)
    return results


def _extra_increment(
    config: DatasetConfig,
    members: list[str],
    template: SourceConfig,
    language: str,
    layout: DatasetLayout,
    token_counter: Callable[[], TokenCounter],
    *,
    should_stop: StopCheck | None,
    config_name: str | None,
) -> _Increment | None:
    """
    The passive increment storing the rows of a language no member of the group names (the reader offers each
    such language once): its source is template with the language replaced, its folder
    sources/<github_code_extra_name>/raw keyed by that source's raw hash, so a later config entry of that name
    (same repo, revision, data_files and text_field) adopts the folder. None, with a warning, when that name is a
    configured source outside the group, or when its folder exists but is stale / outdated / unreadable / without a
    manifest (never deleted here; the user decides) or exhausted.
    """

    name = github_code_extra_name(template, language)
    if name in config.sources and name not in members:
        log.warning("%s: not storing rows of %s: %s is a configured source outside this github_code group", template.hf_id, language, name)
        return None
    source = fetch_source(config, github_code_extra_source(template, language))
    try:
        folder = _raw_folder_for(config, name, source, layout, should_stop=should_stop, config_name=config_name)
    except (RawFolderError, RuntimeError) as error:
        log.warning("%s: not storing rows of %s: %s", template.hf_id, language, error)
        return None
    if folder.exhausted:
        log.warning("%s: not storing rows of %s: %s is marked exhausted", template.hf_id, language, folder.directory)
        return None
    if folder.shard_count == 0 and not folder.manifest.extra:
        folder.manifest.extra = {"github_code_group": list(members), "surplus": True}
    log.info("%s: storing rows of %s (no source of its own) from offset %d -> %s", template.hf_id, language, folder.rows_fetched, folder.directory)
    counters = _IncrementCounters()
    token_step = _TokenStep(source, token_counter(), folder.cap, counters)
    return _Increment(name, source, folder, 0, None, token_step, counters, None, None, passive=True)


def _log_surplus(repo: str, members: list[str], increments: list[_Increment]) -> None:
    """
    One line per pass: the rows members stored past their target, and the languages without a source.
    """

    surplus = [
        f"{increment.name} +{increment.counters.kept - increment.rows_to_keep:,}"
        for increment in increments if increment.name in members and increment.counters.kept > increment.rows_to_keep
    ]
    extras = [f"{increment.name} {increment.counters.kept:,}" for increment in increments if increment.name not in members and increment.counters.kept]
    if surplus:
        log.info("%s: rows stored past the target: %s", repo, ", ".join(surplus))
    if extras:
        log.info("%s: rows stored for languages without a source: %s", repo, ", ".join(extras))


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
    The standardized {instruction, input, output} row for raw, every field a string (None becomes "").

    A malformed row raises ValueError, which the caller counts and skips: the converter's own, or a result
    without instruction / output.
    """

    row = converter(raw) if converter is not None else raw
    missing = [key for key in ("instruction", "output") if key not in row]
    if missing:
        raise ValueError(f"row has no {missing} column; columns: {sorted(row)}")
    return {"instruction": text_or_empty(row["instruction"]), "input": text_or_empty(row.get("input")), "output": text_or_empty(row["output"])}


class _DownloadPostfix:
    """
    The download bar's postfix: source rows consumed, current repo file (refreshed sparsely).
    """

    def __init__(self, bar: Progress) -> None:
        self._bar = bar
        self._values: dict[str, Any] = {"consumed": 0}

    def surplus(self, total: int) -> None:
        """
        Record the running count of rows stored by passive increments; the bar is refreshed every 100 rows.
        """

        self._values["surplus"] = total
        if total % 100 == 0:
            self._refresh()

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
