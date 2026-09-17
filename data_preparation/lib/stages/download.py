# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The download step (sources/<source>/raw/) and the tokenizer step, plus the token counter and manifest helpers
that stages/build.py shares.

The single-source and grouped entry points and the row-dispatch loop remain here. download_state.py owns
conversion and token batches; download_workers.py owns the worker and ordered shutdown; download_groups.py
and download_progress.py provide group planning and dashboard setup.

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
pool is a separate matter (`TOKENIZERS_PARALLELISM`, see :func:`_guard_tokenizers_parallelism`); with a
:class:`~data_preparation.lib.stages.tokenizer_pool.TokenizerPool` (`--tokenizer_threads` above 8) the
:class:`TokenCounter` hands the batches' tokenizer work to separate processes and the worker keeps one batch per
process in flight, still storing in submission order.

The github_code group pass (:func:`download_github_code_group`) keeps everything it decodes: a member that has its
rows stays in the pass as a *passive* increment and stores every further row of its language the pass reads for
the others, and a language no source names gets a raw folder of its own (:func:`_extra_increment`; named by
`loaders.github_code_extra_name`, so a later config entry of that name adopts it). Passive rows are truncated and
counted like any other pretrain row and never bound the pass (`rows_to_keep` 0, never exhausted). A stop or a
failure attempts to preserve everything the pass already consumed (the stop check is suspended,
:class:`_StopGate`; pending batches are submitted, the input stream closed, the worker joined, then every writer
attempts its buffered short final shard, :func:`_flush_partial_shards`). On an ordinary stop, every folder's
offset is the frontier the pass reached: the file index only records row groups the pass consumed whole, and a
passive folder resumes aligned from there. Storage errors propagate and offsets reflect only committed progress.
"""

from __future__ import annotations

import functools
import os
import shutil
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, NamedTuple

from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig, describe_hash_change
from data_preparation.lib.download_profile import measure_rows, profile_source
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.abort import StopCheck
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.converters import get_converter, get_filter
from data_preparation.lib.sources.hub_files import FetchStats, ReadRequest
from data_preparation.lib.sources.loaders import (
    MAX_CACHED_FILE_KEY,
    Row,
    SharedLoaderParameters,
    get_loader,
    github_code_extra_name,
    github_code_extra_source,
    language_request,
    read_github_code_group,
)
from data_preparation.lib.sources.synthetic import write_synthetic_tokenizer
from data_preparation.lib.storage.atomic import _fsync_directory
from data_preparation.lib.storage.ownership import guarded_path
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
from data_preparation.lib.stages.tokenizer_pool import TokenizerPool
from data_preparation.lib.stages.truncation import estimate_tokens, truncate_many
from data_preparation.lib.storage.manifest import Manifest, has_shards, library_versions
from data_preparation.lib.storage.parquet import ShardWriter
from data_preparation.lib.storage.raw_folder import RawFolder, RowProgress
from data_preparation.lib.storage.tokenizer_assessment import assess_tokenizer_folder
from data_preparation.lib.stages.download_groups import create_github_requests, log_group_download_plan, validate_github_group
from data_preparation.lib.stages.download_progress import open_download_progress

from data_preparation.lib.stages.download_state import (
    MALFORMED_WARNINGS_PER_INCREMENT as MALFORMED_WARNINGS_PER_INCREMENT,  # noqa: PLC0414  # preserve the existing stage import
    MAX_CONSECUTIVE_MALFORMED as MAX_CONSECUTIVE_MALFORMED,  # noqa: PLC0414  # preserve the existing stage import
    MalformedSourceError as MalformedSourceError,  # noqa: PLC0414  # preserve the existing stage import
    _DownloadPostfix, _Increment, _IncrementCounters, _TokenStep,
    text_row as text_row,  # noqa: PLC0414  # preserve the existing stage import
)
from data_preparation.lib.stages.download_workers import (
    _DownloadFailures, _StopGate, _TokenWorker, finish_download_pass, open_increment_writer,
)

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000

# --- token counting ----------------------------------------------------------------------------------------------------


class TokenCounter:
    """
    Token counts with the config's tokenizer (token_count: tokenizer, no special tokens) or
    len(text) // 4 (estimate): the text's own tokens, without the BOS and EOS the trainer adds (the token step adds
    truncation.NUMBER_OF_SPECIAL_TOKENS to what it stores). Counts are never capped here: the download truncates pretrain
    *text* at the cap (:meth:`truncate_many`) and drops long instruct rows, so every stored count is a true count.

    With a pool (tokenizer_pool.py; only in tokenizer mode) the token step sends its batches there instead of
    calling :meth:`truncate_many` / :meth:`count_many` here: :attr:`pool` and :attr:`tokenizer_dir` are what it
    needs for that. The methods here stay the in-process path (and what the pool processes run).
    """

    def __init__(self, config: DatasetConfig, layout: DatasetLayout, pool: TokenizerPool | None = None) -> None:
        self.mode = config.token_count
        self.tokenizer_name = config.tokenizer.name
        self.tokenizer_dir = layout.tokenizer_dir(config.tokenizer.name)
        self._tokenizer: SavedTokenizer | None = None
        self.pool: TokenizerPool | None = None
        if self.mode == "tokenizer":
            self._tokenizer = _load_tokenizer(self.tokenizer_dir, self.tokenizer_name)
            self.pool = pool

    @property
    def chat_tokenizer(self) -> SavedTokenizer:
        if self._tokenizer is None:
            raise ValueError("message conversations require token_count: tokenizer")
        return self._tokenizer

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return estimate_tokens(text)
        return len(self._tokenizer.encode(text))

    def count_many(self, texts: list[str]) -> list[int]:
        if self._tokenizer is None:
            return [estimate_tokens(text) for text in texts]
        return self._tokenizer.count_batch(texts)  # no Python ID lists or unused character offsets

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
    manifest still is the one the rows were counted with. Preparation inspects before replacing a same-named
    definition, but another configuration may already have replaced it, so the hash decides.
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


@dataclass(frozen=True)
class TokenizerPlan:
    """Read-only publication decision; valid while the caller holds the dataset lock."""

    directory: Path
    current: Manifest | None

    @property
    def needs_publication(self) -> bool:
        return self.current is None


def inspect_tokenizer(config: DatasetConfig, layout: DatasetLayout) -> TokenizerPlan:
    """Decide whether the tokenizer stage needs work without acquiring or changing any files."""

    directory = layout.tokenizer_dir(config.tokenizer.name)
    guarded_path(layout.root, directory)
    assessment = assess_tokenizer_folder(directory, config.tokenizer_hash())
    # No artifacts is normal first-time preparation, including a precreated empty directory.
    # Keep warnings for existing incomplete/stale tokenizer artifacts: the dashboard retains them.
    if not assessment.ready and directory.exists() and any(directory.iterdir()):
        log.warning("%s; rebuilding from scratch", assessment.problem)
    return TokenizerPlan(directory, assessment.manifest if assessment.ready else None)


def prepare_tokenizer(config: DatasetConfig, layout: DatasetLayout, *, hf_token: str | None = None) -> Manifest:
    """Prepare and publish a tokenizer. Dataset orchestration authorizes repairs before calling this stage."""

    return prepare_planned_tokenizer(config, inspect_tokenizer(config, layout), hf_token=hf_token)


def prepare_planned_tokenizer(config: DatasetConfig, plan: TokenizerPlan, *, hf_token: str | None = None) -> Manifest:
    """Acquire and validate privately, then publish the payload and its manifest together.

    Acquisition, validation and manifest-writing failures preserve the published tokenizer. Only this attempt's
    unique staging directory is cleaned. Publication replaces a nonempty directory by removing it first: failures
    or crashes once removal begins are not rolled back. The orchestrator executes raw/processed repairs only after
    publication succeeds; subsequent failures are likewise not a multi-directory rollback transaction.
    """

    if config.tokenizer.profile:
        from tokenization.profile import recover_profile

        recover_profile(plan.directory)
    if plan.current is not None:
        assessment = assess_tokenizer_folder(plan.directory, config.tokenizer_hash(), validate_payload=True)
        if (assessment.ready and assessment.manifest is not None
                and (not config.tokenizer.profile or SavedTokenizer(plan.directory).profile == config.tokenizer.profile)):
            return assessment.manifest
        log.warning("%s; preparing a validated replacement", assessment.problem)
    tokenizer = config.tokenizer
    tokenizer_dir = plan.directory
    guarded_path(tokenizer_dir.parent.parent, tokenizer_dir)
    log.info("preparing tokenizer %s (%s) -> %s", tokenizer.name, tokenizer.kind, tokenizer_dir)
    tokenizer_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tokenizer-", dir=tokenizer_dir.parent) as staging:
        temporary = Path(staging) / "payload"
        if tokenizer.profile:
            from tokenization.profile import build_profile

            build_profile(temporary, tokenizer_dir.parent / "llama-32k", token=hf_token)
        elif tokenizer.kind == "synthetic":
            write_synthetic_tokenizer(temporary)
        else:
            _auto_tokenizer().from_pretrained(tokenizer.hf_id, revision=tokenizer.revision, token=hf_token).save_pretrained(str(temporary))
        if not (temporary / "tokenizer_config.json").is_file():
            raise ValueError(f"tokenizer {tokenizer.name!r}: prepared replacement has no tokenizer_config.json")
        _load_tokenizer(temporary, tokenizer.name)  # verify it is usable before removing published data
        manifest = new_manifest(config, tokenizer.name, config.tokenizer_hash(), "tokenizer")
        manifest.extra = {"kind": tokenizer.kind, "hf_id": tokenizer.hf_id, "revision": tokenizer.revision}
        manifest.complete_generation(temporary)
        if tokenizer.profile:
            from tokenization.profile import publish_profile

            publish_profile(temporary, tokenizer_dir)
        else:
            if tokenizer_dir.exists():
                shutil.rmtree(tokenizer_dir)  # legacy publication semantics
            os.replace(temporary, tokenizer_dir)
        _fsync_directory(tokenizer_dir.parent)
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
    tokenizer_pool: TokenizerPool | None = None,
) -> Manifest:
    """
    Append raw shards until rows_needed rows are on disk (no-op if they already are). config_name (the
    dataset config's file name) is recorded in a manifest this call creates, for the repair step. tokenizer_pool
    (tokenizer_pool.py) takes the batches' tokenizer work off this process; without it the token worker thread
    tokenizes on the process's own Rust pool.

    The folder's manifest must be current (:func:`inspect_raw`): a stale or outdated one raises
    :class:`RawFolderError` (nothing is deleted here), a missing one starts the folder from shard 0 (refused when
    shards without a manifest are present). manifest.rows_fetched is the loader offset reached (source rows
    consumed). Pretrain sources keep every row (converter applied, text_field alone and a string, the text
    truncated so that tokens, its count with the trainer's specials, is at most dataset_max_sequence_length). Instruct sources
    run the converter and filter at download time and store only standardized {instruction, input, output} rows
    of at most dataset_max_sequence_length tokens; malformed rows (the converter raises ValueError or yields no instruction /
    output) are logged at WARNING and counted in skipped_malformed, longer rows in dropped_too_long;
    :data:`MAX_CONSECUTIVE_MALFORMED` malformed rows in a row fail the download with :class:`MalformedSourceError`
    (a wrong fields / converter is found within the first rows, not after a night of skipping). check_limit bounds the source rows
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

    # Plan the append and return immediately when no rows are needed.
    gate = _StopGate(should_stop)
    folder, increment = _plan_increment(
        config, name, layout, rows_needed, token_counter=lambda: TokenCounter(config, layout, tokenizer_pool), should_stop=gate, config_name=config_name
    )
    if increment is None or increment.passive:  # passive: the rows are on disk; only a group pass reads on
        return folder.manifest

    # Stream the source through conversion, tokenization, and shard publication.
    log.info("%s: fetching %d rows from offset %d -> %s", name, increment.rows_to_keep, folder.rows_fetched, folder.directory)
    loader = get_loader(increment.source.loader)
    fetch_stats = FetchStats()
    with profile_source(name), open_download_progress([increment], name, fetch_stats) as (bar, postfix):
        shared_parameters = SharedLoaderParameters(
            token=hf_token, index_dir=layout.hub_index_dir(), on_file=postfix.on_file, stats=fetch_stats,
            columns=loader_columns(increment.source),
            download_prefetch_mb=config.download_prefetch_mb,
        )
        rows = loader(increment.source, folder.rows_fetched, increment.loader_count, shared_parameters)
        _fetch([increment], _tagged(name, rows), bar, postfix, shard_size, gate)

    # Publish final progress after the fetch and its cleanup succeeded.
    with profile_source(name):
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
    if source.instruction_format == "messages":
        log.info("%s: conversation pass diagnostics: %s", folder.directory, dict(counters.chat))


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
    reads on while the outcome is open (a second pass would re-stream the file prefix). bar tracks the rows kept up
    to each active increment's target (:meth:`_Increment.credit`; postfix: source rows consumed towards a target,
    surplus rows, i.e. rows of passive increments and rows an active increment consumes past its target, and the
    current repo file; the bytes fetched are the counter the bar was created with).

    increments may grow while the pass runs (a group pass discovers languages): a row of a name not seen before
    finds its increment in the list and gets a shard writer then. Cleanup settles pending batches, closes rows,
    joins the worker, salvages partial shards on failure, and finalizes writers, in that order. Every applicable
    action is attempted once, sequentially, even if an earlier action failed. Storage failure may prevent salvage;
    only persisted progress counts. The original substantive failure takes precedence over cooperative stop,
    with secondary failures attached as traceback notes. Writers are never finalized while the worker is live.
    """

    # Open writers as sources enter the pass, then start their shared token worker.
    increments_by_name: dict[str, _Increment] = {}
    writers: dict[str, ShardWriter] = {}
    consumed_total = surplus_total = 0

    failures = _DownloadFailures(gate)
    worker: _TokenWorker | None = None
    try:
        for increment in increments:
            open_increment_writer(increment.name, increments, increments_by_name, writers, shard_size)
        in_flight = max((increment.token_step.parallel_batches for increment in increments), default=1)
        worker = _TokenWorker(",".join(writers), writers, bar, gate, in_flight=in_flight)

        # Keep the row dispatch loop visible; settle instruction batches before reading past their target.
        for name, raw in measure_rows(rows):
            if worker.failed:
                worker.drain()  # raises: stop pulling rows for a worker that stores nothing anymore
            increment = open_increment_writer(name, increments, increments_by_name, writers, shard_size)
            if increment.done:
                if all(increment.done for increment in increments):
                    break
                continue  # this source is done, the others read on
            counters = increment.counters
            counters.consumed += 1
            if increment.passive or counters.consumed > increment.rows_to_keep:
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
                worker.submit(increment, increment.token_step.take())  # settle the possible target before reading on
                worker.drain()
            if all(increment.done for increment in increments):
                break  # enough: stop pulling; the ordered cleanup below closes the stream
    except BaseException as error:  # noqa: BLE001  # preserve the processing failure through cleanup
        failures.record("processing", error)

    # Finish in ownership order and report any processing or cleanup failures.
    finish_download_pass(increments, rows, writers, worker, failures)


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
    tokenizer_pool: TokenizerPool | None = None,
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
    of every source in names and of every extra language the pass stored. tokenizer_pool as in :func:`download`.
    """

    # Validate the group and plan each source without reading any rows.
    validate_github_group(config, names)
    token_counter = functools.cache(lambda: TokenCounter(config, layout, tokenizer_pool))
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

    # Open one shared reader, retaining passive members and discovering extra languages.
    log_group_download_plan(increments)
    requests = create_github_requests(increments)
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

    with profile_source(",".join(names)), open_download_progress(increments, f"{repo} ({len(increments)} languages)", fetch_stats) as (bar, postfix):
        shared_parameters = SharedLoaderParameters(
            token=hf_token, index_dir=layout.hub_index_dir(), on_file=postfix.on_file, stats=fetch_stats, columns=columns,
            download_prefetch_mb=config.download_prefetch_mb,
        )
        rows = read_github_code_group(requests, shared_parameters, discover=discover)
        try:
            _fetch(increments, rows, bar, postfix, shard_size, gate)
        finally:
            for increment in increments:  # after a stop or failure too: the flushed shards moved the offsets
                results[increment.name] = increment.folder.manifest

    # Finalize member counters only after the whole shared pass succeeded.
    for increment in increments:
        with profile_source(increment.name):
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


def loader_columns(source: SourceConfig) -> list[str] | None:
    """
    Column projection for a source's loader (applied whatever the file format): [text_field] for pretrain
    sources read as-is, None (every column) when a converter or fields mapping may need others or the rows are
    instruct rows.
    """

    if source.kind == "instruct" or get_converter(source) is not None:
        return None
    return [source.text_field]
