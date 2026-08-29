# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pipeline stages shared by every source kind: tokenizer, raw download, held-out sets; plus the token counter and
manifest helpers used by ``stages_pretrain.py`` / ``stages_instruct.py``.

Every stage is a function ``(cfg, name, layout, *options) -> Manifest`` that is **idempotent via the manifest**
(a second call with nothing new returns the stored manifest without touching the shards) and **incremental** where
the data allow it. A stored manifest whose ``source_hash`` differs from the current config is stale: the stage logs
a warning and rebuilds the directory from scratch.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data_preparation.lib.storage.parquet import estimate_tokens, list_parquet_files, shard_index, write_dict_rows
from data_preparation.lib.schema.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.schema.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress, progress
from data_preparation.lib.storage.manifest import Manifest, library_versions, shard_rows
from data_preparation.lib.sources import (
    Row,
    get_converter,
    get_filter,
    get_loader,
    list_local_files,
    write_synthetic_tokenizer,
)

log = get_logger(__name__)

DEFAULT_SHARD_SIZE = 10_000
WHOLE_SOURCE = 10**9  # `count` passed to a loader when a source is fetched completely (repeat_to_budget)


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
            tokenizer_dir = layout.tokenizer_dir(cfg.tokenizer.name)
            if not (tokenizer_dir / "tokenizer.json").is_file() and not (tokenizer_dir / "tokenizer_config.json").is_file():
                raise FileNotFoundError(
                    f"tokenizer {cfg.tokenizer.name!r} not found at {tokenizer_dir}; run the tokenizer stage first"
                )
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return min(estimate_tokens(text), self.cap)
        return min(len(self._tokenizer.encode(text, add_special_tokens=False)), self.cap)

    def count_many(self, texts: list[str]) -> list[int]:
        if self._tokenizer is None:
            return [min(estimate_tokens(t), self.cap) for t in texts]
        encoded = self._tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [min(len(ids), self.cap) for ids in encoded]


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
        if index is not None and index >= start_shard:
            manifest.add_shard(path.name, shard_rows(path), tokens.get(path.name) if tokens else None)


def shard_list(manifest: Manifest) -> list[list[Any]]:
    """``[[name, rows], ...]`` — the JSON-friendly identity of a manifest's shards (stored as ``extra["input_shards"]``)."""
    return [[s.name, s.rows] for s in manifest.shards]


def require_manifest(directory: Path, source_hash: str, stage: str, what: str) -> Manifest:
    manifest = current_manifest(directory, source_hash, stage)
    if manifest is None:
        raise FileNotFoundError(f"{what}: no current {stage} manifest in {directory}; run the {stage} stage first")
    return manifest


def text_row(source: SourceConfig, row: Row, name: str) -> Row:
    """Apply a pretrain/holdout source's converter (if any) and check that ``text_field`` is present."""
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

    ``manifest.rows_fetched`` is the loader offset reached (source rows consumed); for pretrain/holdout sources
    every row is kept (converter applied, ``text_field`` guaranteed), for instruct sources the converter and filter
    run at download time and only standardized ``{instruction, input, output}`` rows are stored — malformed rows
    (converter raises ``ValueError``) are skipped and counted in ``extra["skipped_malformed"]``; ``check_limit``
    bounds the number of source rows inspected in total. A loader that yields fewer rows than requested sets
    ``extra["exhausted"]``; ``repeat_to_budget`` sources are fetched whole once (repetition happens in ``process``).
    """
    source = cfg.sources[name]
    source_hash = cfg.source_hash(name)
    out = layout.source_dir(name, "raw")
    manifest = current_manifest(out, source_hash, "raw") or new_manifest(cfg, name, source_hash, "raw")
    if manifest.extra.get("exhausted"):
        log.info("%s: source exhausted after %d rows, nothing more to fetch", name, manifest.rows_fetched)
        return manifest
    if source.repeat_to_budget:
        wanted = WHOLE_SOURCE if manifest.rows_fetched == 0 else 0
    else:
        wanted = rows_needed - manifest.rows()
    if wanted <= 0:
        return manifest
    max_consume = None if source.check_limit is None else source.check_limit - manifest.rows_fetched
    if max_consume is not None and max_consume <= 0:
        manifest.extra["exhausted"] = True
        manifest.save(out)
        return manifest

    log.info("%s: fetching %s rows from offset %d -> %s", name, "all" if wanted == WHOLE_SOURCE else wanted, manifest.rows_fetched, out)
    stats = {"consumed": 0, "kept": 0, "skipped_malformed": 0, "exhausted": False}
    start_shard = len(manifest.shards)
    with progress(total=None if wanted == WHOLE_SOURCE else wanted, desc=f"{name}: download", unit="row") as bar:
        rows = _fetch_rows(source, name, manifest.rows_fetched, wanted, max_consume, hf_token, stats, layout, bar)
        write_dict_rows(rows, out, shard_size, start_shard=start_shard)
    record_new_shards(manifest, out, start_shard)
    manifest.rows_fetched += stats["consumed"]
    manifest.extra["skipped_malformed"] = manifest.extra.get("skipped_malformed", 0) + stats["skipped_malformed"]
    if stats["exhausted"]:
        manifest.extra["exhausted"] = True
    manifest.save(out)
    log.info("%s: kept %d of %d fetched rows (%d rows on disk)", name, stats["kept"], stats["consumed"], manifest.rows())
    return manifest


def _fetch_rows(
    source: SourceConfig,
    name: str,
    offset: int,
    wanted: int,
    max_consume: int | None,
    hf_token: str | None,
    stats: dict[str, Any],
    layout: DatasetLayout,
    bar: Progress,
) -> Iterator[Row]:
    """Rows to store for one download increment; keeps calling the loader until ``wanted`` rows are kept, the
    source is exhausted or ``max_consume`` source rows were inspected (instruct filters may drop rows, so one loader
    call may not be enough). ``bar`` tracks kept rows (postfix: source rows consumed, current repo file)."""
    loader = get_loader(source.loader)
    converter = get_converter(source) if source.kind == "instruct" else None
    row_filter = get_filter(source.filter) if source.filter is not None else None
    if source.kind == "instruct" and converter is None and source.loader != "synthetic":
        raise ValueError(f"{name}: instruct source needs `fields` or `converter`")
    if source.repeat_to_budget and source.loader == "synthetic":
        raise ValueError(f"{name}: repeat_to_budget needs a finite source, the synthetic loader is unbounded")
    postfix: dict[str, Any] = {"consumed": 0}

    def on_file(file: str) -> None:
        postfix["file"] = file.rsplit("/", 1)[-1]
        bar.set_postfix(postfix, refresh=False)

    while stats["kept"] < wanted:
        count = wanted - stats["kept"]
        if max_consume is not None:
            count = min(count, max_consume - stats["consumed"])
            if count <= 0:
                stats["exhausted"] = True
                return
        yielded = 0
        rows = loader(
            source, offset + stats["consumed"], count, token=hf_token, index_dir=layout.hub_index_dir(), on_file=on_file
        )
        for raw in rows:
            yielded += 1
            stats["consumed"] += 1
            postfix["consumed"] = stats["consumed"]
            if stats["consumed"] % 100 == 0:
                bar.set_postfix(postfix, refresh=False)
            if source.kind != "instruct":
                yield text_row(source, raw, name)
                stats["kept"] += 1
                bar.update(1)
                continue
            if row_filter is not None and not row_filter(raw):
                continue
            try:
                row = converter(raw) if converter is not None else dict(raw)
            except ValueError as err:
                stats["skipped_malformed"] += 1
                log.debug("%s: skipping malformed row: %s", name, err)
                continue
            yield {"instruction": row["instruction"], "input": row.get("input", ""), "output": row["output"]}
            stats["kept"] += 1
            bar.update(1)
        if yielded < count:
            stats["exhausted"] = True
            return


# --- holdout -----------------------------------------------------------------------------------------------------------


def holdout(
    cfg: DatasetConfig, name: str, layout: DatasetLayout, *, shard_size: int = DEFAULT_SHARD_SIZE
) -> Manifest:
    """Write the held-out validation rows of a ``holdout`` source: ``source.rows`` rows, shuffled with
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
    source = cfg.sources[name]
    if source.kind != "holdout" or source.rows is None:
        raise ValueError(f"{name}: holdout() needs a source of kind holdout with rows > 0")
    source_hash = cfg.source_hash(name)
    out = layout.holdout_dir(name)
    existing = current_manifest(out, source_hash, "holdout")
    if existing is not None:
        return existing

    offset = 0
    if source.loader == "local":
        total = _local_row_count(Path(str(source.path)))
        offset = max(total - source.rows, 0)
    log.info("%s: holding out %d rows from offset %d -> %s", name, source.rows, offset, out)
    loader = get_loader(source.loader)
    with progress(total=source.rows, desc=f"{name}: holdout", unit="row", leave=False) as bar:
        rows = [
            text_row(source, r, name)
            for r in bar_rows(bar, loader(source, offset, source.rows, index_dir=layout.hub_index_dir()))
        ]
    if len(rows) < source.rows:
        log.warning("%s: only %d of %d requested holdout rows available", name, len(rows), source.rows)
    random.Random(source.seed).shuffle(rows)
    counter = TokenCounter(cfg, layout)
    texts = [str(r[source.text_field]) for r in rows]
    tokens = counter.count_many(texts)
    out_rows = ({"text": t, "source": name, "tokens": n} for t, n in zip(texts, tokens))
    write_dict_rows(out_rows, out, shard_size, start_shard=0)
    manifest = new_manifest(cfg, name, source_hash, "holdout", tokens=True)
    manifest.rows_fetched = offset + len(rows)
    record_new_shards(manifest, out, 0, tokens=_tokens_per_shard(out, tokens))
    manifest.extra = {"offset": offset, "requested_rows": source.rows, "seed": source.seed}
    manifest.save(out)
    return manifest


def bar_rows(bar: Progress, rows: Iterator[Row]) -> Iterator[Row]:
    """Pass ``rows`` through, advancing ``bar`` by one per row."""
    for row in rows:
        bar.update(1)
        yield row


def _local_row_count(directory: Path) -> int:
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
