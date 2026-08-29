# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Streaming MinHash + LSH near-duplicate removal (``dedup.mode: minhash``), first occurrence wins.

Signatures are computed in a ``multiprocessing.Pool`` (``num_workers > 1``) in chunks of ``chunk_size`` rows;
workers return only the ``uint64[num_perm]`` hash values of each document (nothing but numpy arrays is pickled),
the main process rebuilds the ``MinHash`` from that array and queries/inserts the single ``MinHashLSH`` in input
order. Rows stream in and out; at most ``2 * num_workers`` chunks are in flight at any time.

Memory: the LSH index holds the signature of every *kept* row, i.e. O(kept rows). Per kept row this is the
``num_perm`` uint64 hash values (8 * num_perm bytes, 2 KB at num_perm=256) plus ``b`` band keys and the ``doc_<i>``
key strings in Python dicts/sets -- in practice roughly 3-5 KB per kept row at num_perm=256, so ~4 GB per million
kept rows. Signatures use datasketch's default ``seed=1`` and are therefore reproducible across runs and workers.
"""

from __future__ import annotations

import multiprocessing
import time
from collections import deque
from collections.abc import Iterator
from multiprocessing.pool import AsyncResult
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data_preparation.lib.schema.dataset_config import DedupConfig
from data_preparation.lib.stages.row_pipeline import get_ngrams

Row = dict[str, Any]
Signature = NDArray[np.uint64]

CHUNK_SIZE = 1024
MINHASH_SEED = 1  # datasketch default; pinned so signatures are stable

# Signature parameters of *this* process: the n-gram size and the MinHash constructor arguments. Set once per process
# by ``_init_worker`` (the pool initializer, or called directly when num_workers <= 1) before ``_signature`` is used.
_NGRAM: int = 0
_MINHASH_KWARGS: dict[str, Any] = {}


def _import_datasketch() -> tuple[Any, Any]:
    """``(MinHash, MinHashLSH)`` -- imported lazily because datasketch is an optional extra."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "dedup.mode=minhash needs the `datasketch` package (`uv sync --all-extras`), or use dedup.mode=exact"
        ) from exc
    return MinHash, MinHashLSH


def _minhash_kwargs(num_perm: int) -> dict[str, Any]:
    """Constructor arguments that make every ``MinHash`` here comparable: pinned seed and, on datasketch >= 2.0,
    the default hashing ``scheme`` (required explicitly when rebuilding from ``hashvalues``)."""
    MinHash, _ = _import_datasketch()
    kwargs: dict[str, Any] = {"num_perm": num_perm, "seed": MINHASH_SEED}
    scheme = getattr(MinHash(num_perm=1, seed=MINHASH_SEED), "scheme", None)
    if scheme is not None:
        kwargs["scheme"] = scheme
    return kwargs


def _init_worker(num_perm: int, ngram: int) -> None:
    """Set the per-process signature parameters (pool initializer)."""
    global _NGRAM
    _NGRAM = ngram
    _MINHASH_KWARGS.clear()
    _MINHASH_KWARGS.update(_minhash_kwargs(num_perm))


def _signature(text: str) -> Signature:
    """MinHash hash values of the word n-grams of ``text`` (plain numpy array, cheap to pickle)."""
    MinHash, _ = _import_datasketch()
    minhash = MinHash(**_MINHASH_KWARGS)
    for ngram in get_ngrams(text, n=_NGRAM):
        minhash.update(ngram.encode("utf-8"))
    return np.asarray(minhash.hashvalues, dtype=np.uint64)


def _signatures(texts: list[str]) -> list[Signature]:
    """Worker task: the signatures of one chunk of texts."""
    return [_signature(text) for text in texts]


def _chunks(rows: Iterator[Row], size: int) -> Iterator[list[Row]]:
    """Consecutive lists of at most ``size`` rows."""
    chunk: list[Row] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _signatures_in_process(rows: Iterator[Row], dedup: DedupConfig) -> Iterator[tuple[Row, Signature]]:
    """``(row, signature)`` pairs computed in this process."""
    _init_worker(dedup.num_perm, dedup.ngram)
    for row in rows:
        yield row, _signature(row["text"])


def _signatures_in_pool(
    rows: Iterator[Row], dedup: DedupConfig, num_workers: int, chunk_size: int
) -> Iterator[tuple[Row, Signature]]:
    """``(row, signature)`` pairs in input order, signatures computed by a worker pool chunk by chunk.

    Bounded in-order pipeline: at most ``2 * num_workers`` chunks are read ahead of the consumer, so the input keeps
    streaming however slow the LSH side is (``pool.imap`` would read the whole input into its task queue).
    """
    max_inflight = 2 * num_workers
    inflight: deque[tuple[list[Row], AsyncResult[list[Signature]]]] = deque()

    def oldest_finished() -> Iterator[tuple[Row, Signature]]:
        chunk, pending = inflight.popleft()
        return zip(chunk, pending.get())

    with multiprocessing.Pool(num_workers, initializer=_init_worker, initargs=(dedup.num_perm, dedup.ngram)) as pool:
        for chunk in _chunks(rows, chunk_size):
            texts = [row["text"] for row in chunk]
            inflight.append((chunk, pool.apply_async(_signatures, (texts,))))
            if len(inflight) >= max_inflight:
                yield from oldest_finished()
        while inflight:
            yield from oldest_finished()


def fuzzy_dedup(
    rows: Iterator[Row], dedup: DedupConfig, stats: dict[str, Any], num_workers: int = 1, chunk_size: int = CHUNK_SIZE
) -> Iterator[Row]:
    """Yield the rows whose MinHash signature has no near-duplicate (Jaccard >= ``dedup.threshold``) among the rows
    yielded before; ``stats`` gets ``threshold``, ``num_perm``, ``near_duplicates_removed``, ``near_duplicate_rate``
    and ``seconds``. See the module docstring for the memory footprint."""
    MinHash, MinHashLSH = _import_datasketch()
    stats.update({"threshold": dedup.threshold, "num_perm": dedup.num_perm, "near_duplicates_removed": 0})
    lsh = MinHashLSH(threshold=dedup.threshold, num_perm=dedup.num_perm)
    minhash_kwargs = _minhash_kwargs(dedup.num_perm)

    if num_workers <= 1:
        signatures = _signatures_in_process(rows, dedup)
    else:
        signatures = _signatures_in_pool(rows, dedup, num_workers, chunk_size)

    start = time.monotonic()
    for index, (row, signature) in enumerate(signatures):
        minhash = MinHash(hashvalues=signature, **minhash_kwargs)
        if lsh.query(minhash):
            stats["near_duplicates_removed"] += 1
        else:
            lsh.insert(f"doc_{index}", minhash)
            yield row
        rows_seen = index + 1
        stats["near_duplicate_rate"] = stats["near_duplicates_removed"] / rows_seen
        stats["seconds"] = round(time.monotonic() - start, 3)
