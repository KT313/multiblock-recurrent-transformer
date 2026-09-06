# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Streaming MinHash + LSH near-duplicate removal (dedup.mode: minhash), first occurrence wins.

Signatures are computed in a spawn-context multiprocessing.Pool of pass_workers (pass_workers > 1;
spawn because the pool is created from a build worker thread, where a fork could inherit another thread's lock
mid-hold) in chunks of chunk_size rows; workers return only the uint64[num_perm] hash values of each
document (nothing but numpy arrays is pickled), the main process rebuilds the MinHash from that array and
queries/inserts the single MinHashLSH in input order. Rows stream in and out; at most 2 * pass_workers
chunks are in flight at any time.

Memory: the LSH index holds the signature of every *kept* row, i.e. O(kept rows). Per kept row this is the
num_perm uint64 hash values (8 * num_perm bytes, 2 KB at num_perm=256) plus b band keys and the doc_<i>
key strings in Python dicts/sets: in practice roughly 3-5 KB per kept row at num_perm=256, so ~4 GB per million
kept rows. Signatures use datasketch's default seed=1 and are therefore reproducible across runs and workers.
"""

from __future__ import annotations

import multiprocessing
import time
from collections import deque
from collections.abc import Iterator
from functools import partial
from multiprocessing.pool import AsyncResult
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data_preparation.dataset_config import DedupConfig
from data_preparation.lib.iteration import chunks
from data_preparation.lib.stages.row_pipeline import get_ngrams

Row = dict[str, Any]
Signature = NDArray[np.uint64]

CHUNK_SIZE = 1024
MINHASH_SEED = 1  # datasketch default; pinned so signatures are stable

# Signature parameters of a spawn worker: the n-gram size and the MinHash constructor arguments, set once per
# worker by _init_worker (the pool initializer; spawn children start with fresh module globals, its arguments
# are two plain ints). The in-process path (pass_workers <= 1) never touches them: several builds run in threads
# of one process (`lib/build/runner.py`), each with its own dedup settings, so it binds its parameters locally.
_NGRAM: int = 0
_MINHASH_KWARGS: dict[str, Any] = {}


def _import_datasketch() -> tuple[Any, Any]:
    """
    (MinHash, MinHashLSH), imported lazily because datasketch is an optional extra.
    """

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "dedup.mode=minhash needs the `datasketch` package (`uv sync --all-extras`), or use dedup.mode=exact"
        ) from exc
    return MinHash, MinHashLSH


def _minhash_kwargs(num_perm: int) -> dict[str, Any]:
    """
    Constructor arguments that make every MinHash here comparable: pinned seed and, on datasketch >= 2.0,
    the default hashing scheme (required explicitly when rebuilding from hashvalues).
    """

    MinHash, _ = _import_datasketch()
    kwargs: dict[str, Any] = {"num_perm": num_perm, "seed": MINHASH_SEED}
    scheme = getattr(MinHash(num_perm=1, seed=MINHASH_SEED), "scheme", None)
    if scheme is not None:
        kwargs["scheme"] = scheme
    return kwargs


def _init_worker(num_perm: int, ngram: int) -> None:
    """
    Set the signature parameters of a spawn worker (pool initializer).
    """

    global _NGRAM
    _NGRAM = ngram
    _MINHASH_KWARGS.clear()
    _MINHASH_KWARGS.update(_minhash_kwargs(num_perm))


def _signature(text: str, ngram: int, minhash_kwargs: dict[str, Any]) -> Signature:
    """
    MinHash hash values of the word n-grams of text (plain numpy array, cheap to pickle); an empty array for
    a text with fewer than ngram words: such texts have no n-grams, and the empty-set signature would make every
    one of them a near-duplicate of the first.
    """

    MinHash, _ = _import_datasketch()
    ngrams = get_ngrams(text, n=ngram)
    if not ngrams:
        return np.empty(0, dtype=np.uint64)
    minhash = MinHash(**minhash_kwargs)
    for item in ngrams:
        minhash.update(item.encode("utf-8"))
    return np.asarray(minhash.hashvalues, dtype=np.uint64)


def _signatures(texts: list[str]) -> list[Signature]:
    """
    Worker task: the signatures of one chunk of texts, with the worker's parameters of _init_worker.
    """

    return [_signature(text, _NGRAM, _MINHASH_KWARGS) for text in texts]


def _signatures_in_process(rows: Iterator[Row], dedup: DedupConfig) -> Iterator[tuple[Row, Signature]]:
    """
    (row, signature) pairs computed in this process, the parameters bound to this call.
    """

    signature = partial(_signature, ngram=dedup.ngram, minhash_kwargs=_minhash_kwargs(dedup.num_perm))
    for row in rows:
        yield row, signature(row["text"])


def _signatures_in_pool(
    rows: Iterator[Row], dedup: DedupConfig, pass_workers: int, chunk_size: int
) -> Iterator[tuple[Row, Signature]]:
    """
    (row, signature) pairs in input order, signatures computed by a spawn worker pool chunk by chunk.

    Bounded in-order pipeline: at most 2 * pass_workers chunks are read ahead of the consumer, so the input keeps
    streaming however slow the LSH side is (pool.imap would read the whole input into its task queue).
    """

    max_inflight = 2 * pass_workers
    inflight: deque[tuple[list[Row], AsyncResult[list[Signature]]]] = deque()

    def oldest_finished() -> Iterator[tuple[Row, Signature]]:
        chunk, pending = inflight.popleft()
        return zip(chunk, pending.get())

    with multiprocessing.get_context("spawn").Pool(pass_workers, initializer=_init_worker, initargs=(dedup.num_perm, dedup.ngram)) as pool:
        for chunk in chunks(rows, chunk_size):
            texts = [row["text"] for row in chunk]
            inflight.append((chunk, pool.apply_async(_signatures, (texts,))))
            if len(inflight) >= max_inflight:
                yield from oldest_finished()
        while inflight:
            yield from oldest_finished()


def fuzzy_dedup(
    rows: Iterator[Row], dedup: DedupConfig, stats: dict[str, Any], pass_workers: int = 1, chunk_size: int = CHUNK_SIZE
) -> Iterator[Row]:
    """
    Yield the rows whose MinHash signature has no near-duplicate (Jaccard >= dedup.threshold) among the rows
    yielded before; rows too short for a single n-gram pass through untouched (stats["too_short_passed"], they
    are only deduplicated exactly). stats gets threshold, num_perm, near_duplicates_removed,
    near_duplicate_rate and seconds. See the module docstring for the memory footprint.
    """

    MinHash, MinHashLSH = _import_datasketch()
    stats.update({"threshold": dedup.threshold, "num_perm": dedup.num_perm, "near_duplicates_removed": 0, "too_short_passed": 0})
    lsh = MinHashLSH(threshold=dedup.threshold, num_perm=dedup.num_perm)
    minhash_kwargs = _minhash_kwargs(dedup.num_perm)

    if pass_workers <= 1:
        signatures = _signatures_in_process(rows, dedup)
    else:
        signatures = _signatures_in_pool(rows, dedup, pass_workers, chunk_size)

    start = time.monotonic()
    rows_seen = 0  # rows up to and including the last one that went through the LSH
    for index, (row, signature) in enumerate(signatures):
        if signature.size == 0:
            stats["too_short_passed"] += 1
            yield row
            continue
        minhash = MinHash(hashvalues=signature, **minhash_kwargs)
        if lsh.query(minhash):
            stats["near_duplicates_removed"] += 1
        else:
            lsh.insert(f"doc_{index}", minhash)
            yield row
        rows_seen = index + 1
    if rows_seen:
        stats["near_duplicate_rate"] = stats["near_duplicates_removed"] / rows_seen
        stats["seconds"] = round(time.monotonic() - start, 3)
