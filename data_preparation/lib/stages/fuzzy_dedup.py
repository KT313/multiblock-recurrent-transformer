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

# per-process signature parameters (set by the pool initializer, or directly for num_workers <= 1)
_PARAMS: dict[str, int] = {}
_KWARGS: dict[str, Any] = {}


def _import_datasketch() -> tuple[Any, Any]:
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


def _init_signatures(num_perm: int, ngram: int) -> None:
    _PARAMS.update({"num_perm": num_perm, "ngram": ngram})
    _KWARGS.clear()
    _KWARGS.update(_minhash_kwargs(num_perm))


def _signature(text: str) -> Signature:
    """MinHash hash values of the word n-grams of ``text`` (plain numpy array, cheap to pickle)."""
    MinHash, _ = _import_datasketch()
    m = MinHash(**_KWARGS)
    for ngram in get_ngrams(text, n=_PARAMS["ngram"]):
        m.update(ngram.encode("utf-8"))
    return np.asarray(m.hashvalues, dtype=np.uint64)


def _signatures(texts: list[str]) -> list[Signature]:
    return [_signature(text) for text in texts]


def _chunks(rows: Iterator[Row], size: int) -> Iterator[list[Row]]:
    chunk: list[Row] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _signature_stream(
    rows: Iterator[Row], dedup: DedupConfig, num_workers: int, chunk_size: int
) -> Iterator[tuple[Row, Signature]]:
    if num_workers <= 1:
        _init_signatures(dedup.num_perm, dedup.ngram)
        for row in rows:
            yield row, _signature(row["text"])
        return
    # Bounded in-order pipeline: at most ``2 * num_workers`` chunks are read ahead of the consumer, so the input
    # keeps streaming however slow the LSH side is (``pool.imap`` would read the whole input into its task queue).
    inflight: deque[tuple[list[Row], AsyncResult[list[Signature]]]] = deque()
    with multiprocessing.Pool(
        num_workers, initializer=_init_signatures, initargs=(dedup.num_perm, dedup.ngram)
    ) as pool:
        for chunk in _chunks(rows, chunk_size):
            inflight.append((chunk, pool.apply_async(_signatures, ([row["text"] for row in chunk],))))
            if len(inflight) >= 2 * num_workers:
                done, result = inflight.popleft()
                yield from zip(done, result.get())
        while inflight:
            done, result = inflight.popleft()
            yield from zip(done, result.get())


def fuzzy_dedup(
    rows: Iterator[Row], dedup: DedupConfig, stats: dict[str, Any], num_workers: int = 1, chunk_size: int = CHUNK_SIZE
) -> Iterator[Row]:
    """Yield the rows whose MinHash signature has no near-duplicate (Jaccard >= ``dedup.threshold``) among the rows
    yielded before; ``stats`` gets ``threshold``, ``num_perm``, ``near_duplicates_removed``, ``near_duplicate_rate``
    and ``seconds``. See the module docstring for the memory footprint."""
    MinHash, MinHashLSH = _import_datasketch()
    stats.update({"threshold": dedup.threshold, "num_perm": dedup.num_perm, "near_duplicates_removed": 0})
    lsh = MinHashLSH(threshold=dedup.threshold, num_perm=dedup.num_perm)
    kwargs = _minhash_kwargs(dedup.num_perm)
    start = time.monotonic()
    seen = 0
    for index, (row, signature) in enumerate(_signature_stream(rows, dedup, num_workers, chunk_size)):
        seen = index + 1
        m = MinHash(hashvalues=signature, **kwargs)
        if lsh.query(m):
            stats["near_duplicates_removed"] += 1
        else:
            lsh.insert(f"doc_{index}", m)
            yield row
        stats["near_duplicate_rate"] = stats["near_duplicates_removed"] / seen
        stats["seconds"] = round(time.monotonic() - start, 3)
