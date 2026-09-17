# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tokenizing download batches in separate processes: the Rust tokenizer's own thread pool stops scaling long before
a big machine is busy.

Measured on a 64-core node with the llama-32k tokenizer (2026-09-17, 2048-row batches of github_code rows):
8 threads tokenize 29 MB/s, 16 threads 48 MB/s, 32 threads 70 MB/s, 50 threads 76 MB/s, while three processes
with 16 threads each reach 112 MB/s together. One `encode_batch` call is a fork-join over the rows of the batch
(one row is one indivisible work item), so the join waits for the slowest split and most of a large pool idles;
separate processes each have their own join and add up linearly.

A :class:`TokenizerPool` runs ceil(threads / :data:`THREADS_PER_PROCESS`) spawn processes with balanced thread
counts (:func:`plan_processes`: 20 threads become 7 + 7 + 6, not 8 + 8 + 4, because batches are stored in
submission order and a slow process would hold up everything queued behind it). Every process loads the saved
tokenizer once per tokenizer directory (`tokenizer_loader.SavedTokenizer`) and serves the two pure functions of
the download's token step: :func:`truncation.truncate_many` for pretrain texts and `count_batch` for instruct
texts. Results are futures; the token worker (download_workers.py) keeps `processes + 1` batches in flight and
stores them in submission order, so rows, per-row progress, shard boundaries and counters are exactly those of the
in-process path. Chat-message rows are never sent here (they are fitted per row in the worker thread).

Spawn, not fork: the pool is created in a process that already runs threads and a Rust thread pool. A killed
worker (OOM killer, a segfault) breaks the executor and surfaces as a named RuntimeError through the future.
Children report into the `--debug` overview like the cleaning-pass workers (download_debug.py).
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import TypeVar

from data_preparation.lib.download_debug import WorkerDebugOptions, initialize_worker_debug, worker_debug_options

THREADS_PER_PROCESS = 8  # Rust threads per tokenizer process; also the in-process pool size (cli/commands.py)

T = TypeVar("T")
_TOKENIZERS: dict[str, object] = {}  # worker-global: tokenizer directory -> SavedTokenizer


def plan_processes(threads: int) -> list[int]:
    """
    Threads per process for threads tokenizer threads in total: as few processes as :data:`THREADS_PER_PROCESS`
    allows, the threads spread evenly (the counts differ by at most one). 8 -> [8], 20 -> [7, 7, 6], 50 -> [8, 7, 7, 7, 7, 7, 7].
    """

    if threads < 1:
        raise ValueError(f"tokenizer threads must be >= 1, got {threads}")
    count = -(-threads // THREADS_PER_PROCESS)
    base, extra = divmod(threads, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


class TokenizerPool:
    """
    The tokenizer processes of one preparation run (a with block); see the module docstring. :attr:`processes`
    is how many batches the token worker keeps in flight to keep them busy.
    """

    def __init__(self, threads: int) -> None:
        self.plan = plan_processes(threads)
        self._pool: ProcessPoolExecutor | None = None

    @property
    def processes(self) -> int:
        return len(self.plan)

    def __enter__(self) -> TokenizerPool:
        context = multiprocessing.get_context("spawn")
        assigned = context.Value("i", 0)  # the next process takes the next entry of the plan
        self._pool = ProcessPoolExecutor(
            max_workers=self.processes, mp_context=context,
            initializer=_init_tokenizer_process, initargs=(tuple(self.plan), assigned, worker_debug_options()),
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)  # a run that stopped early does not wait for queued batches
            self._pool = None

    def truncate(self, tokenizer_dir: Path, texts: list[str], max_tokens: int) -> Future[list[tuple[str, int]]]:
        """
        `truncation.truncate_many(texts, max_tokens, tokenizer)` with the saved tokenizer of tokenizer_dir, in a
        pool process.
        """

        return _wrapped(self._executor.submit(_truncate_in_process, str(tokenizer_dir), texts, max_tokens))

    def count(self, tokenizer_dir: Path, texts: list[str]) -> Future[list[int]]:
        """
        `SavedTokenizer.count_batch(texts)` with the saved tokenizer of tokenizer_dir, in a pool process.
        """

        return _wrapped(self._executor.submit(_count_in_process, str(tokenizer_dir), texts))

    @property
    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            raise RuntimeError("the tokenizer pool is not open (use it inside its with block)")
        return self._pool


def _wrapped(future: Future[T]) -> Future[T]:
    """
    future with a lost worker renamed: BrokenProcessPool says nothing about what died.
    """

    result: Future[T] = Future()

    def forward(done: Future[T]) -> None:
        error = done.exception()
        if isinstance(error, BrokenProcessPool):
            result.set_exception(RuntimeError(f"a tokenizer process died (killed by the OOM killer, or crashed): {error}"))
        elif error is not None:
            result.set_exception(error)
        else:
            result.set_result(done.result())

    future.add_done_callback(forward)
    return result


def _init_tokenizer_process(plan: tuple[int, ...], assigned: Synchronized[int], debug: WorkerDebugOptions | None) -> None:
    """
    Process initializer: take the next thread count of the plan (Rayon reads RAYON_NUM_THREADS when its pool is
    first used, which is after this), and join the debug overview.
    """

    with assigned.get_lock():
        index = assigned.value
        assigned.value += 1
    os.environ["RAYON_NUM_THREADS"] = str(plan[index % len(plan)])
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    initialize_worker_debug(debug, "tokenizer")


def _tokenizer(tokenizer_dir: str) -> object:
    tokenizer = _TOKENIZERS.get(tokenizer_dir)
    if tokenizer is None:
        from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

        tokenizer = _TOKENIZERS[tokenizer_dir] = SavedTokenizer(tokenizer_dir)
    return tokenizer


def _truncate_in_process(tokenizer_dir: str, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
    from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
    from data_preparation.lib.stages.truncation import truncate_many

    tokenizer = _tokenizer(tokenizer_dir)
    assert isinstance(tokenizer, SavedTokenizer)
    return truncate_many(texts, max_tokens, tokenizer)


def _count_in_process(tokenizer_dir: str, texts: list[str]) -> list[int]:
    from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

    tokenizer = _tokenizer(tokenizer_dir)
    assert isinstance(tokenizer, SavedTokenizer)
    return tokenizer.count_batch(texts)
