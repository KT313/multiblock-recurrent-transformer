# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The tokenizer processes (tokenizer_pool.py) and the token worker's pipeline over them (download_workers.py): the
same rows, counts, shard boundaries and progress as the in-process path, whatever order the batches finish in.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.dataset_config import DatasetConfig, SourceConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.stages import download as download_module
from data_preparation.lib.stages import download_state, download_workers
from data_preparation.lib.stages.download import TokenCounter, download, prepare_tokenizer
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
from data_preparation.lib.stages.tokenizer_pool import THREADS_PER_PROCESS, TokenizerPool, _wrapped, plan_processes
from data_preparation.lib.stages.truncation import estimate_tokens, truncate_many
from data_preparation.lib.storage.manifest import Manifest

CfgFactory = Callable[..., DatasetConfig]
Prep = Callable[[DatasetConfig], DatasetConfig]
Reader = Callable[[Path], list[dict[str, Any]]]


def test_plan_processes_balances_the_threads_over_as_few_processes_as_possible() -> None:
    assert plan_processes(1) == [1]
    assert plan_processes(THREADS_PER_PROCESS) == [THREADS_PER_PROCESS]
    assert plan_processes(9) == [5, 4]
    assert plan_processes(20) == [7, 7, 6]
    assert plan_processes(50) == [8, 7, 7, 7, 7, 7, 7]
    assert plan_processes(64) == [8] * 8
    with pytest.raises(ValueError, match="tokenizer threads must be >= 1"):
        plan_processes(0)


def test_a_lost_tokenizer_process_is_named() -> None:
    source: Future[list[int]] = Future()
    wrapped = _wrapped(source)
    source.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
    with pytest.raises(RuntimeError, match="a tokenizer process died"):
        wrapped.result()
    plain: Future[list[int]] = Future()
    wrapped_plain = _wrapped(plain)
    plain.set_result([1, 2])
    assert wrapped_plain.result() == [1, 2]


def test_pool_processes_truncate_and_count_like_the_process_itself(cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout) -> None:
    """
    Two real spawn processes (9 threads: 5 + 4) give the results of truncation.truncate_many and count_batch
    with the same saved tokenizer, and the pool is closed after the with block.
    """

    cfg = with_tokenizer(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}))
    tokenizer_dir = layout.tokenizer_dir(cfg.tokenizer.name)
    tokenizer = SavedTokenizer(tokenizer_dir)
    texts = [f"document {i} " + "word " * (i * 7) for i in range(12)]
    with TokenizerPool(9) as pool:
        assert pool.processes == 2 and pool.plan == [5, 4]
        truncated = pool.truncate(tokenizer_dir, texts, 20)
        counted = pool.count(tokenizer_dir, texts)
        assert truncated.result(timeout=60) == truncate_many(texts, 20, tokenizer)
        assert counted.result(timeout=60) == tokenizer.count_batch(texts)
    with pytest.raises(RuntimeError, match="not open"):
        pool.count(tokenizer_dir, texts)


def test_a_download_through_the_pool_stores_what_the_in_process_download_stores(
    cfg_factory: CfgFactory, with_tokenizer: Prep, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader, tmp_path: Path,
) -> None:
    monkeypatch.setattr(download_state, "TOKEN_BATCH", 7)  # several batches in flight per shard
    cfg = with_tokenizer(cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}))
    with TokenizerPool(9) as pool:
        pooled = download(cfg, "p", layout, rows_needed=60, shard_size=25, tokenizer_pool=pool)
    other = DatasetLayout(tmp_path / "other")
    prepare_tokenizer(cfg, other)
    reference = download(cfg, "p", other, rows_needed=60, shard_size=25)
    assert [(s.name, s.rows, s.tokens, s.offset) for s in pooled.shards] == [(s.name, s.rows, s.tokens, s.offset) for s in reference.shards]
    assert (pooled.rows(), pooled.tokens(), pooled.rows_fetched) == (reference.rows(), reference.tokens(), reference.rows_fetched)
    assert read_rows(layout.raw_dir("p")) == read_rows(other.raw_dir("p"))


class _ReversingPool:
    """
    A tokenizer pool stand-in (estimate counts) that finishes every group of `processes` batches in reverse
    order, so the worker's in-order storing is what keeps the rows in order. The results are those of the
    estimate mode's truncation, so the manifest matches a plain estimate-mode download.
    """

    processes = 3

    def __init__(self) -> None:
        self.pending: list[tuple[Future[Any], Any]] = []
        self.lock = threading.Lock()
        self.max_in_flight = 0
        self.calls = 0

    def truncate(self, tokenizer_dir: Path, texts: list[str], max_tokens: int) -> Future[list[tuple[str, int]]]:
        return self._submit(truncate_many(texts, max_tokens, None))

    def count(self, tokenizer_dir: Path, texts: list[str]) -> Future[list[int]]:
        return self._submit([estimate_tokens(text) for text in texts])

    def _submit(self, value: Any) -> Future[Any]:
        future: Future[Any] = Future()
        with self.lock:
            self.calls += 1
            self.pending.append((future, value))
            self.max_in_flight = max(self.max_in_flight, len(self.pending))
            if len(self.pending) >= self.processes:
                self._release()
        return future

    def _release(self) -> None:
        group, self.pending = self.pending[::-1], []

        def resolve() -> None:
            for pending, value in group:
                time.sleep(0.005)
                pending.set_result(value)

        threading.Thread(target=resolve, daemon=True).start()

    def flush(self) -> None:
        with self.lock:
            self._release()


class _EstimatePoolCounter:
    """
    A `TokenCounter` stand-in in estimate mode with a :class:`_ReversingPool`.
    """

    pool: Any = _ReversingPool()

    def __init__(self, config: DatasetConfig, layout: DatasetLayout, pool: Any = None) -> None:
        self.tokenizer_dir = layout.tokenizer_dir(config.tokenizer.name)

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        raise AssertionError("the pool truncates")

    def count_many(self, texts: list[str]) -> list[int]:
        raise AssertionError("the pool counts")


def test_batches_finishing_out_of_order_are_stored_in_submission_order(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch, read_rows: Reader, tmp_path: Path,
) -> None:
    monkeypatch.setattr(download_state, "TOKEN_BATCH", 4)
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}, token_count="estimate")
    pool = _ReversingPool()
    monkeypatch.setattr(_EstimatePoolCounter, "pool", pool)
    monkeypatch.setattr(download_module, "TokenCounter", _EstimatePoolCounter)
    # the last, short group of batches would wait forever: release it once the fetch thread is done submitting
    original_close = download_workers._TokenWorker.close

    def close(worker: Any) -> None:
        pool.flush()
        original_close(worker)

    monkeypatch.setattr(download_workers._TokenWorker, "close", close)
    pooled = download(cfg, "p", layout, rows_needed=50, shard_size=15)
    assert pool.calls == 13 and 2 <= pool.max_in_flight <= pool.processes + 1  # 50 rows in batches of 4, at most processes + 1 in flight

    monkeypatch.setattr(download_module, "TokenCounter", TokenCounter)
    other = DatasetLayout(tmp_path / "other")
    reference = download(cfg, "p", other, rows_needed=50, shard_size=15)
    assert [(s.name, s.rows, s.tokens, s.offset) for s in pooled.shards] == [(s.name, s.rows, s.tokens, s.offset) for s in reference.shards]
    assert read_rows(layout.raw_dir("p")) == read_rows(other.raw_dir("p"))


class _FailingPool:
    """
    A pool whose third batch fails (the future raises): the download fails with that error after the shards
    before it were published.
    """

    processes = 2
    calls = 0

    def truncate(self, tokenizer_dir: Path, texts: list[str], max_tokens: int) -> Future[list[tuple[str, int]]]:
        type(self).calls += 1
        future: Future[list[tuple[str, int]]] = Future()
        if type(self).calls == 3:
            future.set_exception(RuntimeError("a tokenizer process died (killed by the OOM killer, or crashed)"))
        else:
            future.set_result(truncate_many(texts, max_tokens, None))
        return future

    def count(self, tokenizer_dir: Path, texts: list[str]) -> Future[list[int]]:
        raise AssertionError("pretrain only")


def test_a_failed_pool_batch_fails_the_download_and_keeps_the_published_shards(
    cfg_factory: CfgFactory, layout: DatasetLayout, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download_state, "TOKEN_BATCH", 5)
    cfg = cfg_factory({"p": SourceConfig(kind="pretrain", loader="synthetic")}, token_count="estimate")
    _FailingPool.calls = 0
    monkeypatch.setattr(_EstimatePoolCounter, "pool", _FailingPool())
    monkeypatch.setattr(download_module, "TokenCounter", _EstimatePoolCounter)
    with pytest.raises(RuntimeError, match="a tokenizer process died"):
        download(cfg, "p", layout, rows_needed=40, shard_size=5)
    manifest = Manifest.load(layout.raw_dir("p"))
    assert manifest is not None and manifest.rows() == manifest.rows_fetched == 10  # the two batches before the failed one
    assert not manifest.exhausted
