# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Offline read-ahead tests, including overlap, bounded fetching and worker ownership."""

from __future__ import annotations

import io
import random
import threading

import pytest

from data_preparation.lib.sources.prefetch import PrefetchReader


def test_prefetch_overlaps_processing_and_stops_after_one_extra_block() -> None:
    fetched_ahead = threading.Event()
    requests: list[tuple[int, int | None]] = []
    workers: set[threading.Thread] = set()

    class Remote(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            workers.add(threading.current_thread())
            requests.append((self.tell(), size))
            result = super().read(size)
            if self.tell() == 16:
                fetched_ahead.set()
            return result

    remote = Remote(bytes(range(100)))
    with PrefetchReader(remote, 100, 8) as reader:
        assert reader.read(1) == b"\x00"
        assert fetched_ahead.wait(5), "the next block must arrive without another consumer read"
        assert requests == [(0, 8), (8, 8)]
        assert reader.tell() == 1  # background fetching must not advance the consumer's position
    assert remote.closed
    assert workers and all(worker is not threading.current_thread() and not worker.is_alive() for worker in workers)


def test_random_reads_seeks_and_readinto_match_a_local_file() -> None:
    data = bytes(range(251)) * 4
    reference = io.BytesIO(data)
    rng = random.Random(42)
    with PrefetchReader(io.BytesIO(data), len(data), 31) as reader:
        for _ in range(200):
            position = rng.randrange(len(data) + 50)
            whence = rng.choice([io.SEEK_SET, io.SEEK_CUR, io.SEEK_END])
            offset = position - (reference.tell() if whence == io.SEEK_CUR else len(data) if whence == io.SEEK_END else 0)
            assert reader.seek(offset, whence) == reference.seek(offset, whence)
            size = rng.randrange(100)
            assert reader.read(size) == reference.read(size)
            target = bytearray(7)
            count = reader.readinto(target)
            expected = reference.read(7)
            assert target[:count] == expected
            assert reader.tell() == reference.tell()
        reader.seek(0)
        assert reader.read() == data
        assert reader.read(1) == b""
        with pytest.raises(ValueError, match="negative"):
            reader.seek(-1)
    with pytest.raises(ValueError):
        reader.read(1)
    reader.close()  # repeated cleanup is safe


def test_prefetch_failure_is_raised_and_worker_closes() -> None:
    failure = OSError("network failed")
    failed = threading.Event()

    class Remote(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 8:
                failed.set()
                raise failure
            return super().read(size)

    remote = Remote(b"x" * 32)
    with pytest.raises(OSError) as caught, PrefetchReader(remote, 32, 8) as reader:
        assert reader.read(8) == b"x" * 8
        assert failed.wait(5)
        reader.read(8)
    assert caught.value is failure
    assert remote.closed


def test_cleanup_preserves_processing_error_and_reports_prefetch_failure() -> None:
    failed = threading.Event()

    class Remote(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if self.tell() == 8:
                failed.set()
                raise OSError("prefetch failed")
            return super().read(size)

    remote = Remote(b"x" * 32)
    with pytest.raises(ValueError, match="processing failed") as caught, PrefetchReader(remote, 32, 8) as reader:
        reader.read(1)
        assert failed.wait(5)
        raise ValueError("processing failed")
    assert "prefetch failed" in " ".join(caught.value.__notes__)
    assert remote.closed


def test_short_remote_read_is_an_error() -> None:
    with pytest.raises(OSError, match="expected 8"), PrefetchReader(io.BytesIO(b"short"), 10, 8) as reader:
        reader.read(8)
