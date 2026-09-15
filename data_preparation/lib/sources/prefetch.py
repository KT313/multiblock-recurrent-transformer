# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded byte prefetch for seekable remote files; never advances dataset row state."""

from __future__ import annotations

import io
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, BinaryIO


class PrefetchReader(io.RawIOBase, BinaryIO):
    """Own a remote handle and read one block ahead on one dedicated thread.

    Only the worker touches the handle until shutdown. The consumer keeps a current block and at most one
    pending block, each bounded by block_size, in addition to caller-owned results and underlying IO buffers.
    Seeks outside these blocks discard read-ahead without changing any row counters. Close waits for the
    current network request (subject to the underlying HTTP timeout), then closes the handle.
    """

    def __init__(self, inner: BinaryIO, size: int, block_size: int) -> None:
        super().__init__()
        if size < 0 or block_size <= 0:
            super().close()
            raise ValueError("prefetch requires a nonnegative file size and a positive block size")
        self._inner = inner
        self._size = size
        self._block_size = block_size
        self._position = 0
        self._buffer = b""
        self._buffer_start = 0
        self._pending: tuple[int, Future[bytes]] | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hub-prefetch")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_CUR:
            offset += self._position
        elif whence == io.SEEK_END:
            offset += self._size
        elif whence != io.SEEK_SET:
            raise ValueError(f"invalid whence: {whence}")
        if offset < 0:
            raise ValueError("negative seek position")
        self._position = offset
        return offset

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        remaining = max(0, self._size - self._position)
        output = bytearray(remaining if size < 0 else min(size, remaining))
        self.readinto(output)
        return bytes(output)

    def readinto(self, buffer: Any) -> int:
        self._checkClosed()
        target = memoryview(buffer).cast("B")
        count = min(len(target), max(0, self._size - self._position))
        written = 0
        while written < count:
            self._load_current_block()
            start = self._position - self._buffer_start
            take = min(count - written, len(self._buffer) - start)
            target[written : written + take] = memoryview(self._buffer)[start : start + take]
            self._position += take
            written += take
        return written

    def _read_block(self, offset: int) -> bytes:
        self._inner.seek(offset)
        wanted = min(self._block_size, self._size - offset)
        data = self._inner.read(wanted)
        if len(data) != wanted:
            raise OSError(f"remote file returned {len(data)} bytes at offset {offset}; expected {wanted}")
        return data

    def _load_current_block(self) -> None:
        if self._buffer_start <= self._position < self._buffer_start + len(self._buffer):
            return

        # Settle the sole pending request before seeking the shared handle elsewhere.
        if self._pending is not None:
            offset, future = self._pending
            if not offset <= self._position < min(offset + self._block_size, self._size):
                self._pending = None
                if not future.cancel():
                    future.result()  # propagate network failures even when a seek discards their bytes
        if self._pending is None:
            self._pending = (self._position, self._pool.submit(self._read_block, self._position))
        offset, future = self._pending
        self._pending = None
        self._buffer = b""  # release the previous block before collecting its replacement
        self._buffer_start, self._buffer = offset, future.result()

        # Fetch the next block while the caller decodes or processes this one.
        next_offset = offset + len(self._buffer)
        if next_offset < self._size:
            self._pending = (next_offset, self._pool.submit(self._read_block, next_offset))

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._pool.shutdown(wait=True, cancel_futures=True)
            pending, self._pending = self._pending, None
            self._buffer = b""
            if pending is not None and not pending[1].cancelled():
                pending[1].result()
        finally:
            try:
                self._inner.close()
            finally:
                super().close()

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None or isinstance(exc, GeneratorExit):
                raise
            exc.add_note(f"Remote prefetch cleanup also failed: {cleanup_error!r}")
