# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Opt-in fatal policy for launcher-owned workers; library calls keep normal unwinding."""

import os
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NoReturn

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import RunLocked

FatalHandler = Callable[[Exception], NoReturn]


def handle_fatal_error(handler: FatalHandler | None, error: BaseException | None) -> None:
    """Expected cancellation/refusal and process-control exceptions retain their existing behavior."""
    if handler is not None and isinstance(error, Exception) and not isinstance(error, (BuildAborted, RunLocked)):
        handler(error)


@contextmanager
def fatal_errors(handler: FatalHandler | None) -> Iterator[None]:
    """Place inside a cleanup-owning context so a fatal callback runs before that context unwinds."""
    try:
        yield
    except Exception as error:
        handle_fatal_error(handler, error)
        raise


def exit_failed_worker(error: Exception) -> NoReturn:
    """Write directly to stderr, then exit without finally/atexit, CUDA calls or logger cleanup.

    Only the torchrun CLI opts in. The launcher owns termination of the other ranks; no distributed
    save or error vote is safe here. Unbuffered writes bypass dashboard/logging locks and preserve causes.
    An unusable stderr must not prevent exit. This cannot solve a native hang that raises no exception.
    """
    try:
        message = f"[rank {os.environ.get('RANK', '?')}] training failed; exiting worker immediately.\n"
        message += ''.join(traceback.format_exception(error))
        pending = memoryview(message.encode('utf-8', errors='backslashreplace'))
        while pending:
            written = os.write(2, pending)
            if written == 0:
                break
            pending = pending[written:]
    finally:
        os._exit(1)
