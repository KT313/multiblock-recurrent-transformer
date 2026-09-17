# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Cooperative CLI stop requests and restoration of process signal handlers."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from training.ui.common import KEEP, TRAINING_LOGGER_NAME

log = logging.getLogger(f"{TRAINING_LOGGER_NAME}.train")


class StopRequest:
    """
    The one stop request of a run (a `StopCheck`: calling it answers "stop?"). `train()` polls it after every
    optimizer step, the in-process dataset build between shards.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def __call__(self) -> bool:
        return self._event.is_set()


@contextmanager
def stop_on_interrupt(*, message: str = "stopping after this step, saving a checkpoint") -> Iterator[StopRequest]:
    """
    Install the Ctrl-C / SIGTERM handling of a run and yield its stop request; the previous handlers are put back
    on exit.

    The first signal sets the request, logs it and hands both signals back to their default handlers, so a second
    Ctrl-C raises `KeyboardInterrupt` as usual (and a second SIGTERM kills). Signal handlers can only be installed
    from the main thread; elsewhere the request is yielded unarmed.
    """

    request = StopRequest()
    if threading.current_thread() is not threading.main_thread():
        yield request
        return

    def on_signal(signum: int, frame: FrameType | None) -> None:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        request.request_stop()
        log.warning(
            "%s received: %s (press Ctrl-C again to abort right away)",
            signal.Signals(signum).name, message,
            extra=KEEP,
        )

    previous = {signum: signal.signal(signum, on_signal) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
