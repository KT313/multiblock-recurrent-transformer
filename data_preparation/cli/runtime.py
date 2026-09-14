# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Signal handling and failure reporting for the dataset preparation CLI."""

from __future__ import annotations

import logging
import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import RunLocked
from data_preparation.lib.build.repair import ConfirmationRequired, RepairError

EXIT_CONFIRMATION_REQUIRED = 2
EXIT_ALREADY_RUNNING = 3
EXIT_INTERRUPTED = 130


def interrupt_on_sigterm(signum: int, frame: FrameType | None) -> None:
    """
    kill ends like Ctrl-C: stop at the next shard, exit 130.
    """

    raise KeyboardInterrupt


def configure_interrupt_handling() -> None:
    """Treat SIGTERM like Ctrl-C when signal registration is allowed."""

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, interrupt_on_sigterm)


@contextmanager
def handle_command_errors(command: str, *, log: logging.Logger) -> Iterator[None]:
    """Report command failures and translate them into the CLI's existing exit codes."""

    try:
        yield
    except SystemExit:
        raise
    except (KeyboardInterrupt, BuildAborted):
        log.warning("%s interrupted; everything published so far is kept, rerun to resume", command)
        raise SystemExit(EXIT_INTERRUPTED) from None
    except ConfirmationRequired as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_CONFIRMATION_REQUIRED) from None
    except RunLocked as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_ALREADY_RUNNING) from None
    except (ValueError, RepairError) as exc:  # a config or repair problem is a message, not a stack trace
        log.error("%s failed: %s", command, exc)
        raise SystemExit(1) from None
    except Exception:
        log.exception("%s failed", command)
        raise SystemExit(1) from None
