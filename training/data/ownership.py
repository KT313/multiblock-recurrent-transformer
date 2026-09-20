# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Rank-zero dataset ownership and explicit coordinated entry/completion boundaries (no torch imports).

Success exits synchronize only after each caller has closed its readers. Exceptional exits do not add a blind
barrier: a failed rank may never reach it. Fatal rank loss therefore requires launcher teardown of all peers;
rank zero's local flock alone is not a crash-proof distributed lease.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Protocol, TypeVar

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import DatasetLease, TRAIN_LOCK_NAME, dataset_lock, run_lock

T = TypeVar("T")


class OwnershipBackend(Protocol):
    is_main: bool
    world_size: int

    def all_gather_object(self, obj: T) -> list[T]: ...
    def barrier(self) -> None: ...


def _exchange_error(backend: OwnershipBackend | None, error: Exception | None, phase: str) -> None:
    status: tuple[str, str] | None = None
    if error is not None:
        status = ("cancelled", str(error)) if isinstance(error, BuildAborted) else ("error", f"{type(error).__name__}: {error}")
    results = [status] if backend is None else backend.all_gather_object(status)
    if error is not None:
        raise error  # retain the originating exception and its cause
    for failure in results:
        if failure is not None:
            kind, message = failure
            if kind == "cancelled":
                raise BuildAborted(f"{phase} cancelled on another rank: {message}")
            raise RuntimeError(f"{phase} failed on another rank ({kind}): {message}")


@contextmanager
def main_rank_phase(backend: OwnershipBackend | None, phase: str) -> Iterator[None]:
    """Every rank enters; callers select who does work. Exchange ordinary errors and cooperative cancellation.

    KeyboardInterrupt/SystemExit bypass the exchange so a forced abort cannot start another blocking collective.
    """

    error: Exception | None = None
    try:
        yield
    except Exception as caught:
        error = caught
    _exchange_error(backend, error, phase)


@contextmanager
def _main_rank_lock(
    lock: AbstractContextManager[T], backend: OwnershipBackend | None, *, complete: bool = False
) -> Iterator[T | None]:
    with ExitStack() as stack:
        owned = None
        with main_rank_phase(backend, "lock acquisition"):
            if backend is None or backend.is_main:
                owned = stack.enter_context(lock)
        yield owned
        # Runs on normal/coordinated return only, after the caller's reader cleanup, before rank zero unlocks.
        if complete and backend is not None:
            backend.barrier()


@contextmanager
def dataset_access(
    root: Path, backend: OwnershipBackend | None = None, *, lease: DatasetLease | None = None, shared: bool = False,
) -> Iterator[DatasetLease | None]:
    """Every rank participates; only rank zero acquires/borrows. Hold through all readers' cleanup."""

    with _main_rank_lock(dataset_lock(root, "training", lease=lease, shared=shared), backend, complete=True) as owned:
        yield owned


@contextmanager
def training_dataset_access(
    root: Path, run_directory: Path, backend: OwnershipBackend, *, shared: bool = False,
) -> Iterator[DatasetLease | None]:
    """Acquire output lock first, dataset lock second. Both refusals reach peers before dataset I/O."""

    with (
        _main_rank_lock(run_lock(run_directory / TRAIN_LOCK_NAME, "training"), backend),
        dataset_access(root, backend, shared=shared) as owned,
    ):
        yield owned
