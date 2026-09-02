# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Shared fixtures of the training tests."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from unittest import mock

import pytest
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter


@pytest.fixture(autouse=True)
def console_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every run that opens its dashboard through `RunLogger.open` gets the console fallback, also under `pytest -s`
    on a terminal; the dashboard tests that want the live display pass `enabled=True` themselves."""
    monkeypatch.setenv("TRAINING_DASHBOARD", "0")


@pytest.fixture(scope="session")
def live_dataloader_iterators() -> Iterator[weakref.WeakSet[_MultiProcessingDataLoaderIter]]:
    """Every multi-process DataLoader iterator handed out during the session that is still alive."""
    live: weakref.WeakSet[_MultiProcessingDataLoaderIter] = weakref.WeakSet()
    plain_iter = DataLoader.__iter__

    def tracked_iter(loader: DataLoader[object]) -> _BaseDataLoaderIter:
        iterator = plain_iter(loader)
        if isinstance(iterator, _MultiProcessingDataLoaderIter):
            live.add(iterator)
        return iterator

    with mock.patch.object(DataLoader, "__iter__", tracked_iter):
        yield live


@pytest.fixture(autouse=True)
def shut_down_abandoned_dataloader_workers(
    live_dataloader_iterators: weakref.WeakSet[_MultiProcessingDataLoaderIter],
) -> Iterator[None]:
    """Shut down the worker processes of the multi-process DataLoader iterators a test left alive.

    A run that ends in a traceback (`pytest.raises` around `train()`) keeps its iterators alive in a reference cycle
    (traceback -> frame -> loaders). Left to the cyclic GC, such an iterator is collected at an arbitrary later
    moment, on whichever thread triggers the collection, and its shutdown then takes 5 s per worker (the worker
    never receives its stop message once the queue objects of the same cycle are finalised, so torch's join times
    out and terminates it); a direct shutdown takes milliseconds. Under xdist that stall hit unrelated tests in the
    same process, so each test shuts its own iterators down here."""
    yield
    for iterator in list(live_dataloader_iterators):
        iterator._shutdown_workers()
