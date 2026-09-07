# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Shared fixtures of the training tests: the console fallback of the dashboard, and the fp32 reference run of
`test_run.py`, built once per session.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from training.testing.golden import ReferenceRun, run_reference


@pytest.fixture(autouse=True)
def console_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Every run that opens its dashboard through `RunLogger.open` gets the console fallback, also under `pytest -s`
    on a terminal; the dashboard tests that want the live display pass `enabled=True` themselves.
    """

    monkeypatch.setenv("TRAINING_DASHBOARD", "0")


@pytest.fixture(scope="session")
def reference_run(session_shared_base: Path, tiny_dataset_dir: Path) -> ReferenceRun:
    """
    The 20-step tiny run in the golden configuration (`run_reference`: fp32, CPU, one thread, deterministic
    algorithms), shared by the golden test, the parity chain and the samples test of `test_run.py`, which used to run
    it once each. Built once per session, under xdist by the first worker that needs it (the others wait on the lock
    and load the record); the record is JSON with the floats as `repr`, so the bit-exact comparisons hold on every
    worker, and the run directory keeps the checkpoints the golden reads its parameter norms from.
    """

    root = session_shared_base / "reference_run"
    record = session_shared_base / "reference_run.json"
    with open(session_shared_base / "reference_run.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not record.exists():
            root.mkdir(exist_ok=True)  # a build that failed halfway leaves the directory; the next attempt reuses it
            record.write_text(run_reference(root, tiny_dataset_dir).to_json())
    return ReferenceRun.from_json(record.read_text())
