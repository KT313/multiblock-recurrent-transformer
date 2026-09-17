# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Private worker staging, publication failure and abrupt parent-exit regressions."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_preparation.lib.build.lock import dataset_lock
from data_preparation.lib.download_debug import WorkerDebugOptions
from data_preparation.lib.stages import build_workers
from data_preparation.lib.stages.build_workers import ShardSource, ShardWorkers
from data_preparation.lib.storage.atomic import write_atomically
from data_preparation.lib.storage.parquet import publish_shard


def _source() -> ShardSource:
    return ShardSource("s", "text", 1, False, True, 4096, 2)


def _raw(path: Path, text: str) -> None:
    pq.write_table(pa.table({"text": [text], "tokens": [3]}), path)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 20
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for {path}")
        time.sleep(0.02)


def _init_delayed_writer(debug: WorkerDebugOptions | None) -> None:
    """Block the actual worker file write until the test has replaced its parent."""
    original = publish_shard
    root = Path(os.environ["MBRT_TEST_SHARD_STAGING"])

    def publish(table: pa.Table, path: Path) -> Path:
        (root / "ready").touch()
        _wait_for(root / "release")
        result = original(table, path)
        with write_atomically(root / "finished") as temporary:
            temporary.write_text(str(result))
        os._exit(0)  # The parent is already gone; leave no orphan executor waiting for another task.

    build_workers.publish_shard = publish  # type: ignore[attr-defined]  # replace the worker's imported writer


def _exit_during_worker_write(root: Path) -> None:
    build_workers._init_build_worker = _init_delayed_writer
    with dataset_lock(root / "dataset"), ShardWorkers(1, _source()) as workers:
        workers.prepare(0, root / "old.parquet")
        prepared = workers.prepared(0)
        with ThreadPoolExecutor(max_workers=1) as thread:
            thread.submit(workers.write, 0, list(range(len(prepared.hashes))),
                          root / "dataset" / "processed" / "s", 0, 2, next_raw_path=None)
            _wait_for(root / "ready")
            os._exit(130)  # Same primitive as the preparation CLI's second-interrupt exit.


def test_worker_cannot_publish_after_parent_exit(tmp_path: Path) -> None:
    _raw(tmp_path / "old.parquet", "old parent's rows")
    _raw(tmp_path / "new.parquet", "replacement parent's rows")
    root = tmp_path / "dataset"
    output = root / "processed" / "s"
    env = os.environ | {"MBRT_TEST_SHARD_STAGING": str(tmp_path)}
    with (tmp_path / "parent.log").open("w") as log:
        parent = subprocess.Popen(
            [sys.executable, "-m", "data_preparation.lib.stages.test_build_workers", str(tmp_path)],
            env=env, stdout=log, stderr=log, start_new_session=True,
        )
        try:
            assert parent.wait(timeout=30) == 130, (tmp_path / "parent.log").read_text()
            with dataset_lock(root), ShardWorkers(1, _source()) as replacement:
                replacement.prepare(0, tmp_path / "new.parquet")
                replacement.prepared(0)
                replacement.write(0, [0], output, 0, 2, next_raw_path=None)
                manifest = output / "MANIFEST.json"
                manifest.write_text('{"owner": "replacement"}')
                before = {path.name: path.read_bytes() for path in output.iterdir()}
                (tmp_path / "release").touch()
                _wait_for(tmp_path / "finished")
                staged = Path((tmp_path / "finished").read_text())
                assert staged.is_relative_to(root / ".build-work")
                assert staged.parent != replacement._scratch, "each invocation owns different staging files"
                assert staged.is_file(), "old worker finished writing only its private file"
                assert {path.name: path.read_bytes() for path in output.iterdir()} == before
        except BaseException:
            # Kill only the isolated process group this test created if synchronization failed.
            with suppress(ProcessLookupError):
                os.killpg(parent.pid, signal.SIGKILL)
            parent.wait(timeout=10)
            raise


def test_failed_parent_publication_preserves_old_shard_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw.parquet"
    _raw(raw, "new rows")
    output = tmp_path / "dataset" / "processed" / "s"
    output.mkdir(parents=True)
    final = output / "data-00000.parquet"
    final.write_bytes(b"previous committed shard")
    original = os.replace

    def fail_final(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == final:
            raise OSError("publication failed")
        original(source, destination)

    monkeypatch.setattr(os, "replace", fail_final)
    with ShardWorkers(1, _source()) as workers:
        workers.prepare(0, raw)
        workers.prepared(0)
        with pytest.raises(OSError, match="publication failed"):
            workers.write(0, [0], output, 0, 2, next_raw_path=None)
        scratch = workers._scratch
        assert scratch is not None and scratch.is_dir()
        assert final.read_bytes() == b"previous committed shard"
        assert list(output.iterdir()) == [final], "failed publication leaves no temporary final-path sibling"
    assert not scratch.exists(), "normal exit joins workers before deleting their private directory"


if __name__ == "__main__":
    _exit_during_worker_write(Path(sys.argv[1]))
