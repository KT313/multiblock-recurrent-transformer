# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Seeded global row shuffle backed by a disposable disk index, not an in-memory row list."""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.log import get_logger

SHUFFLE_POLICY = "sqlite_random128_v1"
SHUFFLE_CACHE_KIB = 32 * 1024
_CHECK_INTERVAL = 1024
log = get_logger(__name__)
Row = dict[str, Any]


@contextmanager
def shuffled_rows(
    rows: Iterator[Row], directory: Path, seed: int, *, should_stop: StopCheck | None = None,
) -> Iterator[Iterator[Row]]:
    """
    Stage processed rows, then stream a global permutation. The caller owns directory and consumes the iterator
    inside the context. Scratch files disappear on success, cancellation or consumer failure; hard crashes leave
    them inside the existing owned build workspace for repair. No scratch database is ever reused.

    Random 128-bit keys are drawn once per surviving row, independent of batch/shard boundaries. The original
    ordinal breaks the vanishingly unlikely key collision without dropping duplicates. JSON preserves the
    processed string/integer columns, including 64-bit hashes, without SQLite integer coercion of their values.
    A rowid table keeps large payloads out of the small shuffle index; an index scan needs no in-memory sort.
    """

    check_stop(should_stop)
    with TemporaryDirectory(prefix="shuffle-", dir=directory) as scratch:
        path = Path(scratch) / "rows.sqlite"
        try:
            with closing(sqlite3.connect(path)) as connection:
                # Private disposable scratch only: a failed transaction is discarded, never recovered/published.
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA mmap_size=0")
                connection.execute(f"PRAGMA cache_size=-{SHUFFLE_CACHE_KIB}")
                connection.execute("PRAGMA cache_spill=ON")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("CREATE TABLE rows (ordinal INTEGER PRIMARY KEY, shuffle_key BLOB NOT NULL, payload TEXT NOT NULL)")
                connection.execute("CREATE INDEX shuffle_order ON rows(shuffle_key, ordinal)")
                rng = random.Random(seed)
                count = 0
                log.info("%s: staging disk-backed shuffle (SQLite cache target %d MiB)", directory, SHUFFLE_CACHE_KIB // 1024)
                for ordinal, row in enumerate(rows):
                    if ordinal % _CHECK_INTERVAL == 0:
                        check_stop(should_stop)
                    connection.execute(
                        "INSERT INTO rows VALUES (?, ?, ?)",
                        (ordinal, rng.getrandbits(128).to_bytes(16, "big"),
                         json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)),
                    )
                    count += 1
                check_stop(should_stop)
                connection.commit()
                log.info("%s: shuffle staged %d rows; writing shuffled output", directory, count)
                cursor = connection.execute("SELECT payload FROM rows INDEXED BY shuffle_order ORDER BY shuffle_key, ordinal")
                with closing(cursor):
                    yield _read_rows(cursor, should_stop)
        except sqlite3.Error as error:
            raise RuntimeError(
                f"Disk-backed shuffle failed in {directory}: {error}. Check free disk space and filesystem access; "
                "the scratch shuffle can be rebuilt from the raw source."
            ) from error


def _read_rows(cursor: sqlite3.Cursor, should_stop: StopCheck | None) -> Iterator[Row]:
    for index, (payload,) in enumerate(cursor):
        if index % _CHECK_INTERVAL == 0:
            check_stop(should_stop)
        yield cast(Row, json.loads(payload))
    check_stop(should_stop)
