# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.data import datasets as datasets_module
from training.data.datasets import DEFAULT_DATA_SIGNATURE, ParquetTextDataset, WeightedMixtureDataset


def _expected_rows(data_dir: Path) -> list[str]:
    rows: list[str] = []
    for f in sorted(data_dir.glob("*.parquet")):
        rows.extend(str(x) for x in pq.read_table(f).column("text").to_pylist())
    return rows


@pytest.fixture
def pretrain_dir(tiny_pretrain_dir: Path) -> Path:
    return tiny_pretrain_dir


@pytest.fixture
def small_dir(tmp_path: Path) -> Path:
    """Three parquet files with numbered rows, written in non-sorted order to check sorted-file iteration."""
    d = tmp_path / "small"
    d.mkdir()
    for name, rows in [("b.parquet", range(10, 20)), ("a.parquet", range(10)), ("c.parquet", range(20, 23))]:
        pq.write_table(pa.table({"text": [f"row {i}" for i in rows], "junk": list(rows)}), d / name)
    return d


def test_len_counts_all_rows(pretrain_dir: Path) -> None:
    ds = ParquetTextDataset(pretrain_dir, "pre")
    assert len(ds) == len(_expected_rows(pretrain_dir)) > 0


def test_yields_every_row_once_in_sorted_file_order(small_dir: Path) -> None:
    ds = ParquetTextDataset(small_dir, "small")
    rows = list(ds)
    assert [r["text"] for r in rows] == [f"row {i}" for i in range(23)]
    assert len(rows) == len(ds)


def test_row_carries_signature_and_data_id(small_dir: Path) -> None:
    sig = {"keys": ["text"], "format_fn": "pass_text"}
    ds = ParquetTextDataset(small_dir, "myprefix", data_signature=sig)
    row = next(iter(ds))
    assert row["data_id"] == "myprefix"
    assert row["data_signature"] == sig
    assert set(row) == {"text", "data_signature", "data_id"}, "only signature keys are read"


def test_default_signature(small_dir: Path) -> None:
    ds = ParquetTextDataset(small_dir, "p")
    assert ds.data_signature == DEFAULT_DATA_SIGNATURE


def test_missing_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        ParquetTextDataset(tmp_path / "empty", "p")


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_sharding_partitions_rows_disjointly_and_completely(small_dir: Path, world: int) -> None:
    expected = _expected_rows(small_dir)
    shards = [[r["text"] for r in ParquetTextDataset(small_dir, "p", shard=(rank, world))] for rank in range(world)]
    for rank, shard in enumerate(shards):
        assert shard == expected[rank::world]
    assert sorted(itertools.chain.from_iterable(shards)) == sorted(expected)


def test_shard_without_worker_info(small_dir: Path) -> None:
    assert ParquetTextDataset(small_dir, "p")._shard() == (0, 1)
    assert ParquetTextDataset(small_dir, "p", shard=(2, 3))._shard() == (2, 3)


def test_shard_combines_rank_and_worker(small_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """shard_id = rank * num_workers + worker_id over world * num_workers shards."""
    monkeypatch.setattr(datasets_module, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=2))
    assert ParquetTextDataset(small_dir, "p", shard=(1, 2))._shard() == (3, 4)
    monkeypatch.setattr(datasets_module, "get_worker_info", lambda: SimpleNamespace(id=0, num_workers=3))
    assert ParquetTextDataset(small_dir, "p", shard=(0, 1))._shard() == (0, 3)


def test_iter_respects_worker_shard(small_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_module, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=2))
    rows = [r["text"] for r in ParquetTextDataset(small_dir, "p", shard=(1, 2))]
    assert rows == _expected_rows(small_dir)[3::4]


@pytest.mark.parametrize(("world", "num_workers"), [(1, 2), (2, 2), (3, 2), (2, 3), (1, 3)])
def test_sharding_grid_world_and_workers(
    small_dir: Path, monkeypatch: pytest.MonkeyPatch, world: int, num_workers: int
) -> None:
    """Every (rank, worker) shard gets exactly rows[shard_id::num_shards]; the shards tile all rows."""
    expected = _expected_rows(small_dir)
    seen: list[str] = []
    for rank in range(world):
        for worker_id in range(num_workers):
            monkeypatch.setattr(
                datasets_module, "get_worker_info", lambda w=worker_id: SimpleNamespace(id=w, num_workers=num_workers)
            )
            rows = [r["text"] for r in ParquetTextDataset(small_dir, "p", shard=(rank, world))]
            shard_id = rank * num_workers + worker_id
            assert rows == expected[shard_id :: world * num_workers]
            seen.extend(rows)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(expected), "shards overlap"


def test_sharding_counts_across_read_batches_and_files(small_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The global row index must not reset at parquet read-batch or file boundaries."""
    monkeypatch.setattr(datasets_module, "PARQUET_READ_BATCH_ROWS", 4)
    expected = _expected_rows(small_dir)
    for rank in range(3):
        assert [r["text"] for r in ParquetTextDataset(small_dir, "p", shard=(rank, 3))] == expected[rank::3]


def test_iteration_is_repeatable(pretrain_dir: Path) -> None:
    ds = ParquetTextDataset(pretrain_dir, "pre")
    assert [r["text"] for r in ds] == [r["text"] for r in ds]


# --- WeightedMixtureDataset -------------------------------------------------------------------------------------------


class _Counting:
    """Finite iterable of tagged items, counts how many times it was (re)started."""

    def __init__(self, prefix: str, n: int) -> None:
        self.prefix, self.n, self.starts = prefix, n, 0

    def __iter__(self) -> Iterator[tuple[str, int]]:
        self.starts += 1
        return iter([(self.prefix, i) for i in range(self.n)])


def test_mixture_validates_lengths() -> None:
    with pytest.raises(ValueError):
        WeightedMixtureDataset([_Counting("a", 3)], [0.5, 0.5], seed=0)
    with pytest.raises(ValueError):
        WeightedMixtureDataset([], [], seed=0)


def test_mixture_normalises_weights() -> None:
    ds = WeightedMixtureDataset([_Counting("a", 3), _Counting("b", 3)], [2, 6], seed=0)
    assert ds.weights == pytest.approx([0.25, 0.75])


def test_mixture_frequencies_match_weights() -> None:
    members = [_Counting("a", 1000), _Counting("b", 1000), _Counting("c", 1000)]
    ds = WeightedMixtureDataset(members, [0.6, 0.3, 0.1], seed=1)
    n = 20_000
    counts = Counter(item[0] for item in itertools.islice(iter(ds), n))
    for prefix, w in zip("abc", [0.6, 0.3, 0.1]):
        # seeded, so this is exact; 0.01 is ~3 sigma at n=20k and far below any weight swap/normalisation error
        assert counts[prefix] / n == pytest.approx(w, abs=0.01)


def test_mixture_restarts_exhausted_members() -> None:
    a, b = _Counting("a", 4), _Counting("b", 1000)
    ds = WeightedMixtureDataset([a, b], [0.5, 0.5], seed=3)
    items = list(itertools.islice(iter(ds), 60))
    a_items = [i for p, i in items if p == "a"]
    assert len(a_items) > 4 and a.starts > 1
    # after restarting, 'a' is replayed from the beginning, in order
    assert a_items == [i % 4 for i in range(len(a_items))]
    # every member's order is preserved
    b_items = [i for p, i in items if p == "b"]
    assert b_items == list(range(len(b_items)))


def test_mixture_deterministic_under_seed() -> None:
    def draw(seed: int) -> list[tuple[str, int]]:
        ds = WeightedMixtureDataset([_Counting("a", 50), _Counting("b", 50)], [0.5, 0.5], seed=seed)
        return list(itertools.islice(iter(ds), 200))

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_mixture_over_parquet_datasets(tiny_holdout_dir: Path, tiny_mixture_dirs: dict[str, Path]) -> None:
    sig = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}
    pre = ParquetTextDataset(tiny_holdout_dir, "pre")
    ft = ParquetTextDataset(tiny_mixture_dirs["validation"], "ft", data_signature=sig)
    ds = WeightedMixtureDataset([pre, ft], [1.0, 1.0], seed=0)
    rows = list(itertools.islice(iter(ds), 100))
    assert {r["data_id"] for r in rows} == {"pre", "ft"}
    assert all("text" in r for r in rows if r["data_id"] == "pre")
    assert all({"instruction", "input", "output"} <= set(r) for r in rows if r["data_id"] == "ft")
