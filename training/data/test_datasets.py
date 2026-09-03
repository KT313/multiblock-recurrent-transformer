# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

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
    """
    Three parquet files with numbered rows, written in non-sorted order to check sorted-file iteration.
    """

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
    """
    shard_id = rank * num_workers + worker_id over world * num_workers shards.
    """

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
    """
    Every (rank, worker) shard gets exactly rows[shard_id::num_shards]; the shards tile all rows.
    """

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
    """
    The global row index must not reset at parquet read-batch or file boundaries.
    """

    monkeypatch.setattr(datasets_module, "PARQUET_READ_BATCH_ROWS", 4)
    expected = _expected_rows(small_dir)
    for rank in range(3):
        assert [r["text"] for r in ParquetTextDataset(small_dir, "p", shard=(rank, 3))] == expected[rank::3]


def test_iteration_is_repeatable(pretrain_dir: Path) -> None:
    ds = ParquetTextDataset(pretrain_dir, "pre")
    assert [r["text"] for r in ds] == [r["text"] for r in ds]


# --- row range (skip_rows / max_rows) ---------------------------------------------------------------------------------

SHARD_SIZES = [7, 12, 5]  # data-00000 .. data-00002, unequal, several row groups each (row_group_size=4)
TOTAL = sum(SHARD_SIZES)


@pytest.fixture
def ranged_dir(tmp_path: Path) -> Path:
    """
    Three unequal shards in the build's naming scheme; rows carry their directory index as unique text.
    """

    d = tmp_path / "ranged"
    d.mkdir()
    first = 0
    for shard, size in enumerate(SHARD_SIZES):
        texts = [f"doc {i}" for i in range(first, first + size)]
        pq.write_table(pa.table({"text": texts}), d / f"data-{shard:05d}.parquet", row_group_size=4)
        first += size
    return d


def _texts(ds: ParquetTextDataset) -> list[str]:
    return [str(r["text"]) for r in ds]


def _docs(start: int, stop: int) -> list[str]:
    return [f"doc {i}" for i in range(start, stop)]


def test_default_range_is_every_row_in_order(ranged_dir: Path) -> None:
    ds = ParquetTextDataset(ranged_dir, "p")
    assert (ds.start, ds.stop, len(ds)) == (0, TOTAL, TOTAL)
    assert _texts(ds) == _docs(0, TOTAL)


def test_skip_rows_inside_first_shard(ranged_dir: Path) -> None:
    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=3)
    assert len(ds) == TOTAL - 3
    assert _texts(ds) == _docs(3, TOTAL)


def test_skip_rows_spanning_whole_shards(ranged_dir: Path) -> None:
    # skip the first shard exactly, and the first shard plus part of the second
    for skip in (SHARD_SIZES[0], SHARD_SIZES[0] + 5, SHARD_SIZES[0] + SHARD_SIZES[1]):
        ds = ParquetTextDataset(ranged_dir, "p", skip_rows=skip)
        assert len(ds) == TOTAL - skip
        assert _texts(ds) == _docs(skip, TOTAL)


def test_max_rows_ending_mid_shard(ranged_dir: Path) -> None:
    ds = ParquetTextDataset(ranged_dir, "p", max_rows=10)  # ends inside data-00001
    assert len(ds) == 10
    assert _texts(ds) == _docs(0, 10)
    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=2, max_rows=3)  # both ends inside data-00000
    assert _texts(ds) == _docs(2, 5)


def test_skip_plus_max_beyond_total_yields_remainder(ranged_dir: Path) -> None:
    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=20, max_rows=1000)
    assert len(ds) == TOTAL - 20
    assert _texts(ds) == _docs(20, TOTAL)


def test_skip_at_or_beyond_total_yields_nothing(ranged_dir: Path) -> None:
    for skip in (TOTAL, TOTAL + 1, 10 * TOTAL):
        ds = ParquetTextDataset(ranged_dir, "p", skip_rows=skip)
        assert len(ds) == 0
        assert _texts(ds) == []
    assert _texts(ParquetTextDataset(ranged_dir, "p", max_rows=0)) == []


def test_negative_range_raises(ranged_dir: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ParquetTextDataset(ranged_dir, "p", skip_rows=-1)
    with pytest.raises(ValueError, match="non-negative"):
        ParquetTextDataset(ranged_dir, "p", max_rows=-1)


@pytest.mark.parametrize("k", [0, 3, SHARD_SIZES[0], SHARD_SIZES[0] + 4, SHARD_SIZES[0] + SHARD_SIZES[1], TOTAL])
@pytest.mark.parametrize("read_batch", [1024, 3])
def test_split_at_k_is_disjoint_and_complete(
    ranged_dir: Path, monkeypatch: pytest.MonkeyPatch, k: int, read_batch: int
) -> None:
    """
    [0, k) as validation and [k, end) as training tile the directory, for k inside shards and on boundaries,
    with read batches larger than the whole directory and smaller than a row group.
    """

    monkeypatch.setattr(datasets_module, "PARQUET_READ_BATCH_ROWS", read_batch)
    val = ParquetTextDataset(ranged_dir, "val", max_rows=k)
    train = ParquetTextDataset(ranged_dir, "train", skip_rows=k)
    assert _texts(val) == _docs(0, k)
    assert _texts(train) == _docs(k, TOTAL)
    assert len(val) + len(train) == TOTAL


@pytest.mark.parametrize(("world", "num_workers"), [(1, 2), (2, 2), (2, 3)])
def test_shards_partition_the_range_only(
    ranged_dir: Path, monkeypatch: pytest.MonkeyPatch, world: int, num_workers: int
) -> None:
    """
    Shard `shard_id` takes range_rows[shard_id::num_shards]; shards tile the range and never leave it.
    """

    skip, max_rows = 5, 13
    expected = _docs(skip, skip + max_rows)
    seen: list[str] = []
    for rank in range(world):
        for worker_id in range(num_workers):
            monkeypatch.setattr(
                datasets_module, "get_worker_info", lambda w=worker_id: SimpleNamespace(id=w, num_workers=num_workers)
            )
            ds = ParquetTextDataset(ranged_dir, "p", shard=(rank, world), skip_rows=skip, max_rows=max_rows)
            rows = _texts(ds)
            assert rows == expected[rank * num_workers + worker_id :: world * num_workers]
            seen.extend(rows)
    assert sorted(seen) == sorted(expected)


def test_dataloader_workers_yield_the_range_once(ranged_dir: Path) -> None:
    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=4, max_rows=15)
    single = Counter(_texts(ds))
    loader: DataLoader[dict[str, str]] = DataLoader(ds, batch_size=None, num_workers=2)
    multi = Counter(str(r["text"]) for r in loader)
    assert multi == single == Counter(_docs(4, 19))


def test_files_and_row_groups_outside_range_are_not_read(ranged_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=SHARD_SIZES[0] + 9, max_rows=1)  # last group of data-00001
    opened: list[str] = []
    real = pq.ParquetFile

    def spy(path: Path) -> pq.ParquetFile:  # datasets.py only ever passes the path
        opened.append(path.name)
        return real(path)

    monkeypatch.setattr(pq, "ParquetFile", spy)  # datasets.py calls it through the same module object
    assert _texts(ds) == _docs(SHARD_SIZES[0] + 9, SHARD_SIZES[0] + 10)
    assert opened == ["data-00001.parquet"]


def test_range_length_is_the_epoch_and_a_mixture_reads_it_once(ranged_dir: Path) -> None:
    """
    The class does not cycle itself: one __iter__ is one epoch over the range, and a WeightedMixtureDataset over
    it ends with that epoch.
    """

    ds = ParquetTextDataset(ranged_dir, "p", skip_rows=6, max_rows=4)
    assert _texts(ds) == _texts(ds) == _docs(6, 10)
    mixture = WeightedMixtureDataset([ds], [1.0], seed=0)
    rows = [str(r["text"]) for r in itertools.islice(iter(mixture), 10)]
    assert rows == _docs(6, 10)


# --- WeightedMixtureDataset -------------------------------------------------------------------------------------------


class _Counting:
    """
    Finite iterable of tagged items, counts how many times it was (re)started.
    """

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
    members = [_Counting("a", 50_000), _Counting("b", 50_000), _Counting("c", 50_000)]  # no member runs out
    ds = WeightedMixtureDataset(members, [0.6, 0.3, 0.1], seed=1)
    n = 20_000
    counts = Counter(item[0] for item in itertools.islice(iter(ds), n))
    for prefix, w in zip("abc", [0.6, 0.3, 0.1]):
        # seeded, so this is exact; 0.01 is ~3 sigma at n=20k and far below any weight swap/normalisation error
        assert counts[prefix] / n == pytest.approx(w, abs=0.01)


def test_mixture_drops_exhausted_members_and_ends_with_the_last() -> None:
    """
    An exhausted member leaves the draw (never restarted), the others carry on with renormalised weights, and
    the mixture ends once every member is read: every item of every member exactly once, in the member's order.
    """

    a, b = _Counting("a", 4), _Counting("b", 1000)
    ds = WeightedMixtureDataset([a, b], [0.5, 0.5], seed=3)
    items = list(iter(ds))
    assert len(items) == 1004 and a.starts == 1 and b.starts == 1
    assert [i for p, i in items if p == "a"] == [0, 1, 2, 3]
    assert [i for p, i in items if p == "b"] == list(range(1000))
    first_b_after_a = items.index(("a", 3)) + 1
    assert all(p == "b" for p, _ in items[first_b_after_a:])  # only b is left to draw from
    # the draws up to the first exhaustion are the plain weighted draw: the same seed gives the same prefix
    again = list(itertools.islice(iter(WeightedMixtureDataset([_Counting("a", 4), _Counting("b", 1000)], [0.5, 0.5], seed=3)), first_b_after_a))
    assert again == items[:first_b_after_a]


def test_mixture_deterministic_under_seed() -> None:
    def draw(seed: int) -> list[tuple[str, int]]:
        ds = WeightedMixtureDataset([_Counting("a", 50), _Counting("b", 50)], [0.5, 0.5], seed=seed)
        return list(itertools.islice(iter(ds), 200))

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


def test_mixture_over_parquet_datasets(tiny_pretrain_dir: Path, tiny_instruct_dir: Path) -> None:
    sig = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}
    pre = ParquetTextDataset(tiny_pretrain_dir, "pre")
    ft = ParquetTextDataset(tiny_instruct_dir, "ft", data_signature=sig)
    ds = WeightedMixtureDataset([pre, ft], [1.0, 1.0], seed=0)
    rows = list(itertools.islice(iter(ds), 100))
    assert {r["data_id"] for r in rows} == {"pre", "ft"}
    assert all("text" in r for r in rows if r["data_id"] == "pre")
    assert all({"instruction", "input", "output"} <= set(r) for r in rows if r["data_id"] == "ft")


def test_missing_signature_column_raises(small_dir: Path) -> None:
    with pytest.raises(ValueError, match=r"lack the column\(s\) \['instruction', 'output'\]"):
        ParquetTextDataset(
            small_dir, "p", data_signature={"keys": ["instruction", "output"], "format_fn": "pass_text"}
        )
    # the default signature needs `text`, which the files have; extra columns are fine
    assert len(ParquetTextDataset(small_dir, "p")) == 23


def test_missing_text_column_raises(tmp_path: Path) -> None:
    d = tmp_path / "notext"
    d.mkdir()
    pq.write_table(pa.table({"content": ["a", "b"]}), d / "a.parquet")
    with pytest.raises(ValueError, match=r"lack the column\(s\) \['text'\].*found \['content'\]"):
        ParquetTextDataset(d, "p")


def test_empty_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No parquet files"):
        ParquetTextDataset(tmp_path / "empty", "p")


# --- resume offsets ------------------------------------------------------------------------------------------------


def test_resume_offset_starts_the_next_epoch_inside_the_range(small_dir: Path) -> None:
    ds = ParquetTextDataset(small_dir, "small")
    ds.set_resume_offset(7)
    assert ds.resume_offset == 7
    assert [r["text"] for r in ds] == [f"row {i}" for i in range(7, 23)]


def test_resume_offset_holds_until_it_is_set_back(small_dir: Path) -> None:
    """
    The dataset keeps the offset for every epoch until it is set back to 0 (`RunDataloaders` does that before
    the second epoch after a resume: a permanent offset would hide the rows before it).
    """

    ds = ParquetTextDataset(small_dir, "small")
    ds.set_resume_offset(20)
    assert len(list(ds)) == 3 and ds.resume_offset == 20
    assert len(list(ds)) == 3
    ds.set_resume_offset(0)
    assert [r["text"] for r in ds] == [f"row {i}" for i in range(23)]


def test_resume_offset_wraps_around_the_range(small_dir: Path) -> None:
    ds = ParquetTextDataset(small_dir, "small")
    ds.set_resume_offset(23 * 3 + 4)
    assert ds.resume_offset == 4


def test_resume_offset_applies_on_top_of_the_validation_split(small_dir: Path) -> None:
    """
    `skip_rows` is the split, the resume offset is counted inside the resulting range.
    """

    ds = ParquetTextDataset(small_dir, "small", skip_rows=5, max_rows=10)
    ds.set_resume_offset(3)
    assert [r["text"] for r in ds] == [f"row {i}" for i in range(8, 15)]


def test_resume_offset_is_shared_by_the_shards(small_dir: Path) -> None:
    """
    The offset counts rows of the range, not of a shard, so the shards together still yield each row once.
    """

    rows: list[str] = []
    for rank in (0, 1):
        ds = ParquetTextDataset(small_dir, "small", shard=(rank, 2))
        ds.set_resume_offset(10)
        rows += [r["text"] for r in ds]
    assert sorted(rows) == sorted(f"row {i}" for i in range(10, 23))


def test_resume_offset_rejects_a_negative_value(small_dir: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ParquetTextDataset(small_dir, "small").set_resume_offset(-1)
