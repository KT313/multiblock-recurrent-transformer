# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TypeVar, cast

import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from training.backend.single_device import SingleDeviceBackend
from training.data.collate import IGNORE_INDEX, Batch, Sample, WorkerBatch, collate_samples
from training.data.dataset_resolver import (
    TRAIN_LOADER_NUM_WORKERS,
    DataEntry,
    ResolvedDataset,
    resolve_dataset,
    validation_batches_available,
)
from training.data.loader import (
    TRAIN_LOADER_BATCH_ROWS,
    TRAIN_LOADER_PREFETCH_FACTOR,
    RunDataloaders,
    build_dataloader,
    build_run_dataloaders,
    dataloader_over,
    entry_dataset,
    sample_length,
    world_batch_micro_batches,
)
from training.data.datasets import ParquetTextDataset, Row
from training.data.tokenizer import Tokenizer
from training.settings import Settings, parse_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
INSTRUCT_SIGNATURE = {"keys": ["instruction", "input", "output"], "format_fn": "concatenate_instruction_input_output"}


@pytest.fixture
def entries(tiny_pretrain_dir: Path, tiny_instruct_dir: Path) -> list[DataEntry]:
    return [
        DataEntry("pre", str(tiny_pretrain_dir), weight=0.7),
        DataEntry("ft", str(tiny_instruct_dir), weight=0.3, data_signature=INSTRUCT_SIGNATURE),
    ]


T = TypeVar("T")


def _batches(loader: Iterable[T], n: int) -> list[T]:
    return list(itertools.islice(iter(loader), n))


def _loader(
    entries: list[DataEntry],
    tokenizer: Tokenizer,
    micro_batch_size: int,
    num_workers: int = 0,
    seed: int = 1337,
    shard: tuple[int, int] = (0, 1),
    padding_multiple: int | None = None,
    training_max_sequence_length: int = 64,
    padded: bool = True,
) -> DataLoader[Row]:
    return build_dataloader(
        entries,
        tokenizer,
        training_max_sequence_length,
        micro_batch_size,
        num_workers=num_workers,
        seed=seed,
        shard=shard,
        padding_multiple=padding_multiple,
        padded=padded,
    )


def _rows_in(directory: Path) -> int:
    """
    Rows of the `data-*.parquet` shards of a processed folder (parquet footers only).
    """

    return sum(pq.read_metadata(path).num_rows for path in sorted(directory.glob("data-*.parquet")))


def _same(a: list[Batch], b: list[Batch]) -> bool:
    return all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b))


# --- DataEntry / build_dataloader ------------------------------------------------------------------------------------


def test_entry_defaults_read_the_whole_text_column(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    entry = DataEntry("p", str(tiny_pretrain_dir))
    assert (entry.weight, entry.data_signature, entry.skip_rows, entry.max_rows) == (1.0, None, 0, None)
    assert len(list(_loader([entry], tokenizer, 1))) == _rows_in(tiny_pretrain_dir)  # None = the default text signature


def test_row_range_reaches_the_dataset(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    """
    `skip_rows` / `max_rows` of an entry restrict its dataset (the resolver's validation split): the validation
    range and the training range of one folder are disjoint and together are the folder, in order.
    """

    directory = str(tiny_pretrain_dir)
    total, k = _rows_in(tiny_pretrain_dir), 3

    def rows(entry: DataEntry) -> list[tuple[int, ...]]:
        return [tuple(b[0][0].tolist()) for b in _loader([entry], tokenizer, 1)]

    val = rows(DataEntry("val", directory, max_rows=k))
    train = rows(DataEntry("train", directory, skip_rows=k))
    assert len(val) == k and len(train) == total - k
    assert not set(val) & set(train)
    assert val + train == rows(DataEntry("all", directory))
    both = _loader(
        [DataEntry("val", directory, weight=0.5, max_rows=k), DataEntry("train", directory, weight=0.5, skip_rows=k)],
        tokenizer,
        1,
    )
    tags = Counter(b[2][0] for b in _batches(both, 100))
    assert set(tags) == {"val", "train"}  # a mixture keeps every member's range


def test_pin_memory_is_off_unless_requested(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    entries = [DataEntry("p", str(tiny_pretrain_dir))]
    assert build_dataloader(entries, tokenizer, 64, 2).pin_memory is False
    assert build_dataloader(entries, tokenizer, 64, 2, pin_memory=True).pin_memory is True


def test_run_dataloaders_pin_only_the_validation_batches(tiny_settings: Settings) -> None:
    """
    The validation loaders yield the padded batches that go to the device, so they pin with the backend; the train
    loaders yield unpadded samples that `pad_and_shift` copies into a fresh pageable micro-batch, so they never pin.
    """

    dataset: ResolvedDataset = resolve_dataset(tiny_settings)
    for pin_memory in (False, True):
        backend = SingleDeviceBackend(device="cpu", precision="32")
        backend.pin_memory = pin_memory  # a CUDA backend's choice, without a GPU (the loaders are never iterated)
        loaders = build_run_dataloaders(tiny_settings, dataset, backend)
        assert all(cast(DataLoader[Row], loader).pin_memory is False for loader in loaders.train_loaders.values())
        assert all(cast(DataLoader[Row], loader).pin_memory is pin_memory for loader in loaders.val_loaders)


def test_duplicate_prefixes_rejected(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    d = str(tiny_pretrain_dir)
    with pytest.raises(ValueError, match="unique"):
        build_dataloader([DataEntry("p", d), DataEntry("p", d)], tokenizer, 64, 2)


def test_single_spec_batches(tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path) -> None:
    loader = _loader(entries[:1], tokenizer, 4, padding_multiple=16)
    batches = _batches(loader, 3)
    for input_ids, labels, data_ids in batches:
        assert input_ids.shape == labels.shape == (4, 64)
        assert input_ids.dtype == torch.long
        assert data_ids == ["pre"] * 4
    # a single dataset is finite: one pass over its rows, the last batch may be short
    assert len(list(loader)) == math.ceil(_rows_in(tiny_pretrain_dir) / 4)


# Mixtures with the instruct folder use a block size no instruct prompt can fill, so no row is dropped for lack of a
# supervised label; the processed instruct rows are only bounded by dataset_max_sequence_length (256).
MIXTURE_BLOCK_SIZE = 128


def test_mixture_loader_mixes_by_weight_and_reads_every_member_once(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path, tiny_instruct_dir: Path
) -> None:
    """
    The draws follow the weights while every member has rows; the loader ends once each member was read once.
    """

    loader = _loader(entries, tokenizer, 2, seed=0, training_max_sequence_length=MIXTURE_BLOCK_SIZE)
    batches = list(loader)
    pre_rows, ft_rows = _rows_in(tiny_pretrain_dir), _rows_in(tiny_instruct_dir)
    assert len(batches) == math.ceil((pre_rows + ft_rows) / 2)
    ids = Counter(itertools.chain.from_iterable(b[2] for b in batches))
    assert ids == {"pre": pre_rows, "ft": ft_rows}
    first = Counter(itertools.chain.from_iterable(b[2] for b in batches[:10]))  # 20 rows, both members still in
    assert first["pre"] / 20 == pytest.approx(0.7, abs=0.2)


def test_validation_mixture_is_finite_and_matches_the_batch_count(
    tokenizer: Tokenizer, tiny_pretrain_dir: Path, tiny_instruct_dir: Path
) -> None:
    """
    A two-entry validation stage: the loader is finite, its batch count is what `validation_batches_available`
    promised at setup, and the smaller member's rows appear exactly once.
    """

    stage_entries = [
        DataEntry("s-pre", str(tiny_pretrain_dir), weight=0.7, max_rows=10),
        DataEntry("s-ft", str(tiny_instruct_dir), weight=0.3, data_signature=INSTRUCT_SIGNATURE, max_rows=3),
    ]
    rows = {str(tiny_pretrain_dir): _rows_in(tiny_pretrain_dir), str(tiny_instruct_dir): _rows_in(tiny_instruct_dir)}
    loader = _loader(stage_entries, tokenizer, 4, seed=0, training_max_sequence_length=MIXTURE_BLOCK_SIZE)
    batches = list(loader)
    assert len(batches) == validation_batches_available(stage_entries, rows, micro_batch_size=4, world_size=1) == 4
    ids = Counter(itertools.chain.from_iterable(b[2] for b in batches))
    assert ids == {"s-pre": 10, "s-ft": 3}


def test_loader_deterministic_under_seed(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    a = _batches(_loader(entries, tokenizer, 2, seed=5, training_max_sequence_length=MIXTURE_BLOCK_SIZE), 10)
    b = _batches(_loader(entries, tokenizer, 2, seed=5, training_max_sequence_length=MIXTURE_BLOCK_SIZE), 10)
    c = _batches(_loader(entries, tokenizer, 2, seed=6, training_max_sequence_length=MIXTURE_BLOCK_SIZE), 10)
    assert _same(a, b)
    assert not _same(a, c)


def test_workers_zero_and_two_identical_with_micro_batch_one(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path
) -> None:
    """
    Rows are dealt round-robin to workers and DataLoader collects worker batches round-robin, so with
    micro_batch_size=1 the two loaders yield the very same sequence.
    """

    a = list(_loader(entries[:1], tokenizer, 1, num_workers=0))
    b = list(_loader(entries[:1], tokenizer, 1, num_workers=2))
    assert len(a) == len(b) == _rows_in(tiny_pretrain_dir)
    assert _same(a, b)


def test_workers_two_micro_batch_gt_one_regroups_rows(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path
) -> None:
    """
    With micro_batch_size>1 each worker batches *its* rows (0,2,4.. / 1,3,5..), so batches differ from
    num_workers=0 in composition but cover exactly the same rows over an epoch.
    """

    a = list(_loader(entries[:1], tokenizer, 2, num_workers=0))
    b = list(_loader(entries[:1], tokenizer, 2, num_workers=2))
    assert len(a) == len(b) == math.ceil(_rows_in(tiny_pretrain_dir) / 2)

    def rows(batches: list[Batch]) -> list[tuple[int, ...]]:
        return sorted(tuple(r.tolist()) for x in batches for r in x[0])

    assert rows(a) == rows(b)
    assert not _same(a, b)
    assert [tuple(r.tolist()) for r in b[0][0]] == [tuple(a[0][0][0].tolist()), tuple(a[1][0][0].tolist())]


def test_workers_two_mixture_is_deterministic(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    a = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, training_max_sequence_length=MIXTURE_BLOCK_SIZE), 12)
    b = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, training_max_sequence_length=MIXTURE_BLOCK_SIZE), 12)
    assert _same(a, b)


def test_unusable_rows_are_dropped_without_ending_the_loader(tokenizer: Tokenizer, tiny_instruct_dir: Path) -> None:
    """
    Regression (T-M4): a row with no supervised label used to raise `StopIteration` out of the collate function,
    which torch's worker loop reads as 'this worker is done' and the single-process loop as 'restart at row 0'. At
    `training_max_sequence_length` 16 most instruct prompts alone fill the window; the loader still walks its whole epoch.
    """

    rows = list(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE)))
    kept = len(collate_samples(rows, tokenizer, training_max_sequence_length=16))
    assert 0 < kept < len(rows), "the fixture must drop some rows and keep others"
    entry = DataEntry("ft", str(tiny_instruct_dir), data_signature=INSTRUCT_SIGNATURE)
    for num_workers in (0, 2):
        loader = build_dataloader([entry], tokenizer, 16, 4, num_workers=num_workers, padded=False)
        batches = list(loader)
        assert sum(len(batch.samples) for batch in batches) == kept
        # rows READ still add up to the whole epoch across the worker shards: what the resume counters are made of
        assert sum(batch.rows_read for batch in batches) == len(rows)


def _worker_batches(loader: Iterable[WorkerBatch]) -> list[WorkerBatch]:
    return list(loader)


def _samples_of(batches: list[WorkerBatch]) -> list[tuple[list[int], list[int], str]]:
    return [(ids.tolist(), labels.tolist(), tag) for batch in batches for ids, labels, tag in batch.samples]


@pytest.mark.parametrize("worker_batch_rows", [1, 7, 64])
def test_worker_batch_rows_regroup_the_same_sample_sequence(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path, worker_batch_rows: int
) -> None:
    """
    The worker batch size of an unpadded loader is a grouping, not an order: the one reader walks its range in
    order whatever the batch size, so the concatenated samples of an epoch are the same sequence as with worker
    batches of `micro_batch_size` rows (the old loaders), every batch but the last holds `worker_batch_rows` rows
    and the `rows_read` counts still add up to the epoch. This is what lets `BatchStream` buffer wide worker batches
    without changing which sample reaches which micro-batch.
    """

    rows = _rows_in(tiny_pretrain_dir)
    reference = _worker_batches(build_dataloader(entries[:1], tokenizer, 64, 2, padded=False))
    regrouped = _worker_batches(
        build_dataloader(entries[:1], tokenizer, 64, 2, padded=False, worker_batch_rows=worker_batch_rows)
    )
    assert [len(batch.samples) for batch in reference] == [2] * (rows // 2) + ([rows % 2] if rows % 2 else [])
    assert [batch.rows_read for batch in regrouped] == [worker_batch_rows] * (rows // worker_batch_rows) + (
        [rows % worker_batch_rows] if rows % worker_batch_rows else []
    )
    assert sum(batch.rows_read for batch in regrouped) == sum(batch.rows_read for batch in reference) == rows
    assert _samples_of(regrouped) == _samples_of(reference)


def test_worker_batch_rows_keep_the_sample_sequence_when_rows_are_dropped(
    tokenizer: Tokenizer, tiny_instruct_dir: Path
) -> None:
    """
    Dropped rows (no supervised label at `training_max_sequence_length` 16) cost one `rows_read` each in whatever batch they fall
    into, so the surviving sample sequence and the total rows read are the same for any worker batch size; only
    the batches are shorter than their row count.
    """

    entry = DataEntry("ft", str(tiny_instruct_dir), data_signature=INSTRUCT_SIGNATURE)
    rows = len(list(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE))))
    reference = _worker_batches(build_dataloader([entry], tokenizer, 16, 4, padded=False))
    wide = _worker_batches(build_dataloader([entry], tokenizer, 16, 4, padded=False, worker_batch_rows=64))
    assert sum(batch.rows_read for batch in wide) == sum(batch.rows_read for batch in reference) == rows
    assert any(len(batch.samples) < batch.rows_read for batch in wide)  # the fixture drops rows at this block size
    assert _samples_of(wide) == _samples_of(reference)


def test_worker_batch_rows_are_refused_where_they_make_no_sense(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    with pytest.raises(ValueError, match="unpadded loaders only"):
        build_dataloader(entries[:1], tokenizer, 64, 2, padded=True, worker_batch_rows=64)
    with pytest.raises(ValueError, match="must be positive"):
        build_dataloader(entries[:1], tokenizer, 64, 2, padded=False, worker_batch_rows=0)


def test_shard_passed_to_datasets(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    full = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1)]
    r0 = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1, shard=(0, 2))]
    r1 = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1, shard=(1, 2))]
    assert r0 == full[0::2] and r1 == full[1::2]


# --- build_run_dataloaders --------------------------------------------------------------------------------------------


@pytest.fixture
def tiny_settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(
        ["--config", str(TINY_YAML), "--dataset_dir", str(tiny_dataset_dir), "--out_dir", str(tmp_path / "out")]
    )


def test_build_run_dataloaders(tiny_settings: Settings, tokenizer: Tokenizer) -> None:
    """
    One train loader per SOURCE (the whole-run readers, one worker each, worker batches of `TRAIN_LOADER_BATCH_ROWS`
    rows kept `TRAIN_LOADER_PREFETCH_FACTOR` batches ahead) and one validation loader per stage of the tiny dataset
    (batches of `micro_batch_size` rows), tokenizer loaded from the resolved directory, train loaders unpadded,
    validation loaders padded and restricted to the held-out rows of the split.
    """

    dataset: ResolvedDataset = resolve_dataset(tiny_settings)
    loaders = build_run_dataloaders(tiny_settings, dataset, SingleDeviceBackend(device="cpu", precision="32"))
    assert isinstance(loaders, RunDataloaders)
    assert loaders.train_sources == ["synthetic_pretrain", "synthetic_instruct"]  # dataset-config order
    assert len(loaders.train_loaders) == 2 and len(loaders.val_loaders) == len(dataset.stages) == 3
    for train_loader in loaders.train_loaders.values():
        assert isinstance(train_loader, DataLoader) and train_loader.num_workers == TRAIN_LOADER_NUM_WORKERS
        assert train_loader.batch_size == TRAIN_LOADER_BATCH_ROWS > tiny_settings.micro_batch_size
        assert train_loader.prefetch_factor == TRAIN_LOADER_PREFETCH_FACTOR
    for val_loader in loaders.val_loaders:
        assert isinstance(val_loader, DataLoader) and val_loader.batch_size == tiny_settings.micro_batch_size
    assert list(loaders.datasets) == loaders.train_sources
    for source, parquet in loaders.datasets.items():
        assert isinstance(parquet, ParquetTextDataset) and parquet.prefix == source
        assert cast(DataLoader[Row], loaders.train_loaders[source]).dataset is parquet
    assert loaders.tokenizer.path == Path(dataset.tokenizer_dir)
    batch = loaders.next_train_batch("synthetic_pretrain")
    samples = batch.samples
    worker_rows = min(TRAIN_LOADER_BATCH_ROWS, loaders.datasets["synthetic_pretrain"].num_rows)
    assert len(samples) == worker_rows
    assert batch.rows_read == worker_rows  # no row dropped
    assert [s[2] for s in samples] == ["synthetic_pretrain"] * worker_rows
    for input_ids, labels, _ in samples:  # unpadded: the true token count, capped at training_max_sequence_length + 1
        assert input_ids.shape == labels.shape and 0 < input_ids.shape[0] <= tiny_settings.training_max_sequence_length + 1
    input_ids, labels, _ = world_batch_micro_batches(
        samples,
        tiny_settings.micro_batch_size,
        tokenizer,
        tiny_settings.training_max_sequence_length,
        sort_by_length=True,
        padding_multiple=tiny_settings.sequence_padding_multiple,
    )[0]
    # padding rounds up to sequence_padding_multiple (capped at training_max_sequence_length + 1), then the label shift drops one
    assert (input_ids.shape[1] + 1) % 128 == 0 or input_ids.shape[1] == tiny_settings.training_max_sequence_length
    assert input_ids.shape[1] <= tiny_settings.training_max_sequence_length
    assert (labels == IGNORE_INDEX).any() or (input_ids != tokenizer.pad_id).all()
    _, _, val_ids = next(iter(loaders.val_loaders[2]))
    assert val_ids == ["finetune-synthetic_instruct"] * tiny_settings.micro_batch_size
    # the validation loaders read only the held-out first rows of the split (a single dataset is one finite epoch)
    for stage_idx, source in ((0, "synthetic_pretrain"), (2, "synthetic_instruct")):
        k = dataset.validation_rows[source]
        assert k >= 1 and len(list(loaders.val_loaders[stage_idx])) == math.ceil(k / tiny_settings.micro_batch_size)


# --- RunDataloaders ---------------------------------------------------------------------------------------------------


def _tagged(tag: str, n: int) -> list[WorkerBatch]:
    """
    A finite 'loader' yielding n one-sample worker batches tagged with `tag`.
    """

    return [WorkerBatch([(torch.full((2,), i), torch.full((2,), i), tag)], 1) for i in range(n)]


def _first(batch: WorkerBatch) -> int:
    """
    The counter value of a `_tagged` worker batch.
    """

    return int(batch.samples[0][0][0])


def test_next_train_batch_cycles_on_exhaustion(tokenizer: Tokenizer) -> None:
    """
    A source that runs dry restarts its loader (an empty source cannot occur: the resolver's
    `check_entry_rows` guarantees at least one training row per source).
    """

    rd = RunDataloaders({"a": _tagged("a", 3), "b": _tagged("b", 2)}, [], tokenizer, {})
    assert [_first(rd.next_train_batch("a")) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert [s[2] for s in rd.next_train_batch("b").samples] == ["b"]
    assert rd._train_iterators["a"] is not None and rd._train_iterators["b"] is not None


def test_post_init_creates_one_slot_per_source_in_order(tokenizer: Tokenizer) -> None:
    rd = RunDataloaders({tag: _tagged(tag, 1) for tag in "abc"}, [], tokenizer, {})
    assert rd._train_iterators == {"a": None, "b": None, "c": None}
    assert rd.train_sources == ["a", "b", "c"] and rd.pending_offsets == {}
    assert RunDataloaders({}, [], tokenizer, {})._train_iterators == {}


def test_iterators_are_lazy_and_independent(tokenizer: Tokenizer) -> None:
    rd = RunDataloaders({"a": _tagged("a", 3), "b": _tagged("b", 3)}, [], tokenizer, {})
    assert rd._train_iterators == {"a": None, "b": None}
    rd.next_train_batch("b")
    assert rd._train_iterators["a"] is None
    assert _first(rd.next_train_batch("b")) == 1
    assert _first(rd.next_train_batch("a")) == 0


def test_set_resume_offsets_are_applied_when_the_iterator_starts(
    tokenizer: Tokenizer, entries: list[DataEntry]
) -> None:
    """
    The offsets wait in `pending_offsets` and land on a source's dataset right before its first iterator is
    created; a source without a pending offset starts at 0.
    """

    pre, ft = entry_dataset(entries[0]), entry_dataset(entries[1])
    loaders: dict[str, Iterable[WorkerBatch]] = {
        "pre": dataloader_over(pre, tokenizer, 64, 2, padded=False),
        "ft": dataloader_over(ft, tokenizer, 64, 2, padded=False),
    }
    rd = RunDataloaders(loaders, [], tokenizer, {"pre": pre, "ft": ft})
    rd.set_resume_offsets({"pre": 3, "gone": 9})  # a name no source has is ignored
    assert rd.pending_offsets == {"pre": 3} and (pre.resume_offset, ft.resume_offset) == (0, 0)
    rd.next_train_batch("pre")
    assert rd.pending_offsets == {} and pre.resume_offset == 3
    rd.next_train_batch("ft")
    assert ft.resume_offset == 0


def test_resume_offset_is_dropped_when_the_loader_restarts(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    """
    The first epoch after a resume starts at the offset; once it ends, the loader reads its whole range again.
    """

    parquet = entry_dataset(DataEntry("pre", str(tiny_pretrain_dir)))
    total = _rows_in(tiny_pretrain_dir)
    rd = RunDataloaders(
        {"pre": dataloader_over(parquet, tokenizer, 64, 1, padded=False)}, [], tokenizer, {"pre": parquet}
    )
    rd.set_resume_offsets({"pre": total - 2})
    first_epoch = [rd.next_train_batch("pre").samples[0][2] for _ in range(2)]
    assert first_epoch == ["pre", "pre"] and parquet.resume_offset == total - 2  # the offset holds for its epoch
    assert len([rd.next_train_batch("pre") for _ in range(total)]) == total  # the restart reads every row
    assert parquet.resume_offset == 0


def test_close_shuts_down_the_worker_iterators(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    """
    `close()` stops the worker processes of every live train iterator right away and forgets the iterators;
    a fake without workers and a second call are no-ops.
    """

    parquet = entry_dataset(DataEntry("pre", str(tiny_pretrain_dir)))
    loader = dataloader_over(parquet, tokenizer, 64, 1, num_workers=1, padded=False)
    rd = RunDataloaders({"pre": loader, "fake": _tagged("fake", 2)}, [], tokenizer, {"pre": parquet})
    rd.next_train_batch("pre")
    rd.next_train_batch("fake")
    workers = cast(Any, rd._train_iterators["pre"])._workers  # the worker processes of the live iterator
    assert all(worker.is_alive() for worker in workers)
    rd.close()
    assert rd._train_iterators == {"pre": None, "fake": None}
    assert not any(worker.is_alive() for worker in workers)
    rd.close()
    assert _first(rd.next_train_batch("fake")) == 0  # a fresh iterator after close


# --- world_batch_micro_batches ----------------------------------------------------------------------------------------

BLOCK = 64  # cap of the fake-sample tests: training_max_sequence_length + 1 = 65 tokens


def _sample(length: int, tag: str) -> Sample:
    """
    An unpadded sample of `length` valid (non-pad, in-vocab) tokens; every position is supervised.
    """

    ids = torch.full((length,), 3, dtype=torch.long)
    return ids, ids.clone(), tag


def _split(
    tokenizer: Tokenizer, samples: list[Sample], micro_batch_size: int, sort: bool, multiple: int | None = None
) -> list[Batch]:
    return world_batch_micro_batches(
        samples, micro_batch_size, tokenizer, BLOCK, sort_by_length=sort, padding_multiple=multiple
    )


def _widths(batches: list[Batch]) -> list[tuple[int, int]]:
    return [(b[0].shape[0], b[0].shape[1]) for b in batches]


def test_sample_length_is_the_token_count() -> None:
    assert sample_length(_sample(7, "a")) == 7


def test_world_batch_sorts_shortest_first_and_pads_each_micro_batch(tokenizer: Tokenizer) -> None:
    samples = [_sample(7, "a"), _sample(2, "b"), _sample(5, "c"), _sample(8, "d")]
    out = _split(tokenizer, samples, micro_batch_size=2, sort=True)
    assert [b[2] for b in out] == [["b", "c"], ["a", "d"]]  # 2,5 then 7,8
    assert _widths(out) == [(2, 4), (2, 7)]  # padded to the longest sample of the micro-batch, then shifted
    assert [int((lab != IGNORE_INDEX).sum()) for b in out for lab in b[1]] == [1, 4, 6, 7]


def test_world_batch_without_sorting_keeps_arrival_order(tokenizer: Tokenizer) -> None:
    samples = [_sample(7, "a"), _sample(2, "b"), _sample(5, "c"), _sample(8, "d")]
    out = _split(tokenizer, samples, micro_batch_size=2, sort=False)
    assert [b[2] for b in out] == [["a", "b"], ["c", "d"]]
    assert _widths(out) == [(2, 6), (2, 7)]


def test_world_batch_ties_keep_arrival_order(tokenizer: Tokenizer) -> None:
    samples = [_sample(3, "a"), _sample(3, "b"), _sample(3, "c"), _sample(1, "d")]
    assert [b[2] for b in _split(tokenizer, samples, micro_batch_size=2, sort=True)] == [["d", "a"], ["b", "c"]]


def test_world_batch_padding_multiple_rounds_the_width_up(tokenizer: Tokenizer) -> None:
    samples = [_sample(5, "a"), _sample(2, "b"), _sample(9, "c"), _sample(1, "d")]
    out = _split(tokenizer, samples, micro_batch_size=2, sort=True, multiple=4)
    # chunks 1,2 -> padded to 4 -> shifted 3 ; 5,9 -> padded to 12 -> shifted 11
    assert _widths(out) == [(2, 3), (2, 11)]


def test_world_batch_width_is_capped_at_block_size(tokenizer: Tokenizer) -> None:
    samples = [_sample(BLOCK + 1, "a"), _sample(BLOCK + 1, "b")]
    assert _widths(_split(tokenizer, samples, micro_batch_size=2, sort=True, multiple=128)) == [(2, BLOCK)]


def test_world_batch_uneven_split_yields_a_partial_micro_batch(tokenizer: Tokenizer) -> None:
    out = _split(tokenizer, [_sample(3, "a"), _sample(1, "b"), _sample(2, "c")], micro_batch_size=2, sort=True)
    assert [b[2] for b in out] == [["b", "c"], ["a"]]


def test_world_batch_width_does_not_depend_on_the_loader_grouping(tokenizer: Tokenizer) -> None:
    """
    The point of assembling before padding: a micro-batch of short rows is no longer widened because some other
    micro-batch of the same world batch happened to contain a long row.
    """

    short = [_sample(5, "a"), _sample(6, "b")]
    long = [_sample(60, "c"), _sample(61, "d")]
    alone = _split(tokenizer, short, micro_batch_size=2, sort=True, multiple=8)
    together = _split(tokenizer, short + long, micro_batch_size=2, sort=True, multiple=8)
    assert _widths(alone) == [(2, 7)] and _widths(together)[0] == (2, 7)


def test_world_batch_on_real_loader_preserves_every_sample(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    loader = build_dataloader(entries[:1], tokenizer, 512, 4, padding_multiple=128, padded=False)
    worker_batches: list[WorkerBatch] = _batches(loader, 2)
    samples = [s for batch in worker_batches for s in batch.samples]
    lengths = [sample_length(s) for s in samples]
    assert lengths != sorted(lengths), "fixture must start unsorted for the test to mean anything"
    out = world_batch_micro_batches(samples, 4, tokenizer, 512, sort_by_length=True, padding_multiple=128)
    assert [b[0].shape[0] for b in out] == [4, 4]
    assert [b[2] for b in out] == [["pre"] * 4, ["pre"] * 4]

    def supervised(batches: list[Batch]) -> list[list[int]]:
        return sorted(lab[lab != IGNORE_INDEX].tolist() for _, labs, _ in batches for lab in labs)

    reference = world_batch_micro_batches(samples, 4, tokenizer, 512, sort_by_length=False, padding_multiple=128)
    assert supervised(out) == supervised(reference)  # sorting only regroups, it never drops a label
    for input_ids, labels, _ in out:
        assert input_ids.shape == labels.shape and (input_ids.shape[1] + 1) % 128 == 0


def _prompt_masked_sample(prompt: int, answer: int, tag: str, pad_id: int) -> Sample:
    """
    An instruct-shaped sample: `prompt` masked positions (pad id in the labels), then `answer` supervised ones.
    """

    ids = torch.full((prompt + answer,), 3, dtype=torch.long)
    labels = ids.clone()
    labels[:prompt] = pad_id
    return ids, labels, tag


def test_world_batch_keeps_every_supervised_label_of_prompt_masked_rows(tokenizer: Tokenizer) -> None:
    """
    An instruct row's labels sit at the END of the row, so the width must come from the full length; a width
    derived from the count of supervised labels would cut the answers off long-prompt rows.
    """

    pad = tokenizer.pad_id
    samples = [_prompt_masked_sample(200, 12, "a", pad), _prompt_masked_sample(150, 30, "b", pad)]
    out = world_batch_micro_batches(samples, 2, tokenizer, 255, sort_by_length=True, padding_multiple=128)
    assert _widths(out) == [(2, 255)]
    # the shift drops the first label of each row; both rows keep every supervised position they had
    assert sorted(int((lab != IGNORE_INDEX).sum()) for _, labs, _ in out for lab in labs) == [12, 30]


def test_world_batch_keeps_the_labels_of_real_instruct_rows(tokenizer: Tokenizer, tiny_instruct_dir: Path) -> None:
    rows = list(itertools.islice(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE)), 8))
    samples = collate_samples(rows, tokenizer, training_max_sequence_length=255)
    expected = sorted(int((lab[1:] != tokenizer.pad_id).sum()) for _, lab, _ in samples)
    out = world_batch_micro_batches(samples, 4, tokenizer, 255, sort_by_length=True, padding_multiple=128)
    assert len(samples) == 8 and expected[0] > 0
    assert sorted(int((lab != IGNORE_INDEX).sum()) for _, labs, _ in out for lab in labs) == expected


def test_the_data_package_re_exports_nothing() -> None:
    """
    `training.data` must stay import-light: a re-export of the torch modules would load torch for everyone
    importing `dataset_resolver` (the framework-neutral module living in this package).
    """

    import training.data as pkg

    assert not hasattr(pkg, "__all__") and not hasattr(pkg, "collate_fn")
