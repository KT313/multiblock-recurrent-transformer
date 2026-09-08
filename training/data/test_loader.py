# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
import logging
import math
import resource
import signal
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from training.backend.single_device import SingleDeviceBackend
from training.data.collate import Batch, WorkerBatch, collate_samples
from training.data.tokenizer import IGNORE_INDEX
from training.data.dataset_resolver import (
    TRAIN_LOADER_NUM_WORKERS,
    DataEntry,
    ResolvedDataset,
    resolve_dataset,
    validation_batches_available,
)
from training.data.loader import (
    RunDataloaders,
    TRAIN_LOADER_BATCH_ROWS,
    TRAIN_LOADER_PREFETCH_FACTOR,
    UNLIMITED_OPEN_FILES,
    build_dataloader,
    build_run_dataloaders,
    dataloader_over,
    entry_dataset,
    raise_open_file_limit,
    worker_init_fn,
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
    batch_size: int,
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
        batch_size,
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
    if len(a) != len(b):  # zip alone would call two runs of different length equal on their common prefix
        return False
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


# Mixtures with the instruct folder use a sequence length no instruct prompt can fill, so no row is dropped for lack of a
# supervised label; the processed instruct rows are only bounded by dataset_max_sequence_length (256).
MIXTURE_SEQUENCE_LENGTH = 128


def test_mixture_loader_mixes_by_weight_and_reads_every_member_once(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path, tiny_instruct_dir: Path
) -> None:
    """
    The draws follow the weights while every member has rows; the loader ends once each member was read once.
    """

    loader = _loader(entries, tokenizer, 2, seed=0, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH)
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
    loader = _loader(stage_entries, tokenizer, 4, seed=0, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH)
    batches = list(loader)
    assert len(batches) == validation_batches_available(stage_entries, rows, validation_batch_size=4, world_size=1) == 4
    ids = Counter(itertools.chain.from_iterable(b[2] for b in batches))
    assert ids == {"s-pre": 10, "s-ft": 3}


def test_loader_deterministic_under_seed(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    a = _batches(_loader(entries, tokenizer, 2, seed=5, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH), 10)
    b = _batches(_loader(entries, tokenizer, 2, seed=5, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH), 10)
    c = _batches(_loader(entries, tokenizer, 2, seed=6, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH), 10)
    assert _same(a, b)
    assert not _same(a, c)


def test_workers_zero_and_two_identical_with_micro_batch_one(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path
) -> None:
    """
    Rows are dealt round-robin to workers and DataLoader collects worker batches round-robin, so with a batch
    size of 1 the two loaders yield the very same sequence.
    """

    a = list(_loader(entries[:1], tokenizer, 1, num_workers=0))
    b = list(_loader(entries[:1], tokenizer, 1, num_workers=2))
    assert len(a) == len(b) == _rows_in(tiny_pretrain_dir)
    assert _same(a, b)


def test_workers_two_micro_batch_gt_one_regroups_rows(
    tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path
) -> None:
    """
    With a batch size above 1 each worker batches *its* rows (0,2,4.. / 1,3,5..), so batches differ from
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
    a = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH), 12)
    b = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, training_max_sequence_length=MIXTURE_SEQUENCE_LENGTH), 12)
    assert _same(a, b)


class _SignalDispositions(IterableDataset[Any]):
    def __iter__(self) -> Iterator[Any]:
        yield signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)


def test_workers_ignore_sigint_but_not_sigterm(tokenizer: Tokenizer) -> None:
    """
    A Ctrl-C reaches the whole process group; a worker must leave it to the parent's handler. SIGTERM stays the
    default: it is how a leftover worker is ended at exit.
    """

    assert dataloader_over(_SignalDispositions(), tokenizer, 64, 1).worker_init_fn is worker_init_fn
    loader = DataLoader(_SignalDispositions(), batch_size=None, num_workers=1, worker_init_fn=worker_init_fn)
    (sigint, sigterm), = list(loader)
    assert sigint == signal.SIG_IGN and sigterm == signal.SIG_DFL


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
    batches of two rows, every batch but the last holds `worker_batch_rows` rows and the `rows_read` counts still
    add up to the epoch. This is what lets `BatchStream` buffer wide worker batches without changing which sample
    reaches which pack.
    """

    rows = _rows_in(tiny_pretrain_dir)
    reference = _worker_batches(build_dataloader(entries[:1], tokenizer, 64, 2, padded=False))
    regrouped = _worker_batches(build_dataloader(entries[:1], tokenizer, 64, worker_batch_rows, padded=False))
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
    wide = _worker_batches(build_dataloader([entry], tokenizer, 16, 64, padded=False))
    assert sum(batch.rows_read for batch in wide) == sum(batch.rows_read for batch in reference) == rows
    assert any(len(batch.samples) < batch.rows_read for batch in wide)  # the fixture drops rows at this block size
    assert _samples_of(wide) == _samples_of(reference)


def test_a_loader_needs_a_positive_batch_size(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_dataloader(entries[:1], tokenizer, 64, 0, padded=False)


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
    (batches of `validation_batch_size` rows), tokenizer loaded from the resolved directory, train loaders unpadded,
    validation loaders padded and restricted to the held-out rows of the split.
    """

    dataset: ResolvedDataset = resolve_dataset(tiny_settings)
    loaders = build_run_dataloaders(tiny_settings, dataset, SingleDeviceBackend(device="cpu", precision="32"))
    assert isinstance(loaders, RunDataloaders)
    assert loaders.train_sources == ["synthetic_pretrain", "synthetic_instruct"]  # dataset-config order
    assert len(loaders.train_loaders) == 2 and len(loaders.val_loaders) == len(dataset.stages) == 3
    for train_loader in loaders.train_loaders.values():
        assert isinstance(train_loader, DataLoader) and train_loader.num_workers == TRAIN_LOADER_NUM_WORKERS
        assert train_loader.batch_size == TRAIN_LOADER_BATCH_ROWS
        assert train_loader.prefetch_factor == TRAIN_LOADER_PREFETCH_FACTOR
    for val_loader in loaders.val_loaders:
        assert isinstance(val_loader, DataLoader) and val_loader.batch_size == tiny_settings.validation_batch_size
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
    input_ids, labels, _ = next(iter(loaders.val_loaders[0]))  # pretrain rows: every ignored label is padding
    # validation rows are padded to a multiple of validation_padding_multiple (capped at training_max_sequence_length + 1),
    # then the label shift drops one position
    assert (input_ids.shape[1] + 1) % 128 == 0 or input_ids.shape[1] == tiny_settings.training_max_sequence_length
    assert input_ids.shape[1] <= tiny_settings.training_max_sequence_length
    assert (input_ids[labels == IGNORE_INDEX] == tokenizer.eos_id).all()  # padding is EOS in the inputs
    _, _, val_ids = next(iter(loaders.val_loaders[2]))
    assert val_ids == ["finetune-synthetic_instruct"] * tiny_settings.validation_batch_size
    # the validation loaders read only the held-out first rows of the split (a single dataset is one finite epoch)
    for stage_idx, source in ((0, "synthetic_pretrain"), (2, "synthetic_instruct")):
        k = dataset.validation_rows[source]
        assert k >= 1 and len(list(loaders.val_loaders[stage_idx])) == math.ceil(k / tiny_settings.validation_batch_size)


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


# --- dropped rows and the epoch check (item 2) -------------------------------------------------------------------

DROPPING_SEQUENCE_LENGTH = 16  # rows are cut to 17 tokens, which the long prompt below fills on its own
LONG_PROMPT = " ".join(f"tok_{i}" for i in range(40))  # masked and longer than the window: the row keeps no label
SHORT_PROMPT = "tok_1 tok_2"


def _instruct_source(
    directory: Path, tokenizer: Tokenizer, prompts: list[str]
) -> tuple[RunDataloaders, ParquetTextDataset]:
    """
    A one-source `RunDataloaders` over a fresh instruct parquet with one row per prompt, read one row per worker
    batch at `DROPPING_SEQUENCE_LENGTH`: a row whose prompt is `LONG_PROMPT` is dropped, one with a short prompt
    keeps its answer.
    """

    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "instruction": prompts,
            "input": [""] * len(prompts),
            "output": [f"tok_{i} tok_{i + 1} tok_{i + 2}" for i in range(len(prompts))],
        }
    )
    pq.write_table(table, directory / "data-00000.parquet")
    parquet = entry_dataset(DataEntry("ft", str(directory), data_signature=INSTRUCT_SIGNATURE))
    loader = dataloader_over(parquet, tokenizer, DROPPING_SEQUENCE_LENGTH, 1, padded=False)
    return RunDataloaders({"ft": loader}, [], tokenizer, {"ft": parquet}, DROPPING_SEQUENCE_LENGTH), parquet


@pytest.mark.timeout(60)
def test_a_source_without_a_usable_row_raises_instead_of_restarting(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """
    Item 2: every restart re-reads the same rows, so a source whose every row the collate drops used to spin
    forever with a frozen step counter. The finished full epoch without a sample is an error naming the cause.
    """

    loaders, _ = _instruct_source(tmp_path / "unusable", tokenizer, [LONG_PROMPT, LONG_PROMPT])
    with pytest.raises(RuntimeError, match="no usable sample in a full epoch") as raised:
        for _ in range(10):  # bounded: the third pull ends the epoch, and before the fix every pull after it hung
            loaders.next_train_batch("ft")
    message = str(raised.value)
    assert "'ft'" in message and "its 2 rows" in message
    assert "training_max_sequence_length + 1 = 17" in message and "Raise training_max_sequence_length" in message
    loaders.close()


def test_dropped_rows_warn_once_and_the_epoch_reports_them(
    tmp_path: Path, tokenizer: Tokenizer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    A source that drops SOME rows trains on the rest: one WARNING for the whole run and one INFO line per epoch.
    """

    prompts = [LONG_PROMPT, SHORT_PROMPT, LONG_PROMPT, SHORT_PROMPT]
    loaders, _ = _instruct_source(tmp_path / "some_dropped", tokenizer, prompts)
    with caplog.at_level(logging.DEBUG, logger="training"):
        batches = [loaders.next_train_batch("ft") for _ in range(8)]  # two epochs of four worker batches
    assert sum(len(batch.samples) for batch in batches) == 4  # the two short-prompt rows of each epoch
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].startswith("ft: 1 of 1 rows dropped")
    assert "training_max_sequence_length + 1 = 17" in warnings[0]
    epochs = [(record.levelno, record.getMessage()) for record in caplog.records if "epoch done" in record.getMessage()]
    assert epochs == [(logging.INFO, "ft: epoch done, 4 rows read, 2 samples, 2 dropped")]
    loaders.close()


def test_an_epoch_without_a_dropped_row_reports_at_debug(
    tokenizer: Tokenizer, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The common case is silent at INFO: a small source finishes an epoch every few pulls and has nothing to report.
    """

    loaders = RunDataloaders({"a": _tagged("a", 2)}, [], tokenizer, {})
    with caplog.at_level(logging.DEBUG, logger="training"):
        [loaders.next_train_batch("a") for _ in range(3)]
    epochs = [(record.levelno, record.getMessage()) for record in caplog.records if "epoch done" in record.getMessage()]
    assert epochs == [(logging.DEBUG, "a: epoch done, 2 rows read, 2 samples, 0 dropped")]
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.mark.timeout(60)
def test_a_resumed_epoch_without_a_sample_restarts_instead_of_raising(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """
    Only a FULL epoch must yield a sample: the first epoch after a resume starts at an offset and may hold nothing
    but dropped rows, which the next epoch over the whole range fixes on its own.
    """

    loaders, parquet = _instruct_source(tmp_path / "resumed", tokenizer, [SHORT_PROMPT, LONG_PROMPT])
    loaders.set_resume_offsets({"ft": 1})  # the resumed epoch holds the long-prompt row alone, and it is dropped
    assert loaders.next_train_batch("ft").samples == []
    assert len(loaders.next_train_batch("ft").samples) == 1  # the restart reads the whole range and yields
    assert parquet.resume_offset == 0
    loaders.close()


class _LosingLoader:
    """
    A train loader whose worker 'loses' every second batch: what torch's loader does when the worker could not hand
    a batch over (out of file descriptors) and then reported the end of its range.
    """

    num_workers = 0

    def __init__(self, loader: Iterable[WorkerBatch]) -> None:
        self._loader = loader

    def __iter__(self) -> Iterator[WorkerBatch]:
        for index, batch in enumerate(self._loader):
            if index % 2 == 0:
                yield batch


@pytest.mark.timeout(60)
def test_an_epoch_that_lost_worker_batches_raises_instead_of_training_on_the_rest(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    """
    The dataset knows how many rows the epoch holds for this rank; a loader that delivered fewer lost batches, and
    the error says so (rows delivered and owed, the file-descriptor cause and this process's limit) instead of
    blaming the collate or restarting over the same broken worker.
    """

    loaders, parquet = _instruct_source(tmp_path / "lost", tokenizer, [SHORT_PROMPT] * 4)
    loaders.train_loaders["ft"] = _LosingLoader(loaders.train_loaders["ft"])
    with pytest.raises(RuntimeError, match="delivered 2 of the 4 rows of this epoch") as raised:
        for _ in range(10):
            loaders.next_train_batch("ft")
    message = str(raised.value)
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert "'ft'" in message and "lost batches" in message and "Too many open files" in message
    assert f"this process may open {soft_limit}" in message
    assert parquet.epoch_rows(0) == 4
    loaders.close()


def test_a_resumed_epoch_owes_only_the_rows_after_the_offset(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """
    The rows an epoch owes start at the resume offset: a resumed epoch that delivers exactly its remainder is
    complete, and the next full epoch owes the whole range again.
    """

    loaders, _ = _instruct_source(tmp_path / "owed", tokenizer, [SHORT_PROMPT] * 3)
    loaders.set_resume_offsets({"ft": 2})
    loaders.next_train_batch("ft")
    assert loaders.epochs["ft"].expected_rows == 1
    loaders.next_train_batch("ft")  # the restart: a full epoch over the range
    assert loaders.epochs["ft"].expected_rows == 3 and loaders.epochs["ft"].full_epoch
    loaders.close()


def test_raise_open_file_limit_lifts_the_soft_limit_to_the_hard_one() -> None:
    """
    The train loaders' batches in flight can hold more descriptors than the usual soft limit of 1024 allows; the
    process lifts its own limit to the administrator's ceiling, or to `UNLIMITED_OPEN_FILES` when there is none.
    """

    original = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft_limit, hard_limit = original
    lowered = min(soft_limit, 1024)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard_limit))
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == lowered
        expected = max(lowered, UNLIMITED_OPEN_FILES) if hard_limit == resource.RLIM_INFINITY else hard_limit
        assert raise_open_file_limit() == expected
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == (expected, hard_limit)
        assert raise_open_file_limit() == expected  # idempotent
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original)


@pytest.mark.timeout(60)
def test_an_offset_of_a_whole_range_is_a_full_epoch(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """
    `set_resume_offset` takes the offset modulo the range, so an offset of exactly the range starts at row 0:
    that epoch is full and owes a sample.
    """

    loaders, _ = _instruct_source(tmp_path / "wrapped", tokenizer, [LONG_PROMPT, LONG_PROMPT])
    loaders.set_resume_offsets({"ft": 2})
    with pytest.raises(RuntimeError, match="no usable sample in a full epoch"):
        for _ in range(10):
            loaders.next_train_batch("ft")
    loaders.close()


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


def test_the_data_package_re_exports_nothing() -> None:
    """
    `training.data` must stay import-light: a re-export of the torch modules would load torch for everyone
    importing `dataset_resolver` (the framework-neutral module living in this package).
    """

    import training.data as pkg

    assert not hasattr(pkg, "__all__") and not hasattr(pkg, "collate_fn")
