# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, TypeVar

import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from training.backend import SingleDeviceBackend
from training.data.collate import IGNORE_INDEX, Batch, Sample, collate_samples
from training.data.dataset_resolver import DataEntry, ResolvedDataset, resolve_dataset
from training.data.loader import (
    SampleBatch,
    StageDataloaders,
    build_dataloader,
    build_stage_dataloaders,
    sample_length,
    sample_stage_batch,
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
    block_size: int = 64,
    padded: bool = True,
) -> DataLoader[Row]:
    return build_dataloader(
        entries,
        tokenizer,
        block_size,
        micro_batch_size,
        num_workers=num_workers,
        seed=seed,
        shard=shard,
        padding_multiple=padding_multiple,
        padded=padded,
    )


def _rows_in(directory: Path) -> int:
    """Rows of the `data-*.parquet` shards of a processed folder (parquet footers only)."""
    return sum(pq.read_metadata(path).num_rows for path in sorted(directory.glob("data-*.parquet")))


def _same(a: list[Batch], b: list[Batch]) -> bool:
    return all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b))


# --- DataEntry / build_dataloader ------------------------------------------------------------------------------------


def test_entry_defaults_read_the_whole_text_column(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    entry = DataEntry("p", str(tiny_pretrain_dir))
    assert (entry.weight, entry.data_signature, entry.skip_rows, entry.max_rows) == (1.0, None, 0, None)
    assert len(list(_loader([entry], tokenizer, 1))) == _rows_in(tiny_pretrain_dir)  # None = the default text signature


def test_row_range_reaches_the_dataset(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    """`skip_rows` / `max_rows` of an entry restrict its dataset (the resolver's validation split): the validation
    range and the training range of one folder are disjoint and together are the folder, in order."""
    directory = str(tiny_pretrain_dir)
    total, k = _rows_in(tiny_pretrain_dir), 3

    def rows(entry: DataEntry) -> list[tuple[int, ...]]:
        return [tuple(b[0][0].tolist()) for b in _loader([entry], tokenizer, 1)]

    val = rows(DataEntry("val", directory, max_rows=k))
    train = rows(DataEntry("train", directory, skip_rows=k))
    assert len(val) == k and len(train) == total - k
    assert not set(val) & set(train)
    assert val + train == rows(DataEntry("all", directory))
    both = _loader([DataEntry("val", directory, weight=0.5, max_rows=k), DataEntry("train", directory, weight=0.5, skip_rows=k)], tokenizer, 1)
    tags = Counter(b[2][0] for b in _batches(both, 100))
    assert set(tags) == {"val", "train"}  # a mixture keeps every member's range


def test_pin_memory_is_off_unless_requested(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    entries = [DataEntry("p", str(tiny_pretrain_dir))]
    assert build_dataloader(entries, tokenizer, 64, 2).pin_memory is False
    assert build_dataloader(entries, tokenizer, 64, 2, pin_memory=True).pin_memory is True


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
# supervised label; the processed instruct rows are only bounded by max_seq_length (256).
MIXTURE_BLOCK_SIZE = 128


def test_mixture_loader_mixes_and_is_infinite(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    loader = _loader(entries, tokenizer, 2, seed=0, block_size=MIXTURE_BLOCK_SIZE)
    ids = Counter(itertools.chain.from_iterable(b[2] for b in _batches(loader, 200)))
    assert set(ids) == {"pre", "ft"}
    assert ids["pre"] / 400 == pytest.approx(0.7, abs=0.06)


def test_loader_deterministic_under_seed(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    a = _batches(_loader(entries, tokenizer, 2, seed=5, block_size=MIXTURE_BLOCK_SIZE), 10)
    b = _batches(_loader(entries, tokenizer, 2, seed=5, block_size=MIXTURE_BLOCK_SIZE), 10)
    c = _batches(_loader(entries, tokenizer, 2, seed=6, block_size=MIXTURE_BLOCK_SIZE), 10)
    assert _same(a, b)
    assert not _same(a, c)


def test_workers_zero_and_two_identical_with_micro_batch_one(tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path) -> None:
    """Rows are dealt round-robin to workers and DataLoader collects worker batches round-robin, so with
    micro_batch_size=1 the two loaders yield the very same sequence."""
    a = list(_loader(entries[:1], tokenizer, 1, num_workers=0))
    b = list(_loader(entries[:1], tokenizer, 1, num_workers=2))
    assert len(a) == len(b) == _rows_in(tiny_pretrain_dir)
    assert _same(a, b)


def test_workers_two_micro_batch_gt_one_regroups_rows(tokenizer: Tokenizer, entries: list[DataEntry], tiny_pretrain_dir: Path) -> None:
    """With micro_batch_size>1 each worker batches *its* rows (0,2,4.. / 1,3,5..), so batches differ from
    num_workers=0 in composition but cover exactly the same rows over an epoch."""
    a = list(_loader(entries[:1], tokenizer, 2, num_workers=0))
    b = list(_loader(entries[:1], tokenizer, 2, num_workers=2))
    assert len(a) == len(b) == math.ceil(_rows_in(tiny_pretrain_dir) / 2)

    def rows(batches: list[Batch]) -> list[tuple[int, ...]]:
        return sorted(tuple(r.tolist()) for x in batches for r in x[0])

    assert rows(a) == rows(b)
    assert not _same(a, b)
    assert [tuple(r.tolist()) for r in b[0][0]] == [tuple(a[0][0][0].tolist()), tuple(a[1][0][0].tolist())]


def test_workers_two_mixture_is_deterministic(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    a = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, block_size=MIXTURE_BLOCK_SIZE), 12)
    b = _batches(_loader(entries, tokenizer, 2, seed=1, num_workers=2, block_size=MIXTURE_BLOCK_SIZE), 12)
    assert _same(a, b)


def test_unusable_rows_are_dropped_without_ending_the_loader(tokenizer: Tokenizer, tiny_instruct_dir: Path) -> None:
    """Regression (T-M4): a row with no supervised label used to raise `StopIteration` out of the collate function,
    which torch's worker loop reads as 'this worker is done' and the single-process loop as 'restart at row 0'. At
    `block_size` 16 most instruct prompts alone fill the window; the loader still walks its whole epoch."""
    rows = list(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE)))
    kept = len(collate_samples(rows, tokenizer, block_size=16))
    assert 0 < kept < len(rows), "the fixture must drop some rows and keep others"
    entry = DataEntry("ft", str(tiny_instruct_dir), data_signature=INSTRUCT_SIGNATURE)
    for num_workers in (0, 2):
        loader = build_dataloader([entry], tokenizer, 16, 4, num_workers=num_workers, padded=False)
        assert sum(len(batch) for batch in loader) == kept


def test_shard_passed_to_datasets(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    full = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1)]
    r0 = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1, shard=(0, 2))]
    r1 = [tuple(b[0][0].tolist()) for b in _loader(entries[:1], tokenizer, 1, shard=(1, 2))]
    assert r0 == full[0::2] and r1 == full[1::2]


# --- build_stage_dataloaders ------------------------------------------------------------------------------------------


@pytest.fixture
def tiny_settings(tmp_path: Path, tiny_dataset_dir: Path) -> Settings:
    return parse_settings(["--config", str(TINY_YAML), "--dataset_dir", str(tiny_dataset_dir), "--out_dir", str(tmp_path / "out")])


def test_build_stage_dataloaders(tiny_settings: Settings, tokenizer: Tokenizer) -> None:
    """One train and one validation loader per stage of the tiny dataset, tokenizer loaded from the resolved
    directory, train loaders unpadded, validation loaders padded and restricted to the held-out rows of the split."""
    dataset: ResolvedDataset = resolve_dataset(tiny_settings)
    loaders = build_stage_dataloaders(tiny_settings, dataset, SingleDeviceBackend(device="cpu", precision="32"))
    assert isinstance(loaders, StageDataloaders)
    assert len(loaders.train_loaders) == len(loaders.val_loaders) == len(dataset.stages) == 3
    assert loaders.tokenizer.path == Path(dataset.tokenizer_dir)
    samples = loaders.next_train_batch(0)
    assert len(samples) == tiny_settings.micro_batch_size
    assert [s[2] for s in samples] == ["pretrain_a-synthetic_pretrain"] * tiny_settings.micro_batch_size
    for input_ids, labels, _ in samples:  # unpadded: the true token count, capped at block_size + 1
        assert input_ids.shape == labels.shape and 0 < input_ids.shape[0] <= tiny_settings.block_size + 1
    input_ids, labels, _ = world_batch_micro_batches(
        samples, tiny_settings.micro_batch_size, tokenizer, tiny_settings.block_size, sort_by_length=True,
        padding_multiple=tiny_settings.sequence_padding_multiple,
    )[0]
    # padding rounds up to sequence_padding_multiple (capped at block_size + 1), then the label shift drops one
    assert (input_ids.shape[1] + 1) % 128 == 0 or input_ids.shape[1] == tiny_settings.block_size
    assert input_ids.shape[1] <= tiny_settings.block_size
    assert (labels == IGNORE_INDEX).any() or (input_ids != tokenizer.pad_id).all()
    _, _, val_ids = next(iter(loaders.val_loaders[2]))
    assert val_ids == ["finetune-synthetic_instruct"] * tiny_settings.micro_batch_size
    # the validation loaders read only the held-out first rows of the split (a single dataset is one finite epoch)
    for stage_idx, source in ((0, "synthetic_pretrain"), (2, "synthetic_instruct")):
        k = dataset.validation_rows[source]
        assert k >= 1 and len(list(loaders.val_loaders[stage_idx])) == math.ceil(k / tiny_settings.micro_batch_size)


# --- StageDataloaders / sample_stage_batch ---------------------------------------------------------------------------


def _tagged(tag: str, n: int) -> list[SampleBatch]:
    """A finite 'loader' yielding n one-sample worker batches tagged with `tag`."""
    return [[(torch.full((2,), i), torch.full((2,), i), tag)] for i in range(n)]


def _first(batch: SampleBatch) -> int:
    """The counter value of a `_tagged` worker batch."""
    return int(batch[0][0][0])


def test_next_train_batch_cycles_on_exhaustion(tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 3), _tagged("s1", 2)], [], tokenizer)
    assert [_first(sd.next_train_batch(0)) for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert [s[2] for s in sd.next_train_batch(1)] == ["s1"]
    assert sd._train_iterators[0] is not None and sd._train_iterators[1] is not None


def test_post_init_creates_one_slot_per_train_loader(tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 1), _tagged("s1", 1), _tagged("s2", 1)], [], tokenizer)
    assert sd._train_iterators == [None, None, None]
    assert StageDataloaders([], [], tokenizer)._train_iterators == []


def test_iterators_are_lazy_and_independent(tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 3), _tagged("s1", 3)], [], tokenizer)
    assert sd._train_iterators == [None, None]
    sd.next_train_batch(1)
    assert sd._train_iterators[0] is None
    assert _first(sd.next_train_batch(1)) == 1
    assert _first(sd.next_train_batch(0)) == 0


def test_train_datasets_reaches_through_a_single_dataset_and_a_mixture(
    tokenizer: Tokenizer, entries: list[DataEntry]
) -> None:
    loaders = StageDataloaders([_loader(entries[:1], tokenizer, 2, padded=False), _loader(entries, tokenizer, 2, padded=False)], [], tokenizer)
    assert [d.prefix for d in loaders.train_datasets(0)] == ["pre"]
    assert [d.prefix for d in loaders.train_datasets(1)] == ["pre", "ft"]
    assert StageDataloaders([_tagged("s0", 1)], [], tokenizer).train_datasets(0) == []  # a plain list of batches


def test_set_and_clear_resume_offsets(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    loaders = StageDataloaders([_loader(entries, tokenizer, 2, padded=False)], [], tokenizer)
    loaders.set_resume_offsets({"pre": 3, "gone": 9})  # a prefix no dataset has is ignored
    assert [d.resume_offset for d in loaders.train_datasets(0)] == [3, 0]
    loaders.clear_resume_offsets(0)
    assert [d.resume_offset for d in loaders.train_datasets(0)] == [0, 0]


def test_resume_offset_is_dropped_when_the_loader_restarts(tokenizer: Tokenizer, tiny_pretrain_dir: Path) -> None:
    """The first epoch after a resume starts at the offset; once it ends, the loader reads its whole range again."""
    entry = DataEntry("pre", str(tiny_pretrain_dir))
    total = _rows_in(tiny_pretrain_dir)
    loaders = StageDataloaders([build_dataloader([entry], tokenizer, 64, 1, padded=False)], [], tokenizer)
    loaders.set_resume_offsets({"pre": total - 2})
    first_epoch = [loaders.next_train_batch(0)[0][2] for _ in range(2)]
    assert first_epoch == ["pre", "pre"] and loaders.train_datasets(0)[0].resume_offset == 0
    assert len([loaders.next_train_batch(0) for _ in range(total)]) == total  # the restart reads every row


def test_sample_stage_batch_outside_transition_is_current(tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 5), _tagged("s1", 5)], [], tokenizer)
    rng = random.Random(0)
    for p in (0.0, 0.5, 1.0):
        assert [s[2] for s in sample_stage_batch(sd, 1, None, p, rng)] == ["s1"]


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.8, 1.0])
def test_sample_stage_batch_bernoulli_frequency(progress: float, tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 5), _tagged("s1", 5)], [], tokenizer)
    rng = random.Random(123)
    n = 4000
    tags = Counter(sample_stage_batch(sd, 1, 0, progress, rng)[0][2] for _ in range(n))
    assert tags["s1"] / n == pytest.approx(progress, abs=0.03)
    if progress in (0.0, 1.0):
        assert len(tags) == 1


class _ScriptedRandom(random.Random):
    """random() returns a fixed script; lets the >= boundary of the Bernoulli draw be checked exactly."""

    def __init__(self, values: list[float]) -> None:
        super().__init__(0)
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def test_sample_stage_batch_bernoulli_exact_boundary(tokenizer: Tokenizer) -> None:
    """Draw u; next stage iff u < progress (so u == progress stays on the previous stage)."""
    sd = StageDataloaders([_tagged("s0", 9), _tagged("s1", 9)], [], tokenizer)
    rng = _ScriptedRandom([0.1, 0.5, 0.49999, 0.9, 0.0])
    tags = [sample_stage_batch(sd, 1, 0, 0.5, rng)[0][2] for _ in range(5)]
    assert tags == ["s1", "s0", "s1", "s0", "s1"]


def test_sample_stage_batch_consumes_rng_only_in_transition(tokenizer: Tokenizer) -> None:
    sd = StageDataloaders([_tagged("s0", 5), _tagged("s1", 5)], [], tokenizer)
    rng = random.Random(0)
    state = rng.getstate()
    sample_stage_batch(sd, 1, None, 0.5, rng)
    assert rng.getstate() == state


# --- world_batch_micro_batches ----------------------------------------------------------------------------------------

BLOCK = 64  # cap of the fake-sample tests: block_size + 1 = 65 tokens


def _sample(length: int, tag: str) -> Sample:
    """An unpadded sample of `length` valid (non-pad, in-vocab) tokens; every position is supervised."""
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
    """The point of assembling before padding: a micro-batch of short rows is no longer widened because some other
    micro-batch of the same world batch happened to contain a long row."""
    short = [_sample(5, "a"), _sample(6, "b")]
    long = [_sample(60, "c"), _sample(61, "d")]
    alone = _split(tokenizer, short, micro_batch_size=2, sort=True, multiple=8)
    together = _split(tokenizer, short + long, micro_batch_size=2, sort=True, multiple=8)
    assert _widths(alone) == [(2, 7)] and _widths(together)[0] == (2, 7)


def test_world_batch_on_real_loader_preserves_every_sample(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    loader = build_dataloader(entries[:1], tokenizer, 512, 4, padding_multiple=128, padded=False)
    worker_batches: list[SampleBatch] = _batches(loader, 2)
    samples = [s for batch in worker_batches for s in batch]
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
    """An instruct-shaped sample: `prompt` masked positions (pad id in the labels), then `answer` supervised ones."""
    ids = torch.full((prompt + answer,), 3, dtype=torch.long)
    labels = ids.clone()
    labels[:prompt] = pad_id
    return ids, labels, tag


def test_world_batch_keeps_every_supervised_label_of_prompt_masked_rows(tokenizer: Tokenizer) -> None:
    """An instruct row's labels sit at the END of the row, so the width must come from the full length; a width
    derived from the count of supervised labels would cut the answers off long-prompt rows."""
    pad = tokenizer.pad_id
    samples = [_prompt_masked_sample(200, 12, "a", pad), _prompt_masked_sample(150, 30, "b", pad)]
    out = world_batch_micro_batches(samples, 2, tokenizer, 255, sort_by_length=True, padding_multiple=128)
    assert _widths(out) == [(2, 255)]
    # the shift drops the first label of each row; both rows keep every supervised position they had
    assert sorted(int((lab != IGNORE_INDEX).sum()) for _, labs, _ in out for lab in labs) == [12, 30]


def test_world_batch_keeps_the_labels_of_real_instruct_rows(tokenizer: Tokenizer, tiny_instruct_dir: Path) -> None:
    rows = list(itertools.islice(iter(ParquetTextDataset(tiny_instruct_dir, "ft", INSTRUCT_SIGNATURE)), 8))
    samples = collate_samples(rows, tokenizer, block_size=255)
    expected = sorted(int((lab[1:] != tokenizer.pad_id).sum()) for _, lab, _ in samples)
    out = world_batch_micro_batches(samples, 4, tokenizer, 255, sort_by_length=True, padding_multiple=128)
    assert len(samples) == 8 and expected[0] > 0
    assert sorted(int((lab != IGNORE_INDEX).sum()) for _, labs, _ in out for lab in labs) == expected


def test_the_data_package_re_exports_nothing() -> None:
    """`training.data` must stay import-light: a re-export of the torch modules would load torch for everyone
    importing `dataset_resolver` (the framework-neutral module living in this package)."""
    import training.data as pkg

    assert not hasattr(pkg, "__all__") and not hasattr(pkg, "collate_fn")
