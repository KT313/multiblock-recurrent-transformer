# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import itertools
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import pytest
import torch

from training.backend import SingleDeviceBackend
from training.data.collate import IGNORE_INDEX, find_multiple
from training.data.dataset_resolver import DataEntry, ResolvedDataset, resolve_dataset
from training.data.loader import (
    Batch,
    StageDataloaders,
    build_dataloader,
    build_stage_dataloaders,
    length_sorted_batches,
    sample_stage_batch,
)
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


def _batches(loader: Iterable[Batch], n: int) -> list[Batch]:
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
) -> Iterable[Batch]:
    return build_dataloader(
        entries,
        tokenizer,
        block_size,
        micro_batch_size,
        num_workers=num_workers,
        seed=seed,
        shard=shard,
        padding_multiple=padding_multiple,
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


# Mixtures with the instruct folder use a block size no instruct prompt can fill: `collate_fn` ends the loader on a
# batch without a single supervised label, and the processed instruct rows are only bounded by max_seq_length (256).
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
    directory, validation loaders restricted to the held-out rows of the split."""
    dataset: ResolvedDataset = resolve_dataset(tiny_settings)
    loaders = build_stage_dataloaders(tiny_settings, dataset, SingleDeviceBackend(device="cpu", precision="32"))
    assert isinstance(loaders, StageDataloaders)
    assert len(loaders.train_loaders) == len(loaders.val_loaders) == len(dataset.stages) == 3
    input_ids, labels, data_ids = loaders.next_train_batch(0)
    assert input_ids.shape[0] == tiny_settings.micro_batch_size and input_ids.shape == labels.shape
    # collate pads to a multiple of sequence_padding_multiple (capped at block_size + 1), then the label shift drops one
    assert (input_ids.shape[1] + 1) % 128 == 0 or input_ids.shape[1] == tiny_settings.block_size
    assert input_ids.shape[1] <= tiny_settings.block_size
    assert data_ids == ["pretrain_a-synthetic_pretrain"] * tiny_settings.micro_batch_size
    assert (labels == IGNORE_INDEX).any() or (input_ids != tokenizer.pad_id).all()
    _, _, val_ids = next(iter(loaders.val_loaders[2]))
    assert val_ids == ["finetune-synthetic_instruct"] * tiny_settings.micro_batch_size
    # the validation loaders read only the held-out first rows of the split (a single dataset is one finite epoch)
    for stage_idx, source in ((0, "synthetic_pretrain"), (2, "synthetic_instruct")):
        k = dataset.validation_rows[source]
        assert k >= 1 and len(list(loaders.val_loaders[stage_idx])) == math.ceil(k / tiny_settings.micro_batch_size)


# --- StageDataloaders / sample_stage_batch ---------------------------------------------------------------------------


def _tagged(tag: str, n: int) -> list[Batch]:
    """A finite 'loader' yielding n batches tagged with `tag`."""
    return [(torch.full((1, 2), i), torch.full((1, 2), i), [tag]) for i in range(n)]


def test_next_train_batch_cycles_on_exhaustion() -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 3), _tagged("s1", 2)], val_loaders=[])
    got = [sd.next_train_batch(0)[0][0, 0].item() for _ in range(7)]
    assert got == [0, 1, 2, 0, 1, 2, 0]
    assert sd.next_train_batch(1)[2] == ["s1"]
    assert sd._train_iterators[0] is not None and sd._train_iterators[1] is not None


def test_post_init_creates_one_slot_per_train_loader() -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 1), _tagged("s1", 1), _tagged("s2", 1)], val_loaders=[])
    assert sd._train_iterators == [None, None, None]
    assert StageDataloaders(train_loaders=[], val_loaders=[])._train_iterators == []


def test_iterators_are_lazy_and_independent() -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 3), _tagged("s1", 3)], val_loaders=[])
    assert sd._train_iterators == [None, None]
    sd.next_train_batch(1)
    assert sd._train_iterators[0] is None
    assert sd.next_train_batch(1)[0][0, 0].item() == 1
    assert sd.next_train_batch(0)[0][0, 0].item() == 0


def test_sample_stage_batch_outside_transition_is_current() -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 5), _tagged("s1", 5)], val_loaders=[])
    rng = random.Random(0)
    for p in (0.0, 0.5, 1.0):
        assert sample_stage_batch(sd, 1, None, p, rng)[2] == ["s1"]


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.8, 1.0])
def test_sample_stage_batch_bernoulli_frequency(progress: float) -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 5), _tagged("s1", 5)], val_loaders=[])
    rng = random.Random(123)
    n = 4000
    tags = Counter(sample_stage_batch(sd, 1, 0, progress, rng)[2][0] for _ in range(n))
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


def test_sample_stage_batch_bernoulli_exact_boundary() -> None:
    """Draw u; next stage iff u < progress (so u == progress stays on the previous stage)."""
    sd = StageDataloaders(train_loaders=[_tagged("s0", 9), _tagged("s1", 9)], val_loaders=[])
    rng = _ScriptedRandom([0.1, 0.5, 0.49999, 0.9, 0.0])
    tags = [sample_stage_batch(sd, 1, 0, 0.5, rng)[2][0] for _ in range(5)]
    assert tags == ["s1", "s0", "s1", "s0", "s1"]


def test_sample_stage_batch_consumes_rng_only_in_transition() -> None:
    sd = StageDataloaders(train_loaders=[_tagged("s0", 5), _tagged("s1", 5)], val_loaders=[])
    rng = random.Random(0)
    state = rng.getstate()
    sample_stage_batch(sd, 1, None, 0.5, rng)
    assert rng.getstate() == state


# --- length_sorted_batches ---------------------------------------------------------------------------------------------


def _padded_batch(lengths: list[int], width: int, pad: int, tag_start: int) -> Batch:
    """Micro-batch where row i has `lengths[i]` non-pad positions (values tag_start+i) then pad."""
    inp = torch.full((len(lengths), width), pad)
    labels = torch.full((len(lengths), width), -100)
    for i, n in enumerate(lengths):
        inp[i, :n] = tag_start + i
        labels[i, :n] = tag_start + i + 1000
    data_ids = [f"d{tag_start + i}" for i in range(len(lengths))]
    return inp, labels, data_ids


def _flatten(batches: Iterable[Batch]) -> list[tuple[list[int], list[int], str]]:
    out: list[tuple[list[int], list[int], str]] = []
    for inp, lab, ids in batches:
        for i in range(inp.shape[0]):
            n = int((lab[i] != -100).sum())  # compare the supervised prefix only: trailing padding may be trimmed
            out.append((inp[i, :n].tolist(), lab[i, :n].tolist(), ids[i]))
    return out


def test_length_sorted_regroups_shortest_first_per_window() -> None:
    pad = 0
    batches = [
        _padded_batch([7, 2], 8, pad, 10),
        _padded_batch([5, 8], 8, pad, 20),
        _padded_batch([1, 6], 8, pad, 30),  # second window starts here
        _padded_batch([3, 4], 8, pad, 40),
    ]
    out = list(length_sorted_batches(batches, micro_batch_size=2, accumulation_steps=2, ignore_index=-100))
    assert len(out) == 4
    # window 1 lengths: 7,2,5,8 -> sorted 2,5,7,8 ; window 2: 1,6,3,4 -> 1,3,4,6
    lengths = [[int((row != pad).sum()) for row in o[0]] for o in out]
    assert lengths == [[2, 5], [7, 8], [1, 3], [4, 6]]
    # every micro-batch is trimmed to its longest sample
    assert [o[0].shape for o in out] == [(2, 5), (2, 8), (2, 3), (2, 6)]
    assert [o[1].shape for o in out] == [(2, 5), (2, 8), (2, 3), (2, 6)]
    assert [o[2] for o in out] == [["d11", "d20"], ["d10", "d21"], ["d30", "d40"], ["d41", "d31"]]
    # exactly the same samples, labels travel with their inputs
    assert sorted(_flatten(out)) == sorted(_flatten(batches))


def test_length_sorted_trailing_partial_window() -> None:
    pad = 0
    batches = [_padded_batch([4, 1], 6, pad, 10), _padded_batch([3, 2], 6, pad, 20), _padded_batch([6, 5], 6, pad, 30)]
    out = list(length_sorted_batches(batches, micro_batch_size=2, accumulation_steps=2, ignore_index=-100))
    assert len(out) == 3
    assert [[int((r != pad).sum()) for r in o[0]] for o in out] == [[1, 2], [3, 4], [5, 6]]
    assert sorted(_flatten(out)) == sorted(_flatten(batches))


def test_length_sorted_uneven_split_yields_partial_micro_batch() -> None:
    pad = 0
    batches = [_padded_batch([3, 1, 2], 4, pad, 10)]
    out = list(length_sorted_batches(batches, micro_batch_size=2, accumulation_steps=1, ignore_index=-100))
    assert [o[0].shape[0] for o in out] == [2, 1]
    assert [o[2] for o in out] == [["d11", "d12"], ["d10"]]


def test_length_sorted_is_lazy_across_windows() -> None:
    pad = 0

    def gen() -> Iterator[Batch]:
        yield _padded_batch([2, 1], 4, pad, 10)
        yield _padded_batch([2, 1], 4, pad, 20)
        raise RuntimeError("should not be pulled before the first window is consumed")

    it = length_sorted_batches(gen(), micro_batch_size=2, accumulation_steps=2, ignore_index=-100)
    first = next(it)
    assert first[2] == ["d11", "d21"]
    next(it)
    with pytest.raises(RuntimeError):
        next(it)


def test_length_sorted_empty_input_yields_nothing() -> None:
    assert list(length_sorted_batches([], micro_batch_size=2, accumulation_steps=2, ignore_index=-100)) == []


def test_length_sorted_single_sample_window() -> None:
    batch = _padded_batch([3], 4, 0, 10)
    out = list(length_sorted_batches([batch], micro_batch_size=4, accumulation_steps=1, ignore_index=-100))
    assert len(out) == 1
    assert torch.equal(out[0][0], batch[0][:, :3]) and torch.equal(out[0][1], batch[1][:, :3]) and out[0][2] == batch[2]


def test_length_sorted_ties_keep_arrival_order() -> None:
    pad = 0
    batches = [_padded_batch([3, 3], 4, pad, 10), _padded_batch([3, 1], 4, pad, 20)]
    out = list(length_sorted_batches(batches, micro_batch_size=2, accumulation_steps=2, ignore_index=-100))
    assert [o[2] for o in out] == [["d21", "d10"], ["d11", "d20"]]


def test_length_sorted_on_real_loader_preserves_samples(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    loader = build_dataloader(entries[:1], tokenizer, 512, 4, padding_multiple=128)
    raw = _batches(loader, 4)
    out = list(length_sorted_batches(raw, micro_batch_size=4, accumulation_steps=2, ignore_index=-100))
    assert [o[0].shape[0] for o in out] == [4, 4, 4, 4]
    assert sorted(_flatten(out)) == sorted(_flatten(raw))


def test_length_sorted_sorts_collated_batches_by_true_length(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    """On real collated batches (pads already replaced by EOS) the sort uses the supervised length and trims."""
    loader = build_dataloader(entries[:1], tokenizer, 512, 4, padding_multiple=128)
    raw = _batches(loader, 2)  # rows are 64..384 words, so these batches carry real padding
    true_lens = [int((lab != -100).sum()) for b in raw for lab in b[1]]
    assert true_lens != sorted(true_lens), "fixture must start unsorted for the test to mean anything"
    out = list(length_sorted_batches(raw, micro_batch_size=4, accumulation_steps=2, ignore_index=-100, padding_multiple=128))
    assert [int((lab != -100).sum()) for o in out for lab in o[1]] == sorted(true_lens)
    for inp, lab, _ in out:
        longest = int((lab != -100).sum(dim=1).max())
        assert inp.shape[1] == lab.shape[1] == min(find_multiple(longest, 128), raw[0][0].shape[1])
    assert sum(o[0].numel() for o in out) < sum(b[0].numel() for b in raw)  # trimming saved positions


def test_length_sorted_padding_multiple_rounds_width_up() -> None:
    batches = [_padded_batch([5, 2], 16, 0, 10), _padded_batch([9, 1], 16, 0, 20)]
    out = list(length_sorted_batches(batches, micro_batch_size=2, accumulation_steps=2, ignore_index=-100, padding_multiple=4))
    # chunks: lengths [1, 2] -> width 4 ; [5, 9] -> width 12
    assert [o[0].shape for o in out] == [(2, 4), (2, 12)]
    assert sorted(_flatten(out)) == sorted(_flatten(batches))


def test_length_sorted_loss_is_unchanged_by_trimming(tokenizer: Tokenizer, entries: list[DataEntry]) -> None:
    """Trimming removes only ignore-index positions, so a per-token loss over the world batch is identical."""
    loader = build_dataloader(entries[:1], tokenizer, 512, 4, padding_multiple=128)
    raw = _batches(loader, 2)
    out = list(length_sorted_batches(raw, micro_batch_size=4, accumulation_steps=2, ignore_index=-100, padding_multiple=128))

    def supervised(batches: list[Batch]) -> list[tuple[int, list[int]]]:
        return sorted((hash(d), lab[lab != -100].tolist()) for _, labs, ids in batches for lab, d in zip(labs, ids))

    assert supervised(out) == supervised(raw)


def test_package_exports_resolve() -> None:
    import training.data as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None
