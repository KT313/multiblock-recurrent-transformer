# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.schema.layout: pure path arithmetic."""

from pathlib import Path

import pytest

from data_preparation.lib.schema.layout import INSTRUCT_MIXTURE_SPLITS, SOURCE_STAGES, DatasetLayout


def test_default_root_is_dataset() -> None:
    assert DatasetLayout().root == Path("dataset")


def test_source_dirs() -> None:
    layout = DatasetLayout(Path("/d"))
    assert layout.source_dir("fineweb", "raw") == Path("/d/sources/fineweb/raw")
    assert layout.source_dir("fineweb", "filtered") == Path("/d/sources/fineweb/filtered")
    assert layout.source_dir("fineweb", "processed") == Path("/d/sources/fineweb/processed")
    assert layout.validation_dir("fineweb_val") == Path("/d/sources/fineweb_val/validation")
    assert SOURCE_STAGES == ("raw", "filtered", "processed")


def test_instruct_mixture_tokenizer_benchmark_dirs() -> None:
    layout = DatasetLayout(Path("root"))
    assert layout.instruct_mixture_dir("crow", "flan", "train") == Path("root/instruct_mixtures/crow/flan/train")
    assert layout.instruct_mixture_dir("crow", "flan", "validation") == Path("root/instruct_mixtures/crow/flan/validation")
    assert layout.tokenizer_dir("llama-32k") == Path("root/tokenizers/llama-32k")
    assert layout.benchmark_cache_dir() == Path("root/benchmarks")
    assert layout.hub_index_dir() == Path("root/hub_index")
    assert INSTRUCT_MIXTURE_SPLITS == ("train", "validation")


def test_unknown_stage_or_split_rejected() -> None:
    layout = DatasetLayout()
    with pytest.raises(ValueError, match="unknown source stage"):
        layout.source_dir("x", "validation")
    with pytest.raises(ValueError, match="unknown mixture split"):
        layout.instruct_mixture_dir("c", "m", "val")


def test_frozen_and_hashable() -> None:
    layout = DatasetLayout(Path("a"))
    assert layout == DatasetLayout(Path("a")) and hash(layout) == hash(DatasetLayout(Path("a")))
    with pytest.raises(AttributeError):
        layout.root = Path("b")  # type: ignore[misc]  # frozen dataclass: assignment must fail
