# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.layout: pure path arithmetic.
"""

from pathlib import Path

import pytest

from data_preparation.layout import (
    INSTRUCT_PROCESSED_COLUMNS,
    PRETRAIN_PROCESSED_COLUMNS,
    PROCESSED_COLUMNS,
    DatasetLayout,
    processed_columns,
)


def test_default_root_is_dataset() -> None:
    assert DatasetLayout().root == Path("dataset")


def test_raw_and_processed_dirs_are_separate_trees() -> None:
    layout = DatasetLayout(Path("/d"))
    assert layout.raw_dir("fineweb") == Path("/d/sources/fineweb/raw")
    assert layout.processed_dir("fineweb") == Path("/d/processed/fineweb")
    # derived data never lives inside the source folder
    assert not layout.processed_dir("fineweb").is_relative_to(layout.raw_dir("fineweb").parent)


def test_no_stage_argument_and_no_removed_dirs() -> None:
    layout = DatasetLayout()
    with pytest.raises(TypeError):
        layout.raw_dir("x", "raw")  # type: ignore[call-arg]  # the stage argument is gone on purpose
    for removed in ("source_dir", "validation_dir", "instruct_mixture_dir"):
        assert not hasattr(layout, removed)


def test_tokenizer_benchmark_hub_index_dirs() -> None:
    layout = DatasetLayout(Path("root"))
    assert layout.tokenizer_dir("llama-32k") == Path("root/tokenizers/llama-32k")
    assert layout.benchmark_cache_dir() == Path("root/benchmarks")
    assert layout.hub_index_dir() == Path("root/hub_index")


def test_processed_columns_per_kind() -> None:
    assert processed_columns("pretrain") == PRETRAIN_PROCESSED_COLUMNS == ("text", "source", "tokens", "hash")
    assert processed_columns("instruct") == INSTRUCT_PROCESSED_COLUMNS == ("instruction", "input", "output", "tokens", "hash")
    assert PROCESSED_COLUMNS is PRETRAIN_PROCESSED_COLUMNS
    assert all("hash" in columns and "tokens" in columns for columns in (PRETRAIN_PROCESSED_COLUMNS, INSTRUCT_PROCESSED_COLUMNS))
    with pytest.raises(ValueError, match="unknown source kind"):
        processed_columns("validation")


def test_frozen_and_hashable() -> None:
    layout = DatasetLayout(Path("a"))
    assert layout == DatasetLayout(Path("a")) and hash(layout) == hash(DatasetLayout(Path("a")))
    with pytest.raises(AttributeError):
        layout.root = Path("b")  # type: ignore[misc]  # frozen dataclass: assignment must fail
