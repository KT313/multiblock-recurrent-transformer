# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.prepare_fineweb_validation with a stubbed Hub."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from typing import cast
import pytest

from data_preparation.lib import prepare_fineweb_validation as pfv
from data_preparation.lib.common import list_parquet_files


def test_parser_only_has_common_args() -> None:
    args = pfv.build_parser().parse_args(["--dataset_dir", "/d", "--cache_dir", "/c"])
    assert str(args.dataset_dir) == "/d" and str(args.cache_dir) == "/c"
    assert set(vars(args)) == {"dataset_dir", "cache_dir"}
    assert pfv.N_VAL == 50_000 and pfv.SHARD_SIZE == 10_000


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hf_datasets: ModuleType, n_rows: int, n_val: int, shard_size: int
) -> tuple[list[Path], pa.Table]:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_load_dataset(path: str, name: str | None = None, split: str | None = None, **kwargs: Any) -> Any:
        calls.append((path, name, split))
        return hf_datasets.Dataset.from_dict(
            {"text": [f"doc {i}" for i in range(n_rows)], "id": list(range(n_rows)), "score": [0.5] * n_rows}
        )

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(pfv, "N_VAL", n_val)
    monkeypatch.setattr(pfv, "SHARD_SIZE", shard_size)
    monkeypatch.setattr(sys, "argv", ["prepare_fineweb_validation", "--dataset_dir", str(tmp_path)])
    pfv.main()
    assert calls == [("HuggingFaceFW/fineweb-edu", "sample-10BT", "train")]
    files = list_parquet_files(tmp_path / "fineweb-edu" / "validation")
    return files, pa.concat_tables([pq.read_table(f) for f in files])


def test_main_output_layout_and_columns(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, table = _run(monkeypatch, tmp_path, hf_datasets, n_rows=60, n_val=10, shard_size=4)
    assert [f.name for f in files] == ["data-00000.parquet", "data-00001.parquet", "data-00002.parquet"]
    assert [pq.read_metadata(f).num_rows for f in files] == [4, 4, 2]
    assert table.column_names == ["text", "id", "score"]  # original columns are kept
    assert table.num_rows == 10
    ids = cast(list[int], table["id"].to_pylist())
    assert len(set(ids)) == 10 and sorted(ids) != ids  # a shuffled hold-out, not the first N rows


def test_main_is_deterministic(tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _, a = _run(monkeypatch, tmp_path / "a", hf_datasets, n_rows=60, n_val=10, shard_size=100)
    _, b = _run(monkeypatch, tmp_path / "b", hf_datasets, n_rows=60, n_val=10, shard_size=100)
    assert a["id"].to_pylist() == b["id"].to_pylist()


def test_main_fails_when_source_smaller_than_holdout(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="test_size=10"):
        _run(monkeypatch, tmp_path, hf_datasets, n_rows=5, n_val=10, shard_size=100)
    assert not (tmp_path / "fineweb-edu").exists()
