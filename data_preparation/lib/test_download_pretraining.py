# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.download_pretraining: pure helpers, per-source handlers and the CLI with a stubbed
Hub."""

import sys
from types import ModuleType
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typing import Any

from data_preparation.lib import download_pretraining as dp
from data_preparation.lib.common import list_parquet_files

# --- pure helpers ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_rows", "target", "expected"),
    [
        (3, 7, [0, 1, 2, 0, 1, 2, 0]),
        (3, 3, [0, 1, 2]),
        (3, 2, [0, 1]),
        (3, 0, []),
        (1, 4, [0, 0, 0, 0]),
    ],
)
def test_repeat_indices(num_rows: int, target: int, expected: object) -> None:
    assert dp.repeat_indices(num_rows, target) == expected


def test_repeat_indices_gsm8k_budget() -> None:
    idx = dp.repeat_indices(7473, 550_000)
    assert len(idx) == 550_000 and idx[0] == 0 and idx[7473] == 0 and max(idx) == 7472


def test_format_gsm8k() -> None:
    row = dp.format_gsm8k({"question": "2+2?", "answer": "4"})
    assert row == {"text": "Question: 2+2?\n\nAnswer: 4", "source": "gsm8k"}


def test_iter_language_filters_and_limits() -> None:
    rows = [{"language": l, "code": i} for i, l in enumerate(["Python", "GO", "Python", "Python", "GO"])]
    assert [r["code"] for r in dp.iter_language(iter(rows), "Python", 2)] == [0, 2]
    assert [r["code"] for r in dp.iter_language(iter(rows), "GO", 10)] == [1, 4]
    assert list(dp.iter_language(iter(rows), "Rust", 10)) == []
    # stops consuming the stream once the limit is reached
    stream = iter(rows)
    list(dp.iter_language(stream, "Python", 1))
    assert next(stream)["code"] == 2


def test_datasets_table_matches_readme() -> None:
    names = [c["name"] for c in dp.DATASETS]
    assert len(names) == len(set(names)) == 19
    by_name = {c["name"]: c for c in dp.DATASETS}
    # the budgets documented in data_preparation/README.md ("Sources and budgets")
    readme_budgets = {
        "fineweb_edu": 9_000_000, "wikipedia": 1_700_000, "books_gutenberg": 600_000, "peso": 600_000,
        "arxiv": 400_000, "openwebmath": 2_000_000, "tinygsm": 1_600_000, "algebraic_stack": 450_000,
        "gsm8k": 550_000, "github_code_clean_python": 3_200_000, "github_code_clean_javascript": 2_200_000,
        "github_code_clean_typescript": 1_100_000, "github_code_clean_java": 1_100_000,
        "github_code_clean_cpp": 900_000, "github_code_clean_go": 800_000, "github_code_clean_rust": 550_000,
        "github_code_clean_shell": 450_000, "github_code_clean_sql": 350_000, "github_code_clean_html": 350_000,
    }  # fmt: skip
    assert {name: c["target_samples"] for name, c in by_name.items()} == readme_budgets
    assert by_name["gsm8k"]["handler"] == "gsm8k"
    github = [c for c in dp.DATASETS if "language" in c]
    assert len(github) == 10 and all(c["name"].startswith("github_code_clean_") for c in github)
    assert all("hf_dataset" not in c and "handler" not in c for c in github)
    assert dp.SHARD_SIZE == 100_000


def test_parser_defaults() -> None:
    args = dp.build_parser().parse_args([])
    assert args.parallel == 1 and args.datasets is None
    args = dp.build_parser().parse_args(["--parallel", "3", "--datasets", "gsm8k", "peso"])
    assert args.parallel == 3 and args.datasets == ["gsm8k", "peso"]


# --- handlers with a stubbed Hub ------------------------------------------------------------------------------------


@pytest.fixture
def stub_hub(
    hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def fake_load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((path, args, kwargs))
        if path == dp.GITHUB_CODE_DATASET:
            assert kwargs["streaming"] is True
            langs = ["Python", "GO", "Python", "Rust", "Python", "Python"]
            return iter([{"code": f"c{i}", "language": l, "path": f"f{i}"} for i, l in enumerate(langs)])
        if path == "gsm8k":
            return hf_datasets.Dataset.from_dict({"question": ["q1", "q2", "q3"], "answer": ["a1", "a2", "a3"]})
        if path == "failing/source":
            raise ConnectionError("offline")
        split = kwargs.get("split", "train")
        n = int(split[len("train[:") : -1]) if split.startswith("train[:") else 5
        n = min(n, 12)
        return hf_datasets.Dataset.from_dict({"text": [f"t{i}" for i in range(n)], "extra": list(range(n))})

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(dp, "SHARD_SIZE", 4)
    return calls


def _table(out_dir: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(f) for f in list_parquet_files(out_dir, "shard")])


def test_download_sliced_keeps_original_columns(
    tmp_path: Path, stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]]
) -> None:
    cfg = {"name": "src", "hf_dataset": "some/ds", "target_samples": 10, "load_kwargs": {"name": "cfg"}}
    shards = dp.download_sliced(cfg, tmp_path)
    assert shards == 3
    assert stub_hub == [("some/ds", (), {"split": "train[:10]", "name": "cfg"})]
    table = _table(tmp_path)
    assert table.column_names == ["text", "extra"] and table.num_rows == 10
    assert [f.name for f in list_parquet_files(tmp_path, "shard")] == [f"shard-{i:05d}.parquet" for i in range(3)]


def test_download_gsm8k_repeats_to_target(
    tmp_path: Path, stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]]
) -> None:
    cfg = {"name": "gsm8k", "hf_dataset": "gsm8k", "target_samples": 7, "load_kwargs": {"name": "main"}}
    shards = dp.download_gsm8k(cfg, tmp_path)
    assert shards == 2
    table = _table(tmp_path)
    assert table.column_names == ["text", "source"]
    assert table["text"].to_pylist() == [f"Question: q{i}\n\nAnswer: a{i}" for i in (1, 2, 3, 1, 2, 3, 1)]
    assert set(table["source"].to_pylist()) == {"gsm8k"}


def test_download_github_code_filters_language(
    tmp_path: Path, stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    cfg = {"name": "github_code_clean_python", "language": "Python", "target_samples": 3}
    shards = dp.download_github_code(cfg, tmp_path)
    assert shards == 1
    assert stub_hub[0][2] == {"split": "train", "streaming": True, "token": "secret"}
    table = _table(tmp_path)
    assert table["code"].to_pylist() == ["c0", "c2", "c4"] and set(table["language"].to_pylist()) == {"Python"}


def test_download_github_code_without_token_passes_none(
    tmp_path: Path, stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = {"name": "github_code_clean_go", "language": "GO", "target_samples": 5}
    assert dp.download_github_code(cfg, tmp_path) == 1
    assert stub_hub[0][2] == {"split": "train", "streaming": True, "token": None}
    assert _table(tmp_path)["code"].to_pylist() == ["c1"]  # only one GO row in the stub stream


def test_save_hf_dataset_uses_shard_prefix(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dp, "SHARD_SIZE", 2)
    ds = hf_datasets.Dataset.from_dict({"text": ["a", "b", "c"]}).select([2, 0, 1])
    assert dp._save_hf_dataset(ds, tmp_path) == 2
    assert [f.name for f in list_parquet_files(tmp_path, "shard")] == ["shard-00000.parquet", "shard-00001.parquet"]
    assert _table(tmp_path)["text"].to_pylist() == ["c", "a", "b"]


def test_download_dataset_dispatch_and_failure(
    tmp_path: Path, stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]], capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    assert dp.download_dataset({"name": "gsm8k", "hf_dataset": "gsm8k", "target_samples": 2, "load_kwargs": {},
                                "handler": "gsm8k"}, raw)  # fmt: skip
    assert dp.download_dataset({"name": "github_code_clean_go", "language": "GO", "target_samples": 5}, raw)
    assert dp.download_dataset({"name": "plain", "hf_dataset": "x/y", "target_samples": 3, "load_kwargs": {}}, raw)
    assert not dp.download_dataset(
        {"name": "broken", "hf_dataset": "failing/source", "target_samples": 3, "load_kwargs": {}}, raw
    )
    assert _table(raw / "gsm8k").num_rows == 2
    assert _table(raw / "github_code_clean_go").num_rows == 1
    assert _table(raw / "plain").num_rows == 3
    assert (raw / "broken").is_dir() and not list_parquet_files(raw / "broken", "shard")
    assert "Error downloading broken: offline" in capsys.readouterr().out


# --- CLI ------------------------------------------------------------------------------------------------------------


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["download_pretraining", *argv])
    dp.main()


@pytest.mark.parametrize("parallel", ["1", "2"])
def test_main_selects_datasets_and_writes_raw_layout(
    tmp_path: Path,
    stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    parallel: str,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # same sources and handlers as the real table, but tiny budgets
    monkeypatch.setattr(dp, "DATASETS", [{**c, "target_samples": 6} for c in dp.DATASETS])
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--parallel", parallel, "--datasets", "gsm8k", "peso",
         "github_code_clean_rust", "not_a_source"],
    )  # fmt: skip
    raw = tmp_path / "pretraining" / "raw"
    assert sorted(d.name for d in raw.iterdir()) == ["github_code_clean_rust", "gsm8k", "peso"]
    assert _table(raw / "gsm8k").num_rows == 6
    assert _table(raw / "github_code_clean_rust")["language"].to_pylist() == ["Rust"]
    assert _table(raw / "peso").column_names == ["text", "extra"]
    requested = {call[0] for call in stub_hub}
    assert requested == {"gsm8k", "nampdn-ai/mini-peS2o", dp.GITHUB_CODE_DATASET}
    out = capsys.readouterr().out
    assert "Warning: HF_TOKEN not set" in out
    assert "Successful: 3 / 3" in out and "Datasets: 3" in out
    assert "Failed" not in out


def test_main_reports_failures(
    tmp_path: Path,
    stub_hub: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dp, "DATASETS", [
        {"name": "ok", "hf_dataset": "x/y", "target_samples": 2, "load_kwargs": {}},
        {"name": "bad", "hf_dataset": "failing/source", "target_samples": 2, "load_kwargs": {}},
    ])  # fmt: skip
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Successful: 1 / 2: ['ok']" in out and "Failed: ['bad']" in out
