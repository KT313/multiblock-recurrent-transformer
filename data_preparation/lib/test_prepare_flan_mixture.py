# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.prepare_flan_mixture: schema conversion, filters, inversions and the CLI with a
stubbed Hub."""

import json
import sys
from types import ModuleType
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from collections.abc import Iterator

from data_preparation.lib import prepare_flan_mixture as pfm
from data_preparation.lib.common import list_parquet_files

# --- standardize_format ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        (  # alpaca style (Evol-Instruct-Code, Code Alpaca, WizardLM)
            {"instruction": "do x", "input": "with y", "output": "done"},
            {"instruction": "do x", "input": "with y", "output": "done"},
        ),
        ({"instruction": "do x", "output": 42}, {"instruction": "do x", "input": "", "output": "42"}),
        (
            {"inputs": "prompt", "targets": "target", "task": "t"},
            {"instruction": "prompt", "input": "", "output": "target"},
        ),
        (  # SlimOrca / ShareGPT role-tagged conversations
            {
                "conversations": [
                    {"from": "system", "value": "be nice"},
                    {"from": "human", "value": "hi"},
                    {"from": "gpt", "value": "hello"},
                ]
            },
            {"instruction": "hi", "input": "be nice", "output": "hello"},
        ),
        (  # role-tagged with an unknown role and no system prompt
            {
                "conversations": [
                    {"from": "human", "value": "q"},
                    {"from": "tool", "value": "x"},
                    {"from": "gpt", "value": "a"},
                ]
            },
            {"instruction": "q", "input": "", "output": "a"},
        ),
        (  # untagged conversations: first two turns
            {"conversations": [{"value": "q"}, {"value": "a"}, {"value": "ignored"}]},
            {"instruction": "q", "input": "", "output": "a"},
        ),
        (  # OpenOrca
            {"question": "q", "response": "r", "system_prompt": "sys", "id": 1},
            {"instruction": "q", "input": "sys", "output": "r"},
        ),
        ({"problem": "p", "solution": "s"}, {"instruction": "p", "input": "", "output": "s"}),
        (
            {"query": "q", "response": "r", "type": "MATH"},
            {"instruction": "q", "input": "", "output": "r"},
        ),  # MetaMathQA
        ({"question": "q", "answer": "a"}, {"instruction": "q", "input": "", "output": "a"}),  # Orca-Math
    ],
)
def test_standardize_format(example: dict[str, Any], expected: object) -> None:
    assert pfm.standardize_format(example) == expected


def test_standardize_format_precedence() -> None:
    # instruction/output wins over every other schema
    ex = {"instruction": "i", "output": "o", "inputs": "x", "targets": "y", "question": "q", "answer": "a"}
    assert pfm.standardize_format(ex) == {"instruction": "i", "input": "", "output": "o"}
    # question+response (OpenOrca) beats question+answer (Orca-Math)
    ex = {"question": "q", "response": "r", "answer": "a"}
    assert pfm.standardize_format(ex)["output"] == "r"


@pytest.mark.parametrize(
    "example",
    [
        {"foo": "bar"},
        {"instruction": "no output"},
        {"conversations": [{"value": "only one turn"}]},
        {"conversations": "not a list"},
        {"conversations": []},
    ],
)
def test_standardize_format_rejects_unknown(example: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Cannot convert example"):
        pfm.standardize_format(example)


# --- filters --------------------------------------------------------------------------------------------------------


def test_check_length_uses_1_3_tokens_per_word() -> None:
    assert pfm.check_length({"instruction": "a", "input": "", "output": "b"})
    # 2048 / 1.3 = 1575.38 words -> 1575 words pass, 1576 fail
    words = " ".join(["w"] * 1573)  # + instruction + output = 1575 words
    assert pfm.check_length({"instruction": "a", "output": "b", "input": words})
    assert not pfm.check_length({"instruction": "a", "output": "b", "input": words + " w"})
    assert pfm.check_length({"instruction": "a", "output": "b", "input": words + " w"}, max_tokens=2100)
    # `input` is optional
    assert pfm.check_length({"instruction": "a", "output": "b"})


def _sharegpt(human: str, gpt: str, first: str = "human", second: str = "gpt") -> dict[str, Any]:
    return {"conversations": [{"from": first, "value": human}, {"from": second, "value": gpt}]}


def test_is_quality_sharegpt() -> None:
    ok_h, ok_g = "h" * 100, "g" * 100
    assert pfm.is_quality_sharegpt(_sharegpt(ok_h, ok_g))
    assert pfm.is_quality_sharegpt(_sharegpt("h" * 50, "g" * 2000))  # inclusive bounds
    assert not pfm.is_quality_sharegpt(_sharegpt("h" * 49, ok_g))
    assert not pfm.is_quality_sharegpt(_sharegpt(ok_h, "g" * 2001))
    assert not pfm.is_quality_sharegpt(_sharegpt(ok_h, ok_g, first="gpt", second="human"))
    assert not pfm.is_quality_sharegpt(_sharegpt(ok_h, ok_g, first="system"))
    assert not pfm.is_quality_sharegpt({"conversations": [{"from": "human", "value": ok_h}]})
    assert not pfm.is_quality_sharegpt({"conversations": None})
    assert not pfm.is_quality_sharegpt({})
    for fence in ("```python", "```Java", "```cpp", "```javascript"):
        assert not pfm.is_quality_sharegpt(_sharegpt(ok_h, ok_g + f"\n{fence}\nx\n```"))
    assert pfm.is_quality_sharegpt(_sharegpt(ok_h, ok_g + "\n```bash\nls\n```"))  # other languages are fine
    assert pfm.is_quality_sharegpt(_sharegpt(ok_h + "```python", ok_g))  # only the answer is checked


# --- inversions / dedup helpers -------------------------------------------------------------------------------------


def test_create_input_inversion_with_and_without_input() -> None:
    ex = {"instruction": "Sort the list", "input": "[3, 1]", "output": "[1, 3]"}
    inv = pfm.create_input_inversion(ex)
    assert inv == {
        "instruction": "Given this output, what was the likely instruction or input?\n\nOutput: [1, 3]",
        "input": "",
        "output": "Sort the list\nInput: [3, 1]",
    }
    ex = {"instruction": "Say hi", "input": "", "output": "hi"}
    assert pfm.create_input_inversion(ex)["output"] == "Say hi"
    assert pfm.create_input_inversion(ex)["instruction"].endswith("Output: hi")


def test_create_input_inversion_noop_cases() -> None:
    assert pfm.create_input_inversion({"foo": 1}) == {"foo": 1}
    assert pfm.create_input_inversion({"instruction": "x"}) == {"instruction": "x"}
    empty = {"instruction": "x", "input": "", "output": ""}
    assert pfm.create_input_inversion(empty) is empty


def test_compute_example_hash_and_required_fields() -> None:
    a = {"instruction": "i", "input": "x", "output": "o"}
    b = {"instruction": "i", "input": "x", "output": "o"}
    c = {"instruction": "i", "input": "", "output": "o"}
    assert pfm.compute_example_hash(a) == pfm.compute_example_hash(b) != pfm.compute_example_hash(c)
    assert len(pfm.compute_example_hash(a)) == 32
    assert pfm.has_required_fields(a)
    assert not pfm.has_required_fields({"instruction": "  ", "input": "", "output": "o"})
    assert not pfm.has_required_fields({"instruction": "i", "input": "", "output": ""})
    assert not pfm.has_required_fields({"instruction": None, "input": "", "output": "o"})


# --- iter_standardized ----------------------------------------------------------------------------------------------


def test_iter_standardized_limits_filters_and_skips_bad_rows() -> None:
    rows: list[dict[str, Any]] = [
        {"question": "q1", "answer": "a1"},
        {"bogus": 1},  # unconvertible -> skipped
        {"instruction": "i", "output": "w " * 2000},  # too long -> skipped
        {"problem": "p", "solution": "s"},
        {"query": "never reached", "response": "r"},
    ]
    out = list(pfm.iter_standardized(iter(rows), target_count=2))
    assert [o["instruction"] for o in out] == ["q1", "p"]
    assert list(pfm.iter_standardized(iter(rows), target_count=0)) == []


def test_iter_standardized_sharegpt_applies_quality_filter_and_check_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pfm, "SHAREGPT_CHECK_LIMIT", 3)
    good = _sharegpt("h" * 60, "g" * 60)
    bad = _sharegpt("short", "g" * 60)
    out = list(pfm.iter_standardized(iter([bad, good, good, good, good]), target_count=10, sharegpt=True))
    # rows 1..3 are checked; the limit triggers right after the third *checked* row yields
    assert len(out) == 2 and out[0] == {"instruction": "h" * 60, "input": "", "output": "g" * 60}
    assert "Reached ShareGPT check limit" in capsys.readouterr().out


# --- CLI ------------------------------------------------------------------------------------------------------------


def test_parser_defaults_and_count_options() -> None:
    args = pfm.build_parser().parse_args([])
    assert args.total_examples == 400000 and args.val_split == 0.05 and args.inversion_ratio == 0.3
    assert not args.add_input_inversions
    for key, _, _, _ in pfm.SOURCES:
        assert getattr(args, f"{key}_count") is None
    args = pfm.build_parser().parse_args(["--flan_count", "7", "--add_input_inversions", "--inversion_ratio", "0.05"])
    assert args.flan_count == 7 and args.add_input_inversions and args.inversion_ratio == 0.05


def test_source_shares_sum_to_one() -> None:
    assert sum(share for _, _, _, share in pfm.SOURCES) == pytest.approx(1.0)
    assert len({key for key, _, _, _ in pfm.SOURCES}) == 8


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_flan_mixture", *argv])
    pfm.main()


def test_main_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--dry_run", "--total_examples", "1000"])
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out and "FLAN Collection" in out
    assert "  Total: 1,000" in out  # 400+150+100+125+25+100+50+50
    assert not (tmp_path / "flan_mixture").exists()


def _fake_rows(key: str, n: int) -> list[dict[str, Any]]:
    """Rows in the native schema of each source, all distinct."""
    if key == "flan":
        return [{"inputs": f"flan q{i}", "targets": f"flan t{i}", "task": "x"} for i in range(n)]
    if key == "metamath":
        return [{"query": f"mm q{i}", "response": f"mm r{i}", "type": "GSM"} for i in range(n)]
    if key == "orca_math":
        return [{"question": f"om q{i}", "answer": f"om a{i}"} for i in range(n)]
    if key in ("evol_code", "code_alpaca", "wizardlm"):
        return [
            {"instruction": f"{key} i{i}", "input": "inp" if i % 2 else "", "output": f"{key} o{i}"} for i in range(n)
        ]
    if key == "slimorca":
        return [
            {"conversations": [{"from": "system", "value": "sys"}, {"from": "human", "value": f"so h{i}"},
                               {"from": "gpt", "value": f"so g{i}"}]}
            for i in range(n)
        ]  # fmt: skip
    if key == "sharegpt":
        return [_sharegpt(f"sg human {i} " + "h" * 60, f"sg gpt {i} " + "g" * 60) for i in range(n)]
    raise AssertionError(key)


@pytest.fixture
def stub_hub(hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """`load_dataset(hf_id, streaming=True)` serves synthetic rows; the local parquet loader stays real.

    Records every Hub id requested. The call kwargs are checked at teardown rather than inside the stub: an
    assertion raised inside would be swallowed by ``download_source``'s ``except Exception`` and only show up
    as a silent count of 0.
    """
    real_load_dataset = hf_datasets.load_dataset
    by_id = {hf_id: key for key, _, hf_id, _ in pfm.SOURCES}
    calls: list[str] = []
    kwargs_seen: list[dict[str, Any]] = []

    def fake_load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == "parquet":
            return real_load_dataset(path, *args, **kwargs)
        kwargs_seen.append({"args": args, **kwargs})
        calls.append(path)
        key = by_id[path]
        if key == "wizardlm":
            raise ConnectionError("gated")  # one failing source must not abort the run
        return iter(_fake_rows(key, 40))

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    yield calls
    assert all(k == {"args": (), "split": "train", "streaming": True} for k in kwargs_seen), kwargs_seen


def _read_split(root: Path, name: str) -> pa.Table:
    files = list_parquet_files(root / name, "data")
    assert files, name
    return pa.concat_tables([pq.read_table(f) for f in files])


def test_main_end_to_end_split_is_disjoint_and_matches_metadata(
    tmp_path: Path, stub_hub: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pfm, "SHARD_SIZE", 25)
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--total_examples", "200", "--val_split", "0.1", "--num_workers", "1",
         "--flan_count", "30"],
    )  # fmt: skip
    out = tmp_path / "flan_mixture"
    assert len(stub_hub) == 8
    meta = json.loads((out / "metadata.json").read_text())
    # requested counts: flan 30 (override), metamath 30, orca 20, evol 25, alpaca 5, slimorca 20, sharegpt 10, wizard 0
    assert meta["dataset_counts"] == {
        "flan": 30, "metamath": 30, "orca_math": 20, "evol_code": 25, "code_alpaca": 5, "slimorca": 20,
        "sharegpt": 10, "wizardlm": 0,
    }  # fmt: skip
    total = 140
    assert meta["total_examples"] == total
    assert meta["train_examples"] == int(total * 0.9) == 126 and meta["val_examples"] == 14
    assert meta["input_inversions"] is False and meta["max_tokens"] == 2048 and meta["random_seed"] == 42

    train, val = _read_split(out, "train"), _read_split(out, "validation")
    assert train.column_names == val.column_names == ["instruction", "input", "output"]
    assert train.num_rows == meta["train_examples"] and val.num_rows == meta["val_examples"]
    assert [f.name for f in list_parquet_files(out / "train", "data")] == [f"data-{i:05d}.parquet" for i in range(6)]
    assert [pq.read_metadata(f).num_rows for f in list_parquet_files(out / "train", "data")] == [25] * 5 + [1]

    train_rows = {pfm.compute_example_hash(r) for r in train.to_pylist()}
    val_rows = {pfm.compute_example_hash(r) for r in val.to_pylist()}
    assert len(train_rows) == train.num_rows and len(val_rows) == val.num_rows  # no duplicates
    assert train_rows.isdisjoint(val_rows)
    assert len(train_rows | val_rows) == total
    # all schemas got converted: system prompts land in `input`, sharegpt quality-filtered rows are present
    all_rows = train.to_pylist() + val.to_pylist()
    assert sum(r["input"] == "sys" for r in all_rows) == 20
    assert sum(r["instruction"].startswith("sg human") for r in all_rows) == 10
    assert not (out / "temp_datasets").exists()


def test_main_is_deterministic_and_shuffled(
    tmp_path: Path, stub_hub: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = ["--total_examples", "100", "--num_workers", "1"]
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path / "a"), *argv])
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path / "b"), *argv])
    a = _read_split(tmp_path / "a" / "flan_mixture", "train").to_pylist()
    b = _read_split(tmp_path / "b" / "flan_mixture", "train").to_pylist()
    assert a == b
    # shuffled: sources are interleaved rather than concatenated in download order
    first_ten = [r["instruction"].split()[0] for r in a[:10]]
    assert len(set(first_ten)) > 1


def test_main_with_input_inversions(tmp_path: Path, stub_hub: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--total_examples", "100", "--num_workers", "1",
         "--add_input_inversions", "--inversion_ratio", "0.5"],
    )  # fmt: skip
    out = tmp_path / "flan_mixture"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["input_inversions"] is True and meta["inversion_ratio"] == 0.5
    rows = _read_split(out, "train").to_pylist() + _read_split(out, "validation").to_pylist()
    inverted = [r for r in rows if r["instruction"].startswith("Given this output, what was the likely instruction")]
    assert len(rows) == meta["total_examples"]
    assert len(inverted) == int(meta["total_examples"] * 0.5)  # no duplicates were created by inverting
    assert all(r["input"] == "" for r in inverted)
    # a row with an `input` keeps it in the inverted output
    assert any("\nInput: inp" in r["output"] for r in inverted)


def test_download_source_direct(tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(path: str, *args: Any, **kwargs: Any) -> Iterator[dict[str, Any]]:
        if path == "bad/id":
            raise ConnectionError("offline")
        return iter(_fake_rows("orca_math", 7))

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(pfm, "SHARD_SIZE", 3)
    assert pfm.download_source("orca_math", "Orca", "good/id", 5, tmp_path / "om") == 5
    files = list_parquet_files(tmp_path / "om", "data")
    assert [pq.read_metadata(f).num_rows for f in files] == [3, 2]
    assert pq.read_table(files[0]).column_names == ["instruction", "input", "output"]
    assert pfm.download_source("orca_math", "Orca", "bad/id", 5, tmp_path / "bad") == 0
    assert not (tmp_path / "bad").exists()


def test_main_all_sources_fail_exits(tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path: str, *args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("offline")

    monkeypatch.setattr(hf_datasets, "load_dataset", boom)
    with pytest.raises(SystemExit, match="No datasets successfully downloaded"):
        _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--total_examples", "80"])


def test_main_dedups_across_sources_and_drops_empty_fields(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact duplicates (same instruction/input/output from two sources) collapse to one row and rows whose
    output is empty after standardisation are dropped, both *after* the download step."""
    real_load_dataset = hf_datasets.load_dataset

    def fake_load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        if path == "parquet":
            return real_load_dataset(path, *args, **kwargs)
        if path == "Open-Orca/FLAN":
            return iter(
                [
                    {"inputs": "shared q", "targets": "shared a"},
                    {"inputs": "flan only", "targets": ""},  # survives iter_standardized, dropped by field check
                    {"inputs": "flan q2", "targets": "flan a2"},
                ]
            )
        if path == "meta-math/MetaMathQA":
            return iter([{"query": "shared q", "response": "shared a"}, {"query": "mm q2", "response": "mm a2"}])
        raise ConnectionError("offline")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--total_examples", "100", "--num_workers", "1"])
    out = tmp_path / "flan_mixture"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["dataset_counts"]["flan"] == 3 and meta["dataset_counts"]["metamath"] == 2
    # 5 downloaded -> 4 after dedup -> 3 after the field check; default 5% split: int(3 * 0.95) = 2 train, 1 val
    assert meta["total_examples"] == 3 and meta["train_examples"] == 2 and meta["val_examples"] == 1
    rows = _read_split(out, "train").to_pylist() + _read_split(out, "validation").to_pylist()
    assert sorted(r["instruction"] for r in rows) == ["flan q2", "mm q2", "shared q"]


def test_main_val_split_zero_writes_no_validation(
    tmp_path: Path, stub_hub: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--total_examples", "80", "--num_workers", "1",
                            "--val_split", "0"])  # fmt: skip
    meta = json.loads((tmp_path / "flan_mixture" / "metadata.json").read_text())
    assert meta["val_examples"] == 0 and meta["train_examples"] == meta["total_examples"]


def test_source_count_zero_disables_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--dry_run", "--total_examples", "1000",
                            "--flan_count", "0"])  # fmt: skip
    out = capsys.readouterr().out
    assert "  Total: 600" in out
