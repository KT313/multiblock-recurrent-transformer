# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Replay safety, fresh inputs, grouped buffering and durable output parity."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.stages.download_state import _IncrementCounters, _TokenStep
from tools.data_preparation.replay import InputRow, load_fixture, make_counter, run_pipeline, summarize_output


def test_fixture_rejects_inconsistent_group_and_bounded_input(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"text":"a","source":"s","group":"a"}\n{"text":"b","source":"s","group":"b"}\n')
    with pytest.raises(ValueError, match="consistent"):
        load_fixture(path, 1000, 10)
    with pytest.raises(ValueError, match="row limits"):
        load_fixture(path, 1000, 1)
    with pytest.raises(ValueError, match="input-mb"):
        load_fixture(path, 1, 10)
    path.write_text(json.dumps({"text": "a", "source": "../dataset"}) + "\n")
    with pytest.raises(ValueError):
        load_fixture(path, 1000, 10)


def test_replay_is_repeatable_without_mutating_original_rows(tmp_path: Path, tiny_tokenizer_dir: Path) -> None:
    rows = [InputRow("main", "group", "tok_1 tok_2 tok_3 " * 4) for _ in range(257)]
    rows += [InputRow("passive", "group", "tok_4 tok_5", True)]
    results: list[dict[str, Any]] = []
    for variant in (True, False, False):
        output = tmp_path / str(len(results))
        run_pipeline(rows, tiny_tokenizer_dir, output, 4, 2, variant)
        results.append(summarize_output(output)["sources"])
    assert results[0] == results[1] == results[2]
    assert rows[0].text == "tok_1 tok_2 tok_3 " * 4
    assert results[0]["main"]["rows_fetched"] == 257
    assert results[0]["passive"]["rows"] == 1
    with pytest.raises(FileExistsError):
        run_pipeline(rows, tiny_tokenizer_dir, tmp_path / "0", 4, 2, False)


@pytest.mark.parametrize("variant", ["original", "candidate", "builtin", "rust"])
def test_replay_counter_contract_and_pipeline_parity(tmp_path: Path, tiny_tokenizer_dir: Path, variant: str) -> None:
    if variant == "rust" and "PREPARER_NATIVE_LIBRARY" not in os.environ:
        pytest.skip("optional native adapter not built; set PREPARER_NATIVE_LIBRARY to test it")
    before = {path.name: path.read_bytes() for path in tiny_tokenizer_dir.iterdir() if path.is_file()}
    counter = make_counter(tiny_tokenizer_dir, variant)
    assert counter.pool is None
    assert counter.tokenizer_dir == tiny_tokenizer_dir
    step = _TokenStep(SourceConfig(kind="pretrain", loader="synthetic"), counter, 4, _IncrementCounters())
    assert step.parallel_batches == 1
    rows = [InputRow("main", "group", "tok_1 tok_2 tok_3 " * 4) for _ in range(17)]
    rows.append(InputRow("passive", "group", "tok_4 tok_5", True))
    run_pipeline(rows, tiny_tokenizer_dir, tmp_path / "original", 4, 1, "original")
    run_pipeline(rows, tiny_tokenizer_dir, tmp_path / "variant", 4, 1, variant)
    assert summarize_output(tmp_path / "original")["sources"] == summarize_output(tmp_path / "variant")["sources"]
    assert before == {path.name: path.read_bytes() for path in tiny_tokenizer_dir.iterdir() if path.is_file()}
