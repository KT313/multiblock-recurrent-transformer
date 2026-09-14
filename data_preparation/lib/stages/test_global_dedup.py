# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Contracts of global comparison, deterministic admission and failure recovery."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from data_preparation.conftest import CfgFactory
from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.stages.global_dedup import (
    GlobalAdmission, GlobalFrontier, global_key, global_policy, ordered_sources,
)


def test_complete_structured_key_and_normalization() -> None:
    row = {"instruction": " Do   THIS ", "input": None, "output": " Answer "}
    assert global_key("instruct", row) == global_key("instruct", {**row, "input": "", "output": "answer"})
    assert global_key("instruct", row) != global_key("instruct", {**row, "output": "another answer"})
    assert global_key("instruct", {"instruction": "a b", "input": "c", "output": "d"}) != global_key(
        "instruct", {"instruction": "a", "input": "b c", "output": "d"}
    )
    assert global_key("pretrain", {"text": "DO\n this answer"}) != global_key("instruct", row)
    assert global_key("pretrain", {"text": " A\n B "}) == global_key("pretrain", {"text": "a b"})


def test_priority_and_policy_identity(cfg_factory: CfgFactory) -> None:
    cfg = cfg_factory({
        "b": SourceConfig(kind="pretrain", loader="synthetic"),
        "val": SourceConfig(kind="pretrain", loader="synthetic", rows=2),
        "a": SourceConfig(kind="pretrain", loader="synthetic"),
    })
    cfg = replace(cfg, bloom_deduplicate_across_sources=True, bloom_dedup_memory_mb=1024)
    assert cfg.bloom_deduplicate_across_sources is True
    assert cfg.bloom_dedup_memory_mb == 1024
    assert ordered_sources(cfg) == ("val", "b", "a")
    reversed_cfg = replace(cfg, sources=dict(reversed(list(cfg.sources.items()))))
    assert cfg.config_hash() != reversed_cfg.config_hash()
    assert global_policy(cfg)["source_order"] == ["val", "b", "a"]
    assert cfg.raw_hash("a") == reversed_cfg.raw_hash("a")
    assert cfg.processed_hash("a") == reversed_cfg.processed_hash("a")
    changed = replace(cfg, bloom_dedup_memory_mb=1)
    assert cfg.config_hash() != changed.config_hash()
    assert cfg.processed_hash("a") == changed.processed_hash("a")
    disabled = replace(cfg, bloom_deduplicate_across_sources=False)
    assert disabled.config_hash() == replace(disabled, bloom_dedup_memory_mb=1).config_hash()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_bad_global_budget(cfg_factory: CfgFactory, value: Any) -> None:
    cfg = cfg_factory({"a": SourceConfig(kind="pretrain", loader="synthetic")})
    with pytest.raises(ValueError, match="bloom_dedup_memory_mb"):
        replace(cfg, bloom_dedup_memory_mb=value)


def test_ordered_admission_and_failed_commit() -> None:
    admission = GlobalAdmission(("val", "a"), memory_mb=1)
    committed: list[dict[str, Any]] = []
    frontiers: list[GlobalFrontier] = []

    def publish(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        committed.extend(rows)
        frontiers.append(frontier)

    admission.commit_batch("val", "pretrain", [{"text": "x"}], publish)
    admission.finish_source("val", publish)
    with pytest.raises(ValueError, match="priority"):
        admission.commit_batch("val", "pretrain", [{"text": "late"}], publish)

    def fail(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        raise OSError("publication failed")

    with pytest.raises(OSError, match="publication failed"):
        admission.commit_batch("a", "pretrain", [{"text": "y"}], fail)
    with pytest.raises(RuntimeError, match="recover"):
        admission.commit_batch("a", "pretrain", [{"text": "y"}], publish)
    recovered = GlobalAdmission(("val", "a"), memory_mb=1, frontier=frontiers[-1],
                                committed_keys=(row["global_hash"] for row in committed))
    recovered.commit_batch("a", "pretrain", [{"text": "X"}, {"text": "y"}], publish)
    assert [row["text"] for row in committed] == ["x", "y"]
    assert recovered.frontier.candidates == 3
    assert recovered.frontier.bloom_positive == 1
    assert recovered.frontier.retained == 2


def test_recovery_rejects_missing_or_changed_keys() -> None:
    admission = GlobalAdmission(("a",), memory_mb=1)
    admission.commit_batch("a", "pretrain", [{"text": "x"}], lambda rows, frontier: None)
    for keys in ([], [123]):
        with pytest.raises(ValueError, match="committed keys"):
            GlobalAdmission(("a",), memory_mb=1, frontier=admission.frontier, committed_keys=keys)
    with pytest.raises(ValueError, match="policy"):
        GlobalAdmission(("a",), memory_mb=2, frontier=admission.frontier, committed_keys=[])


def test_bloom_positive_is_accepted_and_overload_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    admission = GlobalAdmission(("a",), memory_mb=1)
    monkeypatch.setattr(admission.seen, "add_if_new", lambda key: False)
    admission.commit_batch("a", "pretrain", [{"text": "unique"}], lambda rows, frontier: None)
    assert admission.frontier.bloom_positive == 1
    assert admission.frontier.retained == 0
    with pytest.raises(ValueError, match="bloom_dedup_memory_mb"):
        admission.check_capacity(10_000_000)


def _run_candidates(
    prepared: dict[str, list[dict[str, Any]]], *, restart: bool = False,
) -> tuple[list[dict[str, Any]], GlobalFrontier]:
    order = ("a", "b")
    admission = GlobalAdmission(order, memory_mb=1)
    output: list[dict[str, Any]] = []
    saved = admission.frontier

    def publish(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        nonlocal saved
        output.extend(rows)
        saved = frontier

    for name in order:
        for candidate in prepared[name]:
            admission.commit_batch(name, "pretrain", [candidate], publish)
            if restart:
                admission = GlobalAdmission(order, memory_mb=1, frontier=saved,
                                            committed_keys=(row["global_hash"] for row in output))
        admission.finish_source(name, publish)
    return output, admission.frontier


def test_completion_order_and_restarts_cannot_select_winners() -> None:
    prepared = {"a": [{"text": "shared", "source": "a"}],
                "b": [{"text": "SHARED", "source": "b"}, {"text": "unique", "source": "b"}]}
    expected = _run_candidates(prepared)
    assert expected == _run_candidates(dict(reversed(list(prepared.items()))), restart=True)
    assert [(row["source"], row["text"]) for row in expected[0]] == [("a", "shared"), ("b", "unique")]


def test_replay_reclaims_late_higher_priority_duplicate() -> None:
    prepared = {"a": [{"text": "first", "source": "a"}], "b": [{"text": "late", "source": "b"}]}
    assert _run_candidates(prepared)[0][-1]["source"] == "b"
    prepared["a"].append({"text": "late", "source": "a"})
    assert _run_candidates(prepared)[0][-1]["source"] == "a"


def test_preseed_hook_is_explicit_and_bound_to_recovery() -> None:
    key = global_key("pretrain", {"text": "seed"})
    admission = GlobalAdmission(("a",), memory_mb=1, preseed_keys=[key])
    admission.commit_batch("a", "pretrain", [{"text": "SEED"}], lambda rows, frontier: None)
    assert admission.frontier.retained == 0
    assert admission.frontier.preseed_count == 1
    with pytest.raises(ValueError, match="preseed"):
        GlobalAdmission(("a",), memory_mb=1, frontier=admission.frontier)
    restored = GlobalAdmission(("a",), memory_mb=1, frontier=admission.frontier, preseed_keys=[key])
    assert restored.statistics()["bloom_positive"] == 1


def test_frontier_roundtrip_and_corruption() -> None:
    frontier = GlobalAdmission(("a",), memory_mb=1).frontier
    assert GlobalFrontier.from_dict(frontier.to_dict()) == frontier
    for field, value in (("retained", True), ("source_order", "a"), ("key_digest", "broken")):
        with pytest.raises(ValueError, match="corrupt"):
            GlobalFrontier.from_dict({**frontier.to_dict(), field: value})
    with pytest.raises(ValueError, match="schema"):
        GlobalFrontier.from_dict({**frontier.to_dict(), "unexpected_version": 99})


def test_parquet_publication_failure_does_not_commit_reservation(tmp_path: Path) -> None:
    """The durable frontier, not orphan output files, controls recovery."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_preparation.lib.storage.atomic import write_atomically
    from data_preparation.lib.storage.parquet import publish_shard

    admission = GlobalAdmission(("a",), memory_mb=1)
    state_path = tmp_path / "frontier.json"
    state_path.write_text(json.dumps(admission.frontier.to_dict()))
    shard_path = tmp_path / "data-00000.parquet"

    def failed_publish(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        publish_shard(pa.Table.from_pylist(rows), shard_path)
        raise OSError("before frontier publication")

    with pytest.raises(OSError, match="frontier publication"):
        admission.commit_batch("a", "pretrain", [{"text": "rightful"}], failed_publish)
    saved = GlobalFrontier.from_dict(json.loads(state_path.read_text()))
    assert saved.retained == 0
    recovered = GlobalAdmission(("a",), memory_mb=1, frontier=saved, committed_keys=[])

    def commit(rows: list[dict[str, Any]], frontier: GlobalFrontier) -> None:
        publish_shard(pa.Table.from_pylist(rows), shard_path)
        with write_atomically(state_path) as temporary:
            temporary.write_text(json.dumps(frontier.to_dict()))

    recovered.commit_batch("a", "pretrain", [{"text": "rightful"}], commit)
    rows = pq.read_table(shard_path).to_pylist()
    assert [row["text"] for row in rows] == ["rightful"]
    saved = GlobalFrontier.from_dict(json.loads(state_path.read_text()))
    restarted = GlobalAdmission(("a",), memory_mb=1, frontier=saved,
                                committed_keys=(row["global_hash"] for row in rows))
    assert restarted.frontier.retained == 1


@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_malformed_instruction_input_fails(value: Any) -> None:
    with pytest.raises(ValueError, match="string"):
        global_key("instruct", {"instruction": "question", "input": value, "output": "answer"})
