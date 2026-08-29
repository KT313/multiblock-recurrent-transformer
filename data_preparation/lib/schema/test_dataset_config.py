# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.schema.dataset_config: loading the shipped configs, every validation rule, hashes and
budget arithmetic."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from data_preparation.lib.schema import dataset_config as dc
from data_preparation.lib.schema.dataset_config import (
    DatasetConfig,
    DedupConfig,
    MixtureConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
    load_dataset_config,
)

REPO = Path(__file__).resolve().parents[3]
CROW = REPO / "config" / "datasets" / "crow_300m_final.yaml"
TINY = REPO / "config" / "datasets" / "tiny.yaml"


def _minimal() -> dict[str, Any]:
    """A small valid config as a plain dict (mutated by the validation tests)."""
    return {
        "name": "t",
        "tokenizer": {"name": "synthetic", "kind": "synthetic"},
        "sources": {
            "pre": {"kind": "pretrain", "loader": "synthetic"},
            "hold": {"kind": "holdout", "loader": "synthetic", "rows": 10},
            "ins": {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/y", "fields": {"instruction": "a", "output": "b"}},
        },
        "mixtures": {"mix": {"sources": {"ins": 1.0}}},
        "stages": [
            {"name": "s1", "tokens": 1000, "train": {"pre": 1.0}, "val": {"hold": 1.0}},
            {"name": "s2", "tokens": 500, "train": {"mix": 1.0}, "val": {"mix/validation": 1.0}},
        ],
    }


def _build(d: dict[str, Any]) -> DatasetConfig:
    d = copy.deepcopy(d)
    sources = {k: SourceConfig(**({**v, "processing": ProcessingConfig(**v["processing"])} if v.get("processing") else v))
               for k, v in d["sources"].items()}
    return DatasetConfig(
        name=d["name"],
        tokenizer=TokenizerConfig(**d["tokenizer"]),
        sources=sources,
        mixtures={k: MixtureConfig(**v) for k, v in d.get("mixtures", {}).items()},
        stages=[StageConfig(**s) for s in d["stages"]],
        **{k: v for k, v in d.items() if k in ("max_seq_length", "token_count")},
    )


# --- shipped files ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [CROW, TINY])
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_dataset_config(path)
    assert cfg.name in ("crow-300m-final", "tiny")
    assert cfg.stages and cfg.sources


def test_crow_config_matches_thesis_run() -> None:
    cfg = load_dataset_config(CROW)
    assert [s.name for s in cfg.stages] == ["pretrain_phase1", "pretrain_phase2", "finetune"]
    assert [s.tokens for s in cfg.stages] == [3_300_000_000, 1_500_000_000, 150_000_000]
    assert len(cfg.sources_of_kind("pretrain")) == 19
    assert len(cfg.sources_of_kind("instruct")) == 8
    assert cfg.sources_of_kind("holdout") == ["fineweb_val"]
    assert cfg.token_count == "tokenizer" and cfg.max_seq_length == 2048
    assert cfg.processing.dedup.mode == "exact" and not cfg.processing.quality_filter
    assert not cfg.processing.decontamination.enabled
    assert all(s.revision for s in cfg.sources.values()), "every Hub source must pin a revision"
    assert cfg.sources["books_gutenberg"].text_field == "TEXT"
    assert cfg.sources["gsm8k"].repeat_to_budget and cfg.sources["gsm8k"].converter == "gsm8k_question_answer"
    assert cfg.mixtures["flan_mixture"].input_inversions == 0.05


def test_tiny_config_is_synthetic_only() -> None:
    cfg = load_dataset_config(TINY)
    assert cfg.tokenizer.kind == "synthetic"
    assert {s.loader for s in cfg.sources.values()} == {"synthetic"}
    assert cfg.max_seq_length == 256


def test_overrides_apply_to_nested_keys() -> None:
    cfg = load_dataset_config(TINY, ["--max_seq_length", "128", "--processing.dedup.mode", "none"])
    assert cfg.max_seq_length == 128 and cfg.processing.dedup.mode == "none"


def test_load_from_written_yaml(tmp_path: Path) -> None:
    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump(_minimal()))
    cfg = load_dataset_config(path)
    assert cfg.sources["hold"].rows == 10 and cfg.stages[1].train == {"mix": 1.0}


# --- validation rules -------------------------------------------------------------------------------------------------


def test_minimal_is_valid() -> None:
    _build(_minimal())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["stages"][0]["train"].update({"pre": 0.5}), "sum to"),
        (lambda d: d["stages"][0]["train"].update({"nope": 0.0}), "unknown source"),
        (lambda d: d["stages"][0]["train"].update({"hold": 0.0}), "holdout"),
        (lambda d: d["stages"][0]["train"].update({"ins": 0.0}), "instruct source"),
        (lambda d: d["stages"][0]["train"].update({"pre/train": 0.0}), "not a mixture"),
        (lambda d: d["stages"][1]["val"].update({"mix/test": 0.0}), "/validation"),
        (lambda d: d["mixtures"]["mix"]["sources"].update({"pre": 0.0}), "not kind instruct"),
        (lambda d: d["mixtures"]["mix"]["sources"].update({"zzz": 0.0}), "unknown source"),
        (lambda d: d["mixtures"].update({"pre": {"sources": {"ins": 1.0}}}), "shared by sources and mixtures"),
        (lambda d: d["stages"].append(dict(d["stages"][0])), "unique"),
        (lambda d: d["stages"].clear(), "at least one stage"),
        (lambda d: d.update({"name": "a/b"}), "path component"),
        (lambda d: d.update({"max_seq_length": 0}), "max_seq_length"),
        (lambda d: d["stages"][0].update({"tokens": 0}), "tokens must be positive"),
        (lambda d: d["stages"][0].update({"transition_pct": 1.0}), "transition_pct"),
        (lambda d: d["stages"][0].update({"train": {}}), "must not be empty"),
        (lambda d: d["stages"][0]["train"].update({"pre": -1.0, "hold": 2.0}), "non-negative"),
        (lambda d: d["sources"]["hold"].pop("rows"), "holdout requires rows"),
        (lambda d: d["sources"]["ins"].pop("fields"), "requires fields or converter"),
        (lambda d: d["sources"]["ins"].update({"fields": {"instruction": "a"}}), "instruction and output"),
        (lambda d: d["sources"]["ins"].pop("hf_id"), "requires hf_id"),
        (lambda d: d["sources"]["pre"].update({"loader": "github_code"}), "requires language"),
        (lambda d: d["sources"]["pre"].update({"loader": "github_code", "language": "Python"}), "requires hf_id"),
        (lambda d: d["sources"]["pre"].update({"loader": "hf_files", "hf_id": "o/r"}), "requires load_kwargs.data_files"),
        (lambda d: d["sources"]["pre"].update({"loader": "hf_files", "hf_id": "o/r", "load_kwargs": {"data_files": 3}}), "requires load_kwargs.data_files"),
        (lambda d: d["sources"]["pre"].update({"loader": "local"}), "requires path"),
        (lambda d: d["sources"]["pre"].update({"tokens_per_row_estimate": 0}), "tokens_per_row_estimate"),
        (lambda d: d["sources"]["hold"].update({"processing": {"min_chars": 1}}), "only apply to kind pretrain"),
        (lambda d: d["mixtures"]["mix"].update({"val_split": 1.0}), "val_split"),
        (lambda d: d["mixtures"]["mix"].update({"input_inversions": 1.5}), "input_inversions"),
        (lambda d: d.update({"tokenizer": {"name": "x", "kind": "hf"}}), "requires hf_id"),
    ],
)
def test_validation_rejects(mutate: Any, match: str) -> None:
    d = _minimal()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        _build(d)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
        ({"num_perm": 0}, "positive"),
        ({"ngram": 0}, "positive"),
    ],
)
def test_dedup_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DedupConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"min_chars": -1}, {"max_chars": 0}, {"min_chars": 10, "max_chars": 5}])
def test_processing_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="min_chars"):
        ProcessingConfig(**kwargs)


def test_weights_tolerate_float_noise() -> None:
    d = _minimal()
    d["stages"][0]["train"] = {"pre": 0.1 + 0.2 + 0.7}  # 1.0000000000000002
    _build(d)


# --- derived views ----------------------------------------------------------------------------------------------------


def test_source_processing_override() -> None:
    d = _minimal()
    d["sources"]["pre"]["processing"] = {"min_chars": 7, "quality_filter": True}
    cfg = _build(d)
    assert cfg.source_processing("pre").min_chars == 7 and cfg.source_processing("pre").quality_filter
    assert cfg.source_processing("pre") is not cfg.processing
    d = _minimal()
    cfg = _build(d)
    assert cfg.source_processing("pre") is cfg.processing


def test_budget_tokens_use_max_over_stages_not_sum() -> None:
    d = _minimal()
    d["sources"]["pre2"] = {"kind": "pretrain", "loader": "synthetic"}
    d["stages"][0]["train"] = {"pre": 0.6, "pre2": 0.4}
    d["stages"][1] = {"name": "s2", "tokens": 4000, "train": {"pre": 0.1, "pre2": 0.9}, "val": {"hold": 1.0}}
    cfg = _build(d)
    assert cfg.source_budget_tokens("pre") == max(600, 400) == 600
    assert cfg.source_budget_tokens("pre2") == max(400, 3600) == 3600
    assert cfg.source_budget_tokens("hold") == 0  # validation sets have a fixed row count instead


def test_mixture_budget_and_instruct_source_budget() -> None:
    d = _minimal()
    d["sources"]["ins2"] = {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/z", "converter": "first_two_turns"}
    d["mixtures"]["mix"]["sources"] = {"ins": 0.75, "ins2": 0.25}
    d["stages"][1]["tokens"] = 1000
    d["stages"][1]["train"] = {"mix/train": 1.0}
    cfg = _build(d)
    assert cfg.mixture_budget_tokens("mix") == 1000
    assert cfg.source_budget_tokens("ins") == 750 and cfg.source_budget_tokens("ins2") == 250
    assert cfg.mixture_budget_tokens("unused") == 0


def test_sources_of_kind() -> None:
    cfg = _build(_minimal())
    assert cfg.sources_of_kind("pretrain") == ["pre"]
    assert cfg.sources_of_kind("holdout") == ["hold"]
    assert cfg.sources_of_kind("instruct") == ["ins"]


# --- hashes -----------------------------------------------------------------------------------------------------------


def test_source_hash_stable_across_key_order_and_reloads() -> None:
    a = load_dataset_config(TINY)
    b = load_dataset_config(TINY)
    assert a.source_hash("synthetic_pretrain") == b.source_hash("synthetic_pretrain")
    assert a.config_hash() == b.config_hash()
    assert dc._stable_hash({"x": 1, "y": [1, 2]}) == dc._stable_hash({"y": [1, 2], "x": 1})


def test_source_hash_ignores_budget_but_tracks_processing_and_token_mode() -> None:
    base = _build(_minimal())
    h = base.source_hash("pre")

    d = _minimal()
    d["stages"][0]["tokens"] = 999_999  # budget change: same rows on disk
    d["sources"]["pre"]["tokens_per_row_estimate"] = 3  # planner prior: same rows on disk
    assert _build(d).source_hash("pre") == h

    d = _minimal()
    d["sources"]["pre"]["processing"] = {"min_chars": 99}
    assert _build(d).source_hash("pre") != h

    d = _minimal()
    d["token_count"] = "estimate"
    assert _build(d).source_hash("pre") != h

    d = _minimal()
    d["max_seq_length"] = 64
    assert _build(d).source_hash("pre") != h
    assert _build(d).source_hash("hold") == base.source_hash("hold")  # cap only affects pretrain counting

    d = _minimal()
    d["sources"]["pre"]["seed"] = 5
    assert _build(d).source_hash("pre") != h


def test_tokenizer_change_affects_hash_only_when_counting_with_it() -> None:
    d = _minimal()
    base = _build(d)
    d["tokenizer"] = {"name": "other", "kind": "hf", "hf_id": "a/b"}
    assert _build(d).source_hash("pre") != base.source_hash("pre")
    d["token_count"] = "estimate"
    d2 = copy.deepcopy(d)
    d2["tokenizer"] = {"name": "third", "kind": "hf", "hf_id": "c/d"}
    assert _build(d).source_hash("pre") == _build(d2).source_hash("pre")
    assert _build(d).source_hash("ins") != _build(d2).source_hash("ins")  # instruct always tokenizes


def test_mixture_hash_tracks_sources_and_budget() -> None:
    base = _build(_minimal())
    h = base.mixture_hash("mix")
    d = _minimal()
    d["sources"]["ins"]["fields"] = {"instruction": "q", "output": "b"}
    assert _build(d).mixture_hash("mix") != h
    d = _minimal()
    d["stages"][1]["tokens"] = 501
    assert _build(d).mixture_hash("mix") != h
    d = _minimal()
    d["mixtures"]["mix"]["val_split"] = 0.1
    assert _build(d).mixture_hash("mix") != h


def test_config_hash_changes_on_any_field() -> None:
    base = _build(_minimal()).config_hash()
    d = _minimal()
    d["stages"][0]["transition_pct"] = 0.5
    assert _build(d).config_hash() != base
    assert len(base) == 16


def test_dataset_config_fields_and_asdict_roundtrip() -> None:
    assert {"name", "tokenizer", "sources", "stages", "mixtures", "processing"} <= set(dc.dataset_config_fields())
    cfg = _build(_minimal())
    assert asdict(cfg)["sources"]["pre"]["kind"] == "pretrain"
