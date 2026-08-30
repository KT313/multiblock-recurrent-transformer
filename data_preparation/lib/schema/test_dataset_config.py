# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.schema.dataset_config: loading the shipped configs, every validation rule, hashes and
budget arithmetic."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest
import yaml

from data_preparation.lib.schema import dataset_config as dc
from data_preparation.lib.schema.dataset_config import (
    DatasetConfig,
    DedupConfig,
    InstructMixtureConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
    load_dataset_config,
)

REPO = Path(__file__).resolve().parents[3]
CROW = REPO / "config" / "datasets" / "crow_300m_final.yaml"
TINY = REPO / "config" / "datasets" / "tiny.yaml"
MINI = REPO / "config" / "datasets" / "crow_300m_mini.yaml"


def _minimal() -> dict[str, Any]:
    """A small valid config as a plain dict (mutated by the validation tests)."""
    return {
        "name": "t",
        "tokenizer": {"name": "synthetic", "kind": "synthetic"},
        "sources": {
            "pre": {"kind": "pretrain", "loader": "synthetic"},
            "hold": {"kind": "validation", "loader": "synthetic", "rows": 10},
            "ins": {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/y", "fields": {"instruction": "a", "output": "b"}},
        },
        "instruct_mixtures": {"mix": {"sources": {"ins": 1.0}}},
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
        instruct_mixtures={k: InstructMixtureConfig(**v) for k, v in d.get("instruct_mixtures", {}).items()},
        stages=[StageConfig(**s) for s in d["stages"]],
        max_seq_length=d.get("max_seq_length", 2048),
        token_count=d.get("token_count", "tokenizer"),
        processing=_processing(d["processing"]) if d.get("processing") else ProcessingConfig(),
    )


def _processing(d: dict[str, Any]) -> ProcessingConfig:
    d = dict(d)
    if "dedup" in d:
        d["dedup"] = DedupConfig(**d["dedup"])
    if "decontamination" in d:
        d["decontamination"] = dc.DecontaminationConfig(**d["decontamination"])
    return ProcessingConfig(**d)


# --- shipped files ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [CROW, TINY, MINI])
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_dataset_config(path)
    assert cfg.name in ("crow-300m-final", "tiny", "crow-300m-mini")
    assert cfg.stages and cfg.sources


def test_mini_config_is_the_final_config_with_tiny_budgets() -> None:
    """The real-source smoke config differs from the thesis config only in name, budgets and validation size."""
    final, mini = load_dataset_config(CROW), load_dataset_config(MINI)
    assert mini.name == "crow-300m-mini"
    assert [s.tokens for s in mini.stages] == [300_000, 150_000, 60_000]
    assert [(s.name, s.train, s.val, s.transition_pct) for s in mini.stages] == [
        (s.name, s.train, s.val, s.transition_pct) for s in final.stages
    ]
    assert set(mini.sources) == set(final.sources)
    for name, source in final.sources.items():
        expected = replace(source, validation_tokens=40_000) if source.validation_tokens else source
        assert mini.sources[name] == expected, name
    assert mini.sources["fineweb_edu"].validation_tokens == 40_000
    assert mini.instruct_mixtures == final.instruct_mixtures and mini.tokenizer == final.tokenizer
    assert (mini.processing, mini.max_seq_length, mini.token_count) == (final.processing, final.max_seq_length, final.token_count)


def test_crow_config_matches_thesis_run() -> None:
    cfg = load_dataset_config(CROW)
    assert [s.name for s in cfg.stages] == ["pretrain_phase1", "pretrain_phase2", "finetune"]
    assert [s.tokens for s in cfg.stages] == [3_300_000_000, 1_500_000_000, 150_000_000]
    assert len(cfg.sources_of_kind("pretrain")) == 19
    assert len(cfg.sources_of_kind("instruct")) == 8
    assert cfg.sources_of_kind("validation") == []
    assert cfg.sources["fineweb_edu"].validation_tokens == 50_000_000
    assert all(stage.val == {"fineweb_edu/validation": 1.0} for stage in cfg.stages[:2])
    assert cfg.token_count == "tokenizer" and cfg.max_seq_length == 2048
    assert cfg.processing.dedup.mode == "exact" and not cfg.processing.quality_filter
    assert not cfg.processing.decontamination.enabled
    assert all(s.revision for s in cfg.sources.values()), "every Hub source must pin a revision"
    assert cfg.sources["books_gutenberg"].text_field == "TEXT"
    assert cfg.sources["gsm8k"].converter == "gsm8k_question_answer"
    assert cfg.instruct_mixtures["flan_instruct"].input_inversions == 0.05


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
        (lambda d: d["stages"][0]["train"].update({"hold": 0.0}), "validation"),
        (lambda d: d["stages"][0]["train"].update({"ins": 0.0}), "instruct source"),
        (lambda d: d["stages"][0]["train"].update({"pre/train": 0.0}), "take a split"),
        (lambda d: d["stages"][0]["val"].update({"pre/validation": 0.0}), "needs validation_tokens > 0"),
        (lambda d: (d["sources"]["pre"].__setitem__("validation_tokens", 10), d["stages"][0]["train"].update({"pre/validation": 0.0})), "cannot be used for training"),
        (lambda d: d["sources"]["hold"].__setitem__("validation_tokens", 10), "only applies to kind pretrain"),
        (lambda d: d["sources"]["pre"].__setitem__("validation_tokens", -1), "validation_tokens must be >= 0"),
        (lambda d: d["stages"][1]["val"].update({"mix/test": 0.0}), "/validation"),
        (lambda d: d["stages"][1]["val"].update({"mix": 0.0}), "would validate on the train split"),
        (lambda d: d["stages"][1]["val"].update({"mix/train": 0.0}), "would validate on the train split"),
        (lambda d: d["sources"].update({"only_val": {"kind": "pretrain", "loader": "synthetic"}}) or d["stages"][0]["val"].update({"only_val": 0.0}), "never for training"),
        (lambda d: d["instruct_mixtures"]["mix"]["sources"].update({"pre": 0.0}), "not kind instruct"),
        (lambda d: d["instruct_mixtures"]["mix"]["sources"].update({"zzz": 0.0}), "unknown source"),
        (lambda d: d["instruct_mixtures"].update({"pre": {"sources": {"ins": 1.0}}}), "shared by sources and instruct mixtures"),
        (lambda d: d["stages"].append(dict(d["stages"][0])), "unique"),
        (lambda d: d["stages"].clear(), "at least one stage"),
        (lambda d: d.update({"name": "a/b"}), "path component"),
        (lambda d: d.update({"max_seq_length": 0}), "max_seq_length"),
        (lambda d: d["stages"][0].update({"tokens": 0}), "tokens must be positive"),
        (lambda d: d["stages"][0].update({"transition_pct": 1.0}), "transition_pct"),
        (lambda d: d["stages"][0].update({"train": {}}), "must not be empty"),
        (lambda d: d["stages"][0]["train"].update({"pre": -1.0, "hold": 2.0}), "non-negative"),
        (lambda d: d["sources"]["hold"].pop("rows"), "validation requires rows"),
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
        (lambda d: d["instruct_mixtures"]["mix"].update({"val_split": 1.0}), "val_split"),
        (lambda d: d["instruct_mixtures"]["mix"].update({"input_inversions": 1.5}), "input_inversions"),
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


def test_instruct_mixture_budget_and_instruct_source_budget() -> None:
    d = _minimal()
    d["sources"]["ins2"] = {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/z", "converter": "first_two_turns"}
    d["instruct_mixtures"]["mix"]["sources"] = {"ins": 0.75, "ins2": 0.25}
    d["stages"][1]["tokens"] = 1000
    d["stages"][1]["train"] = {"mix/train": 1.0}
    cfg = _build(d)
    assert cfg.instruct_mixture_budget_tokens("mix") == 1000
    assert cfg.source_budget_tokens("ins") == 750 and cfg.source_budget_tokens("ins2") == 250
    assert cfg.instruct_mixture_budget_tokens("unused") == 0


def test_sources_of_kind() -> None:
    cfg = _build(_minimal())
    assert cfg.sources_of_kind("pretrain") == ["pre"]
    assert cfg.sources_of_kind("validation") == ["hold"]
    assert cfg.sources_of_kind("instruct") == ["ins"]


# --- hashes -----------------------------------------------------------------------------------------------------------


def test_max_cached_file_mb_validated_and_not_hashed() -> None:
    d = _minimal()
    d["sources"]["files"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}
    h = _build(d).raw_hash("files")
    d["sources"]["files"]["load_kwargs"]["max_cached_file_mb"] = 1.5
    assert _build(d).raw_hash("files") == h  # how a file is fetched does not change its rows
    d["sources"]["files"]["load_kwargs"]["max_cached_file_mb"] = 0
    assert _build(d).raw_hash("files") == h
    for bad in (-1, "big", True):
        d["sources"]["files"]["load_kwargs"]["max_cached_file_mb"] = bad
        with pytest.raises(ValueError, match="max_cached_file_mb"):
            _build(d)


def test_source_hash_stable_across_key_order_and_reloads() -> None:
    a = load_dataset_config(TINY)
    b = load_dataset_config(TINY)
    assert a.raw_hash("synthetic_pretrain") == b.raw_hash("synthetic_pretrain")
    assert a.processed_hash("synthetic_pretrain") == b.processed_hash("synthetic_pretrain")
    assert a.config_hash() == b.config_hash()
    assert dc._stable_hash({"x": 1, "y": [1, 2]}) == dc._stable_hash({"y": [1, 2], "x": 1})


def test_hash_fields_drops_defaults_recursively() -> None:
    """Schema changes with defaults must not invalidate data on disk: only explicitly set values are hashed."""
    assert dc.hash_fields(DedupConfig()) == {}
    assert dc.hash_fields(DedupConfig(threshold=0.5)) == {"threshold": 0.5}
    proc = ProcessingConfig(min_chars=7, dedup=DedupConfig(mode="minhash"))
    assert dc.hash_fields(proc) == {"min_chars": 7, "dedup": {"mode": "minhash"}}
    src = SourceConfig(kind="pretrain", loader="hf_files", hf_id="x/y", load_kwargs={"data_files": "*.parquet"})
    assert dc.hash_fields(src) == {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}


def test_raw_hash_only_tracks_the_loader_identity() -> None:
    """The raw shards are the bandwidth-expensive part: nothing but a real change of the source may invalidate them."""
    base = _build(_minimal())
    h = base.raw_hash("pre")

    changes: list[Callable[[dict[str, Any]], None]] = [
        lambda d: d["stages"][0].__setitem__("tokens", 999_999),  # budget: same rows on disk
        lambda d: d["sources"]["pre"].__setitem__("tokens_per_row_estimate", 3),  # planner prior
        lambda d: d["sources"]["pre"].__setitem__("processing", {"min_chars": 99}),  # processing → processed only
        lambda d: d["sources"]["pre"].__setitem__("processing", {"dedup": {"mode": "minhash"}}),
        lambda d: d.__setitem__("processing", {"quality_filter": True, "decontamination": {"enabled": True}}),
        lambda d: d.__setitem__("token_count", "estimate"),  # tokens column is recounted in place
        lambda d: d.__setitem__("max_seq_length", 64),
        lambda d: d.__setitem__("tokenizer", {"name": "other", "kind": "hf", "hf_id": "a/b"}),
        lambda d: d["sources"]["pre"].__setitem__("check_limit", 5),  # bounds how far to read, not what is read
    ]
    for change in changes:
        d = _minimal()
        change(d)
        assert _build(d).raw_hash("pre") == h, change

    invalidating: list[Callable[[dict[str, Any]], None]] = [
        lambda d: d["sources"]["pre"].__setitem__("seed", 5),
        lambda d: d["sources"]["pre"].__setitem__("text_field", "body"),
        lambda d: d["sources"]["pre"].__setitem__("converter", "gsm8k_question_answer"),
        lambda d: d["sources"]["pre"].__setitem__("split", "test"),
    ]
    for change in invalidating:
        d = _minimal()
        change(d)
        assert _build(d).raw_hash("pre") != h, change


def test_processed_hash_tracks_processing_and_token_settings() -> None:
    base = _build(_minimal())
    h = base.processed_hash("pre")
    assert h != base.raw_hash("pre")

    d = _minimal()
    d["stages"][0]["tokens"] = 999_999
    d["sources"]["pre"]["tokens_per_row_estimate"] = 3
    assert _build(d).processed_hash("pre") == h

    changes: list[Callable[[dict[str, Any]], None]] = [
        lambda d: d["sources"]["pre"].__setitem__("processing", {"min_chars": 99}),
        lambda d: d.__setitem__("processing", {"max_chars": 99}),
        lambda d: d.__setitem__("token_count", "estimate"),
        lambda d: d.__setitem__("max_seq_length", 64),
        lambda d: d["sources"]["pre"].__setitem__("seed", 5),
    ]
    for change in changes:
        d = _minimal()
        change(d)
        assert _build(d).processed_hash("pre") != h, change

    d = _minimal()
    d["max_seq_length"] = 64
    assert _build(d).validation_hash("hold") != base.validation_hash("hold")  # the cap changes the stored counts
    assert _build(d).raw_hash("hold") == base.raw_hash("hold")


def test_stage_hash_dispatches() -> None:
    cfg = _build(_minimal())
    assert cfg.stage_hash("pre", "raw") == cfg.raw_hash("pre")
    assert cfg.stage_hash("pre", "processed") == cfg.processed_hash("pre")
    assert cfg.stage_hash("hold", "validation") == cfg.validation_hash("hold")
    with pytest.raises(ValueError, match="unknown source stage"):
        cfg.stage_hash("pre", "filtered")


def test_hash_fields_golden_defaults() -> None:
    """`hash_fields` drops default-valued fields, so a changed *default* re-labels data built under the old one.
    Changing any of these defaults must be a conscious, hash-breaking commit: update this golden dict with it."""
    assert asdict(ProcessingConfig()) == {
        "min_chars": 50,
        "max_chars": 20000,
        "dedup": {"mode": "exact", "normalize": True, "threshold": 0.95, "num_perm": 256, "ngram": 5},
        "quality_filter": False,
        "decontamination": {"enabled": False, "benchmarks": list(dc.DEFAULT_BENCHMARKS), "ngram": 13, "threshold": 0.1},
    }
    src = SourceConfig(kind="pretrain", loader="synthetic")
    assert {f: getattr(src, f) for f in ("split", "text_field", "seed", "tokens_per_row_estimate")} == {
        "split": "train", "text_field": "text", "seed": 42, "tokens_per_row_estimate": 500,
    }  # fmt: skip
    cfg = _build(_minimal())
    assert (cfg.max_seq_length, cfg.token_count, cfg.always_range_requests) == (2048, "tokenizer", True)


def test_tokenizer_change_affects_hash_only_when_counting_with_it() -> None:
    d = _minimal()
    base = _build(d)
    d["tokenizer"] = {"name": "other", "kind": "hf", "hf_id": "a/b"}
    assert _build(d).processed_hash("pre") != base.processed_hash("pre")
    d["token_count"] = "estimate"
    d2 = copy.deepcopy(d)
    d2["tokenizer"] = {"name": "third", "kind": "hf", "hf_id": "c/d"}
    assert _build(d).processed_hash("pre") == _build(d2).processed_hash("pre")
    assert _build(d).instruct_mixture_hash("mix") != _build(d2).instruct_mixture_hash("mix")  # instruct always tokenizes
    assert _build(d).raw_hash("ins") == _build(d2).raw_hash("ins")


def test_instruct_mixture_hash_tracks_sources_and_budget() -> None:
    base = _build(_minimal())
    h = base.instruct_mixture_hash("mix")
    d = _minimal()
    d["sources"]["ins"]["fields"] = {"instruction": "q", "output": "b"}
    assert _build(d).instruct_mixture_hash("mix") != h
    d = _minimal()
    d["stages"][1]["tokens"] = 501
    assert _build(d).instruct_mixture_hash("mix") != h
    d = _minimal()
    d["instruct_mixtures"]["mix"]["val_split"] = 0.1
    assert _build(d).instruct_mixture_hash("mix") != h


def test_config_hash_changes_on_any_field() -> None:
    base = _build(_minimal()).config_hash()
    d = _minimal()
    d["stages"][0]["transition_pct"] = 0.5
    assert _build(d).config_hash() != base
    assert len(base) == 16


def test_dataset_config_fields_and_asdict_roundtrip() -> None:
    assert {"name", "tokenizer", "sources", "stages", "instruct_mixtures", "processing"} <= set(dc.dataset_config_fields())
    cfg = _build(_minimal())
    assert asdict(cfg)["sources"]["pre"]["kind"] == "pretrain"


def test_validation_split_key_hash_and_overlap_warning() -> None:
    d = _minimal()
    d["sources"]["pre"]["validation_tokens"] = 100
    d["stages"][0]["val"] = {"pre/validation": 1.0}
    cfg = _build(d)
    assert cfg.stages[0].val == {"pre/validation": 1.0}
    base = _build(_minimal())
    assert cfg.raw_hash("pre") == base.raw_hash("pre"), "the split is a product of processing, raw is untouched"
    assert cfg.processed_hash("pre") != base.processed_hash("pre")
    assert cfg.stage_hash("pre", "validation") == cfg.processed_hash("pre")
    assert cfg.overlap_warnings() == []

    d = _minimal()
    d["sources"]["pre"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "org/repo", "load_kwargs": {"data_files": "data/*.parquet"}}
    d["sources"]["hold"] = {"kind": "validation", "loader": "hf_files", "hf_id": "org/repo", "load_kwargs": {"data_files": "data/sample/*.parquet"}, "rows": 5}
    (warning,) = _build(d).overlap_warnings()
    assert "'hold'" in warning and "'pre'" in warning and "may overlap" in warning
    d["sources"]["hold"]["load_kwargs"] = {"data_files": "other/*.parquet"}
    assert _build(d).overlap_warnings() == []
    assert dc._glob_prefix("data/CC-MAIN-2013-20/*.parquet") == "data/CC-MAIN-2013-20/" and dc._glob_prefix(None) == ""


def test_config_hash_ignores_fetch_and_planner_knobs() -> None:
    d = _minimal()
    d["sources"]["files"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}
    base = _build(d).config_hash()
    d["sources"]["files"]["tokens_per_row_estimate"] = 7
    d["sources"]["files"]["load_kwargs"]["max_cached_file_mb"] = 3
    cfg = _build(d)
    cfg.always_range_requests = False
    assert cfg.config_hash() == base
    d["sources"]["files"]["revision"] = "abc"
    assert _build(d).config_hash() != base
