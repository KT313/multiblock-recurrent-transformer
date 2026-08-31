# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.dataset_config: loading the shipped configs, every validation rule, source usage,
budgets and the raw / processed hash invariants."""

from __future__ import annotations

import copy
import dataclasses
import re
from collections.abc import Callable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from data_preparation import dataset_config as dc
from data_preparation.dataset_config import (
    DatasetConfig,
    DedupConfig,
    ProcessingConfig,
    SourceConfig,
    StageConfig,
    TokenizerConfig,
    load_dataset_config,
)

REPO = Path(__file__).resolve().parents[1]
CROW = REPO / "config" / "datasets" / "crow_300m_final.yaml"
TINY = REPO / "config" / "datasets" / "tiny.yaml"
MINI = REPO / "config" / "datasets" / "crow_300m_mini.yaml"

FINETUNE_SHARES = {"flan": 0.40, "metamath": 0.15, "orca_math": 0.10, "evol_code": 0.125, "code_alpaca": 0.025, "slimorca": 0.10, "sharegpt": 0.05, "wizardlm": 0.05}

Mutation = Callable[[dict[str, Any]], Any]


def _minimal() -> dict[str, Any]:
    """A small valid config as a plain dict (mutated by the validation tests): `pre` is trained and validated on
    (split), `hold` is validation-only (needs `rows`), `ins` is an instruct source trained and validated on."""
    return {
        "name": "t",
        "tokenizer": {"name": "synthetic", "kind": "synthetic"},
        "block_size": 64,
        "sources": {
            "pre": {"kind": "pretrain", "loader": "synthetic"},
            "hold": {"kind": "pretrain", "loader": "synthetic", "rows": 10},
            "ins": {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/y", "fields": {"instruction": "a", "output": "b"}},
        },
        "stages": [
            {"name": "s1", "tokens": 1000, "train": {"pre": 1.0}, "val": {"hold": 0.5, "pre": 0.5}},
            {"name": "s2", "tokens": 500, "train": {"ins": 1.0}, "val": {"ins": 1.0}},
        ],
    }


def _build(d: dict[str, Any]) -> DatasetConfig:
    d = copy.deepcopy(d)
    sources = {k: SourceConfig(**({**v, "processing": _processing(v["processing"])} if v.get("processing") else v))
               for k, v in d["sources"].items()}
    return DatasetConfig(
        name=d["name"],
        tokenizer=TokenizerConfig(**d["tokenizer"]),
        sources=sources,
        stages=[StageConfig(**s) for s in d["stages"]],
        block_size=d["block_size"],
        max_seq_length=d.get("max_seq_length", 2048),
        validation_fraction=d.get("validation_fraction", 0.05),
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


def _write(tmp_path: Path, d: dict[str, Any]) -> Path:
    path = tmp_path / "d.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


# --- shipped files ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [CROW, TINY, MINI])
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_dataset_config(path)
    assert cfg.name in ("crow-300m-final", "tiny", "crow-300m-mini")
    assert cfg.stages and cfg.sources
    assert cfg.block_size <= cfg.max_seq_length


def test_mini_config_is_the_final_config_with_tiny_budgets() -> None:
    """The real-source smoke config differs from the thesis config only in name and budgets."""
    final, mini = load_dataset_config(CROW), load_dataset_config(MINI)
    assert mini.name == "crow-300m-mini"
    assert [s.tokens for s in mini.stages] == [300_000, 150_000, 60_000]
    assert [(s.name, s.train, s.val, s.transition_pct) for s in mini.stages] == [
        (s.name, s.train, s.val, s.transition_pct) for s in final.stages
    ]
    assert mini.sources == final.sources and mini.tokenizer == final.tokenizer
    assert (mini.processing, mini.max_seq_length, mini.block_size, mini.token_count, mini.validation_fraction) == (
        final.processing, final.max_seq_length, final.block_size, final.token_count, final.validation_fraction
    )


def test_crow_config_matches_thesis_run() -> None:
    cfg = load_dataset_config(CROW)
    assert [s.name for s in cfg.stages] == ["pretrain_phase1", "pretrain_phase2", "finetune"]
    assert [s.tokens for s in cfg.stages] == [3_300_000_000, 1_500_000_000, 150_000_000]
    assert len(cfg.sources_of_kind("pretrain")) == 19
    assert len(cfg.sources_of_kind("instruct")) == 8
    assert all(stage.val == {"fineweb_edu": 1.0} for stage in cfg.stages[:2])
    assert cfg.stages[2].train == FINETUNE_SHARES and cfg.stages[2].val == FINETUNE_SHARES
    assert all(cfg.sources[name].input_inversions == 0.05 for name in cfg.sources_of_kind("instruct"))
    assert all(cfg.sources[name].input_inversions == 0.0 for name in cfg.sources_of_kind("pretrain"))
    assert (cfg.token_count, cfg.max_seq_length, cfg.block_size, cfg.validation_fraction) == ("tokenizer", 2048, 2048, 0.05)
    assert cfg.processing.dedup.mode == "exact" and cfg.processing.dedup.bloom_memory_mb == 1024
    assert not cfg.processing.quality_filter and not cfg.processing.decontamination.enabled
    assert all(s.revision for s in cfg.sources.values()), "every Hub source must pin a revision"
    assert all(s.rows is None for s in cfg.sources.values()), "every crow source is trained on"
    assert cfg.sources["books_gutenberg"].text_field == "TEXT"
    assert cfg.sources["gsm8k"].converter == "gsm8k_question_answer"
    assert cfg.validation_fraction_of("fineweb_edu") == 0.05 and cfg.validation_fraction_of("wikipedia") == 0.0
    assert cfg.validation_fraction_of("flan") == 0.05 and cfg.shuffle_of("flan") and not cfg.shuffle_of("fineweb_edu")


def test_tiny_config_is_synthetic_only() -> None:
    cfg = load_dataset_config(TINY)
    assert cfg.tokenizer.kind == "synthetic"
    assert {s.loader for s in cfg.sources.values()} == {"synthetic"}
    assert (cfg.max_seq_length, cfg.block_size) == (256, 256)
    assert set(cfg.sources) == {"synthetic_pretrain", "synthetic_instruct"}
    assert cfg.sources["synthetic_instruct"].input_inversions == 0.1
    assert all(cfg.used_in_train(n) and cfg.used_in_val(n) for n in cfg.sources)
    assert cfg.stages[2].train == {"synthetic_instruct": 1.0} and cfg.stages[2].val == {"synthetic_instruct": 1.0}


def test_overrides_apply_to_nested_keys() -> None:
    cfg = load_dataset_config(TINY, ["--max_seq_length", "512", "--processing.dedup.mode", "none"])
    assert cfg.max_seq_length == 512 and cfg.processing.dedup.mode == "none"


def test_load_from_written_yaml(tmp_path: Path) -> None:
    cfg = load_dataset_config(_write(tmp_path, _minimal()))
    assert cfg.sources["hold"].rows == 10 and cfg.stages[1].train == {"ins": 1.0} and cfg.block_size == 64


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d.update({"instruct_mixtures": {"mix": {"sources": {"ins": 1.0}}}}), r"instruct_mixtures.*\n.*`instruct_mixtures` was removed: mixing"),
        (lambda d: d["sources"]["pre"].update({"validation_tokens": 5}), r"validation_tokens.*\n?.*was removed"),
        (lambda d: d.update({"processing": {"max_chars": 5}}), r"max_chars.*\n?.*was removed"),
        (lambda d: d["sources"]["pre"].update({"tokens_per_row_estimate": 5}), r"tokens_per_row_estimate.*\n?.*describe_tokens_per_row"),
        (lambda d: d["sources"]["hold"].update({"kind": "validation"}), r"Literal\['pretrain', 'instruct'\]"),
        (lambda d: d.pop("block_size"), r"required: block_size"),
        (lambda d: d.update({"bogus": 1}), r"d\.yaml: Option 'bogus' is not accepted$"),
    ],
)
def test_unknown_or_removed_keys_fail_loading_with_a_clear_error(tmp_path: Path, mutate: Mutation, match: str) -> None:
    """Removed keys must not be silently ignored: loading raises a ValueError naming the file and the key (plus a
    hint for the keys of the old schema), never jsonargparse's usage dump + `sys.exit(2)`."""
    d = _minimal()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        load_dataset_config(_write(tmp_path, d))


# --- validation rules -------------------------------------------------------------------------------------------------


def test_minimal_is_valid() -> None:
    _build(_minimal())


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d["stages"][0]["train"].update({"pre": 0.5}), "sum to"),
        (lambda d: d["stages"][0]["train"].update({"pre": 1.0, "ins": 0.0}), "weights must be > 0"),
        (lambda d: d["stages"][0]["val"].update({"hold": 0.0, "pre": 1.0}), "weights must be > 0"),
        (lambda d: d["stages"][0]["train"].update({"pre": 2.0, "ins": -1.0}), "weights must be > 0"),
        (lambda d: d["stages"][0]["train"].update({"pre": 0.5, "nope": 0.5}), "unknown source 'nope'"),
        (lambda d: d["stages"][0].update({"val": {"pre/validation": 0.5, "hold": 0.5}}), "plain source names"),
        (lambda d: d["stages"][0]["train"].update({"pre": 0.5, "pre/train": 0.5}), "plain source names"),
        (lambda d: d.update({"block_size": 0}), "block_size must be positive"),
        (lambda d: d.update({"block_size": 4096}), r"block_size \(4096\) must be <= max_seq_length \(2048\)"),
        (lambda d: d.update({"max_seq_length": 0}), "max_seq_length"),
        (lambda d: d.update({"validation_fraction": 1.0}), r"validation_fraction must be in \[0, 1\)"),
        (lambda d: d.update({"validation_fraction": -0.1}), r"validation_fraction must be in \[0, 1\)"),
        (lambda d: d["sources"]["pre"].update({"validation_fraction": 1.0}), r"validation_fraction must be in \[0, 1\)"),
        (lambda d: d["sources"]["pre"].update({"rows": 5}), "used for training.*drop `rows`"),
        (lambda d: d["sources"]["ins"].update({"rows": 5}), "used for training.*drop `rows`"),
        (lambda d: d["sources"]["hold"].pop("rows"), "used only for validation: give `rows`"),
        (lambda d: d["sources"]["hold"].update({"rows": 0}), "rows must be positive"),
        (lambda d: d["sources"].update({"extra": {"kind": "pretrain", "loader": "synthetic"}}), "'extra' is used by no stage"),
        (lambda d: d["sources"]["pre"].update({"input_inversions": 0.1}), "input_inversions only applies to kind instruct"),
        (lambda d: d["sources"]["ins"].update({"input_inversions": 1.0}), r"input_inversions must be in \[0, 1\)"),
        (lambda d: d["sources"]["ins"].update({"input_inversions": -0.1}), r"input_inversions must be in \[0, 1\)"),
        (lambda d: d["sources"]["pre"].update({"describe_tokens_per_row": 0}), "describe_tokens_per_row"),
        (lambda d: d["sources"]["ins"].update({"processing": {"min_chars": 1}}), "processing only applies to kind pretrain"),
        (lambda d: d["sources"]["ins"].pop("fields"), "requires fields or converter"),
        (lambda d: d["sources"]["ins"].update({"fields": {"instruction": "a"}}), "instruction and output"),
        (lambda d: d["sources"]["ins"].pop("hf_id"), "requires hf_id"),
        (lambda d: d["sources"]["pre"].update({"loader": "github_code", "hf_id": "o/r"}), "requires language"),
        (lambda d: d["sources"]["pre"].update({"loader": "github_code", "language": "Python"}), "requires hf_id"),
        (lambda d: d["sources"]["pre"].update({"loader": "hf_files", "hf_id": "o/r"}), "requires load_kwargs.data_files"),
        (lambda d: d["sources"]["pre"].update({"loader": "hf_files", "hf_id": "o/r", "load_kwargs": {"data_files": 3}}), "requires load_kwargs.data_files"),
        (lambda d: d["sources"]["pre"].update({"loader": "local"}), "requires path"),
        (lambda d: d["stages"].append(dict(d["stages"][0])), "unique"),
        (lambda d: d["stages"].clear(), "at least one stage"),
        (lambda d: d.update({"name": "a/b"}), "path component"),
        (lambda d: d["stages"][0].update({"tokens": 0}), "tokens must be positive"),
        (lambda d: d["stages"][0].update({"transition_pct": 1.0}), "transition_pct"),
        (lambda d: d["stages"][0].update({"train": {}}), "must not be empty"),
        (lambda d: d.update({"tokenizer": {"name": "x", "kind": "hf"}}), "requires hf_id"),
    ],
)
def test_validation_rejects(mutate: Mutation, match: str) -> None:
    d = _minimal()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        _build(d)


# --- the field scope table --------------------------------------------------------------------------------------------


def test_field_scopes_cover_every_source_field() -> None:
    """A new `SourceConfig` field must state where it applies. Without an entry it applies nowhere, so this test
    (and every config that sets it) fails instead of the field being silently accepted for every kind and loader."""
    assert {f.name for f in fields(SourceConfig)} == set(dc.SOURCE_FIELD_SCOPES)
    for name, scope in dc.SOURCE_FIELD_SCOPES.items():
        assert scope.kinds <= dc.ALL_KINDS and scope.loaders <= dc.ALL_LOADERS, name
        assert scope.kinds and scope.loaders, name


def test_a_field_missing_from_the_table_applies_nowhere(monkeypatch: pytest.MonkeyPatch) -> None:
    scopes = {k: v for k, v in dc.SOURCE_FIELD_SCOPES.items() if k != "text_field"}
    monkeypatch.setattr(dc, "SOURCE_FIELD_SCOPES", scopes)
    with pytest.raises(ValueError, match="text_field only applies to no kind or loader"):
        SourceConfig(kind="pretrain", loader="synthetic", text_field="body")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "instruct", "loader": "synthetic", "text_field": "body"}, "text_field only applies to kind pretrain"),
        ({"kind": "pretrain", "loader": "synthetic", "fields": {"instruction": "a", "output": "b"}}, "fields only applies to kind instruct"),
        ({"kind": "pretrain", "loader": "synthetic", "filter": "sharegpt_quality"}, "filter only applies to kind instruct"),
        ({"kind": "pretrain", "loader": "local", "path": "p", "language": "Python"}, "language only applies to loader github_code"),
        ({"kind": "pretrain", "loader": "synthetic", "path": "p"}, "path only applies to loader local"),
        ({"kind": "pretrain", "loader": "synthetic", "hf_id": "o/r"}, "hf_id only applies to loader github_code/hf_files/hf_split/hf_stream"),
        ({"kind": "pretrain", "loader": "local", "path": "p", "revision": "abc"}, "revision only applies to loader"),
        ({"kind": "pretrain", "loader": "local", "path": "p", "load_kwargs": {"name": "cfg"}}, "load_kwargs only applies to loader"),
    ],
)
def test_a_field_set_outside_its_scope_names_the_field_and_the_rule(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=re.escape(match)):
        SourceConfig(**kwargs)


def test_a_field_at_its_default_is_never_out_of_scope() -> None:
    """Only a value a config actually chose is checked, so an instruct source is not rejected for the `text_field`
    default it never mentioned."""
    source = SourceConfig(kind="instruct", loader="synthetic")
    assert (source.text_field, source.split, source.input_inversions) == ("text", "train", 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": 1.5}, "threshold"),
        ({"num_perm": 0}, "positive"),
        ({"ngram": 0}, "positive"),
        ({"bloom_memory_mb": 0}, "bloom_memory_mb must be positive"),
    ],
)
def test_dedup_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DedupConfig(**kwargs)


def test_processing_validation() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        ProcessingConfig(min_chars=-1)
    assert ProcessingConfig(min_chars=0).min_chars == 0


def test_weights_tolerate_float_noise() -> None:
    d = _minimal()
    d["stages"][0]["train"] = {"pre": 0.1 + 0.2 + 0.7}  # 1.0000000000000002
    _build(d)


# --- source usage, split, budgets -------------------------------------------------------------------------------------


def test_used_in_train_and_val() -> None:
    cfg = _build(_minimal())
    assert (cfg.used_in_train("pre"), cfg.used_in_val("pre")) == (True, True)
    assert (cfg.used_in_train("hold"), cfg.used_in_val("hold")) == (False, True)
    assert (cfg.used_in_train("ins"), cfg.used_in_val("ins")) == (True, True)


def test_shuffle_of_defaults_by_kind_and_explicit_value_wins() -> None:
    cfg = _build(_minimal())
    assert not cfg.shuffle_of("pre") and not cfg.shuffle_of("hold") and cfg.shuffle_of("ins")
    d = _minimal()
    d["sources"]["pre"]["shuffle"] = True
    d["sources"]["ins"]["shuffle"] = False
    cfg = _build(d)
    assert cfg.shuffle_of("pre") and not cfg.shuffle_of("ins")


def test_validation_fraction_of() -> None:
    d = _minimal()
    d["sources"]["only_train"] = {"kind": "pretrain", "loader": "synthetic"}
    d["stages"][0]["train"] = {"pre": 0.5, "only_train": 0.5}
    cfg = _build(d)
    assert cfg.validation_fraction_of("pre") == 0.05  # dataset default: in train and val
    assert cfg.validation_fraction_of("hold") == 0.0  # val only: all rows are validation
    assert cfg.validation_fraction_of("only_train") == 0.0  # train only: all rows are training
    d["validation_fraction"] = 0.1
    d["sources"]["pre"]["validation_fraction"] = 0.2
    d["sources"]["hold"]["validation_fraction"] = 0.3  # ignored: not in both
    d["sources"]["only_train"]["validation_fraction"] = 0.3  # ignored: not in both
    cfg = _build(d)
    assert cfg.validation_fraction_of("pre") == 0.2 and cfg.validation_fraction_of("ins") == 0.1
    assert cfg.validation_fraction_of("hold") == 0.0 and cfg.validation_fraction_of("only_train") == 0.0


def test_sequence_budget_is_the_max_over_stages_in_block_size_units() -> None:
    d = _minimal()
    d["sources"]["pre2"] = {"kind": "pretrain", "loader": "synthetic"}
    d["stages"][0]["train"] = {"pre": 0.6, "pre2": 0.4}
    d["stages"][1] = {"name": "s2", "tokens": 4000, "train": {"pre": 0.1, "pre2": 0.9}, "val": {"pre": 1.0}}
    d["stages"].append({"name": "s3", "tokens": 500, "train": {"ins": 1.0}, "val": {"ins": 1.0}})
    cfg = _build(d)  # block_size 64
    assert cfg.sequence_budget("pre") == max(-(-600 // 64), -(-400 // 64)) == 10
    assert cfg.sequence_budget("pre2") == max(-(-400 // 64), -(-3600 // 64)) == 57
    assert cfg.sequence_budget("ins") == 8  # ceil(500 / 64)
    assert cfg.sequence_budget("hold") == 0  # validation only: `rows` says how many to download
    d["block_size"] = 32
    assert _build(d).sequence_budget("pre2") == 113  # ceil(3600 / 32)


def test_sequence_budget_of_the_crow_config() -> None:
    cfg = load_dataset_config(CROW)
    assert cfg.sequence_budget("fineweb_edu") == -(-int(3_300_000_000 * 0.65) // 2048) == 1_047_364
    assert cfg.sequence_budget("gsm8k") == -(-int(1_500_000_000 * 0.022) // 2048) == 16_114
    assert cfg.sequence_budget("flan") == -(-int(150_000_000 * 0.40) // 2048) == 29_297


def test_source_processing_override() -> None:
    d = _minimal()
    d["sources"]["pre"]["processing"] = {"min_chars": 7, "quality_filter": True}
    cfg = _build(d)
    assert cfg.source_processing("pre").min_chars == 7 and cfg.source_processing("pre").quality_filter
    assert cfg.source_processing("pre") is not cfg.processing
    cfg = _build(_minimal())
    assert cfg.source_processing("pre") is cfg.processing and cfg.source_processing("ins") is cfg.processing


def test_sources_of_kind() -> None:
    cfg = _build(_minimal())
    assert cfg.sources_of_kind("pretrain") == ["pre", "hold"]
    assert cfg.sources_of_kind("instruct") == ["ins"]


# --- hashes -----------------------------------------------------------------------------------------------------------

# The hashes of the shipped configs, recorded 2026-09-01. They key data on disk: every folder under `dataset/`
# carries the `raw_hash` / `processed_hash` of the source that produced it (raw folders are the bandwidth-expensive
# part — terabytes on the author's machine — and a mismatch makes one stale, i.e. re-downloaded after confirmation),
# and `config_hash` is what a training checkpoint stores to detect a resume against different data. Refactoring *how*
# the hashes are derived must keep every value below byte-identical; only a deliberate change of *what* a hash counts
# may re-record them, in a commit that says so and accepts that the data on disk are invalidated.
PINNED_HASHES: dict[str, dict[str, Any]] = {
    "tiny": {
        "config": "e828f8d6061c50ae",
        "tokenizer": "cacedea880d7ec37",
        "sources": {
            "synthetic_pretrain": ("f9645dcd93c92e0e", "ca25d628bc0fc423"),
            "synthetic_instruct": ("7a5485356346a9a9", "af48321fbeb25a31"),
        },
    },
    "crow_300m_final": {
        "config": "56a0d60c20c73e91",
        "tokenizer": "128ca0c738434e53",
        "sources": {
            "fineweb_edu": ("04d5d6c0f4ef9279", "b58c044118744cc9"),
            "wikipedia": ("a614ffc5b692de0d", "f37150fbae7ecae3"),
            "books_gutenberg": ("d496b0abd0decd96", "f3c14b65b476c1d2"),
            "peso": ("557afee219124346", "751c373ccb3ac143"),
            "arxiv": ("2d9cbb89d06c547c", "981e30bb496e8308"),
            "openwebmath": ("0ff260c644bbeefb", "ddb2f1336c1256f0"),
            "tinygsm": ("ca8573c25a3a5d1e", "4034a2c025a97177"),
            "algebraic_stack": ("4566347616d999a5", "21efcd652ad03986"),
            "gsm8k": ("df9a15f2b11e2778", "3ffc84246ac300b1"),
            "github_code_clean_python": ("bb88f17dfdaebcff", "700d56cdc7bc9a68"),
            "github_code_clean_javascript": ("6ca946aca0178495", "c9f4ab481591e822"),
            "github_code_clean_typescript": ("0ddb1168bcb4a4ab", "f34c443073fae9d9"),
            "github_code_clean_java": ("0ff50c78d34a5090", "fef4875b7cc60f2a"),
            "github_code_clean_cpp": ("18981d711ba70d0d", "a3b72c190b3b78f4"),
            "github_code_clean_go": ("8b754a6222c3c353", "24b2488bc38d8d0a"),
            "github_code_clean_rust": ("a3d0abda4164a28b", "f997daeef4dfb4d2"),
            "github_code_clean_shell": ("9ec8395c504ab595", "7be5c8aaf54f400e"),
            "github_code_clean_sql": ("d1c5d2ae0a07b2a3", "4554146610fbb315"),
            "github_code_clean_html": ("77dd5d781104dac8", "4cdbab97251f865a"),
            "flan": ("d2ed4c0d9137827a", "a4a0c70625517542"),
            "metamath": ("fbabb8961fdf87ec", "6fca33db6fc6146a"),
            "orca_math": ("4c614e6f6c99cd8d", "559c41d6dc1f9cfa"),
            "evol_code": ("d5b8a54fa3e41a6b", "c86350f441bd855e"),
            "code_alpaca": ("52240fa45709bde5", "ffd74a14ac575a96"),
            "slimorca": ("36fa803348661320", "efdb90d7de8cbe11"),
            "sharegpt": ("9121a99ff55885bb", "b6ba5e08a6d0f91b"),
            "wizardlm": ("3a7cbbfb695c4c3d", "af9512681cc97d79"),
        },
    },
    "crow_300m_mini": {
        "config": "d5e7c9681c2da99d",
        "tokenizer": "128ca0c738434e53",
        "sources": {
            "fineweb_edu": ("04d5d6c0f4ef9279", "b58c044118744cc9"),
            "wikipedia": ("a614ffc5b692de0d", "f37150fbae7ecae3"),
            "books_gutenberg": ("d496b0abd0decd96", "f3c14b65b476c1d2"),
            "peso": ("557afee219124346", "751c373ccb3ac143"),
            "arxiv": ("2d9cbb89d06c547c", "981e30bb496e8308"),
            "openwebmath": ("0ff260c644bbeefb", "ddb2f1336c1256f0"),
            "tinygsm": ("ca8573c25a3a5d1e", "4034a2c025a97177"),
            "algebraic_stack": ("4566347616d999a5", "21efcd652ad03986"),
            "gsm8k": ("df9a15f2b11e2778", "3ffc84246ac300b1"),
            "github_code_clean_python": ("bb88f17dfdaebcff", "700d56cdc7bc9a68"),
            "github_code_clean_javascript": ("6ca946aca0178495", "c9f4ab481591e822"),
            "github_code_clean_typescript": ("0ddb1168bcb4a4ab", "f34c443073fae9d9"),
            "github_code_clean_java": ("0ff50c78d34a5090", "fef4875b7cc60f2a"),
            "github_code_clean_cpp": ("18981d711ba70d0d", "a3b72c190b3b78f4"),
            "github_code_clean_go": ("8b754a6222c3c353", "24b2488bc38d8d0a"),
            "github_code_clean_rust": ("a3d0abda4164a28b", "f997daeef4dfb4d2"),
            "github_code_clean_shell": ("9ec8395c504ab595", "7be5c8aaf54f400e"),
            "github_code_clean_sql": ("d1c5d2ae0a07b2a3", "4554146610fbb315"),
            "github_code_clean_html": ("77dd5d781104dac8", "4cdbab97251f865a"),
            "flan": ("d2ed4c0d9137827a", "a4a0c70625517542"),
            "metamath": ("fbabb8961fdf87ec", "6fca33db6fc6146a"),
            "orca_math": ("4c614e6f6c99cd8d", "559c41d6dc1f9cfa"),
            "evol_code": ("d5b8a54fa3e41a6b", "c86350f441bd855e"),
            "code_alpaca": ("52240fa45709bde5", "ffd74a14ac575a96"),
            "slimorca": ("36fa803348661320", "efdb90d7de8cbe11"),
            "sharegpt": ("9121a99ff55885bb", "b6ba5e08a6d0f91b"),
            "wizardlm": ("3a7cbbfb695c4c3d", "af9512681cc97d79"),
        },
    },
}


@pytest.mark.parametrize("name", list(PINNED_HASHES))
def test_shipped_config_hashes_are_pinned(name: str) -> None:
    cfg = load_dataset_config(REPO / "config" / "datasets" / f"{name}.yaml")
    expected = PINNED_HASHES[name]
    assert cfg.config_hash() == expected["config"]
    assert cfg.tokenizer_hash() == expected["tokenizer"]
    assert {source: (cfg.raw_hash(source), cfg.processed_hash(source)) for source in cfg.sources} == expected["sources"]



def test_max_cached_file_mb_validated_and_not_hashed() -> None:
    d = _minimal()
    d["sources"]["files"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}
    d["stages"][0]["train"] = {"pre": 0.5, "files": 0.5}
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


def test_every_schema_field_carries_a_hash_annotation() -> None:
    """A new field must say which hash it belongs to; without the annotation every hash of it raises rather than
    silently landing in (or missing from) one — the drift this replaces."""
    samples: list[Any] = [
        TokenizerConfig(name="t", kind="synthetic"),
        DedupConfig(),
        dc.DecontaminationConfig(),
        ProcessingConfig(),
        SourceConfig(kind="pretrain", loader="synthetic"),
        StageConfig(name="s", tokens=1, train={"a": 1.0}, val={"a": 1.0}),
        _build(_minimal()),
    ]
    for sample in samples:
        for f in dataclasses.fields(sample):
            assert dc.field_hash_annotation(f, sample) in dc.HASH_ANNOTATIONS, f"{type(sample).__name__}.{f.name}"

    with pytest.raises(TypeError, match="carries no `hash` metadata"):
        dc.hash_payload(_UnannotatedField(value=1), "config")


@dataclasses.dataclass
class _UnannotatedField:
    """What a field added to the schema without a `hash` annotation looks like: hashing it must raise."""

    value: int = 0


def test_hash_payload_drops_defaults_recursively() -> None:
    """Schema changes with defaults must not invalidate data on disk: only explicitly set values are hashed."""
    assert dc.hash_payload(DedupConfig(), "processed") == {}
    assert dc.hash_payload(DedupConfig(mode="minhash", threshold=0.5), "processed") == {"mode": "minhash", "threshold": 0.5}
    proc = ProcessingConfig(min_chars=7, dedup=DedupConfig(mode="minhash"))
    assert dc.hash_payload(proc, "processed") == {"min_chars": 7, "dedup": {"mode": "minhash"}}
    src = SourceConfig(kind="pretrain", loader="hf_files", hf_id="x/y", load_kwargs={"data_files": "*.parquet"})
    assert dc.hash_payload(src, "raw") == {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}


def test_hash_payload_selects_by_annotation() -> None:
    """`raw` and `processed` take exactly their own fields (`processed_hash` folds the raw hash in as one value);
    `config` takes every hashed field, and nothing annotated `none`."""
    src = SourceConfig(kind="instruct", loader="hf_stream", hf_id="x/y", fields={"instruction": "a", "output": "b"},
                       check_limit=5, rows=None, seed=7, input_inversions=0.5, describe_tokens_per_row=9)  # fmt: skip
    assert dc.hash_payload(src, "raw") == {"kind": "instruct", "loader": "hf_stream", "hf_id": "x/y", "fields": {"instruction": "a", "output": "b"}}
    assert dc.hash_payload(src, "processed") == {"seed": 7, "input_inversions": 0.5}  # seed: not the synthetic loader
    assert set(dc.hash_payload(src, "config")) == {
        "kind", "loader", "hf_id", "fields", "check_limit", "seed", "input_inversions",
    }, "config counts raw + processed + config fields, never `describe_tokens_per_row`"

    synthetic = SourceConfig(kind="pretrain", loader="synthetic", seed=7)
    assert dc.hash_payload(synthetic, "raw") == {"kind": "pretrain", "loader": "synthetic", "seed": 7}
    assert dc.hash_payload(synthetic, "processed") == {}


def test_the_seed_is_raw_identity_only_for_the_synthetic_loader() -> None:
    """An instruct source's seed drives inversions and shuffle (build-time): a new seed rebuilds processed, never raw."""
    d = _minimal()
    base = _build(d)
    d["sources"]["ins"]["seed"] = 7
    reseeded = _build(d)
    assert reseeded.raw_hash("ins") == base.raw_hash("ins") and reseeded.processed_hash("ins") != base.processed_hash("ins")
    d["sources"]["pre"]["seed"] = 7  # `pre` is synthetic: the seed generates its rows
    assert _build(d).raw_hash("pre") != base.raw_hash("pre")


def test_config_hash_ignores_the_bloom_budget_like_processed_hash_does() -> None:
    d = _minimal()
    base = _build(d).config_hash()
    d["processing"] = {"dedup": {"bloom_memory_mb": 7}}
    assert _build(d).config_hash() == base
    d["processing"] = {"dedup": {"normalize": False}}
    assert _build(d).config_hash() != base


def test_schema_rejects_a_pretrain_filter_a_non_positive_check_limit_and_an_empty_validation_split() -> None:
    d = _minimal()
    d["sources"]["pre"]["filter"] = "sharegpt_quality"
    with pytest.raises(ValueError, match="filter only applies to kind instruct"):
        _build(d)
    d = _minimal()
    d["sources"]["pre"]["check_limit"] = 0
    with pytest.raises(ValueError, match="check_limit must be positive"):
        _build(d)
    d = _minimal()
    d["validation_fraction"] = 0.0
    with pytest.raises(ValueError, match="validation_fraction is 0"):
        _build(d)


def test_the_processing_payload_keeps_only_the_active_dedup_mode() -> None:
    def payload(processing: ProcessingConfig) -> dict[str, Any]:
        return dc.hash_payload(processing, "processed")

    assert payload(ProcessingConfig()) == {}
    assert payload(ProcessingConfig(dedup=DedupConfig(bloom_memory_mb=1))) == {}  # resource knob
    assert payload(ProcessingConfig(dedup=DedupConfig(threshold=0.5, ngram=3))) == {}  # inactive minhash fields
    assert payload(ProcessingConfig(dedup=DedupConfig(normalize=False))) == {"dedup": {"normalize": False}}
    minhash = ProcessingConfig(min_chars=9, dedup=DedupConfig(mode="minhash", threshold=0.5, normalize=False, bloom_memory_mb=1))
    assert payload(minhash) == {"min_chars": 9, "dedup": {"mode": "minhash", "normalize": False, "threshold": 0.5}}
    assert payload(ProcessingConfig(dedup=DedupConfig(mode="none", threshold=0.5))) == {"dedup": {"mode": "none"}}
    assert payload(ProcessingConfig(dedup=DedupConfig(mode="none", normalize=False))) == {"dedup": {"mode": "none"}}  # nothing is hashed


def test_raw_hash_only_tracks_the_loader_identity_and_token_counting() -> None:
    """The raw shards are the bandwidth-expensive part: nothing but a real change of the source, or of how its
    stored token counts are made, may invalidate them."""
    base = _build(_minimal())
    h = base.raw_hash("pre")

    unchanged: list[Mutation] = [
        lambda d: d["stages"][0].__setitem__("tokens", 999_999),  # budget: same rows on disk
        lambda d: d.__setitem__("block_size", 32),  # sequence budget only
        lambda d: d["sources"]["pre"].__setitem__("describe_tokens_per_row", 3),  # describe only
        lambda d: d["sources"]["pre"].__setitem__("processing", {"min_chars": 99}),  # processing → processed only
        lambda d: d["sources"]["pre"].__setitem__("processing", {"dedup": {"mode": "minhash"}}),
        lambda d: d.__setitem__("processing", {"quality_filter": True, "decontamination": {"enabled": True}}),
        lambda d: d.__setitem__("max_seq_length", 64),  # the raw manifest records the truncation cap itself
        lambda d: d.__setitem__("validation_fraction", 0.2),  # training-time split
        lambda d: d["sources"]["pre"].__setitem__("validation_fraction", 0.2),
        lambda d: d["sources"]["pre"].__setitem__("shuffle", True),  # processed order only
        lambda d: d["sources"]["pre"].__setitem__("check_limit", 5),  # bounds how far to read, not what is read
    ]
    for change in unchanged:
        d = _minimal()
        change(d)
        assert _build(d).raw_hash("pre") == h, change

    invalidating: list[Mutation] = [
        lambda d: d["sources"]["pre"].__setitem__("seed", 5),
        lambda d: d["sources"]["pre"].__setitem__("text_field", "body"),
        lambda d: d["sources"]["pre"].__setitem__("converter", "gsm8k_question_answer"),
        lambda d: d["sources"]["pre"].__setitem__("split", "test"),
        lambda d: d.__setitem__("tokenizer", {"name": "other", "kind": "hf", "hf_id": "a/b"}),  # stored counts
        lambda d: d.__setitem__("token_count", "estimate"),
    ]
    for change in invalidating:
        d = _minimal()
        change(d)
        assert _build(d).raw_hash("pre") != h, change

    d = _minimal()
    d["sources"]["hold"]["rows"] = 99  # how many rows are downloaded, not which
    assert _build(d).raw_hash("hold") == base.raw_hash("hold")
    d = _minimal()
    d["sources"]["ins"]["input_inversions"] = 0.5  # applied by the build
    assert _build(d).raw_hash("ins") == base.raw_hash("ins")


def test_processed_hash_tracks_cap_active_dedup_fields_inversions_and_shuffle() -> None:
    base = _build(_minimal())
    h = base.processed_hash("pre")
    assert h != base.raw_hash("pre")

    unchanged: list[Mutation] = [
        lambda d: d["stages"][0].__setitem__("tokens", 999_999),
        lambda d: d.__setitem__("block_size", 32),
        lambda d: d["sources"]["pre"].__setitem__("describe_tokens_per_row", 3),
        lambda d: d.__setitem__("validation_fraction", 0.2),
        lambda d: d["sources"]["pre"].__setitem__("validation_fraction", 0.2),
        lambda d: d["sources"]["pre"].__setitem__("check_limit", 5),
        lambda d: d.__setitem__("processing", {"dedup": {"bloom_memory_mb": 4}}),  # resource knob of exact mode
        lambda d: d.__setitem__("processing", {"dedup": {"threshold": 0.5, "num_perm": 8, "ngram": 2}}),  # inactive minhash fields
    ]
    for change in unchanged:
        d = _minimal()
        change(d)
        assert _build(d).processed_hash("pre") == h, change

    invalidating: list[Mutation] = [
        lambda d: d.__setitem__("max_seq_length", 64),
        lambda d: d["sources"]["pre"].__setitem__("processing", {"min_chars": 99}),
        lambda d: d.__setitem__("processing", {"min_chars": 99}),
        lambda d: d.__setitem__("processing", {"dedup": {"normalize": False}}),
        lambda d: d.__setitem__("processing", {"dedup": {"mode": "minhash"}}),
        lambda d: d.__setitem__("processing", {"dedup": {"mode": "none"}}),
        lambda d: d.__setitem__("processing", {"quality_filter": True}),
        lambda d: d["sources"]["pre"].__setitem__("shuffle", True),
        lambda d: d["sources"]["pre"].__setitem__("seed", 5),  # through raw_hash
        lambda d: d.__setitem__("token_count", "estimate"),  # through raw_hash
        lambda d: d.__setitem__("tokenizer", {"name": "other", "kind": "hf", "hf_id": "a/b"}),  # through raw_hash
    ]
    for change in invalidating:
        d = _minimal()
        change(d)
        assert _build(d).processed_hash("pre") != h, change

    # minhash fields count once minhash is the active mode
    d = _minimal()
    d["processing"] = {"dedup": {"mode": "minhash"}}
    minhash = _build(d).processed_hash("pre")
    d["processing"] = {"dedup": {"mode": "minhash", "threshold": 0.5}}
    assert _build(d).processed_hash("pre") != minhash
    d["processing"] = {"dedup": {"mode": "minhash", "bloom_memory_mb": 4}}
    assert _build(d).processed_hash("pre") == minhash

    # instruct: inversions and the shuffle default
    d = _minimal()
    d["sources"]["ins"]["input_inversions"] = 0.5
    assert _build(d).processed_hash("ins") != base.processed_hash("ins")
    d = _minimal()
    d["sources"]["ins"]["shuffle"] = False
    assert _build(d).processed_hash("ins") != base.processed_hash("ins")
    d["sources"]["ins"]["shuffle"] = True  # the same as the instruct default
    assert _build(d).processed_hash("ins") == base.processed_hash("ins")


def test_hash_fields_golden_defaults() -> None:
    """`hash_payload` drops default-valued fields, so a changed *default* re-labels data built under the old one.
    Changing any of these defaults must be a conscious, hash-breaking commit: update this golden dict with it."""
    assert asdict(ProcessingConfig()) == {
        "min_chars": 50,
        "dedup": {"mode": "exact", "normalize": True, "bloom_memory_mb": 1024, "threshold": 0.95, "num_perm": 256, "ngram": 5},
        "quality_filter": False,
        "decontamination": {"enabled": False, "benchmarks": list(dc.DEFAULT_BENCHMARKS), "ngram": 13, "threshold": 0.1},
    }
    src = SourceConfig(kind="pretrain", loader="synthetic")
    defaults = ("split", "text_field", "seed", "input_inversions", "shuffle", "validation_fraction", "rows", "describe_tokens_per_row")
    assert {f: getattr(src, f) for f in defaults} == {
        "split": "train", "text_field": "text", "seed": 42, "input_inversions": 0.0, "shuffle": None,
        "validation_fraction": None, "rows": None, "describe_tokens_per_row": 500,
    }  # fmt: skip
    cfg = _build(_minimal())
    assert (cfg.max_seq_length, cfg.validation_fraction, cfg.token_count, cfg.always_range_requests) == (2048, 0.05, "tokenizer", True)
    assert dc.SAFETY_MARGIN == 1.2


def test_tokenizer_hash() -> None:
    a = _build(_minimal())
    d = _minimal()
    d["tokenizer"] = {"name": "other", "kind": "hf", "hf_id": "a/b"}
    assert a.tokenizer_hash() != _build(d).tokenizer_hash() and len(a.tokenizer_hash()) == 16


def test_config_hash_changes_on_any_field() -> None:
    base = _build(_minimal()).config_hash()
    changes: list[Mutation] = [
        lambda d: d["stages"][0].__setitem__("transition_pct", 0.5),
        lambda d: d.__setitem__("block_size", 32),
        lambda d: d.__setitem__("validation_fraction", 0.2),
        lambda d: d["sources"]["ins"].__setitem__("input_inversions", 0.5),
    ]
    for change in changes:
        d = _minimal()
        change(d)
        assert _build(d).config_hash() != base
    assert len(base) == 16


def test_config_hash_ignores_fetch_and_describe_knobs() -> None:
    d = _minimal()
    d["sources"]["files"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}
    d["stages"][0]["train"] = {"pre": 0.5, "files": 0.5}
    base = _build(d).config_hash()
    d["sources"]["files"]["describe_tokens_per_row"] = 7
    d["sources"]["files"]["load_kwargs"]["max_cached_file_mb"] = 3
    cfg = _build(d)
    cfg.always_range_requests = False
    assert cfg.config_hash() == base
    d["sources"]["files"]["revision"] = "abc"
    assert _build(d).config_hash() != base


def test_dataset_config_fields_and_asdict_roundtrip() -> None:
    names = set(dc.dataset_config_fields())
    assert {"name", "tokenizer", "sources", "stages", "block_size", "max_seq_length", "validation_fraction", "processing"} <= names
    assert "instruct_mixtures" not in names
    cfg = _build(_minimal())
    assert asdict(cfg)["sources"]["pre"]["kind"] == "pretrain"


def test_overlap_warnings_for_validation_only_sources() -> None:
    assert _build(_minimal()).overlap_warnings() == []
    d = _minimal()
    d["sources"]["pre"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "org/repo", "load_kwargs": {"data_files": "data/*.parquet"}}
    d["sources"]["hold"] = {"kind": "pretrain", "loader": "hf_files", "hf_id": "org/repo", "load_kwargs": {"data_files": "data/sample/*.parquet"}, "rows": 5}
    (warning,) = _build(d).overlap_warnings()
    assert "'hold'" in warning and "'pre'" in warning and "may overlap" in warning and "validation_fraction" in warning
    d["sources"]["hold"]["load_kwargs"] = {"data_files": "other/*.parquet"}
    assert _build(d).overlap_warnings() == []
    d["sources"]["hold"]["load_kwargs"] = {"data_files": "data/sample/*.parquet"}
    d["stages"][0]["train"] = {"pre": 0.5, "hold": 0.5}  # a source in train is not a held-out set
    d["sources"]["hold"].pop("rows")
    assert _build(d).overlap_warnings() == []
    assert dc._glob_prefix("data/CC-MAIN-2013-20/*.parquet") == "data/CC-MAIN-2013-20/" and dc._glob_prefix(None) == ""
