# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.dataset_config: loading the shipped configs, every validation rule, source usage,
budgets and the raw / processed hash invariants."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict
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
        (lambda d: d["sources"]["ins"].update({"processing": {"min_chars": 1}}), "only apply to kind pretrain"),
        (lambda d: d["sources"]["ins"].pop("fields"), "requires fields or converter"),
        (lambda d: d["sources"]["ins"].update({"fields": {"instruction": "a"}}), "instruction and output"),
        (lambda d: d["sources"]["ins"].pop("hf_id"), "requires hf_id"),
        (lambda d: d["sources"]["pre"].update({"loader": "github_code"}), "requires language"),
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


def test_hash_fields_drops_defaults_recursively() -> None:
    """Schema changes with defaults must not invalidate data on disk: only explicitly set values are hashed."""
    assert dc.hash_fields(DedupConfig()) == {}
    assert dc.hash_fields(DedupConfig(threshold=0.5)) == {"threshold": 0.5}
    proc = ProcessingConfig(min_chars=7, dedup=DedupConfig(mode="minhash"))
    assert dc.hash_fields(proc) == {"min_chars": 7, "dedup": {"mode": "minhash"}}
    src = SourceConfig(kind="pretrain", loader="hf_files", hf_id="x/y", load_kwargs={"data_files": "*.parquet"})
    assert dc.hash_fields(src) == {"kind": "pretrain", "loader": "hf_files", "hf_id": "x/y", "load_kwargs": {"data_files": "*.parquet"}}


def test_processing_hash_fields_keep_only_the_active_dedup_mode() -> None:
    assert dc._processing_hash_fields(ProcessingConfig()) == {}
    assert dc._processing_hash_fields(ProcessingConfig(dedup=DedupConfig(bloom_memory_mb=1))) == {}  # resource knob
    assert dc._processing_hash_fields(ProcessingConfig(dedup=DedupConfig(threshold=0.5, ngram=3))) == {}  # inactive minhash fields
    assert dc._processing_hash_fields(ProcessingConfig(dedup=DedupConfig(normalize=False))) == {"dedup": {"normalize": False}}
    minhash = ProcessingConfig(min_chars=9, dedup=DedupConfig(mode="minhash", threshold=0.5, normalize=False, bloom_memory_mb=1))
    assert dc._processing_hash_fields(minhash) == {"min_chars": 9, "dedup": {"mode": "minhash", "threshold": 0.5}}
    assert dc._processing_hash_fields(ProcessingConfig(dedup=DedupConfig(mode="none", threshold=0.5))) == {"dedup": {"mode": "none"}}


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
    """`hash_fields` drops default-valued fields, so a changed *default* re-labels data built under the old one.
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
