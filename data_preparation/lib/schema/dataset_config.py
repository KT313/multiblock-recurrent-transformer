# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset config schema: the single definition of a dataset (`config/datasets/<name>.yaml`).

Framework-neutral (no torch). Loaded by `data_preparation/prepare.py build` (materialises it under `dataset/`) and
by `training/train.py` (verifies / auto-prepares it and derives the per-stage data mixtures). The run config only
references the file; every data-related setting lives here.

Layout produced on disk (see CLAUDE.md "Dataset-config restructuring"):

    dataset/sources/<source>/{raw,filtered,processed}/   shared by every dataset config, append-only
    dataset/mixtures/<config name>/<mixture>/{train,validation}/
    dataset/tokenizers/<tokenizer name>/
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional

from jsonargparse import ArgumentParser, Namespace

SourceKind = Literal["pretrain", "holdout", "instruct"]
LoaderName = Literal["hf_files", "hf_split", "hf_stream", "github_code", "local", "synthetic"]
DedupMode = Literal["none", "exact", "minhash"]
TokenCountMode = Literal["tokenizer", "estimate"]

DEFAULT_BENCHMARKS = [
    "gsm8k_test",
    "math_test",
    "humaneval",
    "mbpp_test",
    "arc_challenge_test",
    "hellaswag_test",
    "mmlu_test",
    "winogrande_test",
]
SAFETY_MARGIN = 1.2  # rows downloaded = rows needed × this (task 5 planner)


@dataclass
class TokenizerConfig:
    """Which tokenizer defines "a token" for this dataset; saved to `dataset/tokenizers/<name>/`."""

    name: str
    kind: Literal["hf", "synthetic"] = "hf"
    hf_id: Optional[str] = None  # required for kind=hf
    revision: Optional[str] = None  # Hub commit sha; pin it

    def __post_init__(self) -> None:
        if self.kind == "hf" and not self.hf_id:
            raise ValueError(f"tokenizer {self.name!r}: kind=hf requires hf_id")


@dataclass
class DedupConfig:
    """Deduplication of pretrain sources and instruct mixtures."""

    mode: DedupMode = "exact"
    normalize: bool = True  # exact mode: hash lowercased, whitespace-collapsed text
    threshold: float = 0.95  # minhash mode: Jaccard threshold
    num_perm: int = 256  # minhash mode: permutations
    ngram: int = 5  # minhash mode: word n-gram size

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("dedup.threshold must be in (0, 1]")
        if self.num_perm <= 0 or self.ngram <= 0:
            raise ValueError("dedup.num_perm and dedup.ngram must be positive")


@dataclass
class DecontaminationConfig:
    """Drop documents overlapping benchmark test sets (off by default; the thesis run skipped it)."""

    enabled: bool = False
    benchmarks: list[str] = field(default_factory=lambda: list(DEFAULT_BENCHMARKS))
    ngram: int = 13
    threshold: float = 0.1  # share of a document's n-grams found in one benchmark


@dataclass
class ProcessingConfig:
    """Per-source processing options; the dataset-level block is the default, a source may override it."""

    min_chars: int = 50  # drop shorter texts
    max_chars: int = 20000  # truncate longer texts (characters)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    quality_filter: bool = False  # prose heuristics (sentences, caps ratio, repetition); thesis run: off
    decontamination: DecontaminationConfig = field(default_factory=DecontaminationConfig)

    def __post_init__(self) -> None:
        if self.min_chars < 0 or self.max_chars <= 0 or self.max_chars < self.min_chars:
            raise ValueError("processing: need 0 <= min_chars <= max_chars and max_chars > 0")


@dataclass
class SourceConfig:
    """One data source. `kind` selects the pipeline, `loader` how rows are fetched (see `lib/sources.py`)."""

    kind: SourceKind
    loader: LoaderName = "hf_split"
    hf_id: Optional[str] = None  # Hub dataset id (hf_files / hf_split / hf_stream / github_code)
    revision: Optional[str] = None  # Hub commit sha; pin it so row order is stable across increments
    load_kwargs: dict[str, Any] = field(default_factory=dict)  # hf_files: {data_files: <glob>}; else `load_dataset` kwargs
    split: str = "train"
    text_field: str = "text"  # pretrain/holdout: column holding the document
    language: Optional[str] = None  # github_code: language label of codeparrot/github-code-clean
    path: Optional[str] = None  # local: directory of parquet/jsonl files
    converter: Optional[str] = None  # named row converter (lib/sources.py), e.g. gsm8k_question_answer
    fields: Optional[dict[str, str]] = None  # instruct: {instruction: <col>, input: <col>, output: <col>}
    filter: Optional[str] = None  # named row filter, e.g. sharegpt_quality
    check_limit: Optional[int] = None  # instruct: stop after inspecting this many rows even if short of target
    tokens_per_row_estimate: int = 500  # planner prior until the manifest has measured tokens/row
    repeat_to_budget: bool = False  # small sources: download whole, repeat rows to reach the budget
    rows: Optional[int] = None  # holdout: number of rows to hold out
    seed: int = 42  # holdout shuffle / synthetic generator seed
    processing: Optional[ProcessingConfig] = None  # pretrain: override of the dataset-level processing block

    def __post_init__(self) -> None:
        if self.loader == "github_code" and not self.language:
            raise ValueError("loader github_code requires language")
        if self.loader in ("hf_files", "hf_split", "hf_stream", "github_code") and not self.hf_id:
            raise ValueError(f"loader {self.loader} requires hf_id")
        if self.loader == "hf_files" and not isinstance(self.load_kwargs.get("data_files"), str):
            raise ValueError("loader hf_files requires load_kwargs.data_files (a glob relative to the repo root)")
        if self.loader == "local" and not self.path:
            raise ValueError("loader local requires path")
        if self.kind == "holdout" and (self.rows is None or self.rows <= 0):
            raise ValueError("kind holdout requires rows > 0")
        if self.kind == "instruct" and self.fields is None and self.converter is None and self.loader != "synthetic":
            raise ValueError("kind instruct requires fields or converter")
        if self.fields is not None and not {"instruction", "output"} <= set(self.fields):
            raise ValueError("fields must map at least instruction and output")
        if self.tokens_per_row_estimate <= 0:
            raise ValueError("tokens_per_row_estimate must be positive")
        if self.kind != "pretrain" and self.processing is not None:
            raise ValueError("processing overrides only apply to kind pretrain")


@dataclass
class MixtureConfig:
    """An instruct mixture built per dataset config from `instruct` sources (counts derived from the stage budget)."""

    sources: dict[str, float]  # instruct source name -> share of the mixture
    max_tokens: int = 2048  # drop examples whose word-based token estimate exceeds this
    input_inversions: float = 0.0  # share of examples turned into "given the output, what was the instruction?"
    val_split: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        _check_weights("mixture.sources", self.sources)
        if not 0.0 <= self.input_inversions <= 1.0 or not 0.0 <= self.val_split < 1.0:
            raise ValueError("mixture: input_inversions in [0, 1], val_split in [0, 1)")


@dataclass
class StageConfig:
    """One training stage: token budget and the train/val mixtures over sources (or `<mixture>[/validation]`)."""

    name: str
    tokens: int
    train: dict[str, float]
    val: dict[str, float]
    transition_pct: float = 0.0  # fraction of this stage (at its end) blending into the next stage's data/LR

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            raise ValueError(f"stage {self.name}: tokens must be positive")
        if not 0.0 <= self.transition_pct < 1.0:
            raise ValueError(f"stage {self.name}: transition_pct must be in [0, 1)")
        _check_weights(f"stage {self.name}.train", self.train)
        _check_weights(f"stage {self.name}.val", self.val)


@dataclass
class DatasetConfig:
    """The whole dataset definition."""

    name: str
    tokenizer: TokenizerConfig
    sources: dict[str, SourceConfig]
    stages: list[StageConfig]
    mixtures: dict[str, MixtureConfig] = field(default_factory=dict)
    max_seq_length: int = 2048  # token-count cap per document; the run config's block_size must be <= this
    token_count: TokenCountMode = "tokenizer"  # "estimate" = chars / 4
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty path component")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if not self.stages:
            raise ValueError("stages must contain at least one stage")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        if set(self.sources) & set(self.mixtures):
            raise ValueError(f"names shared by sources and mixtures: {sorted(set(self.sources) & set(self.mixtures))}")
        for mixture_name, mixture in self.mixtures.items():
            for src in mixture.sources:
                if src not in self.sources:
                    raise ValueError(f"mixture {mixture_name}: unknown source {src!r}")
                if self.sources[src].kind != "instruct":
                    raise ValueError(f"mixture {mixture_name}: source {src!r} is not kind instruct")
        for stage in self.stages:
            for key in stage.train:
                self._check_stage_key(stage.name, key, is_val=False)
            for key in stage.val:
                self._check_stage_key(stage.name, key, is_val=True)

    def _check_stage_key(self, stage_name: str, key: str, is_val: bool) -> None:
        base, _, split = key.partition("/")
        if base in self.mixtures:
            if split not in ("", "train", "validation"):
                raise ValueError(f"stage {stage_name}: {key!r} must be <mixture>, <mixture>/train or /validation")
            return
        if split:
            raise ValueError(f"stage {stage_name}: {key!r} has a split but {base!r} is not a mixture")
        if base not in self.sources:
            raise ValueError(f"stage {stage_name}: unknown source or mixture {key!r}")
        kind = self.sources[base].kind
        if kind == "instruct":
            raise ValueError(f"stage {stage_name}: instruct source {key!r} can only be used through a mixture")
        if kind == "holdout" and not is_val:
            raise ValueError(f"stage {stage_name}: holdout source {key!r} cannot be used for training")

    # --- derived views ---------------------------------------------------------------------------------------------

    def source_processing(self, source_name: str) -> ProcessingConfig:
        """Effective processing options of a pretrain source (its override or the dataset-level block)."""
        override = self.sources[source_name].processing
        return override if override is not None else self.processing

    def source_hash(self, source_name: str) -> str:
        """Hash of everything that determines a source's rows on disk (loader settings + processing + token mode).

        Manifests key on this; a change means the source directory must be rebuilt. Budgets/weights are NOT part of
        it — they only change how many rows are needed, not what the rows are.
        """
        source = self.sources[source_name]
        payload: dict[str, Any] = {"source": asdict(source), "token_count": self.token_count}
        payload["source"].pop("tokens_per_row_estimate")
        payload["source"].pop("processing")
        if source.kind == "pretrain":
            payload["processing"] = asdict(self.source_processing(source_name))
            payload["max_seq_length"] = self.max_seq_length
        if self.token_count == "tokenizer" or source.kind == "instruct":
            payload["tokenizer"] = asdict(self.tokenizer)
        return _stable_hash(payload)

    def mixture_hash(self, mixture_name: str) -> str:
        """Hash of a mixture definition plus the hashes of the sources it draws from."""
        mixture = self.mixtures[mixture_name]
        payload = {
            "mixture": asdict(mixture),
            "sources": {name: self.source_hash(name) for name in mixture.sources},
            "budget_tokens": self.mixture_budget_tokens(mixture_name),
        }
        return _stable_hash(payload)

    def tokenizer_hash(self) -> str:
        """Hash of the tokenizer definition (the manifest key of `dataset/tokenizers/<name>/`)."""
        return _stable_hash(asdict(self.tokenizer))

    def config_hash(self) -> str:
        """Hash of the complete config (recorded in checkpoints so a resume with different data is detected)."""
        return _stable_hash(asdict(self))

    def mixture_budget_tokens(self, mixture_name: str) -> int:
        """Largest per-stage token demand on a mixture (stages share the built mixture, so max, not sum)."""
        demand = 0
        for stage in self.stages:
            for key, weight in stage.train.items():
                if key.partition("/")[0] == mixture_name:
                    demand = max(demand, int(stage.tokens * weight))
        return demand

    def source_budget_tokens(self, source_name: str) -> int:
        """Largest per-stage token demand on a pretrain source (files are shared between stages → max)."""
        demand = 0
        for stage in self.stages:
            demand = max(demand, int(stage.tokens * stage.train.get(source_name, 0.0)))
        for mixture_name, mixture in self.mixtures.items():
            if source_name in mixture.sources:
                demand = max(demand, int(self.mixture_budget_tokens(mixture_name) * mixture.sources[source_name]))
        return demand

    def sources_of_kind(self, kind: SourceKind) -> list[str]:
        return [name for name, s in self.sources.items() if s.kind == kind]


# --- helpers ----------------------------------------------------------------------------------------------------------


def _check_weights(what: str, weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError(f"{what}: must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{what}: weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what}: weights sum to {total:.6f}, expected 1")


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dataset_config_fields() -> list[str]:
    """Top-level field names (used by the settings/CLI layer to recognise dataset-config keys)."""
    return [f.name for f in fields(DatasetConfig)]


def load_dataset_config(path: str | Path, overrides: Optional[list[str]] = None) -> DatasetConfig:
    """Load a dataset config YAML; `overrides` are jsonargparse `--key value` strings (nested keys with dots)."""
    parser = ArgumentParser(description="Dataset config")
    parser.add_class_arguments(DatasetConfig, nested_key=None)
    ns = parser.parse_path(str(path))
    if overrides:
        ns = parser.parse_args(overrides, namespace=ns)
    instantiated = parser.instantiate_classes(ns)
    values = instantiated.as_dict() if isinstance(instantiated, Namespace) else instantiated
    return DatasetConfig(**values)
