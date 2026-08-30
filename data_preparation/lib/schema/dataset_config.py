# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset config schema: the single definition of a dataset (`config/datasets/<name>.yaml`).

Framework-neutral (no torch). Loaded by `data_preparation/prepare.py build` (materialises it under `dataset/`) and
by `training/train.py` (verifies / auto-prepares it and derives the per-stage data mixtures). The run config only
references the file; every data-related setting lives here.

Layout produced on disk (see CLAUDE.md "Dataset-config restructuring"):

    dataset/sources/<source>/{raw,processed}/            shared by every dataset config, append-only
    dataset/instruct_mixtures/<config name>/<mixture>/{train,validation}/
    dataset/tokenizers/<tokenizer name>/
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, Field, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from jsonargparse import ArgumentParser, Namespace

SourceKind = Literal["pretrain", "validation", "instruct"]
LoaderName = Literal["hf_files", "hf_split", "hf_stream", "github_code", "local", "synthetic"]
DedupMode = Literal["none", "exact", "minhash"]
TokenCountMode = Literal["tokenizer", "estimate"]

HUB_LOADERS: tuple[str, ...] = ("hf_files", "hf_split", "hf_stream", "github_code")  # loaders that need `hf_id`

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
    load_kwargs: dict[str, Any] = field(default_factory=dict)  # hf_files/github_code: {data_files: <glob>, max_cached_file_mb: <MB>}; else `load_dataset` kwargs
    split: str = "train"
    text_field: str = "text"  # pretrain/validation: column holding the document
    language: Optional[str] = None  # github_code: language label of codeparrot/github-code-clean
    path: Optional[str] = None  # local: directory of parquet/jsonl files
    converter: Optional[str] = None  # named row converter (lib/sources.py), e.g. gsm8k_question_answer
    fields: Optional[dict[str, str]] = None  # instruct: {instruction: <col>, input: <col>, output: <col>}
    filter: Optional[str] = None  # named row filter, e.g. sharegpt_quality
    check_limit: Optional[int] = None  # instruct: stop after inspecting this many rows even if short of target
    tokens_per_row_estimate: int = 500  # planner prior until the manifest has measured tokens/row
    rows: Optional[int] = None  # validation: number of rows to hold out
    seed: int = 42  # validation shuffle / synthetic generator seed
    processing: Optional[ProcessingConfig] = None  # pretrain: override of the dataset-level processing block
    validation_tokens: int = 0  # pretrain: hold out the first processed rows worth this many tokens as `<source>/validation`

    def __post_init__(self) -> None:
        self._check_loader_fields()
        self._check_kind_fields()
        if self.fields is not None and not {"instruction", "output"} <= set(self.fields):
            raise ValueError("fields must map at least instruction and output")
        if self.tokens_per_row_estimate <= 0:
            raise ValueError("tokens_per_row_estimate must be positive")

    def _check_loader_fields(self) -> None:
        """Every loader needs some fields the others do not."""
        if self.loader == "github_code" and not self.language:
            raise ValueError("loader github_code requires language")
        if self.loader in HUB_LOADERS and not self.hf_id:
            raise ValueError(f"loader {self.loader} requires hf_id")
        if self.loader == "hf_files" and not isinstance(self.load_kwargs.get("data_files"), str):
            raise ValueError("loader hf_files requires load_kwargs.data_files (a glob relative to the repo root)")
        if self.loader == "local" and not self.path:
            raise ValueError("loader local requires path")
        max_cached_file_mb = self.load_kwargs.get("max_cached_file_mb")
        if max_cached_file_mb is not None and not _is_non_negative_number(max_cached_file_mb):
            raise ValueError("load_kwargs.max_cached_file_mb must be a non-negative number (MB)")

    def _check_kind_fields(self) -> None:
        """Every kind needs some fields the others do not."""
        if self.kind == "validation" and (self.rows is None or self.rows <= 0):
            raise ValueError("kind validation requires rows > 0")
        if self.kind == "instruct":
            has_row_mapping = self.fields is not None or self.converter is not None
            if not has_row_mapping and self.loader != "synthetic":
                raise ValueError("kind instruct requires fields or converter")
        if self.kind != "pretrain" and self.processing is not None:
            raise ValueError("processing overrides only apply to kind pretrain")
        if self.validation_tokens < 0 or (self.kind != "pretrain" and self.validation_tokens):
            raise ValueError("validation_tokens must be >= 0 and only applies to kind pretrain")


@dataclass
class InstructMixtureConfig:
    """An instruct mixture built per dataset config from `instruct` sources (counts derived from the stage budget)."""

    sources: dict[str, float]  # instruct source name -> share of the mixture
    max_tokens: int = 2048  # drop examples longer than this (full token count of instruction + input + output)
    input_inversions: float = 0.0  # share of examples turned into "given the output, what was the instruction?"
    val_split: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        _check_weights("mixture.sources", self.sources)
        if not 0.0 <= self.input_inversions <= 1.0 or not 0.0 <= self.val_split < 1.0:
            raise ValueError("mixture: input_inversions in [0, 1], val_split in [0, 1)")

    def target_tokens(self, budget_tokens: float, source_name: str) -> float:
        """Tokens to take from ``source_name`` so the *train* split reaches its share of ``budget_tokens``: the
        share times ``SAFETY_MARGIN`` (dedup and empty-row removal) over ``1 − val_split`` (the validation split)."""
        return budget_tokens * self.sources[source_name] * SAFETY_MARGIN / (1.0 - self.val_split)


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
    instruct_mixtures: dict[str, InstructMixtureConfig] = field(default_factory=dict)
    max_seq_length: int = 2048  # token-count cap per document; the run config's block_size must be <= this
    always_range_requests: bool = True  # read every Hub file remotely by piece (row groups / stream prefix); False: files
    # up to load_kwargs.max_cached_file_mb are downloaded whole into the Hub cache instead. Traffic only, not part of
    # source hashes.
    token_count: TokenCountMode = "tokenizer"  # "estimate" = chars / 4
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    # --- validation ------------------------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be a non-empty path component")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if not self.stages:
            raise ValueError("stages must contain at least one stage")
        if len({s.name for s in self.stages}) != len(self.stages):
            raise ValueError("stage names must be unique")
        shared_names = set(self.sources) & set(self.instruct_mixtures)
        if shared_names:
            raise ValueError(f"names shared by sources and instruct mixtures: {sorted(shared_names)}")
        for mixture_name, mixture in self.instruct_mixtures.items():
            self._check_mixture_sources(mixture_name, mixture)
        for stage in self.stages:
            for key in stage.train:
                self._check_stage_key(stage.name, key, is_val=False)
            for key in stage.val:
                self._check_stage_key(stage.name, key, is_val=True)
        self._check_val_only_pretrain_sources()

    def _check_mixture_sources(self, mixture_name: str, mixture: InstructMixtureConfig) -> None:
        """A mixture may only draw from declared sources of kind instruct."""
        for source_name in mixture.sources:
            if source_name not in self.sources:
                raise ValueError(f"mixture {mixture_name}: unknown source {source_name!r}")
            if self.sources[source_name].kind != "instruct":
                raise ValueError(f"mixture {mixture_name}: source {source_name!r} is not kind instruct")

    def _check_stage_key(self, stage_name: str, key: str, is_val: bool) -> None:
        """A stage key is `<source>`, `<mixture>`, `<mixture>/train`, `<mixture>/validation` or — in `val` only —
        `<pretrain source>/validation` (the source's held-out split, needs `validation_tokens > 0`)."""
        base, _, split = key.partition("/")
        if base in self.instruct_mixtures:
            if split not in ("", "train", "validation"):
                raise ValueError(f"stage {stage_name}: {key!r} must be <mixture>, <mixture>/train or /validation")
            if is_val and split != "validation":
                raise ValueError(f"stage {stage_name}: {key!r} in val would validate on the train split; use {base}/validation")
            return
        if base not in self.sources:
            raise ValueError(f"stage {stage_name}: unknown source or mixture {key!r}")
        if split:
            source = self.sources[base]
            if split != "validation" or source.kind != "pretrain":
                raise ValueError(f"stage {stage_name}: {key!r}: only <mixture>/... and <pretrain source>/validation take a split")
            if not source.validation_tokens:
                raise ValueError(f"stage {stage_name}: {key!r} needs validation_tokens > 0 on source {base!r}")
            if not is_val:
                raise ValueError(f"stage {stage_name}: {key!r} is a validation split and cannot be used for training")
            return
        kind = self.sources[base].kind
        if kind == "instruct":
            raise ValueError(f"stage {stage_name}: instruct source {key!r} can only be used through a mixture")
        if kind == "validation" and not is_val:
            raise ValueError(f"stage {stage_name}: validation source {key!r} cannot be used for training")

    def _check_val_only_pretrain_sources(self) -> None:
        """A pretrain source only ever used in `val` would get no budget (never downloaded or processed) while
        training expects its directory; hold out a split of a trained source or use `kind: validation` instead."""
        trained = {key.partition("/")[0] for stage in self.stages for key in stage.train}
        for stage in self.stages:
            for key in stage.val:
                base = key.partition("/")[0]
                if base in self.sources and self.sources[base].kind == "pretrain" and base not in trained:
                    raise ValueError(
                        f"stage {stage.name}: pretrain source {base!r} is used for validation but never for training "
                        "(it would never be prepared); use `kind: validation` or a trained source's `/validation` split"
                    )

    # --- derived views ---------------------------------------------------------------------------------------------

    def source_processing(self, source_name: str) -> ProcessingConfig:
        """Effective processing options of a pretrain source (its override or the dataset-level block)."""
        override = self.sources[source_name].processing
        return override if override is not None else self.processing

    def sources_of_kind(self, kind: SourceKind) -> list[str]:
        return [name for name, source in self.sources.items() if source.kind == kind]

    # --- hashes (manifest keys; changing what goes into them invalidates data on disk) ------------------------------

    def raw_hash(self, source_name: str) -> str:
        """Hash of the loader identity of a source: everything that determines **which rows** its ``raw/`` directory
        holds and in what order (loader, repo, revision, files, split, converter/fields/filter, seed, ...).

        The raw manifest keys on this. Deliberately NOT part of it: processing options, ``max_seq_length``,
        ``token_count`` and the tokenizer (they change derived data — the raw ``tokens`` column is recounted in
        place, see ``stages/shared.py:ensure_raw_tokens``), ``check_limit`` (bounds how far to read, not what is
        read), ``tokens_per_row_estimate`` (planner prior), ``load_kwargs.max_cached_file_mb`` (how a file is
        fetched). Budgets/weights only change how many rows are needed. Raw shards are the bandwidth-expensive part
        of a dataset; nothing but a real change of the source may invalidate them.
        """
        source_fields = hash_fields(self.sources[source_name])
        for key in ("tokens_per_row_estimate", "processing", "check_limit", "validation_tokens"):
            source_fields.pop(key, None)
        load_kwargs = source_fields.get("load_kwargs")
        if load_kwargs is not None:
            load_kwargs.pop("max_cached_file_mb", None)
        return _stable_hash({"source": source_fields})

    def token_cap(self, source_name: str) -> Optional[int]:
        """The cap of a source's stored token counts: ``max_seq_length`` for pretrain / validation documents (training
        truncates them there), None for instruct examples (their full length decides ``mixture.max_tokens``)."""
        return None if self.sources[source_name].kind == "instruct" else self.max_seq_length

    def token_settings(self, source_name: str) -> dict[str, Any]:
        """How the ``tokens`` column of a source is counted: mode, tokenizer (when counting with it) and the cap
        (recorded in the manifests so a change recounts in place instead of re-downloading)."""
        source = self.sources[source_name]
        settings: dict[str, Any] = {"token_count": self.token_count, "cap": self.token_cap(source_name)}
        if self.token_count == "tokenizer" or source.kind == "instruct":
            settings["tokenizer"] = hash_fields(self.tokenizer)
        return settings

    def processed_hash(self, source_name: str) -> str:
        """Hash of a pretrain source's ``processed/`` directory: the raw hash plus the effective processing block
        and the token settings. A change rebuilds ``processed/`` from the raw shards (no download)."""
        payload: dict[str, Any] = {
            "raw": self.raw_hash(source_name),
            "processing": hash_fields(self.source_processing(source_name)),
            "tokens": self.token_settings(source_name),
        }
        validation_tokens = self.sources[source_name].validation_tokens
        if validation_tokens:
            payload["validation_tokens"] = validation_tokens  # the first rows go to the validation split instead
        return _stable_hash(payload)

    def validation_hash(self, source_name: str) -> str:
        """Hash of a validation source's directory: the raw hash (which includes ``rows`` and ``seed``) plus the
        token settings; validation sets are small and fetched again when either changes."""
        return _stable_hash({"raw": self.raw_hash(source_name), "tokens": self.token_settings(source_name)})

    def stage_hash(self, source_name: str, stage: str) -> str:
        """The manifest key of one source stage directory (``raw`` / ``processed`` / ``validation``; the validation
        split of a pretrain source is a product of ``process`` and shares its hash)."""
        if stage == "raw":
            return self.raw_hash(source_name)
        if stage == "processed":
            return self.processed_hash(source_name)
        if stage == "validation":
            if self.sources[source_name].kind == "pretrain":
                return self.processed_hash(source_name)
            return self.validation_hash(source_name)
        raise ValueError(f"unknown source stage {stage!r}")

    def overlap_warnings(self) -> list[str]:
        """Validation sources that read the same Hub repo as a pretrain source with the same or a nested
        ``data_files`` glob prefix — such a held-out set is likely not disjoint from the training data (prefer the
        pretrain source's own ``validation_tokens`` split)."""
        warnings: list[str] = []
        for val_name in self.sources_of_kind("validation"):
            val = self.sources[val_name]
            if val.hf_id is None:
                continue
            for train_name in self.sources_of_kind("pretrain"):
                train = self.sources[train_name]
                if train.hf_id != val.hf_id:
                    continue
                a, b = _glob_prefix(val.load_kwargs.get("data_files")), _glob_prefix(train.load_kwargs.get("data_files"))
                if a.startswith(b) or b.startswith(a):
                    warnings.append(
                        f"validation source {val_name!r} reads {val.hf_id} like pretrain source {train_name!r} "
                        f"(data_files {val.load_kwargs.get('data_files')!r} vs {train.load_kwargs.get('data_files')!r}): "
                        "the held-out rows may overlap the training data; consider validation_tokens on the pretrain source"
                    )
        return warnings

    def instruct_mixture_hash(self, instruct_mixture_name: str) -> str:
        """Hash of a mixture definition plus the raw hashes and token settings of the sources it draws from."""
        mixture = self.instruct_mixtures[instruct_mixture_name]
        payload = {
            "instruct_mixture": hash_fields(mixture),
            "sources": {name: self.raw_hash(name) for name in mixture.sources},
            "tokens": {name: self.token_settings(name) for name in mixture.sources},
            "budget_tokens": self.instruct_mixture_budget_tokens(instruct_mixture_name),
        }
        return _stable_hash(payload)

    def tokenizer_hash(self) -> str:
        """Hash of the tokenizer definition (the manifest key of `dataset/tokenizers/<name>/`)."""
        return _stable_hash(hash_fields(self.tokenizer))

    def config_hash(self) -> str:
        """Hash of the complete config (recorded in checkpoints so a resume with different data is detected)."""
        return _stable_hash(asdict(self))

    # --- token budgets ---------------------------------------------------------------------------------------------

    def instruct_mixture_budget_tokens(self, instruct_mixture_name: str) -> int:
        """Largest per-stage token demand on a mixture (stages share the built mixture, so max, not sum)."""
        demand = 0
        for stage in self.stages:
            for key, weight in stage.train.items():
                mixture_name = key.partition("/")[0]
                if mixture_name == instruct_mixture_name:
                    demand = max(demand, int(stage.tokens * weight))
        return demand

    def source_budget_tokens(self, source_name: str) -> int:
        """Largest per-stage token demand on a pretrain source (files are shared between stages → max)."""
        demand = 0
        for stage in self.stages:
            weight = stage.train.get(source_name, 0.0)
            demand = max(demand, int(stage.tokens * weight))
        for mixture_name, mixture in self.instruct_mixtures.items():
            if source_name in mixture.sources:
                mixture_budget = self.instruct_mixture_budget_tokens(mixture_name)
                demand = max(demand, int(mixture_budget * mixture.sources[source_name]))
        return demand


# --- helpers ----------------------------------------------------------------------------------------------------------


def _is_non_negative_number(value: Any) -> bool:
    """True for ints/floats >= 0; bools are not numbers here (`True` would silently mean 1 MB)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and value >= 0


def _check_weights(what: str, weights: dict[str, float]) -> None:
    if not weights:
        raise ValueError(f"{what}: must not be empty")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{what}: weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{what}: weights sum to {total:.6f}, expected 1")


def _glob_prefix(pattern: Any) -> str:
    """The literal directory prefix of a ``data_files`` glob (``data/CC-MAIN-2013-20/*.parquet`` -> ``data/CC-MAIN-2013-20/``)."""
    text = "" if pattern is None else str(pattern)
    for i, char in enumerate(text):
        if char in "*?[":
            return text[:i]
    return text


def _stable_hash(payload: Any) -> str:
    """First 16 hex chars of the sha256 of the payload as sorted-key JSON (independent of dict insertion order)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def hash_fields(obj: Any) -> dict[str, Any]:
    """`asdict(obj)` without the fields that still hold their default value (recursively for nested dataclasses).

    Manifests key on hashes of this, so adding a field with a default to the schema, or removing one, never
    invalidates data on disk; only a value that was explicitly set to something else changes the hash.
    """
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if _holds_default(f, value):
            continue
        out[f.name] = _hashable(value)
    return out


def _holds_default(f: Field[Any], value: Any) -> bool:
    if f.default is not MISSING:
        return bool(value == f.default)
    if f.default_factory is not MISSING:
        return bool(value == f.default_factory())
    return False  # required field: never a default


def _hashable(value: Any) -> Any:
    """Plain dicts/lists/scalars for JSON: nested dataclasses via `hash_fields`, tuples become lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return hash_fields(value)
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    return value


def dataset_config_fields() -> list[str]:
    """Top-level field names (used by the settings/CLI layer to recognise dataset-config keys)."""
    return [f.name for f in fields(DatasetConfig)]


def load_dataset_config(path: str | Path, overrides: Optional[list[str]] = None) -> DatasetConfig:
    """Load a dataset config YAML; `overrides` are jsonargparse `--key value` strings (nested keys with dots)."""
    parser = ArgumentParser(description="Dataset config")
    parser.add_class_arguments(DatasetConfig, nested_key=None)
    namespace = parser.parse_path(str(path))
    if overrides:
        namespace = parser.parse_args(overrides, namespace=namespace)
    instantiated = parser.instantiate_classes(namespace)
    values = instantiated.as_dict() if isinstance(instantiated, Namespace) else instantiated
    return DatasetConfig(**values)
