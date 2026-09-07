# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
prepare.py describe: render a dataset config as a Markdown document (docs/data_mixture.md is generated
with it, so the documentation of the thesis mixture cannot drift from config/datasets/crow_300m_final.yaml).

Pure function of the config file: tokenizer, sequence length and processing defaults, one table per stage (weights,
derived token budgets and estimated row counts), the validation split per source and the source registry. The
leading comment block of the YAML file (the lines starting with # before the first key) is rendered as the "Notes"
section, so config-specific remarks live next to the config. Per-source budgets are the planner's arithmetic
(DatasetConfig.token_budget: the integral of the weight schedule over the run, and DatasetConfig.rows_budget at the
describe_tokens_per_row estimate); the per-stage tables show stage.tokens × weight and the rows that many tokens
are at the estimate.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil
from pathlib import Path

from data_preparation.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig, TokenizerConfig

GENERATED_WITH = "uv run python data_preparation/prepare.py describe --dataset_config {config}"


def describe(config: DatasetConfig, config_path: str | Path, notes: str = "") -> str:
    """
    The Markdown document for config loaded from config_path; notes is inserted verbatim after the
    header (see :func:`leading_comment`).
    """

    config_file = Path(config_path).as_posix()
    lines: list[str] = [
        f"# Dataset `{Path(config_path).stem}`",
        "",
        f"Generated from `{config_file}` with",
        "",
        "```bash",
        GENERATED_WITH.format(config=config_file) + " > docs/data_mixture.md",
        "```",
        "",
        "Do not edit by hand: change the dataset config and regenerate. Token budgets are the stage budgets of the",
        "config times the stage weights; the training loader packs rows end to end, so a",
        "source is consumed by the token length of its rows. The training stream realises the weights as token shares by",
        "filling its packing pool from the source with the largest token deficit (`BatchStream` in `training/step.py`).",
        "The rows columns estimate how many rows that is from",
        "`describe_tokens_per_row` (the rate the planner sizes the first download with, clamped at the training length; the",
        "downloaded shards then measure the real one).",
        "",
    ]
    if notes.strip():
        lines += ["## Notes", "", notes.strip(), ""]
    lines += _general(config)
    lines += _stages(config)
    lines += _validation_split(config)
    lines += _sources(config)
    return "\n".join(lines).rstrip("\n") + "\n"


def leading_comment(config_path: str | Path) -> str:
    """
    The comment block at the top of a YAML file (# lines before the first non-comment line), as prose.
    """

    out: list[str] = []
    for raw in Path(config_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            out.append(line[1:].strip())
        elif line:
            break  # first real YAML line ends the comment block
        else:
            out.append("")  # blank lines inside the block become paragraph breaks
    return "\n".join(out).strip()


# --- sections ----------------------------------------------------------------------------------------------------------


def _general(config: DatasetConfig) -> list[str]:
    total_tokens = sum(stage.tokens for stage in config.stages)
    if config.token_count == "tokenizer":
        token_count = f"`{config.token_count}` (real tokenizer counts)"
    else:
        token_count = f"`{config.token_count}` (chars / 4)"
    return [
        "## Tokenizer, sequence length and token counting",
        "",
        f"- tokenizer: {_tokenizer_label(config.tokenizer)}",
        f"- `training_target_sequence_length`: {config.training_target_sequence_length} (the run trains at this length: a row counts min(its tokens, this) towards the budget)",
        f"- `dataset_max_sequence_length`: {config.dataset_max_sequence_length} (pretrain rows are truncated to this many tokens when downloaded, longer instruct rows are dropped)",
        f"- `token_count`: {token_count}",
        f"- `validation_fraction`: {config.validation_fraction:.0%} of a source used for training and validation is held out",
        f"- training tokens over all stages: {_tokens(total_tokens)}",
        "",
        "## Processing defaults",
        "",
        *_processing_lines(config.processing),
        "",
    ]


def _tokenizer_label(tokenizer: TokenizerConfig) -> str:
    """
    e.g. "`llama-32k` (hf, `hf-internal-testing/llama-tokenizer` @ `<sha>`)".
    """

    label = f"`{tokenizer.name}` ({tokenizer.kind}"
    if tokenizer.hf_id:
        label += f", `{tokenizer.hf_id}`"
        if tokenizer.revision:
            label += f" @ `{tokenizer.revision}`"
    return label + ")"


def _processing_lines(processing: ProcessingConfig) -> list[str]:
    return [
        f"- length filter: {processing.min_chars} <= chars (pretrain only; rows are cut at `dataset_max_sequence_length` tokens when downloaded)",
        f"- dedup: {_dedup_label(processing)}",
        f"- quality filter: {_yn(processing.quality_filter)}",
        f"- decontamination: {_decontamination_label(processing)}",
    ]


def _dedup_label(processing: ProcessingConfig) -> str:
    dedup = processing.dedup
    label = f"`{dedup.mode}`"
    if dedup.mode == "exact":
        label += f" (normalize: {_yn(dedup.normalize)}, Bloom filter {dedup.bloom_memory_mb} MB per source)"
    elif dedup.mode == "minhash":
        label += f" (threshold {dedup.threshold}, {dedup.num_perm} permutations, {dedup.ngram}-grams)"
    return label


def _decontamination_label(processing: ProcessingConfig) -> str:
    decon = processing.decontamination
    label = _yn(decon.enabled)
    if decon.enabled:
        label += f" ({decon.ngram}-grams, threshold {decon.threshold}, benchmarks: {', '.join(decon.benchmarks)})"
    return label


def _stages(config: DatasetConfig) -> list[str]:
    lines = ["## Stages", ""]
    for index, stage in enumerate(config.stages):
        lines += [
            f"### Stage {index + 1}: `{stage.name}` ({_tokens(stage.tokens)} tokens, transition {stage.transition_pct:.0%})",
            "",
            "| Train source | Weight | Tokens | Tokens/row (est.) | Rows (est.) |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, weight in stage.train.items():
            tokens = stage.tokens * weight
            rate = config.tokens_per_row_rate(name)
            lines.append(f"| `{name}` | {weight:.2%} | {_tokens(int(tokens))} | {_rate(rate)} | {ceil(tokens / rate):,} |")
        validation = ", ".join(f"`{name}` at {weight:.0%}" for name, weight in stage.val.items())
        lines += ["", f"Validation: {validation}", ""]
    return lines


def _validation_split(config: DatasetConfig) -> list[str]:
    """
    One line per source: how the training resolver splits its processed rows.
    """

    lines = [
        "## Validation split",
        "",
        "Decided at training time, never on disk: a source used only for training is all training rows, one used only",
        "for validation is all validation rows (`rows` says how many are downloaded), one used for both gives its first",
        "`validation_fraction` of processed rows to validation (deduplicated as one set, so the two never share a document).",
        "",
        "| Source | Used in | Held out |",
        "|---|---|---|",
    ]
    for name in config.sources:
        lines.append(f"| `{name}` | {_usage(config, name)} | {_held_out(config, name)} |")
    return lines + [""]


def _usage(config: DatasetConfig, name: str) -> str:
    in_train, in_val = config.used_in_train(name), config.used_in_val(name)
    if in_train and in_val:
        return "train + val"
    return "train only" if in_train else "val only"


def _held_out(config: DatasetConfig, name: str) -> str:
    if config.used_in_train(name) and config.used_in_val(name):
        return f"{config.validation_fraction_of(name):.0%} held out (the first rows of `processed/{name}`)"
    if config.used_in_val(name):
        rows = config.sources[name].rows
        return f"all rows ({rows:,} downloaded)" if rows is not None else "all rows"
    return "none"


def _sources(config: DatasetConfig) -> list[str]:
    lines = ["## Sources", "", "| Source | Kind | Loader | Origin | Revision | Details |", "|---|---|---|---|---|---|"]
    for name, source in config.sources.items():
        revision = f"`{source.revision[:12]}`" if source.revision else "-"
        details = _details(config, name) or "-"
        lines.append(f"| `{name}` | {source.kind} | `{source.loader}` | {_origin(source)} | {revision} | {details} |")
    return lines + [""]


# --- formatting helpers ------------------------------------------------------------------------------------------------


def _origin(source: SourceConfig) -> str:
    if source.loader == "synthetic":
        return f"generated (seed {source.seed})"
    if source.loader == "local":
        return f"`{source.path}`"
    origin = f"`{source.hf_id}`"
    if source.load_kwargs:
        origin += " " + ", ".join(f"{key}={value}" for key, value in source.load_kwargs.items())
    if source.split != "train":
        origin += f" split={source.split}"
    return origin


def _details(config: DatasetConfig, name: str) -> str:
    source = config.sources[name]
    parts: list[str] = []
    if source.language:
        parts.append(f"language {source.language}")
    if source.text_field != "text" and source.kind != "instruct":
        parts.append(f"text_field `{source.text_field}`")
    if source.fields:
        parts.append("fields " + ", ".join(f"{field}←`{column}`" for field, column in source.fields.items()))
    if source.converter:
        parts.append(f"converter `{source.converter}`")
    if source.filter:
        parts.append(f"filter `{source.filter}`")
    if source.check_limit is not None:
        parts.append(f"check_limit {source.check_limit:,}")
    if config.used_in_train(name):
        rate = config.tokens_per_row_rate(name)
        parts.append(f"budget {_tokens(config.token_budget(name))} tokens (~{config.rows_budget(name):,} rows at {_rate(rate)} tokens/row)")
    else:
        parts.append(f"rows {source.rows:,} (validation only)")
    if source.kind == "instruct":
        parts.append(f"input inversions {source.input_inversions:.0%}")
    if config.shuffle_of(name):
        parts.append(f"shuffled (seed {source.seed})")
    if source.processing is not None:
        override_lines = _processing_lines(source.processing)
        parts.append("processing override: " + "; ".join(line.removeprefix("- ") for line in override_lines))
    return ", ".join(parts)


def _rate(rate: Fraction) -> str:
    """
    A tokens-per-row rate as the planner uses it: the estimate, clamped at the dataset length.
    """

    return f"{float(rate):,.0f}"


def _tokens(n: int) -> str:
    """
    Human-scale token count: 3.30B, 150.0M, 12.5K, 42.
    """

    if n >= 10**9:
        return f"{n / 1e9:.2f}B"
    if n >= 10**6:
        return f"{n / 1e6:.1f}M"
    if n >= 10**3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _yn(flag: bool) -> str:
    return "on" if flag else "off"
