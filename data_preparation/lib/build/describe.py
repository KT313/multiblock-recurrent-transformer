# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``prepare.py describe``: render a dataset config as a Markdown document (``docs/data_mixture.md`` is generated
with it, so the documentation of the thesis mixture cannot drift from ``config/datasets/crow_300m_final.yaml``).

Pure function of the config file: tokenizer and processing defaults, one table per stage (weights and derived
token budgets), one per mixture (shares and derived example counts), validation sources and the source registry. The leading
comment block of the YAML file (the lines starting with ``#`` before the first key) is rendered as the "Notes"
section, so config-specific remarks live next to the config. Numbers are the same arithmetic the planner uses
(``DatasetConfig.source_budget_tokens`` / ``instruct_mixture_budget_tokens``); estimates use ``tokens_per_row_estimate``.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data_preparation.lib.schema.dataset_config import DatasetConfig, ProcessingConfig, SourceConfig, TokenizerConfig

GENERATED_WITH = "uv run python data_preparation/prepare.py describe --dataset_config {config}"


def describe(cfg: DatasetConfig, config_path: str | Path, notes: str = "") -> str:
    """The Markdown document for ``cfg`` loaded from ``config_path``; ``notes`` is inserted verbatim after the
    header (see :func:`leading_comment`)."""
    config = Path(config_path).as_posix()
    lines: list[str] = [
        f"# Dataset `{cfg.name}`",
        "",
        f"Generated from `{config}` with",
        "",
        "```bash",
        GENERATED_WITH.format(config=config) + " > docs/data_mixture.md",
        "```",
        "",
        "Do not edit by hand: change the dataset config and regenerate. Token budgets are the stage budgets of the",
        "config times the mixture weights; row and example counts are estimates from `tokens_per_row_estimate` (the",
        "planner refines them with measured token counts once a source has been processed).",
        "",
    ]
    if notes.strip():
        lines += ["## Notes", "", notes.strip(), ""]
    lines += _general(cfg)
    lines += _stages(cfg)
    lines += _instruct_mixtures(cfg)
    lines += _validations(cfg)
    lines += _sources(cfg)
    return "\n".join(lines).rstrip("\n") + "\n"


def leading_comment(config_path: str | Path) -> str:
    """The comment block at the top of a YAML file (``#`` lines before the first non-comment line), as prose."""
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


def _general(cfg: DatasetConfig) -> list[str]:
    total_tokens = sum(stage.tokens for stage in cfg.stages)
    if cfg.token_count == "tokenizer":
        token_count = f"`{cfg.token_count}` (real tokenizer counts)"
    else:
        token_count = f"`{cfg.token_count}` (chars / 4)"
    return [
        "## Tokenizer and token counting",
        "",
        f"- tokenizer: {_tokenizer_label(cfg.tokenizer)}",
        f"- `max_seq_length`: {cfg.max_seq_length} (token-count cap per document; the run config's `block_size` must not exceed it)",
        f"- `token_count`: {token_count}",
        f"- training tokens over all stages: {_tokens(total_tokens)}",
        "",
        "## Processing defaults (`pretrain` sources)",
        "",
        *_processing_lines(cfg.processing),
        "",
    ]


def _tokenizer_label(tok: TokenizerConfig) -> str:
    """e.g. "`llama-32k` (hf, `hf-internal-testing/llama-tokenizer` @ `<sha>`)"."""
    label = f"`{tok.name}` ({tok.kind}"
    if tok.hf_id:
        label += f", `{tok.hf_id}`"
        if tok.revision:
            label += f" @ `{tok.revision}`"
    return label + ")"


def _processing_lines(p: ProcessingConfig) -> list[str]:
    return [
        f"- length filter: {p.min_chars} <= chars, truncated at {p.max_chars} chars",
        f"- dedup: {_dedup_label(p)}",
        f"- quality filter: {_yn(p.quality_filter)}",
        f"- decontamination: {_decontamination_label(p)}",
    ]


def _dedup_label(p: ProcessingConfig) -> str:
    label = f"`{p.dedup.mode}`"
    if p.dedup.mode == "exact":
        label += f" (normalize: {_yn(p.dedup.normalize)})"
    elif p.dedup.mode == "minhash":
        label += f" (threshold {p.dedup.threshold}, {p.dedup.num_perm} permutations, {p.dedup.ngram}-grams)"
    return label


def _decontamination_label(p: ProcessingConfig) -> str:
    decon = p.decontamination
    label = _yn(decon.enabled)
    if decon.enabled:
        label += f" ({decon.ngram}-grams, threshold {decon.threshold}, benchmarks: {', '.join(decon.benchmarks)})"
    return label


def _stages(cfg: DatasetConfig) -> list[str]:
    lines = ["## Stages", ""]
    for index, stage in enumerate(cfg.stages):
        lines += [
            f"### Stage {index + 1}: `{stage.name}` ({_tokens(stage.tokens)} tokens, transition {stage.transition_pct:.0%})",
            "",
            "| Train source | Weight | Tokens | Tokens/row (est.) |",
            "|---|---:|---:|---:|",
        ]
        for key, weight in stage.train.items():
            lines.append(f"| {_key(cfg, key)} | {weight:.2%} | {_tokens(int(stage.tokens * weight))} | {_estimate(cfg, key)} |")
        validation = ", ".join(f"{_key(cfg, key)} at {weight:.0%}" for key, weight in stage.val.items())
        lines += ["", f"Validation: {validation}", ""]
    return lines


def _instruct_mixtures(cfg: DatasetConfig) -> list[str]:
    if not cfg.instruct_mixtures:
        return []
    lines = ["## Instruct mixtures", ""]
    for name, mixture in cfg.instruct_mixtures.items():
        budget = cfg.instruct_mixture_budget_tokens(name)
        lines += [
            f"### `{name}` ({_tokens(budget)} tokens budget)",
            "",
            f"`max_tokens` {mixture.max_tokens}, input inversions {mixture.input_inversions:.0%}, validation split "
            f"{mixture.val_split:.0%}, seed {mixture.seed}. Examples = budget × share ÷ `tokens_per_row_estimate`.",
            "",
            "| Source | Share | Tokens | Examples (est.) |",
            "|---|---:|---:|---:|",
        ]
        for src, share in mixture.sources.items():
            share_tokens = budget * share
            examples = ceil(share_tokens / cfg.sources[src].tokens_per_row_estimate)
            lines.append(f"| `{src}` | {share:.1%} | {_tokens(int(share_tokens))} | {examples:,} |")
        lines.append("")
    return lines


def _validations(cfg: DatasetConfig) -> list[str]:
    splits = [name for name in cfg.sources_of_kind("pretrain") if cfg.sources[name].validation_tokens]
    names = cfg.sources_of_kind("validation")
    if not names and not splits:
        return []
    lines = ["## Held-out validation sets", ""]
    if splits:
        lines += ["| Split | Tokens | From |", "|---|---:|---|"]
        for name in splits:
            src = cfg.sources[name]
            lines.append(
                f"| `{name}/validation` | {_tokens(src.validation_tokens)} | the first processed rows of `{name}` "
                "(deduplicated together with the training rows, never part of `processed/`) |"
            )
        lines.append("")
    if names:
        lines += ["| Source | Rows | Seed | Loader | Origin |", "|---|---:|---:|---|---|"]
        for name in names:
            src = cfg.sources[name]
            lines.append(f"| `{name}` | {src.rows:,} | {src.seed} | `{src.loader}` | {_origin(src)} |")
        lines.append("")
    return lines


def _sources(cfg: DatasetConfig) -> list[str]:
    lines = ["## Sources", "", "| Source | Kind | Loader | Origin | Revision | Details |", "|---|---|---|---|---|---|"]
    for name, src in cfg.sources.items():
        revision = f"`{src.revision[:12]}`" if src.revision else "-"
        details = _details(cfg, name) or "-"
        lines.append(f"| `{name}` | {src.kind} | `{src.loader}` | {_origin(src)} | {revision} | {details} |")
    return lines + [""]


# --- formatting helpers ------------------------------------------------------------------------------------------------


def _base_name(key: str) -> str:
    """Stage keys name a source or a mixture, optionally with a ``/split`` suffix (``flan_instruct/validation``)."""
    return key.partition("/")[0]


def _key(cfg: DatasetConfig, key: str) -> str:
    if _base_name(key) in cfg.instruct_mixtures:
        return f"`{key}` (instruct_mixture)"
    return f"`{key}`"


def _estimate(cfg: DatasetConfig, key: str) -> str:
    base = _base_name(key)
    if base in cfg.instruct_mixtures:
        return "-"
    return str(cfg.sources[base].tokens_per_row_estimate)


def _origin(src: SourceConfig) -> str:
    if src.loader == "synthetic":
        return f"generated (seed {src.seed})"
    if src.loader == "local":
        return f"`{src.path}`"
    origin = f"`{src.hf_id}`"
    if src.load_kwargs:
        origin += " " + ", ".join(f"{k}={v}" for k, v in src.load_kwargs.items())
    if src.split != "train":
        origin += f" split={src.split}"
    return origin


def _details(cfg: DatasetConfig, name: str) -> str:
    src = cfg.sources[name]
    parts: list[str] = []
    if src.language:
        parts.append(f"language {src.language}")
    if src.text_field != "text" and src.kind != "instruct":
        parts.append(f"text_field `{src.text_field}`")
    if src.fields:
        parts.append("fields " + ", ".join(f"{k}←`{v}`" for k, v in src.fields.items()))
    if src.converter:
        parts.append(f"converter `{src.converter}`")
    if src.filter:
        parts.append(f"filter `{src.filter}`")
    if src.check_limit is not None:
        parts.append(f"check_limit {src.check_limit:,}")
    if src.kind == "pretrain":
        parts.append(f"budget {_tokens(cfg.source_budget_tokens(name))}")
        if src.validation_tokens:
            parts.append(f"validation split {_tokens(src.validation_tokens)}")
        if src.processing is not None:
            override_lines = _processing_lines(src.processing)
            parts.append("processing override: " + "; ".join(line.removeprefix("- ") for line in override_lines))
    return ", ".join(parts)


def _tokens(n: int) -> str:
    """Human-scale token count: 3.30B, 150.0M, 12.5K, 42."""
    if n >= 10**9:
        return f"{n / 1e9:.2f}B"
    if n >= 10**6:
        return f"{n / 1e6:.1f}M"
    if n >= 10**3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _yn(flag: bool) -> str:
    return "on" if flag else "off"


__all__ = ["GENERATED_WITH", "describe", "leading_comment"]
