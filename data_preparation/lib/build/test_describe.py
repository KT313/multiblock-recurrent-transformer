# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `describe`: deterministic Markdown, every section and source present, budgets match the planner."""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data_preparation.lib.build.describe import GENERATED_WITH, describe, leading_comment
from data_preparation.lib.schema.dataset_config import DatasetConfig, load_dataset_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CROW = REPO_ROOT / "config" / "datasets" / "crow_300m_final.yaml"
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"

EXPECTED_TINY_STAGE = """### Stage 1: `pretrain_a` (8.2K tokens, transition 25%)

| Train source | Weight | Tokens | Tokens/row (est.) |
|---|---:|---:|---:|
| `synthetic_pretrain` | 100.00% | 8.2K | 224 |

Validation: `synthetic_val` at 100%
"""

EXPECTED_TINY_INSTRUCT = """### `tiny_instruct` (4.1K tokens budget)

`max_tokens` 256, input inversions 10%, validation split 20%, seed 0. Examples = budget × share ÷ `tokens_per_row_estimate`.

| Source | Share | Tokens | Examples (est.) |
|---|---:|---:|---:|
| `synthetic_instruct` | 100.0% | 4.1K | 64 |
"""


def test_tiny_snippets_and_determinism(tiny_dataset_config: DatasetConfig) -> None:
    text = describe(tiny_dataset_config, "config/datasets/tiny.yaml")
    assert text == describe(tiny_dataset_config, "config/datasets/tiny.yaml")
    assert text.startswith("# Dataset `tiny`\n\nGenerated from `config/datasets/tiny.yaml` with\n")
    assert GENERATED_WITH.format(config="config/datasets/tiny.yaml") + " > docs/data_mixture.md" in text
    assert EXPECTED_TINY_STAGE in text and EXPECTED_TINY_INSTRUCT in text
    assert "| `synthetic_val` | 32 | 1 | `synthetic` | generated (seed 1) |" in text
    assert "- tokenizer: `synthetic` (synthetic)" in text and "- `token_count`: `tokenizer`" in text
    assert "- dedup: `exact` (normalize: on)" in text and "- quality filter: off" in text
    assert "## Notes" not in text  # no notes given
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_notes_and_leading_comment() -> None:
    notes = leading_comment(TINY)
    assert notes.startswith("Synthetic dataset for the smoke run") and "\n" in notes and "#" not in notes.split("\n")[0][:1]
    assert "name: tiny" not in notes  # stops at the first key
    text = describe(load_dataset_config(TINY), TINY, notes=notes)
    assert "## Notes\n\n" + notes + "\n" in text
    assert describe(load_dataset_config(TINY), TINY, notes="   \n") == describe(load_dataset_config(TINY), TINY)


def test_crow_lists_every_source_and_matches_planner_budgets() -> None:
    cfg = load_dataset_config(CROW)
    text = describe(cfg, CROW)
    for name, src in cfg.sources.items():
        assert f"| `{name}` | {src.kind} | `{src.loader}` |" in text
        assert src.hf_id is not None and f"`{src.hf_id}`" in text
        assert src.revision is not None and src.revision[:12] in text
    for stage in cfg.stages:
        assert f"`{stage.name}`" in text
    # per-source token budgets shown in the source table are the planner's (max over stages, not the sum)
    assert f"budget {_tokens(cfg.source_budget_tokens('fineweb_edu'))}" in text and cfg.source_budget_tokens("fineweb_edu") == int(3_300_000_000 * 0.65)
    assert "| `fineweb_edu` | 65.00% | 2.15B | 2000 |" in text
    assert "| `flan_instruct` (instruct_mixture) | 100.00% | 150.0M | - |" in text
    budget = cfg.instruct_mixture_budget_tokens("flan_instruct")
    assert f"| `flan` | 40.0% | {_tokens(int(budget * 0.4))} | {ceil(budget * 0.4 / 300):,} |" in text
    assert "converter `sharegpt_conversations`, filter `sharegpt_quality`, check_limit 100,000" in text
    assert "language Python" in text and "text_field `TEXT`" in text
    assert "`nampdn-ai/mini-peS2o`" in text


def _tokens(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 10**9 else f"{n / 1e6:.1f}M"
