# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `describe`: deterministic Markdown, every section and source present, budgets match the planner.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

from data_preparation.dataset_config import DatasetConfig, SourceConfig, StageConfig, TokenizerConfig, load_dataset_config
from data_preparation.lib.build.describe import GENERATED_WITH, describe, leading_comment

REPO_ROOT = Path(__file__).resolve().parents[3]
CROW = REPO_ROOT / "config" / "datasets" / "crow_300m_final.yaml"
TINY = REPO_ROOT / "config" / "datasets" / "tiny.yaml"

EXPECTED_TINY_STAGE = """### Stage 1: `pretrain_a` (8.2K tokens, transition 25%)

| Train source | Weight | Tokens | Tokens/row (est.) | Rows (est.) |
|---|---:|---:|---:|---:|
| `synthetic_pretrain` | 100.00% | 8.2K | 224 | 37 |

Validation: `synthetic_pretrain` at 100%
"""

EXPECTED_TINY_SPLIT = """| Source | Used in | Held out |
|---|---|---|
| `synthetic_pretrain` | train + val | 5% held out (the first rows of `processed/synthetic_pretrain`) |
| `synthetic_instruct` | train + val | 5% held out (the first rows of `processed/synthetic_instruct`) |
"""


def test_tiny_snippets_and_determinism() -> None:
    tiny_dataset_config = load_dataset_config(TINY)
    text = describe(tiny_dataset_config, "config/datasets/tiny.yaml")
    assert text == describe(tiny_dataset_config, "config/datasets/tiny.yaml")
    assert text.startswith("# Dataset `tiny`\n\nGenerated from `config/datasets/tiny.yaml` with\n")
    assert GENERATED_WITH.format(config="config/datasets/tiny.yaml") + " > docs/data_mixture.md" in text
    assert EXPECTED_TINY_STAGE in text and EXPECTED_TINY_SPLIT in text
    assert "### Stage 3: `finetune` (4.1K tokens, transition 0%)" in text
    assert "| `synthetic_instruct` | 100.00% | 4.1K | 64 | 64 |" in text  # 4096 tokens at the 64 tokens/row estimate
    # budgets are the run-total weight-schedule integral: instruct ramps in over pretrain_b's transition window
    # (8192 × 0.25 / 2 + 4096 = 5120 tokens = 80 rows at 64), pretrain ramps out over it (8192 + 8192 × 0.875 = 15360 tokens = 68.6 rows at 224)
    assert "| `synthetic_instruct` | instruct | `synthetic` | generated (seed 2) | - | budget 5.1K tokens (~80 rows at 64 tokens/row), input inversions 10%, shuffled (seed 2) |" in text
    assert "| `synthetic_pretrain` | pretrain | `synthetic` | generated (seed 0) | - | budget 15.4K tokens (~69 rows at 224 tokens/row) |" in text
    assert "- tokenizer: `synthetic` (synthetic)" in text and "- `token_count`: `tokenizer`" in text
    assert "- `dataset_max_sequence_length`: 256" in text and "- `validation_fraction`: 5%" in text
    assert "- dedup: `exact` (normalize: on, Bloom filter 1 MB per source)" in text and "- quality filter: off" in text
    assert "## Notes" not in text  # no notes given
    assert "mixture" not in text.lower().replace("data_mixture.md", "")  # no mixture section any more
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
    # per-source budgets shown in the source table are the planner's token budgets: the integral of the weight
    # schedule over the whole run (stages sharing a source add up, transition windows count as trapezoids), and
    # the rows they are at the config's tokens-per-row estimate
    budget = cfg.token_budget("fineweb_edu")
    assert budget == 2_594_250_000 and cfg.rows_budget("fineweb_edu") == ceil(budget / 2000) == 1_297_125
    assert f"budget {_tokens(budget)} tokens (~1,297,125 rows at 2,000 tokens/row)" in text
    assert "| `fineweb_edu` | 65.00% | 2.15B | 2,000 | 1,072,500 |" in text  # 2145M tokens at 2000 tokens/row
    # the finetune stage renders like the others: eight instruct sources with their shares
    assert "### Stage 3: `finetune` (150.0M tokens, transition 0%)" in text
    assert "| `flan` | 40.00% | 60.0M | 300 | 200,000 |" in text  # short rows: many of them per token
    assert "Validation: `flan` at 40%, `metamath` at 15%" in text
    assert "| `fineweb_edu` | train + val | 5% held out (the first rows of `processed/fineweb_edu`) |" in text
    assert "| `wikipedia` | train only | none |" in text
    assert "converter `sharegpt_conversations`, filter `sharegpt_quality`, check_limit 100,000" in text
    assert "input inversions 5%, shuffled (seed 42)" in text
    assert "language Python" in text and "text_field `TEXT`" in text
    assert "`nampdn-ai/mini-peS2o`" in text


def test_validation_only_source_renders_rows() -> None:
    cfg = DatasetConfig(
        tokenizer=TokenizerConfig(name="synthetic", kind="synthetic"),
        sources={
            "pre": SourceConfig(kind="pretrain", loader="synthetic"),
            "heldout": SourceConfig(kind="pretrain", loader="synthetic", seed=1, rows=40),
        },
        stages=[StageConfig(name="p", tokens=512, train={"pre": 1.0}, val={"heldout": 1.0})],
        dataset_max_sequence_length=64,
    )
    text = describe(cfg, "t.yaml")
    assert "| `heldout` | val only | all rows (40 downloaded) |" in text
    assert "| `pre` | train only | none |" in text
    assert "| `heldout` | pretrain | `synthetic` | generated (seed 1) | - | rows 40 (validation only) |" in text
    assert "| `pre` | pretrain | `synthetic` | generated (seed 0) | - | budget 512 tokens (~8 rows at 64 tokens/row) |" not in text  # seed 42
    assert "budget 512 tokens (~8 rows at 64 tokens/row)" in text  # the default estimate of 500 clamped at the dataset length 64


def _tokens(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 10**9 else f"{n / 1e6:.1f}M"
