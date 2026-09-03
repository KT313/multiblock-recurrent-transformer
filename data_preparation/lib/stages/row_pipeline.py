# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Pure row-level helpers of the preparation pipeline (no I/O, no config): length filtering, quality heuristics,
benchmark-contamination n-grams, instruct-row inversions and field checks. The stage functions in stages/*.py
apply them to shards.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

Row = dict[str, Any]

_WHITESPACE = re.compile(r"\s+")


# --- pretrain: length filter -------------------------------------------------------------------------------------------


def preprocess_batch(batch: pa.RecordBatch, text_field: str, source_name: str, min_chars: int) -> tuple[list[Row], dict[str, int]]:
    """
    Drop null / shorter-than-min_chars texts (the upper bound is the token truncation at download time).

    Returns the kept rows as {"text": ...} (plus "tokens" when the input batch carries the raw token counts)
    and the batch's statistics: input_samples, removed_invalid, removed_too_short, output_samples.
    """

    if text_field not in batch.schema.names:
        raise ValueError(f"{source_name}: text_field {text_field!r} not in columns {batch.schema.names}")
    stats = {"input_samples": len(batch), "removed_too_short": 0, "removed_invalid": 0}

    # 1. drop null texts
    # the raw shards store text as (large_)string; the cast only narrows the stub type, nothing happens at runtime
    text_array = cast(pa.StringArray, batch[text_field])
    is_valid = pc.is_valid(text_array)
    # pyarrow-stubs restricts `pc.sum` to numeric arrays, but summing a boolean array is valid pyarrow
    stats["removed_invalid"] = len(batch) - pc.sum(is_valid).as_py()  # type: ignore[type-var]
    if stats["removed_invalid"] > 0:
        batch = batch.filter(is_valid)
        text_array = cast(pa.StringArray, batch[text_field])

    # 2. drop texts shorter than min_chars (in code points, not bytes)
    if len(batch) > 0:
        # pyarrow-stubs does not accept a Python int as the second operand, pyarrow does
        is_long_enough = pc.greater_equal(pc.utf8_length(text_array), min_chars)  # type: ignore[call-overload]
        stats["removed_too_short"] = len(batch) - pc.sum(is_long_enough).as_py()
        if stats["removed_too_short"] > 0:
            batch = batch.filter(is_long_enough)
            text_array = cast(pa.StringArray, batch[text_field])

    texts = cast(list[str], text_array.to_pylist())  # nulls are gone: the values are strings
    if "tokens" in batch.schema.names:
        tokens = cast(list[int], batch["tokens"].to_pylist())
        rows: list[Row] = [{"text": text, "tokens": count} for text, count in zip(texts, tokens, strict=True)]
    else:
        rows = [{"text": text} for text in texts]
    stats["output_samples"] = len(rows)
    return rows, stats


# --- pretrain: quality / contamination ---------------------------------------------------------------------------------


def get_ngrams(text: str, n: int = 5) -> list[str]:
    """
    Word n-grams of text (for MinHash).
    """

    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def check_quality(text: str) -> tuple[bool, str]:
    """
    Heuristic prose quality check (>= 3 sentences, <= 30% ALL-CAPS words, >= 25% alphanumeric, <= 30% duplicate
    2-grams, <= 20% duplicate 3-grams); returns (passes, reason).
    """

    if len(text) < 10:
        return False, "too_short"
    sentences = [sentence for sentence in re.split(r"[.!?]+", text) if len(sentence.strip()) > 10]
    if len(sentences) < 3:
        return False, "too_few_sentences"
    words = text.split()
    if len(words) == 0:
        return False, "no_words"
    caps_words = [word for word in words if word.isupper() and len(word) > 1]
    if len(caps_words) / len(words) > 0.3:
        return False, "too_many_caps"
    alphanumeric = sum(1 for char in text if char.isalnum() or char.isspace())
    if alphanumeric / len(text) < 0.25:
        return False, "too_few_alphanumeric"
    if len(words) > 10 and _unique_ratio(get_ngrams(text, 2)) < 0.7:
        return False, "too_repetitive_bigrams"
    if len(words) > 20 and _unique_ratio(get_ngrams(text, 3)) < 0.8:
        return False, "too_repetitive_trigrams"
    return True, "passed"


def _unique_ratio(items: list[str]) -> float:
    """
    Share of distinct entries in items (1.0 = no repetition).
    """

    return len(set(items)) / len(items)


def normalize_text(text: str) -> str:
    """
    Lowercase and collapse whitespace (for contamination checks).
    """

    return _WHITESPACE.sub(" ", text.lower()).strip()


def get_ngram_set(text: str, n: int = 13) -> set[str]:
    """
    Set of normalized word n-grams of text.
    """

    words = normalize_text(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_contamination(
    text: str, benchmark_ngrams: dict[str, set[str]], n: int = 13, threshold: float = 0.1
) -> tuple[bool, list[str]]:
    """
    A document is contaminated if more than threshold of its n-grams occur in any benchmark test set.

    Returns (is_contaminated, names of the contaminating benchmarks).
    """

    doc_ngrams = get_ngram_set(text, n)
    if not doc_ngrams:
        return False, []
    contaminated = []
    for name, test_ngrams in benchmark_ngrams.items():
        if not test_ngrams:
            continue
        overlap = len(doc_ngrams & test_ngrams) / len(doc_ngrams)
        if overlap > threshold:
            contaminated.append(name)
    return len(contaminated) > 0, contaminated


# --- instruct rows -----------------------------------------------------------------------------------------------------


def instruct_text(row: Row) -> str:
    """
    The text whose token count decides an instruct row's length: instruction, input and output joined.
    """

    return f"{row['instruction']}\n{row.get('input') or ''}\n{row['output']}"


def create_input_inversion(row: Row) -> Row:
    """
    Ask for the instruction given the output (swap direction); unchanged when instruction or output is missing
    or empty.
    """

    if not row.get("instruction") or not row.get("output"):
        return row
    inverted_output = str(row["instruction"])
    if row.get("input"):
        inverted_output += f"\nInput: {row['input']}"
    return {
        "instruction": f"Given this output, what was the likely instruction or input?\n\nOutput: {row['output']}",
        "input": "",
        "output": inverted_output,
    }


def has_required_fields(row: Row) -> bool:
    """
    True if instruction and output are non-empty after stripping.
    """

    instruction, output = row.get("instruction"), row.get("output")
    return bool(instruction and str(instruction).strip()) and bool(output and str(output).strip())
