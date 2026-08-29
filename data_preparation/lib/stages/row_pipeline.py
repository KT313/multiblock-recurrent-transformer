# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Pure row-level helpers of the preparation pipeline (no I/O, no config): length filtering, quality heuristics,
benchmark-contamination n-grams, instruct-row inversions and field checks. The stage functions in ``stages/*.py``
apply them to shards.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

Row = dict[str, Any]

# the explicit annotation is what pyarrow-stubs needs to accept the list in ``pa.schema``
_OUTPUT_FIELDS: list[tuple[str, pa.DataType]] = [
    ("text", pa.string()),
    ("source", pa.string()),
    ("original_length", pa.int64()),
]
FILTERED_SCHEMA = pa.schema(_OUTPUT_FIELDS)
FILTERED_SCHEMA_WITH_TOKENS = pa.schema([*_OUTPUT_FIELDS, ("tokens", pa.int64())])
_WHITESPACE = re.compile(r"\s+")


# --- pretrain: length filter -------------------------------------------------------------------------------------------


def preprocess_batch(
    batch: pa.RecordBatch, text_field: str, source_name: str, min_chars: int, max_chars: int
) -> tuple[pa.RecordBatch, dict[str, int]]:
    """Drop null / shorter-than-``min_chars`` texts, truncate to ``max_chars`` characters.

    Returns a ``FILTERED_SCHEMA`` batch (``text``, ``source``, ``original_length``) and per-batch statistics. A
    ``tokens`` column of the input batch (the raw token counts) is carried through as a fourth column, so the
    caller only has to recount the truncated rows (``original_length > max_chars``).
    """
    if text_field not in batch.schema.names:
        raise ValueError(f"{source_name}: text_field {text_field!r} not in columns {batch.schema.names}")
    stats = {"input_samples": len(batch), "removed_too_short": 0, "removed_invalid": 0, "truncated": 0}

    # 1. drop null texts
    # the raw shards store text as (large_)string; the cast only narrows the stub type, nothing happens at runtime
    text_array = cast(pa.StringArray, batch[text_field])
    is_valid = pc.is_valid(text_array)
    # pyarrow-stubs restricts `pc.sum` to numeric arrays, but summing a boolean array is valid pyarrow
    stats["removed_invalid"] = len(batch) - pc.sum(is_valid).as_py()  # type: ignore[type-var]
    if stats["removed_invalid"] > 0:
        batch = batch.filter(is_valid)
        text_array = cast(pa.StringArray, batch[text_field])
    if len(batch) == 0:
        return _empty_filtered_batch(), stats | {"output_samples": 0}

    # 2. drop texts shorter than min_chars
    original_lengths = pc.utf8_length(text_array)
    # pyarrow-stubs does not accept a Python int as the second operand, pyarrow does
    is_long_enough = pc.greater_equal(original_lengths, min_chars)  # type: ignore[call-overload]
    stats["removed_too_short"] = len(batch) - pc.sum(is_long_enough).as_py()
    if stats["removed_too_short"] > 0:
        batch = batch.filter(is_long_enough)
        text_array = cast(pa.StringArray, batch[text_field])
        original_lengths = pc.utf8_length(text_array)
    if len(batch) == 0:
        return _empty_filtered_batch(), stats | {"output_samples": 0}

    # 3. truncate to max_chars (original_length keeps the length before truncation)
    texts = cast(list[str], text_array.to_pylist())  # nulls were filtered above
    truncated_texts = [text[:max_chars] for text in texts]
    stats["truncated"] = sum(1 for text in texts if len(text) > max_chars)

    arrays: list[pa.Array[Any]] = [
        pa.array(truncated_texts, type=pa.string()),
        pa.array([source_name] * len(truncated_texts), type=pa.string()),
        pc.cast(original_lengths, pa.int64()),
    ]
    schema = FILTERED_SCHEMA
    if "tokens" in batch.schema.names:
        arrays.append(pc.cast(batch["tokens"], pa.int64()))
        schema = FILTERED_SCHEMA_WITH_TOKENS
    output = pa.RecordBatch.from_arrays(arrays, schema=schema)
    stats["output_samples"] = len(output)
    return output, stats


def _empty_filtered_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist([], schema=FILTERED_SCHEMA)


# --- pretrain: quality / contamination ---------------------------------------------------------------------------------


def get_ngrams(text: str, n: int = 5) -> list[str]:
    """Word n-grams of ``text`` (for MinHash)."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def check_quality(text: str) -> tuple[bool, str]:
    """Heuristic prose quality check (>= 3 sentences, <= 30% ALL-CAPS words, >= 25% alphanumeric, <= 30% duplicate
    2-grams, <= 20% duplicate 3-grams); returns ``(passes, reason)``."""
    if len(text) < 10:
        return False, "too_short"
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 10]
    if len(sentences) < 3:
        return False, "too_few_sentences"
    words = text.split()
    if len(words) == 0:
        return False, "no_words"
    caps_words = [w for w in words if w.isupper() and len(w) > 1]
    if len(caps_words) / len(words) > 0.3:
        return False, "too_many_caps"
    alphanumeric = sum(1 for c in text if c.isalnum() or c.isspace())
    if alphanumeric / len(text) < 0.25:
        return False, "too_few_alphanumeric"
    if len(words) > 10 and _unique_ratio(get_ngrams(text, 2)) < 0.7:
        return False, "too_repetitive_bigrams"
    if len(words) > 20 and _unique_ratio(get_ngrams(text, 3)) < 0.8:
        return False, "too_repetitive_trigrams"
    return True, "passed"


def _unique_ratio(items: list[str]) -> float:
    """Share of distinct entries in ``items`` (1.0 = no repetition)."""
    return len(set(items)) / len(items)


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace (for contamination checks)."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def get_ngram_set(text: str, n: int = 13) -> set[str]:
    """Set of normalized word n-grams of ``text``."""
    words = normalize_text(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_contamination(
    text: str, benchmark_ngrams: dict[str, set[str]], n: int = 13, threshold: float = 0.1
) -> tuple[bool, list[str]]:
    """A document is contaminated if more than ``threshold`` of its n-grams occur in any benchmark test set.

    Returns ``(is_contaminated, names of the contaminating benchmarks)``.
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
    """The text whose token count decides an instruct row's length: instruction, input and output joined."""
    return f"{row['instruction']}\n{row.get('input') or ''}\n{row['output']}"


def check_length(tokens: int, max_tokens: int) -> bool:
    """Keep an instruct row whose measured token count does not exceed ``max_tokens``."""
    return tokens <= max_tokens


def create_input_inversion(row: Row) -> Row:
    """Ask for the instruction given the output (swap direction); unchanged when instruction or output is missing
    or empty."""
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
    """True if instruction and output are non-empty after stripping."""
    instruction, output = row.get("instruction"), row.get("output")
    return bool(instruction and str(instruction).strip()) and bool(output and str(output).strip())
