# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for data_preparation.lib.stages.row_pipeline: length filter batches, quality heuristics, contamination n-grams,
instruct inversions and field checks.
"""

from typing import Any

import pyarrow as pa
import pytest

from data_preparation.lib.stages import row_pipeline as rp

GOOD = (
    "The quick brown fox jumps over the lazy dog. Then it went home to sleep. It dreamed of chasing rabbits all night."
)


# --- preprocess_batch ------------------------------------------------------------------------------------------------


def test_preprocess_batch_filters_and_counts() -> None:
    batch = pa.RecordBatch.from_pydict(
        {"content": ["short", None, "x" * 50, "y" * 120, "z" * 51], "meta": [1, 2, 3, 4, 5]}
    )
    out, stats = rp.preprocess_batch(batch, "content", "mysrc", min_chars=50)
    assert stats == {"input_samples": 5, "removed_too_short": 1, "removed_invalid": 1, "output_samples": 3}
    assert [len(r["text"]) for r in out] == [50, 120, 51], "nothing is truncated (raw is truncated at download)"
    assert all(set(r) == {"text"} for r in out), "only what the build reads"


def test_preprocess_batch_min_chars_is_inclusive() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["a" * 49, "b" * 50, "c" * 100]})
    out, stats = rp.preprocess_batch(batch, "text", "s", min_chars=50)
    assert stats == {"input_samples": 3, "removed_too_short": 1, "removed_invalid": 0, "output_samples": 2}
    assert [len(r["text"]) for r in out] == [50, 100]


def test_preprocess_batch_carries_the_tokens_column() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["a" * 5, "b" * 50], "tokens": [2, 12]})
    out, stats = rp.preprocess_batch(batch, "text", "s", min_chars=10)
    assert out == [{"text": "b" * 50, "tokens": 12}] and stats["output_samples"] == 1


def test_preprocess_batch_all_invalid_or_short_returns_no_rows() -> None:
    batch = pa.RecordBatch.from_pydict({"text": [None, None]})
    out, stats = rp.preprocess_batch(batch, "text", "s", 50)
    assert out == [] and stats["removed_invalid"] == 2 and stats["output_samples"] == 0

    batch = pa.RecordBatch.from_pydict({"text": ["tiny", "tiny2"]})
    out, stats = rp.preprocess_batch(batch, "text", "s", 50)
    assert out == [] and stats["removed_too_short"] == 2 and stats["output_samples"] == 0


def test_preprocess_batch_uses_unicode_length() -> None:
    batch = pa.RecordBatch.from_pydict({"text": ["é" * 10, "é" * 9]})
    out, stats = rp.preprocess_batch(batch, "text", "s", min_chars=10)
    assert stats["output_samples"] == 1 and stats["removed_too_short"] == 1
    assert out == [{"text": "é" * 10}]


def test_preprocess_batch_missing_text_field_raises() -> None:
    batch = pa.RecordBatch.from_pydict({"body": ["x" * 60]})
    with pytest.raises(ValueError, match="text_field 'text' not in columns"):
        rp.preprocess_batch(batch, "text", "src", 1)


# --- quality -----------------------------------------------------------------------------------------------------------


def _repetitive_trigrams() -> str:
    """
    >20 words; bigram uniqueness 28/39 >= 0.7 but trigram uniqueness 28/38 < 0.8.
    """

    block = [f"b{i}" for i in range(12)]
    unique = [f"u{i}" for i in range(16)]
    words = block + block + unique
    return ". ".join(" ".join(words[i : i + 12]) for i in range(0, len(words), 12)) + "."


def test_get_ngrams() -> None:
    assert rp.get_ngrams("a b c d e f", n=5) == ["a b c d e", "b c d e f"]
    assert rp.get_ngrams("a b", n=5) == []
    assert rp.get_ngrams("x  y\tz", n=2) == ["x y", "y z"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("hi", "too_short"),
        ("This is one sentence. Another one here.", "too_few_sentences"),
        ("THIS IS A LOUD SENTENCE HERE. ANOTHER LOUD SENTENCE. AND ONE MORE LOUD ONE.", "too_many_caps"),
        ("@@@@@@@@@@@@ a. ############ b. $$$$$$$$$$$$ c.", "too_few_alphanumeric_or_space"),
        ("the cat sat. the cat sat. the cat sat. the cat sat. the cat sat.", "too_repetitive_bigrams"),
        (_repetitive_trigrams(), "too_repetitive_trigrams"),
        (GOOD, "passed"),
    ],
)
def test_check_quality_reasons(text: str, reason: str) -> None:
    passes, got = rp.check_quality(text)
    assert got == reason
    assert passes is (reason == "passed")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abcdefghi", "too_short"),
        ("abcdefghij", "too_few_sentences"),
        ("abcdefghij. abcdefghij. abcdefghij.", "too_few_sentences"),
        ("abcdefghijk. abcdefghijk. abcdefghijk.", "passed"),
        ("AA BB CC alpha. beta gamma delta. epsilon zeta eta.", "passed"),
        ("AA BB CC DD. beta gamma delta. epsilon zeta eta.", "too_many_caps"),
        ("aa bb cc dd. ee ff gg hh. ii jj kk ll." + "#" * 102, "passed"),
        ("aa bb cc dd. ee ff gg hh. ii jj kk ll." + "#" * 103, "too_few_alphanumeric_or_space"),
        ("alpha beta gamma. alpha beta gamma. alpha delta epsilon zeta eta.", "passed"),
        ("alpha beta gamma. alpha beta gamma. alpha beta epsilon zeta eta.", "too_repetitive_bigrams"),
        (" ".join(["b0 b1 b2 b3 b4 b5."] * 2 + [f"u{i}" for i in range(9)] + ["u9."]), "passed"),
        (" ".join(["b0 b1 b2 b3 b4 b5 b6."] * 2 + [f"u{i}" for i in range(7)] + ["u7."]), "too_repetitive_trigrams"),
    ],
)
def test_check_quality_threshold_boundaries(text: str, expected: str) -> None:
    assert rp.check_quality(text) == (expected == "passed", expected)


def test_check_quality_ngram_checks_need_enough_words() -> None:
    text = "ab cd. ab cd. ab cd. ab cd. ab cd."
    assert rp.check_quality(text) == (False, "too_few_sentences")  # sentences too short: " ab cd" is 5 chars
    text = "alpha beta gamma. alpha beta gamma. alpha beta gamma. x"  # 10 words, 3 long sentences
    assert len(text.split()) == 10
    assert rp.check_quality(text) == (True, "passed")


# --- contamination -----------------------------------------------------------------------------------------------------


def test_normalize_text_and_ngram_set() -> None:
    assert rp.normalize_text("  Hello\n\tWORLD  x ") == "hello world x"
    assert rp.get_ngram_set("A b C d", n=2) == {"a b", "b c", "c d"}
    assert rp.get_ngram_set("a b", n=13) == set()


def test_check_contamination_with_planted_13gram_overlap() -> None:
    words = [f"w{i}" for i in range(20)]  # 8 13-grams
    doc = " ".join(words)
    doc_ngrams = sorted(rp.get_ngram_set(doc, 13))
    assert len(doc_ngrams) == 8
    benchmarks = {
        "gsm8k_test": set(doc_ngrams[:2]),  # 2/8 = 25% > 10%
        "mmlu_test": set(),  # empty benchmark never flags
        "humaneval": {"totally unrelated " * 13},
    }
    assert rp.check_contamination(doc, benchmarks) == (True, ["gsm8k_test"])
    assert rp.check_contamination(doc, {"b": set(doc_ngrams[:1])}, threshold=0.2) == (False, [])
    assert rp.check_contamination(doc, {"b": set(doc_ngrams[:1])}, threshold=0.1) == (True, ["b"])
    doc22 = " ".join(f"v{i}" for i in range(22))  # 10 13-grams
    grams22 = sorted(rp.get_ngram_set(doc22, 13))
    assert len(grams22) == 10
    assert rp.check_contamination(doc22, {"b": set(grams22[:1])}, threshold=0.1) == (False, [])
    assert rp.check_contamination(doc22, {"b": set(grams22[:2])}, threshold=0.1) == (True, ["b"])
    assert rp.check_contamination("only a few words", benchmarks) == (False, [])
    assert rp.check_contamination(doc.upper().replace(" ", "\n"), benchmarks)[0] is True
    # a different n changes the n-gram set: with n=5 the planted 13-grams do not match
    assert rp.check_contamination(doc, benchmarks, n=5) == (False, [])


# --- instruct rows -----------------------------------------------------------------------------------------------------


def test_instruct_text_is_the_trainers_text() -> None:
    assert rp.instruct_text({"instruction": "a", "input": "b", "output": "c"}) == "a\n\nb\n\nc"
    assert rp.instruct_text({"instruction": " a ", "input": "  ", "output": "c\n"}) == "a\n\nc", "stripped; a blank input is no input"
    assert rp.instruct_text({"instruction": "a", "input": None, "output": "c"}) == "a\n\nc"
    assert rp.instruct_text({"instruction": "a", "output": "c"}) == "a\n\nc"


def test_create_input_inversion_with_and_without_input() -> None:
    ex = {"instruction": "Sort the list", "input": "[3, 1]", "output": "[1, 3]"}
    inv = rp.create_input_inversion(ex)
    assert inv == {
        "instruction": "Given this output, what was the likely instruction or input?\n\nOutput: [1, 3]",
        "input": "",
        "output": "Sort the list\nInput: [3, 1]",
    }
    ex = {"instruction": "Say hi", "input": "", "output": "hi"}
    assert rp.create_input_inversion(ex)["output"] == "Say hi"
    assert rp.create_input_inversion(ex)["instruction"].endswith("Output: hi")


def test_create_input_inversion_noop_cases_including_fixed_guard() -> None:
    assert rp.create_input_inversion({"foo": 1}) == {"foo": 1}
    assert rp.create_input_inversion({"instruction": "x"}) == {"instruction": "x"}
    empty_output = {"instruction": "x", "input": "", "output": ""}
    assert rp.create_input_inversion(empty_output) is empty_output
    # the thesis code inverted rows with an empty instruction when `input` was set; the fixed guard keeps them
    empty_instruction = {"instruction": "", "input": "given", "output": "o"}
    assert rp.create_input_inversion(empty_instruction) is empty_instruction
    empty_output_with_input = {"instruction": "i", "input": "given", "output": ""}
    assert rp.create_input_inversion(empty_output_with_input) is empty_output_with_input


def test_has_required_fields() -> None:
    a: dict[str, Any] = {"instruction": "i", "input": "x", "output": "o"}
    assert rp.has_required_fields(a)
    assert not rp.has_required_fields({"instruction": "  ", "input": "", "output": "o"})
    assert not rp.has_required_fields({"instruction": "i", "input": "", "output": ""})
    assert not rp.has_required_fields({"instruction": None, "input": "", "output": "o"})
    assert not rp.has_required_fields({"output": "o"})
