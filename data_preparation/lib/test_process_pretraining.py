# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.process_pretraining: row-level heuristics, dataset-level steps, and the CLI."""

import json
import sys
from types import ModuleType
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typing import Any

from data_preparation.lib import process_pretraining as pp
from data_preparation.lib.common import list_parquet_files

GOOD = (
    "The quick brown fox jumps over the lazy dog. Then it went home to sleep. It dreamed of chasing rabbits all night."
)


def _repetitive_trigrams() -> str:
    """>20 words; bigram uniqueness 28/39 >= 0.7 but trigram uniqueness 28/38 < 0.8."""
    block = [f"b{i}" for i in range(12)]
    unique = [f"u{i}" for i in range(16)]
    words = block + block + unique
    return ". ".join(" ".join(words[i : i + 12]) for i in range(0, len(words), 12)) + "."


# --- row-level functions --------------------------------------------------------------------------------------------


def test_get_ngrams() -> None:
    assert pp.get_ngrams("a b c d e f", n=5) == ["a b c d e", "b c d e f"]
    assert pp.get_ngrams("a b", n=5) == []
    assert pp.get_ngrams("x  y\tz", n=2) == ["x y", "y z"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("hi", "too_short"),
        ("This is one sentence. Another one here.", "too_few_sentences"),
        ("THIS IS A LOUD SENTENCE HERE. ANOTHER LOUD SENTENCE. AND ONE MORE LOUD ONE.", "too_many_caps"),
        ("@@@@@@@@@@@@ a. ############ b. $$$$$$$$$$$$ c.", "too_few_alphanumeric"),
        ("the cat sat. the cat sat. the cat sat. the cat sat. the cat sat.", "too_repetitive_bigrams"),
        (_repetitive_trigrams(), "too_repetitive_trigrams"),
        (GOOD, "passed"),
    ],
)
def test_check_quality_reasons(text: str, reason: str) -> None:
    passes, got = pp.check_quality(text)
    assert got == reason
    assert passes is (reason == "passed")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # total length: < 10 chars is too_short; exactly 10 goes on to the sentence check
        ("abcdefghi", "too_short"),
        ("abcdefghij", "too_few_sentences"),
        # a sentence counts only when its stripped length is > 10: 3 x 10 chars fail, 3 x 11 chars pass
        ("abcdefghij. abcdefghij. abcdefghij.", "too_few_sentences"),
        ("abcdefghijk. abcdefghijk. abcdefghijk.", "passed"),
        # ALL-CAPS words (len > 1): 3/10 = 30% passes (not > 0.3), 4/10 fails
        ("AA BB CC alpha. beta gamma delta. epsilon zeta eta.", "passed"),
        ("AA BB CC DD. beta gamma delta. epsilon zeta eta.", "too_many_caps"),
        # alphanumeric+space share: 35 of 140 chars = 25% passes, 35 of 141 fails
        ("aa bb cc dd. ee ff gg hh. ii jj kk ll." + "#" * 102, "passed"),
        ("aa bb cc dd. ee ff gg hh. ii jj kk ll." + "#" * 103, "too_few_alphanumeric"),
        # 11 words -> 10 bigrams: 7 unique (0.7) passes, 6 unique (0.6) fails
        ("alpha beta gamma. alpha beta gamma. alpha delta epsilon zeta eta.", "passed"),
        ("alpha beta gamma. alpha beta gamma. alpha beta epsilon zeta eta.", "too_repetitive_bigrams"),
        # 22 words -> 20 trigrams: 16 unique (0.8) passes, 15 unique (0.75) fails; bigram ratio stays >= 0.7
        (" ".join(["b0 b1 b2 b3 b4 b5."] * 2 + [f"u{i}" for i in range(9)] + ["u9."]), "passed"),
        (" ".join(["b0 b1 b2 b3 b4 b5 b6."] * 2 + [f"u{i}" for i in range(7)] + ["u7."]), "too_repetitive_trigrams"),
    ],
)
def test_check_quality_threshold_boundaries(text: str, expected: str) -> None:
    assert pp.check_quality(text) == (expected == "passed", expected)


def test_check_quality_ngram_checks_need_enough_words() -> None:
    # 10 words -> bigram check not applied even though every bigram repeats
    text = "ab cd. ab cd. ab cd. ab cd. ab cd."
    assert pp.check_quality(text) == (False, "too_few_sentences")  # sentences too short: " ab cd" is 5 chars
    text = "alpha beta gamma. alpha beta gamma. alpha beta gamma. x"  # 10 words, 3 long sentences
    assert len(text.split()) == 10
    assert pp.check_quality(text) == (True, "passed")


def test_remove_pii_basic_patterns() -> None:
    text, n = pp.remove_pii("contact a.b@example.com or 192.168.0.1 or call 555-123-4567 now")
    assert text == "contact [EMAIL] or [IP] or call [PHONE] now" and n == 3


@pytest.mark.parametrize(
    ("phone", "masked"),
    [
        ("555-123-4567", True),
        ("555.123.4567", True),
        ("5551234567", True),
        # current behaviour: the leading `\b` cannot match before "(" or "+", so those characters survive the mask
        ("(555)123-4567", "([PHONE]"),
        ("+1-555-123-4567", "+[PHONE]"),
        # and no whitespace is allowed inside the number
        ("(555) 123-4567", False),
        ("555 123 4567", False),
    ],
)
def test_remove_pii_phone_variants(phone: str, masked: bool | str) -> None:
    text, n = pp.remove_pii(f"call {phone} now")
    if masked is True:
        assert text == "call [PHONE] now" and n == 1
    elif masked is False:
        assert text == f"call {phone} now" and n == 0
    else:
        assert text == f"call {masked} now" and n == 1


def test_remove_pii_keys_only_in_key_context() -> None:
    long_token = "A" * 40
    text, n = pp.remove_pii(f"api_key=abcdef123456 and token: XYZ987 plus {long_token}")
    assert text == "[KEY] and [KEY] plus [KEY]" and n == 3
    # no api/key/token context: long alphanumeric strings are left alone
    assert pp.remove_pii(f"no context {long_token}") == (f"no context {long_token}", 0)
    assert pp.remove_pii("my key is abc") == ("my key is abc", 0)


def test_remove_pii_clean_text_untouched() -> None:
    assert pp.remove_pii(GOOD) == (GOOD, 0)


def test_remove_pii_email_and_ip_false_positive_guards() -> None:
    assert pp.remove_pii("reach me at first.last+tag@sub.example.co.uk today") == ("reach me at [EMAIL] today", 1)
    # no domain / no TLD -> not an email
    assert pp.remove_pii("follow @handle now") == ("follow @handle now", 0)
    assert pp.remove_pii("a@b or user@localhost") == ("a@b or user@localhost", 0)
    # three octets or a >3-digit octet are not IPs; a 7-digit local number is not a phone number
    assert pp.remove_pii("version 1.2.3 and 1234.5.6.7") == ("version 1.2.3 and 1234.5.6.7", 0)
    assert pp.remove_pii("ext 555-1234") == ("ext 555-1234", 0)


def test_remove_pii_key_context_is_a_plain_substring_match() -> None:
    """Current (thesis-run) behaviour, pinned: the api/key/token context check is a substring test on the whole
    document, so ordinary prose containing "token" or "key" (even inside "monkey") gets masked."""
    assert pp.remove_pii("The token is valid and the API key management page") == (
        "The [KEY] valid and the API key management page",
        1,
    )
    assert pp.remove_pii("a monkey saw " + "A" * 40) == ("a monkey saw [KEY]", 1)


def test_normalize_text_and_ngram_set() -> None:
    assert pp.normalize_text("  Hello\n\tWORLD  x ") == "hello world x"
    assert pp.get_ngram_set("A b C d", n=2) == {"a b", "b c", "c d"}
    assert pp.get_ngram_set("a b", n=13) == set()


def test_check_contamination_with_planted_13gram_overlap() -> None:
    words = [f"w{i}" for i in range(20)]  # 8 13-grams
    doc = " ".join(words)
    doc_ngrams = sorted(pp.get_ngram_set(doc, 13))
    assert len(doc_ngrams) == 8
    benchmarks = {
        "gsm8k_test": set(doc_ngrams[:2]),  # 2/8 = 25% > 10%
        "mmlu_test": set(),  # empty benchmark never flags
        "humaneval": {"totally unrelated " * 13},
    }
    assert pp.check_contamination(doc, benchmarks) == (True, ["gsm8k_test"])
    # single overlapping 13-gram with threshold 0.2 -> 1/8 = 12.5% stays clean
    assert pp.check_contamination(doc, {"b": set(doc_ngrams[:1])}, threshold=0.2) == (False, [])
    assert pp.check_contamination(doc, {"b": set(doc_ngrams[:1])}, threshold=0.1) == (True, ["b"])
    # exact boundary: overlap == threshold is clean (strict >), one more n-gram flags it
    doc22 = " ".join(f"v{i}" for i in range(22))  # 10 13-grams
    grams22 = sorted(pp.get_ngram_set(doc22, 13))
    assert len(grams22) == 10
    assert pp.check_contamination(doc22, {"b": set(grams22[:1])}, threshold=0.1) == (False, [])
    assert pp.check_contamination(doc22, {"b": set(grams22[:2])}, threshold=0.1) == (True, ["b"])
    # too short for a 13-gram -> never contaminated
    assert pp.check_contamination("only a few words", benchmarks) == (False, [])
    # normalization: case and whitespace differences still match
    assert pp.check_contamination(doc.upper().replace(" ", "\n"), benchmarks)[0] is True


# --- dataset-level steps --------------------------------------------------------------------------------------------


def _ds(hf_datasets: ModuleType, texts: list[str]) -> Any:
    return hf_datasets.Dataset.from_dict({"text": texts, "source": ["s"] * len(texts)})


def test_exact_deduplicate_first_occurrence_wins(hf_datasets: ModuleType) -> None:
    ds = _ds(hf_datasets, ["a", "b", "a", "c", "b", "a"])
    deduped, stats = pp.exact_deduplicate(ds, num_workers=1, desc="t")
    assert deduped["text"] == ["a", "b", "c"]
    assert deduped.column_names == ["text", "source"]
    assert stats == {"original_count": 6, "unique_count": 3, "duplicates_removed": 3, "duplicate_rate": 0.5}


def test_exact_deduplicate_empty(hf_datasets: ModuleType) -> None:
    ds = _ds(hf_datasets, []).select([])
    deduped, stats = pp.exact_deduplicate(ds, num_workers=1, desc="t")
    assert len(deduped) == 0 and stats["duplicate_rate"] == 0


def test_quality_filter_counts_reasons(hf_datasets: ModuleType) -> None:
    ds = _ds(hf_datasets, [GOOD, "hi", "This is one sentence. Another one here.", GOOD + " More text here."])
    filtered, stats = pp.quality_filter(ds, num_workers=1, desc="t")
    assert len(filtered) == 2 and filtered.column_names == ["text", "source"]
    assert stats["rejection_reasons"] == {"too_short": 1, "too_few_sentences": 1}
    assert (stats["original_count"], stats["passed_count"], stats["filtered_count"]) == (4, 2, 2)
    assert stats["filter_rate"] == pytest.approx(0.5)


def test_apply_pii_removal(hf_datasets: ModuleType) -> None:
    ds = _ds(hf_datasets, ["mail a@b.com", "ip 10.0.0.1 and 10.0.0.2", GOOD])
    cleaned, stats = pp.apply_pii_removal(ds, num_workers=1, desc="t")
    assert cleaned["text"] == ["mail [EMAIL]", "ip [IP] and [IP]", GOOD]
    assert cleaned.column_names == ["text", "source"]
    assert stats == {"documents_processed": 3, "pii_instances_removed": 3, "avg_pii_per_document": 1.0}


def test_decontaminate(hf_datasets: ModuleType) -> None:
    words = " ".join(f"w{i}" for i in range(20))
    ds = _ds(hf_datasets, [words, GOOD, words + " tail"])
    grams = pp.get_ngram_set(words, 13)
    clean, stats = pp.decontaminate(ds, {"gsm8k_test": grams, "mmlu_test": set()}, num_workers=1, desc="t")
    assert clean["text"] == [GOOD] and clean.column_names == ["text", "source"]
    assert stats["contaminated_count"] == 2 and stats["contaminated_by_benchmark"] == {"gsm8k_test": 2}


def test_fuzzy_deduplicate_removes_near_duplicates(hf_datasets: ModuleType) -> None:
    pytest.importorskip("datasketch")
    base = " ".join(f"word{i}" for i in range(200))
    near = base.replace("word100", "changed")  # 5 of 196 5-grams differ -> Jaccard ~0.95
    other = " ".join(f"other{i}" for i in range(200))
    # shares its first 100 words with `base`: 96 common 5-grams of 296 -> Jaccard ~0.32, well below 0.8 -> kept
    partial = " ".join(f"word{i}" for i in range(100)) + " " + " ".join(f"new{i}" for i in range(100))
    ds = _ds(hf_datasets, [base, near, other, base, partial])
    deduped, stats = pp.fuzzy_deduplicate(ds, threshold=0.8, num_perm=64, num_workers=1, desc="t")
    assert deduped["text"] == [base, other, partial]
    assert deduped.column_names == ["text", "source"]
    assert stats["near_duplicates_removed"] == 2 and stats["threshold"] == 0.8 and stats["num_perm"] == 64
    assert stats["near_duplicate_rate"] == pytest.approx(0.4)


def test_dataset_steps_on_empty_dataset(hf_datasets: ModuleType) -> None:
    """`map` on an empty dataset adds no columns; every step must still return the input schema and zero rates."""
    empty = _ds(hf_datasets, [])
    filtered, qstats = pp.quality_filter(empty, 1, "t")
    cleaned, pstats = pp.apply_pii_removal(empty, 1, "t")
    clean, dstats = pp.decontaminate(empty, {"gsm8k_test": {"x"}}, 1, "t")
    for out in (filtered, cleaned, clean):
        assert len(out) == 0 and out.column_names == ["text", "source"]
    assert qstats["filter_rate"] == 0 and qstats["rejection_reasons"] == {}
    assert pstats["avg_pii_per_document"] == 0
    assert dstats["contamination_rate"] == 0 and dstats["contaminated_by_benchmark"] == {}


def test_fuzzy_deduplicate_without_datasketch_raises(hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasketch", None)  # makes `import datasketch` raise ImportError
    with pytest.raises(ImportError, match="skip_fuzzy_dedup"):
        pp.fuzzy_deduplicate(_ds(hf_datasets, ["a b c d e f"]), 0.8, 16, 1, "t")


def test_load_benchmark_ngrams_with_stubbed_hub(
    hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    words = " ".join(f"q{i}" for i in range(15))
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_load_dataset(name: str, config: str | None = None, split: str | None = None) -> Any:
        calls.append((name, config, split))
        if name == "hellaswag":
            raise OSError("offline")
        return hf_datasets.Dataset.from_dict({"question": [words], "choices": [["c1 " * 20, "c2"]], "answer": [3]})

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    grams = pp.load_benchmark_ngrams(n=13)
    assert set(grams) == set(pp.BENCHMARK_DATASETS)
    assert grams["hellaswag_test"] == set()
    assert ("gsm8k", "main", "test") in calls and ("openai_humaneval", None, "test") in calls
    expected = pp.get_ngram_set(words + " " + "c1 " * 20 + " c2", 13)
    assert grams["gsm8k_test"] == expected and len(expected) > 0
    assert "Error loading hellaswag_test" in capsys.readouterr().out


def test_prepare_dataset_adds_source_and_char_estimate(hf_datasets: ModuleType) -> None:
    ds = hf_datasets.Dataset.from_dict({"text": ["a" * 8, "b" * 11]})
    out = pp.prepare_dataset(ds, "src", 1, tokenizer_path=None, max_seq_length=None)
    assert out["source"] == ["src", "src"] and out["estimated_tokens"] == [2, 2]
    # existing columns are kept as is
    again = pp.prepare_dataset(out, "other", 1, None, None)
    assert again["source"] == ["src", "src"]


def test_prepare_dataset_with_tokenizer_truncates(hf_datasets: ModuleType, tiny_tokenizer_path: Path) -> None:
    ds = hf_datasets.Dataset.from_dict({"text": [" ".join(f"tok_{i}" for i in range(10))]})
    out = pp.prepare_dataset(ds, "src", 1, str(tiny_tokenizer_path), max_seq_length=4)
    assert out["estimated_tokens"] == [4]
    assert out["text"] == ["tok_0 tok_1 tok_2 tok_3"]
    # without a limit the real token count replaces the char/4 estimate and the text is untouched
    full = pp.prepare_dataset(ds, "src", 1, str(tiny_tokenizer_path), max_seq_length=None)
    assert full["estimated_tokens"] == [10] and full["text"] == ds["text"]


def test_column_and_drop_columns(hf_datasets: ModuleType) -> None:
    ds = _ds(hf_datasets, ["a", "b"])
    assert pp.column(ds, "text") == ["a", "b"]
    assert pp.column(ds, "missing") == []
    dropped = pp.drop_columns(ds, ["source", "missing"])
    assert dropped.column_names == ["text"]
    assert pp.drop_columns(ds, []).column_names == ["text", "source"]


def test_load_filtered_dataset(tmp_path: Path, hf_datasets: ModuleType) -> None:
    assert pp.load_filtered_dataset(tmp_path / "missing") is None
    (tmp_path / "empty").mkdir()
    assert pp.load_filtered_dataset(tmp_path / "empty") is None
    _write_filtered(tmp_path, "src", ["a", "b", "c", "d", "e"], shard_size=2)
    ds = pp.load_filtered_dataset(tmp_path / "pretraining" / "filtered" / "src")
    assert ds is not None and ds["text"] == ["a", "b", "c", "d", "e"]
    assert ds.column_names == ["text", "source", "original_length"]


def test_save_dataset_and_verification_samples(tmp_path: Path, hf_datasets: ModuleType) -> None:
    ds = hf_datasets.Dataset.from_dict(
        {
            "text": ["x" * 600, "y", "z", "w"],
            "source": ["s"] * 4,
            "estimated_tokens": [150, 0, 0, 0],
            "extra": [1, 2, 3, 4],
        }
    )
    info = pp.save_dataset(ds, "s", tmp_path / "merged", shard_size=3)
    assert info == {"dataset_name": "s", "tokens": 150, "documents": 4, "num_shards": 2}
    table = pq.read_table(list_parquet_files(tmp_path / "merged" / "s", "data")[0])
    assert table.column_names == ["text", "source", "estimated_tokens"]  # `extra` is dropped
    (tmp_path / "merged" / "no_files").mkdir()
    (tmp_path / "merged" / "stray.txt").write_text("")
    pp.write_verification_samples(tmp_path / "merged", tmp_path / "samples.txt")
    samples = (tmp_path / "samples.txt").read_text()
    assert "no_files" not in samples and samples.count("--- Sample") == 3
    assert "Estimated tokens: 150" in samples and "x" * 500 + "..." in samples and "x" * 501 not in samples


# --- CLI ------------------------------------------------------------------------------------------------------------


def _write_filtered(root: Path, name: str, texts: list[str], shard_size: int = 3) -> None:
    d = root / "pretraining" / "filtered" / name
    d.mkdir(parents=True)
    for i in range(0, len(texts), shard_size):
        chunk = texts[i : i + shard_size]
        table = pa.table({"text": chunk, "source": [name] * len(chunk), "original_length": [len(t) for t in chunk]})
        pq.write_table(table, d / f"data-{i // shard_size:05d}.parquet")


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["process_pretraining", *argv])
    pp.main()


def test_parser_defaults_and_max_seq_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    args = pp.build_parser().parse_args([])
    assert args.datasets is None and args.shard_size == 10000 and args.fuzzy_threshold == 0.8
    assert args.minhash_num_perm == 256 and not args.dry_run
    with pytest.raises(SystemExit, match="--max_seq_length requires --tokenizer_path"):
        _run_main(monkeypatch, ["--max_seq_length", "10"])


def test_main_dry_run_prints_plan_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_main(monkeypatch, ["--dataset_dir", str(tmp_path), "--dry_run", "--skip_fuzzy_dedup"])
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out and "fuzzy dedup: skipped" in out and "exact dedup: enabled" in out
    assert not (tmp_path / "pretraining" / "processed").exists()


def test_main_end_to_end_without_fuzzy_and_decontamination(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    pii = GOOD + " Write to me at someone@example.org for details about this."
    texts = [GOOD, GOOD, "hi", pii, GOOD + " Extra sentence number four here.", GOOD + " Fifth one."]
    _write_filtered(tmp_path, "alpha", texts, shard_size=4)
    _write_filtered(tmp_path, "beta", [GOOD + " One.", GOOD + " Two.", GOOD + " Three."])

    _run_main(
        monkeypatch,
        [
            "--dataset_dir", str(tmp_path), "--datasets", "alpha", "beta", "missing_source",
            "--skip_fuzzy_dedup", "--skip_decontamination", "--num_workers", "1", "--shard_size", "2",
        ],
    )  # fmt: skip

    processed = tmp_path / "pretraining" / "processed"
    merged = processed / "merged"
    alpha_files = list_parquet_files(merged / "alpha", "data")
    assert [f.name for f in alpha_files] == ["data-00000.parquet", "data-00001.parquet"]
    alpha = pa.concat_tables([pq.read_table(f) for f in alpha_files])
    assert alpha.column_names == ["text", "source", "estimated_tokens"]
    assert alpha.schema.field("estimated_tokens").type == pa.int64()
    rows = alpha.to_pylist()
    assert [r["text"] for r in rows] == [GOOD, pii.replace("someone@example.org", "[EMAIL]"), texts[4], texts[5]]
    assert all(r["source"] == "alpha" for r in rows)
    # estimated_tokens is computed on the loaded text, i.e. before PII masking shortens it
    assert [r["estimated_tokens"] for r in rows] == [len(t) // 4 for t in (GOOD, pii, texts[4], texts[5])]
    assert not (merged / "missing_source").exists()

    stats = json.loads((processed / "preprocessing_stats.json").read_text())
    cfg = stats["preprocessing_config"]
    assert cfg["exact_dedup"] and cfg["quality_filter"] and cfg["pii_removal"]
    assert not cfg["fuzzy_dedup"] and not cfg["decontamination"]
    assert stats["random_seed"] == 42 and "timestamp" in stats
    assert set(stats["statistics"]) == {"alpha", "beta"}
    a = stats["statistics"]["alpha"]
    assert list(a) == ["exact_dedup", "quality_filter", "pii_removal", "save"]
    assert a["exact_dedup"]["duplicates_removed"] == 1
    assert a["quality_filter"]["rejection_reasons"] == {"too_short": 1}
    assert a["pii_removal"]["pii_instances_removed"] == 1
    assert a["save"] == {"dataset_name": "alpha", "tokens": sum(r["estimated_tokens"] for r in rows), "documents": 4,
                         "num_shards": 2}  # fmt: skip
    assert stats["statistics"]["beta"]["save"]["documents"] == 3

    samples = (processed / "verification_samples.txt").read_text()
    assert samples.startswith("Pretraining dataset verification samples\n")
    assert samples.index("\nalpha\n") < samples.index("\nbeta\n")
    # only the first shard (2 rows here) is sampled
    assert samples.count("--- Sample 1 ---") == 2 and samples.count("--- Sample 2 ---") == 2
    assert "--- Sample 3 ---" not in samples
    assert "Source: alpha" in samples and "[EMAIL]" in samples


def test_main_decontamination_uses_benchmark_ngrams(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaked = " ".join(f"q{i}" for i in range(20)) + ". Some more words for sentence two. And a third sentence here."
    _write_filtered(tmp_path, "src", [GOOD, leaked])
    monkeypatch.setattr(pp, "load_benchmark_ngrams", lambda: {"gsm8k_test": pp.get_ngram_set(leaked, 13)})
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--datasets", "src", "--skip_fuzzy_dedup", "--skip_exact_dedup",
         "--skip_quality_filter", "--skip_pii_removal", "--num_workers", "1"],
    )  # fmt: skip
    table = pq.read_table(list_parquet_files(tmp_path / "pretraining" / "processed" / "merged" / "src", "data")[0])
    assert table["text"].to_pylist() == [GOOD]
    stats = json.loads((tmp_path / "pretraining" / "processed" / "preprocessing_stats.json").read_text())
    assert list(stats["statistics"]["src"]) == ["decontamination", "save"]
    assert stats["statistics"]["src"]["decontamination"]["contaminated_by_benchmark"] == {"gsm8k_test": 1}


def test_main_all_steps_skipped_keeps_everything(
    tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_filtered(tmp_path, "src", ["a", "a", "hi"])
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--datasets", "src", "--skip_exact_dedup", "--skip_fuzzy_dedup",
         "--skip_quality_filter", "--skip_pii_removal", "--skip_decontamination", "--num_workers", "1"],
    )  # fmt: skip
    table = pq.read_table(list_parquet_files(tmp_path / "pretraining" / "processed" / "merged" / "src", "data")[0])
    assert table["text"].to_pylist() == ["a", "a", "hi"]
    assert table["estimated_tokens"].to_pylist() == [0, 0, 0]


def test_main_with_fuzzy_dedup(tmp_path: Path, hf_datasets: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("datasketch")
    base = " ".join(f"word{i}" for i in range(200))
    near = base.split()
    near[7] = "x"
    _write_filtered(tmp_path, "src", [base, " ".join(near), "different " * 50])
    _run_main(
        monkeypatch,
        ["--dataset_dir", str(tmp_path), "--datasets", "src", "--skip_exact_dedup", "--skip_quality_filter",
         "--skip_pii_removal", "--skip_decontamination", "--num_workers", "1", "--minhash_num_perm", "32"],
    )  # fmt: skip
    table = pq.read_table(list_parquet_files(tmp_path / "pretraining" / "processed" / "merged" / "src", "data")[0])
    assert table.num_rows == 2 and table["text"].to_pylist()[0] == base
    stats = json.loads((tmp_path / "pretraining" / "processed" / "preprocessing_stats.json").read_text())
    assert stats["statistics"]["src"]["fuzzy_dedup"]["near_duplicates_removed"] == 1
    assert stats["preprocessing_config"]["minhash_num_perm"] == 32


def test_sources_list_matches_readme_budgets() -> None:
    assert len(pp.SOURCES) == 19 and len(set(pp.SOURCES)) == 19
    assert sum(s.startswith("github_code_clean_") for s in pp.SOURCES) == 10
