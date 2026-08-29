# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Deduplicate, filter, scrub and decontaminate the filtered sources into
``dataset/pretraining/processed/merged/<source>/data-*.parquet`` (the directories the training config points at).

Pipeline per source, in order (each step can be skipped with ``--skip_*``):

1. exact deduplication (MD5 of the text)
2. fuzzy deduplication (MinHash over 5-grams + LSH, Jaccard threshold 0.8, 256 permutations; needs ``datasketch``)
3. quality filtering (>= 3 sentences, <= 30% ALL-CAPS words, >= 25% alphanumeric, <= 30% duplicate 2-grams,
   <= 20% duplicate 3-grams)
4. PII masking (emails, IPs, phone numbers, API keys)
5. benchmark decontamination (documents with > 10% 13-gram overlap with a benchmark test set are dropped)

Output columns: ``text``, ``source``, ``estimated_tokens``. Also writes ``preprocessing_stats.json`` and
``verification_samples.txt`` under ``dataset/pretraining/processed/``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_preparation.lib.common import (
    RANDOM_SEED,
    add_common_args,
    configure_hf_cache,
    estimate_tokens,
    iter_dataset_tables,
    list_parquet_files,
    md5_hex,
    print_header,
    write_parquet_shards,
)

if TYPE_CHECKING:  # `datasets` is imported lazily at runtime (after the HF cache is configured)
    from datasets import Dataset

SOURCES = [
    "algebraic_stack",
    "arxiv",
    "books_gutenberg",
    "fineweb_edu",
    "github_code_clean_cpp",
    "github_code_clean_go",
    "github_code_clean_html",
    "github_code_clean_java",
    "github_code_clean_javascript",
    "github_code_clean_python",
    "github_code_clean_rust",
    "github_code_clean_shell",
    "github_code_clean_sql",
    "github_code_clean_typescript",
    "gsm8k",
    "openwebmath",
    "peso",
    "tinygsm",
    "wikipedia",
]

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b")
API_KEY_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9]{32,}\b"),
    re.compile(r"api[_\-]?key[_\-:=\s]+[A-Za-z0-9]+", re.IGNORECASE),
    re.compile(r"token[_\-:=\s]+[A-Za-z0-9]+", re.IGNORECASE),
]

# (dataset, config, split) of the benchmark test sets used for decontamination.
BENCHMARK_DATASETS = {
    "gsm8k_test": ("gsm8k", "main", "test"),
    "math_test": ("hendrycks_math", "all", "test"),
    "humaneval": ("openai_humaneval", None, "test"),
    "mbpp_test": ("mbpp", None, "test"),
    "arc_challenge_test": ("ai2_arc", "ARC-Challenge", "test"),
    "hellaswag_test": ("hellaswag", None, "test"),
    "mmlu_test": ("cais/mmlu", "all", "test"),
    "winogrande_test": ("winogrande", "winogrande_xl", "test"),
}


# --- row-level functions (pure, reusable) ---------------------------------------------------------------------------


def get_ngrams(text: str, n: int = 5) -> list[str]:
    """Word n-grams of ``text`` (for MinHash)."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def check_quality(text: str) -> tuple[bool, str]:
    """Heuristic quality check; returns ``(passes, reason)``."""
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
    if len(words) > 10:
        bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        if len(set(bigrams)) / len(bigrams) < 0.7:
            return False, "too_repetitive_bigrams"
    if len(words) > 20:
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) / len(trigrams) < 0.8:
            return False, "too_repetitive_trigrams"
    return True, "passed"


def remove_pii(text: str) -> tuple[str, int]:
    """Mask emails, IPs, phone numbers and (in key/token context) API keys; returns ``(text, num_replaced)``."""
    num_removed = 0
    for pattern, replacement in ((EMAIL_PATTERN, "[EMAIL]"), (IP_PATTERN, "[IP]"), (PHONE_PATTERN, "[PHONE]")):
        text, n = pattern.subn(replacement, text)
        num_removed += n
    for pattern in API_KEY_PATTERNS:
        lowered = text.lower()
        if "api" in lowered or "key" in lowered or "token" in lowered:
            text, n = pattern.subn("[KEY]", text)
            num_removed += n
    return text, num_removed


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace (for contamination checks)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def get_ngram_set(text: str, n: int = 13) -> set[str]:
    """Set of normalized word n-grams of ``text``."""
    words = normalize_text(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_contamination(
    text: str, benchmark_ngrams: dict[str, set[str]], n: int = 13, threshold: float = 0.1
) -> tuple[bool, list[str]]:
    """A document is contaminated if more than ``threshold`` of its n-grams occur in any benchmark test set."""
    doc_ngrams = get_ngram_set(text, n)
    if not doc_ngrams:
        return False, []
    contaminated = []
    for name, test_ngrams in benchmark_ngrams.items():
        if test_ngrams and len(doc_ngrams & test_ngrams) / len(doc_ngrams) > threshold:
            contaminated.append(name)
    return len(contaminated) > 0, contaminated


# --- dataset-level steps --------------------------------------------------------------------------------------------


def column(dataset: "Dataset", name: str) -> list[Any]:
    """Values of a column, or ``[]`` when the dataset is empty (``map`` adds no columns to an empty dataset)."""
    return list(dataset[name]) if name in dataset.column_names else []


def drop_columns(dataset: "Dataset", names: list[str]) -> "Dataset":
    """Remove the temporary columns that exist."""
    return dataset.remove_columns([n for n in names if n in dataset.column_names])


def exact_deduplicate(dataset: "Dataset", num_workers: int, desc: str) -> "tuple[Dataset, dict[str, Any]]":
    """Drop rows whose text MD5 was already seen (first occurrence wins)."""
    print("  Running exact deduplication...")
    hashed = dataset.map(lambda ex: {"md5_hash": md5_hex(ex["text"])}, num_proc=num_workers, desc=f"  {desc} - hashes")
    seen: set[str] = set()
    unique_indices = []
    for idx, h in enumerate(column(hashed, "md5_hash")):
        if h not in seen:
            seen.add(h)
            unique_indices.append(idx)
    deduped = dataset.select(unique_indices)
    stats: dict[str, Any] = {
        "original_count": len(dataset),
        "unique_count": len(deduped),
        "duplicates_removed": len(dataset) - len(deduped),
        "duplicate_rate": 1 - len(deduped) / len(dataset) if len(dataset) > 0 else 0,
    }
    print(f"  Removed {stats['duplicates_removed']:,} duplicates ({100 * stats['duplicate_rate']:.1f}%)")
    return deduped, stats


def fuzzy_deduplicate(
    dataset: "Dataset", threshold: float, num_perm: int, num_workers: int, desc: str
) -> "tuple[Dataset, dict[str, Any]]":
    """MinHash + LSH near-duplicate removal over 5-gram sets."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError("Fuzzy deduplication needs `pip install datasketch` (or pass --skip_fuzzy_dedup)") from exc
    from tqdm.auto import tqdm

    print(f"  Running fuzzy deduplication (threshold={threshold}, num_perm={num_perm})...")
    start = time.time()

    def compute_minhash(example: dict[str, Any]) -> dict[str, bytes]:
        m = MinHash(num_perm=num_perm)
        for ngram in get_ngrams(example["text"], n=5):
            m.update(ngram.encode("utf-8"))
        return {"minhash_bytes": pickle.dumps(m)}

    with_minhash = dataset.map(compute_minhash, num_proc=num_workers, desc=f"  {desc} - MinHash")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = []
    for idx, example in enumerate(tqdm(with_minhash, desc=f"  {desc} - LSH")):
        # `datasets` declares __iter__ as yielding dict | list, so pyright cannot index the row by name
        m = pickle.loads(example["minhash_bytes"])  # pyright: ignore[reportCallIssue, reportArgumentType]
        if not lsh.query(m):
            lsh.insert(f"doc_{idx}", m)
            unique_indices.append(idx)
    deduped = dataset.select(unique_indices)
    stats: dict[str, Any] = {
        "original_count": len(dataset),
        "unique_count": len(deduped),
        "near_duplicates_removed": len(dataset) - len(deduped),
        "near_duplicate_rate": 1 - len(deduped) / len(dataset) if len(dataset) > 0 else 0,
        "threshold": threshold,
        "num_perm": num_perm,
        "time_seconds": time.time() - start,
    }
    print(f"  Removed {stats['near_duplicates_removed']:,} near-duplicates ({100 * stats['near_duplicate_rate']:.1f}%)")
    return deduped, stats


def quality_filter(dataset: "Dataset", num_workers: int, desc: str) -> "tuple[Dataset, dict[str, Any]]":
    """Keep only rows passing :func:`check_quality`."""
    print("  Running quality filtering...")

    def add_quality_check(example: dict[str, Any]) -> dict[str, Any]:
        passes, reason = check_quality(example["text"])
        return {"passes_quality": passes, "rejection_reason": "" if passes else reason}

    checked = dataset.map(add_quality_check, num_proc=num_workers, desc=f"  {desc} - checking")
    rejection_reasons = Counter(r for r in column(checked, "rejection_reason") if r)
    filtered = checked.filter(lambda ex: ex["passes_quality"], desc=f"  {desc} - filtering")
    filtered = drop_columns(filtered, ["passes_quality", "rejection_reason"])
    stats: dict[str, Any] = {
        "original_count": len(dataset),
        "passed_count": len(filtered),
        "filtered_count": len(dataset) - len(filtered),
        "filter_rate": 1 - len(filtered) / len(dataset) if len(dataset) > 0 else 0,
        "rejection_reasons": dict(rejection_reasons),
    }
    print(f"  Filtered {stats['filtered_count']:,} low-quality documents ({100 * stats['filter_rate']:.1f}%)")
    for reason, count in rejection_reasons.most_common(5):
        print(f"    - {reason}: {count:,}")
    return filtered, stats


def apply_pii_removal(dataset: "Dataset", num_workers: int, desc: str) -> "tuple[Dataset, dict[str, Any]]":
    """Apply :func:`remove_pii` to every row."""
    print("  Running PII removal...")

    def remove_pii_from_example(example: dict[str, Any]) -> dict[str, Any]:
        text, num_removed = remove_pii(example["text"])
        return {"text": text, "pii_count": num_removed}

    with_counts = dataset.map(remove_pii_from_example, num_proc=num_workers, desc=f"  {desc}")
    total_removed = sum(column(with_counts, "pii_count"))
    cleaned = drop_columns(with_counts, ["pii_count"])
    stats: dict[str, Any] = {
        "documents_processed": len(dataset),
        "pii_instances_removed": total_removed,
        "avg_pii_per_document": total_removed / len(dataset) if len(dataset) > 0 else 0,
    }
    print(f"  Removed {total_removed:,} PII instances ({stats['avg_pii_per_document']:.2f} per document)")
    return cleaned, stats


def load_benchmark_ngrams(n: int = 13) -> dict[str, set[str]]:
    """Download every benchmark test set and collect its normalized n-grams (a failing benchmark yields an empty set)."""
    from datasets import load_dataset

    print("\nLoading benchmark test sets for decontamination...")
    all_ngrams: dict[str, set[str]] = {}
    for name, (dataset_name, config, split) in BENCHMARK_DATASETS.items():
        try:
            print(f"  Loading {name}...")
            ds = load_dataset(dataset_name, config, split=split) if config else load_dataset(dataset_name, split=split)
            ngrams: set[str] = set()
            for example in ds:
                parts = []
                for value in example.values():
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, list):
                        parts.extend(str(v) for v in value if isinstance(v, str))
                ngrams.update(get_ngram_set(" ".join(parts), n))
            all_ngrams[name] = ngrams
            print(f"    {len(ngrams):,} {n}-grams from {len(ds):,} examples")
        except Exception as exc:  # a missing benchmark must not abort processing
            print(f"    Error loading {name}: {exc}")
            all_ngrams[name] = set()
    print(f"  Total: {sum(len(s) for s in all_ngrams.values()):,} unique {n}-grams")
    return all_ngrams


def decontaminate(
    dataset: "Dataset", benchmark_ngrams: dict[str, set[str]], num_workers: int, desc: str
) -> "tuple[Dataset, dict[str, Any]]":
    """Drop rows flagged by :func:`check_contamination`."""
    print("  Running benchmark decontamination...")

    def check(example: dict[str, Any]) -> dict[str, Any]:
        is_contaminated, benchmarks = check_contamination(example["text"], benchmark_ngrams)
        return {"is_clean": not is_contaminated, "contaminated_benchmarks": benchmarks}

    checked = dataset.map(check, num_proc=num_workers, desc=f"  {desc} - checking")
    contaminated_counts = Counter(b for benches in column(checked, "contaminated_benchmarks") for b in benches)
    clean = checked.filter(lambda ex: ex["is_clean"], desc=f"  {desc} - filtering")
    clean = drop_columns(clean, ["is_clean", "contaminated_benchmarks"])
    stats: dict[str, Any] = {
        "original_count": len(dataset),
        "clean_count": len(clean),
        "contaminated_count": len(dataset) - len(clean),
        "contamination_rate": 1 - len(clean) / len(dataset) if len(dataset) > 0 else 0,
        "contaminated_by_benchmark": dict(contaminated_counts),
    }
    print(
        f"  Removed {stats['contaminated_count']:,} contaminated documents ({100 * stats['contamination_rate']:.2f}%)"
    )
    for bench, count in contaminated_counts.most_common():
        print(f"    - {bench}: {count:,}")
    return clean, stats


# --- loading / saving -----------------------------------------------------------------------------------------------


def load_filtered_dataset(dataset_dir: Path) -> "Dataset | None":
    """Load ``data-*.parquet`` of one filtered source as a ``datasets.Dataset`` (None if absent)."""
    from datasets import Dataset, concatenate_datasets

    parquet_files = list_parquet_files(dataset_dir, "data")
    if not parquet_files:
        return None
    return concatenate_datasets([Dataset.from_parquet(str(f)) for f in parquet_files])


def prepare_dataset(
    dataset: "Dataset", name: str, num_workers: int, tokenizer_path: str | None, max_seq_length: int | None
) -> "Dataset":
    """Ensure ``source`` and ``estimated_tokens`` columns exist (char/4, or real counts with ``tokenizer_path``)."""
    if "source" not in dataset.column_names:
        dataset = dataset.add_column("source", [name] * len(dataset))
    if "estimated_tokens" in dataset.column_names:
        return dataset
    if tokenizer_path is None:
        print("    Using char/4 token estimation (pass --tokenizer_path for exact counts)")
        return dataset.map(
            lambda ex: {"estimated_tokens": estimate_tokens(ex["text"])}, num_proc=num_workers, desc="    Estimating"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(
        f"    Using tokenizer {tokenizer_path}" + (f", truncating to {max_seq_length} tokens" if max_seq_length else "")
    )

    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        text = example["text"]
        if max_seq_length and len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        return {"text": text, "estimated_tokens": len(tokens)}

    return dataset.map(tokenize, num_proc=num_workers, desc="    Tokenizing")


def save_dataset(dataset: "Dataset", name: str, output_dir: Path, shard_size: int) -> dict[str, Any]:
    """Write ``text``/``source``/``estimated_tokens`` shards to ``output_dir/<name>``."""
    dataset_dir = output_dir / name
    total_tokens = sum(dataset["estimated_tokens"])
    print(f"\n  Saving {name}: {len(dataset):,} documents, {total_tokens / 1e9:.2f}B tokens")
    tables = iter_dataset_tables(dataset.select_columns(["text", "source", "estimated_tokens"]))
    num_shards = write_parquet_shards(tables, dataset_dir, shard_size)
    print(f"    Saved to {dataset_dir} ({num_shards} shards)")
    return {"dataset_name": name, "tokens": total_tokens, "documents": len(dataset), "num_shards": num_shards}


def write_verification_samples(merged_dir: Path, path: Path) -> None:
    """Write the first three documents of every processed source to a text file for eyeballing."""
    import pyarrow.parquet as pq

    with path.open("w", encoding="utf-8") as f:
        f.write("Pretraining dataset verification samples\n")
        for source_dir in sorted(d for d in merged_dir.iterdir() if d.is_dir()):
            files = list_parquet_files(source_dir, "data")
            if not files:
                continue
            f.write(f"\n{'=' * 80}\n{source_dir.name}\n{'=' * 80}\n\n")
            for i, row in enumerate(pq.read_table(files[0]).slice(0, 3).to_pylist(), 1):
                f.write(f"--- Sample {i} ---\nSource: {row['source']}\nEstimated tokens: {row['estimated_tokens']:,}\n")
                f.write(f"Text (first 500 chars):\n{row['text'][:500]}...\n\n")


# --- CLI ------------------------------------------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register this command's options on ``parser`` (used by ``prepare.py`` and ``build_parser``)."""
    add_common_args(parser)
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None, help=f"Source names to process (default: {' '.join(SOURCES)})"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None, help="Tokenizer for exact token counts (default: char/4 estimate)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Truncate texts to this many tokens (requires --tokenizer_path)",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Worker processes for datasets.map (default: 4)")
    parser.add_argument("--skip_exact_dedup", action="store_true", help="Skip exact deduplication")
    parser.add_argument("--skip_fuzzy_dedup", action="store_true", help="Skip MinHash/LSH fuzzy deduplication")
    parser.add_argument("--skip_quality_filter", action="store_true", help="Skip quality filtering")
    parser.add_argument("--skip_pii_removal", action="store_true", help="Skip PII masking")
    parser.add_argument("--skip_decontamination", action="store_true", help="Skip benchmark decontamination")
    parser.add_argument("--fuzzy_threshold", type=float, default=0.8, help="Jaccard threshold (default: 0.8)")
    parser.add_argument("--minhash_num_perm", type=int, default=256, help="MinHash permutations (default: 256)")
    parser.add_argument("--shard_size", type=int, default=10000, help="Rows per output parquet file (default: 10000)")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit")


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for this command."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute the command with parsed ``args``."""
    if args.max_seq_length and not args.tokenizer_path:
        raise SystemExit("--max_seq_length requires --tokenizer_path")
    configure_hf_cache(args.cache_dir)

    input_dir = args.dataset_dir / "pretraining" / "filtered"
    output_dir = args.dataset_dir / "pretraining" / "processed"
    merged_dir = output_dir / "merged"
    sources = args.datasets or SOURCES

    print_header("Pretraining dataset processing plan")
    print(f"Input: {input_dir}\nOutput: {merged_dir}\nSources: {', '.join(sources)}")
    steps = {
        "exact dedup": not args.skip_exact_dedup,
        "fuzzy dedup": not args.skip_fuzzy_dedup,
        "quality filter": not args.skip_quality_filter,
        "PII removal": not args.skip_pii_removal,
        "decontamination": not args.skip_decontamination,
    }
    for i, (step, enabled) in enumerate(steps.items(), 1):
        print(f"  {i}. {step}: {'enabled' if enabled else 'skipped'}")
    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without processing.")
        return

    benchmark_ngrams = load_benchmark_ngrams() if steps["decontamination"] else {}

    all_stats: dict[str, dict[str, Any]] = {}
    for name in sources:
        print_header(f"Processing {name}")
        dataset = load_filtered_dataset(input_dir / name)
        if dataset is None:
            print("  Dataset not found, skipping")
            continue
        print(f"  Loaded {len(dataset):,} documents")
        dataset = prepare_dataset(dataset, name, args.num_workers, args.tokenizer_path, args.max_seq_length)

        dataset_stats: dict[str, dict[str, Any]] = {}
        if steps["exact dedup"]:
            dataset, dataset_stats["exact_dedup"] = exact_deduplicate(dataset, args.num_workers, f"{name} exact")
        if steps["fuzzy dedup"]:
            dataset, dataset_stats["fuzzy_dedup"] = fuzzy_deduplicate(
                dataset, args.fuzzy_threshold, args.minhash_num_perm, args.num_workers, f"{name} fuzzy"
            )
        if steps["quality filter"]:
            dataset, dataset_stats["quality_filter"] = quality_filter(dataset, args.num_workers, f"{name} quality")
        if steps["PII removal"]:
            dataset, dataset_stats["pii_removal"] = apply_pii_removal(dataset, args.num_workers, f"{name} PII")
        if steps["decontamination"]:
            dataset, dataset_stats["decontamination"] = decontaminate(
                dataset, benchmark_ngrams, args.num_workers, f"{name} decontam"
            )
        print(f"  Final: {len(dataset):,} documents")
        dataset_stats["save"] = save_dataset(dataset, name, merged_dir, args.shard_size)
        all_stats[name] = dataset_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "preprocessing_stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "preprocessing_config": {
                    "exact_dedup": steps["exact dedup"],
                    "fuzzy_dedup": steps["fuzzy dedup"],
                    "fuzzy_threshold": args.fuzzy_threshold,
                    "minhash_num_perm": args.minhash_num_perm,
                    "quality_filter": steps["quality filter"],
                    "pii_removal": steps["PII removal"],
                    "decontamination": steps["decontamination"],
                },
                "statistics": all_stats,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "random_seed": RANDOM_SEED,
            },
            indent=2,
        )
    )
    verification_file = output_dir / "verification_samples.txt"
    if merged_dir.is_dir():
        write_verification_samples(merged_dir, verification_file)
    print_header("Processing complete")
    print(f"Output: {merged_dir}\nStatistics: {stats_file}\nVerification: {verification_file}")


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv``) and run."""
    run(build_parser().parse_args(argv))
