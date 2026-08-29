# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Download the raw pretraining sources to ``dataset/pretraining/raw/<source>/shard-*.parquet``.

No filtering happens here (see ``filter_pretraining``); rows are stored with their original HuggingFace columns.
Sample targets are the 300M-model budgets used for the thesis run. Sources are loaded with ``train[:N]`` split
slicing except GSM8K (question/answer concatenated and repeated to the target) and the GitHub code sources
(streamed from ``codeparrot/github-code-clean`` and filtered by language).
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data_preparation.common import (
    add_common_args,
    configure_hf_cache,
    iter_dataset_tables,
    print_header,
    write_dict_rows,
    write_parquet_shards,
)

if TYPE_CHECKING:  # `datasets` is imported lazily at runtime (after the HF cache is configured)
    from datasets import Dataset

SHARD_SIZE = 100_000
GITHUB_CODE_DATASET = "codeparrot/github-code-clean"

DATASETS: list[dict[str, Any]] = [
    {
        "name": "fineweb_edu",
        "hf_dataset": "HuggingFaceFW/fineweb-edu",
        "target_samples": 9_000_000,  # ~18B tokens @ 2000 tok/sample
        "load_kwargs": {"name": "CC-MAIN-2013-20"},
    },
    {
        "name": "wikipedia",
        "hf_dataset": "wikipedia",
        "target_samples": 1_700_000,  # ~2.5B tokens @ 1500 tok/sample
        "load_kwargs": {"name": "20220301.en", "trust_remote_code": True},
    },
    {
        "name": "books_gutenberg",
        "hf_dataset": "sedthh/gutenberg_english",
        "target_samples": 600_000,  # ~1.8B tokens @ 3000 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "peso",
        "hf_dataset": "nampdn-ai/mini-peS2o",
        "target_samples": 600_000,  # ~0.9B tokens @ 1500 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "arxiv",
        "hf_dataset": "common-pile/arxiv_papers_filtered",
        "target_samples": 400_000,  # ~0.6B tokens @ 1500 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "openwebmath",
        "hf_dataset": "open-web-math/open-web-math",
        "target_samples": 2_000_000,  # ~1.5B tokens @ 750 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "tinygsm",
        "hf_dataset": "ostapeno/tinygsm-mind",
        "target_samples": 1_600_000,  # ~0.5B tokens @ 300 tok/sample
        "load_kwargs": {},
    },
    {
        "name": "algebraic_stack",
        "hf_dataset": "EleutherAI/proof-pile-2",
        "target_samples": 450_000,  # ~0.34B tokens @ 750 tok/sample
        "load_kwargs": {"name": "algebraic-stack", "trust_remote_code": True},
    },
    {
        "name": "gsm8k",
        "hf_dataset": "gsm8k",
        "target_samples": 550_000,  # ~0.17B tokens @ 300 tok/sample (with repetition)
        "load_kwargs": {"name": "main"},
        "handler": "gsm8k",
    },
    # GitHub code, ten languages; the language strings are those of the codeparrot dataset.
    {"name": "github_code_clean_python", "language": "Python", "target_samples": 3_200_000},  # ~1.6B tok
    {"name": "github_code_clean_javascript", "language": "JavaScript", "target_samples": 2_200_000},  # ~1.1B
    {"name": "github_code_clean_typescript", "language": "TypeScript", "target_samples": 1_100_000},  # ~0.55B
    {"name": "github_code_clean_java", "language": "Java", "target_samples": 1_100_000},  # ~0.55B
    {"name": "github_code_clean_cpp", "language": "C++", "target_samples": 900_000},  # ~0.45B
    {"name": "github_code_clean_go", "language": "GO", "target_samples": 800_000},  # ~0.4B
    {"name": "github_code_clean_rust", "language": "Rust", "target_samples": 550_000},  # ~0.28B
    {"name": "github_code_clean_shell", "language": "Shell", "target_samples": 450_000},  # ~0.23B
    {"name": "github_code_clean_sql", "language": "SQL", "target_samples": 350_000},  # ~0.18B
    {"name": "github_code_clean_html", "language": "HTML", "target_samples": 350_000},  # ~0.18B
]


def _save_hf_dataset(dataset: "Dataset", out_dir: Path) -> int:
    """Write an in-memory ``datasets.Dataset`` to ``shard-*.parquet`` files."""
    return write_parquet_shards(iter_dataset_tables(dataset), out_dir, SHARD_SIZE, prefix="shard")


def download_sliced(config: dict[str, Any], out_dir: Path) -> int:
    """Load ``train[:target_samples]`` of a Hub dataset and save it unchanged."""
    from datasets import load_dataset

    dataset = load_dataset(config["hf_dataset"], split=f"train[:{config['target_samples']}]", **config["load_kwargs"])
    print(f"  Loaded {len(dataset):,} samples")
    return _save_hf_dataset(dataset, out_dir)


def download_gsm8k(config: dict[str, Any], out_dir: Path) -> int:
    """GSM8K is small: combine question + answer into ``text`` and repeat examples up to the target count."""
    from datasets import load_dataset

    dataset = load_dataset(config["hf_dataset"], split="train", **config["load_kwargs"])
    dataset = dataset.map(format_gsm8k, remove_columns=dataset.column_names)
    indices = repeat_indices(len(dataset), config["target_samples"])
    print(f"  Original: {len(dataset):,} samples, target: {config['target_samples']:,}")
    return _save_hf_dataset(dataset.select(indices), out_dir)


def format_gsm8k(example: dict[str, Any]) -> dict[str, str]:
    """Concatenate a GSM8K question and answer into a single ``text`` field."""
    return {"text": f"Question: {example['question']}\n\nAnswer: {example['answer']}", "source": "gsm8k"}


def repeat_indices(num_rows: int, target: int) -> list[int]:
    """Indices that cycle through ``range(num_rows)`` until ``target`` rows are covered."""
    full_copies, remainder = divmod(target, num_rows)
    return list(range(num_rows)) * full_copies + list(range(remainder))


def download_github_code(config: dict[str, Any], out_dir: Path) -> int:
    """Stream ``codeparrot/github-code-clean`` and keep the first ``target_samples`` files of one language."""
    from datasets import load_dataset

    stream = load_dataset(GITHUB_CODE_DATASET, split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
    rows = iter_language(stream, config["language"], config["target_samples"])
    return write_dict_rows(rows, out_dir, SHARD_SIZE, prefix="shard")


def iter_language(rows: Iterable[dict[str, Any]], language: str, limit: int) -> Iterator[dict[str, Any]]:
    """Yield up to ``limit`` rows whose ``language`` column equals ``language``."""
    taken = 0
    for row in rows:
        if taken >= limit:
            return
        if row["language"] == language:
            taken += 1
            yield row


def download_dataset(config: dict[str, Any], raw_dir: Path) -> bool:
    """Download one source into ``raw_dir/<name>``; returns True on success."""
    out_dir = raw_dir / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print_header(config["name"])
    print(f"  Target: {config['target_samples']:,} samples -> {out_dir}")
    try:
        if config.get("handler") == "gsm8k":
            shards = download_gsm8k(config, out_dir)
        elif "language" in config:
            shards = download_github_code(config, out_dir)
        else:
            shards = download_sliced(config, out_dir)
        print(f"  Saved {shards} shards for {config['name']}")
        return True
    except Exception as exc:  # one failing source must not abort the others
        print(f"  Error downloading {config['name']}: {exc}")
        traceback.print_exc()
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--parallel", type=int, default=1, help="Datasets to download concurrently (default: 1)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="Source names to download (default: all)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_hf_cache(args.cache_dir)
    raw_dir = args.dataset_dir / "pretraining" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    configs = DATASETS if not args.datasets else [c for c in DATASETS if c["name"] in set(args.datasets)]
    print_header(f"Pretraining downloader ({args.parallel} concurrent)")
    print(f"Output: {raw_dir}")
    print(f"Datasets: {len(configs)}, total samples: {sum(c['target_samples'] for c in configs):,}")
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN not set; gated sources may fail.")

    start = time.time()
    successful: list[str] = []
    failed: list[str] = []
    if args.parallel == 1:
        for config in configs:
            (successful if download_dataset(config, raw_dir) else failed).append(config["name"])
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(download_dataset, c, raw_dir): c["name"] for c in configs}
            for future in as_completed(futures):
                (successful if future.result() else failed).append(futures[future])

    print_header("Download summary")
    print(f"Successful: {len(successful)} / {len(configs)}: {successful}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Total time: {(time.time() - start) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
