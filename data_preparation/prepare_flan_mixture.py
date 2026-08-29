# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Build the instruction-finetuning mixture at ``dataset/flan_mixture/{train,validation}/data-*.parquet``.

Eight sources are streamed from the Hub, converted to ``instruction`` / ``input`` / ``output`` rows and
length-filtered (<= 2048 estimated tokens) while downloading. Default composition of ``--total_examples``:

    FLAN Collection 40%, MetaMathQA 15%, Orca-Math 10%, Evol-Instruct-Code 12.5%, Code Alpaca 2.5%,
    SlimOrca-Dedup 10%, ShareGPT (quality filtered) 5%, WizardLM Evol V2 5%

Then: optional input inversions (``--add_input_inversions``), exact deduplication, empty-field removal, shuffle
(seed 42) and a train/validation split. The thesis run used ``--add_input_inversions --inversion_ratio 0.05``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from data_preparation.common import (
    RANDOM_SEED,
    add_common_args,
    configure_hf_cache,
    iter_dataset_tables,
    md5_hex,
    print_header,
    write_dict_rows,
    write_parquet_shards,
)

MAX_TOKENS = 2048
SHARD_SIZE = 10_000
SHAREGPT_CHECK_LIMIT = 100_000

# (display name, HF dataset id, CLI count option, default share of --total_examples)
SOURCES = [
    ("flan", "FLAN Collection", "Open-Orca/FLAN", 0.40),
    ("metamath", "MetaMathQA", "meta-math/MetaMathQA", 0.15),
    ("orca_math", "Orca-Math", "microsoft/orca-math-word-problems-200k", 0.10),
    ("evol_code", "Evol-Instruct-Code", "nickrosh/Evol-Instruct-Code-80k-v1", 0.125),
    ("code_alpaca", "Code Alpaca", "sahil2801/CodeAlpaca-20k", 0.025),
    ("slimorca", "SlimOrca-Dedup", "Open-Orca/SlimOrca-Dedup", 0.10),
    ("sharegpt", "ShareGPT (filtered)", "anon8231489123/ShareGPT_Vicuna_unfiltered", 0.05),
    ("wizardlm", "WizardLM Evol V2", "WizardLM/WizardLM_evol_instruct_V2_196k", 0.05),
]


# --- row-level functions --------------------------------------------------------------------------------------------


def standardize_format(example: dict[str, Any]) -> dict[str, str]:
    """Convert the various source schemas to ``{"instruction", "input", "output"}``."""
    if "instruction" in example and "output" in example:
        return {
            "instruction": str(example["instruction"]),
            "input": str(example.get("input", "")),
            "output": str(example["output"]),
        }
    if "inputs" in example and "targets" in example:  # FLAN
        return {"instruction": str(example["inputs"]), "input": "", "output": str(example["targets"])}
    if "conversations" in example and isinstance(example["conversations"], list):
        convs = example["conversations"]
        if convs and "from" in convs[0]:  # SlimOrca / ShareGPT with role tags
            system_msg = human_msg = gpt_msg = ""
            for turn in convs:
                if turn.get("from") == "system":
                    system_msg = turn.get("value", "")
                elif turn.get("from") == "human":
                    human_msg = turn.get("value", "")
                elif turn.get("from") == "gpt":
                    gpt_msg = turn.get("value", "")
            return {"instruction": human_msg, "input": system_msg, "output": gpt_msg}
        if len(convs) >= 2:
            return {"instruction": convs[0].get("value", ""), "input": "", "output": convs[1].get("value", "")}
    if "question" in example and "response" in example:  # OpenOrca
        return {
            "instruction": str(example["question"]),
            "input": str(example.get("system_prompt", "")),
            "output": str(example["response"]),
        }
    if "problem" in example and "solution" in example:
        return {"instruction": str(example["problem"]), "input": "", "output": str(example["solution"])}
    if "query" in example and "response" in example:  # MetaMathQA
        return {"instruction": str(example["query"]), "input": "", "output": str(example["response"])}
    if "question" in example and "answer" in example:  # Orca-Math
        return {"instruction": str(example["question"]), "input": "", "output": str(example["answer"])}
    raise ValueError(f"Cannot convert example to standard format: {list(example.keys())}")


def check_length(example: dict[str, Any], max_tokens: int = MAX_TOKENS) -> bool:
    """Whitespace-token estimate (1.3 tokens per word) must not exceed ``max_tokens``."""
    text = f"{example['instruction']} {example.get('input', '')} {example['output']}"
    return len(text.split()) * 1.3 <= max_tokens


def is_quality_sharegpt(example: dict[str, Any]) -> bool:
    """ShareGPT quality filter: human→gpt opening, 50-2000 chars per side, no code blocks in the answer."""
    convs = example.get("conversations")
    if not convs or len(convs) < 2:
        return False
    if convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
        return False
    human_text, gpt_text = convs[0].get("value", ""), convs[1].get("value", "")
    if not 50 <= len(human_text) <= 2000 or not 50 <= len(gpt_text) <= 2000:
        return False
    return not any(p in gpt_text.lower() for p in ("```python", "```java", "```cpp", "```javascript"))


def create_input_inversion(example: dict[str, Any]) -> dict[str, Any]:
    """Ask for the instruction given the output (swap direction); no-op on rows without instruction/output."""
    if "instruction" not in example or "output" not in example:
        return example
    if not example.get("input", "") and not example.get("output", ""):
        return example
    inverted_output = example["instruction"]
    if example.get("input"):
        inverted_output += f"\nInput: {example['input']}"
    return {
        "instruction": f"Given this output, what was the likely instruction or input?\n\nOutput: {example['output']}",
        "input": "",
        "output": inverted_output,
    }


def compute_example_hash(example: dict[str, Any]) -> str:
    """Content hash for exact deduplication."""
    return md5_hex(f"{example['instruction']}\n{example['input']}\n{example['output']}")


def has_required_fields(example: dict[str, Any]) -> bool:
    """True if instruction and output are non-empty after stripping."""
    return bool(example["instruction"] and example["instruction"].strip()) and bool(
        example["output"] and example["output"].strip()
    )


# --- download -------------------------------------------------------------------------------------------------------


def iter_standardized(
    stream: Iterable[dict[str, Any]], target_count: int, sharegpt: bool = False
) -> Iterator[dict[str, str]]:
    """Yield up to ``target_count`` standardized, length-checked rows from a streaming dataset."""
    saved = 0
    checked = 0
    for example in stream:
        if saved >= target_count:
            return
        checked += 1
        if sharegpt and not is_quality_sharegpt(example):
            continue
        try:
            formatted = standardize_format(example)
        except Exception:  # unconvertible rows are skipped, as in the original pipeline
            continue
        if not check_length(formatted):
            continue
        saved += 1
        yield formatted
        if sharegpt and checked >= SHAREGPT_CHECK_LIMIT:
            print(f"  Reached ShareGPT check limit of {SHAREGPT_CHECK_LIMIT:,} examples")
            return


def download_source(key: str, display: str, hf_dataset: str, target_count: int, out_dir: Path) -> int:
    """Stream one source to ``out_dir`` shards; returns the number of rows saved (0 on failure)."""
    from datasets import load_dataset

    if target_count <= 0:
        print(f"\nSkipping {display} (count 0)")
        return 0
    print(f"\nDownloading {display} (target: {target_count:,} examples)...")
    saved = 0

    def counted() -> Iterator[dict[str, str]]:
        nonlocal saved
        for row in iter_standardized(stream, target_count, sharegpt=key == "sharegpt"):
            saved += 1
            yield row

    try:
        stream = load_dataset(hf_dataset, split="train", streaming=True)
        write_dict_rows(counted(), out_dir, SHARD_SIZE)
    except Exception as exc:  # one failing source must not abort the mixture
        print(f"  Error downloading {display}: {exc}")
        return 0
    print(f"  Saved {saved:,} examples")
    return saved


# --- CLI ------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--total_examples", type=int, default=400000, help="Total examples (default: 400,000)")
    for key, display, _, share in SOURCES:
        parser.add_argument(
            f"--{key}_count", type=int, default=None, help=f"{display} examples (default: {share * 100:g}%% of total)"
        )
    parser.add_argument("--add_input_inversions", action="store_true", help="Invert a share of the examples")
    parser.add_argument("--inversion_ratio", type=float, default=0.3, help="Share of examples to invert (default: 0.3)")
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation share (default: 0.05)")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker processes for datasets.map (default: 4)")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_hf_cache(args.cache_dir)
    output_dir = args.dataset_dir / "flan_mixture"
    counts = {
        key: int(args.total_examples * share) if getattr(args, f"{key}_count") is None else getattr(args, f"{key}_count")
        for key, _, _, share in SOURCES
    }

    print_header("FLAN-focused balanced mixture preparation")
    print(f"Target: {args.total_examples:,} examples -> {output_dir}\nComposition:")
    for key, display, _, share in SOURCES:
        print(f"  - {display:<22} {counts[key]:8,}  ({share:5.1%})")
    print(f"  Total: {sum(counts.values()):,}")
    inv = f"enabled ({args.inversion_ratio:.0%})" if args.add_input_inversions else "disabled"
    print(f"Input inversions: {inv}; dedup, length filter (max {MAX_TOKENS} tokens), field verification: enabled")
    print(f"Train/val split: {1 - args.val_split:.0%} / {args.val_split:.0%}")
    if args.dry_run:
        print("\n[DRY RUN] Plan displayed. Exiting without downloading.")
        return

    from datasets import concatenate_datasets, load_dataset

    random.seed(RANDOM_SEED)
    temp_dir = output_dir / "temp_datasets"
    temp_dir.mkdir(parents=True, exist_ok=True)
    dataset_counts = {
        key: download_source(key, display, hf_dataset, counts[key], temp_dir / key)
        for key, display, hf_dataset, _ in SOURCES
    }
    successful = [key for key, n in dataset_counts.items() if n > 0]
    if not successful:
        raise SystemExit("No datasets successfully downloaded.")
    print(f"\nDownloaded {len(successful)} / {len(SOURCES)} sources, {sum(dataset_counts.values()):,} examples")

    combined = concatenate_datasets(
        [load_dataset("parquet", data_dir=str(temp_dir / key), split="train") for key in successful]
    )
    print(f"Combined: {len(combined):,} examples")

    if args.add_input_inversions:
        inversion_count = int(len(combined) * args.inversion_ratio)
        inversion_indices = set(random.sample(range(len(combined)), inversion_count))
        combined = combined.map(
            lambda ex, idx: create_input_inversion(ex) if idx in inversion_indices else ex,
            with_indices=True,
            num_proc=args.num_workers,
            desc="Creating inversions",
        )
        print(f"Inverted {inversion_count:,} examples")

    print("Deduplicating...")
    seen: set[str] = set()
    unique_indices = []
    for idx, example in enumerate(combined):
        # `datasets` declares __iter__ as yielding dict | list; rows are dicts here
        content_hash = compute_example_hash(example)  # pyright: ignore[reportArgumentType]
        if content_hash not in seen:
            seen.add(content_hash)
            unique_indices.append(idx)
    print(f"  Removed {len(combined) - len(unique_indices):,} duplicates")
    combined = combined.select(unique_indices)
    before = len(combined)
    combined = combined.filter(has_required_fields, desc="Verifying fields")
    print(f"  Removed {before - len(combined):,} examples with empty fields; final {len(combined):,}")

    combined = combined.shuffle(seed=RANDOM_SEED)
    split_idx = int(len(combined) * (1 - args.val_split))
    splits = {
        "train": combined.select(range(split_idx)),
        "validation": combined.select(range(split_idx, len(combined))) if split_idx < len(combined) else combined.select([]),
    }
    for name, split in splits.items():
        shards = write_parquet_shards(iter_dataset_tables(split), output_dir / name, SHARD_SIZE)
        print(f"  {name}: {len(split):,} examples in {shards} shards -> {output_dir / name}")
    shutil.rmtree(temp_dir, ignore_errors=True)

    metadata = {
        "total_examples": len(combined),
        "train_examples": len(splits["train"]),
        "val_examples": len(splits["validation"]),
        "dataset_counts": dataset_counts,
        "input_inversions": args.add_input_inversions,
        "inversion_ratio": args.inversion_ratio,
        "max_tokens": MAX_TOKENS,
        "random_seed": RANDOM_SEED,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print_header("Dataset preparation complete")
    print(f"Output: {output_dir} (train/, validation/, metadata.json)")


if __name__ == "__main__":
    main()
