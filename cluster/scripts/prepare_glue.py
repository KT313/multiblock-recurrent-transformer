#!/usr/bin/env python3
"""
Download and prepare GLUE benchmark datasets for finetuning.

GLUE (General Language Understanding Evaluation) consists of 9 tasks:
- CoLA: Corpus of Linguistic Acceptability
- SST-2: Stanford Sentiment Treebank
- MRPC: Microsoft Research Paraphrase Corpus
- QQP: Quora Question Pairs
- STS-B: Semantic Textual Similarity Benchmark
- MNLI: Multi-Genre Natural Language Inference
- QNLI: Question Natural Language Inference
- RTE: Recognizing Textual Entailment
- WNLI: Winograd Natural Language Inference

Usage:
    # Download all GLUE tasks
    python prepare_glue.py

    # Download specific tasks
    python prepare_glue.py --tasks cola sst2 mnli

    # Custom output directory
    python prepare_glue.py --output_dir /path/to/glue
"""

from datasets import load_dataset
from pathlib import Path
from datasets.utils.logging import set_verbosity_info, enable_progress_bar
import argparse

set_verbosity_info()
enable_progress_bar()

# GLUE task configurations
GLUE_TASKS = {
    "cola": "cola",
    "sst2": "sst2",
    "mrpc": "mrpc",
    "qqp": "qqp",
    "stsb": "stsb",
    "mnli": "mnli",
    "mnli_matched": "mnli",  # alias
    "mnli_mismatched": "mnli",  # alias
    "qnli": "qnli",
    "rte": "rte",
    "wnli": "wnli",
}

# Tasks that have a test split with labels (most GLUE test sets don't have labels)
# For evaluation, we typically use validation split
TASKS_WITH_TEST = []  # GLUE test sets don't have labels

# Default tasks to download (all tasks)
DEFAULT_TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]


def format_glue_for_causal_lm(example, task_name):
    """
    Format GLUE examples for causal language modeling.
    Converts classification/regression tasks into text generation format.
    """
    # Different tasks have different input formats
    if task_name in ["cola"]:
        # Single sentence
        text = f"Sentence: {example['sentence']}\nAcceptable: "
        label_text = "yes" if example.get('label', 0) == 1 else "no"

    elif task_name in ["sst2"]:
        # Sentiment classification
        text = f"Review: {example['sentence']}\nSentiment: "
        label_text = "positive" if example.get('label', 0) == 1 else "negative"

    elif task_name in ["mrpc", "qqp"]:
        # Sentence pair similarity
        text = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nEquivalent: "
        label_text = "yes" if example.get('label', 0) == 1 else "no"

    elif task_name == "stsb":
        # Regression task - similarity score
        text = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nSimilarity (0-5): "
        label_text = f"{example.get('label', 0.0):.1f}"

    elif task_name in ["mnli"]:
        # Natural language inference
        text = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nRelation: "
        labels = ["entailment", "neutral", "contradiction"]
        label_text = labels[example.get('label', 0)]

    elif task_name in ["qnli"]:
        # Question answering NLI
        text = f"Question: {example['question']}\nSentence: {example['sentence']}\nAnswer contained: "
        label_text = "yes" if example.get('label', 0) == 0 else "no"  # entailment = 0

    elif task_name in ["rte"]:
        # Recognizing textual entailment
        text = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nEntailment: "
        label_text = "yes" if example.get('label', 0) == 0 else "no"  # entailment = 0

    elif task_name in ["wnli"]:
        # Winograd NLI
        text = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nEntailment: "
        label_text = "yes" if example.get('label', 0) == 1 else "no"

    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Combine into single text for causal LM
    full_text = text + label_text

    return {"text": full_text}


def download_glue_task(task_name, base_dir, format_for_clm=True):
    """Download and save a single GLUE task."""
    print(f"\n{'='*80}")
    print(f"Processing GLUE task: {task_name.upper()}")
    print(f"{'='*80}")

    task_config = GLUE_TASKS[task_name]
    task_dir = base_dir / task_name

    # Create directories
    (task_dir / "train").mkdir(parents=True, exist_ok=True)
    (task_dir / "validation").mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading {task_name} from HuggingFace...")
    ds = load_dataset("glue", task_config)

    # MNLI has matched and mismatched validation sets
    if task_name == "mnli":
        print("Processing MNLI (with matched and mismatched validation)...")
        (task_dir / "validation_matched").mkdir(parents=True, exist_ok=True)
        (task_dir / "validation_mismatched").mkdir(parents=True, exist_ok=True)

        # Process and save
        if format_for_clm:
            train = ds["train"].map(lambda x: format_glue_for_causal_lm(x, task_name), remove_columns=ds["train"].column_names)
            val_matched = ds["validation_matched"].map(lambda x: format_glue_for_causal_lm(x, task_name), remove_columns=ds["validation_matched"].column_names)
            val_mismatched = ds["validation_mismatched"].map(lambda x: format_glue_for_causal_lm(x, task_name), remove_columns=ds["validation_mismatched"].column_names)
        else:
            train = ds["train"]
            val_matched = ds["validation_matched"]
            val_mismatched = ds["validation_mismatched"]

        train.save_to_disk(str(task_dir / "train"))
        val_matched.save_to_disk(str(task_dir / "validation_matched"))
        val_mismatched.save_to_disk(str(task_dir / "validation_mismatched"))

        print(f"✓ Saved train: {train.num_rows:,} examples")
        print(f"✓ Saved validation_matched: {val_matched.num_rows:,} examples")
        print(f"✓ Saved validation_mismatched: {val_mismatched.num_rows:,} examples")
    else:
        # Standard train/validation split
        if format_for_clm:
            train = ds["train"].map(lambda x: format_glue_for_causal_lm(x, task_name), remove_columns=ds["train"].column_names)
            validation = ds["validation"].map(lambda x: format_glue_for_causal_lm(x, task_name), remove_columns=ds["validation"].column_names)
        else:
            train = ds["train"]
            validation = ds["validation"]

        train.save_to_disk(str(task_dir / "train"))
        validation.save_to_disk(str(task_dir / "validation"))

        print(f"✓ Saved train: {train.num_rows:,} examples")
        print(f"✓ Saved validation: {validation.num_rows:,} examples")

    print(f"✓ Task saved to: {task_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download GLUE benchmark datasets")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=list(GLUE_TASKS.keys()),
        help="GLUE tasks to download (default: all tasks)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/shared_storage/recpre/datasets/glue",
        help="Base directory to save datasets"
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Don't format for causal LM, keep original GLUE format"
    )

    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GLUE tasks: {', '.join(args.tasks)}")
    print(f"Output directory: {base_dir}")
    print(f"Format for causal LM: {not args.no_format}")

    for task in args.tasks:
        try:
            download_glue_task(task, base_dir, format_for_clm=not args.no_format)
        except Exception as e:
            print(f"✗ Error processing {task}: {e}")
            continue

    print(f"\n{'='*80}")
    print("All tasks downloaded successfully!")
    print(f"{'='*80}")
    print(f"Datasets saved to: {base_dir}")
    print("\nTo use in training config:")
    print("data_config:")
    print("  train_data:")
    print("    - type: hfds")
    print("      prefix: glue-train")
    print(f"      data_dir: {base_dir}/<task_name>/train")
    print("      weight: 1")


if __name__ == "__main__":
    main()
