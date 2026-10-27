# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Read-only word n-gram overlap between benchmark evaluation docs and the rows a checkpoint was trained on.

    uv run --frozen --no-sync python evaluation/contamination_check.py --config config/<run>.yaml --workers 32 \
        --output outputs/<run>/contamination

Benchmark side: the evaluation docs of every task as lm-eval scores them (test split, or validation where the task
uses it). Each string field of a doc gives its normalized 13-grams (`get_ngram_set`, the normalization of the
build's decontamination); a doc whose fields are all too short falls back to its fields joined, and a doc still
shorter than 13 words is reported as unchecked. Group tasks (mmlu) report under the requested name, while every
example keeps its lm-eval leaf task and position (`mmlu_anatomy`, 12). A task that cannot be loaded (a gated
dataset without access, a missing optional dependency) is skipped and named in the report.

Training side: the checkpoint's own record of the data split and the rows its stream read (`validation_rows`,
`source_rows`, `data_stream.consumed_rows`). Sources are read without shuffling from the first training row, so
the trained rows of a source are `[validation_rows, validation_rows + consumed)`, or the whole training range when
the stream wrapped around (`--scope all` always takes the whole training range). Shards are only read: no lock,
no dataset preparation, no write outside `--output`; a source whose row count on disk differs from the
checkpoint's record is an error, because the row ranges would then point at other rows.

Report (`contamination.md`, `contamination.json`): per task, how many examples share any n-gram with the training
rows, at least half of their n-grams ("likely contaminated") and all of them, with the sources responsible; per
source, the rows read and the rows sharing n-grams with a benchmark. Isolated 13-gram hits are mostly shared
boilerplate (licence headers, common phrasings); the half-of-n-grams share is the meaningful figure.

Sample lists for benchmark runs, per leaf task by position in its evaluation documents: `contaminated_samples.json`
(examples with at least `--flag_share` of their n-grams in the training rows) and `clean_samples.json` (all
others, unchecked short examples included), the format `benchmark_checkpoint.py --samples` reads; plus
`example_shares.json` with the share of every example that has any hit, for other thresholds later. A leaf task
whose examples are all flagged has no clean list and cannot be run clean.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# drop this folder from the import path: its evaluate.py would shadow the `evaluate` package that lm-eval's code
# tasks (humaneval, mbpp) import
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_preparation.lib.stages.row_pipeline import get_ngram_set, instruct_text  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_TASKS: tuple[str, ...] = (
    "arc_challenge", "hellaswag", "mmlu", "winogrande",  # the benchmarks of the training configs
    "piqa", "boolq", "race", "truthfulqa_mc2", "arc_easy", "openbookqa",  # the extra multiple-choice grid
    "gsm8k", "humaneval", "mbpp",  # the generation tasks
    "leaderboard_mmlu_pro", "anli", "gpqa_main_zeroshot", "gpqa_diamond_zeroshot", "minerva_math500",  # the suite
    "cb", "copa", "rte", "wic", "wsc", "sst2", "mrpc", "cola",
)
NGRAM = 13
LIKELY_SHARE = 0.5  # an example with at least this share of its n-grams in the training rows counts as contaminated
SAMPLES_FORMAT = "mbrt_benchmark_samples_v1"
ROW_SHARE = 0.1  # a training row with more than this share of its n-grams in one task's docs (the build's threshold)
WORST_EXAMPLES = 10  # per task in the JSON, for manual inspection
PREVIEW_CHARS = 200
BATCH_ROWS = 1024

_INDEX: dict[str, int] = {}  # n-gram -> bit mask of the task indices it occurs in; set before the workers fork
_KEYS: frozenset[str] = frozenset()  # its keys, for C-level set intersection with each row's n-grams


@dataclass(frozen=True)
class Example:
    task: str  # the requested task (mmlu)
    leaf: str  # the lm-eval task that holds the doc (mmlu_anatomy), the key of a sample list
    index: int  # position in the leaf task's evaluation docs
    ngrams: frozenset[str]
    preview: str


@dataclass(frozen=True)
class Unit:
    """Rows [start, stop) of one shard file (indices inside the file) of one source."""

    source: str
    path: str
    start: int
    stop: int


@dataclass
class UnitResult:
    source: str
    rows: int = 0
    million_chars: float = 0.0
    hits: set[str] = field(default_factory=set)  # benchmark n-grams found in these rows
    rows_with_hit: int = 0
    rows_over_share: dict[int, int] = field(default_factory=dict)  # task index -> rows above ROW_SHARE


def doc_strings(value: Any) -> Iterator[str]:
    """Every string inside a doc, depth first (dicts by value, lists in order)."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from doc_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from doc_strings(item)


def example_ngrams(doc: Mapping[str, Any], n: int = NGRAM) -> frozenset[str]:
    """
    The n-grams of each string of a doc on its own, so no n-gram spans two fields that training text would not have
    side by side; the fields joined when none of them alone is long enough.
    """

    strings = list(doc_strings(doc))
    ngrams: set[str] = set()
    for text in strings:
        ngrams |= get_ngram_set(text, n)
    return frozenset(ngrams or get_ngram_set(" ".join(strings), n))


def load_lm_eval_docs(tasks: Sequence[str]) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, str]]:
    """
    ({task: {leaf task: evaluation docs}}, {task: why it could not be loaded}) through lm-eval itself, one task
    at a time, so one that fails (gated, a missing dependency) is reported instead of stopping the check.
    """

    from lm_eval.tasks import TaskManager, get_task_dict

    def collect(mapping: Mapping[Any, Any], into: dict[str, list[dict[str, Any]]]) -> None:
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                collect(value, into)
            else:
                into[str(key)] = [dict(doc) for doc in value.eval_docs]

    manager = TaskManager()
    docs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    skipped: dict[str, str] = {}
    for task in tasks:
        try:
            leaves: dict[str, list[dict[str, Any]]] = {}
            collect(get_task_dict([task], manager), leaves)
        except Exception as error:  # reported per task; the check continues with the others
            skipped[task] = f"{type(error).__name__}: {error}"
            log.warning("task %s skipped, it cannot be loaded: %s", task, skipped[task])
            continue
        docs[task] = leaves
    return docs, skipped


def build_examples(
    docs: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], n: int = NGRAM,
) -> tuple[list[Example], dict[str, int], dict[str, int]]:
    """The checkable examples, per task the number too short to have an n-gram, and per leaf task its doc count."""

    examples: list[Example] = []
    unchecked: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for task, leaves in docs.items():
        unchecked[task] = 0
        for leaf, leaf_docs in leaves.items():
            sizes[leaf] = len(leaf_docs)
            for index, doc in enumerate(leaf_docs):
                ngrams = example_ngrams(doc, n)
                if not ngrams:
                    unchecked[task] += 1
                    continue
                preview = " ".join(" ".join(doc_strings(doc)).split())[:PREVIEW_CHARS]
                examples.append(Example(task, leaf, index, ngrams, preview))
    return examples, unchecked, sizes


def build_index(examples: Iterable[Example], tasks: Sequence[str]) -> dict[str, int]:
    position = {task: index for index, task in enumerate(tasks)}
    index: dict[str, int] = {}
    for example in examples:
        bit = 1 << position[example.task]
        for ngram in example.ngrams:
            index[ngram] = index.get(ngram, 0) | bit
    return index


def trained_ranges(
    validation_rows: Mapping[str, int], source_rows: Mapping[str, int], consumed_rows: Mapping[str, int], *,
    scope: str, sources: Sequence[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """
    {source: [first, end) processed rows to scan}. A source counts as trained when the stream read from it; the
    rows read run on from the first training row and wrap to cover the whole training range.
    """

    if scope not in ("trained", "all"):
        raise ValueError(f"scope must be 'trained' or 'all', got {scope!r}")
    names = [name for name, rows in consumed_rows.items() if rows > 0]
    if sources is not None:
        unknown = sorted(set(sources) - set(names))
        if unknown:
            raise KeyError(f"source(s) {unknown} were not trained on; trained: {sorted(names)}")
        names = [name for name in names if name in sources]
    ranges: dict[str, tuple[int, int]] = {}
    for name in sorted(names):
        first, end = validation_rows.get(name, 0), source_rows[name]
        if scope == "trained":
            end = min(end, first + consumed_rows[name])
        if end > first:
            ranges[name] = (first, end)
    return ranges


def plan_units(directories: Mapping[str, Path], ranges: Mapping[str, tuple[int, int]]) -> list[Unit]:
    """One unit per shard file overlapping a source's range, in the order the training stream reads them."""

    import pyarrow.parquet as pq

    from data_preparation.lib.storage.parquet import SHARD_PATTERN

    units: list[Unit] = []
    for source, (first, end) in ranges.items():
        files = sorted(path for path in directories[source].glob("*.parquet") if SHARD_PATTERN.match(path.name))
        offset = 0
        for path in files:
            rows = pq.ParquetFile(path).metadata.num_rows  # footer only
            start, stop = max(first - offset, 0), min(end - offset, rows)
            if stop > start:
                units.append(Unit(source, str(path), start, stop))
            offset += rows
    return units


def _row_texts(batch: Any) -> list[str]:
    columns = set(batch.schema.names)
    if "text" in columns:
        return [text or "" for text in batch.column("text").to_pylist()]
    if "messages" in columns:
        return ["\n\n".join(message.get("content") or "" for message in messages or [])
                for messages in batch.column("messages").to_pylist()]
    if "instruction" in columns:
        return [instruct_text(row) for row in batch.to_pylist()]
    raise ValueError(f"no text column among {sorted(columns)}")


def _text_columns(path: str) -> list[str]:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    for wanted in (["text"], ["messages"], ["instruction", "input", "output"]):
        if set(wanted) <= names:
            return wanted
    raise ValueError(f"{path}: no text column among {sorted(names)}")


def scan_unit(unit: Unit, n: int = NGRAM) -> UnitResult:
    """Read one unit's rows (read-only) and match their n-grams against the module index."""

    import pyarrow.parquet as pq

    result = UnitResult(unit.source)
    row = 0
    with pq.ParquetFile(unit.path) as shard:
        for batch in shard.iter_batches(batch_size=BATCH_ROWS, columns=_text_columns(unit.path)):
            begin, end = row, row + batch.num_rows
            row = end
            if end <= unit.start:
                continue
            if begin >= unit.stop:
                break
            batch = batch.slice(max(unit.start - begin, 0), min(unit.stop, end) - max(unit.start, begin))
            for text in _row_texts(batch):
                result.rows += 1
                result.million_chars += len(text) / 1e6
                ngrams = get_ngram_set(text, n)
                hits = ngrams & _KEYS
                if not hits:
                    continue
                result.rows_with_hit += 1
                result.hits |= hits
                per_task: dict[int, int] = {}
                for ngram in hits:
                    mask = _INDEX[ngram]
                    while mask:
                        bit = mask & -mask
                        task = bit.bit_length() - 1
                        per_task[task] = per_task.get(task, 0) + 1
                        mask ^= bit
                for task, count in per_task.items():
                    if count / len(ngrams) > ROW_SHARE:
                        result.rows_over_share[task] = result.rows_over_share.get(task, 0) + 1
    return result


def scan(units: Sequence[Unit], index: dict[str, int], workers: int,
         progress: Callable[[int, int, float], None] | None = None) -> list[UnitResult]:
    """Scan every unit, in forked worker processes that share the index copy-on-write when workers > 1."""

    global _INDEX, _KEYS
    _INDEX, _KEYS = index, frozenset(index)
    try:
        if workers <= 1:
            results = []
            for done, unit in enumerate(units, 1):
                results.append(scan_unit(unit))
                if progress is not None:
                    progress(done, len(units), sum(item.million_chars for item in results))
            return results
        gc.freeze()  # keep the index's pages shared: no collector pass touches them in the children
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            results = []
            for result in pool.imap_unordered(scan_unit, units):
                results.append(result)
                if progress is not None:
                    progress(len(results), len(units), sum(item.million_chars for item in results))
            return results
    finally:
        _INDEX, _KEYS = {}, frozenset()
        gc.unfreeze()


def hits_by_source(results: Iterable[UnitResult]) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = {}
    for result in results:
        hits.setdefault(result.source, set()).update(result.hits)
    return hits


def sample_lists(
    examples: Iterable[Example], sizes: Mapping[str, int], results: Sequence[UnitResult], flag_share: float,
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, dict[int, float]]]:
    """
    (clean, contaminated, shares) per leaf task: positions with at least `flag_share` of their n-grams in the
    training rows are contaminated, every other position is clean (unchecked short ones included); shares hold
    every example with any hit. A leaf task without clean or without contaminated positions has no entry there.
    """

    found = set().union(*hits_by_source(results).values()) if results else set()
    flagged: dict[str, set[int]] = {}
    shares: dict[str, dict[int, float]] = {}
    for example in examples:
        matched = len(example.ngrams & found)
        if not matched:
            continue
        share = matched / len(example.ngrams)
        shares.setdefault(example.leaf, {})[example.index] = round(share, 4)
        if share >= flag_share:
            flagged.setdefault(example.leaf, set()).add(example.index)
    clean = {leaf: [index for index in range(count) if index not in flagged.get(leaf, set())] for leaf, count in sizes.items()}
    contaminated = {leaf: sorted(indices) for leaf, indices in sorted(flagged.items())}
    return {leaf: indices for leaf, indices in sorted(clean.items()) if indices}, contaminated, shares


def write_sample_lists(
    output: Path, clean: Mapping[str, list[int]], contaminated: Mapping[str, list[int]],
    shares: Mapping[str, Mapping[int, float]], criterion: Mapping[str, Any],
) -> None:
    for name, samples in (("clean_samples.json", clean), ("contaminated_samples.json", contaminated)):
        record = {"format": SAMPLES_FORMAT, "criterion": {**criterion, "selection": name.removesuffix("_samples.json")},
                  "samples": dict(samples)}
        (output / name).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (output / "example_shares.json").write_text(json.dumps({"criterion": dict(criterion), "shares": {
        leaf: {str(index): share for index, share in sorted(values.items())} for leaf, values in sorted(shares.items())
    }}, indent=1) + "\n", encoding="utf-8")


def summarize(
    examples: Sequence[Example], unchecked: Mapping[str, int], tasks: Sequence[str], results: Sequence[UnitResult],
    ranges: Mapping[str, tuple[int, int]], flag_share: float = LIKELY_SHARE,
) -> dict[str, Any]:
    by_source_hits = hits_by_source(results)
    sources: dict[str, dict[str, Any]] = {}
    for result in results:
        entry = sources.setdefault(result.source, {
            "rows_scanned": 0, "million_chars": 0.0, "rows_with_hit": 0, "rows_over_share": {},
            "row_range": list(ranges[result.source]),
        })
        entry["rows_scanned"] += result.rows
        entry["million_chars"] += result.million_chars
        entry["rows_with_hit"] += result.rows_with_hit
        for task, count in result.rows_over_share.items():
            entry["rows_over_share"][tasks[task]] = entry["rows_over_share"].get(tasks[task], 0) + count
    all_hits = set().union(*by_source_hits.values()) if by_source_hits else set()

    report_tasks: dict[str, dict[str, Any]] = {
        task: {"examples": unchecked.get(task, 0), "unchecked": unchecked.get(task, 0), "any_hit": 0, "likely": 0,
               "complete": 0, "flagged": 0, "likely_by_source": {}, "worst": []}
        for task in tasks
    }
    for example in examples:
        entry = report_tasks[example.task]
        entry["examples"] += 1
        found = example.ngrams & all_hits
        if not found:
            continue
        share = len(found) / len(example.ngrams)
        entry["any_hit"] += 1
        entry["likely"] += share >= LIKELY_SHARE
        entry["complete"] += share == 1.0
        entry["flagged"] += share >= flag_share
        by_source = {source: len(example.ngrams & hits) / len(example.ngrams)
                     for source, hits in by_source_hits.items() if example.ngrams & hits}
        for source, source_share in by_source.items():
            if source_share >= LIKELY_SHARE:
                entry["likely_by_source"][source] = entry["likely_by_source"].get(source, 0) + 1
        entry["worst"].append({"leaf": example.leaf, "index": example.index, "share": round(share, 4),
                               "ngrams": len(example.ngrams),
                               "sources": {source: round(value, 4) for source, value in
                                           sorted(by_source.items(), key=lambda item: -item[1])},
                               "preview": example.preview})
    for entry in report_tasks.values():
        entry["worst"] = sorted(entry["worst"], key=lambda item: -item["share"])[:WORST_EXAMPLES]
    return {"ngram": NGRAM, "likely_share": LIKELY_SHARE, "flag_share": flag_share, "row_share": ROW_SHARE,
            "tasks": report_tasks, "sources": sources}


def markdown_report(report: Mapping[str, Any], header: Mapping[str, Any]) -> str:
    lines = ["# Benchmark contamination check", ""]
    lines += [f"- {key}: {value}" for key, value in header.items()]
    lines += [
        "",
        f"Word {report['ngram']}-grams after lowercasing and whitespace normalization. \"Likely\": at least "
        f"{report['likely_share']:.0%} of an example's n-grams occur in the scanned training rows; \"complete\": all of "
        "them. Isolated hits are mostly shared boilerplate; the likely share is the meaningful figure. \"Excluded\": "
        f"examples with at least {report['flag_share']:.0%} of their n-grams found, listed in contaminated_samples.json "
        "and left out of clean_samples.json (for `benchmark_checkpoint.py --samples`).",
        "",
        "## By task",
        "",
        "| Task | Examples | Unchecked (too short) | Any hit | Likely | Complete | Excluded | Likely, by source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, entry in report["tasks"].items():
        checked = entry["examples"] - entry["unchecked"]

        def share(count: int, checked: int = checked) -> str:
            return f"{count} ({100 * count / checked:.1f}%)" if checked else str(count)

        by_source = ", ".join(f"{source} {count}" for source, count in
                              sorted(entry["likely_by_source"].items(), key=lambda item: -item[1])) or "-"
        lines.append(f"| {task} | {entry['examples']} | {entry['unchecked']} | {share(entry['any_hit'])} | "
                     f"{share(entry['likely'])} | {share(entry['complete'])} | {share(entry['flagged'])} | {by_source} |")
    lines += [
        "",
        "## By source",
        "",
        f"Rows over share: training rows with more than {report['row_share']:.0%} of their n-grams in one task's docs "
        "(the build decontamination's rule), per task.",
        "",
        "| Source | Row range | Rows scanned | M chars | Rows with any hit | Rows over share |",
        "|---|---|---:|---:|---:|---|",
    ]
    for source, entry in sorted(report["sources"].items()):
        over = ", ".join(f"{task} {count}" for task, count in sorted(entry["rows_over_share"].items(),
                                                                       key=lambda item: -item[1])) or "-"
        first, end = entry["row_range"]
        lines.append(f"| {source} | [{first:,}, {end:,}) | {entry['rows_scanned']:,} | {entry['million_chars']:,.0f} | "
                     f"{entry['rows_with_hit']:,} | {over} |")
    return "\n".join(lines) + "\n"


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", required=True, help="training config of the run")
    parser.add_argument("--checkpoint", default=None, help="default: the run's newest regular checkpoint")
    parser.add_argument("--output", required=True, help="new directory for the report; must not exist")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated lm-eval task names")
    parser.add_argument("--sources", default=None, help="comma-separated subset of the trained sources")
    parser.add_argument("--scope", choices=("trained", "all"), default="trained",
                        help="trained: the rows the checkpoint's stream read; all: every training row")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flag_share", type=float, default=LIKELY_SHARE,
                        help="share of an example's n-grams found in training that excludes it from clean_samples.json")
    arguments = parser.parse_args(argv)
    if not 0.0 < arguments.flag_share <= 1.0:
        parser.error("--flag_share must be in (0, 1]")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = parse_arguments(argv)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)  # fail before any work when a report is there already

    import torch

    from data_preparation import DatasetLayout, load_dataset_config
    from training.checkpoint import CheckpointMetadata, find_latest_checkpoint
    from training.data.dataset_resolver import processed_rows
    from training.execution.setup import get_run_directory
    from training.settings import parse_settings

    settings = parse_settings(["--config", arguments.config])
    checkpoint = (Path(arguments.checkpoint) if arguments.checkpoint
                  else find_latest_checkpoint(get_run_directory(settings), settings.run_name))
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint {checkpoint or ''} for run {settings.run_name}")
    # map the file instead of reading it: only the small metadata entries are touched
    metadata = CheckpointMetadata.from_state(torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True))
    dataset_config = load_dataset_config(settings.dataset_config)
    layout = DatasetLayout(Path(settings.dataset_dir)).for_config(dataset_config)
    sources = arguments.sources.split(",") if arguments.sources else None
    ranges = trained_ranges(metadata.validation_rows, metadata.source_rows, metadata.data_stream["consumed_rows"],
                            scope=arguments.scope, sources=sources)
    directories = {source: layout.processed_dir(source) for source in ranges}
    for source, directory in directories.items():
        on_disk = processed_rows(directory, f"source {source!r}")  # footers, cross-checked against the manifest
        if on_disk != metadata.source_rows[source]:
            raise RuntimeError(f"source {source!r}: {on_disk} rows in {directory}, but the checkpoint trained on "
                               f"{metadata.source_rows[source]}; the data changed since, row ranges would be wrong")
    units = plan_units(directories, ranges)
    log.info("checkpoint %s (step %d): %d sources, %d shard units, scope %s", checkpoint, metadata.step,
             len(ranges), len(units), arguments.scope)

    docs, skipped = load_lm_eval_docs(list(dict.fromkeys(arguments.tasks.split(","))))
    tasks = list(docs)
    examples, unchecked, sizes = build_examples(docs)
    index = build_index(examples, tasks)
    log.info("%d benchmark examples, %d distinct %d-grams", len(examples), len(index), NGRAM)

    started = time.monotonic()
    last = [started]

    def progress(done: int, total: int, million_chars: float) -> None:
        now = time.monotonic()
        if now - last[0] >= 60 or done == total:
            last[0] = now
            log.info("%d/%d units, %.0f M chars, %.1f M chars/s", done, total, million_chars,
                     million_chars / max(now - started, 1e-9))

    results = scan(units, index, arguments.workers, progress)
    report = summarize(examples, unchecked, tasks, results, ranges, arguments.flag_share)
    header = {"run": settings.run_name, "checkpoint": str(checkpoint.resolve()), "step": metadata.step,
              "scope": arguments.scope, "dataset_config": settings.dataset_config,
              "scan_minutes": round((time.monotonic() - started) / 60, 1)}
    clean, contaminated, shares = sample_lists(examples, sizes, results, arguments.flag_share)
    no_clean = sorted(set(sizes) - set(clean))
    if skipped:
        header["skipped tasks (could not load)"] = "; ".join(f"{task} ({reason[:120]})" for task, reason in skipped.items())
    if no_clean:
        header["leaf tasks with no clean example (cannot run clean)"] = ", ".join(no_clean)
    write_sample_lists(output, clean, contaminated, shares, {
        "flag_share": arguments.flag_share, "ngram": NGRAM, "checkpoint": header["checkpoint"], "step": metadata.step,
        "scope": arguments.scope, "dataset_config": settings.dataset_config,
    })
    (output / "contamination.json").write_text(json.dumps({"header": header, **report}, indent=2) + "\n", encoding="utf-8")
    (output / "contamination.md").write_text(markdown_report(report, header), encoding="utf-8")
    log.info("report: %s", output / "contamination.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
