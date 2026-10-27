# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Read-only word n-gram overlap between benchmark evaluation docs and the rows a checkpoint was trained on.

    uv run --frozen --no-sync python evaluation/contamination_check.py --config config/<run>.yaml --workers 32 \
        --output outputs/<run>/contamination

Benchmark side: the evaluation docs of every task as lm-eval scores them (test split, or validation where the task
uses it). Each string field of a doc gives its normalized 13-grams (`get_ngram_set`, the normalization of the
build's decontamination); a doc whose fields are all too short falls back to its fields joined, and a doc still
shorter than 13 words is reported as unchecked. Group tasks (mmlu) report under the requested name.

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
)
NGRAM = 13
LIKELY_SHARE = 0.5  # an example with at least this share of its n-grams in the training rows counts as contaminated
ROW_SHARE = 0.1  # a training row with more than this share of its n-grams in one task's docs (the build's threshold)
WORST_EXAMPLES = 10  # per task in the JSON, for manual inspection
PREVIEW_CHARS = 200
BATCH_ROWS = 1024

_INDEX: dict[str, int] = {}  # n-gram -> bit mask of the task indices it occurs in; set before the workers fork
_KEYS: frozenset[str] = frozenset()  # its keys, for C-level set intersection with each row's n-grams


@dataclass(frozen=True)
class Example:
    task: str
    index: int  # position in the task's evaluation docs
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


def load_lm_eval_docs(tasks: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """{task: evaluation docs} through lm-eval itself; the docs of a group's subtasks go under the group's name."""

    from lm_eval.tasks import TaskManager, get_task_dict

    def collect(value: Any) -> Iterator[dict[str, Any]]:
        if isinstance(value, Mapping):
            for item in value.values():
                yield from collect(item)
        else:
            yield from (dict(doc) for doc in value.eval_docs)

    loaded = get_task_dict(list(tasks), TaskManager())
    docs: dict[str, list[dict[str, Any]]] = {}
    for key, value in loaded.items():
        name = str(getattr(key, "group_name", key))
        docs.setdefault(name, []).extend(collect(value))
    missing = [task for task in tasks if task not in docs]
    if missing:
        raise KeyError(f"lm-eval returned no docs for {missing}; got {sorted(docs)}")
    return {task: docs[task] for task in tasks}


def build_examples(docs: Mapping[str, Sequence[Mapping[str, Any]]], n: int = NGRAM) -> tuple[list[Example], dict[str, int]]:
    """The checkable examples of every task, and per task the number too short to have an n-gram."""

    examples: list[Example] = []
    unchecked: dict[str, int] = {}
    for task, task_docs in docs.items():
        unchecked[task] = 0
        for index, doc in enumerate(task_docs):
            ngrams = example_ngrams(doc, n)
            if not ngrams:
                unchecked[task] += 1
                continue
            preview = " ".join(" ".join(doc_strings(doc)).split())[:PREVIEW_CHARS]
            examples.append(Example(task, index, ngrams, preview))
    return examples, unchecked


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


def summarize(
    examples: Sequence[Example], unchecked: Mapping[str, int], tasks: Sequence[str], results: Sequence[UnitResult],
    ranges: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    hits_by_source: dict[str, set[str]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for result in results:
        hits_by_source.setdefault(result.source, set()).update(result.hits)
        entry = sources.setdefault(result.source, {
            "rows_scanned": 0, "million_chars": 0.0, "rows_with_hit": 0, "rows_over_share": {},
            "row_range": list(ranges[result.source]),
        })
        entry["rows_scanned"] += result.rows
        entry["million_chars"] += result.million_chars
        entry["rows_with_hit"] += result.rows_with_hit
        for task, count in result.rows_over_share.items():
            entry["rows_over_share"][tasks[task]] = entry["rows_over_share"].get(tasks[task], 0) + count
    all_hits = set().union(*hits_by_source.values()) if hits_by_source else set()

    report_tasks: dict[str, dict[str, Any]] = {
        task: {"examples": unchecked.get(task, 0), "unchecked": unchecked.get(task, 0), "any_hit": 0, "likely": 0,
               "complete": 0, "likely_by_source": {}, "worst": []}
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
        by_source = {source: len(example.ngrams & hits) / len(example.ngrams)
                     for source, hits in hits_by_source.items() if example.ngrams & hits}
        for source, source_share in by_source.items():
            if source_share >= LIKELY_SHARE:
                entry["likely_by_source"][source] = entry["likely_by_source"].get(source, 0) + 1
        entry["worst"].append({"index": example.index, "share": round(share, 4), "ngrams": len(example.ngrams),
                               "sources": {source: round(value, 4) for source, value in
                                           sorted(by_source.items(), key=lambda item: -item[1])},
                               "preview": example.preview})
    for entry in report_tasks.values():
        entry["worst"] = sorted(entry["worst"], key=lambda item: -item["share"])[:WORST_EXAMPLES]
    return {"ngram": NGRAM, "likely_share": LIKELY_SHARE, "row_share": ROW_SHARE, "tasks": report_tasks,
            "sources": sources}


def markdown_report(report: Mapping[str, Any], header: Mapping[str, Any]) -> str:
    lines = ["# Benchmark contamination check", ""]
    lines += [f"- {key}: {value}" for key, value in header.items()]
    lines += [
        "",
        f"Word {report['ngram']}-grams after lowercasing and whitespace normalization. \"Likely\": at least "
        f"{report['likely_share']:.0%} of an example's n-grams occur in the scanned training rows; \"complete\": all of "
        "them. Isolated hits are mostly shared boilerplate; the likely share is the meaningful figure.",
        "",
        "## By task",
        "",
        "| Task | Examples | Unchecked (too short) | Any hit | Likely | Complete | Likely, by source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, entry in report["tasks"].items():
        checked = entry["examples"] - entry["unchecked"]

        def share(count: int, checked: int = checked) -> str:
            return f"{count} ({100 * count / checked:.1f}%)" if checked else str(count)

        by_source = ", ".join(f"{source} {count}" for source, count in
                              sorted(entry["likely_by_source"].items(), key=lambda item: -item[1])) or "-"
        lines.append(f"| {task} | {entry['examples']} | {entry['unchecked']} | {share(entry['any_hit'])} | "
                     f"{share(entry['likely'])} | {share(entry['complete'])} | {by_source} |")
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
    return parser.parse_args(argv)


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

    tasks = arguments.tasks.split(",")
    examples, unchecked = build_examples(load_lm_eval_docs(tasks))
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
    report = summarize(examples, unchecked, tasks, results, ranges)
    header = {"run": settings.run_name, "checkpoint": str(checkpoint.resolve()), "step": metadata.step,
              "scope": arguments.scope, "dataset_config": settings.dataset_config,
              "scan_minutes": round((time.monotonic() - started) / 60, 1)}
    (output / "contamination.json").write_text(json.dumps({"header": header, **report}, indent=2) + "\n", encoding="utf-8")
    (output / "contamination.md").write_text(markdown_report(report, header), encoding="utf-8")
    log.info("report: %s", output / "contamination.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
