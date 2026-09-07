# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The build step for both source kinds: sources/<name>/raw -> processed/<name>.

pretrain rows go through the length filter (min_chars; the upper bound is the token truncation at download)
-> quality filter -> decontamination -> exact dedup (hash column, first occurrence wins); instruct rows
(converter and filter already applied at download) get their input inversions -> empty-field and over-cap removal
-> exact dedup over instruction\\ninput\\noutput. Both kinds publish layout.processed_columns(kind); the
tokens of a pretrain row is the stored raw count clamped to the current dataset_max_sequence_length.

Two write modes:

* per raw shard, resumable (pretrain sources without shuffle): the survivors of one raw shard are published
  before the next raw shard is read and the manifest records the raw shard as covered (input_shards), so a
  failure or a stop request (checked between raw shards) loses at most one raw shard of work and the next call
  resumes behind the last covered one. The dedup filter (:class:`SeenDocuments`, a Bloom filter under
  dedup.bloom_memory_mb) is refilled from the hash column of the processed shards at the start of every
  build, so the rows kept are exactly those of one full pass.
* all at once (config.shuffle_of(name), the default for instruct sources, and dedup.mode: minhash): every
  raw shard is read, the survivors are shuffled with random.Random(source.seed) (or, for minhash, run through
  the LSH index), written into processed/<name>.tmp and swapped into place rename-aside
  (:func:`_swap_into_place`; the repair step finishes an interrupted swap). Why shuffle: the training loader reads
  a source's shards in order and only mixes between sources; instruct repositories are sorted by task, so without
  a shuffle the model would see one task for thousands of steps. Instruct sources are small, so rebuilding them
  whole is cheap; a top-up rebuilds the folder from every raw shard.

A build is a no-op when the processed manifest is current and covers every raw shard. It starts from a fresh
manifest, deleting the whole processed folder first, when the manifest is stale, when the covered shards are no
longer a prefix of the raw shards or when the folder predates the current columns. A raw folder that is exhausted
with zero shards still gets a (zero-shard) processed manifest, so the source counts as complete.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.pool
import random
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from data_preparation.dataset_config import DatasetConfig, DecontaminationConfig, SourceConfig
from data_preparation.layout import DatasetLayout, processed_columns
from data_preparation.lib.abort import StopCheck, check_stop
from data_preparation.lib.build.assessment import ProcessedAssessment, assess_processed_folder
from data_preparation.lib.iteration import chunks
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.stages.benchmarks import load_benchmark_ngrams
from data_preparation.lib.stages.exact_dedup import SeenDocuments, stored_hashes, text_hash64
from data_preparation.lib.stages.fuzzy_dedup import fuzzy_dedup
from data_preparation.lib.stages.row_pipeline import (
    check_contamination,
    check_quality,
    create_input_inversion,
    has_required_fields,
    instruct_text,
    preprocess_batch,
)
from data_preparation.lib.stages.download import (
    DEFAULT_SHARD_SIZE,
    TokenCounter,
    inspect_raw,
    new_manifest,
)
from data_preparation.lib.stages.truncation import NUMBER_OF_SPECIAL_TOKENS
from data_preparation.lib.storage.manifest import shard_list, Manifest, ShardInfo
from data_preparation.lib.storage.parquet import publish_shard, shard_name
from data_preparation.lib.ui.dashboard import progress

log = get_logger(__name__)

Row = dict[str, Any]
INSTRUCT_RAW_COLUMNS: tuple[str, ...] = ("instruction", "input", "output", "tokens")


# --- build_source ------------------------------------------------------------------------------------------------------


def build_source(
    config: DatasetConfig,
    name: str,
    layout: DatasetLayout,
    *,
    pass_workers: int = 1,
    shard_size: int = DEFAULT_SHARD_SIZE,
    should_stop: StopCheck | None = None,
    rows_target: int | None = None,
) -> Manifest:
    """
    Turn the raw shards of source name into processed/<name> (see the module docstring) and return the
    processed manifest. pass_workers sizes the spawn process pool of the optional cleaning passes
    (decontamination / minhash; 1 = in-process). rows_target caps a per-raw-shard build: it stops after the raw
    shard that brings the processed rows to that many (the planner's rows_sufficient; raw rows past the budget
    stay unbuilt until a larger target asks for them); an all-at-once build ignores it. Raises FileNotFoundError
    without a current raw manifest (run the download first).
    """

    source = config.sources[name]
    processing = config.source_processing(name)
    source_hash = config.processed_hash(name)
    raw_dir, processed_dir = layout.raw_dir(name), layout.processed_dir(name)
    raw = inspect_raw(config, name, layout).current_manifest
    if raw is None:
        raise FileNotFoundError(f"{name}: no current raw manifest in {raw_dir}; run the download stage first")
    all_at_once = config.shuffle_of(name) or (source.kind == "pretrain" and processing.dedup.mode == "minhash")
    assessment = assess_processed_folder(config, name, processed_dir, shard_list(raw.shards), check_files=False)
    if assessment.problem == "unreadable_manifest":
        # never deleted here, on either path: the repair step does that, after the user confirmed
        raise RuntimeError(
            f"{name}: {processed_dir / 'MANIFEST.json'} cannot be parsed; the repair step deletes the folder after "
            "confirmation (`prepare` asks, `--yes` answers), or fix or delete it by hand"
        )

    if all_at_once:
        if assessment.problem == "none" and assessment.manifest is not None:
            return assessment.manifest  # built from exactly the current raw shards
        output = ProcessedOutput(_fresh_manifest(config, name, source_hash), _temporary_dir(processed_dir), is_new=True)
        pending = list(raw.shards)
    else:
        output = ProcessedOutput.resume(config, name, source_hash, processed_dir, assessment)
        pending = raw.shards[output.covered() :]
        if not pending:
            if output.is_new:
                output.save([])  # an exhausted raw folder with zero shards still gets its processed manifest
            return output.manifest

    stats = output.manifest.stats
    seen = SeenDocuments(memory_mb=processing.dedup.bloom_memory_mb) if processing.dedup.mode != "none" else None
    if seen is not None:
        if not output.is_new:
            seen.add_all(output.stored_hashes())
        # the raw row count is the honest upper bound of what this build can insert
        log.info("%s: %s", name, seen.describe(raw.rows()))
    log.info("%s: building %d raw shard(s) (%d already covered) -> %s", name, len(pending), output.covered(), processed_dir)

    # per-shard builds check the stop request after every published shard; an all-at-once build reads every raw
    # shard before it writes anything, so its readers check between raw shards instead
    pipeline = RowPipeline(config, name, layout, pass_workers, shard_size, stats, seen=seen, should_stop=should_stop if all_at_once else None)
    pending_rows = sum(shard.rows for shard in pending)
    with pipeline, progress(total=pending_rows, desc=name, unit="row", panel="builds") as bar:
        pipeline.bar = bar
        if all_at_once:
            _build_all_at_once(pipeline, raw_dir, raw, output, processed_dir, shard_size, should_stop)
        else:
            _build_per_raw_shard(pipeline, raw_dir, raw, pending, output, shard_size, should_stop, rows_target)
    log.info("%s: %d processed rows, %s tokens", name, output.manifest.rows(), output.manifest.tokens())
    return output.manifest


def _build_per_raw_shard(
    pipeline: RowPipeline,
    raw_dir: Path,
    raw: Manifest,
    pending: list[ShardInfo],
    output: ProcessedOutput,
    shard_size: int,
    should_stop: StopCheck | None,
    rows_target: int | None = None,
) -> None:
    """
    One raw shard at a time: its survivors become the next processed shard(s), published and recorded (with the
    raw shard as covered) before the next raw shard starts; the stop request is checked in between. With
    rows_target the loop ends once the processed rows reach it (checked before every raw shard).
    """

    first_row_index = sum(shard.rows for shard in raw.shards[: output.covered()])
    for index, shard in enumerate(pending, start=1):
        if rows_target is not None and output.manifest.rows() >= rows_target:
            log.info(
                "%s: %d processed rows serve the budget of %d; %d raw shard(s) left unbuilt",
                raw.source, output.manifest.rows(), rows_target, len(pending) - index + 1,
            )
            return
        if pipeline.bar is not None:
            pipeline.bar.set_postfix({"shard": f"{index}/{len(pending)}"}, refresh=False)
        pipeline.stats["input_rows"] += shard.rows
        survivors = list(pipeline.run(raw_dir, [shard], first_row_index=first_row_index))
        output.publish(survivors, shard_size)
        output.save([*output.manifest.input_shards, [shard.name, shard.rows]])
        first_row_index += shard.rows
        check_stop(should_stop)


def _build_all_at_once(
    pipeline: RowPipeline,
    raw_dir: Path,
    raw: Manifest,
    output: ProcessedOutput,
    processed_dir: Path,
    shard_size: int,
    should_stop: StopCheck | None,
) -> None:
    """
    Every raw shard through the pipeline (plus fuzzy dedup in minhash mode), shuffled when the source asks for
    it, written into output.directory (the .tmp sibling) and swapped over processed_dir rename-aside
    (:func:`_swap_into_place`).
    """

    pipeline.stats["input_rows"] += raw.rows()
    rows = pipeline.run(raw_dir, list(raw.shards), first_row_index=0)
    if pipeline.kind == "pretrain" and pipeline.processing.dedup.mode == "minhash":
        rows = fuzzy_dedup(rows, pipeline.processing.dedup, pipeline.stats["dedup"], pipeline.pass_workers)
    survivors = list(rows)
    if output.manifest.shuffled:
        random.Random(pipeline.source.seed).shuffle(survivors)
    check_stop(should_stop)

    temporary = output.directory
    if temporary.exists():
        log.warning("removing leftover %s of an interrupted build", temporary)
        shutil.rmtree(temporary)
    output.publish(survivors, shard_size)
    output.save(shard_list(raw.shards))
    _swap_into_place(temporary, processed_dir)
    output.directory = processed_dir


def _swap_into_place(temporary: Path, processed_dir: Path) -> None:
    """
    Replace processed_dir by the complete temporary folder without a moment where neither exists: the
    old folder steps aside (processed/<name>.old), the new one is renamed into place, and only then is anything
    deleted. A crash between the renames leaves the .old next to the complete .tmp (the repair step
    finishes the swap), one after them leaves the new folder in place next to a stale .old (the repair step
    removes it).
    """

    old = _old_dir(processed_dir)
    if old.exists():
        shutil.rmtree(old)  # leftover of an earlier crashed swap; the folder that replaced it is in place or in `temporary`
    if processed_dir.exists():
        processed_dir.rename(old)
    temporary.rename(processed_dir)
    if old.exists():
        shutil.rmtree(old)


def _temporary_dir(processed_dir: Path) -> Path:
    """
    processed/<name>.tmp: where an all-at-once build writes before the swap into place.
    """

    return processed_dir.with_name(processed_dir.name + ".tmp")


def _old_dir(processed_dir: Path) -> Path:
    """
    processed/<name>.old: where :func:`_swap_into_place` parks the folder it replaces until the new one is
    in place.
    """

    return processed_dir.with_name(processed_dir.name + ".old")


# --- the processed folder ------------------------------------------------------------------------------------------------


@dataclass
class ProcessedOutput:
    """
    The processed folder of one build: its manifest, the directory shards are published into (the final folder,
    or the .tmp sibling of an all-at-once build) and whether the manifest was created by this call (a new
    manifest is saved even when nothing is appended, so an empty source still counts as built).
    """

    manifest: Manifest
    directory: Path
    is_new: bool

    @classmethod
    def resume(cls, config: DatasetConfig, name: str, source_hash: str, processed_dir: Path, assessment: ProcessedAssessment) -> ProcessedOutput:
        """
        The stored manifest if new raw shards can be appended to it (the shared verdict says built or behind
        raw: current hash, expected columns, covered shards a prefix of the raw shards); otherwise a fresh one, and
        the folder is deleted first, so no shard of the previous build survives unlisted. A manifest that cannot be
        parsed never gets here (:func:`build_source` refuses it before choosing a path).
        """

        if assessment.problem in ("none", "behind_raw") and assessment.manifest is not None:
            return cls(assessment.manifest, processed_dir, is_new=False)
        if assessment.problem != "absent":
            log.warning("%s: processed %s, rebuilding everything", name, assessment.reason)
        if processed_dir.exists():
            log.info("%s: removing %s before the rebuild", name, processed_dir)
            shutil.rmtree(processed_dir)
        return cls(_fresh_manifest(config, name, source_hash), processed_dir, is_new=True)

    def covered(self) -> int:
        """
        Raw shards the manifest already covers.
        """

        return len(self.manifest.input_shards)

    def stored_hashes(self) -> Iterator[int]:
        """
        The exact-dedup keys of every processed row on disk, in manifest order (refills the dedup filter).
        """

        return stored_hashes(self.directory / shard.name for shard in self.manifest.shards)

    def publish(self, rows: list[Row], shard_size: int) -> None:
        """
        Append rows as shard(s) of at most shard_size rows, each recorded in the manifest.
        """

        for start in range(0, len(rows), shard_size):
            chunk = rows[start : start + shard_size]
            path = publish_shard(pa.Table.from_pylist(chunk), self.directory / shard_name(len(self.manifest.shards)))
            self.manifest.add_shard(path.name, len(chunk), sum(int(row["tokens"]) for row in chunk))

    def save(self, covered: list[list[Any]]) -> None:
        self.manifest.input_shards = list(covered)
        self.manifest.save(self.directory)
        self.is_new = False


def _fresh_manifest(config: DatasetConfig, name: str, source_hash: str) -> Manifest:
    source = config.sources[name]
    processing = config.source_processing(name)
    manifest = new_manifest(config, name, source_hash, "processed", tokens=True)
    stats: dict[str, Any] = {
        "input_rows": 0,
        "dedup": {"mode": processing.dedup.mode, "duplicates_removed": 0},
    }
    if source.kind == "pretrain":
        stats["length_filter"] = {"input_samples": 0, "removed_too_short": 0, "removed_invalid": 0, "output_samples": 0}
        stats["quality_filter"] = {"enabled": processing.quality_filter, "filtered_count": 0, "rejection_reasons": {}}
        stats["decontamination"] = {
            "enabled": processing.decontamination.enabled, "contaminated_count": 0, "contaminated_by_benchmark": {},
        }  # fmt: skip
    else:
        stats["inverted"] = 0  # rows replaced by their input inversion (`source.input_inversions` share, seeded per row)
        stats["removed_empty"] = 0  # rows without instruction or output after stripping
        stats["removed_too_long"] = 0  # rows over `dataset_max_sequence_length` tokens (a safety net; the download already drops them)
    manifest.columns = list(processed_columns(source.kind))
    manifest.shuffled = config.shuffle_of(name)
    manifest.shuffle_seed = source.seed
    manifest.stats = stats
    return manifest


# --- the row pipeline ----------------------------------------------------------------------------------------------------


class RowPipeline:
    """
    The row pipeline of one build call, reusable per raw shard: the dedup filter, the statistics, the token
    counter (instruct inversions) and the decontamination worker pool (a with resource) persist across run
    calls. Pretrain: length filter -> quality filter -> decontamination -> hash -> exact dedup; instruct: input
    inversions -> empty / over-cap removal -> hash -> exact dedup. The filters run before the dedup, so the hashes on
    disk are exactly the dedup's "seen" set and an incremental build keeps the same rows as a full pass.
    """

    def __init__(
        self,
        config: DatasetConfig,
        name: str,
        layout: DatasetLayout,
        pass_workers: int,
        batch_size: int,
        stats: dict[str, Any],
        *,
        seen: SeenDocuments | None,
        should_stop: StopCheck | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.layout = layout
        self.source: SourceConfig = config.sources[name]
        self.kind = self.source.kind
        self.processing = config.source_processing(name)
        self.dataset_max_sequence_length = config.dataset_max_sequence_length
        self.pass_workers = pass_workers
        self.batch_size = batch_size
        self.stats = stats
        self.seen = seen
        self.should_stop = should_stop
        self.decontaminator = Decontaminator(self.processing.decontamination, pass_workers, layout, stats.get("decontamination", {}))
        self.bar: Progress | None = None
        self._token_counter: TokenCounter | None = None  # instruct inversions only, created on first use

    def __enter__(self) -> RowPipeline:
        if self.kind == "pretrain":
            self.decontaminator.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.decontaminator.__exit__(exc_type, exc, tb)

    def run(self, raw_dir: Path, shards: list[ShardInfo], *, first_row_index: int) -> Iterator[Row]:
        """
        The processed rows of shards (lazy). first_row_index is the global raw row index of the first row
        of shards[0] (instruct inversions are keyed by it).
        """

        if self.kind == "pretrain":
            rows = self._pretrain_rows(raw_dir, shards)
        else:
            rows = self._instruct_rows(raw_dir, shards, first_row_index)
        if self.seen is not None:  # dedup on (minhash = this cheap exact pass first, then fuzzy, all at once)
            rows = _exact_dedup(rows, self.seen, self.stats["dedup"])
        return rows

    # --- pretrain --------------------------------------------------------------------------------------------------

    def _pretrain_rows(self, raw_dir: Path, shards: list[ShardInfo]) -> Iterator[Row]:
        processing = self.processing
        rows = self._length_filtered_rows(raw_dir, shards)
        if processing.quality_filter:
            rows = _quality_filter(rows, self.stats["quality_filter"])
        if processing.decontamination.enabled:
            rows = self.decontaminator(rows)
        normalize = processing.dedup.normalize
        for row in rows:
            # stored counts are clamped to the current cap: lowering `dataset_max_sequence_length` after the download never
            # re-downloads (the raw manifest's `truncated_at_tokens` bounds the stored texts), so a raw count may
            # exceed the cap; training cuts rows at its own length, at most dataset_max_sequence_length, anyway
            yield {
                "text": row["text"],
                "source": self.name,
                "tokens": min(int(row["tokens"]), self.dataset_max_sequence_length),
                "hash": text_hash64(row["text"], normalize),
            }

    def _length_filtered_rows(self, raw_dir: Path, shards: list[ShardInfo]) -> Iterator[Row]:
        """
        {text, tokens} rows of the raw shards through the length filter (preprocess_batch per Arrow batch:
        null / shorter than min_chars dropped); statistics summed into stats["length_filter"], the bar
        advanced per raw row.
        """

        text_field = self.source.text_field
        stats: dict[str, int] = self.stats["length_filter"]
        for shard in shards:
            check_stop(self.should_stop)
            parquet = pq.ParquetFile(raw_dir / shard.name)
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=[text_field, "tokens"]):
                kept, batch_stats = preprocess_batch(batch, text_field, self.name, self.processing.min_chars)
                for key, value in batch_stats.items():
                    stats[key] += value
                self._advance(len(batch))
                yield from kept

    # --- instruct --------------------------------------------------------------------------------------------------

    def _instruct_rows(self, raw_dir: Path, shards: list[ShardInfo], first_row_index: int) -> Iterator[Row]:
        """
        {instruction, input, output, tokens, hash} rows of the raw shards: inversions decided per row by
        random.Random(f"{seed}:{global row index}") (deterministic, independent of shard boundaries and of a
        resume), then rows without instruction / output and rows over dataset_max_sequence_length tokens dropped (an inversion
        prepends a fixed instruction, so it can push a row over the cap that fitted before).
        """

        share = self.source.input_inversions
        normalize = self.processing.dedup.normalize
        row_index = first_row_index
        for shard in shards:
            check_stop(self.should_stop)
            parquet = pq.ParquetFile(raw_dir / shard.name)
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=list(INSTRUCT_RAW_COLUMNS)):
                rows: list[Row] = batch.to_pylist()
                for row in rows:
                    row["input"] = row["input"] or ""
                if share > 0:
                    self._invert_sample(rows, row_index, share)
                row_index += len(rows)
                self._advance(len(rows))
                for row in rows:
                    if not has_required_fields(row):
                        self.stats["removed_empty"] += 1
                        continue
                    if int(row["tokens"]) > self.dataset_max_sequence_length:
                        self.stats["removed_too_long"] += 1
                        continue
                    yield {
                        "instruction": row["instruction"],
                        "input": row["input"],
                        "output": row["output"],
                        "tokens": int(row["tokens"]),
                        "hash": text_hash64(instruct_text(row), normalize),
                    }

    def _invert_sample(self, rows: list[Row], first_row_index: int, share: float) -> None:
        """
        Replace the seeded sample of rows (in place) by their input inversion and recount their tokens; rows
        the inversion leaves unchanged (empty instruction or output) are not counted.
        """

        seed = self.source.seed
        chosen: list[int] = []
        for offset, row in enumerate(rows):
            if random.Random(f"{seed}:{first_row_index + offset}").random() >= share:
                continue
            inverted = create_input_inversion(row)
            if inverted is row:
                continue
            rows[offset] = inverted
            chosen.append(offset)
        if not chosen:
            return
        counts = self._counter().count_many([instruct_text(rows[offset]) for offset in chosen])
        for offset, count in zip(chosen, counts):
            rows[offset]["tokens"] = count + NUMBER_OF_SPECIAL_TOKENS  # the download's rule: the trainer's specials included
        self.stats["inverted"] += len(chosen)

    def _counter(self) -> TokenCounter:
        if self._token_counter is None:
            self._token_counter = TokenCounter(self.config, self.layout)
        return self._token_counter

    def _advance(self, rows: int) -> None:
        if self.bar is not None:
            self.bar.update(rows)


def _exact_dedup(rows: Iterator[Row], seen: SeenDocuments, stats: dict[str, Any]) -> Iterator[Row]:
    """
    First occurrence wins: drop rows whose hash the filter has seen (the keys of every processed row on disk
    plus the rows kept earlier in this pass). A Bloom false positive drops a unique row, never keeps a duplicate.
    """

    for row in rows:
        if not seen.add_if_new(row["hash"]):
            stats["duplicates_removed"] += 1
            continue
        yield row


def _quality_filter(rows: Iterator[Row], stats: dict[str, Any]) -> Iterator[Row]:
    """
    Drop rows failing check_quality; counts the rejections per reason in stats.
    """

    for row in rows:
        passes, reason = check_quality(row["text"])
        if not passes:
            stats["filtered_count"] += 1
            _increment(stats["rejection_reasons"], reason)
            continue
        yield row


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


# --- decontamination -----------------------------------------------------------------------------------------------------

# the parameters of `_contaminated_by` in a spawn worker, set by `_init_decontamination`, the pool initializer
# (spawn children start with fresh module globals, so the n-grams are handed over as picklable init args, loaded
# once in the parent). The in-process path never touches them: several builds run in threads of one process
# (`lib/build/runner.py`), each with its own decontamination settings, so `Decontaminator` keeps its own.
_BENCHMARK_NGRAMS: dict[str, set[str]] = {}
_DECONTAM: dict[str, Any] = {}


def _init_decontamination(ngrams: dict[str, set[str]], n: int, threshold: float) -> None:
    global _BENCHMARK_NGRAMS
    _BENCHMARK_NGRAMS = ngrams
    _DECONTAM.update({"n": n, "threshold": threshold})


def _contaminated_by(text: str) -> list[str]:
    """
    Benchmarks text is contaminated by, using the worker-global n-grams of _init_decontamination.
    """

    return check_contamination(text, _BENCHMARK_NGRAMS, _DECONTAM["n"], _DECONTAM["threshold"])[1]


class Decontaminator:
    """
    Drops rows contaminated by a benchmark (counts hits per benchmark in stats); the benchmark n-grams are
    loaded once, in this process, and checked in-process (pass_workers <= 1, against the n-grams this object
    holds) or in a pool of pass_workers spawn processes that lives for the whole with block. Spawn, not fork:
    the pool is created from a build worker thread (`lib/build/runner.py` runs one build per thread), and a fork
    of a multi-threaded process can inherit a lock another thread holds mid-operation; spawn children start clean.
    """

    def __init__(self, config: DecontaminationConfig, pass_workers: int, layout: DatasetLayout, stats: dict[str, Any]) -> None:
        self.config = config
        self.pass_workers = pass_workers
        self.stats = stats
        self.cache_dir = str(layout.benchmark_cache_dir())
        self._ngrams: dict[str, set[str]] = {}
        self._pool: multiprocessing.pool.Pool | None = None

    def __enter__(self) -> Decontaminator:
        if not self.config.enabled:
            return self
        self._ngrams = load_benchmark_ngrams(list(self.config.benchmarks), self.config.ngram, self.cache_dir)
        if self.pass_workers > 1:
            init_args = (self._ngrams, self.config.ngram, self.config.threshold)
            self._pool = multiprocessing.get_context("spawn").Pool(self.pass_workers, initializer=_init_decontamination, initargs=init_args)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __call__(self, rows: Iterator[Row]) -> Iterator[Row]:
        for row, contaminated in self._checks(rows):
            if contaminated:
                self.stats["contaminated_count"] += 1
                for benchmark in contaminated:
                    _increment(self.stats["contaminated_by_benchmark"], benchmark)
                continue
            yield row

    def _checks(self, rows: Iterator[Row]) -> Iterator[tuple[Row, list[str]]]:
        """
        (row, contaminating benchmarks) for every row.
        """

        if self._pool is None:
            for row in rows:
                yield row, check_contamination(row["text"], self._ngrams, self.config.ngram, self.config.threshold)[1]
            return
        for chunk in chunks(rows, 1024):
            texts = [row["text"] for row in chunk]
            yield from zip(chunk, self._pool.map(_contaminated_by, texts, chunksize=64))
