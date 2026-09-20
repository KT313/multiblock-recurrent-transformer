# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Per-source conversion, token batches, counters, and progress during a download pass."""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.conversation_format import fit_conversation, validate_messages
from data_preparation.lib.sources.instruction_messages import ExcludedConversation
from data_preparation.lib.log import get_logger
from data_preparation.lib.progress import Progress
from data_preparation.lib.sources.conversations import OrphanAssistantOpening
from data_preparation.lib.sources.converters import Filter, expected_format, get_converter, text_or_empty
from data_preparation.lib.sources.loaders import Row
from data_preparation.lib.stages.row_pipeline import instruct_text
from data_preparation.lib.stages.truncation import NUMBER_OF_SPECIAL_TOKENS
from data_preparation.lib.storage.raw_folder import RawFolder, RowProgress

if TYPE_CHECKING:
    from data_preparation.lib.stages.download import TokenCounter

log = get_logger("data_preparation.lib.stages.download")

UNBOUNDED_COUNT = 2**62  # instruct downloads stop after their kept-row target
StoredRow = tuple[Row, RowProgress]  # stored row and the fetch progress immediately after it
TOKEN_BATCH = 2048  # rows per tokenizer call (measured 2026-09-17: 256 rows leave most of a big Rust pool idle, see tokenizer_pool.py)
MAX_CONSECUTIVE_MALFORMED = 10  # consecutive schema errors that fail the download
MALFORMED_WARNINGS_PER_INCREMENT = 100  # subsequent malformed warnings use DEBUG


@dataclass
class _IncrementCounters:
    """
    What one download increment did so far (updated while :func:`_fetch` runs, read back at the end).

    Two threads write here, each its own fields: the fetch thread `consumed` and `skipped_malformed`, the token
    worker `kept` and `dropped_too_long` (they follow the tokenizer); either may read the other's. `exhausted` is
    set after the worker joined.
    """

    consumed: int = 0  # source rows the loader yielded (the loader offset advances by this much)
    kept: int = 0  # rows written to disk
    skipped_malformed: int = 0  # instruct rows whose filter/converter raised ValueError or left out instruction / output
    dropped_too_long: int = 0  # instruct rows with more than `dataset_max_sequence_length` tokens
    chat: Counter[str] = field(default_factory=Counter)  # per-pass diagnostics, not durable offsets
    exhausted: bool = False  # the loader ran dry, or check_limit was reached


class _TokenStep:
    """
    The token step of a download, in two halves used from two threads. The fetch thread feeds it row by row:
    add(row, progress) returns a full batch of :data:`TOKEN_BATCH` rows (else []), take() whatever is
    buffered; every row comes with the :class:`RowProgress` right after it. The token worker calls start(batch)
    on those batches (the tokenizer's part: a future, computed in a tokenizer process when the counter has a
    pool, tokenizer_pool.py, else right there) and finish(batch, result) in submission order, which gives the
    rows ready to store and advances the drop counters; tokenize(batch) is both in one call.

    Every stored tokens counts the text plus :data:`NUMBER_OF_SPECIAL_TOKENS` (the BOS and EOS the trainer adds), the one
    place the specials enter a count. Pretrain rows: text_field is truncated (truncation.py) so that this sum is at
    most max_tokens. Instruct rows: tokens counts the trainer's text (row_pipeline.instruct_text) uncapped; a row
    over max_tokens is dropped (counters.dropped_too_long), never cut, and every stored row's progress carries the
    drop count of the rows before it (exact per row, so a resume never double counts).
    """

    def __init__(self, source: SourceConfig, counter: TokenCounter, max_tokens: int, counters: _IncrementCounters) -> None:
        if max_tokens < NUMBER_OF_SPECIAL_TOKENS:
            raise ValueError(f"dataset_max_sequence_length {max_tokens} leaves no room for the {NUMBER_OF_SPECIAL_TOKENS} special tokens of a row")
        self._counter = counter
        self._max_tokens = max_tokens
        self._counters = counters
        self._is_instruct = source.kind == "instruct"
        self._messages = source.instruction_format == "messages"
        if self._messages:
            tokenizer = counter.chat_tokenizer
            if any(token_id is None or not 0 <= token_id < tokenizer.vocab_size for token_id in (tokenizer.bos_id, tokenizer.eos_id)):
                raise ValueError("message conversations require BOS and EOS IDs in the tokenizer base vocabulary")
        self._text_field = source.text_field
        self._batch: list[StoredRow] = []

    @property
    def pending(self) -> int:
        """
        Rows buffered for the next tokenizer call.
        """

        return len(self._batch)

    def add(self, row: Row, progress: RowProgress) -> list[StoredRow]:
        """
        Buffer row; a full batch (:data:`TOKEN_BATCH` rows) is released, untokenized.
        """

        self._batch.append((row, progress))
        return self.take() if len(self._batch) >= TOKEN_BATCH else []

    def take(self) -> list[StoredRow]:
        """
        The buffered rows (possibly none), untokenized; the buffer is empty afterwards.
        """

        batch, self._batch = self._batch, []
        return batch

    @property
    def parallel_batches(self) -> int:
        """
        Batches the token worker keeps in flight: one per tokenizer process plus one queued, or one in-process.
        """

        pool = self._counter.pool
        return 1 if pool is None else pool.processes + 1

    def start(self, batch: list[StoredRow]) -> Future[Any] | None:
        """
        The tokenizer's part of batch: a future of the texts' (prefix, count) pairs (pretrain) or counts
        (instruct), computed in a tokenizer process when the counter has a pool, else right here before this
        returns. None for an empty batch and for message conversations, which finish fits per row.
        """

        if not batch or self._messages:
            return None
        pool = self._counter.pool
        if self._is_instruct:
            texts = [instruct_text(row) for row, _ in batch]
            return _completed(self._counter.count_many(texts)) if pool is None else pool.count(self._counter.tokenizer_dir, texts)
        texts = [row[self._text_field] for row, _ in batch]
        max_tokens = self._max_tokens - NUMBER_OF_SPECIAL_TOKENS
        return _completed(self._counter.truncate_many(texts, max_tokens)) if pool is None else pool.truncate(self._counter.tokenizer_dir, texts, max_tokens)

    def finish(self, batch: list[StoredRow], result: Any) -> list[StoredRow]:
        """
        The rows of batch ready to store, given the result of start's future (None where start gave none):
        pretrain rows truncated and counted, instruct rows counted or dropped. Batches finish in submission order:
        the drop counter every stored row's progress carries advances here.
        """

        if not batch:
            return []
        if self._messages:
            return self._fit_message_rows(batch)
        return self._drop_long_instruct_rows(batch, result) if self._is_instruct else self._truncate_pretrain_rows(batch, result)

    def tokenize(self, batch: list[StoredRow]) -> list[StoredRow]:
        """
        start and finish in one call: the rows of batch ready to store.
        """

        future = self.start(batch)
        return self.finish(batch, None if future is None else future.result())

    def _fit_message_rows(self, batch: list[StoredRow]) -> list[StoredRow]:
        stored: list[StoredRow] = []
        for row, before in batch:
            encoded = fit_conversation(row["messages"], self._counter.chat_tokenizer, self._max_tokens)
            self._counters.chat["trimmed_trailing_user"] += encoded.trimmed_user
            self._counters.chat["removed_exchanges"] += encoded.removed_exchanges
            if not encoded.messages:
                self._counters.chat["no_fitting_exchange"] += 1
                self._counters.dropped_too_long += 1
                continue
            row = {"messages": encoded.messages, "exchange_ends": encoded.exchange_ends, "tokens": len(encoded.ids)}
            stored.append((row, RowProgress(before.consumed, before.skipped_malformed, self._counters.dropped_too_long)))
        return stored

    def _truncate_pretrain_rows(self, batch: list[StoredRow], truncated: list[tuple[str, int]]) -> list[StoredRow]:
        for (row, _), (cut, tokens) in zip(batch, truncated, strict=True):
            row[self._text_field] = cut
            row["tokens"] = tokens + NUMBER_OF_SPECIAL_TOKENS
        return batch

    def _drop_long_instruct_rows(self, batch: list[StoredRow], counts: list[int]) -> list[StoredRow]:
        stored: list[StoredRow] = []
        for (row, before), count in zip(batch, counts, strict=True):
            tokens = count + NUMBER_OF_SPECIAL_TOKENS
            if tokens > self._max_tokens:
                self._counters.dropped_too_long += 1
                continue
            row["tokens"] = tokens
            stored.append((row, RowProgress(before.consumed, before.skipped_malformed, self._counters.dropped_too_long)))
        return stored


def _completed(value: Any) -> Future[Any]:
    """
    A future that already holds value (the in-process token step, computed before start returns).
    """

    future: Future[Any] = Future()
    future.set_result(value)
    return future


class MalformedSourceError(RuntimeError):
    """
    :data:`MAX_CONSECUTIVE_MALFORMED` source rows in a row came out malformed: the source's `fields` / `converter`
    does not fit the rows, so the download fails with what the converter expects and what the rows looked like
    (column name -> value type name, and the converter's reason). It propagates like any download failure: the
    shards published so far stay on disk, the runner reports the job failed.
    """

    def __init__(self, name: str, *, expected: str, samples: list[tuple[dict[str, str], str]]) -> None:
        found = " | ".join(f"{columns} ({reason})" for columns, reason in samples)
        super().__init__(
            f"{name}: {len(samples)} consecutive rows could not be converted; expected format: {expected}; "
            f"formats of the last {len(samples)} rows: {found}"
        )
        self.name = name
        self.expected = expected
        self.samples = samples


def _row_format(raw: Row) -> dict[str, str]:
    """
    The shape of a source row for the malformed-row error: column name -> type name of its value.
    """

    return {str(column): type(value).__name__ for column, value in raw.items()}


@dataclass
class _Increment:
    """
    One source's part of a download pass: what it still wants, how a source row becomes a stored row, and what
    the pass did for it so far (:attr:`counters`).

    :attr:`consecutive_malformed` and :attr:`last_malformed` (the shapes and reasons of the last
    :data:`MAX_CONSECUTIVE_MALFORMED` malformed rows) belong to the fetch thread, like :meth:`convert`.
    """

    name: str
    source: SourceConfig
    folder: RawFolder
    rows_to_keep: int  # rows to keep in this pass
    max_consume: int | None  # source rows this pass may consume (`check_limit` less the offset reached); None = no bound
    token_step: _TokenStep
    counters: _IncrementCounters
    converter: Callable[[Row], Row] | None  # instruct sources: the standardizing converter (None: rows are standard already)
    row_filter: Filter | None
    passive: bool = False  # stores whatever rows the pass hands it (a group member past its target, or a language without a source); never bounds the pass
    submitted: int = 0  # rows handed to the token worker (fetch thread)
    settled: int = 0  # rows the token worker stored or dropped (worker thread)
    consecutive_malformed: int = 0  # malformed rows since the last converted one (a filter rejection is neither)
    last_malformed: deque[tuple[dict[str, str], str]] = field(default_factory=lambda: deque(maxlen=MAX_CONSECUTIVE_MALFORMED))

    @property
    def is_instruct(self) -> bool:
        return self.source.kind == "instruct"

    @property
    def in_flight(self) -> int:
        """
        Rows handed to the token worker whose fate (stored or dropped) is not settled yet.
        """

        return self.submitted - self.settled

    @property
    def loader_count(self) -> int:
        """
        What the loader is asked for. Pretrain rows are all kept: rows_to_keep (or the consume budget when that is
        smaller); a remote loader may still finish its row group beyond it. Instruct rows may be dropped by the
        filter, the converter or the token step: everything within the budget, or without bound.
        """

        if self.passive:
            return 0
        if self.is_instruct:
            return UNBOUNDED_COUNT if self.max_consume is None else self.max_consume
        return self.rows_to_keep if self.max_consume is None else min(self.rows_to_keep, self.max_consume)

    def credit(self, stored: int) -> int:
        """
        How many of stored rows just appended (counters.kept already advanced) the bar counts: the ones up to
        :attr:`rows_to_keep`. None for a passive increment, none past the target (a member reading on for the
        other languages, a loader finishing a remote row group).
        """

        if self.passive:
            return 0
        kept = self.counters.kept
        return min(kept, self.rows_to_keep) - min(kept - stored, self.rows_to_keep)

    @property
    def done(self) -> bool:
        """
        No more source rows are taken: the consume budget is spent (check_limit bounds the source rows
        consumed, whatever the loader yields beyond its count), or an instruct source kept its rows_to_keep rows.
        """

        if self.passive:
            return False
        if self.max_consume is not None and self.counters.consumed >= self.max_consume:
            return True
        return self.is_instruct and self.counters.kept >= self.rows_to_keep

    def convert(self, name: str, raw: Row) -> Row | None:
        """
        The row to store for source row raw, or None when the filter rejects it or the filter/converter finds it
        malformed (ValueError: logged at WARNING, counted in skipped_malformed). A converted row ends a run of
        malformed ones; a filter rejection neither extends nor ends it. The :data:`MAX_CONSECUTIVE_MALFORMED`-th
        malformed row in a row raises :class:`MalformedSourceError`.

        Orphan GPT openings are counted and warned about, but neither extend nor reset the schema-error streak.
        """

        if not self.is_instruct:
            return text_row(self.source, raw, name)
        try:
            if self.row_filter is not None and not self.row_filter(raw):
                if self.source.instruction_format == "messages":
                    self.counters.chat["quality_filter"] += 1
                return None
            if self.source.instruction_format == "messages":
                converted = self.converter(raw) if self.converter else raw
                row = {"messages": validate_messages(converted.get("messages"))}
            else:
                row = _instruct_row(raw, self.converter)
        except ExcludedConversation as err:
            self.counters.chat[err.reason] += 1
            return None
        except OrphanAssistantOpening as err:
            # Preserve the existing manifest counter/offset contract without treating a known unsuitable
            # conversation as evidence of a changed source schema. Never search later turns for another pair.
            self.counters.skipped_malformed += 1
            log.warning("%s: orphan assistant opening skipped (%s)", name, err)
            return None
        except ValueError as err:
            self._malformed(name, raw, str(err))
            return None
        self.consecutive_malformed = 0
        self.last_malformed.clear()
        return row

    def _malformed(self, name: str, raw: Row, reason: str) -> None:
        """
        Count, log and remember a malformed row; fail the download once :data:`MAX_CONSECUTIVE_MALFORMED` came in a row.
        """

        self.counters.skipped_malformed += 1
        self.consecutive_malformed += 1
        self.last_malformed.append((_row_format(raw), reason))
        level = logging.WARNING if self.counters.skipped_malformed <= MALFORMED_WARNINGS_PER_INCREMENT else logging.DEBUG
        log.log(level, "%s: malformed row skipped (%s)", name, reason)
        if self.consecutive_malformed >= MAX_CONSECUTIVE_MALFORMED:
            raise MalformedSourceError(name, expected=expected_format(self.source), samples=list(self.last_malformed))


def text_row(source: SourceConfig, row: Row, name: str) -> Row:
    """
    The pretrain row to store for a source row: the converter (if any) applied, then text_field alone, as a
    string (None becomes "", which the build's min_chars filter drops). Only the Hub file reader projects
    columns; hf_split and hf_stream deliver every source column, and a surplus column of varying type would
    fail the shard write, a nested one bloat it.
    """

    converter = get_converter(source)
    if converter is not None:
        row = converter(row)
    if source.text_field not in row:
        raise ValueError(f"{name}: row has no {source.text_field!r} column; columns: {sorted(row)}")
    return {source.text_field: text_or_empty(row[source.text_field])}


def _instruct_row(raw: Row, converter: Callable[[Row], Row] | None) -> Row:
    """
    The standardized {instruction, input, output} row for raw, every field a string (None becomes "").

    A malformed row raises ValueError, which the caller counts and skips: the converter's own, or a result
    without instruction / output.
    """

    row = converter(raw) if converter is not None else raw
    missing = [key for key in ("instruction", "output") if key not in row]
    if missing:
        raise ValueError(f"row has no {missing} column; columns: {sorted(row)}")
    return {"instruction": text_or_empty(row["instruction"]), "input": text_or_empty(row.get("input")), "output": text_or_empty(row["output"])}


class _DownloadPostfix:
    """
    The download bar's postfix: source rows consumed towards a target, surplus rows (stored, but not counted by
    the bar: passive increments' rows and rows past a target), current repo file (refreshed sparsely).
    """

    def __init__(self, bar: Progress) -> None:
        self._bar = bar
        self._values: dict[str, Any] = {"consumed": 0}

    def surplus(self, total: int) -> None:
        """
        Record the running count of surplus rows (passive increments' rows and rows consumed past a target); the
        bar is refreshed every 100 rows.
        """

        self._values["surplus"] = total
        if total % 100 == 0:
            self._refresh()

    def on_file(self, file: str) -> None:
        """
        Loader callback: a new repo file is being read.
        """

        self._values["file"] = file.rsplit("/", 1)[-1]
        self._refresh()

    def consumed(self, total: int) -> None:
        """
        Record the running count of source rows consumed towards a target; the bar is refreshed every 100 rows.
        """

        self._values["consumed"] = total
        if total % 100 == 0:
            self._refresh()

    def _refresh(self) -> None:
        self._bar.set_postfix(self._values, refresh=False)
