# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Replay original text through the real fetch/token/store pipeline in an empty output tree."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.identifiers import validate_identifier
from data_preparation.lib.progress import NoProgress
from data_preparation.lib.stages.download import TokenCounter, _fetch, _finish_increment
from data_preparation.lib.stages.download_state import _DownloadPostfix, _Increment, _IncrementCounters, _TokenStep
from data_preparation.lib.stages.download_workers import _StopGate
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.raw_folder import RawFolder
from tools.data_preparation.reference import truncate_original


@dataclass(frozen=True)
class InputRow:
    source: str
    group: str
    text: str
    passive: bool = False


def load_fixture(path: Path, max_bytes: int, max_rows: int) -> list[InputRow]:
    """Read bounded JSONL: text plus optional source, group and passive fields."""
    if path.stat().st_size > max_bytes:
        raise ValueError("fixture exceeds --max-input-mb")
    rows: list[InputRow] = []
    identities: dict[str, tuple[str, bool]] = {}
    used = 0
    with path.open("rb") as handle:
        for line in handle:
            used += len(line)
            if used > max_bytes or len(rows) >= max_rows:
                raise ValueError("fixture exceeds input byte/row limits")
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError("each fixture line must be an object with string text")
            source = value.get("source", "source")
            group = value.get("group", source)
            passive = value.get("passive", False)
            if not isinstance(source, str) or not isinstance(group, str) or not isinstance(passive, bool):
                raise ValueError("source/group must be strings; passive must be boolean")
            validate_identifier(source, field="fixture source")
            validate_identifier(group, field="fixture group")
            identity = group, passive
            if identities.setdefault(source, identity) != identity:
                raise ValueError("each source must belong to one group with consistent passive status")
            rows.append(InputRow(source, group, value["text"], passive))
    if not rows:
        raise ValueError("fixture must contain rows")
    return rows


class ReplayCounter(TokenCounter):
    """Use an existing artifact read-only, without tokenizer preparation or repair."""

    def __init__(self, path: Path, original: bool) -> None:
        self.mode = "tokenizer"
        self.tokenizer_name = path.name
        self._tokenizer = SavedTokenizer(path)
        self.original = original

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        if self.original:
            return truncate_original(texts, max_tokens, self._tokenizer)
        return super().truncate_many(texts, max_tokens)

    def count_many(self, texts: list[str]) -> list[int]:
        if self.original:
            assert self._tokenizer is not None
            return [len(encoding.ids) for encoding in self._tokenizer.encode_batch(texts)]
        return super().count_many(texts)


def make_counter(path: Path, variant: str | bool) -> ReplayCounter:
    if variant in ("builtin", "rust"):
        from tools.data_preparation.native_backends import BuiltinCounter, RustCounter
        return RustCounter(path) if variant == "rust" else BuiltinCounter(path)
    return ReplayCounter(path, variant is True or variant == "original")


def run_pipeline(rows: list[InputRow], tokenizer: Path, output: Path, cap: int, jobs: int, original: str | bool) -> None:
    """Publish all input, joining every worker before returning; never reuse row dictionaries."""
    output.mkdir(exist_ok=False)
    groups: dict[str, list[InputRow]] = defaultdict(list)
    for row in rows:
        groups[row.group].append(row)

    def run_group(inputs: list[InputRow]) -> None:
        counts = Counter(row.source for row in inputs)
        counter = make_counter(tokenizer, original)
        increments: list[_Increment] = []
        gate = _StopGate(None)
        for name, count in counts.items():
            passive = next(row.passive for row in inputs if row.source == name)
            source = SourceConfig(kind="pretrain", loader="synthetic")
            manifest = Manifest(source=name, source_hash="offline-replay", stage="raw", truncated_at_tokens=cap)
            folder = RawFolder(output / name, manifest, config_cap=cap)
            counters = _IncrementCounters()
            increments.append(_Increment(name, source, folder, 0 if passive else count, None,
                                         _TokenStep(source, counter, cap, counters), counters, None, None, passive=passive))
        bar = NoProgress()
        stream = ((row.source, {"text": row.text}) for row in inputs)
        _fetch(increments, stream, bar, _DownloadPostfix(bar), 10_000, gate)
        for increment in increments:
            _finish_increment(increment.folder, increment.source, increment.counters)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # Collect every result even if an earlier job fails; executor exit joins the others.
        futures = [pool.submit(run_group, inputs) for inputs in groups.values()]
        for future in futures:
            future.result()


def summarize_output(output: Path) -> dict[str, Any]:
    """Hash ordered rows and durable metadata after timing; source completion order is irrelevant."""
    sources: dict[str, Any] = {}
    size = 0
    for directory in sorted(output.iterdir()):
        manifest = Manifest.load(directory)
        assert manifest is not None
        digest = hashlib.sha256()
        schemas: set[str] = set()
        for shard in manifest.shards:
            path = directory / shard.name
            size += path.stat().st_size
            parquet = pq.ParquetFile(path)
            schemas.add(str(parquet.schema_arrow))
            for batch in parquet.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
        sources[directory.name] = {
            "digest": digest.hexdigest(), "schemas": sorted(schemas), "rows": manifest.rows(),
            "tokens": manifest.tokens(), "rows_fetched": manifest.rows_fetched, "exhausted": manifest.exhausted,
            "skipped_malformed": manifest.skipped_malformed, "dropped_too_long": manifest.dropped_too_long,
            "shards": [asdict(shard) for shard in manifest.shards],
        }
    return {"sources": sources, "compressed_bytes": size}
