# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Source registry: how rows of a `SourceConfig` are fetched (loaders), reshaped (converters) and filtered.

Everything is looked up **by name** from the dataset config (`config/datasets/<name>.yaml`):

    LOADERS[source.loader](source, offset, count)   -> Iterator[Row]   at most `count` raw rows from row `offset`
    get_converter(source)                           -> Callable[[Row], Row] | None   (fields mapping or converter)
    get_filter(source.filter)                       -> Callable[[Row], bool]

Loaders are deterministic for a pinned `revision`: row `i` of a source is always the same row, so an incremental
download (`offset` = rows already on disk) appends exactly the next slice. `hf_split` uses `train[a:b]` slicing,
`hf_stream` streaming with `skip(offset)`, `github_code` streams codeparrot/github-code-clean and counts only rows
of the requested `language`, `local` reads a directory of parquet/jsonl files, `synthetic` generates the tiny
smoke-test data on demand (row `i` depends only on `(seed, i)`).

Converters map a raw row to the standard shape: pretrain `{"text": ...}` (only `gsm8k_question_answer`; other
pretrain sources are used as is and `text_field` is applied downstream) or instruct
`{"instruction", "input", "output"}` with string values. A missing required column raises `ValueError` listing the
row's keys — a malformed source is a failed build, never a silently smaller dataset.

`datasets` is imported lazily inside the loaders so the HF cache environment can be configured before import.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from data_preparation.lib.dataset_config import SourceConfig

Row = dict[str, Any]
Converter = Callable[[Row], Row]
Filter = Callable[[Row], bool]


class Loader(Protocol):
    def __call__(self, source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]: ...


# --- synthetic tokenizer / data constants (shared with the tiny run and its tests) ------------------------------------

SPECIALS = ["<pad>", "<bos>", "<eos>"]  # ids 0, 1, 2
N_WORD_TOKENS = 256  # tok_0..tok_255 -> ids 3..258
VOCAB_SIZE = len(SPECIALS) + N_WORD_TOKENS  # 259; the tiny model preset pads its vocab to 512
SYNTHETIC_DOC_WORDS = (64, 384)  # pretrain/holdout documents: words per row (inclusive)
SYNTHETIC_INSTRUCTION_WORDS = (4, 32)
SYNTHETIC_OUTPUT_WORDS = (8, 64)

GITHUB_CODE_DATA_FILES = "data/*.parquet"


# --- loaders ----------------------------------------------------------------------------------------------------------


def _check_offset_count(offset: int, count: int) -> None:
    if offset < 0 or count < 0:
        raise ValueError(f"offset and count must be non-negative, got offset={offset}, count={count}")


def _take(rows: Iterable[Row], count: int) -> Iterator[Row]:
    """Yield at most `count` rows as fresh dicts (stops pulling from `rows` as soon as the quota is reached)."""
    for row in islice(rows, count):
        yield dict(row)


def _load_dataset(**kwargs: Any) -> Any:
    """`datasets.load_dataset`, imported lazily (the HF cache env must be configurable before import)."""
    from datasets import load_dataset

    return load_dataset(**kwargs)


def load_hf_split(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via `split[a:b]` slicing (materialised download, deterministic order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    dataset = _load_dataset(
        path=source.hf_id,
        split=f"{source.split}[{offset}:{offset + count}]",
        revision=source.revision,
        token=token,
        **source.load_kwargs,
    )
    yield from _take(dataset, count)


def load_hf_stream(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Rows `offset..offset+count` of `hf_id` via streaming with `skip(offset)` (used for instruct sources)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    stream = _load_dataset(
        path=source.hf_id,
        split=source.split,
        streaming=True,
        revision=source.revision,
        token=token,
        **source.load_kwargs,
    )
    if offset:
        stream = stream.skip(offset)
    yield from _take(stream, count)


def iter_language(rows: Iterable[Row], language: str, limit: int, offset: int = 0) -> Iterator[Row]:
    """Yield up to `limit` rows whose `language` column equals `language`, skipping the first `offset` matches."""
    if limit <= 0:
        return
    seen = 0
    taken = 0
    for row in rows:
        if row["language"] != language:
            continue
        if seen < offset:
            seen += 1
            continue
        taken += 1
        yield dict(row)
        if taken >= limit:
            return


def load_github_code(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Stream `hf_id` (codeparrot/github-code-clean) and keep rows of `source.language`.

    `offset` counts rows *of that language* already consumed, so an incremental fetch continues where the previous
    one stopped (the stream is re-read from the start; the pinned `revision` keeps its order stable).
    """
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.language is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("github_code loader requires source.language")
    load_kwargs = dict(source.load_kwargs)
    data_files = load_kwargs.pop("data_files", GITHUB_CODE_DATA_FILES)
    stream = _load_dataset(
        path=source.hf_id,
        split=source.split,
        streaming=True,
        revision=source.revision,
        data_files=data_files,
        token=token,
        **load_kwargs,
    )
    yield from iter_language(stream, source.language, count, offset)


def list_local_files(directory: Path) -> list[Path]:
    """`*.parquet` and `*.jsonl` files directly under `directory`, sorted by name (the source's row order)."""
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in (".parquet", ".jsonl")]
    return sorted(files)


def _iter_local_file(path: Path) -> Iterator[Row]:
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches():
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def load_local(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Rows `offset..offset+count` of the parquet/jsonl files under `source.path` (files in sorted order)."""
    _check_offset_count(offset, count)
    if count == 0:
        return
    if source.path is None:  # validated by SourceConfig; repeated for the type checker
        raise ValueError("local loader requires source.path")
    directory = Path(source.path)
    if not directory.is_dir():
        raise FileNotFoundError(f"local source directory not found: {directory}")

    def all_rows() -> Iterator[Row]:
        for file in list_local_files(directory):
            yield from _iter_local_file(file)

    yield from islice(all_rows(), offset, offset + count)


def _synthetic_words(rng: random.Random, low: int, high: int) -> str:
    return " ".join(f"tok_{rng.randrange(N_WORD_TOKENS)}" for _ in range(rng.randint(low, high)))


def synthetic_row(kind: str, seed: int, index: int) -> Row:
    """Row `index` of a synthetic source; depends only on `(seed, index)` so any offset yields the same rows."""
    rng = random.Random(f"{seed}:{index}")
    if kind == "instruct":
        return {
            "instruction": _synthetic_words(rng, *SYNTHETIC_INSTRUCTION_WORDS),
            "input": "",
            "output": _synthetic_words(rng, *SYNTHETIC_OUTPUT_WORDS),
        }
    return {"text": _synthetic_words(rng, *SYNTHETIC_DOC_WORDS)}


def load_synthetic(source: SourceConfig, offset: int, count: int, *, token: str | None = None) -> Iterator[Row]:
    """Deterministic random-word rows seeded by `source.seed` (`{"text"}` for pretrain/holdout, instruct triple)."""
    _check_offset_count(offset, count)
    for index in range(offset, offset + count):
        yield synthetic_row(source.kind, source.seed, index)


LOADERS: dict[str, Loader] = {
    "hf_split": load_hf_split,
    "hf_stream": load_hf_stream,
    "github_code": load_github_code,
    "local": load_local,
    "synthetic": load_synthetic,
}


def get_loader(name: str) -> Loader:
    if name not in LOADERS:
        raise ValueError(f"unknown loader {name!r}; known loaders: {sorted(LOADERS)}")
    return LOADERS[name]


def write_synthetic_tokenizer(path: Path) -> None:
    """A WordLevel tokenizer.json (<pad>=0, <bos>=1, <eos>=2, tok_i=3+i) written by hand so there is no magic."""
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for i in range(N_WORD_TOKENS):
        vocab[f"tok_{i}"] = len(vocab)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab[t],
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for t in SPECIALS
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<pad>"},
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "model_max_length": 1_000_000,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    (path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}, indent=2)
    )


# --- converters -------------------------------------------------------------------------------------------------------


def _require(row: Row, *keys: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise ValueError(f"row is missing column(s) {missing}; available columns: {sorted(row)}")


def gsm8k_question_answer(row: Row) -> Row:
    """GSM8K `question` + `answer` -> one pretraining document."""
    _require(row, "question", "answer")
    return {"text": f"Question: {row['question']}\n\nAnswer: {row['answer']}"}


def _conversations(row: Row) -> list[Any]:
    _require(row, "conversations")
    convs = row["conversations"]
    if not isinstance(convs, list):
        raise ValueError(f"'conversations' must be a list, got {type(convs).__name__}; columns: {sorted(row)}")
    return convs


def sharegpt_conversations(row: Row) -> Row:
    """SlimOrca / ShareGPT `conversations` with `from`/`value` turns: system->input, human->instruction, gpt->output.

    Later turns of the same role overwrite earlier ones (as in the thesis pipeline: only one exchange is kept).
    """
    system_msg = human_msg = gpt_msg = ""
    for turn in _conversations(row):
        if not isinstance(turn, dict) or "from" not in turn:
            raise ValueError(f"sharegpt_conversations: turns need 'from'/'value' keys, got {turn!r}")
        role, value = turn["from"], str(turn.get("value", ""))
        if role == "system":
            system_msg = value
        elif role == "human":
            human_msg = value
        elif role == "gpt":
            gpt_msg = value
    return {"instruction": human_msg, "input": system_msg, "output": gpt_msg}


def first_two_turns(row: Row) -> Row:
    """`conversations` without role tags (WizardLM): first `value` = instruction, second = output."""
    convs = _conversations(row)
    if len(convs) < 2:
        raise ValueError(f"first_two_turns: need at least two turns, got {len(convs)}; columns: {sorted(row)}")
    first, second = convs[0], convs[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError(f"first_two_turns: turns must be dicts with a 'value' key, got {convs[:2]!r}")
    return {"instruction": str(first.get("value", "")), "input": "", "output": str(second.get("value", ""))}


def instruction_input_output(row: Row) -> Row:
    """Rows that already carry `instruction`/`output` (and optionally `input`); missing input -> ""."""
    return fields_converter({"instruction": "instruction", "input": "input", "output": "output"})(row)


def fields_converter(fields: dict[str, str]) -> Converter:
    """Converter mapping `{instruction: <col>, input: <col>?, output: <col>}` to the standard instruct row."""
    if not {"instruction", "output"} <= set(fields):
        raise ValueError(f"fields must map at least instruction and output, got {sorted(fields)}")
    unknown = set(fields) - {"instruction", "input", "output"}
    if unknown:
        raise ValueError(f"fields has unknown keys {sorted(unknown)}; allowed: instruction, input, output")
    instruction_col, output_col = fields["instruction"], fields["output"]
    input_col = fields.get("input")

    def convert(row: Row) -> Row:
        _require(row, instruction_col, output_col)
        input_value = row.get(input_col, "") if input_col is not None else ""
        return {
            "instruction": str(row[instruction_col]),
            "input": "" if input_value is None else str(input_value),
            "output": str(row[output_col]),
        }

    return convert


CONVERTERS: dict[str, Converter] = {
    "gsm8k_question_answer": gsm8k_question_answer,
    "sharegpt_conversations": sharegpt_conversations,
    "first_two_turns": first_two_turns,
    "instruction_input_output": instruction_input_output,
}


def get_converter(source: SourceConfig) -> Converter | None:
    """`fields` mapping first, then the named `converter`, else None (row used as is; `text_field` applied later)."""
    if source.fields is not None:
        return fields_converter(source.fields)
    if source.converter is None:
        return None
    if source.converter not in CONVERTERS:
        raise ValueError(f"unknown converter {source.converter!r}; known converters: {sorted(CONVERTERS)}")
    return CONVERTERS[source.converter]


# --- filters ----------------------------------------------------------------------------------------------------------


def sharegpt_quality(row: Row) -> bool:
    """ShareGPT quality filter: human->gpt opening, 50-2000 chars per side, no code blocks in the answer."""
    convs = row.get("conversations")
    if not isinstance(convs, list) or len(convs) < 2:
        return False
    first, second = convs[0], convs[1]
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if first.get("from") != "human" or second.get("from") != "gpt":
        return False
    human_text, gpt_text = str(first.get("value", "")), str(second.get("value", ""))
    if not 50 <= len(human_text) <= 2000 or not 50 <= len(gpt_text) <= 2000:
        return False
    return not any(p in gpt_text.lower() for p in ("```python", "```java", "```cpp", "```javascript"))


FILTERS: dict[str, Filter] = {"sharegpt_quality": sharegpt_quality}


def get_filter(name: str) -> Filter:
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; known filters: {sorted(FILTERS)}")
    return FILTERS[name]


# --- misc -------------------------------------------------------------------------------------------------------------


def repeat_indices(num_rows: int, target: int) -> list[int]:
    """Indices that cycle through `range(num_rows)` until `target` rows are covered (`repeat_to_budget` sources)."""
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    full_copies, remainder = divmod(target, num_rows)
    return list(range(num_rows)) * full_copies + list(range(remainder))
