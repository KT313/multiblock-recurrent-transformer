# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Optional native experiments, never imported by production preparation."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import os
from pathlib import Path
from typing import Protocol, cast

from tokenizers import Tokenizer

from tools.data_preparation.replay import ReplayCounter


class NativeOperations(Protocol):
    def truncate_many(self, texts: list[str], cap: int) -> list[tuple[str, int]]: ...
    def count_many(self, texts: list[str]) -> list[int]: ...


class RustCounter(ReplayCounter):
    def __init__(self, path: Path) -> None:
        super().__init__(path, False)
        if importlib.metadata.version("tokenizers") != "0.23.1":
            raise ValueError("native experiment pins tokenizers 0.23.1; use the matching Python version")
        library = os.environ.get("PREPARER_NATIVE_LIBRARY")
        if library is None:
            raise ValueError("rust variant requires --native-library pointing to the experimental compiled adapter")
        spec = importlib.util.spec_from_file_location("preparer_native", library)
        if spec is None or spec.loader is None:
            raise ValueError("could not load native experiment")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert self._tokenizer is not None
        backend = self._tokenizer._tokenizer
        self._native = cast(NativeOperations, module.NativeCounter(backend.to_str(), backend.encode_special_tokens))

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        return self._native.truncate_many(texts, max_tokens)

    def count_many(self, texts: list[str]) -> list[int]:
        return self._native.count_many(texts)


class BuiltinCounter(ReplayCounter):
    """Existing APIs only: native cap+1 truncation for boundaries, fast no-offset count encoding."""
    def __init__(self, path: Path) -> None:
        super().__init__(path, False)
        assert self._tokenizer is not None
        self._bounded = Tokenizer.from_str(self._tokenizer._tokenizer.to_str())
        self._bounded.encode_special_tokens = self._tokenizer._tokenizer.encode_special_tokens
        if self._bounded.truncation is not None or self._bounded.padding is not None:
            raise ValueError("built-in truncation experiment requires an artifact without padding/truncation")

    def count_many(self, texts: list[str]) -> list[int]:
        assert self._tokenizer is not None
        return [len(encoding) for encoding in self._tokenizer._tokenizer.encode_batch_fast(texts, add_special_tokens=False)]

    def truncate_many(self, texts: list[str], max_tokens: int) -> list[tuple[str, int]]:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        self._bounded.enable_truncation(max_tokens + 1, stride=0)
        current = [text[:32 * max_tokens] for text in texts]
        counts = [0] * len(texts)
        pending = list(range(len(texts)))
        while pending:
            encoded = self._bounded.encode_batch([current[i] for i in pending], add_special_tokens=False)
            remaining = []
            for index, encoding in zip(pending, encoded, strict=True):
                if len(encoding) <= max_tokens:
                    counts[index] = len(encoding)
                else:
                    offset = encoding.token_to_chars(max_tokens)
                    if offset is None:
                        offset = encoding.offsets[max_tokens]
                    current[index] = current[index][:min(offset[0], len(current[index]) - 1)]
                    remaining.append(index)
            pending = remaining
        return list(zip(current, counts, strict=True))
