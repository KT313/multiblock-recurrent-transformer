# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
A saved tokenizer directory (`tokenizer.json` + `tokenizer_config.json`, what `save_pretrained` writes) loaded
through the `tokenizers` library alone.

transformers' tokenizer classes import torch and, through the generation utilities, scikit-learn: 2.5 s and 300 MB
in every process that only counts or encodes (the download jobs, the trainer and its DataLoader workers, every test
worker), for a wrapper whose encoding, offsets and decoding are the Rust tokenizer's own. The ids, offsets and
decoded text are those transformers' loader gives; `test_tokenizer_loader.py` checks that for every tokenizer a
shipped dataset config names, because the stored token counts and the raw hash depend on them. The Hub download
and `save_pretrained` (`download.prepare_tokenizer`) stay on transformers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Encoding, Tokenizer

SPECIAL_TOKEN_NAMES = ("bos_token", "eos_token", "pad_token", "unk_token")


class SavedTokenizer:
    """
    The tokenizer of a saved directory. `encode` and `encode_batch` add no special tokens: the download counts the
    text's own tokens and the training formats place BOS and EOS themselves (transformers' loader was used the same
    way, with `add_special_tokens=False` and `add_bos_token=False` / `add_eos_token=False`). The special-token ids
    come from `tokenizer_config.json`, with `special_tokens_map.json` as the fallback; a token given as a string or
    as an `{"content": ...}` entry (older saves), null means the tokenizer has none.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        file = self.path / "tokenizer.json"
        if not file.is_file():
            raise ValueError(f"no tokenizer.json in {self.path}: only a fast tokenizer directory (as save_pretrained writes it) can be loaded")
        self._tokenizer = Tokenizer.from_file(str(file))
        config = _read_json(self.path / "tokenizer_config.json")
        if config.get("clean_up_tokenization_spaces"):
            # transformers' decode would rewrite spaces around punctuation; decoding here is the raw tokenizer's
            raise ValueError(f"{self.path}: clean_up_tokenization_spaces is set, which this loader does not replicate")
        fallback = _read_json(self.path / "special_tokens_map.json")
        ids = {name: self._special_id(name, config.get(name) if config.get(name) is not None else fallback.get(name)) for name in SPECIAL_TOKEN_NAMES}
        self.bos_id: int | None = ids["bos_token"]
        self.eos_id: int | None = ids["eos_token"]
        self.pad_id: int | None = ids["pad_token"]
        self.unk_id: int | None = ids["unk_token"]

    def _special_id(self, name: str, value: Any) -> int | None:
        if value is None:
            return None
        token = value["content"] if isinstance(value, dict) else value
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"{self.path}: {name} {token!r} is not in the vocabulary")
        return token_id

    @property
    def vocab_size(self) -> int:
        """
        The base vocabulary without added tokens (what `training.data.collate` masks labels against).
        """

        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[Encoding]:
        """
        One `Encoding` (ids, offsets) per text; an empty list gives an empty list.
        """

        return self._tokenizer.encode_batch(texts, add_special_tokens=False)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
