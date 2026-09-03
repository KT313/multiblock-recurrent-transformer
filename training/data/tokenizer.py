# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Thin wrapper around a HuggingFace fast tokenizer directory (``tokenizer.json`` + ``tokenizer_config.json``)."""

import logging
from pathlib import Path
from typing import cast

log = logging.getLogger(__name__)

# Tried in order when picking the padding id (see `resolve_pad_id`).
_PAD_ID_FALLBACKS = ("pad_token_id", "unk_token_id", "eos_token_id")


def resolve_pad_id(processor: object, path: Path) -> int:
    """The id used for padding and for masking labels.

    Tokenizers without a pad token (the Llama tokenizer) fall back to the unk token (Llama's `<unk>`, id 0), then to
    EOS. Pad positions in the inputs are replaced by EOS in the collate function anyway; in the labels they become
    the ignore index.
    """
    for attribute in _PAD_ID_FALLBACKS:
        token_id = cast(int | None, getattr(processor, attribute, None))
        if token_id is None:
            continue
        if attribute != "pad_token_id":
            log.warning(
                "Tokenizer at %s defines no pad token; using its %s (id %d) for padding", path, attribute, token_id
            )
        return token_id
    raise ValueError(f"Tokenizer at {path} defines no pad, unk or eos token; padding/label masking needs one.")


class Tokenizer:
    """Loads a HF tokenizer directory and encodes text without automatic special tokens.

    BOS/EOS are added explicitly by :meth:`encode` so that formatting functions control them.
    """

    def __init__(self, path: str | Path) -> None:
        from transformers import AutoTokenizer

        self.path = Path(path)
        if not (self.path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"No tokenizer.json in {self.path}")
        self.processor = AutoTokenizer.from_pretrained(str(self.path), add_bos_token=False, add_eos_token=False)
        self.bos_id: int | None = self.processor.bos_token_id
        self.eos_id: int | None = self.processor.eos_token_id
        self.pad_id: int = resolve_pad_id(self.processor, self.path)

    @property
    def vocab_size(self) -> int:
        """Size of the base vocabulary (without added tokens), used to mask out-of-range labels."""
        return self.processor.vocab_size

    def __len__(self) -> int:
        return len(self.processor)

    def __reduce__(self) -> tuple[type["Tokenizer"], tuple[Path]]:
        """Pickle as (class, (path,)) so worker processes reload the tokenizer from disk instead of copying it."""
        return (self.__class__, (self.path,))

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """Tokenize ``text``; prepend BOS / append EOS when requested and the tokenizer defines them."""
        tokens: list[int] = self.processor.encode(text)
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        # decode() of a flat id list is always a str; the stub's `str | list[str]` covers the batched overload.
        return cast(str, self.processor.decode(ids, skip_special_tokens=skip_special_tokens))
