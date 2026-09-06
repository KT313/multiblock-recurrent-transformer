# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Thin wrapper around a HuggingFace fast tokenizer directory (tokenizer.json + tokenizer_config.json).
"""

from pathlib import Path
from typing import cast

# Label value of positions without a loss (padding, masked prompts, out-of-vocab); the model defaults to it too.
# Never a token id: a real `<unk>` or `<pad>` token in a document is a supervised label like any other.
IGNORE_INDEX = -100


def resolve_pad_id(processor: object, eos_id: int) -> int:
    """
    The id generation pads with (`evaluation.wrapper`, `evaluation.samples`): the pad token if the tokenizer defines
    one, else EOS. Training never pads with it: pad positions are EOS in the inputs and `IGNORE_INDEX` in the labels.
    """

    pad_id = cast(int | None, getattr(processor, "pad_token_id", None))
    return eos_id if pad_id is None else pad_id


class Tokenizer:
    """
    Loads a HF tokenizer directory and encodes text without automatic special tokens.

    BOS/EOS are added explicitly by :meth:`encode` so that formatting functions control them. Both must exist: the
    formats prepend BOS, every document ends in EOS, and pack tails are EOS.
    """

    def __init__(self, path: str | Path) -> None:
        from transformers import AutoTokenizer

        self.path = Path(path)
        if not (self.path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"No tokenizer.json in {self.path}")
        self.processor = AutoTokenizer.from_pretrained(str(self.path), add_bos_token=False, add_eos_token=False)
        bos_id, eos_id = self.processor.bos_token_id, self.processor.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError(f"Tokenizer at {self.path} must define a BOS and an EOS token")
        self.bos_id: int = bos_id
        self.eos_id: int = eos_id
        self.pad_id: int = resolve_pad_id(self.processor, self.eos_id)

    @property
    def vocab_size(self) -> int:
        """
        Size of the base vocabulary (without added tokens), used to mask out-of-range labels.
        """

        return self.processor.vocab_size

    def __len__(self) -> int:
        return len(self.processor)

    def __reduce__(self) -> tuple[type["Tokenizer"], tuple[Path]]:
        """
        Pickle as (class, (path,)) so worker processes reload the tokenizer from disk instead of copying it.
        """

        return (self.__class__, (self.path,))

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """
        Tokenize text; prepend BOS / append EOS when requested.
        """

        tokens: list[int] = self.processor.encode(text)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        # decode() of a flat id list is always a str; the stub's `str | list[str]` covers the batched overload.
        return cast(str, self.processor.decode(ids, skip_special_tokens=skip_special_tokens))
