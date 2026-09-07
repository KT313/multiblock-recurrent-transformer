# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""
Thin wrapper around a saved tokenizer directory (tokenizer.json + tokenizer_config.json), loaded without transformers.
"""

from pathlib import Path
from typing import Any, cast

from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

# Label value of positions without a loss (padding, masked prompts, out-of-vocab); the model defaults to it too.
# Never a token id: a real `<unk>` or `<pad>` token in a document is a supervised label like any other.
IGNORE_INDEX = -100

# Text the automatic-BOS self-check encodes; any text does, the check only compares its two encodings.
BOS_PROBE = "a b"


def resolve_pad_id(processor: object, eos_id: int) -> int:
    """
    The id generation pads with (`evaluation.wrapper`, `evaluation.samples`): the pad token if the tokenizer defines
    one, else EOS. Training never pads with it: pad positions are EOS in the inputs and `IGNORE_INDEX` in the labels.
    """

    pad_id = cast(int | None, getattr(processor, "pad_id", None))
    return eos_id if pad_id is None else pad_id


def check_automatic_bos(processor: Any, bos_id: int, path: Path) -> None:
    """
    Verify that `add_bos_token=True` really makes `add_special_tokens=True` prepend BOS.

    That the flag's setter installs a `bos $A` post-processor is a transformers implementation detail. If a later
    version drops it, lm-eval would keep asking for special tokens and keep getting none, and every benchmark would
    be scored on contexts no training row looks like: a few points lower, with nothing in the logs. Checked once per
    load instead, so a benchmark run on such a version dies at setup.
    """

    import transformers

    with_specials = processor.encode(BOS_PROBE, add_special_tokens=True)
    without = processor.encode(BOS_PROBE, add_special_tokens=False)
    if with_specials == [bos_id] + without and not (without and without[0] == bos_id):
        return
    raise RuntimeError(
        f"The tokenizer at {path}, loaded with add_bos_token=True, does not prepend BOS ({bos_id}) when asked for "
        f"special tokens: transformers {transformers.__version__} encodes {BOS_PROBE!r} as {with_specials} with "
        f"special tokens and as {without} without them. Benchmarks would be scored on contexts that start unlike "
        "every training row; pin transformers to a version whose add_bos_token installs the BOS template."
    )


class Tokenizer:
    """
    Loads a saved tokenizer directory and encodes text without automatic special tokens.

    BOS/EOS are added explicitly by :meth:`encode` so that formatting functions control them (:attr:`processor`, the
    transformers object lm-eval encodes through, is the exception: it prepends BOS itself). Both must exist: the
    formats prepend BOS, every document ends in EOS, and pack tails are EOS. Both must also be base-vocabulary
    tokens, because labels are masked against `vocab_size` (see :meth:`__init__`).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"No tokenizer.json in {self.path}")
        self._backend = SavedTokenizer(self.path)
        self._processor: Any = None
        bos_id, eos_id = self._backend.bos_id, self._backend.eos_id
        if bos_id is None or eos_id is None:
            raise ValueError(f"Tokenizer at {self.path} must define a BOS and an EOS token")
        outside = [f"{name} {token_id}" for name, token_id in (("BOS", bos_id), ("EOS", eos_id)) if token_id >= self.vocab_size]
        if outside:
            raise ValueError(
                f"Tokenizer at {self.path}: {' and '.join(outside)} lie(s) outside its base vocabulary of "
                f"{self.vocab_size} tokens (they are added tokens). Labels are bounded by the base vocabulary "
                "(`training.data.collate` masks every id >= vocab_size to IGNORE_INDEX), so such an EOS would be "
                "masked out of every document and the model would never learn to stop. Rebuild the tokenizer "
                "directory with its specials in the base vocabulary."
            )
        self.bos_id: int = bos_id
        self.eos_id: int = eos_id
        # pad_id is never a label (generation pads with it, training pads inputs with EOS), so it is not bounded here
        self.pad_id: int = resolve_pad_id(self._backend, self.eos_id)

    @property
    def processor(self) -> Any:
        """
        The transformers tokenizer over the same directory, which lm-eval (`evaluation.benchmarks`) needs: built on
        first use, because transformers imports torch and scikit-learn.

        Automatic BOS is on, so a context lm-eval encodes starts with BOS as every training row does; automatic EOS
        is off, because a context is a prefix the model continues. `check_automatic_bos` confirms the flag took
        effect before the object is handed out.
        """

        if self._processor is None:
            from transformers import AutoTokenizer

            processor = AutoTokenizer.from_pretrained(str(self.path), add_bos_token=True, add_eos_token=False)
            check_automatic_bos(processor, self.bos_id, self.path)
            self._processor = processor
        return self._processor

    @property
    def vocab_size(self) -> int:
        """
        Size of the base vocabulary (without added tokens), used to mask out-of-range labels.
        """

        return self._backend.vocab_size

    def __len__(self) -> int:
        return len(self._backend)

    def __reduce__(self) -> tuple[type["Tokenizer"], tuple[Path]]:
        """
        Pickle as (class, (path,)) so worker processes reload the tokenizer from disk instead of copying it.
        """

        return (self.__class__, (self.path,))

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """
        Tokenize text; prepend BOS / append EOS when requested.
        """

        tokens = self._backend.encode(text)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)
