# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import json
import pickle
import shutil
from pathlib import Path

import pytest

from data_preparation.lib.sources.synthetic import N_WORD_TOKENS, SPECIALS
from training.data.tokenizer import Tokenizer


def test_special_ids_and_sizes(tokenizer: Tokenizer) -> None:
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id) == (0, 1, 2)
    assert tokenizer.vocab_size == len(SPECIALS) + N_WORD_TOKENS
    assert len(tokenizer) == tokenizer.vocab_size


def test_encode_maps_words_to_ids(tokenizer: Tokenizer) -> None:
    assert tokenizer.encode("tok_0 tok_5 tok_255") == [3, 8, 258]


@pytest.mark.parametrize(("bos", "eos"), [(False, False), (True, False), (False, True), (True, True)])
def test_encode_bos_eos(tokenizer: Tokenizer, bos: bool, eos: bool) -> None:
    out = tokenizer.encode("tok_7 tok_8", bos=bos, eos=eos)
    expected = ([1] if bos else []) + [10, 11] + ([2] if eos else [])
    assert out == expected


def test_encode_never_adds_specials_implicitly(tokenizer: Tokenizer) -> None:
    out = tokenizer.encode("tok_3")
    assert tokenizer.bos_id not in out and tokenizer.eos_id not in out


def test_unknown_word_maps_to_unk(tokenizer: Tokenizer) -> None:
    # The synthetic tokenizer uses <pad> as its unk token.
    assert tokenizer.encode("definitely_not_a_token") == [tokenizer.pad_id]


def test_decode_round_trip(tokenizer: Tokenizer) -> None:
    text = "tok_1 tok_2 tok_3"
    assert isinstance(tokenizer.decode([3]), str)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode(text, bos=True, eos=True), skip_special_tokens=True) == text


def test_reduce_returns_class_and_path(tokenizer: Tokenizer) -> None:
    cls, args = tokenizer.__reduce__()
    assert cls is Tokenizer and args == (tokenizer.path,)


def test_picklable_by_path(tokenizer: Tokenizer) -> None:
    clone = pickle.loads(pickle.dumps(tokenizer))
    assert clone.path == tokenizer.path
    assert clone.encode("tok_9 tok_10", bos=True) == tokenizer.encode("tok_9 tok_10", bos=True)


def test_missing_tokenizer_json_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Tokenizer(tmp_path)


def _without_pad(tiny_tokenizer_dir: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "nopad"
    shutil.copytree(tiny_tokenizer_dir, dst)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        cfg = json.loads((dst / name).read_text())
        cfg.pop("pad_token", None)
        (dst / name).write_text(json.dumps(cfg))
    return dst


def test_missing_pad_token_falls_back(tiny_tokenizer_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The synthetic tokenizer exposes no unk at the HF level, so without a pad token it falls back to EOS."""
    with caplog.at_level("WARNING"):
        tokenizer = Tokenizer(_without_pad(tiny_tokenizer_dir, tmp_path))
    assert tokenizer.processor.pad_token_id is None
    assert tokenizer.pad_id == tokenizer.eos_id == 2
    assert "no pad token" in caplog.text and "eos_token_id" in caplog.text


def test_resolve_pad_id_order() -> None:
    """Thesis behaviour for the Llama tokenizer: no pad token -> `<unk>` (id 0); then EOS; then an error."""
    from types import SimpleNamespace

    from training.data.tokenizer import resolve_pad_id

    assert resolve_pad_id(SimpleNamespace(pad_token_id=5, unk_token_id=0, eos_token_id=2), Path("t")) == 5
    assert resolve_pad_id(SimpleNamespace(pad_token_id=None, unk_token_id=0, eos_token_id=2), Path("t")) == 0
    assert resolve_pad_id(SimpleNamespace(pad_token_id=None, unk_token_id=None, eos_token_id=2), Path("t")) == 2
    with pytest.raises(ValueError, match="no pad, unk or eos token"):
        resolve_pad_id(SimpleNamespace(pad_token_id=None, unk_token_id=None, eos_token_id=None), Path("t"))
