# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import json
import pickle
import shutil
from pathlib import Path

import pytest

from data_preparation.lib.sources.synthetic import N_WORD_TOKENS, SPECIALS
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer
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
    # The synthetic tokenizer uses <pad> (id 0) as its unk token; it is a token like any other, not a sentinel.
    assert tokenizer.encode("definitely_not_a_token") == [0]


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


def _without(tiny_tokenizer_dir: Path, tmp_path: Path, token: str) -> Path:
    dst = tmp_path / f"no_{token}"
    shutil.copytree(tiny_tokenizer_dir, dst)
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        cfg = json.loads((dst / name).read_text())
        cfg.pop(token, None)
        (dst / name).write_text(json.dumps(cfg))
    return dst


def test_missing_pad_token_pads_generation_with_eos(tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    """
    The Llama case: no pad token, so generation pads with EOS. Training never uses the pad id.
    """

    tokenizer = Tokenizer(_without(tiny_tokenizer_dir, tmp_path, "pad_token"))
    assert SavedTokenizer(tokenizer.path).pad_id is None
    assert tokenizer.pad_id == tokenizer.eos_id == 2


@pytest.mark.parametrize("token", ["bos_token", "eos_token"])
def test_missing_bos_or_eos_token_raises(tiny_tokenizer_dir: Path, tmp_path: Path, token: str) -> None:
    path = _without(tiny_tokenizer_dir, tmp_path, token)
    with pytest.raises(ValueError, match=f"{path}.*BOS and an EOS"):
        Tokenizer(path)


def test_resolve_pad_id_prefers_the_pad_token() -> None:
    from types import SimpleNamespace

    from training.data.tokenizer import resolve_pad_id

    assert resolve_pad_id(SimpleNamespace(pad_id=5), eos_id=2) == 5
    assert resolve_pad_id(SimpleNamespace(pad_id=None), eos_id=2) == 2


def _tokenizer_dir_with_an_added_token_eos(tmp_path: Path) -> Path:
    """
    A hand-written tokenizer directory (never the Hub) whose EOS is an added token, not a base-vocabulary one:
    base vocab <pad>=0, <bos>=1, tok_0=2, and <eos>=3 in `added_tokens`. That is the shape of most modern HF
    tokenizers (Llama-3, SmolLM2, Qwen), where `get_vocab_size(with_added_tokens=False)` excludes the specials.
    """

    path = tmp_path / "added_token_eos"
    path.mkdir()
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 3, "content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"<pad>": 0, "<bos>": 1, "tok_0": 2}, "unk_token": "<pad>"},
    }
    special_tokens_map = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}
    (path / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", **special_tokens_map}))
    (path / "special_tokens_map.json").write_text(json.dumps(special_tokens_map))
    return path


def test_specials_outside_the_base_vocabulary_are_refused_at_load(tmp_path: Path) -> None:
    """
    `collate.mask_label_ids` masks every id >= vocab_size, so an added-token EOS would silently erase the EOS
    label of every document: the load must fail instead.
    """

    path = _tokenizer_dir_with_an_added_token_eos(tmp_path)
    backend = SavedTokenizer(path)
    # EOS 3 sits past the base vocabulary (3 tokens) and only `with_added_tokens=True` counts it: the guard's case
    assert (backend.eos_id, backend.vocab_size, len(backend)) == (3, 3, 4)
    with pytest.raises(ValueError, match="EOS 3 lie.*outside its base vocabulary of 3 tokens"):
        Tokenizer(path)
