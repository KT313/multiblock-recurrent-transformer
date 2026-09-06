# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
`SavedTokenizer` against transformers' loader, for every tokenizer a shipped dataset config names: the download's
counts and offsets and the training ids must not move by one token when the loader does (the stored counts and
the raw hash depend on them), so transformers stays the oracle here. The Hub is never reached (the root conftest
forces offline mode); a Hub tokenizer missing from the local cache skips with a reason.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from data_preparation.dataset_config import TokenizerConfig, load_dataset_config
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]

CORPUS = [
    "Hello world",
    " leading space",
    "  two spaces",
    "trailing space ",
    "",
    " ",
    "\n",
    "line one\nline two\n\n",
    "tabs\tand\ttabs",
    "<s> inside </s> text <unk> <bos> <eos> <pad>",
    "</s>",
    "▁metaspace char",
    "naïve café résumé",
    "日本語のテキストと中文和한국어",
    "emoji 😀🎉 and math ∑∫√",
    "1234567890 3.14159 1e-10",
    "def f(x):\n    return x**2  # comment",
    "https://example.com/a?b=c&d=e",
    "MixedCASE and CamelCase and snake_case",
    "a" * 500,
    "tok_1 " * 300,
    "quotes “smart” 'single' \"double\"",
    "hyphen-ated words, punctuation!?;: (parens) [brackets]",
    "  <s>  ",
    "##hash ##tags",
]
_PIECES = ["the", "quick", "brown", "über", "naïve", "西", "🙂", "<s>", "</s>", "\n", "  ", "3.5", "x_y", "Ab", "tok_7", "tok_255", "<bos>"]
_random = random.Random(1)
CORPUS += [" ".join(_random.choice(_PIECES) for _ in range(_random.randint(0, 60))) for _ in range(300)]


def _shipped_tokenizers() -> list[TokenizerConfig]:
    seen: dict[tuple[str, str | None, str | None], TokenizerConfig] = {}
    for path in sorted((REPO_ROOT / "config" / "datasets").glob("*.yaml")):
        tokenizer = load_dataset_config(path).tokenizer
        seen.setdefault((tokenizer.kind, tokenizer.hf_id, tokenizer.revision), tokenizer)
    return list(seen.values())


@pytest.fixture(params=_shipped_tokenizers(), ids=lambda tokenizer: tokenizer.name)
def tokenizer_dir(request: pytest.FixtureRequest, tiny_tokenizer_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    The tokenizer's directory as `prepare_tokenizer` stores it: the tiny fixture's synthetic one, or a Hub tokenizer
    from the local cache saved through `save_pretrained`.
    """

    tokenizer: TokenizerConfig = request.param
    if tokenizer.kind == "synthetic":
        return tiny_tokenizer_dir
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    assert tokenizer.hf_id is not None
    try:
        snapshot = snapshot_download(tokenizer.hf_id, revision=tokenizer.revision, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{tokenizer.hf_id} at revision {tokenizer.revision} is not in the local HF cache")
    directory = tmp_path_factory.mktemp(tokenizer.name)
    AutoTokenizer.from_pretrained(snapshot).save_pretrained(str(directory))
    return directory


def test_ids_offsets_specials_and_decode_match_transformers(tokenizer_dir: Path) -> None:
    loader = SavedTokenizer(tokenizer_dir)
    counting = AutoTokenizer.from_pretrained(str(tokenizer_dir))  # what TokenCounter loaded
    training = AutoTokenizer.from_pretrained(str(tokenizer_dir), add_bos_token=False, add_eos_token=False)  # what Tokenizer loaded

    encoded = counting(CORPUS, add_special_tokens=False, return_offsets_mapping=True)
    batch = loader.encode_batch(CORPUS)
    assert [encoding.ids for encoding in batch] == encoded["input_ids"]
    assert [encoding.offsets for encoding in batch] == encoded["offset_mapping"]
    assert [loader.encode(text) for text in CORPUS] == [training.encode(text) for text in CORPUS]
    for ids in encoded["input_ids"]:
        assert loader.decode(ids) == training.decode(ids, skip_special_tokens=False)
        assert loader.decode(ids, skip_special_tokens=True) == training.decode(ids, skip_special_tokens=True)
    specials = (loader.bos_id, loader.eos_id, loader.pad_id, loader.unk_id)
    assert specials == (training.bos_token_id, training.eos_token_id, training.pad_token_id, training.unk_token_id)
    assert loader.bos_id is not None and loader.eos_id is not None
    assert loader.vocab_size == training.vocab_size and len(loader) == len(training)
    assert loader.encode_batch([]) == []


def _copy_with_config(source: Path, destination: Path, **changes: object) -> Path:
    shutil.copytree(source, destination)
    config = json.loads((destination / "tokenizer_config.json").read_text())
    config.update(changes)
    (destination / "tokenizer_config.json").write_text(json.dumps(config))
    return destination


def test_special_tokens_in_dict_form_and_from_the_fallback_file(tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    directory = _copy_with_config(tiny_tokenizer_dir, tmp_path / "forms", bos_token={"content": "<bos>", "special": True}, eos_token=None)
    loader = SavedTokenizer(directory)
    assert (loader.bos_id, loader.eos_id, loader.pad_id) == (1, 2, 0), "eos from special_tokens_map.json, pad from the config"
    (directory / "special_tokens_map.json").unlink()
    assert SavedTokenizer(directory).eos_id is None, "null and no fallback: the tokenizer has no such token"


def test_refuses_a_directory_without_tokenizer_json(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match=f"no tokenizer.json in {tmp_path}"):
        SavedTokenizer(tmp_path)


def test_refuses_a_special_token_outside_the_vocabulary(tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    directory = _copy_with_config(tiny_tokenizer_dir, tmp_path / "unknown", bos_token="<nope>")
    with pytest.raises(ValueError, match=f"{directory}: bos_token '<nope>' is not in the vocabulary"):
        SavedTokenizer(directory)


def test_refuses_clean_up_tokenization_spaces(tiny_tokenizer_dir: Path, tmp_path: Path) -> None:
    directory = _copy_with_config(tiny_tokenizer_dir, tmp_path / "cleanup", clean_up_tokenization_spaces=True)
    with pytest.raises(ValueError, match="clean_up_tokenization_spaces"):
        SavedTokenizer(directory)
