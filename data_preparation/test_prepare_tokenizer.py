# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.prepare_tokenizer with the Hub download stubbed by the tiny tokenizer."""

import sys
from pathlib import Path
from typing import Any

import pytest

from data_preparation import prepare_tokenizer as pt


def test_parser_and_constants() -> None:
    args = pt.build_parser().parse_args([])
    assert str(args.dataset_dir) == "dataset" and args.cache_dir is None
    assert pt.TOKENIZER_NAME == "hf-internal-testing/llama-tokenizer"


def test_main_saves_tokenizer_to_dataset_tokenizer(
    tmp_path: Path, tiny_tokenizer_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    real_from_pretrained = transformers.AutoTokenizer.from_pretrained
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_from_pretrained(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return real_from_pretrained(str(tiny_tokenizer_path), *args, **kwargs)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(sys, "argv", ["prepare_tokenizer", "--dataset_dir", str(tmp_path)])
    pt.main()

    assert calls == [(pt.TOKENIZER_NAME, {"use_fast": True})]
    out = tmp_path / "tokenizer"
    assert (out / "tokenizer.json").is_file() and (out / "tokenizer_config.json").is_file()
    reloaded = real_from_pretrained(str(out))
    assert reloaded.encode("tok_0 tok_255", add_special_tokens=False) == [3, 258]
    assert reloaded.pad_token_id == 0 and reloaded.bos_token_id == 1 and reloaded.eos_token_id == 2


def test_main_configures_cache_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Path | None] = {}

    def fake_configure(cache_dir: Path | None) -> None:
        seen["cache_dir"] = cache_dir

    class FakeTok:
        def save_pretrained(self, path: Path) -> None:
            seen["saved_to"] = path

    transformers = pytest.importorskip("transformers")
    monkeypatch.setattr(pt, "configure_hf_cache", fake_configure)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr(
        sys, "argv", ["prepare_tokenizer", "--dataset_dir", str(tmp_path), "--cache_dir", str(tmp_path / "c")]
    )
    pt.main()
    assert seen == {"cache_dir": tmp_path / "c", "saved_to": tmp_path / "tokenizer"}
    assert (tmp_path / "tokenizer").is_dir()
