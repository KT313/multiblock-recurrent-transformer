# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for data_preparation.lib.make_tiny_dataset: layout, tokenizer ids and determinism."""

import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from typing import cast
import pytest

from data_preparation.lib import make_tiny_dataset as mtd


def test_constants() -> None:
    assert mtd.SPECIALS == ["<pad>", "<bos>", "<eos>"]
    assert mtd.N_WORD_TOKENS == 256 and mtd.VOCAB_SIZE == 259
    assert set(mtd.SPLITS) == {"pretrain", "finetune"} and all(set(v) == {"train", "val"} for v in mtd.SPLITS.values())


def test_layout_matches_splits(tiny_dataset_dir: Path) -> None:
    assert (tiny_dataset_dir / "tokenizer" / "tokenizer.json").is_file()
    assert (tiny_dataset_dir / "tokenizer" / "tokenizer_config.json").is_file()
    assert (tiny_dataset_dir / "tokenizer" / "special_tokens_map.json").is_file()
    for name, splits in mtd.SPLITS.items():
        for split, (n_files, rows) in splits.items():
            files = sorted((tiny_dataset_dir / name / split).glob("*.parquet"))
            assert [f.name for f in files] == [f"part_{i:03d}.parquet" for i in range(n_files)]
            for f in files:
                table = pq.read_table(f)
                assert table.column_names == ["text"] and table.num_rows == rows
                for text in cast(list[str], table["text"].to_pylist()):
                    words = text.split()
                    assert 64 <= len(words) <= 384
                    assert all(w.startswith("tok_") and 0 <= int(w[4:]) < 256 for w in words)


def test_tokenizer_json_ids(tiny_dataset_dir: Path) -> None:
    tok = json.loads((tiny_dataset_dir / "tokenizer" / "tokenizer.json").read_text())
    vocab = tok["model"]["vocab"]
    assert tok["model"]["type"] == "WordLevel" and tok["model"]["unk_token"] == "<pad>"
    assert len(vocab) == 259
    assert vocab["<pad>"] == 0 and vocab["<bos>"] == 1 and vocab["<eos>"] == 2
    assert vocab["tok_0"] == 3 and vocab["tok_255"] == 258
    assert sorted(vocab.values()) == list(range(259))
    assert [t["id"] for t in tok["added_tokens"]] == [0, 1, 2] and all(t["special"] for t in tok["added_tokens"])
    assert tok["pre_tokenizer"] == {"type": "Whitespace"}
    cfg = json.loads((tiny_dataset_dir / "tokenizer" / "tokenizer_config.json").read_text())
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast" and cfg["pad_token"] == "<pad>"


def test_tokenizer_loads_with_transformers_and_roundtrips(tiny_tokenizer_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(tiny_tokenizer_path))
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2)
    assert len(tok) == 259
    ids = tok.encode("tok_0 tok_255 tok_17", add_special_tokens=False)
    assert ids == [3, 258, 20]
    assert tok.decode(ids) == "tok_0 tok_255 tok_17"
    assert tok.encode("unknown", add_special_tokens=False) == [0]  # unk -> <pad>


def test_documents_tokenize_to_word_count(tiny_dataset_dir: Path, tiny_tokenizer_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(tiny_tokenizer_path))
    table = pq.read_table(tiny_dataset_dir / "pretrain" / "train" / "part_000.parquet")
    for text in cast(list[str], table["text"].to_pylist())[:5]:
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) == len(text.split()) and 0 not in ids and min(ids) >= 3


def test_deterministic_for_fixed_seed(tmp_path: Path) -> None:
    mtd.make_tiny_dataset(tmp_path / "a", seed=0)
    mtd.make_tiny_dataset(tmp_path / "b", seed=0)
    mtd.make_tiny_dataset(tmp_path / "c", seed=1)

    def read(root: Path) -> list[list[str]]:
        return [cast(list[str], pq.read_table(f)["text"].to_pylist()) for f in sorted(root.rglob("*.parquet"))]

    assert read(tmp_path / "a") == read(tmp_path / "b")
    assert read(tmp_path / "a") != read(tmp_path / "c")
    assert (tmp_path / "a" / "tokenizer" / "tokenizer.json").read_bytes() == (
        tmp_path / "c" / "tokenizer" / "tokenizer.json"
    ).read_bytes()


def test_default_seed_matches_session_fixture(tmp_path: Path, tiny_dataset_dir: Path) -> None:
    mtd.make_tiny_dataset(tmp_path)
    a = pq.read_table(tmp_path / "finetune" / "val" / "part_001.parquet")["text"].to_pylist()
    b = pq.read_table(tiny_dataset_dir / "finetune" / "val" / "part_001.parquet")["text"].to_pylist()
    assert a == b


def test_write_split_returns_word_count(tmp_path: Path) -> None:
    total = mtd.write_split(tmp_path / "s", n_files=2, rows_per_file=3, rng=random.Random(5))
    tables = [pq.read_table(f) for f in sorted((tmp_path / "s").glob("*.parquet"))]
    assert len(tables) == 2
    assert total == sum(len(t.split()) for tab in tables for t in cast(list[str], tab["text"].to_pylist()))


def test_build_tokenizer_direct(tmp_path: Path) -> None:
    mtd.build_tokenizer(tmp_path / "tok")
    assert sorted(p.name for p in (tmp_path / "tok").iterdir()) == [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    specials = json.loads((tmp_path / "tok" / "special_tokens_map.json").read_text())
    assert specials == {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}
    # idempotent: writing into an existing directory overwrites with identical content
    before = (tmp_path / "tok" / "tokenizer.json").read_bytes()
    mtd.build_tokenizer(tmp_path / "tok")
    assert (tmp_path / "tok" / "tokenizer.json").read_bytes() == before


def test_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["make_tiny_dataset", "--out", str(tmp_path / "tiny")])
    mtd.main()
    assert (tmp_path / "tiny" / "pretrain" / "train" / "part_003.parquet").is_file()
    assert "259 tokenizer entries" in capsys.readouterr().out
