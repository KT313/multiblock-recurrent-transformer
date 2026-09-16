# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validate actual tokenizer payloads and build the one supported immutable chat profile."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .chat import ASSISTANT, BASE_REPO, BASE_REVISION, BASE_SIZE, CHAT_TEMPLATE, FORMAT_VERSION, PROFILE, USER, VOCAB_SIZE

# Transformers 5 Llama conversion of the pinned upstream payload (unchanged vocab/merges, Metaspace first).
BASE_CORE_DIGEST = "8cc25fa9bd77ea2d63f5053e4090929e4a3f2f3456a04b9aa39b373523e5c4df"
CONTRACT_FILE = "tokenizer_contract.json"


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def check_base_payload(payload: dict[str, Any]) -> None:
    core = {key: payload[key] for key in ("model", "normalizer", "pre_tokenizer", "decoder")}
    if digest(core) != BASE_CORE_DIGEST:
        raise ValueError("tokenizer base payload differs from the pinned Llama 32K tokenizer")


def build_contract(path: Path) -> dict[str, Any]:
    payload, config = read_json(path / "tokenizer.json"), read_json(path / "tokenizer_config.json")
    check_base_payload(payload)
    template = (path / "chat_template.jinja").read_text(encoding="utf-8")
    if template != CHAT_TEMPLATE or config.get("literal_profile") != PROFILE or config.get("chat_template", CHAT_TEMPLATE) != CHAT_TEMPLATE:
        raise ValueError(f"{path}: missing or changed literal chat profile/template")
    if config.get("tokenizer_class") != "LlamaChatTokenizer" or config.get("split_special_tokens") is not True:
        raise ValueError(f"{path}: missing literal-aware tokenizer adapter/settings")
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(json.dumps(payload))
    vocabulary = backend.get_vocab()
    if len(vocabulary) != VOCAB_SIZE or set(vocabulary.values()) != set(range(VOCAB_SIZE)):
        raise ValueError(f"{path}: expected {VOCAB_SIZE} contiguous usable tokenizer IDs")
    for spelling, token_id in (("<s>", 1), ("</s>", 2), ("<unk>", 0), ("<user>", USER), ("<assistant>", ASSISTANT)):
        if vocabulary.get(spelling) != token_id:
            raise ValueError(f"{path}: incorrect token ID for {spelling}")
    added = {item["content"]: item for item in payload["added_tokens"]}
    for spelling in ("<user>", "<assistant>"):
        item = added[spelling]
        if not item["special"] or item["normalized"] or item["lstrip"] or item["rstrip"] or item["single_word"]:
            raise ValueError(f"{path}: invalid AddedToken flags for {spelling}")
    for key, expected in (("bos_token", "<s>"), ("eos_token", "</s>"), ("unk_token", "<unk>")):
        value = config.get(key)
        if isinstance(value, dict):
            value = value.get("content")
        if value != expected:
            raise ValueError(f"{path}: invalid {key}")
    pad = config.get("pad_token")
    if isinstance(pad, dict):
        pad = pad.get("content")
    if pad not in (None, "</s>"):
        raise ValueError("chat profile padding must use EOS")
    if (path / "chat_templates").exists():
        raise ValueError("chat profile does not support alternate template files")
    # Hash only stable semantic metadata; HF rewrites paths, class bookkeeping and library version on export.
    options = {key: config.get(key) for key in ("bos_token", "eos_token", "unk_token", "add_bos_token", "add_eos_token", "clean_up_tokenization_spaces", "split_special_tokens")}
    code = {name: (path / name).read_text(encoding="utf-8") for name in ("hf.py", "chat.py")}
    reference = Path(__file__).parent
    if any(text != (reference / name).read_text(encoding="utf-8") for name, text in code.items()):
        raise ValueError(f"{path}: tokenizer adapter version differs from this formatter; rerun tokenizer preparation")
    return {"schema_version": 1, "profile": PROFILE, "format": FORMAT_VERSION, "base_size": BASE_SIZE,
            "vocab_size": VOCAB_SIZE, "special_ids": {"bos": 1, "eos": 2, "user": USER, "assistant": ASSISTANT},
            "digest": digest({"schema": 1, "payload": payload, "options": options, "template": template, "code": code}),
            "relative_path": "tokenizer"}


def validate_profile(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    config = read_json(path / "tokenizer_config.json")
    if config.get("literal_profile") is None:
        if (path / CONTRACT_FILE).exists() or config.get("tokenizer_class") == "LlamaChatTokenizer":
            raise ValueError(f"{path}: incomplete literal tokenizer profile")
        return None
    expected = read_json(path / CONTRACT_FILE)
    actual = build_contract(path)
    if expected != actual:
        raise ValueError(f"{path}: tokenizer contract does not match actual payload; rerun tokenizer preparation")
    return actual


def load_processor(path: str | Path, *, add_bos_token: bool | None = None) -> Any:
    """Load the known adapter directly; never execute unverified tokenizer code from a dataset directory."""
    from transformers import AutoTokenizer

    if validate_profile(path) is None:
        options = {"add_bos_token": add_bos_token, "add_eos_token": False} if add_bos_token is not None else {}
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True, **options)
    from .hf import LlamaChatTokenizer

    processor = LlamaChatTokenizer.from_pretrained(str(path), local_files_only=True)
    return processor


def build_profile(destination: Path, cached_base: Path, *, token: str | None = None) -> None:
    """Use a verified cached base offline, otherwise obtain the pinned base through HF's cache."""
    from transformers import AddedToken, AutoTokenizer
    from .hf import LlamaChatTokenizer

    base = None
    if (cached_base / "tokenizer.json").is_file():
        try:
            check_base_payload(read_json(cached_base / "tokenizer.json"))
            candidate = AutoTokenizer.from_pretrained(str(cached_base), local_files_only=True)
            if len(candidate) == BASE_SIZE:
                base = candidate
        except (ValueError, OSError):
            pass  # an unverified cache is not accepted; obtain the pinned upstream instead
    if base is None:
        base = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REVISION, token=token)
    check_base_payload(json.loads(base.backend_tokenizer.to_str()))
    if len(base) != BASE_SIZE:
        raise ValueError("expected an unextended 32000-token base")
    original = base.get_vocab()
    base.add_special_tokens({"additional_special_tokens": [AddedToken(value, normalized=False, special=True)
                                                           for value in ("<user>", "<assistant>")]})
    updated = base.get_vocab()
    if any(updated.get(key) != value for key, value in original.items()):
        raise ValueError("adding chat tokens changed existing IDs")
    processor = LlamaChatTokenizer(tokenizer_object=base.backend_tokenizer, bos_token="<s>", eos_token="</s>",
                                   unk_token="<unk>", additional_special_tokens=["<user>", "<assistant>"],
                                   add_bos_token=True, add_eos_token=False, clean_up_tokenization_spaces=False,
                                   chat_template=CHAT_TEMPLATE, literal_profile=PROFILE)
    processor.save_pretrained(destination)
    contract = build_contract(destination)
    (destination / CONTRACT_FILE).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    validate_profile(destination)


def copy_profile(source: Path, destination: Path) -> None:
    """Copy the small immutable artifact without reserializing its payload."""
    contract = validate_profile(source)
    if contract is None:
        raise ValueError("cannot publish a legacy tokenizer as a chat profile")
    shutil.copytree(source, destination)
    if validate_profile(destination) != contract:
        raise ValueError("tokenizer copy changed its contract")


def publish_profile(temporary: Path, destination: Path) -> None:
    """Publish under the preparation lock, preserving the old directory if replacement fails."""
    import os

    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        raise ValueError(f"unfinished tokenizer publication at {backup}; recover it before replacing artifacts")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def recover_profile(destination: Path) -> None:
    """Recover an interrupted profile publication under the same preparation lock."""
    import os

    backup = destination.with_name(f".{destination.name}.previous")
    if not backup.exists():
        return
    if not destination.exists():
        os.replace(backup, destination)
        return
    if validate_profile(destination) is None:
        raise ValueError("replacement tokenizer has no profile; preserving previous publication")
    shutil.rmtree(backup)


def inspect_processor_contract(processor: Any) -> dict[str, Any] | None:
    """Validate a supplied in-memory processor privately before export changes any existing output."""
    from tempfile import TemporaryDirectory

    if getattr(processor, "literal_profile", None) is None:
        return None
    with TemporaryDirectory(prefix="mbrt-tokenizer-export-") as temporary:
        path = Path(temporary)
        processor.save_pretrained(path)
        return validate_profile(path)
