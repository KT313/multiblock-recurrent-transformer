# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Validate export metadata, transfer weights and package flat remote-code modules."""

import re
from copy import deepcopy
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..model import RecurrentGPT

def validate_special_token_id(name: str, value: object, vocab_size: int) -> int | list[int] | None:
    """Validate against real vocabulary rows; only EOS supports a nonempty list of IDs."""
    if value is None:
        return None
    if name == "eos_token_id" and isinstance(value, list) and value:
        return [validate_single_token_id(name, item, vocab_size) for item in value]
    return validate_single_token_id(name, value, vocab_size)


def validate_single_token_id(name: str, value: object, vocab_size: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < vocab_size:
        raise ValueError(f"{name} must contain integer IDs in the real vocabulary [0, {vocab_size}); got {value!r}")
    return value


def resolve_special_token_ids(
    vocab_size: int, tokenizer: PreTrainedTokenizerBase | None = None, *,
    bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None,
    pad_token_id: int | None = None, allow_missing_generation_metadata: bool = False,
) -> dict[str, int | list[int] | None]:
    """Pure metadata resolution. None means unspecified, so it never clears a tokenizer's existing ID."""
    resolved: dict[str, int | list[int] | None] = {}
    for name, value in (("bos_token_id", bos_token_id), ("eos_token_id", eos_token_id), ("pad_token_id", pad_token_id)):
        explicit = validate_special_token_id(name, value, vocab_size)
        saved = validate_special_token_id(name, getattr(tokenizer, name, None), vocab_size)
        if explicit is not None and saved is not None:
            # A singleton EOS list and its scalar are equivalent; preserve the explicit representation on export.
            explicit_ids = explicit if isinstance(explicit, list) else [explicit]
            saved_ids = saved if isinstance(saved, list) else [saved]
            if explicit_ids != saved_ids:
                raise ValueError(f"{name} conflicts: explicit {explicit!r}, tokenizer {saved!r}")
        resolved[name] = saved if explicit is None else explicit
    if resolved["eos_token_id"] is None and not allow_missing_generation_metadata:
        raise ValueError(
            "HF export requires eos_token_id from a tokenizer or explicit IDs for EOS stopping; "
            "pass allow_missing_generation_metadata=True for an intentional model-only export"
        )
    if tokenizer is not None:
        known_ids = set(tokenizer.get_vocab().values())
        for name, value in resolved.items():
            token_ids = value if isinstance(value, list) else ([] if value is None else [value])
            if any(token_id not in known_ids for token_id in token_ids):
                raise ValueError(f"{name} {value!r} contains IDs unknown to the tokenizer")
            if len(token_ids) > 1:
                raise ValueError("a tokenizer has one EOS token; use explicit-only export for multiple EOS IDs")
    return resolved


# Tokenizer metadata plus common fast/slow tokenizer artifacts. Refuse ambiguous reuse; never delete these files.
_TOKENIZER_EXPORT_ARTIFACTS = (
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "added_tokens.json",
    "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
    "chat_template.jinja", "chat_templates", "tokenizer_contract.json", "hf.py", "chat.py", "profile.py",
)
# One `from .x import` / `from ..x.y import` line: leading whitespace, the dots, the dotted module name.
_RELATIVE_IMPORT = re.compile(r"^(?P<indent>[ \t]*)from[ \t]+(?P<dots>\.+)(?P<name>[\w.]*)[ \t]+import\b", re.MULTILINE)


def flat_module_name(module: Path) -> str:
    """
    Top-level name of a package module in the export folder: `layers/norms.py` -> `layers_norms`.
    """

    return "_".join(module.with_suffix("").parts)


def flatten_relative_imports(source: str, module: Path, package_dir: Path) -> str:
    """
    Rewrite the package-relative imports of `module` (its path relative to `package_dir`) for the flat export.

    Only `from .x import` lines change: `from .layers.norms import X` -> `from .layers_norms import X`, `from ..config
    import Y` -> `from .config import Y`. An import that resolves to a package (`from .layers import X` or
    `from . import x`) raises: `__init__.py` files are not exported, so imports must name the defining module.
    """

    # Directory of `module` inside the package, e.g. ("hf",) for hf/modeling.py or () for a top-level module.
    module_package = module.parent.parts

    def rewrite(match: re.Match[str]) -> str:
        line = match[0].strip()
        num_dots = len(match["dots"])
        levels_up = num_dots - 1  # `.` = same package, `..` = one package up, ...
        if not match["name"] or levels_up > len(module_package):
            # `from . import x` names no module; more dots than packages would leave the package.
            raise ValueError(f"{module}: {line!r} cannot be flattened (import from the defining module)")

        base_package = module_package[: len(module_package) - levels_up]
        target_module = (*base_package, *match["name"].split("."))
        if not package_dir.joinpath(*target_module).with_suffix(".py").is_file():
            raise ValueError(f"{module}: {line!r} does not name a module file (import from the defining module)")
        flat_name = "_".join(target_module)
        return f"{match['indent']}from .{flat_name} import"

    return _RELATIVE_IMPORT.sub(rewrite, source)


def export_sources(package_dir: Path, out_dir: Path) -> list[Path]:
    """
    Write every module of the package (no `__init__.py`, tests or `__pycache__`) flat into `out_dir`.
    """

    written: list[Path] = []
    for source in sorted(package_dir.rglob("*.py")):
        module = source.relative_to(package_dir)
        if "__pycache__" in module.parts or module.name == "__init__.py" or module.name.startswith("test_"):
            continue
        target = out_dir / f"{flat_module_name(module)}.py"
        if target in written:
            raise ValueError(f"{module} and another module both flatten to {target.name}")
        text = flatten_relative_imports(source.read_text(encoding="utf-8"), module, package_dir)
        text = re.sub(r"from tokenization\.(\w+) import", r"from .tokenization_\1 import", text)
        target.write_text(text, encoding="utf-8")
        written.append(target)
    tokenizer_package = package_dir.parent / "tokenization"
    if tokenizer_package.is_dir():
        for source in sorted(tokenizer_package.glob("*.py")):
            if source.name == "__init__.py" or source.name.startswith("test_"):
                continue
            text = re.sub(r"from \.(\w+) import", r"from .tokenization_\1 import", source.read_text(encoding="utf-8"))
            # Setup-only parity checks are not part of exported model imports.
            target = out_dir / f"tokenization_{source.name}"
            target.write_text(text, encoding="utf-8")
            written.append(target)
    return written



def prepare_export_metadata(
    vocab_size: int, out_dir: str | Path, tokenizer_dir: str | Path | None, tokenizer: PreTrainedTokenizerBase | None,
    bos_token_id: int | None, eos_token_id: int | list[int] | None, pad_token_id: int | None,
    allow_missing_generation_metadata: bool,
) -> tuple[Path, PreTrainedTokenizerBase | None, dict[str, int | list[int] | None]]:
    if tokenizer_dir is not None and tokenizer is not None:
        raise ValueError("supply either tokenizer_dir or tokenizer, not both")
    out_dir = Path(out_dir)
    if tokenizer_dir is None and tokenizer is None:
        existing = [name for name in _TOKENIZER_EXPORT_ARTIFACTS if (out_dir / name).exists() or (out_dir / name).is_symlink()]
        if existing:
            raise ValueError(
                f"HF export destination {out_dir} contains tokenizer artifacts ({', '.join(existing)}); "
                "use a fresh directory or supply a tokenizer to avoid stale special-token metadata"
            )
    if tokenizer_dir is not None:
        from tokenization.profile import load_processor

        tokenizer = load_processor(tokenizer_dir)
    elif tokenizer is not None:
        tokenizer = deepcopy(tokenizer)  # metadata resolution and serialization use this one independent instance
    metadata = resolve_special_token_ids(
        vocab_size, tokenizer, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
        pad_token_id=pad_token_id, allow_missing_generation_metadata=allow_missing_generation_metadata,
    )
    if tokenizer is not None:
        # Fill missing special-token roles without adding tokens or mutating the caller's tokenizer.
        for name, value in metadata.items():
            if value is not None and getattr(tokenizer, name) is None:
                token_id = value[0] if isinstance(value, list) else value
                setattr(tokenizer, name.removesuffix("_id"), tokenizer.convert_ids_to_tokens(token_id))
                if getattr(tokenizer, name) != token_id:
                    raise ValueError(f"tokenizer cannot represent {name}={token_id}")

    return out_dir, tokenizer, metadata


def transfer_export_weights(model: RecurrentGPT, hf_model: PreTrainedModel) -> None:
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        state_dict[f"model.{name}"] = tensor.detach().cpu()
    state_dict["model.freqs_cis"] = model.freqs_cis.detach().cpu()  # persistent in the wrapper only (see its __init__)
    hf_model.load_state_dict(state_dict, assign=True)
