# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Exportable HF adapter preserving literal message content while inserting chat control IDs explicitly."""
from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer as RustTokenizer
from transformers import BatchEncoding, PreTrainedTokenizerFast

from .chat import CHAT_TEMPLATE, PROFILE, VOCAB_SIZE, build_literal_encoder, encode_chat


class LlamaChatTokenizer(PreTrainedTokenizerFast):
    """Use apply_chat_template(tokenize=True) for chat; plain encode always treats strings literally.

    tokenize=False returns a readable rendering only. Re-encoding that string cannot recover which spellings
    came from content, so it deliberately has plain-text semantics. The adapter is saved with the tokenizer.
    """

    @classmethod
    def convert_to_native_format(cls, trust_remote_code: bool = False, **kwargs: Any) -> dict[str, Any]:
        # Preserve the saved Rust pipeline; generic reconstruction may discard model-specific pretokenization.
        result = dict(kwargs)
        file = result.pop("tokenizer_file", None)
        if file is not None:
            result["tokenizer_object"] = RustTokenizer.from_file(file)
        return result

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["split_special_tokens"] = True
        super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]
        self.literal_profile = PROFILE
        from .profile import check_base_payload
        import json

        check_base_payload(json.loads(self.backend_tokenizer.to_str()))
        if len(self) != VOCAB_SIZE or self.chat_template != CHAT_TEMPLATE:
            raise ValueError("invalid literal chat tokenizer vocabulary/template")
        for spelling, expected in (("<s>", 1), ("</s>", 2), ("<unk>", 0), ("<user>", 32000), ("<assistant>", 32001)):
            if self.convert_tokens_to_ids(spelling) != expected:
                raise ValueError(f"incorrect tokenizer ID for {spelling}")
        self._literal = build_literal_encoder(self.backend_tokenizer, chat_body=True)

    def _encode_plus(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("split_special_tokens") is False:
            raise ValueError("this tokenizer requires literal text encoding; use apply_chat_template for chat structure")
        kwargs["split_special_tokens"] = True
        return super()._encode_plus(*args, **kwargs)

    def save_pretrained(self, save_directory: Any, *args: Any, **kwargs: Any) -> Any:
        from pathlib import Path
        from .profile import build_contract, CONTRACT_FILE
        import json

        previous = self.backend_tokenizer
        self._tokenizer = RustTokenizer.from_str(previous.to_str())
        self._tokenizer.no_padding()
        self._tokenizer.no_truncation()
        try:
            result = super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            self._tokenizer = previous
        path = Path(save_directory)
        contract = build_contract(path)
        (path / CONTRACT_FILE).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        return result

    def encode_literal(self, text: str) -> list[int]:
        return self._literal.encode(text, add_special_tokens=False).ids

    def apply_chat_template(
        self, conversation: Any, tools: Any = None, documents: Any = None, chat_template: str | None = None,
        add_generation_prompt: bool = False, continue_final_message: bool | str = False, tokenize: bool = True,
        padding: Any = False, truncation: bool = False, max_length: int | None = None, return_tensors: Any = None,
        return_dict: bool = False, return_assistant_tokens_mask: bool = False, tokenizer_kwargs: Any = None, **kwargs: Any,
    ) -> Any:
        if self.chat_template != CHAT_TEMPLATE or chat_template not in (None, CHAT_TEMPLATE):
            raise ValueError("chat template differs from the literal-aware tokenizer profile")
        if continue_final_message or truncation or kwargs or tools is not None or documents is not None or tokenizer_kwargs:
            raise ValueError("unsupported chat option; shorten complete exchanges explicitly, without assistant prefill or tools")
        batched = isinstance(conversation, list) and bool(conversation) and isinstance(conversation[0], list)
        rows = conversation if batched else [conversation]
        encoded = [encode_chat(row, self.encode_literal, generation_prompt=add_generation_prompt) for row in rows]
        if max_length is not None and any(len(row.ids) > max_length for row in encoded):
            raise ValueError("chat exceeds max_length; shorten complete exchanges explicitly")
        if not tokenize:
            if return_assistant_tokens_mask or return_tensors is not None:
                raise ValueError("chat masks/tensors require tokenize=True")
            return [row.text for row in encoded] if batched else encoded[0].text
        if return_assistant_tokens_mask and not return_dict:
            raise ValueError("return_assistant_tokens_mask requires return_dict=True")
        if padding or return_tensors is not None:
            result = self.pad({"input_ids": [row.ids for row in encoded]}, padding=padding, max_length=max_length)
            if return_assistant_tokens_mask:
                masks = []
                for row, ids in zip(encoded, result["input_ids"], strict=True):
                    tail = [0] * (len(ids) - len(row.ids))
                    mask = [int(value) for value in row.supervised]
                    masks.append(tail + mask if self.padding_side == "left" else mask + tail)
                result["assistant_masks"] = masks
            if return_tensors is not None:
                result.convert_to_tensors(return_tensors)
            elif not batched:
                result = BatchEncoding({key: value[0] for key, value in result.items()})
        else:
            result = BatchEncoding({"input_ids": [row.ids for row in encoded],
                                    "attention_mask": [[1] * len(row.ids) for row in encoded]})
            if return_assistant_tokens_mask:
                result["assistant_masks"] = [[int(value) for value in row.supervised] for row in encoded]
            if not batched:
                result = BatchEncoding({key: value[0] for key, value in result.items()})
        return result if return_dict else result["input_ids"]


LlamaChatTokenizer.register_for_auto_class("AutoTokenizer")  # type: ignore[no-untyped-call]
