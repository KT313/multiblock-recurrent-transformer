# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Keep structured chat provenance through lm-eval's string-only request interface."""
from __future__ import annotations

import json
from typing import Any

import torch

# Internal request transport only, never model input. JSON's parser identifies the exact continuation boundary.
CHAT_ENVELOPE = "\x00MBRT_CHAT_V1:"


def create_literal_harness_class(base: Any, *, chat: bool) -> Any:
    class LiteralHFLM(base):  # type: ignore[misc]
        def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
            if not chat or not add_generation_prompt:
                raise ValueError("literal chat benchmarks require a final user and an assistant generation prefix; prefill is unsupported")
            self.tokenizer.apply_chat_template(chat_history, tokenize=True, add_generation_prompt=True)
            return CHAT_ENVELOPE + json.dumps(chat_history, ensure_ascii=True, separators=(",", ":"))

        def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
            context_ids, answer_ids = super()._encode_pair(context, continuation)
            if chat and context.startswith(CHAT_ENVELOPE):
                if not answer_ids:
                    raise ValueError("chat likelihood requires a nonempty candidate")
                if len(context_ids) + len(answer_ids) - 1 > self.max_length:
                    raise ValueError("chat likelihood exceeds context; shorten few-shot history explicitly")
            return list(context_ids), list(answer_ids)

        def loglikelihood_rolling(self, requests: list[Any], **kwargs: Any) -> Any:
            if chat:
                raise ValueError("rolling likelihood has no chat protocol; select plain prompting")
            return super().loglikelihood_rolling(requests, **kwargs)

        def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
            from evaluation.literal_generation import generate_literal_until
            return generate_literal_until(self, requests, disable_tqdm)

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, left_truncate_len: int | None = None, **kwargs: Any) -> list[int]:
            if chat and string.startswith(CHAT_ENVELOPE):
                messages, end = json.JSONDecoder().raw_decode(string[len(CHAT_ENVELOPE):])
                ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                suffix = string[len(CHAT_ENVELOPE) + end:]
                if suffix:
                    ids = ids + self.tokenizer.encode_literal(suffix)
                if left_truncate_len is not None and len(ids) > left_truncate_len:
                    raise ValueError("chat benchmark prompt exceeds context; shorten few-shot history explicitly")
            else:
                # A literal '<s>' in a plain prompt is not evidence that BOS was already inserted.
                ids = self.tokenizer.encode(string, add_special_tokens=self.add_bos_token if add_special_tokens is None else add_special_tokens, **kwargs)
                if left_truncate_len:
                    ids = ids[-left_truncate_len:]
            return list(ids)

        def tok_batch_encode(self, strings: list[str], padding_side: str = "left", left_truncate_len: int | None = None, truncation: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            rows = [self.tok_encode(text, left_truncate_len=left_truncate_len) for text in strings]
            width = max(map(len, rows))
            ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, row in enumerate(rows):
                start = width - len(row) if padding_side == "left" else 0
                ids[index, start:start + len(row)] = torch.tensor(row, dtype=torch.long)
                mask[index, start:start + len(row)] = 1
            return ids, mask

    return LiteralHFLM
