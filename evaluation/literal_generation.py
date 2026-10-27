# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""HFLM generation with token EOS and only task-requested textual stops.

Use the harness's collation, option normalization, forward and postprocessing APIs.
Two narrow differences from HFLM 0.4.13: decoded EOS is not treated as a text stop (a literal '</s>' encoded
with ordinary tokens is valid generated content), and a completion is decoded after its prompt
(`decode_continuation`), so its leading whitespace survives SentencePiece decoding.
"""
from copy import deepcopy
from typing import Any


def decode_continuation(lm: Any, prompt: list[int], continuation: list[int]) -> str:
    """
    The text of `continuation` as it reads after `prompt`.

    SentencePiece tokenizers (the Llama family) drop the leading space of the first token of a decoded sequence, so
    decoding a completion on its own turns a function body indented by four spaces into three and the code no longer
    parses (HumanEval). Decoding it after the prompt and cutting off the prompt's text keeps every space. When the
    prompt's text is not a prefix of the joint text (a character split across the boundary), the completion is
    decoded on its own as before.
    """

    head: str = lm.tok_decode(prompt)
    whole: str = lm.tok_decode(prompt + continuation)
    if whole.startswith(head):
        return whole[len(head):]
    alone: str = lm.tok_decode(continuation)
    return alone


def generate_literal_until(lm: Any, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
    from lm_eval.models.utils import Collator, handle_stop_sequences, normalize_gen_kwargs, postprocess_generated_text
    from tqdm import tqdm  # type: ignore[import-untyped]

    if not requests:
        return []
    if lm.backend != "causal" or lm.think_end_token is not None:
        raise ValueError("literal generation supports causal models without thinking-token postprocessing")
    batch_size = lm._detect_batch_size() if lm.batch_size == "auto" else lm.batch_size
    collator = Collator(
        [deepcopy(request.args) for request in requests],
        sort_fn=lambda item: (-len(lm.tok_encode(item[0])), item[0]),
        group_by="gen_kwargs", group_fn=lambda item: item[1],
    )
    responses: list[str] = []
    with tqdm(total=len(requests), disable=disable_tqdm, desc="Running generate_until requests") as progress:
        for chunk in collator.get_batched(n=batch_size):
            contexts, options = zip(*chunk, strict=True)
            kwargs: dict[str, Any] = dict(normalize_gen_kwargs(options[0], lm.max_gen_toks))
            stops = handle_stop_sequences(kwargs.pop("until", None), eos=None)
            reserve = kwargs.pop("max_gen_toks")
            if reserve <= 0 or reserve >= lm.max_length:
                raise ValueError("max_gen_toks must be positive and smaller than the model context")
            ids, mask = lm.tok_batch_encode(contexts, left_truncate_len=lm.max_length - reserve, truncation=lm.truncation)
            cap = kwargs.pop("max_length", ids.shape[1] + reserve)
            if cap > lm.max_length or cap <= ids.shape[1]:
                raise ValueError("generation max_length must fit the model and leave room for a completion")
            kwargs["synced_gpus"] = False
            generated = lm._model_generate(
                context=ids.to(lm.device), attention_mask=mask.to(lm.device), stop=stops, max_length=cap, **kwargs,
            )
            for tokens, context, keep in zip(generated.tolist(), contexts, mask.bool().tolist(), strict=True):
                prompt = [token for token, kept in zip(tokens[:ids.shape[1]], keep, strict=True) if kept]  # no padding
                suffix = tokens[ids.shape[1]:]
                if lm.eot_token_id in suffix:
                    suffix = suffix[:suffix.index(lm.eot_token_id)]
                text = postprocess_generated_text(decode_continuation(lm, prompt, suffix), stop=stops, think_end_token=None)
                responses.append(text)
                lm.cache_hook.add_partial("generate_until", (context, options[0]), text)
                progress.update(1)
    return list(collator.get_original(responses))
