# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""HFLM generation with token EOS and only task-requested textual stops.

Use the harness's collation, option normalization, forward and postprocessing APIs.
The narrow difference from HFLM 0.4.13 is not treating decoded EOS as a text stop:
a literal '</s>' encoded with ordinary tokens is valid generated content.
"""
from copy import deepcopy
from typing import Any


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
            for tokens, context in zip(generated.tolist(), contexts, strict=True):
                suffix = tokens[ids.shape[1]:]
                if lm.eot_token_id in suffix:
                    suffix = suffix[:suffix.index(lm.eot_token_id)]
                text = postprocess_generated_text(lm.tok_decode(suffix), stop=stops, think_end_token=None)
                responses.append(text)
                lm.cache_hook.add_partial("generate_until", (context, options[0]), text)
                progress.update(1)
    return list(collator.get_original(responses))
