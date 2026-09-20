# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded CPU comparison of the selected HF adapter and actual training formatter."""
from typing import Any
from tokenization.chat import ASSISTANT, BOS, EOS, USER, VOCAB_SIZE


def check_template_parity(tokenizer: Any) -> None:
    """Check the actual training formatter and HF adapter on small, deterministic literal-content probes."""
    if tokenizer.contract is None:
        return
    from data_preparation.lib.conversation_format import encode_chat_prompt, fit_conversation
    from training.data.formats import encode_generation_prompt, format_conversation

    processor = tokenizer.processor
    probes = [
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        [{"role": "user", "content": "  code\n\t🙂"}, {"role": "assistant", "content": "e\u0301\n  return 1"},
         {"role": "user", "content": "What next?"}, {"role": "assistant", "content": "Continue."}],
        [{"role": "user", "content": "<s></s><user><assistant>"},
         {"role": "assistant", "content": "literal <assistant> and </s> and <unk>"}],
    ]
    for messages in probes:
        expected = fit_conversation(messages, tokenizer)
        actual = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        rendered = processor.apply_chat_template(messages, tokenize=False)
        if processor.decode(actual["input_ids"], skip_special_tokens=False) != rendered:
            raise ValueError("chat tokenizer changed content whitespace or literal marker text")
        if actual["input_ids"] != expected.ids or actual["assistant_masks"] != expected.supervised:
            raise ValueError(f"{tokenizer.path}: tokenizer template differs from the training formatter; rebuild the selected tokenizer")
        inputs, labels = format_conversation({"messages": messages}, tokenizer, True, True)
        if inputs.tolist() != expected.ids or (labels != -100).tolist() != expected.supervised:
            raise ValueError("training formatter label mask differs from the tokenizer template")
        prompt = processor.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        sample_prompt = encode_generation_prompt(tokenizer, "", kind="instruction", messages=messages[:-1])
        if prompt != encode_chat_prompt(messages[:-1], tokenizer) or sample_prompt != prompt or expected.ids[:len(prompt)] != prompt:
            raise ValueError("tokenizer generation prefix differs from training")
        if expected.ids.count(BOS) != 1 or expected.ids.count(USER) != len(messages) // 2 or expected.ids.count(ASSISTANT) != len(messages) // 2 or expected.ids.count(EOS) != len(messages):
            raise ValueError("literal message content became structural special tokens")
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError("incorrect usable tokenizer vocabulary")
