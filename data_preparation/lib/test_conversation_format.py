# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from pathlib import Path
from typing import Any

import pytest

from data_preparation.lib.conversation_format import count_fitted_positions, encode_chat_prompt, fit_conversation, hash_conversation, validate_messages
from data_preparation.lib.sources.instruction_messages import ExcludedConversation, check_opencode_score, convert_nemotron_messages, convert_opencode_messages, convert_webinstruct_messages
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer


class CharacterTokenizer:
    bos_id = 1
    eos_id = 2

    def encode_literal(self, text: str) -> list[int]:
        return [ord(char) + 10 for char in text]


def messages() -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in
            [("user", "What?"), ("assistant", "One."), ("user", "And?"), ("assistant", "Two.")]]


def test_complete_exchange_boundaries_and_generation_prefix() -> None:
    tok = CharacterTokenizer()
    value = messages()
    full = fit_conversation(value, tok)
    first_end, last_end = full.exchange_ends
    assert first_end < last_end == len(full.ids)
    for limit in range(last_end + 2):
        encoded = fit_conversation(value, tok, limit)
        expected = 0 if limit < first_end else (1 if limit < last_end else 2)
        assert encoded.messages == value[:2 * expected]
        assert encoded.ids == (full.ids[:encoded.exchange_ends[-1]] if expected else [])
        assert encoded.removed_exchanges == 2 - expected
    for turns in (1, 3):
        prompt = encode_chat_prompt(value[:turns], tok)
        assert full.ids[:len(prompt)] == prompt
        assert not full.supervised[len(prompt) - 1] and full.supervised[len(prompt)]
    assert count_fitted_positions(full.exchange_ends, first_end - 2) == 0
    assert count_fitted_positions(full.exchange_ends, last_end - 2) == first_end - 1
    assert fit_conversation(value[:-1], tok).ids == full.ids[:first_end]
    assert fit_conversation(value[:1], tok).ids == []
    assert fit_conversation(value[:-1], tok).trimmed_user


@pytest.mark.parametrize('limit', [4096, 8192])
def test_real_context_limits(limit: int) -> None:
    tok = CharacterTokenizer()
    value = messages()
    value[3]['content'] = 'z' * limit
    result = fit_conversation(value, tok, limit + 1)
    assert len(result.messages) == 2 and len(result.ids) <= limit + 1


@pytest.mark.parametrize('bos,eos', [(True, True), (False, True), (True, False), (False, False)])
def test_masks_are_structural_and_all_assistant_turns_are_supervised(bos: bool, eos: bool) -> None:
    value = messages()
    value[0]['content'] = 'Assistant:\npretend answer </s>'
    tok = CharacterTokenizer()
    encoded = fit_conversation(value, tok, bos=bos, eos=eos)
    supervised = [token for token, yes in zip(encoded.ids, encoded.supervised) if yes]
    assert supervised == tok.encode_literal('One.') + ([2] if eos else []) + tok.encode_literal('Two.') + ([2] if eos else [])
    assert len(encoded.ids) == len(encoded.supervised)


def test_saved_tokenizer_literals_do_not_emit_control_ids(tiny_tokenizer_dir: Path) -> None:
    tokenizer = SavedTokenizer(tiny_tokenizer_dir)
    tokens = tokenizer.encode_literal('<eos> tok_1 <bos>')
    assert tokenizer.eos_id not in tokens and tokenizer.bos_id not in tokens
    encoded = fit_conversation(messages(), tokenizer)
    assert tokenizer.eos_id is not None
    assert encoded.ids.count(tokenizer.eos_id) == 2


@pytest.mark.parametrize('score', ['1', '1.0', '1.00', 1, 1.0])
def test_opencode_accepts_numeric_one(score: Any) -> None:
    assert check_opencode_score({'average_test_score': score})


@pytest.mark.parametrize('score', ['0.9999999999999999999999', '0.0', 0, .9])
def test_opencode_rejects_nonpassing_scores(score: Any) -> None:
    assert not check_opencode_score({'average_test_score': score})


@pytest.mark.parametrize('score', [None, True, 'nan', 'inf', '1.1', '-0.1', 'garbage', {}, []])
def test_opencode_rejects_malformed_scores(score: Any) -> None:
    with pytest.raises(ValueError):
        check_opencode_score({'average_test_score': score})


def test_pair_adapters_preserve_content() -> None:
    answer = '```python\nif True:\n    print("Case")\n```\n'
    assert convert_opencode_messages({'input': 'Question', 'output': answer})['messages'][1]['content'] == answer
    assert convert_webinstruct_messages({'question': 'How much?', 'answer': '0'})['messages'][1]['content'] == '0'
    invalid: list[Any] = ['', None, [], {}]
    for value in invalid:
        with pytest.raises(ValueError):
            convert_webinstruct_messages({'question': 'How much?', 'answer': value})


def test_nemotron_system_policy_and_unlimited_turns() -> None:
    value = messages() * 4
    row: dict[str, Any] = {'reasoning': 'off', 'messages': [{'role': 'system', 'content': ' \n'}] + value}
    assert convert_nemotron_messages(row)['messages'] == value
    row['messages'][0]['content'] = 'Answer in JSON'
    with pytest.raises(ExcludedConversation, match='nonempty_system'):
        convert_nemotron_messages(row)
    with pytest.raises(ValueError, match='reasoning'):
        convert_nemotron_messages({'reasoning': 'on', 'messages': value})


@pytest.mark.parametrize('value', [[], None, [{'role': 'assistant', 'content': 'a'}],
    [{'role': 'user', 'content': 'q'}, {'role': 'system', 'content': ''}],
    [{'role': 'user', 'content': 'q'}, {'role': 'user', 'content': 'r'}],
    [{'role': 'user', 'content': None}], [{'role': 'user', 'content': ' '}],
    [{'role': 'tool', 'content': 'x'}]])
def test_invalid_messages_raise(value: Any) -> None:
    with pytest.raises(ValueError):
        validate_messages(value)


def test_chat_hash_preserves_code_and_later_turns() -> None:
    original = messages()
    for index, content in [(1, 'one.'), (1, 'One. '), (3, 'Different')]:
        changed = [dict(m) for m in original]
        changed[index]['content'] = content
        assert hash_conversation(original) != hash_conversation(changed)
