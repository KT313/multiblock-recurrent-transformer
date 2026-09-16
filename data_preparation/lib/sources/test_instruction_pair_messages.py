# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import CfgFactory
from data_preparation.lib.conversation_format import fit_conversation
from data_preparation.lib.dataset_config import SourceConfig
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.sources.conversations import OrphanAssistantOpening, SHAREGPT_EXCHANGE_POLICY
from data_preparation.lib.sources.converters import get_converter, sharegpt_quality
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.download import download, prepare_tokenizer
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer


@pytest.mark.parametrize(('converter', 'raw', 'question', 'answer'), [
    ('sharegpt_conversations', {'conversations': [
        {'from': 'system', 'value': '  context <s>\n'}, {'from': 'human', 'value': 'question <user>'},
        {'from': 'gpt', 'value': '  answer </s>'}, {'from': 'gpt', 'value': 'ignored later turn'},
    ]}, 'question <user>\n\n  context <s>\n', '  answer </s>'),
    ('first_two_turns', {'conversations': [{'value': '  question <assistant>'}, {'value': 'answer'}, {'value': 'ignored'}]},
     '  question <assistant>', 'answer'),
    ('instruction_input_output', {'instruction': 'question', 'input': '  context', 'output': '  answer\n'},
     'question\n\n  context', '  answer\n'),
])
def test_pair_adapters_keep_selected_content_and_legacy_output(
    converter: str, raw: dict[str, Any], question: str, answer: str,
) -> None:
    source = SourceConfig(kind='instruct', loader='local', path='unused', converter=converter)
    legacy = get_converter(source)
    adapted = get_converter(replace(source, instruction_format='messages'))
    assert legacy is not None and adapted is not None
    original = legacy(raw)
    assert set(original) == {'instruction', 'input', 'output'}
    assert adapted(raw) == {'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}
    assert legacy(raw) == original


def test_sharegpt_message_adapter_preserves_orphan_and_filter_policy() -> None:
    source = SourceConfig(kind='instruct', loader='local', path='unused', converter='sharegpt_conversations',
                          instruction_format='messages', filter='sharegpt_quality')
    convert = get_converter(source)
    assert convert is not None
    with pytest.raises(OrphanAssistantOpening):
        convert({'conversations': [{'from': 'gpt', 'value': 'orphan'}]})
    with pytest.raises(ValueError, match='missing turn'):
        convert({'conversations': [{'from': 'human', 'value': 'question'}]})
    row = {'conversations': [{'from': 'human', 'value': 'q' * 60}, {'from': 'gpt', 'value': 'a' * 60}]}
    assert sharegpt_quality(row)
    assert convert(row)['messages'][1]['content'] == 'a' * 60
    assert not sharegpt_quality({'conversations': [{'from': 'system', 'value': 'context'}, *row['conversations']]})


def test_sharegpt_message_identity_keeps_both_policies(cfg_factory: CfgFactory) -> None:
    source = SourceConfig(kind='instruct', loader='local', path='unused', converter='sharegpt_conversations', filter='sharegpt_quality')
    legacy = cfg_factory({'chat': source})
    chat = replace(legacy, sources={'chat': replace(source, instruction_format='messages')})
    assert legacy.raw_hash_payload('chat')['row_semantics'] == {'sharegpt_exchange': SHAREGPT_EXCHANGE_POLICY}
    semantics = chat.raw_hash_payload('chat')['row_semantics']
    assert set(semantics) == {'conversation', 'tokenizer', 'sharegpt_exchange'}
    assert legacy.raw_hash('chat') != chat.raw_hash('chat')
    changed = replace(chat, tokenizer=replace(chat.tokenizer, name='different'))
    assert changed.raw_hash('chat') != chat.raw_hash('chat')


@pytest.mark.parametrize('converter', ['sharegpt_conversations', 'first_two_turns', 'instruction_input_output'])
def test_pair_messages_download_build_and_count_with_real_stages(
    converter: str, tmp_path: Path, cfg_factory: CfgFactory, write_local: Callable[..., Path],
) -> None:
    raw = {'instruction': 'tok_1', 'output': 'tok_2', 'conversations': [
        {'from': 'human', 'value': 'tok_1'}, {'from': 'gpt', 'value': 'tok_2'},
        {'from': 'human', 'value': 'tok_3'}, {'from': 'gpt', 'value': 'tok_4'},
    ]}
    directory = tmp_path / 'input'
    write_local(directory, [raw], '.jsonl')
    source = SourceConfig(kind='instruct', loader='local', path=str(directory), converter=converter, instruction_format='messages')
    config = cfg_factory({'chat': source}, tokens=128, dataset_max_sequence_length=128)
    layout = DatasetLayout(tmp_path / 'dataset')
    prepare_tokenizer(config, layout)
    download(config, 'chat', layout, rows_needed=1)
    build_source(config, 'chat', layout, pass_workers=1)
    raw_rows = [row for file in layout.raw_dir('chat').glob('data-*.parquet') for row in pq.read_table(file).to_pylist()]
    rows = [row for file in layout.processed_dir('chat').glob('data-*.parquet') for row in pq.read_table(file).to_pylist()]
    expected = [{'role': 'user', 'content': 'tok_1'}, {'role': 'assistant', 'content': 'tok_2'}]
    assert len(raw_rows) == len(rows) == 1 and raw_rows[0]['messages'] == rows[0]['messages'] == expected
    tokenizer = SavedTokenizer(layout.tokenizer_dir(config.tokenizer.name))
    encoded = fit_conversation(expected, tokenizer)
    assert rows[0]['tokens'] == len(encoded.ids) and rows[0]['exchange_ends'] == encoded.exchange_ends
    assert [token for token, supervised in zip(encoded.ids, encoded.supervised, strict=True) if supervised] == tokenizer.encode_literal('tok_2') + [tokenizer.eos_id]
