# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pyarrow.parquet as pq
import pytest

from data_preparation.conftest import CfgFactory, FakeHub, REPO, REV
from data_preparation.lib.build.runner import prepare
from data_preparation.lib.build.planner import measured_tokens_per_row
from data_preparation.lib.conversation_format import count_fitted_positions, fit_conversation, hash_conversation
from data_preparation.lib.dataset_config import SourceConfig, load_dataset_config
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.stages.download import download, prepare_tokenizer
from data_preparation.lib.stages.global_dedup import global_key
from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

Writer = Callable[..., Path]
ConfigFile = Callable[..., Path]


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [row for file in sorted(path.glob('data-*.parquet')) for row in pq.read_table(file).to_pylist()]


def chat(index: int) -> list[dict[str, str]]:
    return [{'role': 'user', 'content': f'tok_{index} tok_1'}, {'role': 'assistant', 'content': 'tok_2 tok_3'},
            {'role': 'user', 'content': 'tok_4'}, {'role': 'assistant', 'content': 'tok_5 tok_6'}]


def test_message_sources_prepare_shuffle_global_dedup_and_reuse(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, config_file: ConfigFile,
) -> None:
    code = [{'input': f'tok_{i}', 'output': '  tok_2\n', 'average_test_score': score}
            for i, score in enumerate(['0', '1', '1.0', '0.9', '1'])]
    web = [{'question': f'tok_{i}', 'answer': 'tok_7'} for i in range(10, 20)]
    nemo = [{'reasoning': 'off', 'messages': [{'role': 'system', 'content': ''}] + chat(i)} for i in range(20, 30)]
    nemo += [nemo[0], {'reasoning': 'off', 'messages': [{'role': 'system', 'content': 'constraint'}] + chat(40)}]
    sources = {}
    for name, converter, rows in [('code', 'opencode_messages', code), ('web', 'webinstruct_messages', web), ('nemo', 'nemotron_messages', nemo)]:
        path = tmp_path / name
        write_local(path, rows, '.jsonl')
        sources[name] = SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(path),
                                     converter=converter, filter='opencode_passed_tests' if name == 'code' else None)
    sources['text'] = SourceConfig(kind='pretrain', loader='synthetic')
    cfg = cfg_factory(sources, dataset_max_sequence_length=256, tokens=256, tokens_per_row=32,
                      bloom_deduplicate_across_sources=True)
    path = config_file(cfg)
    root = tmp_path / 'dataset'
    result = prepare(path, root, num_workers=1, pass_workers=1, assume_yes=True)
    assert result.complete
    layout = DatasetLayout(root).for_config(cfg)
    tokenizer = SavedTokenizer(layout.tokenizer_dir(cfg.tokenizer.name))
    for name in ['code', 'web', 'nemo']:
        rows = read_rows(layout.processed_dir(name))
        assert rows
        for row in rows:
            assert row['messages'][-1]['role'] == 'assistant'
            assert row['hash'] == row['global_hash'] == hash_conversation(row['messages'])
            encoded = fit_conversation(row['messages'], tokenizer)
            assert row['tokens'] == len(encoded.ids) and row['exchange_ends'] == encoded.exchange_ends
        schema = pq.read_schema(next(layout.processed_dir(name).glob('data-*.parquet')))
        assert schema.field('messages').type.value_type.field('role').type == schema.field('messages').type.value_type.field('content').type
    before = {p: p.stat().st_mtime_ns for p in root.rglob('data-*.parquet')}
    assert prepare(path, root, num_workers=1, pass_workers=1, assume_yes=True).complete
    assert before == {p: p.stat().st_mtime_ns for p in root.rglob('data-*.parquet')}


def test_download_resume_and_target_counts_use_complete_exchanges(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer,
) -> None:
    source_dir = tmp_path / 'input'
    write_local(source_dir, [{'reasoning': 'off', 'messages': chat(i)} for i in range(10)], '.jsonl')
    src = SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(source_dir), converter='nemotron_messages')
    cfg = cfg_factory({'chat': src}, dataset_max_sequence_length=128, tokens=100)
    layout = DatasetLayout(tmp_path / 'data')
    prepare_tokenizer(cfg, layout)
    download(cfg, 'chat', layout, rows_needed=3, shard_size=2)
    first = read_rows(layout.raw_dir('chat'))
    download(cfg, 'chat', layout, rows_needed=8, shard_size=2)
    rows = read_rows(layout.raw_dir('chat'))
    assert len(rows) == 8 and rows[:3] == first
    assert [r['messages'] for r in rows] == [chat(i) for i in range(8)]
    raw = Manifest.load(layout.raw_dir('chat'))
    assert raw is not None
    target = rows[0]['exchange_ends'][0] - 1
    shorter = replace(cfg, training_target_sequence_length=target)
    expected = sum(count_fitted_positions(r['exchange_ends'], target) for r in rows) / len(rows)
    assert measured_tokens_per_row(shorter, 'chat', layout, raw) == expected
    assert expected < target + 1 < rows[0]['tokens']
    assert read_rows(layout.raw_dir('chat')) == rows


def test_web_file_glob_excludes_test_and_legacy(hub: FakeHub, cfg_factory: CfgFactory, layout: DatasetLayout) -> None:
    for file, question in [('train-0.parquet', 'tok_1'), ('train_legacy-0.parquet', 'tok_2'), ('test-0.parquet', 'tok_3')]:
        hub.add('data/' + file, [{'question': question, 'answer': 'tok_4'}])
    source = SourceConfig(kind='instruct', instruction_format='messages', loader='hf_files', hf_id=REPO, revision=REV,
                          load_kwargs={'data_files': 'data/train-*.parquet'}, converter='webinstruct_messages')
    cfg = cfg_factory({'web': source})
    prepare_tokenizer(cfg, layout)
    download(cfg, 'web', layout, rows_needed=3)
    rows = read_rows(layout.raw_dir('web'))
    assert len(rows) == 1 and rows[0]['messages'][0]['content'] == 'tok_1'


def test_smoke_config_and_legacy_identity() -> None:
    config = load_dataset_config(Path('config/datasets/instruction_sources_smoke.yaml'))
    assert sum(s.tokens for s in config.stages) == 524288
    assert config.sources['fineweb_edu_350bt_smoke'].load_kwargs['data_files'] == 'sample/350BT/*.parquet'
    legacy = load_dataset_config(Path('config/datasets/tiny.yaml'))
    payload = legacy.raw_hash_payload('synthetic_instruct')
    assert 'instruction_format' not in payload['source'] and 'row_semantics' not in payload
    assert global_key('instruct', {'instruction': 'q', 'input': '', 'output': 'a'}) != global_key('messages', {'messages': chat(0)})
    with pytest.raises(ValueError, match='input_inversions'):
        replace(config.sources['nemotron_chat_off_smoke'], input_inversions=.1)
    with pytest.raises(ValueError, match='token_count'):
        replace(config, token_count='estimate')


def test_chat_tokenizer_identity_and_unsupported_benchmark_policy() -> None:
    cfg = load_dataset_config(Path('config/datasets/instruction_sources_smoke.yaml'))
    changed = replace(cfg, tokenizer=replace(cfg.tokenizer, revision='different'))
    assert cfg.raw_hash('nemotron_chat_off_smoke') != changed.raw_hash('nemotron_chat_off_smoke')
    assert cfg.raw_hash('fineweb_edu_350bt_smoke') == changed.raw_hash('fineweb_edu_350bt_smoke')
    with pytest.raises(ValueError, match='benchmark Bloom exclusion'):
        replace(cfg, bloom_deduplicate_across_sources_add_benchmarks=['arc_challenge'])


def test_message_storage_trims_complete_exchanges_and_counts_exclusions(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    from data_preparation.lib.stages.build import build_source

    directory = tmp_path / 'input'
    trailing = chat(0) + [{'role': 'user', 'content': 'unanswered'}]
    long = chat(1)
    long[-1]['content'] = ' '.join(['tok_3'] * 300)
    nofit = [{'role': 'user', 'content': ' '.join(['tok_4'] * 300)}, {'role': 'assistant', 'content': 'tok_1'}]
    rows = [{'reasoning': 'off', 'messages': value} for value in [trailing, long, nofit]]
    rows.append({'reasoning': 'off', 'messages': [{'role': 'system', 'content': 'obey'}] + chat(2)})
    write_local(directory, rows, '.jsonl')
    source = SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(directory), converter='nemotron_messages')
    cfg = cfg_factory({'chat': source}, dataset_max_sequence_length=64, tokens=64)
    layout = DatasetLayout(tmp_path / 'dataset')
    prepare_tokenizer(cfg, layout)
    caplog.set_level(logging.INFO)
    download(cfg, 'chat', layout, rows_needed=10)
    raw = read_rows(layout.raw_dir('chat'))
    assert [r['messages'] for r in raw] == [chat(0), chat(1)[:2]]
    manifest = Manifest.load(layout.raw_dir('chat'))
    assert manifest is not None and manifest.dropped_too_long == 1 and manifest.skipped_malformed == 0
    assert 'nonempty_system' in caplog.text and 'trimmed_trailing_user' in caplog.text
    assert 'no_fitting_exchange' in caplog.text and 'removed_exchanges' in caplog.text
    lower = raw[0]['exchange_ends'][0]
    shorter = replace(cfg, dataset_max_sequence_length=lower, training_target_sequence_length=lower)
    build_source(shorter, 'chat', layout, pass_workers=1)
    processed = read_rows(layout.processed_dir('chat'))
    assert len(processed) == 2 and all(len(r['messages']) == 2 for r in processed)
    assert all(r['tokens'] <= lower for r in processed)


def test_interrupted_chat_download_resumes_exactly(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer,
) -> None:
    from data_preparation.lib.abort import BuildAborted

    directory = tmp_path / 'input'
    write_local(directory, [{'reasoning': 'off', 'messages': chat(i)} for i in range(16)], '.jsonl')
    source = SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(directory), converter='nemotron_messages')
    cfg = cfg_factory({'chat': source}, dataset_max_sequence_length=64, tokens=64)
    interrupted, full = DatasetLayout(tmp_path / 'interrupted'), DatasetLayout(tmp_path / 'full')
    for layout in (interrupted, full):
        prepare_tokenizer(cfg, layout)
    def stop() -> bool:
        return len(list(interrupted.raw_dir('chat').glob('data-*.parquet'))) >= 1
    with pytest.raises(BuildAborted):
        download(cfg, 'chat', interrupted, rows_needed=12, shard_size=2, should_stop=stop)
    download(cfg, 'chat', interrupted, rows_needed=12, shard_size=2)
    download(cfg, 'chat', full, rows_needed=12, shard_size=2)
    assert read_rows(interrupted.raw_dir('chat')) == read_rows(full.raw_dir('chat'))
