# (c) 2025-2026 Tobias Kerner. Apache-2.0.
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from data_preparation.conftest import CfgFactory, FakeHub, REPO, REV
from data_preparation.lib.dataset_config import SourceConfig, ProcessingConfig
from data_preparation.inspection.pipeline import inspect_pipeline
from data_preparation.inspection.artifacts import read_rows
from training.data.packing import pack_samples
from training.data.tokenizer import Tokenizer

ConfigFile = Callable[..., Path]
Writer = Callable[..., Path]


def test_inspection_runs_actual_filters_global_dedup_and_packing(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, config_file: ConfigFile,
) -> None:
    inputs = tmp_path / 'input'
    rows = [{'input': f'tok_{i}', 'output': 'tok_4 tok_5', 'average_test_score': score}
            for i, score in enumerate(['0', '1', '1', '0.8', '1', '1', '1'])]
    write_local(inputs, rows, '.jsonl')
    sources = {
        'code': SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(inputs),
                             converter='opencode_messages', filter='opencode_passed_tests'),
        'text': SourceConfig(kind='pretrain', loader='synthetic'),
    }
    config = cfg_factory(sources, dataset_max_sequence_length=64, tokens=256, bloom_deduplicate_across_sources=True)
    path = config_file(config)
    original = path.read_bytes()
    output = tmp_path / 'output'
    output.mkdir()
    report = inspect_pipeline(path, output, count=4, packs_per_stage=2, pack_length=128)
    assert report['status'] == 'complete'
    assert path.read_bytes() == original
    assert report['sources']['code']['raw_rows'] == report['sources']['text']['raw_rows'] == 4
    assert report['sources']['code']['source_rows_examined'] == 6
    assert report['sources']['code']['training_samples'] > 0
    assert (output / 'raw/code.jsonl').is_file() and (output / 'prepared/code.jsonl').is_file()
    formatted = json.loads((output / 'formatted/code.json').read_text())
    assert formatted[0]['split'] == 'validation' and formatted[-1]['split'] == 'train'
    tokenizer = Tokenizer(output / 'dataset/tokenizers/synthetic')
    samples = {record['id']: (torch.tensor([row[0] for row in record['tokens']]),
                             torch.tensor([row[1] for row in record['tokens']]), record['id']) for record in formatted}
    for record in formatted:
        assert 'input_ids' not in record and 'labels' not in record and 'loss_mask' not in record and 'decoded' not in record
        for token, label, mask, decoded in record['tokens']:
            assert mask == (label != -100)
            assert decoded == tokenizer.decode([token], skip_special_tokens=False)
    first_row = formatted[0]['tokens'][0]
    assert first_row == [tokenizer.bos_id, -100, False, '<bos>']
    assert json.dumps(first_row) in (output / 'formatted/code.json').read_text()
    for file in sorted((output / 'packed/sources/code').glob('*.pt')):
        saved = torch.load(file, weights_only=True)
        expected = pack_samples([samples[name] for name in saved['data_ids']], 128, tokenizer)
        for key, value in expected._asdict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(saved[key], value)
            else:
                assert saved[key] == value
        exported = json.loads(file.with_suffix('.json').read_text())
        assert 'input_ids' not in exported and 'labels' not in exported and 'loss_mask' not in exported
        assert [row[0] for row in exported['tokens']] == saved['input_ids'][0].tolist()
        assert [row[1] for row in exported['tokens']] == saved['labels'][0].tolist()
        assert [row[2] for row in exported['tokens']] == (saved['labels'][0] != -100).tolist()
        for row in exported['tokens']:
            assert row[3] == tokenizer.decode([row[0]], skip_special_tokens=False)
            assert json.dumps(row, ensure_ascii=False) in file.with_suffix('.json').read_text()
        for key in ('position_ids', 'document_ids'):
            assert exported[key] == saved[key].tolist()
        for key in ('data_ids', 'data_tokens', 'padding_tokens'):
            assert exported[key] == saved[key]
        assert file.with_suffix('.txt').is_file() and file.with_suffix('.tsv').is_file()
    assert sum(len(torch.load(f, weights_only=True)['data_ids']) for f in (output / 'packed/sources/code').glob('*.pt')) == len(formatted)
    for stage in report['stages'].values():
        assert stage['packs'] == 2
    for file in (output / 'packed/stages').rglob('*.pt'):
        saved = torch.load(file, weights_only=True)
        assert all(name != formatted[0]['id'] for name in saved['data_ids'])
    with pytest.raises(ValueError, match='new empty directory'):
        inspect_pipeline(path, output, count=4)


def test_inspection_reports_empty_sources_without_reweighting(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, config_file: ConfigFile,
) -> None:
    directory = tmp_path / 'input'
    write_local(directory, [{'input': 'q', 'output': 'a', 'average_test_score': '0'}] * 10, '.jsonl')
    source = SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(directory),
                          converter='opencode_messages', filter='opencode_passed_tests')
    cfg = cfg_factory({'code': source}, tokens=128, bloom_deduplicate_across_sources=True)
    output = tmp_path / 'result'
    output.mkdir()
    report = inspect_pipeline(config_file(cfg), output, count=2, max_source_rows=3)
    assert report['sources']['code']['raw_rows'] == report['sources']['code']['processed'] == 0
    assert report['sources']['code']['source_rows_examined'] == 3
    assert report['sources']['code']['raw_shortfall'] == 2
    assert report['mixed_packs'] == 0 and report['stages']['finetune']['skipped']
    assert (output / 'raw/code.jsonl').read_text() == ''


def test_inspection_bounds_pretrain_raw_rows_at_row_group(
    hub: FakeHub, cfg_factory: CfgFactory, config_file: ConfigFile, tmp_path: Path,
) -> None:
    hub.add('data/train-0.parquet', [{'text': f'tok_{i} tok_1 tok_2'} for i in range(20)])
    cfg = cfg_factory({'web': SourceConfig(kind='pretrain', loader='hf_files', hf_id=REPO, revision=REV,
                       load_kwargs={'data_files': 'data/*.parquet'})}, tokens=128)
    output = tmp_path / 'out'
    output.mkdir()
    report = inspect_pipeline(config_file(cfg), output, count=3, packs_per_stage=0)
    assert report['sources']['web']['raw_rows'] == 3
    assert len(list(read_rows(Path(report['sources']['web']['raw_directory'])))) == 3


def test_global_dedup_changes_final_prepared_view(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, config_file: ConfigFile,
) -> None:
    sources = {}
    for name in ['a', 'b']:
        directory = tmp_path / name
        write_local(directory, [{'text': 'shared text tok_1'}, {'text': f'{name} unique text tok_2'}], '.jsonl')
        sources[name] = SourceConfig(kind='pretrain', loader='local', path=str(directory))
    cfg = cfg_factory(sources, tokens=128, processing=ProcessingConfig(min_chars=1), bloom_deduplicate_across_sources=True)
    out = tmp_path / 'out'
    out.mkdir()
    report = inspect_pipeline(config_file(cfg), out, count=2, packs_per_stage=0)
    assert report['sources']['a']['processed'] == 2
    assert report['sources']['b']['candidate_rows'] == 2 and report['sources']['b']['processed'] == 1
    assert len(list((out / 'dataset').rglob('MANIFEST.json'))) > 3


@pytest.mark.parametrize('kwargs', [{'count': 0}, {'pack_length': 1}, {'sequence_length': 99999}, {'max_source_rows': 0}])
def test_inspection_validates_limits_before_download(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        inspect_pipeline(Path('config/datasets/tiny.yaml'), tmp_path, **kwargs)


def test_inspection_preserves_both_assistant_turns_in_saved_labels(
    tmp_path: Path, cfg_factory: CfgFactory, write_local: Writer, config_file: ConfigFile,
) -> None:
    directory = tmp_path / 'input'
    rows = [{'reasoning': 'off', 'messages': [
        {'role': 'system', 'content': ''}, {'role': 'user', 'content': f'tok_{i}'},
        {'role': 'assistant', 'content': 'tok_7'}, {'role': 'user', 'content': 'tok_8'},
        {'role': 'assistant', 'content': 'tok_9'},
    ]} for i in range(2)]
    write_local(directory, rows, '.jsonl')
    cfg = cfg_factory({'chat': SourceConfig(kind='instruct', instruction_format='messages', loader='local',
                       path=str(directory), converter='nemotron_messages')}, tokens=128)
    out = tmp_path / 'out'
    out.mkdir()
    inspect_pipeline(config_file(cfg), out, count=2, pack_length=128)
    tokenizer = Tokenizer(out / 'dataset/tokenizers/synthetic')
    expected = tokenizer.encode('tok_7') + [tokenizer.eos_id] + tokenizer.encode('tok_9') + [tokenizer.eos_id]
    saved = torch.load(out / 'packed/sources/chat/pack-0000.pt', weights_only=True)
    assert saved['labels'][saved['labels'] != -100].tolist() == expected * 2
    assert saved['position_ids'][0, 0] == 0
    assert len(saved['data_ids']) == 2
    assert all(len(json.loads(line)['messages']) == 4 for line in (out / 'prepared/chat.jsonl').read_text().splitlines())
    text = (out / 'packed/sources/chat/pack-0000.txt').read_text()
    assert 'tok_7' in text and 'tok_9' in text and '[NO LOSS' in text and '[LOSS' in text
