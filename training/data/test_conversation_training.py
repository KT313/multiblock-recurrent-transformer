# (c) 2025-2026 Tobias Kerner. Apache-2.0.
import io
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from data_preparation.lib.conversation_format import encode_chat_prompt, fit_conversation
from training.data.collate import collate_samples, pad_and_shift
from training.data.formats import format_conversation
from training.data.packing import pack_samples, PackPool
from training.data.tokenizer import Tokenizer, IGNORE_INDEX
from model import build_model
from model.layers.attention import document_attention_mask
from model.test_config import TINY_ARCHITECTURE
import model.model as model_module


def test_conversation_loss_packing_and_resume(tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, 'initialize_state', torch.zeros_like)
    messages = [{'role': 'user', 'content': 'tok_1 tok_2'}, {'role': 'assistant', 'content': 'tok_3 tok_4'},
                {'role': 'user', 'content': 'tok_5'}, {'role': 'assistant', 'content': 'tok_6'}]
    row = {'messages': messages, 'data_signature': {'format_fn': 'format_conversation'}, 'data_id': 'chat'}
    inputs, labels = format_conversation(row, tokenizer, True, True)
    expected = tokenizer.encode_literal('tok_3 tok_4') + [tokenizer.eos_id] + tokenizer.encode_literal('tok_6') + [tokenizer.eos_id]
    assert labels[labels != IGNORE_INDEX].tolist() == expected
    encoded = fit_conversation(messages, tokenizer)
    prefix = encode_chat_prompt(messages[:3], tokenizer)
    assert inputs[:len(prefix)].tolist() == prefix
    shortened = collate_samples([row], tokenizer, encoded.exchange_ends[0] - 1)
    assert len(shortened) == 1 and shortened[0][0].tolist() == inputs[:encoded.exchange_ends[0]].tolist()
    assert collate_samples([row], tokenizer, encoded.exchange_ends[0] - 2) == []

    samples = collate_samples([row, row], tokenizer, 256)
    padded = pad_and_shift(samples, tokenizer, 256)
    packed = pack_samples(samples, 128, tokenizer)
    assert (packed.labels != IGNORE_INDEX).sum().item() == 2 * len(expected)
    assert packed.position_ids[0, len(inputs) - 1].item() == 0  # next conversation resets, not next turn
    assert packed.position_ids[0, encoded.exchange_ends[0]].item() == encoded.exchange_ends[0]

    torch.manual_seed(9)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False).train()
    kwargs = {'num_steps': (1, 2), 'return_loss_statistics': True}
    result = model(padded.input_ids, labels=padded.labels, **kwargs)
    logits = model(padded.input_ids, return_logits=True, num_steps=(1, 2))['logits']
    manual = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), padded.labels.reshape(-1), ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(result['loss'], manual)
    assert result['supervised_count'].item() == 2 * len(expected)
    result['loss'].backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    other = model(packed.input_ids, labels=packed.labels, position_ids=packed.position_ids,
                  attention_mask=document_attention_mask(packed.document_ids), **kwargs)
    torch.testing.assert_close(result['loss'], other['loss'], atol=1e-6, rtol=1e-5)
    other['loss'].backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(grads[name], parameter.grad, atol=1e-6, rtol=1e-4)

    pool = PackPool(128)
    for sample in samples:
        pool.add(sample)
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'pool': pool.state()}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    new_pool = PackPool(128)
    new_pool.restore(restored['pool'])
    model.load_state_dict(restored['model'])
    restored_pack = pack_samples(new_pool.take_pack(), 128, tokenizer)
    torch.testing.assert_close(restored_pack.labels, packed.labels)
    torch.testing.assert_close(restored_pack.input_ids, packed.input_ids)


@pytest.mark.slow
@pytest.mark.timeout(60)
def test_real_training_loop_resumes_mixed_chat_and_text(tmp_path: Path) -> None:
    import json
    from dataclasses import asdict
    import yaml
    from data_preparation import DatasetConfig, DatasetLayout
    from data_preparation.lib.dataset_config import SourceConfig, StageConfig, TokenizerConfig, ProcessingConfig, DedupConfig
    from data_preparation.lib.build.runner import prepare
    from training.testing.golden import write_tiny_yaml, single_thread_deterministic
    from training.backend.single_device import SingleDeviceBackend
    from training.settings import parse_settings
    from training.run import train
    from training.checkpoint import checkpoint_dir

    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    messages = [[{'role': 'user', 'content': f'tok_{i}'}, {'role': 'assistant', 'content': 'tok_2 tok_3'},
                 {'role': 'user', 'content': 'tok_4'}, {'role': 'assistant', 'content': f'tok_{i+1}'}] for i in range(100)]
    (inputs / 'rows.jsonl').write_text(''.join(json.dumps({'reasoning': 'off', 'messages': row}) + '\n' for row in messages))
    cfg = DatasetConfig(tokenizer=TokenizerConfig(name='synthetic', kind='synthetic'), training_target_sequence_length=128,
        dataset_max_sequence_length=128, sources={
            'text': SourceConfig(kind='pretrain', loader='synthetic'),
            'chat': SourceConfig(kind='instruct', instruction_format='messages', loader='local', path=str(inputs), converter='nemotron_messages'),
        }, stages=[StageConfig(name='mixed', tokens=2048, train={'text': .5, 'chat': .5}, val={'text': .5, 'chat': .5})],
        processing=ProcessingConfig(min_chars=1, dedup=DedupConfig(bloom_memory_mb=1)), bloom_dedup_memory_mb=1)
    dataset_config = tmp_path / 'dataset.yaml'
    dataset_config.write_text(yaml.safe_dump(asdict(cfg)))
    layout = DatasetLayout(tmp_path / 'data')
    prepare(dataset_config, layout.root, num_workers=1, pass_workers=1, assume_yes=True)
    overrides = dict(dataset_config=str(dataset_config), precision='32', stage_base_lrs=[3e-4], training_max_sequence_length=128,
                     warmup_steps=0, cooldown_steps=0, save_step_interval=1, eval_step_interval=100,
                     log_gradient_metrics_interval=0, partial_depth_eval=[])
    full_dir = tmp_path / 'full'
    resumed_dir = tmp_path / 'resumed'
    for directory in (full_dir, resumed_dir):
        directory.mkdir()
    with single_thread_deterministic():
        settings = parse_settings(['--config', str(write_tiny_yaml(full_dir, layout.root, full_dir / 'out', **overrides))])
        complete = train(settings, backend=SingleDeviceBackend(device='cpu', precision='32'), keep_history=True)
        checkpoint = checkpoint_dir(full_dir / 'out' / 'tiny') / 'step-00000001-tiny.pth'
        assert checkpoint.is_file()
        settings = parse_settings(['--config', str(write_tiny_yaml(resumed_dir, layout.root, resumed_dir / 'out',
                         resume=True, resume_checkpoint_path=str(checkpoint), **overrides))])
        resumed = train(settings, backend=SingleDeviceBackend(device='cpu', precision='32'), keep_history=True)
    assert complete.completed_steps == resumed.completed_steps == 2
    assert complete.history[2]['loss'] == resumed.history[2]['loss']
    full_state = torch.load(checkpoint_dir(full_dir / 'out' / 'tiny') / 'step-00000002-tiny.pth', weights_only=False)
    resumed_state = torch.load(checkpoint_dir(resumed_dir / 'out' / 'tiny') / 'step-00000002-tiny.pth', weights_only=False)
    for name, value in full_state['model'].items():
        torch.testing.assert_close(value, resumed_state['model'][name], atol=0, rtol=0)
    assert full_state['data_stream']['consumed_rows'] == resumed_state['data_stream']['consumed_rows']
