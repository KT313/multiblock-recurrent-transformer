# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Use the production collation, packing pool, and stage sampler on a small finite fixture."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.inspection.artifacts import write_formatted_samples, write_json, write_pack
from training.data.collate import Sample, WorkerBatch, collate_samples
from training.data.dataset_resolver import CHAT_DATA_SIGNATURE, INSTRUCT_DATA_SIGNATURE, validation_rows_of
from training.data.datasets import DEFAULT_DATA_SIGNATURE
from training.data.entries import ResolvedStage
from training.data.loader_state import RunDataloaders
from training.data.packing import PackPool, pack_samples
from training.data.tokenizer import IGNORE_INDEX, Tokenizer
from training.settings import Settings
from training.stage_manager import StageManager
from training.steps.batches import BatchStream
from training.steps.state import TrainingProgress


log = logging.getLogger(__name__)

def format_rows(
    config: DatasetConfig, name: str, rows: list[dict[str, Any]], tokenizer: Tokenizer, sequence_length: int, output: Path,
) -> tuple[list[Sample], list[Sample], dict[str, int]]:
    source = config.sources[name]
    signature = (CHAT_DATA_SIGNATURE if source.instruction_format == 'messages' else INSTRUCT_DATA_SIGNATURE) if source.kind == 'instruct' else DEFAULT_DATA_SIGNATURE
    validation_count = validation_rows_of(config, name, len(rows))
    all_samples, train_samples = [], []
    records = []
    for index, row in enumerate(rows):
        identifier = f'{name}:{index:06d}'
        samples = collate_samples([{**row, 'data_id': identifier, 'data_signature': signature}], tokenizer, sequence_length)
        split = 'train' if config.used_in_train(name) and index >= validation_count else 'validation'
        record: dict[str, Any] = {'id': identifier, 'prepared_row_index': index, 'split': split, 'usable': bool(samples), 'tokens': []}
        if samples:
            inputs, labels, _ = samples[0]
            record['tokens'] = [[token, label, label != IGNORE_INDEX, tokenizer.decode([token], skip_special_tokens=False)]
                                for token, label in zip(inputs.tolist(), labels.tolist(), strict=True)]
            all_samples.extend(samples)
            if split == 'train':
                train_samples.extend(samples)
        else:
            record['drop_reason'] = 'no supervised target after production collation at the requested length'
        records.append(record)
    write_formatted_samples(output / 'formatted' / f'{name}.json', records)
    return all_samples, train_samples, {'processed': len(rows), 'formatted': len(all_samples), 'training_samples': len(train_samples),
                                       'validation_rows': validation_count if config.used_in_train(name) else len(rows),
                                       'collation_dropped': len(rows) - len(all_samples)}


def pack_finite(samples: Iterator[Sample], tokenizer: Tokenizer, pack_length: int, output: Path) -> int:
    pool = PackPool(pack_length)
    exhausted = False
    index = 0
    while True:
        while not exhausted and pool.needs_refill():
            sample = next(samples, None)
            if sample is None:
                exhausted = True
            elif not pool.add(sample):
                raise ValueError('Inspection sample does not fit the pack; increase --pack-length')
        if not len(pool):
            return index
        selected = pool.take_pack()
        if not selected:
            raise RuntimeError('Packing pool failed to produce a nonempty pack')
        write_pack(output, index, pack_samples(selected, pack_length, tokenizer), tokenizer)
        index += 1


def interleave_sources(samples: dict[str, list[Sample]]) -> Iterator[Sample]:
    for group in zip_longest(*samples.values()):
        for sample in group:
            if sample is not None:
                yield sample


def pack_stages(
    config: DatasetConfig, samples: dict[str, list[Sample]], tokenizer: Tokenizer,
    sequence_length: int, pack_length: int, count: int, output: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    settings = Settings(dataset_config='inspection', model_architecture_config='inspection-no-model',
                        tokens_per_micro_batch=pack_length, micro_batches_per_step=1,
                        training_max_sequence_length=sequence_length, stage_base_lrs=[1e-4], warmup_steps=0, cooldown_steps=0)
    for index, stage in enumerate(config.stages):
        weights = {name: weight for name, weight in stage.train.items() if weight > 0}
        missing = [name for name in weights if not samples[name]]
        directory = output / 'packed' / 'stages' / f'{index:02d}'
        if missing:
            log.warning('Skipping stage preview %s: no usable training samples for %s', stage.name, missing)
            result = {'skipped': True, 'reason': 'no usable training samples after split/collation', 'sources': missing}
            results[stage.name] = result
            write_json(directory / 'summary.json', result)
            continue
        # Real BatchStream cycles the finite per-source examples; each stage starts at steady weights.
        loaders = RunDataloaders({name: [WorkerBatch(samples[name], len(samples[name]))] for name in weights}, [], tokenizer, {})
        manager = StageManager([ResolvedStage(stage.name, max(stage.tokens, count * pack_length), 1e-4, 0, weights, [])], pack_length)
        progress = TrainingProgress()
        stream = BatchStream(settings, loaders, manager, progress)
        try:
            for number in range(count):
                write_pack(directory, number, next(stream), tokenizer)
                progress.step += 1
            result = {'packs': count, 'weights': weights, 'consumed_rows_including_lookahead': stream.consumed_rows,
                      'mode': 'independent stage preview; real BatchStream, steady weights, finite training samples cycle; no transitions'}
            results[stage.name] = result
            write_json(directory / 'summary.json', result)
        finally:
            loaders.close()
    return results
