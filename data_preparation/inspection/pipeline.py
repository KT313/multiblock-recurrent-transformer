# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Sample with production download/build stages, then retain inspection artifacts."""
from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from data_preparation.lib.dataset_config import DatasetConfig, load_dataset_config
from data_preparation.lib.layout import DatasetLayout
from data_preparation.lib.build.lock import dataset_lock
from data_preparation.lib.stages.download import download, prepare_tokenizer
from data_preparation.lib.stages.build import build_source
from data_preparation.lib.stages.global_build import build_global_source
from data_preparation.lib.stages.global_dedup import ordered_sources
from data_preparation.lib.stages.benchmark_seeds import load_benchmark_seeds
from data_preparation.inspection.artifacts import export_rows, write_json
from data_preparation.inspection.packing import format_rows, interleave_sources, pack_finite, pack_stages
from training.data.collate import Sample
from training.data.tokenizer import Tokenizer

log = logging.getLogger(__name__)


def limit_sources(config: DatasetConfig, count: int, max_source_rows: int) -> DatasetConfig:
    sources = {}
    for name, source in config.sources.items():
        limit = count if source.kind == 'pretrain' else max_source_rows
        if source.check_limit is not None:
            limit = min(limit, source.check_limit)
        sources[name] = replace(source, check_limit=limit)
    return replace(config, sources=sources, download_prefetch_mb=0)


def prepare_samples(config: DatasetConfig, layout: DatasetLayout, count: int, output: Path, hf_token: str | None, summary: dict[str, Any]) -> DatasetLayout:
    prepare_tokenizer(config, layout, hf_token=hf_token)
    for index, name in enumerate(config.sources, 1):
        log.info('[%d/%d] Download up to %d retained rows: %s', index, len(config.sources), count, name)
        raw = download(config, name, layout, rows_needed=count, shard_size=min(count, 1000), hf_token=hf_token,
                       config_name='inspection-effective.yaml')
        summary[name] = {'requested_raw': count, 'raw_rows': raw.rows(), 'source_rows_examined': raw.rows_fetched,
                         'download_malformed': raw.skipped_malformed, 'download_too_long': raw.dropped_too_long,
                         'download_filtered_or_excluded': raw.rows_fetched - raw.rows() - raw.skipped_malformed - raw.dropped_too_long,
                         'raw_shortfall': max(0, count - raw.rows()), 'raw_directory': str(layout.raw_dir(name))}
        if raw.rows() > count:
            raise RuntimeError(f'{name}: downloader exceeded the inspection row bound ({raw.rows()} > {count})')
        if raw.rows() < count:
            log.warning('%s: retained %d/%d raw samples before exhaustion or the source-row limit', name, raw.rows(), count)
        export_rows(layout.raw_dir(name), output / 'raw' / f'{name}.jsonl')

        log.info('Prepare all retained raw rows: %s', name)
        processed = build_source(config, name, layout, pass_workers=1, shard_size=min(count, 1000))
        summary[name]['candidate_rows'] = processed.rows()
        summary[name]['preparation_stats'] = processed.stats

    # Run the same ordered global admission, without budget top-ups beyond the sampled raw rows.
    final = layout.for_config(config)
    if config.bloom_deduplicate_across_sources:
        with load_benchmark_seeds(config.bloom_deduplicate_across_sources_add_benchmarks,
                                  memory_mb=config.bloom_dedup_memory_mb, hf_token=hf_token) as seeds:
            frontier = seeds.frontier(ordered_sources(config), config.bloom_dedup_memory_mb)
            for name in ordered_sources(config):
                log.info('Apply global deduplication to sampled rows: %s', name)
                frontier, _ = build_global_source(config, name, final, frontier, rows_target=0, exhausted=True, preseed_keys=seeds.keys())
    return final


def inspect_pipeline(
    config_path: Path, output: Path, *, count: int = 10, sequence_length: int | None = None,
    pack_length: int | None = None, max_source_rows: int | None = None, packs_per_stage: int = 2, hf_token: str | None = None,
) -> dict[str, Any]:
    original = load_dataset_config(config_path)
    sequence_length = original.training_target_sequence_length if sequence_length is None else sequence_length
    pack_length = sequence_length if pack_length is None else pack_length
    max_source_rows = max(1000, 100 * count) if max_source_rows is None else max_source_rows
    if count < 1 or max_source_rows < 1 or packs_per_stage < 0:
        raise ValueError('sample count and source-row limit must be positive; packs-per-stage must be nonnegative')
    if not 0 < sequence_length <= original.dataset_max_sequence_length or pack_length < sequence_length:
        raise ValueError('require 0 < sequence-length <= dataset_max_sequence_length and pack-length >= sequence-length')
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError('inspection output must be a new empty directory; existing data is never overwritten')
    config = replace(limit_sources(original, count, max_source_rows), training_target_sequence_length=sequence_length)
    (output / 'original.yaml').write_text(config_path.read_text(), encoding='utf-8')
    (output / 'inspection-effective.yaml').write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding='utf-8')
    (output / 'README.txt').write_text(INSTRUCTIONS, encoding='utf-8')
    manifest: dict[str, Any] = {'status': 'running', 'dataset_config': str(config_path.resolve()), 'samples_per_source': count,
        'sequence_length': sequence_length, 'pack_length': pack_length, 'packs_per_stage': packs_per_stage,
        'max_source_rows': max_source_rows, 'effective_config_hash': config.config_hash(), 'sources': {},
        'overrides': {'download_prefetch_mb': 0, 'check_limit': {name: s.check_limit for name, s in config.sources.items()}},
        'packing_scope': 'finite source/mixed views include validation rows for inspection; stage previews use training rows only'}
    write_json(output / 'summary.json', manifest)
    layout = DatasetLayout(output / 'dataset')
    logger = logging.getLogger('data_preparation')
    previous_level = logger.level
    handler = logging.FileHandler(output / 'pipeline.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with dataset_lock(layout.root, 'pipeline inspection'):
            summary = manifest['sources']
            final = prepare_samples(config, layout, count, output, hf_token, summary)
            tokenizer = Tokenizer(layout.tokenizer_dir(config.tokenizer.name))
            all_samples: dict[str, list[Sample]] = {}
            train_samples: dict[str, list[Sample]] = {}
            for name in config.sources:
                rows = export_rows(final.processed_dir(name), output / 'prepared' / f'{name}.jsonl')
                all_samples[name], train_samples[name], counts = format_rows(config, name, rows, tokenizer, sequence_length, output)
                summary[name].update(counts)
                summary[name]['prepared_directory'] = str(final.processed_dir(name))
                summary[name]['packs'] = pack_finite(iter(all_samples[name]), tokenizer, pack_length, output / 'packed' / 'sources' / name)
            manifest['mixed_packs'] = pack_finite(interleave_sources(all_samples), tokenizer, pack_length, output / 'packed' / 'mixed')
            manifest['stages'] = pack_stages(config, train_samples, tokenizer, sequence_length, pack_length, packs_per_stage, output) if packs_per_stage else {}
        manifest['status'] = 'complete'
        return manifest
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
        write_json(output / 'summary.json', manifest)


INSTRUCTIONS = '''Dataset pipeline inspection (retained temporary artifacts; remove manually when finished).

original.yaml: input configuration. inspection-effective.yaml: exact bounded download configuration.
summary.json: stage status, counts, drops, paths, and diagnostic overrides. pipeline.log: preparation diagnostics.
dataset/: actual tokenizer, raw Parquet shards/manifests, source-local processed candidates, and global scoped output.
raw/*.jsonl: readable copies of standardized downloader output, AFTER download-time conversion/filtering/token limits.
prepared/*.jsonl: readable final prepared rows, AFTER source-local and optional global deduplication.
formatted/*.json: each sample's tokens are [input_id, label, loss_mask, decoded_token] rows before shifting;
source:index links to the prepared JSONL row. Each token row occupies one line; sample ID and split are retained.
packed/sources/: each source's usable prepared samples, packed once through the actual training PackPool/pack_samples.
packed/mixed/: all usable prepared samples, interleaved in config source order, packed once.
packed/stages/: real BatchStream previews at each stage's steady weights; finite training samples cycle.

Each pack has .pt tensors/metadata, .json token rows, .tsv per-position input/next-target/mask, and .txt decoded spans.
Packed JSON tokens are [input_id, label, loss_mask, decoded_token], one row per line AFTER the next-token shift.
The decoded token belongs to input_id; position_ids, document_ids and source/padding metadata are retained.
Labels are NEXT-token targets. -100 means no loss. Document IDs and positions encode the actual causal document mask;
no quadratic attention matrix is saved. Special tokens remain visible. Individual decoded token spacing is a display aid.

Finite views include validation rows to inspect every sample, and drain the final partial pool. They are not the training
mixture. Stage previews respect the row validation split and use the real deficit sampler, lookahead, formatter and packer,
but start independently at fixed stage weights: they do not reproduce transitions, production epoch offsets, or a whole run.
A missing usable training source causes that stage preview to be explicitly skipped, never silently reweighted.

n targets RETAINED RAW rows, not unfiltered upstream rows. Download filters may examine additional input rows; source limits
and exhaustion are reported. Preparation may leave fewer than n. There are no preparation top-ups. Row limits do not bound
network bytes: Parquet row groups and some loaders may fetch more data. Original quality/dedup/benchmark policies are retained.
Read-ahead is disabled; sources run sequentially and cleaning uses one worker. All local dataset writes stay under dataset/.
The actual loader may use its normal Hub cache; CLI invocations put that cache in this output folder.

This is a diagnostic sample, not a complete training dataset. Production stage token budgets are not materialized and no
training-ready dataset snapshot is published. No model is loaded or trained, and no GPU is used.
'''
