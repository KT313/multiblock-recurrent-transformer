# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Readable inspection exports alongside the real Parquet and tensor artifacts."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from data_preparation.lib.storage.manifest import Manifest
from training.data.packing import PackedBatch
from training.data.tokenizer import IGNORE_INDEX, Tokenizer


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def format_token_record(record: dict[str, Any], *, indent: int = 0) -> str:
    """Render metadata compactly and each aligned token row on its own line."""
    prefix = ' ' * indent
    fields = [f'{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}' for key, value in record.items() if key != 'tokens']
    rows = ',\n'.join(prefix + '    ' + json.dumps(row, ensure_ascii=False) for row in record['tokens'])
    fields.append('"tokens": [\n' + rows + '\n' + prefix + '  ]' if rows else '"tokens": []')
    return prefix + '{\n' + prefix + '  ' + (',\n' + prefix + '  ').join(fields) + '\n' + prefix + '}'


def write_formatted_samples(path: Path, records: list[dict[str, Any]]) -> None:
    """Keep each aligned token row on one line while retaining sample metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [format_token_record(record, indent=2) for record in records]
    path.write_text('[\n' + ',\n'.join(samples) + '\n]\n', encoding='utf-8')


def read_rows(directory: Path) -> Iterator[dict[str, Any]]:
    manifest = Manifest.load(directory)
    if manifest is None:
        raise ValueError(f'No manifest in {directory}')
    for shard in manifest.shards:
        for batch in pq.ParquetFile(directory / shard.name).iter_batches(batch_size=128):
            yield from batch.to_pylist()


def export_rows(directory: Path, destination: Path) -> list[dict[str, Any]]:
    rows = list(read_rows(directory))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({'row_index': index, **row}, ensure_ascii=False) + '\n')
    return rows


def write_pack(directory: Path, index: int, pack: PackedBatch, tokenizer: Tokenizer) -> None:
    stem = directory / f'pack-{index:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    values = pack._asdict()
    torch.save(values, stem.with_suffix('.pt'))
    ids, labels, positions, documents = (tensor[0].tolist() for tensor in (pack.input_ids, pack.labels, pack.position_ids, pack.document_ids))
    record = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in values.items()
              if key not in ('input_ids', 'labels')}
    record['attention_rule'] = 'causal AND equal document_ids; padding has its own document ID'
    record['tokens'] = [[token, label, label != IGNORE_INDEX, tokenizer.decode([token], skip_special_tokens=False)]
                        for token, label in zip(ids, labels, strict=True)]
    stem.with_suffix('.json').write_text(format_token_record(record) + '\n', encoding='utf-8')

    # Each row aligns one input with its NEXT-token target, not with its own label.
    with stem.with_suffix('.tsv').open('w', encoding='utf-8') as stream:
        stream.write('slot\tdocument\tposition\tinput_id\tinput_token_json\tlabel_id\ttarget_token_json\tloss\n')
        for slot, (token, label, position, document) in enumerate(zip(ids, labels, positions, documents, strict=True)):
            decoded = json.dumps(tokenizer.decode([token], skip_special_tokens=False), ensure_ascii=False)
            target = '' if label == IGNORE_INDEX else json.dumps(tokenizer.decode([label], skip_special_tokens=False), ensure_ascii=False)
            stream.write(f'{slot}\t{document}\t{position}\t{token}\t{decoded}\t{label}\t{target}\t{int(label != IGNORE_INDEX)}\n')

    lines = [f'Pack {index}: {len(ids)} positions, {pack.padding_tokens} padding positions.',
             'Labels are shifted next-token targets; exact IDs/masks are in JSON, PT and TSV.',
             'Decoded individual spans are a display aid; tokenizer spacing may differ across span boundaries.', '']
    offset = 0
    for document, (source, length) in enumerate(zip(pack.data_ids, pack.data_tokens, strict=True)):
        lines += [f'=== Document {document}: {source}, slots [{offset}, {offset + length}) ===',
                  'INPUT:', tokenizer.decode(ids[offset:offset + length], skip_special_tokens=False), 'TARGET SPANS:']
        start = offset
        while start < offset + length:
            supervised = labels[start] != IGNORE_INDEX
            end = start + 1
            while end < offset + length and (labels[end] != IGNORE_INDEX) == supervised:
                end += 1
            text = tokenizer.decode(labels[start:end], skip_special_tokens=False) if supervised else '(masked)'
            lines.append(f'[{"LOSS" if supervised else "NO LOSS"} slots {start}:{end}] {text}')
            start = end
        lines.append('')
        offset += length
    lines.append(f'Padding: slots [{offset}, {len(ids)}), all labels ignored.')
    stem.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
