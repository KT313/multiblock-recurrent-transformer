# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Retain raw, prepared, and packed samples: python -m data_preparation.inspection --dataset_config FILE -n 10."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_config', '--dataset-config', type=Path, required=True)
    parser.add_argument('-n', '--n', '--samples', type=int, default=10, help='target retained raw rows per source')
    parser.add_argument('--output-dir', type=Path, help='new directory; otherwise create and retain a temporary directory')
    parser.add_argument('--sequence-length', type=int, help='document cap; default: dataset training_target_sequence_length')
    parser.add_argument('--pack-length', type=int, help='tokens per pack; default: sequence length')
    parser.add_argument('--max-source-rows', type=int, help='instruction rows examined per source; default max(1000, 100*n), respecting config limits')
    parser.add_argument('--packs-per-stage', type=int, default=2, help='real BatchStream previews per stage; 0 disables')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        output = Path(tempfile.mkdtemp(prefix='mbrt-pipeline-inspection-'))
    else:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    print(f'Inspection artifacts: {output}', flush=True)

    # Configure cache/thread defaults before importing the downloader or training libraries.
    os.environ['HF_HOME'] = str(output / 'cache' / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(output / 'cache' / 'huggingface' / 'hub')
    os.environ['HF_DATASETS_CACHE'] = str(output / 'cache' / 'huggingface' / 'datasets')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('RAYON_NUM_THREADS', '2')
    from data_preparation.lib.log import configure_logging
    from data_preparation.inspection.pipeline import inspect_pipeline
    configure_logging()
    result = inspect_pipeline(args.dataset_config, output, count=args.n, sequence_length=args.sequence_length,
                              pack_length=args.pack_length, max_source_rows=args.max_source_rows,
                              packs_per_stage=args.packs_per_stage, hf_token=os.environ.get('HF_TOKEN'))
    print(f"Done: {output}\nStart with README.txt, summary.json, and packed/mixed/*.txt ({result['mixed_packs']} packs).")


if __name__ == '__main__':
    main()
