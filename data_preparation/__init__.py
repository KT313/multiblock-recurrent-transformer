# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset preparation driven by a dataset config (``config/datasets/<name>.yaml``).

Entry point: ``python data_preparation/prepare.py tiny [--dataset_config F] [--dataset_dir dataset]`` (run from
the repo root; ``dataset_config.py`` (the config schema = the config reference) and ``layout.py`` (paths under
``dataset/``) live in this package, the rest in ``data_preparation/lib/``: ``sources`` loaders/converters, ``stages``
pipeline steps, ``storage`` manifests/parquet, ``build`` planner/runner). Outputs:

    dataset/sources/<source>/raw/      rows as downloaded (both kinds), shared by every dataset config
    dataset/processed/<source>/        cleaned rows (what training reads); the validation split is made at training time
    dataset/tokenizers/<name>/
"""
