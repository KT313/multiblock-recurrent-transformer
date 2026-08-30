# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset preparation driven by a dataset config (``config/datasets/<name>.yaml``).

Entry point: ``python data_preparation/prepare.py tiny [--dataset_config F] [--dataset_dir dataset]`` (run from
the repo root; ``dataset_config.py`` (the config schema = the config reference) and ``layout.py`` (paths under
``dataset/``) live in this package, the rest in ``data_preparation/lib/``: ``sources`` loaders/converters, ``stages``
pipeline functions, ``storage`` manifests/parquet, ``build`` planner/runner). Outputs:

    dataset/sources/<source>/{raw,processed}/            shared source cache (pretrain sources)
    dataset/sources/<source>/validation/                     held-out validation rows
    dataset/instruct_instruct_mixtures/<config>/<mixture>/{train,validation}/
    dataset/tokenizers/<name>/
"""
