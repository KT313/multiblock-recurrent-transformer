# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset preparation CLIs for the crow-300m-final training run.

Every module is a standalone command (``python -m data_preparation.<name> --help``). Run them from the repo root
in this order; all outputs default to ``dataset/`` (override with ``--dataset_dir``):

    download_pretraining        -> dataset/pretraining/raw/<source>/shard-*.parquet      (raw HF columns)
    filter_pretraining          -> dataset/pretraining/filtered/<source>/data-*.parquet  (text, source, original_length)
    process_pretraining         -> dataset/pretraining/processed/merged/<source>/data-*.parquet
                                   (text, source, estimated_tokens) + preprocessing_stats.json + verification_samples.txt
    prepare_fineweb_validation  -> dataset/fineweb-edu/validation/data-*.parquet         (text, ...)
    prepare_flan_mixture        -> dataset/flan_mixture/{train,validation}/data-*.parquet (instruction, input, output)
    prepare_tokenizer           -> dataset/tokenizer/                                   (HF tokenizer files)

The training config references ``processed/merged/<source>`` for the two pretraining stages,
``fineweb-edu/validation`` for pretraining validation, ``flan_mixture/{train,validation}`` for the finetuning stage
and ``tokenizer`` as ``tokenizer_path``.
"""
