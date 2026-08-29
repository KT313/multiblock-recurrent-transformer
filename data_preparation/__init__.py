# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Dataset preparation for the crow-300m-final training run.

Single entry point: ``python data_preparation/prepare.py <command> [options]`` (run from the repo root; the
implementation lives in ``data_preparation/lib/``). All outputs default to ``dataset/`` (``--dataset_dir``):

    download            -> dataset/pretraining/raw/<source>/data-*.parquet      (raw HF columns)
    filter              -> dataset/pretraining/filtered/<source>/data-*.parquet  (text, source, original_length)
    process             -> dataset/pretraining/processed/merged/<source>/data-*.parquet
                           (text, source, estimated_tokens) + preprocessing_stats.json + verification_samples.txt
    fineweb-validation  -> dataset/fineweb-edu/validation/data-*.parquet         (text, ...)
    flan-mixture        -> dataset/flan_mixture/{train,validation}/data-*.parquet (instruction, input, output)
    tokenizer           -> dataset/tokenizer/                                   (HF tokenizer files)
    tiny                -> dataset/tiny/                                        (synthetic smoke data + tokenizer)

The training config references ``processed/merged/<source>`` for the two pretraining stages,
``fineweb-edu/validation`` for pretraining validation, ``flan_mixture/{train,validation}`` for the finetuning stage
and ``tokenizer`` as ``tokenizer_path``.
"""
