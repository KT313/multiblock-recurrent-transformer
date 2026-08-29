# Data preparation

Standalone CLIs that rebuild the datasets consumed by `config/crow_300m_final.yaml`. Every command is run from the
repo root as `python data_preparation/prepare.py <command>` (implementation in `lib/`) and writes below `dataset/` (`--dataset_dir` overrides the root,
`--cache_dir` the HuggingFace cache). The logic (sources, sample budgets, filters, deduplication, PII masking,
decontamination, mixture shares, seeds) is the one used for the thesis run; only the cluster wrappers, hard-coded
paths and unused code paths were removed.

## Pipeline

| Step | Command | Writes |
|---|---|---|
| 1 | `python data_preparation/prepare.py download [--parallel N] [--datasets ...]` | `dataset/pretraining/raw/<source>/shard-*.parquet` (raw HF columns) |
| 2 | `python data_preparation/prepare.py filter [--min_chars 50] [--max_chars 20000]` | `dataset/pretraining/filtered/<source>/data-*.parquet` (`text`, `source`, `original_length`) |
| 3 | `python data_preparation/prepare.py process [--skip_fuzzy_dedup] [--skip_decontamination] [--dry_run]` | `dataset/pretraining/processed/merged/<source>/data-*.parquet` (`text`, `source`, `estimated_tokens`), `preprocessing_stats.json`, `verification_samples.txt` |
| 4 | `python data_preparation/prepare.py fineweb-validation` | `dataset/fineweb-edu/validation/data-*.parquet` |
| 5 | `python data_preparation/prepare.py flan-mixture --add_input_inversions --inversion_ratio 0.05 [--dry_run]` | `dataset/flan_mixture/{train,validation}/data-*.parquet` (`instruction`, `input`, `output`), `metadata.json` |
| 6 | `python data_preparation/prepare.py tokenizer` | `dataset/tokenizer/` |

Steps 4-6 are independent of 1-3. Step 3 needs `datasketch` for fuzzy deduplication (`--skip_fuzzy_dedup`
otherwise) and downloads the benchmark test sets for decontamination. Sources 1-3 that need authentication
(GitHub code) read `HF_TOKEN` from the environment.

### Sources and budgets

Pretraining (`download_pretraining.DATASETS`, sample targets sized for the 300M run): fineweb_edu (9.0M docs),
wikipedia (1.7M), books_gutenberg (600k), peso (600k), arxiv (400k), openwebmath (2.0M), tinygsm (1.6M),
algebraic_stack (450k), gsm8k (7.5k train rows repeated to 550k) and ten `github_code_clean_<language>` splits
(python 3.2M, javascript 2.2M, typescript 1.1M, java 1.1M, cpp 900k, go 800k, rust 550k, shell 450k, sql 350k,
html 350k). The per-source training weights of the two pretraining stages live in the training config, not here.

Finetuning mixture (`prepare_flan_mixture.SOURCES`, 400k examples): FLAN Collection 40%, MetaMathQA 15%, Orca-Math
10%, Evol-Instruct-Code 12.5%, Code Alpaca 2.5%, SlimOrca-Dedup 10%, ShareGPT (quality filtered) 5%, WizardLM Evol V2
5%; 95/5 train/validation split. The thesis run enabled input inversions on 5% of the examples (the command above).

### Processing steps (step 3)

Per source, in order: exact deduplication (MD5), fuzzy deduplication (MinHash over 5-grams + LSH, Jaccard 0.8, 256
permutations), quality filter (>= 3 sentences, <= 30% ALL-CAPS words, >= 25% alphanumeric characters, <= 30% repeated
2-grams, <= 20% repeated 3-grams), PII masking (`[EMAIL]`, `[IP]`, `[PHONE]`, `[KEY]`) and benchmark decontamination
(drop documents whose 13-grams overlap > 10% with the GSM8K, MATH, HumanEval, MBPP, ARC-Challenge, HellaSwag, MMLU or
WinoGrande test sets). Each step has a `--skip_*` flag.

## Resulting layout

```
dataset/
├── pretraining/
│   ├── raw/<source>/shard-*.parquet
│   ├── filtered/<source>/data-*.parquet          (+ preprocessing_stats.json)
│   └── processed/
│       ├── merged/<source>/data-*.parquet        <- train_data of the two pretraining stages
│       ├── preprocessing_stats.json
│       └── verification_samples.txt
├── fineweb-edu/validation/data-*.parquet         <- val_data of the pretraining stages
├── flan_mixture/
│   ├── train/data-*.parquet                      <- finetune stage train_data
│   ├── validation/data-*.parquet                 <- finetune stage val_data
│   └── metadata.json
└── tokenizer/                                    <- tokenizer_path
```

`raw/` and `filtered/` are intermediates and can be deleted once `processed/merged/` exists.
