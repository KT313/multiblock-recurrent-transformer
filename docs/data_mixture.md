# Dataset `crow-300m-final`

Generated from `config/datasets/crow_300m_final.yaml` with

```bash
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/crow_300m_final.yaml > docs/data_mixture.md
```

Do not edit by hand: change the dataset config and regenerate. Token budgets are the stage budgets of the
config times the stage weights; sequences are those tokens divided by `block_size` (what the training loader
draws and what the planner sizes downloads with); rows are an estimate from `describe_tokens_per_row`, which
nothing but this document uses.

## Notes

Dataset definition of the thesis run (crow-300m-final). Referenced from `config/crow_300m_final.yaml` via
`dataset_config`. Materialise with `python data_preparation/prepare.py prepare --dataset_config <this file>`;
training does it automatically when something is missing (`auto_prepare`). `nampdn-ai/mini-peS2o` is gated:
export `HF_TOKEN` before building. This comment block is rendered into `docs/data_mixture.md` by
`prepare.py describe`.

Relation to the thesis run: the thesis data was prepared with exact deduplication, fuzzy (MinHash) deduplication
at Jaccard threshold 0.95 and real tokenizer counts truncated to 2048 tokens, with the quality filter, benchmark
decontamination and PII masking all skipped. This config mirrors that as closely as the restructured pipeline
allows: exact dedup on, tokenizer counts capped at 2048, quality filter and decontamination off. Fuzzy dedup is
off here (it is available as `dedup: {mode: minhash, threshold: 0.95, num_perm: 256}`), and PII masking no longer
exists. The per-stage token budgets are the thesis budgets (3.3B + 1.5B + 0.15B tokens); the stage weights are
the thesis run's; per-source download sizes are derived from the sequence budget (stage tokens × weight ÷
block_size, × 1.2) instead of the fixed row counts of the thesis scripts. Several HuggingFace ids had to move (wikipedia -> wikimedia/wikipedia 20231101.en,
gsm8k -> openai/gsm8k, arxiv -> common-pile/arxiv_papers_filtered), and every source is pinned to a revision.

## Tokenizer, sequence length and token counting

- tokenizer: `llama-32k` (hf, `hf-internal-testing/llama-tokenizer` @ `d02ad6cb9dd2c2296a6332199fa2fdca5938fef0`)
- `max_seq_length`: 2048 (pretrain rows are truncated to this many tokens when downloaded, longer instruct rows are dropped)
- `block_size`: 2048 (training sequence length; the run config must use the same value)
- `token_count`: `tokenizer` (real tokenizer counts)
- `validation_fraction`: 5% of a source used for training and validation is held out
- training tokens over all stages: 4.95B (2,416,993 sequences)

## Processing defaults

- length filter: 50 <= chars (pretrain only; rows are cut at `max_seq_length` tokens when downloaded)
- dedup: `exact` (normalize: on, Bloom filter 1024 MB per source)
- quality filter: off
- decontamination: off

## Stages

### Stage 1: `pretrain_phase1` (3.30B tokens, transition 10%)

| Train source | Weight | Tokens | Sequences | Tokens/row (est.) | Rows (est.) |
|---|---:|---:|---:|---:|---:|
| `fineweb_edu` | 65.00% | 2.15B | 1,047,364 | 2000 | 1,072,500 |
| `wikipedia` | 9.00% | 297.0M | 145,020 | 1500 | 198,000 |
| `books_gutenberg` | 6.00% | 198.0M | 96,680 | 3000 | 66,000 |
| `github_code_clean_python` | 3.60% | 118.8M | 58,008 | 500 | 237,600 |
| `github_code_clean_javascript` | 2.40% | 79.2M | 38,672 | 500 | 158,400 |
| `github_code_clean_typescript` | 1.20% | 39.6M | 19,336 | 500 | 79,200 |
| `github_code_clean_java` | 1.20% | 39.6M | 19,336 | 500 | 79,200 |
| `github_code_clean_cpp` | 0.96% | 31.7M | 15,469 | 500 | 63,360 |
| `github_code_clean_go` | 0.84% | 27.7M | 13,536 | 500 | 55,440 |
| `github_code_clean_rust` | 0.60% | 19.8M | 9,668 | 500 | 39,600 |
| `github_code_clean_shell` | 0.48% | 15.8M | 7,735 | 500 | 31,680 |
| `github_code_clean_sql` | 0.36% | 11.9M | 5,801 | 500 | 23,760 |
| `github_code_clean_html` | 0.36% | 11.9M | 5,801 | 500 | 23,760 |
| `peso` | 3.00% | 99.0M | 48,340 | 1500 | 66,000 |
| `arxiv` | 2.00% | 66.0M | 32,227 | 1500 | 44,000 |
| `openwebmath` | 3.00% | 99.0M | 48,340 | 750 | 132,000 |

Validation: `fineweb_edu` at 100%

### Stage 2: `pretrain_phase2` (1.50B tokens, transition 10%)

| Train source | Weight | Tokens | Sequences | Tokens/row (est.) | Rows (est.) |
|---|---:|---:|---:|---:|---:|
| `fineweb_edu` | 35.00% | 525.0M | 256,348 | 2000 | 262,500 |
| `github_code_clean_python` | 8.40% | 126.0M | 61,524 | 500 | 252,001 |
| `github_code_clean_javascript` | 5.60% | 84.0M | 41,016 | 500 | 168,000 |
| `github_code_clean_typescript` | 2.80% | 42.0M | 20,508 | 500 | 84,000 |
| `github_code_clean_java` | 2.80% | 42.0M | 20,508 | 500 | 84,000 |
| `github_code_clean_cpp` | 2.24% | 33.6M | 16,407 | 500 | 67,200 |
| `github_code_clean_go` | 1.96% | 29.4M | 14,356 | 500 | 58,800 |
| `github_code_clean_rust` | 1.40% | 21.0M | 10,254 | 500 | 42,000 |
| `github_code_clean_shell` | 1.12% | 16.8M | 8,204 | 500 | 33,600 |
| `github_code_clean_sql` | 0.84% | 12.6M | 6,153 | 500 | 25,200 |
| `github_code_clean_html` | 0.84% | 12.6M | 6,153 | 500 | 25,200 |
| `openwebmath` | 8.80% | 132.0M | 64,454 | 750 | 176,000 |
| `tinygsm` | 6.60% | 99.0M | 48,340 | 300 | 330,000 |
| `algebraic_stack` | 4.40% | 66.0M | 32,227 | 750 | 88,000 |
| `gsm8k` | 2.20% | 33.0M | 16,114 | 300 | 110,000 |
| `peso` | 9.00% | 135.0M | 65,918 | 1500 | 90,000 |
| `arxiv` | 6.00% | 90.0M | 43,946 | 1500 | 60,000 |

Validation: `fineweb_edu` at 100%

### Stage 3: `finetune` (150.0M tokens, transition 0%)

| Train source | Weight | Tokens | Sequences | Tokens/row (est.) | Rows (est.) |
|---|---:|---:|---:|---:|---:|
| `flan` | 40.00% | 60.0M | 29,297 | 300 | 200,000 |
| `metamath` | 15.00% | 22.5M | 10,987 | 300 | 75,000 |
| `orca_math` | 10.00% | 15.0M | 7,325 | 300 | 50,000 |
| `evol_code` | 12.50% | 18.8M | 9,156 | 400 | 46,875 |
| `code_alpaca` | 2.50% | 3.8M | 1,832 | 200 | 18,750 |
| `slimorca` | 10.00% | 15.0M | 7,325 | 300 | 50,000 |
| `sharegpt` | 5.00% | 7.5M | 3,663 | 400 | 18,750 |
| `wizardlm` | 5.00% | 7.5M | 3,663 | 400 | 18,750 |

Validation: `flan` at 40%, `metamath` at 15%, `orca_math` at 10%, `evol_code` at 12%, `code_alpaca` at 2%, `slimorca` at 10%, `sharegpt` at 5%, `wizardlm` at 5%

## Validation split

Decided at training time, never on disk: a source used only for training is all training rows, one used only
for validation is all validation rows (`rows` says how many are downloaded), one used for both gives its first
`validation_fraction` of processed rows to validation (deduplicated as one set, so the two never share a document).

| Source | Used in | Held out |
|---|---|---|
| `fineweb_edu` | train + val | 5% held out (the first rows of `processed/fineweb_edu`) |
| `wikipedia` | train only | none |
| `books_gutenberg` | train only | none |
| `peso` | train only | none |
| `arxiv` | train only | none |
| `openwebmath` | train only | none |
| `tinygsm` | train only | none |
| `algebraic_stack` | train only | none |
| `gsm8k` | train only | none |
| `github_code_clean_python` | train only | none |
| `github_code_clean_javascript` | train only | none |
| `github_code_clean_typescript` | train only | none |
| `github_code_clean_java` | train only | none |
| `github_code_clean_cpp` | train only | none |
| `github_code_clean_go` | train only | none |
| `github_code_clean_rust` | train only | none |
| `github_code_clean_shell` | train only | none |
| `github_code_clean_sql` | train only | none |
| `github_code_clean_html` | train only | none |
| `flan` | train + val | 5% held out (the first rows of `processed/flan`) |
| `metamath` | train + val | 5% held out (the first rows of `processed/metamath`) |
| `orca_math` | train + val | 5% held out (the first rows of `processed/orca_math`) |
| `evol_code` | train + val | 5% held out (the first rows of `processed/evol_code`) |
| `code_alpaca` | train + val | 5% held out (the first rows of `processed/code_alpaca`) |
| `slimorca` | train + val | 5% held out (the first rows of `processed/slimorca`) |
| `sharegpt` | train + val | 5% held out (the first rows of `processed/sharegpt`) |
| `wizardlm` | train + val | 5% held out (the first rows of `processed/wizardlm`) |

## Sources

| Source | Kind | Loader | Origin | Revision | Details |
|---|---|---|---|---|---|
| `fineweb_edu` | pretrain | `hf_files` | `HuggingFaceFW/fineweb-edu` data_files=data/CC-MAIN-2013-20/*.parquet | `87f09149ef47` | budget 1,047,364 sequences (2.15B tokens) |
| `wikipedia` | pretrain | `hf_files` | `wikimedia/wikipedia` data_files=20231101.en/*.parquet | `b04c8d1ceb2f` | budget 145,020 sequences (297.0M tokens) |
| `books_gutenberg` | pretrain | `hf_files` | `sedthh/gutenberg_english` data_files=data/*.parquet | `28973b04f28f` | text_field `TEXT`, budget 96,680 sequences (198.0M tokens) |
| `peso` | pretrain | `hf_files` | `nampdn-ai/mini-peS2o` data_files=train-*.parquet | `18a60ef8d79f` | budget 65,918 sequences (135.0M tokens) |
| `arxiv` | pretrain | `hf_files` | `common-pile/arxiv_papers_filtered` data_files=arxiv-papers-*.json.gz | `033cf7f53f9b` | budget 43,946 sequences (90.0M tokens) |
| `openwebmath` | pretrain | `hf_files` | `open-web-math/open-web-math` data_files=data/*.parquet | `fde8ef8de230` | budget 64,454 sequences (132.0M tokens) |
| `tinygsm` | pretrain | `hf_files` | `ostapeno/tinygsm-mind` data_files=data/*.parquet | `f5ecf416b715` | budget 48,340 sequences (99.0M tokens) |
| `algebraic_stack` | pretrain | `hf_files` | `EleutherAI/proof-pile-2` data_files=algebraic-stack/train/*.jsonl.zst | `901a9273a770` | budget 32,227 sequences (66.0M tokens) |
| `gsm8k` | pretrain | `hf_split` | `openai/gsm8k` name=main | `740312add88f` | converter `gsm8k_question_answer`, budget 16,114 sequences (33.0M tokens) |
| `github_code_clean_python` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Python, text_field `code`, budget 61,524 sequences (126.0M tokens) |
| `github_code_clean_javascript` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language JavaScript, text_field `code`, budget 41,016 sequences (84.0M tokens) |
| `github_code_clean_typescript` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language TypeScript, text_field `code`, budget 20,508 sequences (42.0M tokens) |
| `github_code_clean_java` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Java, text_field `code`, budget 20,508 sequences (42.0M tokens) |
| `github_code_clean_cpp` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language C++, text_field `code`, budget 16,407 sequences (33.6M tokens) |
| `github_code_clean_go` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language GO, text_field `code`, budget 14,356 sequences (29.4M tokens) |
| `github_code_clean_rust` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Rust, text_field `code`, budget 10,254 sequences (21.0M tokens) |
| `github_code_clean_shell` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Shell, text_field `code`, budget 8,204 sequences (16.8M tokens) |
| `github_code_clean_sql` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language SQL, text_field `code`, budget 6,153 sequences (12.6M tokens) |
| `github_code_clean_html` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language HTML, text_field `code`, budget 6,153 sequences (12.6M tokens) |
| `flan` | instruct | `hf_files` | `Open-Orca/FLAN` data_files=flan_zsopt_data/*.parquet | `6845b1b3b53c` | fields instruction←`inputs`, output←`targets`, budget 29,297 sequences (60.0M tokens), input inversions 5%, shuffled (seed 42) |
| `metamath` | instruct | `hf_files` | `meta-math/MetaMathQA` data_files=MetaMathQA-395K.json | `aa4f34d3d2d3` | fields instruction←`query`, output←`response`, budget 10,987 sequences (22.5M tokens), input inversions 5%, shuffled (seed 42) |
| `orca_math` | instruct | `hf_files` | `microsoft/orca-math-word-problems-200k` data_files=data/*.parquet | `29255d1770cc` | fields instruction←`question`, output←`answer`, budget 7,325 sequences (15.0M tokens), input inversions 5%, shuffled (seed 42) |
| `evol_code` | instruct | `hf_files` | `nickrosh/Evol-Instruct-Code-80k-v1` data_files=EvolInstruct-Code-80k.json | `3ae930c20d54` | fields instruction←`instruction`, output←`output`, budget 9,156 sequences (18.8M tokens), input inversions 5%, shuffled (seed 42) |
| `code_alpaca` | instruct | `hf_files` | `sahil2801/CodeAlpaca-20k` data_files=code_alpaca_20k.json | `152bb5e9a296` | fields instruction←`instruction`, input←`input`, output←`output`, budget 1,832 sequences (3.8M tokens), input inversions 5%, shuffled (seed 42) |
| `slimorca` | instruct | `hf_files` | `Open-Orca/SlimOrca-Dedup` data_files=data/*.parquet | `bd7d445aa1ff` | converter `sharegpt_conversations`, budget 7,325 sequences (15.0M tokens), input inversions 5%, shuffled (seed 42) |
| `sharegpt` | instruct | `hf_files` | `anon8231489123/ShareGPT_Vicuna_unfiltered` data_files=ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json | `192ab2185289` | converter `sharegpt_conversations`, filter `sharegpt_quality`, check_limit 100,000, budget 3,663 sequences (7.5M tokens), input inversions 5%, shuffled (seed 42) |
| `wizardlm` | instruct | `hf_files` | `WizardLM/WizardLM_evol_instruct_V2_196k` data_files=WizardLM_evol_instruct_V2_143k.json | `8a7d15a83028` | converter `first_two_turns`, budget 3,663 sequences (7.5M tokens), input inversions 5%, shuffled (seed 42) |
