# Dataset `crow-300m-final`

Generated from `config/datasets/crow_300m_final.yaml` with

```bash
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/crow_300m_final.yaml > docs/data_mixture.md
```

Do not edit by hand: change the dataset config and regenerate. Token budgets are the stage budgets of the
config times the mixture weights; row and example counts are estimates from `tokens_per_row_estimate` (the
planner refines them with measured token counts once a source has been processed).

## Notes

Dataset definition of the thesis run (crow-300m-final). Referenced from `config/crow_300m_final.yaml` via
`dataset_config`. Materialise with `python data_preparation/prepare.py build --dataset_config <this file>`;
training does it automatically when something is missing (`auto_prepare`). `nampdn-ai/mini-peS2o` is gated:
export `HF_TOKEN` before building. This comment block is rendered into `docs/data_mixture.md` by
`prepare.py describe`.

Relation to the thesis run: the thesis data was prepared with exact deduplication, fuzzy (MinHash) deduplication
at Jaccard threshold 0.95 and real tokenizer counts truncated to 2048 tokens, with the quality filter, benchmark
decontamination and PII masking all skipped. This config mirrors that as closely as the restructured pipeline
allows: exact dedup on, tokenizer counts capped at 2048, quality filter and decontamination off. Fuzzy dedup is
off here (it is available as `dedup: {mode: minhash, threshold: 0.95, num_perm: 256}`), and PII masking no longer
exists. The per-stage token budgets are the thesis budgets (3.3B + 1.5B + 0.15B tokens); the stage weights are
the thesis run's; per-source download sizes are derived from budget × weight × 1.2 instead of the fixed row
counts of the thesis scripts. Several HuggingFace ids had to move (wikipedia -> wikimedia/wikipedia 20231101.en,
gsm8k -> openai/gsm8k, arxiv -> common-pile/arxiv_papers_filtered), and every source is pinned to a revision.

## Tokenizer and token counting

- tokenizer: `llama-32k` (hf, `hf-internal-testing/llama-tokenizer` @ `d02ad6cb9dd2c2296a6332199fa2fdca5938fef0`)
- `max_seq_length`: 2048 (token-count cap per document; the run config's `block_size` must not exceed it)
- `token_count`: `tokenizer` (real tokenizer counts)
- training tokens over all stages: 4.95B

## Processing defaults (`pretrain` sources)

- length filter: 50 <= chars, truncated at 20000 chars
- dedup: `exact` (normalize: on)
- quality filter: off
- decontamination: off

## Stages

### Stage 1: `pretrain_phase1` (3.30B tokens, transition 10%)

| Train source | Weight | Tokens | Tokens/row (est.) |
|---|---:|---:|---:|
| `fineweb_edu` | 65.00% | 2.15B | 2000 |
| `wikipedia` | 9.00% | 297.0M | 1500 |
| `books_gutenberg` | 6.00% | 198.0M | 3000 |
| `github_code_clean_python` | 3.60% | 118.8M | 500 |
| `github_code_clean_javascript` | 2.40% | 79.2M | 500 |
| `github_code_clean_typescript` | 1.20% | 39.6M | 500 |
| `github_code_clean_java` | 1.20% | 39.6M | 500 |
| `github_code_clean_cpp` | 0.96% | 31.7M | 500 |
| `github_code_clean_go` | 0.84% | 27.7M | 500 |
| `github_code_clean_rust` | 0.60% | 19.8M | 500 |
| `github_code_clean_shell` | 0.48% | 15.8M | 500 |
| `github_code_clean_sql` | 0.36% | 11.9M | 500 |
| `github_code_clean_html` | 0.36% | 11.9M | 500 |
| `peso` | 3.00% | 99.0M | 1500 |
| `arxiv` | 2.00% | 66.0M | 1500 |
| `openwebmath` | 3.00% | 99.0M | 750 |

Validation: `fineweb_val` at 100%

### Stage 2: `pretrain_phase2` (1.50B tokens, transition 10%)

| Train source | Weight | Tokens | Tokens/row (est.) |
|---|---:|---:|---:|
| `fineweb_edu` | 35.00% | 525.0M | 2000 |
| `github_code_clean_python` | 8.40% | 126.0M | 500 |
| `github_code_clean_javascript` | 5.60% | 84.0M | 500 |
| `github_code_clean_typescript` | 2.80% | 42.0M | 500 |
| `github_code_clean_java` | 2.80% | 42.0M | 500 |
| `github_code_clean_cpp` | 2.24% | 33.6M | 500 |
| `github_code_clean_go` | 1.96% | 29.4M | 500 |
| `github_code_clean_rust` | 1.40% | 21.0M | 500 |
| `github_code_clean_shell` | 1.12% | 16.8M | 500 |
| `github_code_clean_sql` | 0.84% | 12.6M | 500 |
| `github_code_clean_html` | 0.84% | 12.6M | 500 |
| `openwebmath` | 8.80% | 132.0M | 750 |
| `tinygsm` | 6.60% | 99.0M | 300 |
| `algebraic_stack` | 4.40% | 66.0M | 750 |
| `gsm8k` | 2.20% | 33.0M | 300 |
| `peso` | 9.00% | 135.0M | 1500 |
| `arxiv` | 6.00% | 90.0M | 1500 |

Validation: `fineweb_val` at 100%

### Stage 3: `finetune` (150.0M tokens, transition 0%)

| Train source | Weight | Tokens | Tokens/row (est.) |
|---|---:|---:|---:|
| `flan_mixture` (mixture) | 100.00% | 150.0M | - |

Validation: `flan_mixture/validation` (mixture) at 100%

## Instruct mixtures

### `flan_mixture` (150.0M tokens budget)

`max_tokens` 2048, input inversions 5%, validation split 5%, seed 42. Examples = budget × share ÷ `tokens_per_row_estimate`.

| Source | Share | Tokens | Examples (est.) |
|---|---:|---:|---:|
| `flan` | 40.0% | 60.0M | 200,000 |
| `metamath` | 15.0% | 22.5M | 75,000 |
| `orca_math` | 10.0% | 15.0M | 50,000 |
| `evol_code` | 12.5% | 18.8M | 46,875 |
| `code_alpaca` | 2.5% | 3.8M | 18,750 |
| `slimorca` | 10.0% | 15.0M | 50,000 |
| `sharegpt` | 5.0% | 7.5M | 18,750 |
| `wizardlm` | 5.0% | 7.5M | 18,750 |

## Held-out validation sets

| Source | Rows | Seed | Loader | Origin |
|---|---:|---:|---|---|
| `fineweb_val` | 50,000 | 42 | `hf_files` | `HuggingFaceFW/fineweb-edu` data_files=sample/10BT/*.parquet |

## Sources

| Source | Kind | Loader | Origin | Revision | Details |
|---|---|---|---|---|---|
| `fineweb_edu` | pretrain | `hf_files` | `HuggingFaceFW/fineweb-edu` data_files=data/CC-MAIN-2013-20/*.parquet | `87f09149ef47` | budget 2.15B |
| `wikipedia` | pretrain | `hf_files` | `wikimedia/wikipedia` data_files=20231101.en/*.parquet | `b04c8d1ceb2f` | budget 297.0M |
| `books_gutenberg` | pretrain | `hf_files` | `sedthh/gutenberg_english` data_files=data/*.parquet | `28973b04f28f` | text_field `TEXT`, budget 198.0M |
| `peso` | pretrain | `hf_files` | `nampdn-ai/mini-peS2o` data_files=train-*.parquet | `18a60ef8d79f` | budget 135.0M |
| `arxiv` | pretrain | `hf_files` | `common-pile/arxiv_papers_filtered` data_files=arxiv-papers-*.json.gz | `033cf7f53f9b` | budget 90.0M |
| `openwebmath` | pretrain | `hf_files` | `open-web-math/open-web-math` data_files=data/*.parquet | `fde8ef8de230` | budget 132.0M |
| `tinygsm` | pretrain | `hf_files` | `ostapeno/tinygsm-mind` data_files=data/*.parquet | `f5ecf416b715` | budget 99.0M |
| `algebraic_stack` | pretrain | `hf_files` | `EleutherAI/proof-pile-2` data_files=algebraic-stack/train/*.jsonl.zst | `901a9273a770` | budget 66.0M |
| `gsm8k` | pretrain | `hf_split` | `openai/gsm8k` name=main | `740312add88f` | converter `gsm8k_question_answer`, repeated to budget, budget 33.0M |
| `github_code_clean_python` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Python, text_field `code`, budget 126.0M |
| `github_code_clean_javascript` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language JavaScript, text_field `code`, budget 84.0M |
| `github_code_clean_typescript` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language TypeScript, text_field `code`, budget 42.0M |
| `github_code_clean_java` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Java, text_field `code`, budget 42.0M |
| `github_code_clean_cpp` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language C++, text_field `code`, budget 33.6M |
| `github_code_clean_go` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language GO, text_field `code`, budget 29.4M |
| `github_code_clean_rust` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Rust, text_field `code`, budget 21.0M |
| `github_code_clean_shell` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language Shell, text_field `code`, budget 16.8M |
| `github_code_clean_sql` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language SQL, text_field `code`, budget 12.6M |
| `github_code_clean_html` | pretrain | `github_code` | `codeparrot/github-code-clean` | `c48d40f9e70f` | language HTML, text_field `code`, budget 12.6M |
| `fineweb_val` | holdout | `hf_files` | `HuggingFaceFW/fineweb-edu` data_files=sample/10BT/*.parquet | `87f09149ef47` | - |
| `flan` | instruct | `hf_files` | `Open-Orca/FLAN` data_files=flan_zsopt_data/*.parquet | `6845b1b3b53c` | fields instruction←`inputs`, output←`targets` |
| `metamath` | instruct | `hf_split` | `meta-math/MetaMathQA` data_files=MetaMathQA-395K.json | `aa4f34d3d2d3` | fields instruction←`query`, output←`response` |
| `orca_math` | instruct | `hf_files` | `microsoft/orca-math-word-problems-200k` data_files=data/*.parquet | `29255d1770cc` | fields instruction←`question`, output←`answer` |
| `evol_code` | instruct | `hf_split` | `nickrosh/Evol-Instruct-Code-80k-v1` data_files=EvolInstruct-Code-80k.json | `3ae930c20d54` | fields instruction←`instruction`, output←`output` |
| `code_alpaca` | instruct | `hf_split` | `sahil2801/CodeAlpaca-20k` data_files=code_alpaca_20k.json | `152bb5e9a296` | fields instruction←`instruction`, input←`input`, output←`output` |
| `slimorca` | instruct | `hf_files` | `Open-Orca/SlimOrca-Dedup` data_files=data/*.parquet | `bd7d445aa1ff` | converter `sharegpt_conversations` |
| `sharegpt` | instruct | `hf_split` | `anon8231489123/ShareGPT_Vicuna_unfiltered` data_files=ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json | `192ab2185289` | converter `sharegpt_conversations`, filter `sharegpt_quality`, check_limit 100,000 |
| `wizardlm` | instruct | `hf_split` | `WizardLM/WizardLM_evol_instruct_V2_196k` data_files=WizardLM_evol_instruct_V2_143k.json | `8a7d15a83028` | converter `first_two_turns` |
