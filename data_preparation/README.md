# Data preparation

A **dataset config** (`config/datasets/<name>.yaml`) is the single definition of a dataset: its sources, the
tokenizer, the processing options, the per-stage token budgets with their train/validation mixtures, and the
instruct mixtures. `python data_preparation/prepare.py build --dataset_config <file>` materialises it under
`dataset/`; a run config (`config/<run>.yaml`) references it via `dataset_config:` and `training/train.py` verifies
the prepared data before training, building whatever is missing by default. Everything runs from the repo root
with `uv run ...`; the implementation lives in `lib/` (tests beside every module).

## The dataset config

```yaml
name: crow-300m-final                    # -> dataset/mixtures/crow-300m-final/
tokenizer: {name: llama-32k, kind: hf, hf_id: hf-internal-testing/llama-tokenizer, revision: <sha>}
max_seq_length: 2048                     # token-count cap per document; the run config's block_size must be <= this
token_count: tokenizer                   # count with the real tokenizer (or: estimate = chars / 4)
processing:                              # defaults for every pretrain source; a source may carry its own block
  min_chars: 50                          # drop shorter documents
  max_chars: 20000                       # truncate longer documents
  dedup: {mode: exact, normalize: true}  # or {mode: minhash, threshold: 0.95, num_perm: 256, ngram: 5}, or none
  quality_filter: false
  decontamination: {enabled: false}      # benchmarks: [gsm8k_test, math_test, ...], ngram: 13, threshold: 0.1
sources:                                 # keyed by name; `kind` selects the pipeline, `loader` how rows are fetched
  fineweb_edu: {kind: pretrain, loader: hf_split, hf_id: HuggingFaceFW/fineweb-edu, revision: <sha>,
                load_kwargs: {name: CC-MAIN-2013-20}, tokens_per_row_estimate: 2000}
  gsm8k:       {kind: pretrain, loader: hf_split, hf_id: openai/gsm8k, revision: <sha>, load_kwargs: {name: main},
                converter: gsm8k_question_answer, repeat_to_budget: true, tokens_per_row_estimate: 300}
  fineweb_val: {kind: holdout, loader: hf_split, hf_id: HuggingFaceFW/fineweb-edu, revision: <sha>,
                load_kwargs: {name: sample-10BT}, rows: 50000, seed: 42}
  flan:        {kind: instruct, loader: hf_stream, hf_id: Open-Orca/FLAN, revision: <sha>,
                fields: {instruction: inputs, output: targets}, tokens_per_row_estimate: 300}
mixtures:                                # instruct mixtures, built per dataset config from `instruct` sources
  flan_mixture: {sources: {flan: 1.0}, max_tokens: 2048, input_inversions: 0.05, val_split: 0.05, seed: 42}
stages:                                  # the training curriculum; weights per stage sum to 1
  - {name: pretrain_phase1, tokens: 3_300_000_000, transition_pct: 0.10,
     train: {fineweb_edu: 0.99, gsm8k: 0.01}, val: {fineweb_val: 1.0}}
  - {name: finetune, tokens: 150_000_000, train: {flan_mixture: 1.0}, val: {flan_mixture/validation: 1.0}}
```

Three source kinds: `pretrain` (a `text` column, goes through the processing pipeline and can be used in `train`),
`holdout` (a fixed number of rows for validation only, no dedup/filters) and `instruct` (`instruction/input/output`
rows, only usable through a mixture; `<mixture>`, `<mixture>/train` and `<mixture>/validation` are the stage keys).
The schema with every field and the validation rules is `lib/schema/dataset_config.py`; loading fails on unknown
loader/converter names, weights that do not sum to 1, holdouts in `train`, and so on. The configs in the tree are
`config/datasets/crow_300m_final.yaml` (the thesis run; `docs/data_mixture.md` is generated from it),
`config/datasets/crow_300m_mini.yaml` (the same sources with 300k / 150k / 60k-token budgets and a 40-row holdout:
a real-source smoke build of a few MB that finishes in minutes and exercises every loader; needs `HF_TOKEN` for
`mini-peS2o`; `tools/capped_download.sh 500 uv run python data_preparation/prepare.py build --dataset_config
config/datasets/crow_300m_mini.yaml` runs it under a hard download cap) and `config/datasets/tiny.yaml` (synthetic,
builds in seconds, used by the tests and `config/tiny.yaml`).

## On-disk layout and manifests

```
dataset/
├── sources/<source>/
│   ├── raw/          MANIFEST.json + data-*.parquet     rows as fetched (converter applied), append-only
│   ├── filtered/     MANIFEST.json + data-*.parquet     length filter, one shard per raw shard
│   ├── processed/    MANIFEST.json + data-*.parquet     dedup / filters / token counts   <- training reads this
│   └── holdout/      MANIFEST.json + data-*.parquet     `holdout` sources only            <- validation reads this
├── mixtures/<config name>/<mixture>/{train,validation}/   MANIFEST.json + shards           <- finetune stage
├── tokenizers/<tokenizer name>/                            MANIFEST.json + tokenizer files
├── hub_index/<repo>@<revision>/<glob hash>.json            file lists, sizes, row counts + parquet row-group layout of `hf_files` / `github_code` repos
└── benchmarks/                                             cached benchmark test sets (decontamination only)
```

`sources/` is shared by every dataset config and by every stage that uses the source (stages 1 and 2 of the thesis
config draw from the same `fineweb_edu/processed`, only with different weights). Only mixtures are per config.
Every directory carries a `MANIFEST.json` (`lib/storage/manifest.py`): the `source_hash` of the config that built
it (loader settings + processing + token-count mode; budgets and weights are *not* part of it), rows and tokens per
shard, the loader offset reached, tokenizer and library versions. A manifest whose hash differs from the current
config is stale and its directory is rebuilt; shard writes go to a `.tmp` directory first and are renamed into
place, so an interrupted build never leaves a half-written stage behind.

## Commands

```bash
uv run python data_preparation/prepare.py build    --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
        [--sources NAME ...] [--steps tokenizer download filter process holdout mixtures] [--num_workers N]
        [--hf_token T] [--cache_dir DIR] [--dry_run]
uv run python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml > docs/data_mixture.md
uv run python data_preparation/prepare.py tiny     # = build --dataset_config config/datasets/tiny.yaml
```

- `build` computes the plan (what the config needs versus what the manifests say is there) and runs only the
  missing parts, in the order tokenizer → pretrain sources (download → filter → process) → holdouts → mixtures.
  `--sources` / `--steps` restrict it, `--dry_run` prints the plan and writes nothing. A failing source is a
  failing build (exit 1) — never a silently smaller dataset.
- `status` prints the same table (rows/tokens present versus needed, fetch increment, manifest state) and exits 0
  iff the dataset is complete.
- `describe` renders the config as Markdown (tokenizer, processing defaults, stage tables with derived token
  budgets, mixture tables with derived example counts, holdouts, sources). The comment block at the top of the YAML
  becomes its "Notes" section. `docs/data_mixture.md` is that output for the crow config; regenerate it after
  editing the config.

## Budgets and incremental builds

The config states token budgets; the planner (`lib/build/planner.py`) turns them into row counts. Per pretrain
source the needed tokens are the **largest** per-stage demand (`stage.tokens × weight`, max over stages, since the
files are shared), the needed rows are `tokens ÷ tokens_per_row × 1.2` (safety margin), where `tokens_per_row` is
the measured value from the processed manifest when it is current and `tokens_per_row_estimate` from the config
before the first build. The download increment is `rows needed − rows present`, clamped at 0 (a remote parquet fetch may leave more rows
than needed on disk, see "Where things are cached"; that is fine, the next plan simply has nothing to fetch). If the processed
tokens still fall short of the budget, the runner (`lib/build/runner.py`) refines the estimate with the measured
value and fetches another increment — at most 5 rounds, then an error. A loader that runs dry marks the source
`exhausted`; the build completes with a warning. `repeat_to_budget` sources (gsm8k) are fetched whole once and
their rows repeated at process time.

Sources are append-only and deterministic: loaders read the repo's files in sorted order, one file at a time
(`hf_files`, `github_code`), fetch `train[offset:offset+count]` at a pinned `revision` (`hf_split`), stream with
`skip(offset)` (`hf_stream`) or read local files in sorted order (`local`), so a config or stage that needs more
rows appends the next slice and one that needs fewer reads a prefix. `filter`
processes only raw shards newer than its manifest; `process` re-runs the streaming exact dedup over old + new
shards (first occurrence wins, so old rows are kept unchanged). Mixtures are rebuilt whenever their definition, their
budget or any input source's raw shard list changed.

## Where things are cached

Two caches, with different lifetimes:

* **Hub cache** (`~/.cache/huggingface/hub`, or `HF_HOME` / `--cache_dir`): the original repo files fetched by
  `hf_files` / `github_code` (`hf_hub_download`, one file at a time, never twice) and the `datasets` cache of
  `hf_split` sources. Deleting it costs a re-download; nothing else depends on it. Only files up to
  `max_cached_file_mb` (default 32, per source via `load_kwargs.max_cached_file_mb`) land here: larger parquet
  files are read remotely row group by row group, projected to the columns the source needs (a top-up seeks
  straight to the row group it needs; fineweb-edu's 2.4 GB files cost a few MB per 1000 rows). A remote parquet
  fetch keeps **every** row of the row groups it read — `rows needed` is a minimum, the raw shards and
  `rows_fetched` advance to the row-group boundary — so a top-up never re-downloads a row group; row groups can be
  large for book-like sources (gutenberg is ~300 MB per 1,000 rows, so the smallest fetch costs one such group).
  Larger `.jsonl[.zst|.gz]` and plain `.json` array files are streamed from the start until exactly enough rows
  were read (`.json` arrays are parsed incrementally with `ijson`; their row count is only recorded once read to
  the end, so a top-up inside a partially consumed file re-streams that one file from its start — a prefix read,
  a few MB for the first few hundred rows).
* **`dataset/sources/<source>/raw/`**: the rows this pipeline kept, in shards, with a manifest — the append-only
  source cache that `filter` / `process` / mixtures are built from. `dataset/hub_index/` holds the small JSON file
  indexes (file list per glob, rows per file, row-group layout and per-language rows per row group) that let `hf_files` fetch at an offset without opening earlier files;
  it is safe to delete (rebuilt on demand from the repo listing, the file sizes and the cached / remote files).

## Progress bars

`build` shows a bar over the plan items and every stage shows its own (`<source>: download` with the rows kept,
source rows consumed and the current repo file; `length_filter` per shard; `process` per row with the running token
count; `holdout`; `build_mixture` per source and per step). Bars go to stderr and are disabled automatically when
stderr is not a terminal or when `DATA_PREP_PROGRESS=0` is set; log lines are written through `tqdm.write` so they
do not garble the bars. `hf_hub_download` prints its own byte-level bar per file.

## Training auto-prepares

`training/train.py` loads the run config's `dataset_config`, runs `status`, and — with `auto_prepare: true` (the
default) — runs `build` in-process on the main rank (`prepare_num_workers` workers), then re-verifies. With
`auto_prepare: false` a missing dataset is a hard error quoting the `prepare.py build` command. Gated sources
(`nampdn-ai/mini-peS2o` in the crow config) need `HF_TOKEN` in the environment (or `--hf_token` for `build`).
The dataset config's hash is written into every checkpoint; resuming with a changed dataset config is an error
unless the run config sets `allow_dataset_change: true`.

## Adding a source

One YAML entry under `sources:` plus the weight where it is used (a stage's `train`, or a mixture's `sources`).
Pin the `revision` (`git ls-remote` on the Hub repo, or the commit shown on the dataset page) so row order is stable
across increments, and give a `tokens_per_row_estimate` in the right ballpark so the first download is not far off.

Loaders (`lib/sources/loaders.py`, `loader:`; all are `(source, offset, count) -> Iterator[row]`):

| Loader | Use for | Notes |
|---|---|---|
| `hf_files` | **default for Hub repos with many files** | `load_kwargs: {data_files: <glob>, max_cached_file_mb: 32}` (`data_files` required, relative to the repo root); files sorted by path; files up to `max_cached_file_mb` are downloaded one at a time into the Hub cache on demand and read locally, larger `.parquet` files are read remotely by row group (every row of a fetched row group is kept, so a top-up never re-downloads one) and larger `.jsonl`, `.jsonl.zst`, `.jsonl.gz`/`.json.gz` files and plain `.json` arrays (incrementally via `ijson`) are streamed from the start; a file index under `dataset/hub_index/` lets a top-up skip files already consumed |
| `hf_split` | split-name based repos (e.g. `gsm8k` with `name: main`) | `train[a:b]` slicing; `datasets` downloads and caches the file once and slices locally; `load_kwargs` go to `load_dataset` (`name`, `data_files`, ...) |
| `hf_stream` | fallback | `datasets` streaming with `skip(offset)`; **caches nothing** — every fetch re-streams from the start, so avoid it for anything large |
| `github_code` | `codeparrot/github-code-clean` | `hf_files` over `data/*.parquet` keeping rows of `language:` (`text_field: code`); all language sources read the same cached files and the shared index stores per-language row counts (per row group for partially read parquet files) |
| `local` | your own data | `path:` directory of `*.parquet` / `*.jsonl` files, read in sorted file order |
| `synthetic` | tests / smoke runs | random-word rows from `seed:` |

Pretrain and holdout sources need a text column (`text_field`, default `text`); instruct sources need either
`fields: {instruction: <col>, input: <col>, output: <col>}` (input optional) or a `converter`. Converters and filters
(`lib/sources/converters.py`) turn one source row into the row the pipeline expects: `gsm8k_question_answer`
(question + answer → `text`), `sharegpt_conversations` (`from`/`value` turns → instruction/output),
`first_two_turns` (a `conversations` list), `instruction_input_output` (already standardized); filter
`sharegpt_quality` (keeps human→gpt openings with 50–2000 characters per side and no code block in the answer;
`check_limit` bounds the rows inspected). Write a
new converter when the source's schema is not a simple column mapping: add a function `(row) -> row` to
`converters.py`, register it in `CONVERTERS` (or `FILTERS`), test it on a hand-written row in `test_sources.py`, and
reference it by name in the YAML. A converter raising `ValueError` skips the row (counted in the manifest as
`skipped_malformed`).

### A local dataset

```yaml
sources:
  my_corpus: {kind: pretrain, loader: local, path: /data/my_corpus, text_field: text, tokens_per_row_estimate: 800}
  my_val:    {kind: holdout,  loader: local, path: /data/my_corpus, rows: 2000, seed: 1}
```

`local` reads every `*.parquet` / `*.jsonl` file directly under `path` in sorted order. A holdout over the same
directory takes the **last** `rows` rows, so it stays disjoint from a training source reading the first rows as long
as the training source needs fewer than `total − rows` rows (for `hf_split` holdouts choose a different split or
subset, as the crow config does with fineweb-edu's `sample-10BT`; `hf_stream` holdouts take the first rows).

## Processing toggles

`process` (`lib/stages/pretrain.py`) runs per pretrain source, in this order: exact dedup → quality filter →
decontamination → token counting → fuzzy dedup → repetition (`repeat_to_budget`). Each is configured in the
`processing` block (dataset-level default, per-source override):

- `dedup.mode: exact` (default) hashes the normalized text (lowercased, whitespace collapsed; `normalize: false` for
  verbatim) in one streaming pass, first occurrence wins. `minhash` adds MinHash/LSH near-duplicate removal
  (`threshold`, `num_perm`, `ngram`; `lib/stages/fuzzy_dedup.py`, signatures computed with `--num_workers`
  processes into a compact array, needs the `datasketch` extra). `none` disables both.
- `quality_filter: true` keeps documents with ≥ 3 sentences, ≤ 30 % ALL-CAPS words, ≥ 25 % alphanumeric characters,
  ≤ 30 % repeated 2-grams and ≤ 20 % repeated 3-grams — prose heuristics, wrong for code.
- `decontamination.enabled: true` drops documents whose 13-grams overlap more than `threshold` with a benchmark test
  set (`benchmarks`: gsm8k_test, math_test, humaneval, mbpp_test, arc_challenge_test, hellaswag_test, mmlu_test,
  winogrande_test — downloaded once into `dataset/benchmarks/`, `lib/stages/benchmarks.py`). Needs `datasets` and
  network access on first use.
- Token counting (`token_count`) is done with the real tokenizer, capped at `max_seq_length`, **without rewriting
  the text** (training truncates); `estimate` uses chars / 4.

Instruct mixtures (`lib/stages/instruct.py`) take `ceil(budget × share ÷ measured tokens/row)` rows per source,
drop examples longer than `max_tokens`, apply input inversions on a seeded sample, the same normalized exact dedup,
remove rows with empty fields, shuffle and split.

## Differences from the thesis run

The thesis data was prepared with exact dedup, fuzzy dedup at Jaccard 0.95, tokenizer counts truncated to 2048
tokens and the quality filter, decontamination and PII masking skipped; the pretrain sources were fetched as fixed
row counts (e.g. 9.0M fineweb-edu documents) and the finetune mixture as 400k examples. The crow config mirrors
that with exact dedup on, tokenizer counts capped at 2048 and everything else off, but: fuzzy dedup is off by
default (available as `dedup: {mode: minhash, threshold: 0.95}`), PII masking no longer exists, download sizes
follow the token budgets (× 1.2) instead of fixed row counts, several HuggingFace ids moved
(`wikimedia/wikipedia`, `openai/gsm8k`, `common-pile/arxiv_papers_filtered`) and every source is pinned to a
revision. `docs/data_mixture.md` lists the resulting budgets.
