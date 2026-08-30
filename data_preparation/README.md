# Data preparation

A **dataset config** (`config/datasets/<name>.yaml`) is the single definition of a dataset: its sources, the
tokenizer, the processing options, the per-stage token budgets with their train/validation mixtures, and the
instruct mixtures. `python data_preparation/prepare.py build --dataset_config <file>` materialises it under
`dataset/`; a run config (`config/<run>.yaml`) references it via `dataset_config:` and `training/train.py` verifies
the prepared data before training, building whatever is missing by default. Everything runs from the repo root
with `uv run ...`; the implementation lives in `lib/` (tests beside every module).

## The dataset config

Config reference: `data_preparation/dataset_config.py` itself (the `DatasetConfig` docstring lists the top-level keys,
every field carries a `# ...` comment, `__post_init__` holds the validation rules); `layout.py` beside it maps a config
to its directories under `dataset/`.

```yaml
name: crow-300m-final                    # -> dataset/instruct_mixtures/crow-300m-final/
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
  fineweb_edu: {kind: pretrain, loader: hf_files, hf_id: HuggingFaceFW/fineweb-edu, revision: <sha>,
                load_kwargs: {data_files: "data/CC-MAIN-2013-20/*.parquet"}, tokens_per_row_estimate: 2000,
                validation_tokens: 50_000_000}   # hold the first 50M processed tokens out as fineweb_edu/validation
  gsm8k:       {kind: pretrain, loader: hf_split, hf_id: openai/gsm8k, revision: <sha>, load_kwargs: {name: main},
                converter: gsm8k_question_answer, tokens_per_row_estimate: 300}
  other_val:   {kind: validation, loader: hf_files, hf_id: some/other-corpus, revision: <sha>,
                load_kwargs: {data_files: "*.parquet"}, rows: 5000, seed: 42}   # an external held-out set
  flan:        {kind: instruct, loader: hf_stream, hf_id: Open-Orca/FLAN, revision: <sha>,
                fields: {instruction: inputs, output: targets}, tokens_per_row_estimate: 300}
instruct_mixtures:                                # instruct mixtures, built per dataset config from `instruct` sources
  flan_instruct: {sources: {flan: 1.0}, max_tokens: 2048, input_inversions: 0.05, val_split: 0.05, seed: 42}
stages:                                  # the training curriculum; weights per stage sum to 1
  - {name: pretrain_phase1, tokens: 3_300_000_000, transition_pct: 0.10,
     train: {fineweb_edu: 0.99, gsm8k: 0.01}, val: {fineweb_edu/validation: 0.5, other_val: 0.5}}
  - {name: finetune, tokens: 150_000_000, train: {flan_instruct: 1.0}, val: {flan_instruct/validation: 1.0}}
```

Three source kinds: `pretrain` (a `text` column, goes through the processing pipeline and can be used in `train`;
with `validation_tokens` the **first** processed rows worth that many tokens are held out as the stage key
`<source>/validation` — deduplicated together with the training rows, so the two never share a document, and fixed
by the append-only layout, so a later top-up never moves the boundary; this is how the crow config validates),
`validation` (an external held-out set: a fixed number of rows, no dedup/filters — make sure it is disjoint from
every training source, the planner warns when one reads the same Hub repo and file prefix as a pretrain source)
and `instruct` (`instruction/input/output` rows, only usable through a mixture; `<mixture>/train` in `train` and
`<mixture>/validation` in `val` are the stage keys; a bare `<mixture>` means `/train`). The schema with every field
and the validation rules is `data_preparation/dataset_config.py`; loading fails on unknown loader/converter names,
weights that do not sum to 1, validation sources or splits in `train`, a mixture's train split in `val`, a pretrain
source that only appears in `val` (it would never be prepared), and so on. The configs in the tree are
`config/datasets/crow_300m_final.yaml` (the thesis run; `docs/data_mixture.md` is generated from it),
`config/datasets/crow_300m_mini.yaml` (the same sources with 300k / 150k / 60k-token budgets and a 40k-token validation split:
a real-source smoke build of a few MB that finishes in minutes and exercises every loader; needs `HF_TOKEN` for
`mini-peS2o`; `tools/capped_download.sh 500 uv run python data_preparation/prepare.py build --dataset_config
config/datasets/crow_300m_mini.yaml` runs it under a hard download cap) and `config/datasets/tiny.yaml` (synthetic,
builds in seconds, used by the tests and `config/tiny.yaml`).

## On-disk layout and manifests

```
dataset/
├── sources/<source>/
│   ├── raw/          MANIFEST.json + data-*.parquet     rows as fetched (converter applied) + `tokens`, append-only
│   ├── processed/    MANIFEST.json + data-*.parquet     length filter / dedup / filters / token counts, append-only   <- training reads this
│   └── validation/   MANIFEST.json + data-*.parquet     `validation` sources, or the `validation_tokens` split of a pretrain source   <- validation reads this
├── instruct_mixtures/<config name>/<mixture>/{train,validation}/   MANIFEST.json + shards           <- finetune stage
├── tokenizers/<tokenizer name>/                            MANIFEST.json + tokenizer files
├── hub_index/<repo>@<revision>/<glob hash>.json            file lists, sizes, row counts + parquet row-group layout of `hf_files` / `github_code` repos
└── benchmarks/                                             cached benchmark test sets (decontamination only)
```

`sources/` is shared by every dataset config and by every stage that uses the source (stages 1 and 2 of the thesis
config draw from the same `fineweb_edu/processed`, only with different weights). Only instruct mixtures are per config.
Every directory carries a `MANIFEST.json` (`lib/storage/manifest.py`): the hash of the config that built it, rows
and tokens per shard, the loader offset reached, how tokens were counted (mode, tokenizer, cap), tokenizer and
library versions. `raw/` is keyed on `DatasetConfig.raw_hash` — the loader identity only (repo, revision, files,
split, converter, seed, ...), so processing options, `max_seq_length`, `token_count`, the tokenizer, `check_limit`,
budgets and weights never invalidate downloaded shards (a changed token setting recounts the `tokens` column in
place). `processed/` is keyed on `processed_hash` (raw hash + processing block + token settings) and is rebuilt
from the raw shards when it changes; validation dirs and mixtures likewise on the raw hashes of their sources plus
the token settings. Raw downloads and `processed/` are append-only and publish **shard by shard** (each shard is
written to a `.tmp` file, renamed into place and recorded in the manifest — raw shards with the loader offset
after their last row), so a network error, a crash or Ctrl-C keeps everything fetched so far and the next run
resumes behind the last complete shard; a corrupt raw shard only drops that shard and the ones after it. Stages
that rewrite a directory as a whole (validation sets, instruct mixtures, `processed/` in minhash mode) write into
a `.tmp` directory and rename it into place. Ctrl-C stops every running stage at its next shard.

## Commands

```bash
uv run python data_preparation/prepare.py build    --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
        [--sources NAME ...] [--steps tokenizer download process validation instruct_mixtures] [--num_workers N] [--max_parallel_downloads N]
        [--hf_token T] [--cache_dir DIR] [--dry_run]
uv run python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml > docs/data_mixture.md
uv run python data_preparation/prepare.py tiny     # = build --dataset_config config/datasets/tiny.yaml
```

- `build` computes the plan (what the config needs versus what the manifests say is there) and runs only the
  missing parts: the tokenizer first, then every pretrain source (download → process, with the estimate → measured
  refinement rounds), validation source, instruct source and instruct mixture as a work item in a bounded thread
  pool — `--max_parallel_downloads` items download and `--num_workers` items process at any time, so the network
  and the CPU overlap across sources; the `github_code` sources of one repo download together in a single pass over
  its files, instruct mixtures run after their sources. `--sources` / `--steps` restrict it, `--dry_run` prints the
  plan and writes nothing. A failing item is a failing build (exit 1, the other items stop at their next shard) —
  never a silently smaller dataset; Ctrl-C likewise stops every item at its next shard and exits 130. One build at
  a time per dataset directory (`dataset/.build.lock`; a second `prepare.py build` or `train.py` auto-prepare on the
  same directory fails fast naming the holder). Everything published so far survives both; rerun to resume.
- `status` prints the same table (rows/tokens present versus needed, fetch increment, manifest state) and exits 0
  iff the dataset is complete.
- `describe` renders the config as Markdown (tokenizer, processing defaults, stage tables with derived token
  budgets, mixture tables with derived example counts, validation sources, sources). The comment block at the top of the YAML
  becomes its "Notes" section. `docs/data_mixture.md` is that output for the crow config; regenerate it after
  editing the config.

## Budgets and incremental builds

The config states token budgets; the planner (`lib/build/planner.py`) turns them into row counts. Per pretrain
source the needed tokens are the **largest** per-stage demand (`stage.tokens × weight`, max over stages, since the
files are shared) plus the source's `validation_tokens`, the needed rows are `tokens ÷ tokens_per_row × 1.2` (safety
margin), where `tokens_per_row` is the measured value — processed (+ validation split) tokens per raw row once
something is processed, the raw manifest's own counts right after a download — and `tokens_per_row_estimate` from
the config before the first download. The download increment is `rows needed − rows present`, clamped at 0 (a remote parquet fetch may leave more rows
than needed on disk, see "Where things are cached"; that is fine, the next plan simply has nothing to fetch). If the processed
tokens still fall short of the budget, the runner (`lib/build/runner.py`) refines the estimate with the measured
value and fetches another increment — at most 5 rounds, then an error. A loader that runs dry marks the source
`exhausted`; the build completes with a warning. Nothing is repeated on disk: the training sampler draws every
source with its stage weight and restarts a source that runs out, so a small source (gsm8k in the full run) is simply
cycled more often — the `epochs` column of the status table (`budget ÷ tokens on disk`) shows how often.

Sources are append-only and deterministic: loaders read the repo's files in sorted order, one file at a time
(`hf_files`, `github_code`), fetch `train[offset:offset+count]` at a pinned `revision` (`hf_split`), stream with
`skip(offset)` (`hf_stream`) or read local files in sorted order (`local`), so a config or stage that needs more
rows appends the next slice and one that needs fewer reads a prefix. `process` handles only the raw shards its
manifest does not cover yet, one at a time, deduplicating them against the `hash` column of every processed (and
validation-split) row already on disk (first occurrence wins, so old rows are kept unchanged and the result equals
a full pass). Mixtures are rebuilt whenever their definition, their budget or any input source's raw shard list
changed; a mixture reads each instruct source only as far as its target (`budget × share × 1.2 ÷ (1 − val_split)`
tokens, so the train split reaches the budget after dedup and the split) and tops a short source up with the kept
tokens per row it measured.

## Where things are cached

Two caches, with different lifetimes:

* **Hub cache** (`~/.cache/huggingface/hub`, or `HF_HOME` / `--cache_dir`): the original repo files fetched by
  `hf_files` / `github_code` (`hf_hub_download`, one file at a time, never twice) and the `datasets` cache of
  `hf_split` sources. Deleting it costs a re-download; nothing else depends on it. `always_range_requests: true`
  (the dataset-level default) reads every Hub file remotely by piece and caches nothing here; set it to `false` to
  let files up to `max_cached_file_mb` (default 32, per source via `load_kwargs.max_cached_file_mb`) land here,
  while larger parquet files are still read remotely row group by row group, projected to the columns the source needs (a top-up seeks
  straight to the row group it needs; fineweb-edu's 2.4 GB files cost a few MB per 1000 rows). A remote parquet
  fetch keeps **every** row of the row groups it read — `rows needed` is a minimum, the raw shards and
  `rows_fetched` advance to the row-group boundary — so a top-up never re-downloads a row group; row groups can be
  large for book-like sources (gutenberg is ~300 MB per 1,000 rows, so the smallest fetch costs one such group).
  Larger `.jsonl[.zst|.gz]` and plain `.json` array files are streamed from the start until exactly enough rows
  were read (`.json` arrays are parsed incrementally with `ijson`; their row count is only recorded once read to
  the end, so a top-up inside a partially consumed file re-streams that one file from its start — a prefix read,
  a few MB for the first few hundred rows).
* **`dataset/sources/<source>/raw/`**: the rows this pipeline kept, in shards, with a manifest — the append-only
  source cache that `process` / instruct mixtures are built from; every row carries its token count (counted at
  download time on the `max_chars` prefix a pretrain document keeps, or the whole instruct example), and a raw
  directory counted with other settings is recounted in place, never fetched again. `dataset/hub_index/` holds the small JSON file
  indexes (file list per glob, rows per file, row-group layout and per-language rows per row group) that let `hf_files` fetch at an offset without opening earlier files;
  it is safe to delete (rebuilt on demand from the repo listing, the file sizes and the cached / remote files).

## Progress display and logs

On a terminal `build` runs inside a live dashboard (`lib/ui/dashboard.py`, `rich`): one bar per running task — the
plan items, `<source>: download` with the rows kept, source rows consumed, current repo file and MB read,
`<source>: process` per row with the running token count, `validation`, `<mixture>: build_instruct_mixture` per
source and per step — above a panel with the latest log lines; tables such as the status summary are printed
unwrapped into the scrollback. The HuggingFace libraries' own bars are silenced for the duration. Every log line
also goes to `dataset/build.log`. When stderr is not a terminal (`nohup`, redirects) or `DATA_PREP_PROGRESS=0` is
set, there is no dashboard and plain timestamped log lines are written instead.

## Training auto-prepares

`training/train.py` loads the run config's `dataset_config`, runs `status`, and — with `auto_prepare: true` (the
default) — runs `build` in-process on the main rank (`prepare_num_workers` items processed and
`prepare_max_parallel_downloads` items downloading at a time, the same dashboard and `build.log`, the same build
lock), then re-verifies. With
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

Pretrain and validation sources need a text column (`text_field`, default `text`); instruct sources need either
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
  my_val:    {kind: validation,  loader: local, path: /data/my_corpus, rows: 2000, seed: 1}
```

`local` reads every `*.parquet` / `*.jsonl` file directly under `path` in sorted order. A validation over the same
directory takes the **last** `rows` rows, so it stays disjoint from a training source reading the first rows as long
as the training source needs fewer than `total − rows` rows (`hf_files` / `hf_split` / `hf_stream` validation sources
take the first rows of their files / split / stream, so point them at files no training source reads — or skip the
separate source altogether and hold a split out of the training source with `validation_tokens`, as the crow config
does with fineweb-edu).

## Processing toggles

`process` (`lib/stages/pretrain.py`) runs per pretrain source, raw shard by raw shard, in this order: length filter
(`min_chars` drops, `max_chars` truncates) → quality filter → decontamination → exact dedup → token count (reused
from the raw shard) → fuzzy dedup. Each is configured in the `processing` block (dataset-level default, per-source
override):

- `dedup.mode: exact` (default) hashes the normalized text (lowercased, whitespace collapsed; `normalize: false` for
  verbatim) in one streaming pass, first occurrence wins; the 64-bit key is stored as the `hash` column. `minhash`
  runs that exact pass first and then MinHash/LSH near-duplicate removal (`threshold`, `num_perm`, `ngram`;
  `lib/stages/fuzzy_dedup.py`, signatures computed with `--num_workers` processes into a compact array, needs the
  `datasketch` extra; documents shorter than one n-gram pass through). Minhash mode is a single all-or-nothing pass
  over the whole source every time it runs (the LSH index must see every signature) and needs roughly 4 GB of RAM
  per million kept rows at `num_perm: 256` — over 10 GB for fineweb-edu at the crow budget. `none` disables both.
- `quality_filter: true` keeps documents with ≥ 3 sentences, ≤ 30 % ALL-CAPS words, ≥ 25 % alphanumeric characters,
  ≤ 30 % repeated 2-grams and ≤ 20 % repeated 3-grams — prose heuristics, wrong for code.
- `decontamination.enabled: true` drops documents whose 13-grams overlap more than `threshold` with a benchmark test
  set (`benchmarks`: gsm8k_test, math_test, humaneval, mbpp_test, arc_challenge_test, hellaswag_test, mmlu_test,
  winogrande_test — downloaded once into `dataset/benchmarks/`, `lib/stages/benchmarks.py`). Needs `datasets` and
  network access on first use.
- Token counting (`token_count`) happens at download time with the real tokenizer over the `max_chars` prefix of
  a document, capped at `max_seq_length`, **without rewriting the text** (`max_seq_length` is a storage / counting
  cap independent of the training sequence length: a dataset capped at 2048 serves a `block_size` 512 run
  unchanged, training truncates); `estimate` uses chars / 4. Instruct examples are counted whole (no cap), so
  `max_tokens` drops them even when it equals `max_seq_length`.
- Note on budgets: the planner treats `stage.tokens × weight` as tokens drawn from a source, while the training
  sampler draws *rows* by weight and pads each to `block_size`; for sources with short rows the row counts and the
  `epochs` column are therefore upper bounds (see CLAUDE.md, "Known issue").

Instruct mixtures (`lib/stages/instruct.py`) read each source in order until its kept rows hold the target
(`budget × share × 1.2 ÷ (1 − val_split)` tokens; the `tokens` column is counted at download time, so only that
prefix is read), drop examples longer than `max_tokens`, apply input inversions on a seeded sample, the same
normalized exact dedup, remove rows with empty fields, shuffle and split. A mixture whose train split ends up empty
is never reported complete.

## Differences from the thesis run

The thesis data was prepared with exact dedup, fuzzy dedup at Jaccard 0.95, tokenizer counts truncated to 2048
tokens and the quality filter, decontamination and PII masking skipped; the pretrain sources were fetched as fixed
row counts (e.g. 9.0M fineweb-edu documents) and the finetune mixture as 400k examples. The crow config mirrors
that with exact dedup on, tokenizer counts capped at 2048 and everything else off, but: fuzzy dedup is off by
default (available as `dedup: {mode: minhash, threshold: 0.95}`), PII masking no longer exists, download sizes
follow the token budgets (× 1.2) instead of fixed row counts, validation is a held-out split of the fineweb-edu
training source (`validation_tokens`) instead of fineweb-edu's `sample-10BT` (which overlapped the training dump:
about a fifth of that validation set was training data), several HuggingFace ids moved
(`wikimedia/wikipedia`, `openai/gsm8k`, `common-pile/arxiv_papers_filtered`) and every source is pinned to a
revision. `docs/data_mixture.md` lists the resulting budgets.
