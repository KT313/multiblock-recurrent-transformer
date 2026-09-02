# Data preparation

A **dataset config** (`config/datasets/<name>.yaml`) is the single definition of a dataset: its sources, the
tokenizer, the sequence length, the processing options and the stage list (token budget and train/val weights over
sources). `uv run python data_preparation/prepare.py prepare --dataset_config <file>` materialises it under
`dataset/`; a run config (`config/<run>.yaml`) references it via `dataset_config:` and `training/train.py` verifies
the prepared data before training, building whatever is missing by default. The pipeline does two things per
source — **download** rows and **build** a cleaned copy — and nothing else: mixing sources by weight and splitting
a source into training and validation rows happen in the training dataloader. Everything runs from the repo root
with `uv run ...`; the implementation lives in `lib/` (tests beside every module).

## The dataset config

**Config reference:** `data_preparation/dataset_config.py` itself — the `DatasetConfig` docstring lists the
top-level keys, every field carries a `# ...` comment, `__post_init__` holds the validation rules;
`data_preparation/layout.py` beside it maps a config to its directories under `dataset/`. Nothing here duplicates
that file; this is the shape:

```yaml
name: crow-300m-final
tokenizer: {name: llama-32k, kind: hf, hf_id: hf-internal-testing/llama-tokenizer, revision: <sha>}
max_seq_length: 2048        # pretrain rows are truncated to this many tokens WHEN DOWNLOADED, instruct rows longer than
                            # this are dropped; raising it re-downloads raw (after confirmation), lowering it costs nothing
block_size: 2048            # training sequence length; the planner counts sequences with it; the run config must match
token_count: tokenizer      # or: estimate (chars / 4)
validation_fraction: 0.05   # share of a source's rows held out when the source is used for training AND validation
processing:                 # defaults for every source; a pretrain source may carry its own block
  min_chars: 50             # drop shorter documents (pretrain only; the upper bound is max_seq_length at download)
  dedup: {mode: exact, normalize: true, bloom_memory_mb: 1024}   # or {mode: minhash, ...} (not for scale), or none
  quality_filter: false
  decontamination: {enabled: false}
sources:                    # keyed by name; `kind` selects the converter and the training-side formatting
  fineweb_edu: {kind: pretrain, loader: hf_files, hf_id: HuggingFaceFW/fineweb-edu, revision: <sha>,
                load_kwargs: {data_files: "data/CC-MAIN-2013-20/*.parquet"}}
  gsm8k:       {kind: pretrain, loader: hf_split, hf_id: openai/gsm8k, revision: <sha>, load_kwargs: {name: main},
                converter: gsm8k_question_answer}
  heldout:     {kind: pretrain, loader: hf_files, hf_id: some/other-corpus, revision: <sha>,
                load_kwargs: {data_files: "*.parquet"}, rows: 5000}     # only in `val` below -> all rows validation
  flan:        {kind: instruct, loader: hf_files, hf_id: Open-Orca/FLAN, revision: <sha>,
                load_kwargs: {data_files: "flan_zsopt_data/*.parquet"},
                fields: {instruction: inputs, output: targets}, input_inversions: 0.05}
stages:                     # the curriculum; weights per stage are > 0 and sum to 1; keys are plain source names
  - {name: pretrain_phase1, tokens: 3_300_000_000, transition_pct: 0.10,
     train: {fineweb_edu: 0.99, gsm8k: 0.01}, val: {fineweb_edu: 0.5, heldout: 0.5}}
  - {name: finetune, tokens: 150_000_000, train: {flan: 1.0}, val: {flan: 1.0}}
```

Two source kinds: `pretrain` (one text column, `text_field`) and `instruct` (`instruction / input / output`, from a
`fields` mapping or a `converter`). Both go through the same download and build steps; the kind only decides the
row shape, the filters that apply and how training formats a row. Stage keys are source names, in `train` and in
`val` alike; how a source is used decides its split at training time (below). Loading fails on unknown keys
(removed ones — `instruct_mixtures`, `validation_tokens`, `max_chars`, `max_tokens`, `tokens_per_row_estimate` —
get a hint saying what replaced them), weights that are zero or do not sum to 1, a source used by no stage, a
source used only in `val` without `rows`, a training source with `rows`, and so on.

The configs in the tree: `config/datasets/crow_300m_final.yaml` (the thesis run; `docs/data_mixture.md` is
generated from it), `config/datasets/crow_300m_mini.yaml` (the same sources with tiny budgets: a real-source smoke
build of a few MB that exercises every loader; needs `HF_TOKEN` for `mini-peS2o`; `tools/capped_download.sh 500 uv
run python data_preparation/prepare.py prepare --dataset_config config/datasets/crow_300m_mini.yaml` runs it under a
hard download cap) and `config/datasets/tiny.yaml` (synthetic, builds in seconds, used by the tests and
`config/tiny.yaml`).

## On-disk layout: two trees

```
dataset/
├── sources/<source>/raw/     MANIFEST.json + data-*.parquet   rows as downloaded (converter applied, pretrain text
│                                                              truncated to max_seq_length tokens, `tokens` column);
│                                                              append-only; the ONLY tree the download step writes
├── processed/<source>/       MANIFEST.json + data-*.parquet   rows after cleaning; derived from raw, cheap to rebuild;
│                                                              the ONLY tree the build step writes  <- training reads this
├── tokenizers/<name>/        MANIFEST.json + tokenizer files
├── hub_index/<repo>@<rev>/   file lists, row counts and row-group layout of `hf_files` / `github_code` repos (safe to delete)
├── benchmarks/               cached benchmark test sets (decontamination only)
├── build.log                 every log line of every prepare run
└── .build.lock               one build per directory
```

Both trees are shared by every dataset config (stages 1 and 2 of the thesis config draw from the same
`processed/fineweb_edu`, only with different weights). Every folder carries a `MANIFEST.json`
(`lib/storage/manifest.py`): the hash of the settings that produced it, rows and tokens per shard, the loader
offset and the rejected-row totals after each raw shard, how tokens were counted and — for raw — `truncated_at_tokens`,
the cap the rows were cut or dropped at. One object owns a raw folder's bookkeeping (`lib/storage/raw_folder.py`:
`RawFolder` — the cap, the loader offset, the `skipped_malformed` / `dropped_too_long` counters, the exhaustion flag
and the truncation to a good prefix); the per-source download, the `github_code` group pass and the repair step all
go through it, so a repair followed by a resume restores every counter instead of only the offset.

- `raw/` is keyed on `DatasetConfig.raw_hash`: the loader identity (repo, revision, files, split, converter, fields,
  filter, seed, ...) plus `token_count` and the tokenizer. Processing options, `max_seq_length`, budgets, weights,
  `rows`, `check_limit`, `validation_fraction`, `input_inversions` and `shuffle` are *not* part of it.
- `processed/` is keyed on `processed_hash`: the raw hash, `max_seq_length`, the processing block reduced to the
  fields of the active dedup mode, `input_inversions` and the resolved `shuffle`. A change rebuilds `processed/`
  from the raw shards; nothing is downloaded.

Which hash a setting belongs to is declared once, on the field itself: every field of `dataset_config.py` carries a
`field(metadata={"hash": "raw" | "processed" | "config" | "none"})` annotation (a callable for the two conditional
cases — `seed` is raw identity only for `loader: synthetic`, and a dedup field only counts for the modes that use
it). `hash_payload` walks those annotations and the three hash methods assemble their payload from it; a new field
without an annotation makes hashing raise. Only non-default values enter, so adding or removing a field with a
default never invalidates data on disk.

Shards are published **one at a time** (written to a `.tmp` file, renamed, recorded in the manifest — raw shards
with the loader offset **and** the rejected-row totals as of their last row), so a network error, a crash or Ctrl-C
keeps everything fetched so far and the next run resumes behind the last complete shard without counting a skipped
or dropped source row twice. All-at-once builds (shuffled sources, minhash) write into
`processed/<source>.tmp` and rename it into place.

## Commands

```bash
uv run python data_preparation/prepare.py prepare  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
        [--sources NAME ...] [--steps tokenizer download build] [--yes] [--dry_run]
        [--num_workers N] [--pass_workers N] [--max_parallel_downloads N] [--hf_token T] [--cache_dir DIR]
uv run python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml > docs/data_mixture.md
uv run python data_preparation/prepare.py tiny     # = prepare --dataset_config config/datasets/tiny.yaml
```

- `prepare` materialises the config (missing parts only; a second run is a no-op). `--sources` / `--steps` restrict
  it, `--dry_run` prints what the repair step and the first round would do and writes nothing (not even the lock).
  Exit codes: 0 ok, 1 a failed source (logged with its traceback; the other jobs stop at their next shard — a
  failed source is a failed build, never a silently smaller dataset), 2 an unconfirmed raw deletion (below), 130
  Ctrl-C (every running step stops at its next shard, everything published is kept; rerun to resume). One build at a
  time per dataset directory (`dataset/.build.lock`; a second `prepare` or a `train.py` auto-prepare on the same
  directory fails fast naming the holder's pid and host).
- `status` is read-only: what the repair step *would* do plus the status table (rows needed / raw / processed /
  epochs / state / reason per source and the tokenizer); exit 0 iff the dataset is complete. A source the repair
  step would touch counts as incomplete.
- `describe` renders the config as Markdown: tokenizer and token counting, processing defaults, one table per stage
  (weights, token budgets, sequences, an estimated row count), the validation split per source and the source
  registry. The comment block at the top of the YAML becomes its "Notes" section. `docs/data_mixture.md` is that
  output for the crow config; regenerate it after editing the config.

## What `prepare` does

`lib/build/runner.py:prepare` is the whole pipeline, readable top to bottom; trimmed to its shape:

```python
def prepare(config_path, dataset_dir, *, num_workers, pass_workers, max_parallel_downloads, assume_yes, dry_run, steps, sources, ...):
    config = load_dataset_config(config_path)
    layout = DatasetLayout(Path(dataset_dir))
    with build_lock(layout.root):
        prepare_tokenizer(config, layout)                                    # tokenizers/<name>/ (downloads count with it)
        repair_report = repair_broken_and_stale_folders(config, layout, assume_yes=assume_yes, dry_run=dry_run, confirm=confirm)
        for round_number in range(1, MAX_ROUNDS + 1):                        # MAX_ROUNDS = 5
            download_plan = plan_downloads(config, layout, sources=selected)  # rows still missing per source
            download_and_build_missing(download_plan, config, layout, ...)   # downloads (sources/<s>/raw) and builds (processed/<s>)
                                                                             # side by side; a source is built as soon as its download finished
            if every_source_satisfies_its_budget(config, layout, sources=selected):
                break
            if not another_round_can_fetch_more(config, layout, active_steps, selected):
                break                                                        # still short, nothing left to download: reported
        report = summarize_dataset_state(config, layout)                     # the status table
    return report
```

`download_and_build_missing` is the only place with thread-pool code: a pool of `--max_parallel_downloads` download
jobs (the `github_code` sources of one repo form one job and are read in a single pass over the repo's files) and a
pool of `--num_workers` build jobs (threads) run side by side under one stop flag; each build additionally holds a
spawn process pool of `--pass_workers` for its optional cleaning passes (decontamination / minhash — off in the
shipped configs), so those toggles cost up to `num_workers × pass_workers` worker processes (2 × 4 = 8 with the
defaults). Sources with nothing to download are built right away, every other source the
moment its download job finished (the members of a `github_code` group after the group pass), so a source is never
built while its own download runs; a failure or Ctrl-C stops both pools at their next shard. Because the two pools
overlap, peak memory is the downloads *plus* `--num_workers` builds (each holding a `dedup.bloom_memory_mb` filter),
no longer the larger of the two.

A round is normally enough. A second one happens when a loader returned fewer rows than asked without being
exhausted, or when the length filter and the dedup dropped more than the 20 % safety margin covers — and it really
tops the source up: one `SourceLedger` per source (`lib/build/planner.py`) answers both "what is still to download?"
and "is this source done?" from one read of the config and the manifests, so the plan sees a folder whose *raw* rows
suffice but whose *processed* rows do not, and asks for `shortfall ÷ observed yield × 1.2` more raw rows. The loop
stops when every source serves its budget, when nothing more can be fetched, or after `MAX_ROUNDS` (5) — a source
still short then is reported, not looped on forever.

### Download (`lib/stages/download.py`)

Appends raw shards to `sources/<source>/raw/` until `rows_needed` rows are on disk. Loaders are deterministic and
read at an offset (below), so a config that needs more rows appends the next slice and one that needs fewer reads
a prefix. Pretrain rows keep every row, with `text_field` **cut at the token boundary `max_seq_length`**
(`lib/stages/truncation.py`: the stored text is a prefix of the document and `tokens` is the true count of the
stored text, so storage is bounded and no count is ever wrong; `token_count: estimate` cuts at `4 × max_seq_length`
characters). Instruct rows run through the converter and filter at download time and are stored as
`{instruction, input, output, tokens}`; a row over `max_seq_length` tokens is **dropped**, never cut (an answer
missing its end would be worse than a missing row; `dropped_too_long` in the manifest), a malformed one is skipped
(`skipped_malformed`); `check_limit` bounds the source rows inspected. A loader that runs dry marks the source
`exhausted`; the source completes with a warning and the training sampler cycles what is there.

The download **never deletes** a raw folder. A folder whose manifest is *stale* (identity or tokenizer changed) or
*outdated* (stored with a smaller `max_seq_length` than the config asks for now) is an error at this point; only the
repair step removes it, and only after confirmation.

### Build (`lib/stages/build.py`)

Turns the raw shards of a source into `processed/<source>/`, in this order — **pretrain**: length filter
(`min_chars`; the upper bound is the truncation at download) → quality filter → decontamination → exact dedup;
**instruct**: input inversions (a seeded per-row decision keyed by `(seed, global row index)`, so it is independent
of shard boundaries and of a resume) → drop rows with an empty instruction or output → exact dedup over
`instruction\ninput\noutput`. Every processed row carries `tokens` (the raw count, clamped to the current
`max_seq_length`) and `hash` (the 64-bit exact-dedup key). Two write modes:

- **per raw shard, resumable** (pretrain sources): the survivors of one raw shard are published before the next raw
  shard is read and the manifest records the raw shard as covered, so a stop loses at most one raw shard of work
  and the next build continues behind the last covered one; a top-up only builds the new raw shards.
- **all at once** (`shuffle: true` — the default for instruct sources — and minhash mode): every raw shard is read,
  the survivors are shuffled with `random.Random(seed)`, written into `processed/<source>.tmp` and renamed into
  place; a top-up rebuilds the folder whole. Why shuffle at all: the training loader reads a source's shards **in
  order** and only mixes *between* sources; instruct repositories are sorted by task, so without a shuffle the model
  would see one task for thousands of steps and the "first k rows" validation split (below) would be a single task.
  Instruct sources are small (the eight of the thesis run hold about 150 M tokens together), so rebuilding them
  whole is cheap. Pretrain sources are not shuffled (`shuffle: false` unless set).

**Exact dedup** (`lib/stages/exact_dedup.py`) hashes the normalized text (lowercased, whitespace collapsed;
`normalize: false` for verbatim) and keeps the first occurrence. The "seen" set is a Bloom filter (`rbloom`) under a
fixed memory budget, `dedup.bloom_memory_mb` (default 1024 MB, per source): fixed size, O(1) per document, and its
only error is a **false positive — a unique document dropped as a duplicate — never a kept duplicate**. Sized for a
0.1 % false-positive rate at its nominal capacity (~600 M documents at 1 GiB), the rate is negligible below that
(≈ 3e-7 at 200 M rows, unmeasurably small at the crow budgets; the build logs the expected rate once per source:
`dedup filter: 1024 MB, ~2.6 M rows -> FPR ≈ ...`). The filter is not persisted: every build refills it from the
`hash` column of the processed shards
already on disk, which is exactly the set a full pass would have accumulated, so an incremental build keeps the
same rows as a full one. Changing `bloom_memory_mb` does not invalidate processed folders (it is a resource knob).

**Minhash** (`dedup: {mode: minhash, threshold: 0.95, num_perm: 256, ngram: 5}`, `lib/stages/fuzzy_dedup.py`,
`datasketch` extra) runs the exact pass first and then MinHash/LSH near-duplicate removal over the whole source at
once (the LSH index needs every signature; documents shorter than one n-gram pass through). It is **not meant for
large sources**: an all-or-nothing pass every time the source changes and roughly 4 GB of RAM per million kept rows
at `num_perm: 256` — over 10 GB for fineweb-edu at the crow budget. The thesis run had it on; the crow config has it
off. Fuzzy dedup at scale is what `datatrove` or `rensa` are for; wiring one of them in is not planned. It applies
to **pretrain sources only**: a config that puts an instruct source under `mode: minhash` is rejected when it is
loaded (it would silently get exact dedup), so ask for minhash in the `processing` block of each pretrain source
rather than in the dataset-level one.

### Repair and the confirmation rule (`lib/build/repair.py`)

Before anything is downloaded or built, one pass over every source folder decides what has to go, and one
report says what it did:

- **raw** (downloaded, expensive): a *stale* folder (identity or tokenizer changed) or an *outdated* one
  (`max_seq_length` raised above `truncated_at_tokens`) is deleted and downloaded again — **only after the user
  confirmed**. A folder with a *broken* shard (missing, unreadable, wrong row count) is truncated to its good prefix
  and the next download resumes there (when no prefix can be kept, it is queued for the same confirmed deletion).
  Shards without a manifest are an error (nothing says where those rows came from).
- **processed** (derived, cheap): deleted without confirmation when stale, broken, without a manifest, built from
  raw shards that no longer exist, or when its raw folder is being deleted; a leftover `.tmp` folder goes too.

Nothing is touched until every folder was inspected; the queued raw deletions are confirmed **once**, with one
list ("The following raw folders will be deleted and downloaded again: fineweb_edu: outdated: max_seq_length 2048
-> 4096 ... Continue? [y/N]"). `--yes` answers it; without a terminal and without `--yes` the command prints the
list and exits 2 with nothing changed. `train.py`'s auto-prepare never prompts and **never deletes raw**: it fails
with the same list and the `prepare.py prepare --yes` command. Lowering `max_seq_length` never touches raw (rows are
at most `truncated_at_tokens` long; the build clamps the stored counts, training truncates at `block_size` anyway).

### Sequences, not tokens (`lib/build/planner.py`)

The trainer draws **rows** from one continuous reader per source, one draw per sample, weighted by the stage
schedule (the stage's constant weight, linearly interpolated across a transition window), and pads or truncates
every row to `block_size` (no packing). The run therefore consumes the integral of a source's weight schedule over
the stage token budgets, ÷ `block_size`. That is the planner's unit, the **sequence budget**: stages sharing a
source ADD UP (the reader continues across stage boundaries instead of re-reading), each stage contributing
`(tokens − transition tokens) × weight` plus the trapezoid `transition tokens × (weight + next stage's weight) / 2`
for the window at its end:

```
rows_needed(source)     = ceil(sequence_budget × 1.2 ÷ (1 − validation_fraction_of(source)))   # source used in train
                        = source.rows                                                          # source used only in val
rows_sufficient(source) = rows_needed ÷ 1.2                                                    # processed rows that serve it
epochs (status table)   = sequence_budget ÷ training rows after the split
```

The `× 1.2` covers what the length filter and the dedup drop, the division keeps the *training* part at the
sequence budget after the resolver holds `validation_fraction` out. There is no tokens-per-row estimate anywhere in
this arithmetic (`describe_tokens_per_row` on a source feeds only the row column of `describe`).

Whether a source is **satisfied** is one `SourceLedger.satisfaction()` case, and the plan, the round loop and the
status table all read it:

| case | when | satisfied |
|---|---|---|
| `OK` | processed rows ≥ `rows_sufficient` | yes |
| `EXHAUSTED_SMALL` | the loader ran dry with fewer, but some, rows | yes, with a warning (the sampler cycles them) |
| `EXHAUSTED_EMPTY` | the loader ran dry and **not one** row survived the build | **no** — a wrong `fields` / `converter` / `filter` / `language`, and a failed source is a failed build |
| `SHORT_BUT_FETCHABLE` | too few processed rows, the loader has more | no — the next round tops it up |
| `NOT_BUILT` | `processed/` missing, stale or behind the raw shards | no — build it |
| `RAW_MISSING` / `RAW_BROKEN` | nothing downloaded / stale or outdated raw | no — download, or let the repair step delete it |

The consequence to keep in mind: **the weights mix rows, not tokens.** The realised token share of a source in a
stage is proportional to `weight × mean_tokens_per_row` (rows capped at `block_size`), so a stage's token mix is
`weight × mean_tokens_per_row ÷ block_size`-weighted. Example: `block_size` 2048, one 1 B-token stage,
`{fineweb: 0.5 (~1000 tokens/row), gsm8k: 0.5 (~300 tokens/row, 7.5 k rows)}` — the stage draws 488 k sequences,
244 k rows of each source, a realised token mix of about 77 / 23, and gsm8k is cycled ~33 times (`epochs` 33 in the
status table). The planner downloads 293 k fineweb rows for it (244 k × 1.2), not the 600 k the old token-based
planner asked for. Set weights with the row lengths of the sources in mind; `docs/data_mixture.md` prints
tokens, sequences and an estimated row count per source and stage.

### The validation split happens at training time

The pipeline writes **one** folder per source and never decides what is validation. `training/data/dataset_resolver.py`
decides, once per source and run, from how the stages use it:

| the source appears in | training rows | validation rows |
|---|---|---|
| `train` only | all | none |
| `val` only (`rows` says how many to download) | none | all |
| both | the rest | the **first** `ceil(validation_fraction_of(source) × rows)` processed rows |

`rows` is the row count of `processed/<source>` on disk (parquet footers, cross-checked against the manifest); the
per-source `validation_fraction` overrides the dataset default (0.05). Because a source's processed folder was
deduplicated as one set, the two parts never share a document, and the same source gets the same split in every
stage of the run. The chosen `validation_rows` per source travel with every checkpoint next to the dataset config
hash and are verified on resume. A validation-only source reading the same Hub repo and file prefix as a training
source draws a warning (prefer listing the training source in `val` too). The resolver also checks every stage key
directly on disk — folder present, at least one shard, a non-empty row range for its part — independent of the
manifests, and that the run config's `block_size` equals the dataset config's.

## Where things are cached

Two caches, with different lifetimes:

* **Hub cache** (`~/.cache/huggingface/hub`, or `HF_HOME` / `--cache_dir`): the original repo files fetched by
  `hf_files` / `github_code` (`hf_hub_download`, one file at a time, never twice) and the `datasets` cache of
  `hf_split` sources. Deleting it costs a re-download; nothing else depends on it. `always_range_requests: true`
  (the dataset-level default) reads every Hub file remotely by piece and caches nothing here; set it to `false` to
  let files up to `max_cached_file_mb` (default 32, per source via `load_kwargs.max_cached_file_mb`) land here,
  while larger parquet files are still read remotely row group by row group, projected to the columns the source
  needs (a top-up seeks straight to the row group it needs; fineweb-edu's 2.4 GB files cost a few MB per 1000 rows).
  A remote parquet fetch keeps **every** row of the row groups it read — `rows_needed` is a minimum, the raw shards
  and the loader offset advance to the row-group boundary — so a top-up never re-downloads a row group; row groups
  can be large for book-like sources (gutenberg is ~300 MB per 1,000 rows). Larger `.jsonl[.zst|.gz]` and plain
  `.json` array files are streamed from the start until enough rows were read (`.json` arrays incrementally with
  `ijson`; a top-up inside a partially consumed file re-streams that one file's prefix).
* **`dataset/sources/<source>/raw/`**: the rows this pipeline kept — the append-only cache that `processed/` is
  built from, deleted only by the confirmed repair above. `dataset/hub_index/` holds the small JSON file indexes
  (file list per glob, rows per file, row-group layout, per-language rows per row group) that let `hf_files` fetch
  at an offset without opening earlier files; it is safe to delete (rebuilt on demand).

## Progress display and logs

On a terminal `prepare` runs inside a live dashboard (`lib/ui/dashboard.py`, `rich`): a header (config, round,
step, elapsed), a **downloads** panel (one row per running download — rows kept / wanted, rate, elapsed, source rows
consumed, current repo file, MB read — and a summary line: jobs done, rows of the round, MB, elapsed), a **builds**
panel (one row per running build — raw rows processed, current raw shard — plus its summary line), the **log**
panel with the latest lines, and a footer naming `dataset/build.log` (every log line goes there). Finished rows
disappear into the summary; at most eight rows are shown per panel ("… and k more"). Nothing else reaches the
terminal while the display is up: every `logging` record (the HuggingFace libraries' included), `warnings` and
stray prints land in the log panel, the libraries' own bars are silenced. Warnings and the tables (plan, repair,
status) are *kept* and printed once, unwrapped, after the display closed — the scrollback of a run is those lines
and the final table, no frame. Ctrl-C and SIGTERM (`tools/capped_download.sh`) leave the same way. When stderr is
not a terminal (`nohup`, redirects) or `DATA_PREP_PROGRESS=0` is set, there is no dashboard and plain timestamped
log lines are written instead.

## Training auto-prepares

`training/train.py` loads the run config's `dataset_config`, runs `status`, and — with `auto_prepare: true` (the
default) — runs `prepare` in-process on the main rank (`prepare_num_workers` / `prepare_pass_workers` / `prepare_max_parallel_downloads`,
the same dashboard, `build.log` and lock; `assume_yes=False`, so it never deletes raw), then re-verifies. With
`auto_prepare: false` a missing dataset is a hard error quoting the `prepare.py prepare` command. Gated sources
(`nampdn-ai/mini-peS2o` in the crow config) need `HF_TOKEN` in the environment (or `--hf_token` for `prepare`).
The dataset config's hash and the validation split are written into every checkpoint; resuming with a changed
dataset config is an error unless the run config sets `allow_dataset_change: true`.

## Adding a source

One YAML entry under `sources:` plus its weight in the stages that use it. Pin the `revision` (`git ls-remote` on
the Hub repo, or the commit shown on the dataset page) so row order is stable across increments; give
`describe_tokens_per_row` a ballpark value if you care about the row column of `describe`.

Loaders (`lib/sources/loaders.py`, `loader:`; all are `(source, offset, count, shared_parameters) -> Iterator[row]`, the
last a `SharedLoaderParameters` — token, index directory, file callback, download counters, column projection):

| Loader | Use for | Notes |
|---|---|---|
| `hf_files` | **default for Hub repos with many files** | `load_kwargs: {data_files: <glob>, max_cached_file_mb: 32}` (`data_files` required, relative to the repo root); files sorted by path; files up to `max_cached_file_mb` are downloaded one at a time into the Hub cache on demand and read locally, larger `.parquet` files are read remotely by row group (every row of a fetched row group is kept, so a top-up never re-downloads one) and larger `.jsonl`, `.jsonl.zst`, `.jsonl.gz`/`.json.gz` files and plain `.json` arrays (incrementally via `ijson`) are streamed from the start; a file index under `dataset/hub_index/` lets a top-up skip files already consumed |
| `hf_split` | split-name based repos (e.g. `gsm8k` with `name: main`) | `train[a:b]` slicing; `datasets` downloads and caches the file once and slices locally; `load_kwargs` go to `load_dataset` (`name`, `data_files`, ...) |
| `hf_stream` | fallback | `datasets` streaming with `skip(offset)`; **caches nothing** — every fetch re-streams from the start, so avoid it for anything large |
| `github_code` | `codeparrot/github-code-clean` | `hf_files` over `data/*.parquet` keeping rows of `language:` (`text_field: code`); all language sources of a repo download in one pass over the shared files and the index stores per-language row counts (per row group for partially read parquet files) |
| `local` | your own data | `path:` directory of `*.parquet` / `*.jsonl` files, read in sorted file order |
| `synthetic` | tests / smoke runs | random-word rows from `seed:` |

Pretrain sources need a text column (`text_field`, default `text`); instruct sources need either
`fields: {instruction: <col>, input: <col>, output: <col>}` (input optional) or a `converter`. Converters and filters
(`lib/sources/converters.py`) turn one source row into the row the pipeline expects: `gsm8k_question_answer`
(question + answer → `text`), `sharegpt_conversations` (`from`/`value` turns → instruction/output),
`first_two_turns` (a `conversations` list), `instruction_input_output` (already standardized); filter
`sharegpt_quality` (keeps human→gpt openings with 50–2000 characters per side and no code block in the answer;
`check_limit` bounds the rows inspected). Write a new converter when the source's schema is not a simple column
mapping: add a function `(row) -> row` to `converters.py`, register it in `CONVERTERS` (or `FILTERS`), test it on a
hand-written row in `test_sources.py`, and reference it by name in the YAML. For an instruct source a converter
raising `ValueError` skips the row (counted in the manifest as `skipped_malformed`); for a pretrain source it fails the
download (a pretrain converter maps whole columns, a failing row means a wrong mapping).

### A local dataset

```yaml
sources:
  my_corpus: {kind: pretrain, loader: local, path: /data/my_corpus, text_field: text}
stages:
  - {name: pretrain, tokens: 100_000_000, train: {my_corpus: 1.0}, val: {my_corpus: 1.0}}   # first 5 % of rows = validation
```

`local` reads every `*.parquet` / `*.jsonl` file directly under `path` in sorted order. Listing the source in `val`
too gives a held-out split without a second source; a separate validation-only source (`rows: N`, `val` only)
should point at files no training source reads.

## Processing toggles

Configured in the `processing` block (dataset-level default, per-pretrain-source override); see "Build" above for
the order they run in.

- `dedup.mode`: `exact` (default; Bloom filter, `normalize`, `bloom_memory_mb`), `minhash` (exact first, then
  MinHash/LSH; `threshold`, `num_perm`, `ngram`; not for scale, pretrain sources only — an instruct source under
  `minhash` is a config error) or `none`.
- `quality_filter: true` keeps documents with ≥ 3 sentences, ≤ 30 % ALL-CAPS words, ≥ 25 % alphanumeric characters,
  ≤ 30 % repeated 2-grams and ≤ 20 % repeated 3-grams — prose heuristics, wrong for code.
- `decontamination.enabled: true` drops documents whose 13-grams overlap more than `threshold` with a benchmark test
  set (`benchmarks`: gsm8k_test, math_test, humaneval, mbpp_test, arc_challenge_test, hellaswag_test, mmlu_test,
  winogrande_test — downloaded once into `dataset/benchmarks/`, `lib/stages/benchmarks.py`). Needs network access
  on first use.
- `min_chars` drops shorter pretrain documents; there is no upper character bound — the token truncation at
  download is the upper bound.
- Token counting (`token_count`) happens at download time: `tokenizer` counts with the config's tokenizer (no
  special tokens), `estimate` uses chars / 4. Instruct rows are counted whole (instruction + input + output).

## Differences from the thesis run

The thesis data was prepared with exact dedup, fuzzy dedup at Jaccard 0.95, tokenizer counts truncated to 2048
tokens and the quality filter, decontamination and PII masking skipped; the pretrain sources were fetched as fixed
row counts (e.g. 9.0M fineweb-edu documents) and the finetune data as a 400k-example mixture built up front. The
crow config mirrors that with exact dedup on, `max_seq_length` 2048 and everything else off, but: fuzzy dedup is
off by default (available as `dedup: {mode: minhash, threshold: 0.95}`), PII masking no longer exists, texts are
truncated at the token cap when downloaded instead of stored whole, download sizes follow the sequence budgets
(× 1.2) instead of fixed row counts, the finetune stage mixes the eight instruct sources by weight in the
dataloader (no prebuilt mixture, no cross-source dedup of instruct data), validation is the first 5 % of the
fineweb-edu training source (`validation_fraction`) instead of fineweb-edu's `sample-10BT` (which overlapped the
training dump: about a fifth of that validation set was training data), several HuggingFace ids moved
(`wikimedia/wikipedia`, `openai/gsm8k`, `common-pile/arxiv_papers_filtered`) and every source is pinned to a
revision. `docs/data_mixture.md` lists the resulting budgets.
