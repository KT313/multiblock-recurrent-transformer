# Data preparation

A **dataset config** (`config/datasets/<name>.yaml`) is the single definition of a dataset: its sources, the
tokenizer, the sequence length, the processing options and the stage list (token budget and train/val weights over
sources). `uv run python data_preparation/prepare.py prepare --dataset_config <file>` materialises it under
`dataset/`; a run config (`config/<run>.yaml`) references it via `dataset_config:` and `training/train.py` verifies
the prepared data before training, building whatever is missing by default. The pipeline does two things per
source, **download** rows and **build** a cleaned copy, and nothing else: mixing sources by weight and splitting
a source into training and validation rows happen in the training dataloader. Everything runs from the repo root
with `uv run ...`; the implementation lives in `lib/` (tests beside every module).

## The dataset config

**Config reference:** `data_preparation/dataset_config.py` itself: the `DatasetConfig` docstring lists the
top-level keys, every field carries a `# ...` comment, `__post_init__` holds the validation rules;
`data_preparation/layout.py` beside it maps a config to its directories under `dataset/`. Nothing here duplicates
that file; this is the shape:

```yaml
tokenizer: {name: llama-32k, kind: hf, hf_id: hf-internal-testing/llama-tokenizer, revision: <sha>}
training_target_sequence_length: 2048   # the run trains at this length: a row counts min(its tokens, this) towards the
                                        # download budget; at most dataset_max_sequence_length
dataset_max_sequence_length: 2048   # pretrain rows are truncated to this many tokens WHEN DOWNLOADED, instruct rows longer
                                    # than this are dropped; raising it re-downloads raw (after confirmation), lowering it
                                    # costs nothing; a storage cap (no 100k-token documents stored whole), not the training length
token_count: tokenizer      # or: estimate (chars / 4)
validation_fraction: 0.05   # share of a source's rows held out when the source is used for training AND validation
processing:                 # defaults for every source; a pretrain source may carry its own block
  min_chars: 50             # drop shorter documents (pretrain only; the upper bound is dataset_max_sequence_length at download)
  dedup: {mode: exact, normalize: true, bloom_memory_mb: 1024}   # or {mode: minhash, ...} (not for scale), or none
  quality_filter: false
  decontamination: {enabled: false}
sources:                    # keyed by name; `kind` selects the converter and the training-side formatting
  fineweb_edu: {kind: pretrain, loader: hf_files, hf_id: HuggingFaceFW/fineweb-edu, revision: <sha>,
                load_kwargs: {data_files: "data/CC-MAIN-2013-20/*.parquet"}}
  gsm8k:       {kind: pretrain, loader: hf_split, hf_id: openai/gsm8k, revision: <sha>, load_kwargs: {name: main},
                converter: gsm8k_question_answer}
  heldout:     {kind: pretrain, loader: hf_files, hf_id: some/other-corpus, revision: <sha>,
                load_kwargs: {data_files: "*.parquet"}, rows: 5000}     # only in `val` below -> all rows validation (5000 delivered)
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
`val` alike; how a source is used decides its split at training time (below). Loading fails on unknown keys,
weights that are zero or do not sum to 1, a source used by no stage, a source used only in `val` without `rows`, a
training source with `rows`, and so on. The keys of the earlier schema (`instruct_mixtures`, `validation_tokens`,
`max_chars`, `max_tokens`, `tokens_per_row_estimate`, `<source>/validation` stage keys) are simply unknown now:
mixing and the validation split are the training dataloader's job, rows are truncated to `dataset_max_sequence_length` tokens at
download, and the planner counts tokens (`describe_tokens_per_row` is its starting rate, not a description).

The configs in the tree: `config/datasets/crow_300m_final.yaml` (the thesis run; `docs/data_mixture.md` is
generated from it), `config/datasets/crow_300m_mini.yaml` (the same sources with tiny budgets: a real-source smoke
build of a few MB that exercises every loader; needs `HF_TOKEN` for `mini-peS2o`) and `config/datasets/tiny.yaml`
(synthetic, builds in seconds, used by the tests and `config/tiny.yaml`).

## On-disk layout: two trees

```
dataset/
├── sources/<source>/raw/     MANIFEST.json + data-*.parquet   rows as downloaded (converter applied, pretrain text
│                                                              truncated to dataset_max_sequence_length tokens, `tokens` column);
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
(`lib/storage/manifest.py`): the hash of the settings that produced it and the exact dict it was computed from
(`hash_payload`), rows and tokens per shard, the loader offset and the rejected-row totals after each raw shard, how
tokens were counted (`token_count`, `tokenizer`, and for raw the tokenizer's definition hash `tokenizer_hash`) and,
for raw, `truncated_at_tokens`, the cap the rows were cut or dropped at. The stage-specific fields are typed on
`Manifest`: a raw manifest carries `exhausted`, `check_limit_reached`, `skipped_malformed` and `dropped_too_long`, a processed one `input_shards`,
`columns`, `shuffled`, `shuffle_seed` and `stats`. One object owns a raw folder's bookkeeping (`lib/storage/raw_folder.py`:
`RawFolder`: the cap, the loader offset, the `skipped_malformed` / `dropped_too_long` counters, the exhaustion flag
and the truncation to a good prefix); the per-source download, the `github_code` group pass and the repair step all
go through it, so a repair followed by a resume restores every counter instead of only the offset.

- `raw/` is keyed on `DatasetConfig.raw_hash`: exactly the settings that change which rows the loader stores (kind,
  loader, repo, revision, files, language, path, converter, fields, filter; `split` only for `hf_split` / `hf_stream`,
  `text_field` only for pretrain rows, `seed` only for `loader: synthetic`). The tokenizer and `token_count` are
  *not* part of it: they only make the stored token counts and the truncation, so the raw manifest records them
  (`tokenizer`, `tokenizer_hash`, `token_count`) and a later change is offered as a choice (see the repair section)
  instead of costing a re-download. Processing options, `dataset_max_sequence_length`, budgets, weights, `rows`,
  `check_limit`, `validation_fraction`, `input_inversions` and `shuffle` are not part of it either.
- `processed/` is keyed on `processed_hash`: the raw hash, the tokenizer definition, `token_count` and the counting
  rule, `dataset_max_sequence_length`, the processing block as the build applies it (pretrain: reduced to the fields
  of the active dedup mode; instruct: the dedup block alone, the other passes run for pretrain rows only),
  `input_inversions` (instruct) and the resolved `shuffle` and `seed`. A change rebuilds `processed/` from the raw
  shards after confirmation; nothing is downloaded.

Both manifests store `hash_payload`, the exact dict the hash was computed from, so a mismatch is explained field by
field (`processing.dedup.normalize: true -> false`) in the confirmation prompt and the status table
(`DatasetConfig.describe_hash_change`).

Which hash a setting belongs to is declared once, on the field itself: every field of `dataset_config.py` carries a
`field(metadata={"hash": "raw" | "processed" | "tokenizer" | "config" | "none"})` annotation (a callable for the
conditional cases: `seed`, `split` and `text_field` are raw identity only where a loader reads them, and a dedup
field only counts for the modes that use it). `hash_payload` walks those annotations and the hash methods assemble their payload from it; a new field
without an annotation makes hashing raise. Every counted field enters with its resolved value, default or not: a
changed default invalidates the data built under the old one, and adding a field to the schema changes the hashes
once (one rebuild). `config_hash` nests the others: every source's `processed_hash` (which folds in its `raw_hash`)
beside the source's own `config` fields, the tokenizer hash and the dataset-level `config` fields.

Shards are published **one at a time** (written to a `.tmp` file, renamed, recorded in the manifest; raw shards
with the loader offset **and** the rejected-row totals as of their last row), so a network error, a crash or Ctrl-C
keeps everything fetched so far and the next run resumes behind the last complete shard without counting a skipped
or dropped source row twice. All-at-once builds (shuffled sources, minhash) write into
`processed/<source>.tmp` and rename it into place.

## Commands

```bash
uv run python data_preparation/prepare.py prepare  --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
        [--sources NAME ...] [--steps tokenizer download build] [--reopen NAME ...] [--yes] [--dry_run]
        [--num_workers N] [--pass_workers N] [--max_parallel_downloads N] [--hf_token T] [--cache_dir DIR]
uv run python data_preparation/prepare.py download --dataset_config config/datasets/<name>.yaml [same options; --steps tokenizer download]
uv run python data_preparation/prepare.py status   --dataset_config config/datasets/<name>.yaml [--dataset_dir dataset]
uv run python data_preparation/prepare.py describe --dataset_config config/datasets/<name>.yaml > docs/data_mixture.md
uv run python data_preparation/prepare.py tiny     # = prepare --dataset_config config/datasets/tiny.yaml
```

- `prepare` materialises the config (missing parts only; a second run is a no-op). `--sources` / `--steps` restrict
  it, `--reopen` clears the exhausted flag of the named sources first (below), `--dry_run` prints what the repair
  step and the first round would do and writes nothing (not even the lock).
- `download` (`make download`) is `prepare` without the build step: tokenizer and raw shards only, for the long
  download on a machine that builds later. Its verdict is about the raw side (exit 0 with "download complete" iff
  every source has its raw rows, else 1 naming the sources); no processed folder is expected after it, `prepare`
  (`make prepare`) builds them from the raw shards without downloading again.
  Exit codes: 0 ok, 1 a failed source (logged with its traceback; the other jobs stop at their next shard: a
  failed source is a failed build, never a silently smaller dataset), 2 an unconfirmed raw deletion (below), 3
  another data preparation still running, 130 Ctrl-C (every running step stops at its next shard, everything
  published is kept; rerun to resume). One run at a time (`dataset/.build.lock`, `lib/build/lock.py`; training holds
  `<out_dir>/.train.lock` the same way): a second `prepare` or a `train.py` auto-prepare on the same directory exits 3
  right away, naming the running one's start time and pid and how to stop it (`kill -INT <pid>`); the lock is the
  OS's, released when the holder ends, so it never goes stale.
- `status` is read-only: what the repair step *would* do plus the status table (rows needed / tokens per row / raw /
  processed / epochs / state / reason per source and the tokenizer); exit 0 iff the dataset is complete. A source the
  repair step would touch counts as incomplete.
- `describe` renders the config as Markdown: tokenizer and token counting, processing defaults, one table per stage
  (weights, token budgets, the tokens-per-row estimate and the rows it makes of them), the validation split per
  source and the source registry. The comment block at the top of the YAML becomes its "Notes" section. `docs/data_mixture.md` is that
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
        reopen_sources(config, layout, reopened, dry_run=dry_run)            # --reopen: clear the exhausted latch
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
jobs (the `github_code` sources of one repo form one job, every member of the repo included, and are read in a single
pass over the repo's files) and a pool of `--num_workers` build jobs (threads) run side by side under one stop flag; each build additionally holds a
spawn process pool of `--pass_workers` for its optional cleaning passes (decontamination / minhash, off in the
shipped configs), so those toggles cost up to `num_workers × pass_workers` worker processes (2 × 4 = 8 with the
defaults). Sources with nothing to download are built right away, every other source the
moment its download job finished (the members of a `github_code` group after the group pass), so a source is never
built while its own download runs; a failure or Ctrl-C stops both pools at their next shard, and a second Ctrl-C
while they wait for the running jobs ends the process right away (exit 130; a transfer of hundreds of MB is not
waited for, everything published so far is kept). Because the two pools
overlap, peak memory is the downloads *plus* `--num_workers` builds (each holding a `dedup.bloom_memory_mb` filter),
not the larger of the two. Inside a download job, fetching the next row group and tokenizing the previous batches
overlap too (a token worker thread per job; the rows are still written in order), and `prepare.py` turns the
tokenizer's own thread pool on (`TOKENIZERS_PARALLELISM=true`, off by library default because a training run that
prepares data in-process forks DataLoader workers afterwards) with 8 threads (`RAYON_NUM_THREADS`; the pool is one
per process and shared by every download job, and past 8 threads a batch barely gets faster). A single download is
therefore bound by its network fetch; `--max_parallel_downloads` scales from there.

A round is normally enough. A second one happens when the raw shards of the first measured fewer tokens per row than
the estimate the download was sized with (by more than the 20 % safety margin covers), or when the length filter and
the dedup dropped more than that margin, and it really tops the source up: one `SourceLedger` per source
(`lib/build/planner.py`) answers both "what is still to download?" and "is this source done?" from one read of the
config and the manifests, so the plan sees a folder whose *raw* rows suffice but whose *processed* rows do not, and
asks for `shortfall ÷ observed yield × 1.2` more raw rows, and it never re-downloads a source whose processed rows
already serve the budget. A loader that yielded fewer rows than asked does **not** get a second round: it is
exhausted (below). The loop stops when every source serves its budget, when nothing more can be fetched, or after
`MAX_ROUNDS` (5); a source still short then is reported, not looped on forever.

### Download (`lib/stages/download.py`)

Appends raw shards to `sources/<source>/raw/` until `rows_needed` rows are on disk. Loaders are deterministic and
read at an offset (below), so a config that needs more rows appends the next slice and one that needs fewer reads
a prefix. Pretrain rows keep every row as `{text_field, tokens}` (a string, whatever the loader delivered), with
`text_field` **cut at a token boundary** (`lib/stages/truncation.py`: the stored text is a prefix of the document,
so storage is bounded and no count is ever wrong; `token_count: estimate` cuts at 4 characters per token). `tokens`
is the length the trainer sees: the true count of the stored text plus the BOS and EOS the trainer adds around
every row (`NUMBER_OF_SPECIAL_TOKENS`), and never exceeds `dataset_max_sequence_length`. Instruct rows run through the converter and
filter at download time and are stored as `{instruction, input, output, tokens}` with `tokens` counted the same way
over the text the trainer formats from them (`instruct_text`); a row whose `tokens` exceeds `dataset_max_sequence_length` is
**dropped**, never cut (an answer missing its end would be worse than a missing row; `dropped_too_long` in the
manifest), a malformed one (converter error, no instruction / output) is skipped (`skipped_malformed`);
`check_limit` bounds the source rows inspected. A loader that yields fewer rows than asked
marks the source `exhausted`, whatever the reason (the source really ended, a partial listing, a loader bug): the
source completes with a warning (after the status table) and the training sampler cycles what is there. The flag is
a latch: a later run does not read on by itself, a bigger budget fetches nothing. The one exception is a source
stopped by its own `check_limit`: the manifest records the limit, and a grown (or removed) limit reopens the source.
Everything else is `prepare --reopen NAME`: it clears the flag and the download resumes at the recorded offset.

The download **never deletes** a raw folder. A folder whose manifest is *stale* (identity or tokenizer changed) or
*outdated* (stored with a smaller `dataset_max_sequence_length` than the config asks for now) is an error at this point; only the
repair step removes it, and only after confirmation.

**The `github_code` pass keeps every row it decodes.** The repo's parquet files mix all ~30 languages, and a row group
(the unit a range request fetches) is read whole for whichever language still needs rows, so the pass decodes far
more rows than any one language asks for (a rare language decides how deep the pass reads). Nothing decoded is
thrown away: a member that has its rows (from the start, or once it reached its target) stays in the pass
*passively* and stores every further row of its language the pass reads for the others, and a language no source
names gets a raw folder of its own, `sources/<repo tail>_<language slug>/raw` (`codeparrot/github-code-clean` +
`C#` -> `github_code_clean_csharp`; `lib/sources/loaders.py: github_code_extra_name`), downloaded as the group's
first member with the language replaced, truncated and counted like every other pretrain row. Such a folder is
adopted by a config entry of that name with the same repo, revision, `data_files` and `text_field` and the
language (its raw hash is the folder's); until then nothing builds or trains on it and the repair step leaves it
alone. A passively fed folder is only as long as the pass was: its rows end where the last active language was
satisfied, and an entry added later downloads on from that offset. Passive rows are stored only while their offset
is *aligned* with the pass (carried through every earlier row group by a recorded count or by reading it); a pass
that seeks past a passive folder's position (a top-up resuming deeper in the repo) stores nothing for it rather
than a gap. For that the shared file index counts *every* language of every fully decoded row group
(`full_counts` in `dataset/hub_index/`), and a stop or a failure publishes each folder's buffered rows as a short
final shard first, so every folder's offset is the frontier the pass reached and the next pass resumes them all
aligned. The cost is disk (the whole decoded volume, zstd, instead of the wanted languages), the tokenizer running
over every row (the 8-thread pool keeps up with a row group's fetch), and one shard buffer per language (passive
shards are a quarter of `shard_size`). `build.log` lists the rows stored past each target and per extra language.

### Build (`lib/stages/build.py`)

Turns the raw shards of a source into `processed/<source>/`, in this order. **Pretrain**: length filter
(`min_chars`; the upper bound is the truncation at download) → quality filter → decontamination → exact dedup;
**instruct**: input inversions (a seeded per-row decision keyed by `(seed, global row index)`, so it is independent
of shard boundaries and of a resume) → drop rows with an empty instruction or output → exact dedup over the
trainer's text (`instruct_text`). Every processed row carries `tokens` (the raw count, clamped to the current
`dataset_max_sequence_length`) and `hash` (the 64-bit exact-dedup key). Two write modes:

- **per raw shard, resumable** (pretrain sources): the survivors of one raw shard are published before the next raw
  shard is read and the manifest records the raw shard as covered, so a stop loses at most one raw shard of work
  and the next build continues behind the last covered one; a top-up only builds the new raw shards. **The build
  is capped at the budget**: it stops after the raw shard that brings the processed rows to `rows_sufficient`
  (`lib/build/planner.py`; `build_source(rows_target=)`), so raw rows past the budget (a `github_code` member fed
  on after its target, above) cost raw disk only. A processed folder behind raw that serves the budget is healthy,
  satisfied (`ok, N raw shard(s) past the budget unbuilt` in the status table) and not a pending build; a larger
  budget, or a lower measured tokens-per-row rate, builds the next shards.
- **all at once** (`shuffle: true`, the default for instruct sources, and minhash mode): every raw shard is read,
  the survivors are shuffled with `random.Random(seed)`, written into `processed/<source>.tmp` and renamed into
  place; a top-up rebuilds the folder whole. Why shuffle at all: the training loader reads a source's shards **in
  order** and only mixes *between* sources; instruct repositories are sorted by task, so without a shuffle the model
  would see one task for thousands of steps and the "first k rows" validation split (below) would be a single task.
  Instruct sources are small (the eight of the thesis run hold about 150 M tokens together), so rebuilding them
  whole is cheap. Pretrain sources are not shuffled (`shuffle: false` unless set).

**Exact dedup** (`lib/stages/exact_dedup.py`) hashes the normalized text (lowercased, whitespace collapsed;
`normalize: false` for verbatim) and keeps the first occurrence. The "seen" set is a Bloom filter (`rbloom`) under a
fixed memory budget, `dedup.bloom_memory_mb` (default 1024 MB, per source): fixed size, O(1) per document, and its
only error is a **false positive (a unique document dropped as a duplicate), never a kept duplicate**. Sized for a
0.1 % false-positive rate at its nominal capacity (597 M documents at 1 GiB, nine probes per key), the rate is
negligible below that (≈ 3e-7 at 200 M rows, unmeasurably small at the crow budgets). The build logs the expected
rate once per source, counting the source's raw rows as the upper bound of what it inserts:

```
dedup filter: 1024 MB, 2,600,000 rows on disk (upper bound of insertions) -> FPR ≈ 0.00 % (0 % of the nominal 597 M rows)
```

Past the nominal capacity the build warns and names the `bloom_memory_mb` that brings it back under (raising it
invalidates nothing, see below); past **twice** it, where the rate is already ≈ 5 %, the build refuses to run
rather than drop that many unique documents. The processed manifest keeps both numbers under `stats.dedup`:
`rows_on_disk` and `expected_false_positive_rate` from the start of the build, `items_in_filter` and
`measured_false_positive_rate` (rbloom's estimate from the set bits) as of the last shard it wrote.

The filter is not persisted: every build refills it from the `hash` column of the processed shards
already on disk, which is exactly the set a full pass would have accumulated, so an incremental build keeps the
same rows as a full one. Changing `bloom_memory_mb` does not invalidate processed folders (it is a resource knob).

**Minhash** (`dedup: {mode: minhash, threshold: 0.95, num_perm: 256, ngram: 5}`, `lib/stages/fuzzy_dedup.py`,
`datasketch` extra) runs the exact pass first and then MinHash/LSH near-duplicate removal over the whole source at
once (the LSH index needs every signature; documents shorter than one n-gram pass through). It is **not meant for
large sources**: an all-or-nothing pass every time the source changes and roughly 4 GB of RAM per million kept rows
at `num_perm: 256`, over 10 GB for fineweb-edu at the crow budget. The thesis run had it on; the crow config has it
off. Fuzzy dedup at scale is what `datatrove` or `rensa` are for; wiring one of them in is not planned. It applies
to **pretrain sources only**: a config that puts an instruct source under `mode: minhash` is rejected when it is
loaded (it would silently get exact dedup), so ask for minhash in the `processing` block of each pretrain source
rather than in the dataset-level one.

### Repair and the confirmation rule (`lib/build/repair.py`)

Before anything is downloaded or built, one pass over every source folder decides what has to go, and one
report (`RepairReport`) lists every action with whether it was carried out (`performed`; False for `status` /
`--dry_run` and for the report a refused confirmation carries):

- **raw** (downloaded, expensive): a *stale* folder (the loader identity changed; the prompt lists the fields, e.g.
  `source.revision: "abc" -> "def"`) or an *outdated* one (`dataset_max_sequence_length` raised above
  `truncated_at_tokens`) is deleted and downloaded again, **only after the user confirmed**. A folder whose rows were
  counted with another tokenizer or `token_count` than the config's (*tokenizer_changed*; the raw manifest records
  `tokenizer_hash`) is not re-downloaded: the prompt names the old and the new tokenizer and the rows counted under
  the old one, and on yes the folder is *adopted*: its manifest is re-labelled with the new tokenizer (the switch is
  logged under `tokenizer_changes` in the manifest) and the rows stay, at the cost that their stored token counts
  and the truncation of pretrain texts do not match the new tokenizer; later downloads count with the new one. A
  folder with a *broken* shard (missing, unreadable, wrong row count) is truncated to its good prefix
  and the next download resumes there (when no prefix can be kept, it is queued for the same confirmed deletion).
  Shards without a manifest are an error (nothing says where those rows came from). Raw folders are shared by
  source name across dataset configs and their manifest records the config file they were downloaded under, so a
  deletion that another config's folder would suffer needs `--allow_foreign_raw` on top of the confirmation.
- **processed** (derived, cheap): a *stale* folder (the processed fingerprint changed: a processing option, the
  tokenizer, `dataset_max_sequence_length`, ...; the prompt lists the fields) joins the confirmation like a raw
  deletion, as does a manifest that cannot be parsed (the build refuses such a folder until then). Broken shards,
  unlisted shards, a missing manifest, a folder built from raw shards that no longer exist or whose raw folder is
  being deleted, and a leftover `.tmp` folder are deleted without asking.

Nothing is touched until every folder was inspected; the queued confirmations are answered **once**, with one
list ("The following folders will be deleted, truncated or re-labelled (...): fineweb_edu: outdated:
dataset_max_sequence_length 2048 -> 4096 / flan: stale: processing.dedup.normalize: true -> false ... Continue?
[y/N]"), default no. `--yes` answers it; without a terminal and without `--yes` the command prints the list and
exits 2 with nothing changed. `train.py`'s auto-prepare never prompts, so it **never deletes raw and never rebuilds
a stale processed folder on its own**: it fails with the same list and the `prepare.py prepare --yes` command.
Lowering `dataset_max_sequence_length` never touches raw (rows are
at most `truncated_at_tokens` long; the build clamps the stored counts, training cuts rows at its own length anyway).

### Tokens, not sequences (`lib/build/planner.py`)

The trainer draws **rows** from one continuous reader per source, one draw per sample, weighted by the stage
schedule (the stage's constant weight, linearly interpolated across a transition window), and packs them end to end,
cut at the length the dataset config plans for, `training_target_sequence_length`: a row serves
`min(its tokens, target)` of the budget (a 530-token row 530 tokens, a 4000-token row the target; a 50k-token row is
stored cut at `dataset_max_sequence_length` and serves the target too). The run therefore needs the
integral of a source's weight schedule over the stage token budgets, in tokens. That is the planner's unit, the
**token budget**: stages sharing a source ADD UP (the reader continues across stage boundaries instead of
re-reading), each stage contributing `(tokens − transition tokens) × weight` plus the trapezoid
`transition tokens × (weight + next stage's weight) / 2` for the window at its end. Rows are what a loader delivers,
so the budget is divided by a **tokens-per-row rate**: the source's `describe_tokens_per_row` (clamped at the target)
until its first raw shard is on disk, the mean of the capped row lengths from then on, read from the `tokens` column
of the raw shards:

```
rate                    = mean over raw rows of min(tokens, training_target_sequence_length), or min(describe_tokens_per_row, target) before the first shard
rows_needed(source)     = ceil(token_budget ÷ rate × 1.2 ÷ (1 − validation_fraction_of(source)))   # source used in train
                        = ceil(source.rows × 1.2)                                                  # source used only in val
rows_sufficient(source) = rows_needed ÷ 1.2, or source.rows                                        # processed rows that serve it
epochs (status table)   = (token_budget ÷ rate) ÷ training rows after the split
```

The `× 1.2` covers what the length filter and the dedup drop and an estimate that ran high, the division keeps the
*training* part at the budget after the resolver holds `validation_fraction` out. The first download is sized at the
estimate; the round loop tops the source up at the measured rate when the estimate was more than 20 % too high
(rows measured shorter), and re-downloads nothing when the processed rows already serve the budget. A run that pads
instead of packing consumes one row per sequence, so it is over-provisioned by `training_target_sequence_length ÷ rate` and never short.
A run whose `training_max_sequence_length` differs from the target is warned about at startup: cut shorter, the rows
serve fewer tokens than budgeted (the sampler cycles the source); cut longer, more. A validation-only source's
`rows` are delivered rows: `× 1.2` downloaded, `rows` of them have to survive the build.

Whether a source is **satisfied** is `SourceLedger.satisfaction()`: `(satisfied, reason)`, the reason being the
status table's last column. The plan, the round loop and the status table all read it:

| reason | when | satisfied |
|---|---|---|
| `ok` | processed rows ≥ `rows_sufficient` | yes |
| `ok, N raw shard(s) past the budget unbuilt` | the same, with raw shards the capped build did not need | yes |
| `exhausted at N of M rows` | the loader yielded fewer rows than asked, some of them training rows | yes, state `exhausted` (the sampler cycles them; `--reopen` if the source has more) |
| `exhausted and NOT ONE of N raw rows survived the build` | the loader ran dry and every row was rejected | **no**: a wrong `fields` / `converter` / `filter` / `language`, and a failed source is a failed build |
| `exhausted, and … leaves 0 training rows` | the few rows all go to the validation holdout | **no**: lower the source's `validation_fraction` or give it more rows |
| `processed rows N < M` | too few processed rows, the loader has more | no: the next round tops it up |
| `processed <reason>` | `processed/` missing, stale, or behind the raw shards while short of the budget | no: build it |
| `raw <reason>` | nothing downloaded / stale or outdated raw, or a raw manifest nobody can parse | no: download, let the repair step delete it, or fix the manifest by hand |

The consequence to keep in mind: **the weights mix rows, not tokens.** The realised token share of a source in a
stage is proportional to `weight × mean_tokens_per_row`. Example: rows of at most 2048 tokens, one 1 B-token stage,
`{fineweb: 0.5 (~1000 tokens/row), gsm8k: 0.5 (~300 tokens/row, 7.5 k rows)}`: the planner books 500 M tokens per
source, 500 k fineweb rows (600 k downloaded) and 1.67 M gsm8k draws, so gsm8k is cycled ~220 times (`epochs` in the
status table) for a realised token mix of about 77 / 23. Set weights with the row lengths of the sources in mind;
`docs/data_mixture.md` prints tokens, the tokens-per-row estimate and the rows it makes of them per source and stage.

### The validation split happens at training time

The pipeline writes **one** folder per source and never decides what is validation. `training/data/dataset_resolver.py`
decides, once per source and run, from how the stages use it:

| the source appears in | training rows | validation rows |
|---|---|---|
| `train` only | all | none |
| `val` only (`rows` says how many processed rows it delivers) | none | all |
| both | the rest | the **first** `ceil(validation_fraction_of(source) × rows)` processed rows |

`rows` is the row count of `processed/<source>` on disk (parquet footers, cross-checked against the manifest); the
per-source `validation_fraction` overrides the dataset default (0.05). Because a source's processed folder was
deduplicated as one set, the two parts never share a document, and the same source gets the same split in every
stage of the run. The chosen `validation_rows` per source travel with every checkpoint next to the dataset config
hash and are verified on resume. A validation-only source reading the same Hub repo and file prefix as a training
source draws a warning (prefer listing the training source in `val` too). The resolver also checks every stage key
directly on disk (folder present, at least one shard, a non-empty row range for its part), independent of the
manifests, and that the run's `training_max_sequence_length` is at most `model_max_sequence_length` and `dataset_max_sequence_length`.

## Where things are cached

Two caches, with different lifetimes:

* **Hub cache** (`~/.cache/huggingface/hub`, or `HF_HOME` / `--cache_dir`): the original repo files fetched by
  `hf_files` / `github_code` (`hf_hub_download`, one file at a time, never twice) and the `datasets` cache of
  `hf_split` sources. Deleting it costs a re-download; nothing else depends on it. `always_range_requests: true`
  (the dataset-level default) reads every Hub file remotely by piece and caches nothing here; set it to `false` to
  let files up to `max_cached_file_mb` (default 32, per source via `load_kwargs.max_cached_file_mb`) land here,
  while larger parquet files are still read remotely row group by row group, projected to the columns the source
  needs (a top-up seeks straight to the row group it needs; fineweb-edu's 2.4 GB files cost a few MB per 1000 rows).
  A remote parquet fetch keeps **every** row of the row groups it read (`rows_needed` is a minimum, the raw shards
  and the loader offset advance to the row-group boundary), so a top-up never re-downloads a row group; row groups
  can be large for book-like sources (gutenberg is ~300 MB per 1,000 rows). Larger `.jsonl[.zst|.gz]` and plain
  `.json` array files are streamed from the start until enough rows were read (`.json` arrays incrementally with
  `ijson`; a top-up inside a partially consumed file re-streams that one file's prefix).
* **`dataset/sources/<source>/raw/`**: the rows this pipeline kept: the append-only cache that `processed/` is
  built from, deleted only by the confirmed repair above. For a `github_code` repo that is every language the pass
  decoded (members past their target, and `sources/<repo tail>_<slug>/raw` folders for the languages without a
  source), roughly the fetched volume in zstd; the build turns only the budgeted part into `processed/`. `dataset/hub_index/` holds the small JSON file indexes
  (file list per glob, rows per file, row-group layout, per-language rows per row group) that let `hf_files` fetch
  at an offset without opening earlier files; it is safe to delete (rebuilt on demand).

## Progress display and logs

On a terminal `prepare` runs inside a live dashboard (`lib/ui/dashboard.py`, `rich`): a header (config, round,
step, elapsed), a **downloads** panel (one row per running download: rows on disk / target, rate, current download
speed and bytes fetched, elapsed, source rows consumed, surplus rows, current repo file; plus a summary line: jobs
done, rows of the round, bytes fetched, elapsed), a **builds** panel (one row per running build: raw rows processed, current raw
shard; plus its summary line), the **log**
panel with the latest lines, and a footer naming `dataset/build.log` (every log line goes there). Finished rows
disappear into the summary; at most eight rows are shown per panel ("… and k more"). A download row opens at the
rows the source already has on disk (a resumed source starts where it stood) and counts only rows up to its target;
rows fetched past it (a loader finishing a remote row group, a github_code member reading on for the other languages,
a language without a source) are stored and show as `surplus` in the postfix, not in the count. Resizing the terminal redraws
the frame from a cleared screen (both dashboards; `ui/display.py`). Nothing else reaches the
terminal while the display is up: every `logging` record (the HuggingFace libraries' included), `warnings` and
stray prints land in the log panel, the libraries' own bars are silenced. Warnings and the tables (plan, repair,
status) are *kept* and printed once, unwrapped, after the display closed; the scrollback of a run is those lines
and the final table, no frame. Ctrl-C and SIGTERM leave the same way. When stderr is
not a terminal (`nohup`, redirects) or `DATA_PREP_PROGRESS=0` is set, there is no dashboard and plain timestamped
log lines are written instead. A terminal that dies mid-run (closed window, dropped SSH session) does not end the
run: the display closes itself and the run continues headless. `tail -f dataset/build.log` shows it; start long
runs under tmux to come back to a live display.

## Training auto-prepares

`training/train.py` loads the run config's `dataset_config`, runs `status`, and with `auto_prepare: true` (the
default) runs `prepare` in-process on the main rank (`prepare_num_workers` / `prepare_pass_workers` / `prepare_max_parallel_downloads`,
the same dashboard, `build.log` and lock; `assume_yes=False`, so it never deletes raw), then re-verifies. With
`auto_prepare: false` a missing dataset is a hard error quoting the `prepare.py prepare` command. Gated sources
(`nampdn-ai/mini-peS2o` in the crow config) need `HF_TOKEN` in the environment (or `--hf_token` for `prepare`).
The dataset config's hash and the validation split are written into every checkpoint; resuming with a changed
dataset config is an error unless the run config sets `allow_dataset_change: true`.

## Adding a source

One YAML entry under `sources:` plus its weight in the stages that use it. Pin the `revision` (`git ls-remote` on
the Hub repo, or the commit shown on the dataset page) so row order is stable across increments; give
`describe_tokens_per_row` the source's ballpark mean row length in tokens: the planner sizes the first download with
it (the raw shards then measure the real rate; an estimate within 20 % above the truth costs no second round).

Loaders (`lib/sources/loaders.py`, `loader:`; all are `(source, offset, count, shared_parameters) -> Iterator[row]`, the
last a `SharedLoaderParameters`: token, index directory, file callback, download counters, column projection):

| Loader | Use for | Notes |
|---|---|---|
| `hf_files` | **default for Hub repos with many files** | `load_kwargs: {data_files: <glob>, max_cached_file_mb: 32}` (`data_files` required, relative to the repo root); files sorted by path; files up to `max_cached_file_mb` are downloaded one at a time into the Hub cache on demand and read locally, larger `.parquet` files are read remotely by row group (every row of a fetched row group is kept, so a top-up never re-downloads one) and larger `.jsonl`, `.jsonl.zst`, `.jsonl.gz`/`.json.gz` files and plain `.json` arrays (incrementally via `ijson`) are streamed from the start; a file index under `dataset/hub_index/` lets a top-up skip files already consumed |
| `hf_split` | split-name based repos (e.g. `gsm8k` with `name: main`) | `train[a:b]` slicing; `datasets` downloads and caches the file once and slices locally; `load_kwargs` go to `load_dataset` (`name`, `data_files`, ...) |
| `hf_stream` | fallback | `datasets` streaming with `skip(offset)`; **caches nothing**: every fetch re-streams from the start, so avoid it for anything large |
| `github_code` | `codeparrot/github-code-clean` | `hf_files` over `data/*.parquet` keeping rows of `language:` (`text_field: code`); all language sources of a repo download in one pass over the shared files and the index stores per-language row counts (per row group for partially read parquet files) |
| `local` | your own data | `path:` directory of `*.parquet`, `*.jsonl` (plain, `.zst` or `.gz`), `*.json.gz` or `*.json` files (the Hub reader's formats), read in sorted file order |
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

`local` reads every file directly under `path` in a format the Hub reader knows (`*.parquet`, `*.jsonl`, `*.jsonl.zst`,
`*.jsonl.gz`, `*.json.gz`, `*.json`) in sorted order. Listing the source in `val`
too gives a held-out split without a second source; a separate validation-only source (`rows: N`, `val` only)
should point at files no training source reads.

## Processing toggles

Configured in the `processing` block (dataset-level default, per-pretrain-source override); see "Build" above for
the order they run in.

- `dedup.mode`: `exact` (default; Bloom filter, `normalize`, `bloom_memory_mb`), `minhash` (exact first, then
  MinHash/LSH; `threshold`, `num_perm`, `ngram`; not for scale, pretrain sources only; an instruct source under
  `minhash` is a config error) or `none`.
- `quality_filter: true` keeps documents with ≥ 3 sentences, ≤ 30 % ALL-CAPS words, ≥ 25 % alphanumeric characters,
  ≤ 30 % repeated 2-grams and ≤ 20 % repeated 3-grams: prose heuristics, wrong for code.
- `decontamination.enabled: true` drops documents whose 13-grams overlap more than `threshold` with a benchmark test
  set (`benchmarks`: gsm8k_test, math_test, humaneval, mbpp_test, arc_challenge_test, hellaswag_test, mmlu_test,
  winogrande_test; each pinned to a Hub commit that enters the processed hash, downloaded once into
  `dataset/benchmarks/`, `lib/stages/benchmarks.py`). Needs network access on first use.
- `min_chars` drops shorter pretrain documents; there is no upper character bound; the token truncation at
  download is the upper bound.
- Token counting (`token_count`) happens at download time: `tokenizer` counts with the config's tokenizer,
  `estimate` uses chars / 4; both add the two special tokens the trainer puts around a row. Instruct rows are
  counted whole, as the trainer formats them (instruction, input and output joined by blank lines).

## Differences from the thesis run

The thesis data was prepared with exact dedup, fuzzy dedup at Jaccard 0.95, tokenizer counts truncated to 2048
tokens and the quality filter, decontamination and PII masking skipped; the pretrain sources were fetched as fixed
row counts (e.g. 9.0M fineweb-edu documents) and the finetune data as a 400k-example mixture built up front. The
crow config mirrors that with exact dedup on, `dataset_max_sequence_length` 2048 and everything else off, but: fuzzy dedup is
off by default (available as `dedup: {mode: minhash, threshold: 0.95}`), PII masking no longer exists, texts are
truncated at the token cap when downloaded instead of stored whole, download sizes follow the token budgets
(÷ the measured tokens per row, × 1.2) instead of fixed row counts, the finetune stage mixes the eight instruct sources by weight in the
dataloader (no prebuilt mixture, no cross-source dedup of instruct data), validation is the first 5 % of the
fineweb-edu training source (`validation_fraction`) instead of fineweb-edu's `sample-10BT` (which overlapped the
training dump: about a fifth of that validation set was training data), several HuggingFace ids moved
(`wikimedia/wikipedia`, `openai/gsm8k`, `common-pile/arxiv_papers_filtered`) and every source is pinned to a
revision. `docs/data_mixture.md` lists the resulting budgets.
