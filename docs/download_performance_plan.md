# Download preparation performance plan

Date: 2026-09-17

Analysis baseline: `abf6f2e1323b424a7497fec2e3ad148ce582e77e`

Code review: 2026-09-17 against `38873609436dc4f9386d8f6c8345885b32ecef0e` and the working tree.
The review snapshot contained unrelated merge conflicts; recheck current status and relevant diffs
before further implementation.

Status: accessor/count optimizations, an offline replay, and opt-in diagnostics implemented;
see [implementation results](download_performance_results.md). Native comparison tooling is
experimental. Remote-copy and concurrency phases remain conditional; dataset artifacts are unchanged.

## Objective and scope

Increase sustained throughput of the **download stage**, which includes decoding, tokenization,
text truncation, and publishing raw Parquet shards. Preserve the exact dataset content and safe
resume behavior. A constant network rate is not itself the success criterion: retained rows/s,
stored tokens/s, and elapsed time for identical input are.

Primary workload: `config/datasets/final_100B_tokens.yaml`, eight download jobs, 64 MiB remote
prefetch, and a Slurm allocation of 32 CPUs. The latest server command requested 30 Rayon threads,
four source-build threads, and one optional cleaning-pass worker. It used `prepare`, which can
start source builds later; benchmark the download stage separately to avoid that confounder.

Do not change training, evaluation, tokenizer vocabulary, chat serialization, deduplication,
or processed-output architecture. Preserve unrelated dirty configuration files. Never push.

## Execution order and decision gates

| Step | Deliverable | Decision before proceeding |
|---|---|---|
| 0 | Small offline replay and baseline report | Repeated baseline runs agree semantically; workload and resource limits are recorded |
| 1 | Direct encoding length/offset access | Exact parity, failure/resume checks, and matched performance measurements |
| 2 | Targeted diagnostic report for unexplained waits | Attribute the remaining stalls before selecting another optimization |
| 3 | Remote read-copy patch, if justified | Preserve read/close/error contracts and demonstrate reduced cost |
| 4 | One scheduling experiment at a time | Promote only a measured improvement with bounded memory and unchanged semantics |

Keep Phase 0 minimal enough to evaluate Phase 1 promptly. Full production diagnostics and remote
profiling are not prerequisites for the accessor patch. Phases 3–4 are conditional, not a requirement
to implement every proposed optimization. Stop expanding scope when further gains are unmeasured.

## Evidence and its limits

- The original eight-thread run averaged 7.06 CPU cores. Eight native threads accounted for
  roughly 89% of the process CPU. The later 30-thread request averaged 6.07 cores and peaked at
  15.79. Different input positions and profiling overhead prevent a speed comparison.
- The later profile showed full-queue waits in at least 98.6% of GitHub fetch-thread samples,
  91.8% in downloads_1, 58.1% in downloads_2, 60.7% in downloads_4, 91.9% in downloads_6,
  and 82.7% in downloads_7. These are per-thread sampled weights, not CPU percentages.
- FineWeb, OpenWebMath, TinyGSM, and GitHub tokenizer callers spent approximately 69% of samples
  waiting for an `encode_batch` completion. Gutenberg was approximately 65%, Wikipedia 50%.
  The summary does not expose the native Rayon workers' internal hot functions.
- Some sources are input-starved: one Parquet consumer waited for input throughout the capture;
  arXiv's token worker waited for input throughout its capture.
- Timed condition-variable waits beneath SSL, gzip, and Python postprocessing are consistent
  with GIL contention. They do not identify the GIL owner or prove that all HTTP-path waits are
  remote-server latency. Avoid diagnosing network throttling from a Python frame name alone.
- Raw data already uses buffered Parquet writes, normally 10,000 rows per shard; passive GitHub
  languages use 2,500. No per-row file write/flush is present in that path.
- The server filesystem is NFS over RDMA. A short 256 MiB write probe achieved 1,066 MiB/s with
  stable 28–32 ms write-plus-fsync chunks, but reused the same 32 MiB payload and omitted
  shard/manifest metadata operations. It does not prove sustained real-workload storage speed.
- Earlier NFS interval counters recorded only two completed writes, approximately 1.85 MiB.
  Those counters are mount-wide and not contemporaneous with every later profile. The very large
  RPC backlog number and occasional RPC wait remain unexplained; do not declare NFS universally
  healthy. Local NVMe iostat does not measure this mount.
- The Python summary retained only the three most common exact stacks per thread. Absence of
  writes from that summary does not establish zero write cost.
- The Rayon thread count requested through the environment still needs to be verified in the
  launched process. This profile is not an inventory of all native worker threads.

Working diagnosis: several fetchers are blocked behind the tokenization/storage consumer, while
other sources are waiting on input. Queue backpressure establishes where to investigate; it does
not distinguish encoding from Python result handling or synchronous writes. Increasing download
concurrency alone cannot resolve a full downstream queue, and these captures do not establish an
optimal Rayon setting.

Confirmed local inefficiency: with tokenizers 0.23.1 and a synthetic 16,384-token encoding,
`len(encoding.ids)` took about 159 microseconds versus 0.03 for `len(encoding)`;
`encoding.offsets[index]` took 658 microseconds versus 0.07 for `token_to_chars(index)`.
These are isolated accessor measurements, not an end-to-end speedup estimate.

During this analysis, direct length equality and individual-offset equality were checked on
nine strings with the locally available legacy llama-32k and synthetic tokenizer artifacts:
6,537 and 2,025 offsets respectively, with no missing direct offsets or mismatches. This does
not cover every tokenizer or the server's llama-32k-chat-v1 artifact.

## Current call paths

| Area | Files and responsibilities |
|---|---|
| CLI/thread setup | `data_preparation/prepare.py::_materialise`, `cli/commands.py::configure_preparation_environment`: defaults TOKENIZERS_PARALLELISM=true and RAYON_NUM_THREADS=8; explicit environment wins |
| Download orchestration | `lib/build/runner.py::download_and_build_missing`: separate download and build pools; `num_workers` controls source builds |
| Fetch/dispatch | `lib/stages/download.py::_fetch`: feeds converted rows, preserves per-source offsets, stops at configured targets |
| Token batch/worker | `download_state.py::_TokenStep`, `download_workers.py::_TokenWorker`: 256 rows per token batch, three queued batches per pass, ordered tokenization followed by synchronous storage |
| Tokenizer | `tokenizer_loader.py::SavedTokenizer`, `download.py::TokenCounter`: tokenizers Rust backend; one process-wide Rayon pool |
| Text truncation | `truncation.py::truncate_many`: pre-cut at 32 characters per token of cap, encode, cut at a token start, re-encode cut prefixes until counts fit |
| Remote reads | `sources/loaders.py::hub_fetcher`, `sources/hub_files.py::HubFetcher`, `_CountingRaw`, `sources/prefetch.py::PrefetchReader` |
| Remote Parquet | `hub_files.py::read_rows_multi`, `_parquet_rows`, `row_group_batches`: source/group skipping, projections, row-group completion, grouped language collection |
| Decode/Python conversion | `hub_files.py::row_group_batches`, `_parse_json_lines`, `_Increment.convert`: Arrow decoding and `to_pylist`, gzip/JSON decoding, filters, row copies, and text conversion before tokenization |
| Input identity/cache | `sources/hub_files.py::FileIndex`, `sources/hf_cache.py`, `sources/loaders.py`: resolved Hub revision, saved file/group counts, cache location, per-source overrides |
| Instruction fitting | `download_state.py::_fit_message_rows`, `lib/conversation_format.py`, `tokenization/chat.py`: message fitting uses a separate path from pretrain truncation |
| Raw publication | `storage/parquet.py::ShardWriter/publish_shard`, `raw_folder.py::record_shard`, `manifest.py::save`, `atomic.py::write_atomically` |
| Shutdown | `download_workers.py::finish_download_pass`: settle, close reader, join token worker, salvage partial shards, release writers, establish exhaustion |
| Speed display | `stages/download_progress.py`, `lib/ui/dashboard.py`: completed-read byte counts, five-second rate window |

The remote wrapper retains a current block plus one pending block, not a deep producer queue.
At 64 MiB and eight active remote files, those blocks alone can occupy approximately 1 GiB;
Arrow, underlying fsspec buffers, caller results, Python rows, and encodings add to that.
That is not a total memory bound: decoded batches contain up to 1,000 original rows, queued
256-row token batches still contain untruncated input, and each language has its own partial
token batch and shard buffer. Include dynamically discovered passive languages in memory accounting.

The layered read path is `Arrow/gzip -> optional PrefetchReader -> _CountingRaw -> HfFileSystem
-> fsspec cache -> HTTP`. Files at or below the per-source cache threshold (default 32 MiB) instead
use `hf_hub_download` and a local file. Parquet and stream fsspec blocks default to 1 and 8 MiB.
Inspect effective per-source settings; a single global prefetch value does not describe every input.

## Non-negotiable correctness contracts

1. Identical stored text prefixes, token counts, row order, accepted/dropped rows, and special-token
   accounting for identical original input and tokenizer artifacts.
2. Preserve the 32-characters-per-token pre-cut, exact boundary selection, strict shortening guard,
   and prefix recount loop. Merely satisfying the token cap is not sufficient parity.
3. Preserve raw/processed identity hashes and tokenizer metadata for behavior-preserving changes.
   Do not invalidate existing raw data, request a rebuild, or silently adopt different token rules.
4. Preserve per-row RowProgress, grouped language routing, passive-language collection, source
   discovery, row-group alignment, exhaustion, and exact instruction-target behavior.
5. Preserve exception identity/visibility, secondary cleanup errors, bounded buffering, and worker
   ownership. Durable consumed offsets may include rejected rows, including trailing rejects,
   but must never skip an accepted row whose storage has not committed. Preserve the existing
   per-shard and final-offset rules. Never finalize writers before worker termination. No retries
   that duplicate already-published output.
6. Keep existing file and directory fsync, manifest commit ordering, compression, and shard sizes
   in the first implementation. A separate durability change is not justified by these profiles.
7. All benchmarks write only to explicitly owned temporary output directories. Never run repair,
   overwrite, or benchmark publication against the user's production dataset.
8. Preserve lock ownership at public preparation entry points. A lower-level replay must have
   exclusive ownership of its output: `ShardWriter.__enter__` deletes shards at/after start_shard,
   so pointing it at an existing dataset is destructive even without a repair flag.

## Phase 0: establish a reproducible baseline

Add a bounded, offline benchmark module under `tools/data_preparation/`, separate from the
GPU/model-oriented `tools.bench` CLI. Proposed entry point:
`python -m tools.data_preparation.benchmark_download` (does not exist yet).

The harness should:

- Accept an existing tokenizer directory, an explicit source fixture, an owned scratch-output
  root, source concurrency, and repetition count. Refuse input/output path overlap and production
  dataset destinations. Resolve symlinks for overlap checks; create a unique empty child directory
  for every run and keep all manifests, indexes, and caches there. Never reuse an output as a fresh
  run. Do not call repair or mutate input manifests or the supplied tokenizer artifact.
- Exercise three levels independently: encoding metadata access; actual `truncate_many` plus
  `TokenCounter.count_many`; and replay through the real fetch/token/store pipeline with network
  replaced by a deterministic offline source.
- Use representative original documents: short FineWeb-like prose, long books/papers, code,
  math, Unicode, and pathological long runs. Fix selection/seed/order and record the fixture hash.
  Already-downloaded raw text has been truncated; it is useful for a short-document workload but
  cannot stand in for original long documents or their first truncation pass.
- Use a bounded original-input fixture, e.g. a fixed 64 MiB sample plus explicit long-document
  cases. Obtain any new remote fixture separately from the production preparer and retain its
  immutable source revision, file, row range, projection, and content digest.
- Reload or reconstruct fresh row objects for every replay: `_TokenStep` mutates row text and
  token fields. Reusing dictionaries would silently benchmark already-truncated inputs.
- Include enough distinct short/medium rows to publish multiple full 10,000-row shards and a
  final partial shard, plus a separate bounded long-document case. Include a mixed-language
  group that exercises passive 2,500-row publication. A 64 MiB sample alone does not guarantee
  this coverage. Do not repeat a few identical strings just to fill shards; tokenizer caches and
  unusually compressible output would bias the result. Record output shard counts and sizes.
- Run each measured repetition in a fresh process, with its warm-up and environment configured
  before tokenizer use; this also makes peak-RSS comparisons meaningful between trials.
  Use the same saved tokenizer and input batch order for baseline and candidate.
- Include a warm-up, at least three measured repetitions, and median/spread. Measure the full
  lifetime of encodings, including release of the final batch, rather than ending before cleanup.
  Time through worker join, final partial-shard publication, manifest save, and reader close.
  Report startup separately from steady processing. Alternate baseline/candidate order, keep
  warm-up/cache policy identical, and collect throughput without a sampling profiler attached.
- Record wall time, process CPU seconds, peak RSS, original input characters, kept rows, stored
  tokens, compressed output bytes, and a streaming digest of ordered semantic output.
  Exclude nondeterministic generation IDs/timestamps from semantic equality checks.
  Compute ordered digests separately for each source, then aggregate in fixed source-name order;
  independent sources have no required cross-thread completion order. Compare schema and durable
  progress/counters as well as text/tokens. Verify outputs outside the timed interval.
- Record git revision/diff identity, Python and tokenizers/PyArrow versions, tokenizer hash,
  effective CPU affinity, Slurm allocation, cgroup CPU quota/throttling, and explicit pool settings.
  Inspect ancestor cgroup limits as well as the task cgroup, and measure throttling deltas during
  the trial. Record PyArrow CPU/I/O pool sizes, TOKENIZERS_PARALLELISM, Rayon, OpenMP/BLAS settings,
  and any child processes. CPU usage of only the parent misses work in spawned build/pass workers.

Use a small workload matrix: pretrain short/long documents, grouped active/passive GitHub, legacy
instruction count/drop, and message instruction fitting. Time the observed pretrain workloads first;
the other branches are regression coverage, not grounds for claiming the same speedup everywhere.
Remote-wrapper benchmarks must inject a fake `HubFetcher.remote` over local Parquet/JSON data and
force the remote branch; simply replaying a local row iterator bypasses Arrow/prefetch entirely.
Replay with fixed eight-job membership to isolate the patch, then separately exercise the real
runner's job turnover. After a download-only win, validate a bounded mixed prepare run in scratch.

Benchmark baseline versus Phase 1 with eight download jobs, 256-row token batches, unchanged
queue depth, and unchanged prefetch. First keep Rayon at 30 to isolate the code change;
then compare 8, 16, and 30 in fresh processes. Do not change multiple dimensions at once.
Reproduce the 32-CPU experiment only on an allocation that can support it; use an explicitly
smaller matched local case otherwise and label it separately.

Gate: repeated baseline runs first establish semantic stability; then report exact candidate parity
and measurement variability before claiming an improvement. Bound input rows/bytes, output bytes,
wall time, and process-tree memory; record an incomplete trial as a failure, not a throughput result.
The existing accessor timings alone do not satisfy this gate.

## Phase 1: eliminate unnecessary tokenizer metadata materialization

Highest-priority production change; small scope and no new concurrency.

### 1A. Count directly

In `truncation.py::truncate_many`, calculate `count = len(encoding)` once and reuse it for the
cap check and retained result. In `download.py::TokenCounter.count_many`, use `len(encoding)`
instead of `len(encoding.ids)`.

Keep `SavedTokenizer.encode` returning IDs; training and chat need that API. A dedicated scalar
count method is optional only if scalar `TokenCounter.count` is measured as significant.

Do not mechanically replace other `len(encoded.ids)` expressions. In message paths,
EncodedConversation/ChatEncoding.ids is already an ordinary Python list, not the allocating
tokenizers Encoding property.

### 1B. Retrieve only the cut offset

Replace full `encoding.offsets` materialization with `encoding.token_to_chars(max_tokens)`,
using the returned start coordinate. Change the private cut helper to accept the start coordinate
and preserve `text[:min(start, len(text) - 1)]` exactly.

The direct accessor returns an optional result. If it returns None, preserve existing semantics
by explicitly retrieving `encoding.offsets[max_tokens]` for that exceptional case. This is a
documented equivalent lookup, not an estimated offset or silent content change. If neither lookup
is valid, allow a visible error. Never guess a character position.

### Validation

Extend `stages/test_truncation.py` with an independent copy of the previous algorithm as an
oracle and compare complete (text, count) results. Cover empty inputs/batches, caps 0/1/exact/over,
negative caps, leading whitespace, multilingual text, emoji, combining characters, literal
special-token spellings, overlapping Metaspace offsets, and repeated shortening.

Update `_ViterbiLikeStub`: its SimpleNamespace encodings currently expose only IDs/offsets.
Provide an explicit test encoding supporting length and individual-offset lookup while retaining
the existing three-round recount assertion. Add a case for the None-offset compatibility path.

Extend `test_tokenizer_loader.py` to compare direct length/offset lookup with existing properties
on offline fixtures and the actual pinned chat tokenizer when available. Required offline
coverage must not rely exclusively on optional cache-dependent tests that can all skip.
Before production rollout, require a successful parity run with the exact server tokenizer artifact
and tokenizers version. If unavailable locally, mark that gate pending for the server; a synthetic
tokenizer success cannot close it. Preserve single-string, no-added-specials encoding; do not apply
the direct-offset shortcut to paired encodings with sequence-relative coordinates without analysis.

Retain `test_download.py` count/estimate tests, and verify row counts, stored prefixes, token
budgets, and manifest offsets after stop/resume with the optimized path.

Gate: all parity checks pass, retained tokens and text are unchanged, and matched pipeline
measurements show the magnitude of improvement or clearly report that it is negligible.

## Phase 2: measure remaining waits with bounded diagnostics

Add opt-in diagnostic output, not per-row logging or a new always-on dashboard subsystem.
Proposed `--download_profile PATH` writes a versioned JSON report outside dataset identity;
wire it through `cli/arguments.py`, `cli/commands.py`, the runner and download jobs as runtime
options, not fields included in dataset hashes. Start with stage totals; add expensive detail only
where those totals justify it.

Use per-worker accumulators and batch-level timings. Report a final summary and optionally
a coarse 30-second snapshot; bound histograms and avoid a shared lock on every row.

Measure separately:

- Fetch owner: time inside next(input iterator), conversion/dispatch, and queue.put. Split iterator
  time into Arrow read/decode, `RecordBatch.to_pylist`, gzip/JSON decode, and grouped routing when
  needed. These Python conversion paths can limit throughput independently of tokenization.
- Token worker: time in queue.get, encode_batch wall time, truncation/result handling,
  encoding release, and storing rows.
- Writer: Arrow table construction, compression/write, file sync, directory sync, and manifest
  publication. Nested times must be identified so users do not add them twice.
- Remote worker: underlying read duration, requested/returned bytes, read-call counts, completed
  block counts, consumer Future wait, and blocks discarded by seeks.
- Token workload: characters entering encoding, rows, truncation rounds, output token counts,
  batch completion latency, and per-source progress, including passive GitHub output.

Time a remote read as a whole; do not label it pure network latency because it can include
Python scheduling and local allocation. Do not sum overlapping thread durations into elapsed
time. Do not infer GIL ownership from a futex symbol.

Separate consumer bytes, prefetch/underlying read bytes, and actual HTTP requests/response bytes.
`FetchStats.bytes_fetched` measures wrapper reads plus an mtime-based whole-file cache heuristic;
it is not a wire counter. fsspec cache hits, read-ahead, retries and discarded speculation can make
these differ. Only report HTTP request counts when measured at the HTTP boundary; otherwise mark
them unavailable. Preserve current dashboard semantics while adding explicitly named diagnostics.

Publish diagnostic output after download cleanup with one owning aggregator, never let competing
workers overwrite the same report, and avoid timing callbacks that can disrupt shard publication.
Validate the report destination before work starts. On report-write failure, surface the failure
after normal worker/writer cleanup; retain any primary preparation error and attach the reporting
error as secondary. Reports should contain counts/timings, not document bodies or credentials.
Test success, cancellation, failed readers/writers, and report failure. Compare diagnostics on/off
for semantic parity and overhead, and exclude diagnostic runs from headline speed comparisons.

Measure publication with both fresh and realistically sized existing scratch manifests/indexes:
`Manifest.save` serializes the complete shard list each time, and index saves include metadata work.
Two atomic publications per raw shard normally mean two file syncs plus two directory sync attempts;
the existing helper can ignore unsupported directory-sync errors. Preserve that behavior rather
than claiming a stronger crash guarantee. Include mkdir/stat/rename and publication tail latency.

Test diagnostics with injected clocks or controlled blocking events. Disabled diagnostics
must avoid timing calls in inner row loops and must leave output and exception behavior intact.

If needed, collect a short native CPU profile of the actual Rayon worker TIDs, subject to existing
cluster permissions. The supplied py-spy summary cannot identify native worker utilization,
long-document imbalance, allocator/destructor cost, or shared-pool fairness by itself.
Correlate any GIL-owner samples with native worker CPU, queue waits and completed batches before
attributing a stall. If progress-display locking appears material, compare a no-progress replay
with the real dashboard separately; preserve logging and rendering behavior in the first patch.

Gate: measurements identify whether the next target is encoding, Python handling, remote reads,
byte copying, or publication. Avoid introducing process workers based only on caller wait time.

## Phase 3: reduce remote wrapper copies

Two concrete allocation sites deserve a separate, measured patch:

1. `hub_files.py::_CountingRaw` overrides readinto but inherits RawIOBase.read. The prefetch worker
   calls read on it, which routes through an additional buffer and readinto copy.
   Implement an explicit read method that delegates to the underlying read and updates
   bytes_fetched once. Preserve readinto support, closed-handle behavior, EOF, exceptions,
   and return types; verify read(None), read(-1), read(0), and short reads against the old API.
2. `PrefetchReader.read` always allocates a zero-filled bytearray, copies data into it, and then
   creates immutable bytes. Add a contiguous-current-block fast path returning bytes directly
   (or one slice for a partial read), advancing the cursor exactly as before. Keep the existing
   cross-block/readinto path initially. Do not return a mutable buffer or retain extra old blocks.

Keep at most current plus one pending block internally. Returning immutable bytes may let a caller
retain an old block; account for that as caller-owned memory, as with today's returned result.

Extend `sources/test_prefetch.py`, `test_hf_files.py`, and `test_sources.py`: zero-length reads
must not trigger fetching, negative/all reads match the old contract, EOF/beyond-EOF seeks,
cross-block/random reads, non-aligned Parquet footer seeks, counting exactly once, short-read
failure, discarded-prefetch errors, early stop, and joined/closed worker ownership.

Use deterministic in-memory remote handles with recorded read ranges, then offline Parquet
and compressed-JSON fixtures, both with prefetch disabled and enabled. Compare output bytes,
logical demand ranges, read-call counts, and exceptions. Cancellation timing may change which
speculative call completes; test its bound and failure propagation with controlled events rather
than demanding identical concurrent completion order. Any changed read-all call grouping must be
explicitly reviewed for extra transfer and timeout behavior, not silently assumed equivalent.
Benchmark 1/8/64 MiB reads and actual Arrow/gzip consumers with RSS and CPU time.

Gate: less copy/allocation cost or higher matched pipeline throughput without changing delivered
bytes, required ranges, row order, failure visibility, or retained-buffer bounds; report any change
in speculative traffic. HTTP read timeouts are not whole-operation deadlines: progress/retries
can extend a read and therefore close/join. Do not promise a 30-second shutdown bound.

## Phase 4: tune scheduling only after the small fixes

Keep experiments out of defaults until they win on the target server.

- Sweep Rayon 8/16/30 with identical eight-source replay. Investigate load imbalance from long
  documents and verify effective worker count/CPU quota before concluding that more cores help.
- Only then compare smaller/larger token batches (e.g. 64/256/512 rows). Larger batches may improve
  scheduling efficiency but worsen tail latency, encoding lifetime, instruction overshoot handling,
  and memory. Count rows and characters; do not increase all queues with batch size.
- If character-based batch limits are useful, make them explicit and preserve instruction drain
  conditions, per-language partial batches, and end-of-pass settlement.
- If GIL-bound handling still dominates, prototype a bounded **spawn** process pool for pure
  tokenize/truncate work. Return compact texts/counts, not Encoding objects or expanded ID lists.
  Keep writes, ordered row bookkeeping, counters, and manifests in the owning process.
  Assign sequence numbers and bound both submitted work and out-of-order completed results.
  Preserve original RowProgress and rebuild dropped-row accounting in input order.
- Initialize tokenizers inside children, avoid forking an initialized Rayon pool, and bound
  process-count times native-thread-count within the allocation with room for fetching/writing.
  Include tokenizer duplication, IPC copies, child RSS, cancellation, and failure costs in results.
  Try a small pool first; preserve the current threaded path as the default unless the evidence
  supports a separate opt-in mode.
- If remote-read latency dominates instead, measure demand bytes versus speculative bytes,
  seek discards, Arrow read ranges, and underlying fsspec read-ahead before tuning prefetch.
  Test 0 (off)/8/16/64 MiB with all other dimensions fixed. Do not unconditionally increase read-ahead
  for random-access Parquet or download whole files.
- Split writing onto another bounded worker only if publication stalls are measured as material.
  This needs explicit ownership and shutdown design and separate crash/resume tests.

Do not lower the text pre-cut, replace exact counts with estimates, omit recounts, disable passive
language collection, or change tokenizer normalization to obtain a performance number.

## Validation and rollout

Use this regression matrix to select tests for the touched code. Cooperative stop and an abrupt
process exit are different cases; do not promise automatic repair of an orphan shard after a crash.
Compare with the baseline's existing refusal/recovery behavior in owned scratch directories.

| Contract | Required cases and existing test anchors |
|---|---|
| Encoding parity | Independent previous-algorithm oracle, count/offset getters, repeated shortening, actual server artifact; `stages/test_truncation.py`, `test_tokenizer_loader.py` |
| Raw row semantics | Pretrain/estimate, instruction drops, message fitting, stored folder cap overriding a smaller requested cap; `stages/test_download.py`, `tokenization/test_chat.py` |
| Grouped sources | Active/passive/discovered languages, unequal saved offsets, skipped row groups, target reached mid-group; `stages/test_download.py`, `sources/test_hf_files.py` |
| Publication/resume | Full and partial shard, stop/resume, trailing rejected rows, offsets/counters/schema; `storage/test_raw_folder.py`, `test_manifest.py`, `test_parquet.py` |
| Failure ownership | Read/encode/write/manifest callback failure, failure after publication, interrupted join, secondary cleanup failure; `stages/test_download_cleanup.py`, `storage/test_atomic.py` |
| Read wrappers | Cache and remote paths, prefetch off/on, random seeks, zero/all/short reads, EOF, close twice, failed speculative read; `sources/test_prefetch.py`, `test_hf_files.py`, `test_hub_http.py` |
| Runtime options | Environment overrides, diagnostics excluded from identity, no builds in download-only mode, mixed prepare job handoff; `cli/test_prepare.py`, `lib/build/test_runner.py`, `test_global_runner.py` |

For process/writer concurrency proposals, additionally test a worker dying with outstanding batches,
out-of-order completion, cancellation with full queues, and no live workers after failure. Put
timeouts around controlled concurrency tests to turn deadlocks into failures. These are prerequisites
for Phase 4 concurrency changes, not extra work required to replace an encoding accessor.

For Phase 1, run:

```bash
uv run --no-sync pytest data_preparation/lib/stages/test_truncation.py data_preparation/lib/stages/test_tokenizer_loader.py data_preparation/lib/stages/test_download.py data_preparation/lib/stages/test_download_cleanup.py -n 0
```

For Phases 2–3, add:

```bash
uv run --no-sync pytest data_preparation/lib/sources/test_prefetch.py data_preparation/lib/sources/test_hf_files.py data_preparation/lib/sources/test_sources.py data_preparation/lib/sources/test_hub_http.py data_preparation/cli/test_prepare.py -n 0
```

Before review run `make lint`, `make typecheck`, and the relevant broader offline data-preparation
suite. Record pre-existing failures separately with a baseline comparison; do not hide them.
Add focused diagnostic/harness tests at their new paths. Run resource-heavy benchmarks serially.
When shared storage/atomic code is touched, include the storage tests above and callers outside
the download stage. If unrelated merge conflicts prevent broad checks, run focused checks in an
isolated, coherent snapshot and report the remaining validation gap; do not resolve unrelated work.
Plan-only documentation does not require executing these implementation checks now.

For the server comparison, preserve the production dataset. Use immutable inputs and fresh
scratch outputs for baseline/candidate repetitions. Stop the production download before CPU-heavy
replays, or use a separate allocation. Compare local scratch and NFS only with identical content.
For network trials, keep revision/file/row ranges and cache state explicit; compare several trials
and label internet variability. Never claim matched performance from two successive resume
positions or from changing both worker settings and code.

Prefer separate reviewable changes: benchmark baseline, accessor optimization, diagnostic reporting,
read-copy optimization, and finally any evidence-backed concurrency change. A behavior-preserving
patch should resume existing data without a tokenizer/hash migration. Rollback should require only
reverting code and restarting normally, not deleting shards.

Success criteria:

- Exact semantic parity and unchanged resume/failure contracts.
- Improvement exceeds observed run-to-run noise on matched fixtures; report rows/s, tokens/s,
  wall time, CPU cost, and memory, with no fabricated target speedup.
- No unbounded queue, unexpectedly multiplied tokenizer pools, or growing memory over repeated batches.
- Server evidence distinguishes application processing stalls from remote reads and NFS operations.
- No production data removed or rewritten for testing.

## Reference implementations

- [Tokenizer Encoding accessors](https://github.com/huggingface/tokenizers/blob/main/bindings/python/src/encoding.rs):
  length, allocating ID/offset properties, and token_to_chars. Local API probes used 0.23.1;
  record and verify the server version before implementation validation.
- [CPython 3.11 GIL](https://github.com/python/cpython/blob/3.11/Python/ceval_gil.h):
  timed condition-variable waits; a wait symbol alone is not attribution.
- [py-spy](https://github.com/benfred/py-spy): sampled stacks and idle/native profiling limitations.
