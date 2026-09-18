**Implementation plan: reuse global admission state and parallelize key computation**

Status: implemented on 2026-09-18 after critical review. Baseline: `eed1208`. See [validation and performance results](global_dedup_performance_results.md). Repository: `/home/tobi/Desktop/multiblock-recurrent-transformer`.

The user is preparing an already-downloaded 100B-token dataset on a remote server. Source-local builds have finished; the final cross-source deduplication pass uses roughly one CPU core. One observed rate was about 9,200 candidate rows/s. This observation does not establish the time split between normalization, hashing, Bloom operations, decoding, and storage.

Implement two independent improvements, in order:

1. Retain the global Bloom filter and associated admission state across compatible source transitions.
2. Optionally compute document keys in multiple processes, while admitting and publishing batches in their original order.

The changes must preserve existing prepared data, resumability, deduplication decisions, and dataset identities. This plan does not authorize stopping the remote job or operating on its data.

**Review conclusions and implementation order**

Both improvements are reasonable, but they address different costs. Reuse removes repeated recovery between sources; it does not accelerate hashing within the currently running FineWeb pass or remove the first recovery after a restart. Parallel hashing can improve active admission only if its saved CPU time exceeds serialization and process overhead. Ordered Bloom operations remain serial.

The review found these implementation traps, resolved below:

| Risk in the initial plan | Required clarification |
|---|---|
| Vague cache provenance could allow an incorrect source handoff | Specify an immutable reuse stamp and invalidate by default during a pass |
| Skipping the constructor also skips checks it currently performs | Preserve policy/capacity validation and distinguish metadata checks from full disk recovery |
| Filling the hash window could surface a later reader error before earlier batches | Preserve the input error position and drain the preceding ordered work |
| A batch-count limit could be mistaken for a RAM limit | Account for reader, writer, serialization, and worker copies; retain the serial default |
| Blocking waits and cleanup could obscure cancellation or the original failure | Check stop around waits/admission, stop refilling, and preserve the primary exception |

Deliver phase 1 with its own correctness and performance evidence before implementing phase 2. Do not combine them into a new general pipeline framework. Keep the benchmark a small offline harness, not another preparation entry point.

**Current pipeline and relevant files**

```text
data_preparation/prepare.py
  -> cli/commands.py::prepare_requested_steps
  -> lib/build/runner.py::prepare
  -> prepare_global
       download_and_build_missing: source-local candidate builds
       run_global_admission_rounds: ordered final pass
         build_source: ensure sufficient source-local candidates
         stages/global_build.py::build_global_source
           validate candidate/output generations and restore committed progress
           construct GlobalAdmission by reinserting committed keys
           reader thread -> ordered admission -> writer thread
           finish source and complete output generation
```

Relevant paths are relative to `data_preparation/` unless otherwise noted:

| File | Responsibility |
|---|---|
| `lib/build/runner.py` | Invocation lifetime, source order, top-up loop, exclusive dataset lease |
| `lib/build/global_preparation.py` | Candidate targets and complete-snapshot fast path |
| `lib/stages/global_build.py` | Per-source admission, threaded reader/writer, publication |
| `lib/stages/global_dedup.py` | `global_key`, `GlobalAdmission`, `GlobalFrontier`, policy constants |
| `lib/stages/global_output.py` | Candidate iteration, committed-key recovery, generation validation/replay |
| `lib/stages/exact_dedup.py` | Existing Bloom implementation and `mix128` hash adapter |
| `cli/arguments.py`, `cli/commands.py` | Runtime-option parsing and propagation |
| `lib/download_debug.py`, `lib/download_profile.py` | Existing worker initialization and bounded diagnostics |
| `lib/stages/test_global_dedup.py` | Key, admission, recovery and policy tests |
| `lib/stages/test_global_build.py` | Reader/writer ordering, cancellation and failure tests |
| `lib/build/test_global_runner.py` | End-to-end preparation, replay, top-up and corruption tests |

Current storage layout:

```text
dataset/processed/<source>/                         source-local candidates
dataset/.dataset-scopes/<config-hash>/processed/<source>/  final global output
```

The final source manifest commits its listed shards, candidate offset, cumulative counts, retained-key digest, candidate generation, and preceding output generations. It is the recovery authority. The in-memory Bloom filter is not persisted.

`GLOBAL_BATCH_ROWS` is currently 4,096. `_Reader` queues two batches; `_Writer` serializes publication with a queue of one pending batch. Worker completion order must never determine which duplicate wins.

**Rules both improvements must preserve**

- Keep `global_key()` semantics exactly: normalization, format separation, JSON serialization, signed 64-bit SHA-256 prefix, and structured-conversation hashing. Local candidate `hash` values are not substitutes for global keys.
- Keep the Bloom implementation, bit budget, false-positive behavior, insertion order, `mix128`, benchmark preseeding, policy constants, and source priority unchanged.
- Keep row order, logical batch boundaries, output shard boundaries, schema, compression, counters, and retained-key digests unchanged. Admission/commit units remain whole batches. Read-ahead can change how much work is pending when a real-time stop arrives; it must never advance durable progress for an unadmitted batch.
- Keep final writes in the lock-owning parent process. Hash workers are pure computation workers and must not write files, allocate their own Bloom filters, or mutate progress.
- Do not weaken ownership, candidate-generation, dependency-generation, frontier, or corruption checks for the sake of cache reuse.
- Existing partial global output must resume without a format migration. Complete snapshots remain no-ops. Runtime worker count must not affect dataset/config hashes.
- Unexpected worker or publication failures stop visibly. No silent switch to serial execution, skipped batch, or guessed recovery cursor.
- Preserve the protection introduced by `eed1208`: source-build workers write private files and the parent publishes them; satisfied candidate budgets start no shard workers.
- Preserve unrelated dirty files. Do not run `git push`.

Fresh builds naturally create different UUIDs/timestamps. Parity comparisons may normalize those incidental values; comparisons against an existing resumed build must preserve its actual generation identities and references.

**Phase 0: establish a bounded reference**

Before changing runtime code, create an isolated baseline checkout/archive of `eed1208` and deterministic offline fixtures. Record any relevant uncommitted changes separately: the working tree already contains unrelated CLI/profiling edits, and a HEAD archive does not contain them. Include several pretraining sources with cross-source duplicates, legacy instruction pairs, structured messages, short and long documents, and a separate benchmark-preseed fixture. The current configuration disallows benchmark preseeding together with message sources; preserve that restriction.

Record retained rows and their order, per-shard bytes, semantic manifest fields, frontiers, policy/config hashes, and recovery outcomes. Keep old-format partial-output fixtures for the compatibility tests below.

Measure two separate workloads:

- A multi-source workload with a large first source, to expose repeated Bloom restoration.
- A long-document admission workload, to expose normalization/hash cost and process-transfer overhead.

The existing `tools/data_preparation/benchmark_download.py` measures downloading/tokenization, not this final pass. Add a dedicated bounded benchmark under `tools/data_preparation/benchmark_global.py` during implementation. Use disposable output directories outside the real dataset. Record fixture/config digests, versions, CPU allocation, filesystem, elapsed time, aggregate process CPU/RSS, rows/s, and recovery time. Set explicit row/byte/time limits and retain comparison artifacts.

**Phase 1: retain admission state across sources**

Current problem: `build_global_source()` creates `GlobalAdmission` for every unfinished source. Its constructor rereads and reinserts all retained keys from earlier sources, plus this source's committed prefix. That is required for restart recovery but redundant at an ordinary source boundary within one run.

Introduce a small invocation-owned `GlobalAdmissionSession` in a focused module such as `lib/stages/global_session.py`. It owns the existing `GlobalAdmission` instance, its validated provenance, and whether the instance is safe to reuse. Keep orchestration visible in `run_global_admission_rounds()` and `build_global_source()`.

1. Create one session for `run_global_admission_rounds()` under the existing exclusive dataset lease. Allocate the Bloom state lazily at the first source that actually needs admission.
2. Preserve the current candidate/output validation and `restore_output_generation()` decisions before selecting cached state. A cache hit must not bypass these checks.
3. On first use, construct `GlobalAdmission` with the existing recovery iterator: benchmark preseeds, previously committed source keys, and the active source's committed keys. Keep its count and digest validation intact.
4. On an ordinary transition to the next source, reuse the instance after confirming its frontier matches the required committed frontier and its consumed-prefix provenance still matches the validated generation chain.
5. After each source pass, drain and join the writer, finish all required manifest/generation writes, and only then mark the session reusable. A source that remains budget-short may return a clean partial state; the next call still has to validate it.
6. Invalidate/discard the session on every abnormal exit from admission, reader/hash processing, writer flush/shutdown, statistics publication, or generation completion. Recovery must reconstruct from disk rather than repair speculative in-memory mutations.

The session must keep the filter, retained-key checksum object, preseeding state, and full frontier together. Copying counters onto a filter containing another prefix is incorrect.

Reuse validation needs more than equal counters. Track the consumed prefix's output-generation dependencies and the active candidate/output generation when resuming within a source. An explicit successful source handoff advances that provenance. A newly created next-source manifest does not by itself invalidate the completed preceding prefix.

Make that contract concrete:

- Bind the session to one dataset root/scope, ordered policy, and the invocation's immutable benchmark-seed identity. The runner owns the seed spool for the entire session; obtain a fresh `seeds.keys()` iterator only when reconstruction needs one. Do not store and later reuse a consumed iterator, or let standalone calls silently substitute different seeds into an existing session.
- Store a frozen stamp containing the full committed frontier, an ordered tuple of completed `(source, output_generation_id)` dependencies, and, for an unfinished active source, its name, candidate generation, output generation, and starting frontier. Store values, not references to the writer's mutable manifest/dictionaries. Zero-retained-row sources still contribute generation dependencies.
- For same-source continuation, require an exact stamp match against the validated restored manifest. For a next-source handoff, require the cached completed dependency chain to equal the next manifest's dependencies and the cached frontier to equal the next manifest's committed frontier. An already-partially-processed next source will usually be ahead of the cache and therefore needs recovery including its own keys.
- Remove the reusable entry before beginning a pass. Only reinstall it after every writer and manifest operation, the final stop check, and a check that `source_frontier(manifest) == admission.frontier` succeed. Any exception then leaves no reusable entry, including failures before admission begins. A complete-output early return must either preserve an exactly matching completed stamp or discard stale state; it must not advance a filter by assigning new counters.
- Keep a single cached entry. A legitimate mismatch discards it before constructing its replacement, avoiding two full Bloom allocations. Retain constructor policy/frontier validation on recovery; reuse must validate the same policy invariants through the immutable stamp and run the existing per-source capacity check. Do not bypass validation by calling an internal constructor-less path.

The optimization relies on the exclusive dataset lease and managed files remaining immutable between parent-controlled writes. It intentionally stops rereading previously committed key columns at every source boundary. It cannot promise detection of manual or external shard edits that preserve metadata during a live invocation. On actual disk reconstruction, missing/extra keys and digest mismatches must still fail exactly as before; metadata corruption must still fail before reuse. This is not a new full-content integrity scanner.

Use these decisions:

| Situation | Required action |
|---|---|
| Fresh first source | Initialize once, including benchmark preseeds |
| Existing complete prefix, then unfinished source | Skip complete sources without eagerly restoring; recover once at the first unfinished source |
| Compatible next source after successful publication | Reuse the current admission instance |
| Same candidate generation and exact committed frontier | Reuse if session provenance matches |
| Candidate generation changed and existing logic replays the source | Discard state and reconstruct the validated preceding prefix |
| Earlier output dependency changed | Apply existing downstream replay rules and reconstruct the required prefix |
| Corrupt/mismatched recovery metadata | Raise the existing explanatory error; do not reinterpret it as a harmless cache miss |
| Publication, worker, reader, or cancellation exception | Invalidate the session; next attempt recovers from committed output |
| Entire dataset already complete | Allocate neither Bloom state nor hash workers |

Top-ups are particularly important: rebuilding source-local candidates can change their generation and reset that source's global output to its starting frontier. The old Bloom filter can still contain keys from the discarded attempt. Reusing it would incorrectly reject replayed rows. Keep the existing replay behavior, even when this requires another reconstruction. The target is one reconstruction per uninterrupted compatible suffix, not one reconstruction regardless of state changes.

Keep standalone `build_global_source()` callers working through an optional session argument or a short-lived local session. Do not introduce a module-global cache. Emit concise logs/timing for restore versus reuse, with retained-key counts, so the avoided work is observable.

**Phase 2: ordered parallel key computation**

Add `--global_hash_workers N`, default **1**. This is a runtime-only preparation argument, not a dataset YAML field. Validate it as a positive integer before resources are created. Default 1 keeps the existing serial key path; values above 1 use a spawn process pool.

Wire it through:

```text
cli/arguments.py
  -> cli/commands.py::prepare_requested_steps
  -> runner.prepare
  -> prepare_global
  -> run_global_admission_rounds
  -> global hashing pipeline
```

Existing library callers, including training auto-preparation, retain the default. Do not add training settings for this change. `--num_workers`, `--pass_workers`, and `--tokenizer_threads` keep their current meanings. Do not allocate a hash pool for download-only runs, local-only deduplication, dry runs, or complete-snapshot no-ops.

Introduce `lib/stages/global_hash_workers.py` with a small spawn-based pool and a top-level picklable batch-key function that calls the existing `global_key()` for every row. Workers receive only the kind and the fields that key computation needs; the parent retains the original output rows. Workers return keys only. Do not add a persistent key cache or another on-disk intermediate format.

Pipeline:

```text
existing reader thread
  -> bounded submissions to hash worker processes
  -> consume futures in input-batch order
  -> parent performs ordered Bloom admission and frontier updates
  -> existing writer thread publishes shards/manifests
```

Use an explicit deque containing each original batch and its future. Consume the oldest future, not completion order. Start with at most N outstanding batches, including completed results waiting behind a slower batch. Refill only after the oldest batch has been admitted, so popping a future does not silently allow N queued batches plus another batch waiting for admission. Keep the existing reader/writer queue bounds in addition to this limit; do not eagerly submit the entire source with `Executor.map`.

Handle read-ahead termination explicitly. If reading the next batch fails while filling/refilling the window, save that exception at its input position, stop reading, admit the preceding queued batches in order, and then raise it. A hash failure at an earlier batch takes precedence; no later batch may be admitted. A stop request takes priority over draining speculative hashing: discard unadmitted results and let the writer settle already-enqueued commits using its existing behavior. Test these rules with event-gated tasks, not timing-dependent sleeps.

This bounds batches, **not bytes**. The parent also holds reader batches (two queued plus a producer batch), an admission batch, writer survivors (one queued plus one being written), and the decoder's current rows. Pickle buffers and workers add copies of key fields and normalization temporaries. Token counts do not impose a maximum stored text length. Document this limitation and measure long-document peak memory before recommending N; do not claim memory is only N times a batch or automatically derive N from all server CPUs. Byte-based scheduling or splitting a logical batch into smaller hash tasks is a separate follow-up if these measurements require it.

Extend `GlobalAdmission.commit_batch()` with an optional keyword-only sequence of precomputed keys. Preserve the current serial behavior when keys are absent and share the same ordered admission loop. For supplied keys, check count and signed-64-bit integer validity before mutating the filter (`type(key) is int`, excluding booleans). Keep source/kind and capacity validation on both paths. Do not recompute keys in the parent on the parallel path. The payload projection must preserve missing/null legacy `input` semantics and the complete structured `messages` value; it must not normalize or coerce fields independently of `global_key()`.

Reuse a lazily created pool across compatible source passes within the invocation. Drain source-specific tasks before a source transition, candidate rebuild, or replay. Retain ordinary 4,096-row admission batches even if worker tasks complete in a different order. Correctness must also hold for smaller test batches, final partial batches, and batches whose rows are all rejected by the Bloom filter.

Keep the process boundary simple first. Python serialization of long documents may be expensive; measure it before adding shared memory, native extensions, or workers that reopen candidate files. Those designs are not part of this implementation.

On failure or stop, stop submissions, cancel unstarted jobs, and join active workers before ordinary cleanup finishes. Check `should_stop` before submission, while waiting for results using short timed waits, and immediately before admission. A wait timeout is only a chance to check cancellation, not a reason to fail or retry a task. Running Python 3.11 executor tasks cannot be cancelled through `Future.cancel()`; ordinary cleanup can still wait for the current bounded tasks. Do not promise a fixed shutdown time, reach into executor internals, or add a new forced-termination policy. Preserve existing CLI interrupt handling and keep the dataset lease until parent writers have stopped.

The invocation owns the lazy pool outside individual source passes; each pass owns its reader/writer and pending deque. On errors, stop the reader, settle/join the writer, clear speculative batches, and close the failed pool. Preserve the original exception if cleanup also fails, attaching/logging the secondary error. On a successful source boundary no tasks remain pending; an idle pool may be reused even when admission state later needs reconstruction. Never carry tasks across candidate rebuilds.

Report source and candidate-batch offset for hash-worker exceptions or a broken process pool. Add existing-style worker diagnostics via `initialize_worker_debug`/`measured_worker`; distinguish hashing, result waits, and filter restoration without per-row logging. Because the pool spans sources, pass the source name with each task and bind its profiling context per task; an initializer's first-source label would become misleading. Treat diagnostics as optional and keep workers from writing dataset files or shared logs directly.

The asynchronous writer needs special care: enqueueing a commit does not prove durability. Its failure may surface only at the next publish, `flush()`, or context exit. Every such failure must invalidate the reusable admission session, even if `GlobalAdmission.commit_batch()` itself returned successfully.

**Regression tests and acceptance gates**

Add focused tests beside the new helpers and extend the existing integration tests. Cover:

1. **Reuse actually happens:** a fresh three-source pass creates one Bloom filter, preseeds once, and never rereads earlier committed keys between compatible sources. A resume behind a complete prefix reconstructs once at the first unfinished source, then reuses it.
2. **Output parity:** serial baseline, reuse-only, and hashing with 2/4 processes retain identical ordered rows, keys, shard bytes, frontiers, counters, and policy/config hashes. Exercise Unicode, escaped/control characters, normalized duplicates, legacy pairs, and multi-turn messages.
3. **Order independence:** deliberately complete later hash batches first; the earlier duplicate still wins. Compare a restored 4,096-row boundary and a mid-candidate-shard offset.
4. **Queue bounds and error ordering:** block the earliest future while later tasks finish and assert submissions/buffered batches never exceed the selected window, including during refill. Inject a reader failure during initial filling and later refill; earlier queued batches must commit before that error surfaces. Include long rows and an all-rejected batch; measure memory separately rather than treating the queue test as a byte-bound proof.
5. **Existing resume compatibility:** stop using the old implementation and resume using the new implementation with 1/2/4 workers. Compare to uninterrupted baseline output. Worker count may change between restarts without changing identity.
6. **Generation completion edge:** recover a source whose finish frontier was committed but whose generation-complete flag was not yet written.
7. **Top-up/replay safety:** cover same-source top-ups, exhausted/zero-yield rounds, rebuilt candidate generations, reopened higher-priority sources, and changed dependency generations. Verify stale keys do not reject replayed candidates and later duplicates still lose to earlier sources.
8. **Failure recovery:** inject worker death, hash exceptions, reader errors, shard-write failures, manifest failures, and delayed writer failures at flush/context exit. Retry from disk and compare output. Test cache invalidation even when another exception is already unwinding.
9. **Corruption remains visible:** on disk recovery, missing/extra committed keys and wrong digests still raise. On both recovery and reuse, reject inconsistent starting frontiers, later-source offset corruption, and incompatible session/preseed policy. Test metadata checks even with a populated cache. Include a completed-prefix handoff into an already-partial next source, a completed-output early return, zero-row source dependencies, and a capacity failure on the reused path.
10. **Resource lifecycle:** no hash pool/admission-filter allocation for no-ops or disabled modes; invalid worker counts (including boolean library arguments) fail early; cancellation while filling, waiting, and before admission joins workers and leaves recoverable progress. Assert cleanup preserves the original failure. Use real spawn processes for smoke/parity/worker-death tests with a test timeout and cleanup of owned processes; use deterministic fakes/events for scheduling tests. Retain the existing parent-death publication regression from `eed1208`.

For partial-output compatibility tests, create a baseline fixture once, clone it for each resumed run, and compare with an uninterrupted continuation from that same generation history. A separately fresh build has different generation IDs and cannot be compared literally. Require identical Parquet bytes only with the same PyArrow/compression versions and write settings; ordered rows, schemas, shard boundaries and semantic manifests remain the essential correctness gate. Exercise the real Bloom filter at meaningful load as well as tiny duplicate fixtures, so reuse is compared with reconstruction in the presence of false positives.

Suggested focused command, adding the new test modules as they are created:

```bash
uv run --no-sync pytest \
  data_preparation/lib/stages/test_global_dedup.py \
  data_preparation/lib/stages/test_global_build.py \
  data_preparation/lib/build/test_global_runner.py \
  data_preparation/lib/build/test_runner.py \
  data_preparation/lib/stages/test_build.py \
  data_preparation/lib/stages/test_build_workers.py \
  data_preparation/lib/storage/test_atomic.py \
  data_preparation/lib/storage/test_ownership.py -n 0
```

Also run affected CLI and debug tests plus Ruff, strict mypy, and basedpyright on changed Python files. Use the project's actual virtual environment for type-checker import discovery. Keep validation offline and resource-bounded; do not run the default eight-worker full suite alongside performance measurements.

**Performance comparison and rollout**

Measure sequentially on the same input and output filesystem, with identical Bloom size and admission batches:

| Case | Admission lifetime | Hash workers |
|---|---|---:|
| Baseline | Original per-source reconstruction | 1 |
| Improvement 1 | Reused compatible session | 1 |
| Improvement 2a | Reused compatible session | 2 |
| Improvement 2b | Reused compatible session | 4 |
| Optional scaling check | Reused compatible session | 8 |

Report startup/recovery, active admission, inter-source gaps, and whole-run wall time separately. Count restored keys to demonstrate eliminated work, rather than inferring it only from elapsed time. Include at least three comparable trials for throughput, report the spread, and identify cache-warm versus cold conditions. Use a fresh output copy for every trial so a completed-snapshot fast path cannot masquerade as speedup. Track total parent-plus-child RSS and CPU, not only the parent in btop; report the measurement method and acknowledge shared-page double counting in summed RSS. Never drop host caches or benchmark against live preparation data. Do not extrapolate local-filesystem timings to the remote `/mnt/vast` storage without a target-server measurement.

Do not promise a numerical speedup before measurement. Higher CPU utilization alone is not success. Keep default hashing at 1 until target-server measurements establish a useful worker count; retain the explicit serial setting if process transfer costs dominate. Reuse is enabled for compatible runs once its correctness gates pass.

Implement and validate phase 1 before phase 2 so each effect is attributable and independently reviewable. Document the runtime flag, recovery/replay exceptions, resource bounds, and observed performance. Do not change the Bloom size or source ordering to obtain a better benchmark.

For deployment, let any running process continue while the changes are developed. A running Python process will not adopt edited modules. To use the new implementation on the remote server, stop preparation gracefully, wait for it to exit, deploy the validated code, and rerun the same configuration/data directory with the chosen `--global_hash_workers` value. Existing compatible progress must resume; one initial Bloom reconstruction remains expected. Do not automatically deploy or stop the remote process as part of implementation.

Non-goals: native/Rust Bloom replacement, sharded or unordered Bloom admission, changed deduplication policies, changed publication batch sizes, persistent Bloom snapshots, dataset migration/redownload, planner-wide caching, and a dashboard redesign.
