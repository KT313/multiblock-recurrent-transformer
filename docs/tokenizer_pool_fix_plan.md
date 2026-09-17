# Tokenizer pool review fixes

Baseline: `60c1bd1e0bd170dcf5402e83445d90433e68c4c0`

Status: implemented and validated on 2026-09-17. Scope remains the three reproduced review findings.

## Objective and boundaries

Make cancellation reliably settle returned futures, restore the offline pipeline replay, and make
tokenizer-process debug reports distinguish task execution from time outside tasks.

Preserve tokenizer output, exact text prefixes/counts, row order, rejection counters, durable offsets,
shard boundaries, and existing failure precedence. Keep `TOKEN_BATCH=2048`, thread allocation,
spawn-based execution, the existing in-flight limit, and parent-owned writers unchanged. No dataset
hash migration, production-data modification, new dependency, or Rust change is required.

The working tree contains unrelated configuration edits and an uncommitted JSON-profile extension.
Implement and validate these fixes against the commit above plus only the intended changes, using an
isolated checkout if needed. Do not include unrelated edits in a later commit. Never push.

## 1. Settle cancelled proxy futures

**Files:** `data_preparation/lib/stages/tokenizer_pool.py::_wrapped` and
`data_preparation/lib/stages/test_tokenizer_pool.py`.

The callback currently calls `done.exception()` before checking cancellation. That call raises
`CancelledError` for a cancelled source future, leaving the returned proxy pending forever.
`TokenizerPool.__exit__` uses `shutdown(cancel_futures=True)`, so cancellation is a supported path.

Required behavior:

| Source/proxy condition | Result |
|---|---|
| Source cancelled, before or after wrapping | Proxy becomes cancelled and terminal; `result()` raises `CancelledError` |
| Source succeeds | Forward the original result |
| Source fails normally | Preserve the original exception object in the same-process bridge |
| Source fails with `BrokenProcessPool` | Preserve the existing named tokenizer-process `RuntimeError` |
| Caller cancels proxy before source finishes | Later forwarding does not raise `InvalidStateError` or log a callback traceback |

Implementation approach:

1. Handle source cancellation before calling `exception()` or `result()`.
2. Make the proxy's cancellation-versus-result transition race-safe. A separate `if not result.done()`
   followed by `set_result()` is insufficient because cancellation can happen between them. Use the
   public Future transition mechanism, such as `set_running_or_notify_cancel()`, when claiming delivery.
3. Ensure forwarded cancellation notifies Future waiters, including `wait()`/`as_completed()`, rather
   than checking only `cancelled()`. Test this explicitly against the supported Python version.
4. Keep this a one-way result/state bridge. Do not add worker termination, retries, or an implied
   guarantee that cancelling the proxy stops underlying computation. Reverse cancellation would be
   a separate contract change. Document this distinction in `_wrapped`.

Add deterministic unit tests for every table row, including already-completed/already-cancelled
sources and cancellation racing with success and failure. Use events/barriers and bounded waits;
assert both terminal state and absence of `concurrent.futures` callback-error logs.

Add a small real spawn-pool shutdown test: hold workers with events, enqueue enough bounded work to
leave a genuinely pending task, trigger shutdown with cancellation, and verify its proxy settles.
Do not assume the second submitted task is cancellable: the executor may already have marked it
running. Capture underlying futures or use controlled test seams to identify queued work. Release
all gates in `finally` and enforce an outer timeout so a regression cannot hang the test runner.

**Gate:** every source completion/cancellation produces an observable terminal proxy state; ordinary
results and error naming remain unchanged; shutdown tests finish without callback exceptions.

## 2. Restore the replay counter's initialization contract

**Files:** `tools/data_preparation/replay.py::ReplayCounter.__init__`,
`tools/data_preparation/test_replay.py`, and `tools/data_preparation/test_native_backends.py`.

`_TokenStep.parallel_batches` and `start` now read `counter.pool`; pooled submissions also need
`counter.tokenizer_dir`. `ReplayCounter` overrides the parent constructor without initializing these
fields. The existing pipeline replay currently fails with `AttributeError` before processing rows.

Initialize `self.pool = None` and `self.tokenizer_dir = path` in `ReplayCounter`, alongside its existing
fields. Keep the supplied tokenizer read-only. Do not call tokenizer preparation/repair or create a
dataset layout merely to satisfy this constructor. Do not hide the missing contract with a production
`getattr(..., None)` fallback in `_TokenStep`.

Check the other `TokenCounter` subclasses/stand-ins for the same initialization requirements.
`BuiltinCounter` and `RustCounter` call the replay constructor and should inherit this fix. The
baseline replay must continue using the in-process path; do not silently enable the new process pool
or claim that this fixes benchmarking of process-pool scaling.

Validation:

- Restore the existing failing `test_replay_is_repeatable_without_mutating_original_rows`.
- Assert that replay counters have no pool, retain the supplied tokenizer directory, and report one
  parallel batch through `_TokenStep`.
- Exercise original, candidate, and built-in variants through the pipeline, comparing stored rows,
  counts, schemas, and durable progress. Keep the compiled Rust variant optional and explicitly report
  a skip when its artifact is unavailable.
- Run one bounded CLI pipeline replay with the existing tiny tokenizer, temporary JSONL input and a
  fresh temporary output root. This is a functional smoke test, not a new speedup measurement.

**Gate:** pipeline replay starts and finishes for supported variants, preserves original input objects,
and retains the existing refusal to reuse output directories.

## 3. Mark tokenizer tasks as worker activity

**Files:** `data_preparation/lib/stages/tokenizer_pool.py::_truncate_in_process`, `_count_in_process`,
`data_preparation/lib/test_download_debug.py`, and `docs/download_debug.md`.

The initializer starts a `pool_idle_or_dispatch` span. Cleaning workers suspend that span using
`measured_worker`, but the new tokenizer entry points do not. Consequently, tokenization is reported
as both task work and idle/dispatch time. The review reproduced approximately 0.288 seconds of counting
and 0.293 seconds of reported idle time for the same operation.

Import and apply the existing `measured_worker` decorator to both entry points. Use distinct task
labels, for example `tokenizer_truncate` and `tokenizer_count`. Wrap the whole task, including lazy
tokenizer loading, and retain the existing nested `truncate_batch`, `encode_batch`, and `count_batch`
measurements. Do not duplicate the idle-state machinery or start another reporter thread.

Keep the entry points at module scope and preserve their metadata so spawn-process pickling still
works. The decorator's `finally` must restore the idle/dispatch span on both success and failure.

Extend the existing controlled spawn-worker debug tests to include both tokenizer entry points:

- Block inside a real task through a test seam; receive a child-PID report before releasing it.
- Compare two snapshots wholly inside the blocked task. Task time must advance while
  `pool_idle_or_dispatch` does not. An interval crossing task entry may legitimately include some idle time.
- Release the task, confirm its result, and verify idle/dispatch time resumes afterward.
- Make a task raise, then verify idle state is restored and a subsequent task can run.
- Preserve debug-disabled behavior and parent-mediated `--debug-file` output. Do not mutate worker
  globals in the main pytest process; use isolated spawned workers.

Update the debug guide with the two new outer task labels. Explain that nested encoding stages
overlap with task time, while `pool_idle_or_dispatch` means time outside task execution, including
IPC/deserialization; it is not a pure kernel-wait measurement. Do not promise per-source attribution
for the shared tokenizer pool as part of this fix.

**Gate:** busy tokenizer processes no longer accumulate idle/dispatch time during their task body;
results, failure propagation, reporter cleanup, and disabled-debug behavior remain unchanged.

## Validation and delivery

Start by adding the regressions above and confirming they fail on the baseline. Implement the fixes
in the order listed. Focused checks:

```bash
uv run --no-sync pytest data_preparation/lib/stages/test_tokenizer_pool.py tools/data_preparation/test_replay.py tools/data_preparation/test_native_backends.py data_preparation/lib/test_download_debug.py data_preparation/lib/test_download_debug_file.py -n 0
```

Then cover the existing integration contracts:

```bash
uv run --no-sync pytest data_preparation/lib/stages/test_download.py data_preparation/lib/stages/test_download_cleanup.py data_preparation/lib/build/test_runner.py data_preparation/cli/test_prepare.py -n 0
```

Run Ruff, strict mypy, and basedpyright on the affected code and tests, plus `git diff --check`.
Distinguish baseline/environment failures from newly introduced failures. Use offline fixtures and
small worker pools; serialize resource-heavy checks. No production download restart is required to
verify these three fixes.

Deliver a cohesive fix commit only when requested, with the regression results and optional-test
skips recorded. None of these changes should require deleting shards, rebuilding raw data, changing
thread settings, or altering the download command.


## Implementation and validation record

- `_wrapped` acknowledges source/caller cancellation through `set_running_or_notify_cancel`, preserving
  results and named worker-loss errors without leaving cancelled proxies pending in Future waiters.
- `ReplayCounter` initializes its tokenizer directory and explicitly stays in process (`pool=None`).
- Both tokenizer process entry points use `measured_worker`; their task bodies, including lazy loading,
  suspend idle/dispatch accounting and restore it on success or failure. Dedicated regressions live in
  `data_preparation/lib/stages/test_tokenizer_debug.py` to keep the pending JSON-profile edits separate.
- Before the fixes, new cancellation, replay-contract, and spawned-task-accounting tests reproduced
  the reviewed failures.
- Against an isolated archive of the baseline plus only these fixes: focused tests passed (50 passed,
  two optional Rust tests initially skipped), and the download/cleanup/runner/CLI integration checks
  passed (200 passed). With the existing optional Rust library supplied, the replay, native parity,
  and tokenizer debug checks passed (10 passed, no skips).
- Ruff, strict mypy and basedpyright passed for all five affected Python files; whitespace checks passed.
- A 32-row offline CLI pipeline replay in the working checkout completed for original, candidate and
  built-in variants with exact output parity. Its report is under
  `/tmp/tokenizer-replay-smoke-k87c4mps/checkout-output/`; this functional check is not a throughput claim.
  Unit/integration validation above excluded the unrelated working-tree changes.

Dataset outputs, batch/thread allocation and resume contracts remain unchanged. No production download
or HPC throughput test was run.
