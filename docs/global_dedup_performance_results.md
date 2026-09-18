# Global deduplication performance and compatibility

Implemented on 2026-09-18 against `eed1208`: reuse the global admission state across compatible source passes, and optionally compute document keys in ordered spawn workers. The serial default remains `--global_hash_workers 1`. No dataset identity, shard layout or policy changes are required.

## Local measurements

Three independent fresh-output trials per case, run sequentially on local ext4/NVMe with warm input caches. Python 3.11.14, PyArrow 25.0.1, rbloom 1.5.4, 24 CPUs in the process affinity mask, 8 MiB global Bloom filter, 4,096-row admission batches. The baseline used an isolated archive of `eed1208`; the other cases used the working implementation. These small synthetic fixtures establish eliminated recovery work and expose worker overhead; they do not predict throughput on the remote `/mnt/vast` filesystem.

The short fixture has 90,000 rows across three sources (60,000/15,000/15,000), with a 64-character text body plus an identifying prefix. The long fixture has 36,000 rows (24,000/6,000/6,000), with a 4,096-character body plus prefix. Every fifth row uses a source-independent prefix to create cross-source duplicates.

| Fixture | Case | Median wall seconds | Range | Peak tree RSS MiB | Restored prior keys |
|---|---|---:|---:|---:|---:|
| short | baseline | 1.579 | 1.539–1.626 | 221 | 132,000 |
| short | reuse | 1.330 | 1.288–1.407 | 218 | 0 |
| short | hash2 | 1.276 | 1.208–1.344 | 398 | 0 |
| short | hash4 | 1.371 | 1.298–1.412 | 554 | 0 |
| long | baseline | 1.704 | 1.696–1.724 | 413 | 52,800 |
| long | reuse | 1.601 | 1.579–1.660 | 433 | 0 |
| long | hash2 | 1.566 | 1.561–1.609 | 716 | 0 |
| long | hash4 | 1.482 | 1.462–1.551 | 865 | 0 |

`baseline` reconstructs the filter for each unfinished source; `reuse` hashes serially with the retained session; `hash2`/`hash4` add two/four hash processes. Wall time includes initial recovery, worker startup/shutdown, admission and publication, but excludes candidate fixture creation and final comparison. RSS is the maximum 100 ms sampled sum across the benchmark process and descendants, including fixture setup; shared pages are counted per process. It is neither an exact peak nor unique physical memory.

Reuse removed all inter-source key restoration in both fixtures. Baseline median restoration time was 0.221 s for short text and 0.092 s for long text. Four-worker hashing improved the longer-text fixture, but was slightly slower than reuse alone for short text. Two-worker results overlap the serial/reuse trial ranges. Worker CPU and memory costs increased, so there is no justification for enabling a large worker count by default.

All trials matched the baseline for fixture/config hashes, ordered output shard bytes, frontiers and statistics. Raw reports retain source-code digests, library versions, process affinity, per-source wall times, restoration times/counts, aggregate parent-plus-joined-child CPU seconds, and sampled RSS: [short](../tools/data_preparation/results/global_dedup_20260918/short.json), [long](../tools/data_preparation/results/global_dedup_20260918/long.json).

## Compatibility and correctness

A separate fixture was prepared with the original `eed1208` code and interrupted immediately after committing candidate row 8,192 within a 10,000-row candidate shard. Four clones were continued with the old code and with the new code using 1, 2 and 4 workers. All matched for shard bytes, frontiers, statistics, candidate-generation references and the existing partial-source generation ID. The fixture included Unicode/control characters, normalized duplicates, legacy instruction rows with null inputs, and structured message rows. Newly created output UUIDs were excluded from cross-run equality. [Comparison artifact](../tools/data_preparation/results/global_dedup_20260918/resume_compatibility.json).

Tests cover live reuse, same-source continuation, already-partial next sources, top-up replay, benchmark seeds, zero-retained source dependencies, capacity checks, metadata corruption, delayed writer failures, hash errors/worker death, reader error ordering, bounded submissions, cancellation and cleanup. A real 1 MiB Bloom test inserts 550,000 distinct candidate keys, observes false positives, then checks that live and reconstructed filters make identical subsequent decisions.

Validation: **290 focused tests passed** across global admission/session/workers, runner/top-ups, source builds, storage ownership/atomic writes, CLI and diagnostics. Ruff, strict mypy and basedpyright passed on all 14 changed Python files. Remote H100/server throughput was not tested.

```bash
uv run --no-sync pytest \
  data_preparation/lib/stages/test_global_dedup.py \
  data_preparation/lib/stages/test_global_build.py \
  data_preparation/lib/stages/test_global_session.py \
  data_preparation/lib/stages/test_global_hash_workers.py \
  data_preparation/lib/build/test_global_runner.py \
  data_preparation/lib/build/test_runner.py \
  data_preparation/lib/stages/test_build.py \
  data_preparation/lib/stages/test_build_workers.py \
  data_preparation/lib/storage/test_atomic.py \
  data_preparation/lib/storage/test_ownership.py \
  data_preparation/cli/test_prepare.py \
  data_preparation/lib/test_download_debug.py \
  data_preparation/lib/test_download_debug_file.py \
  data_preparation/lib/test_download_profile.py -n 0
```

## Reproduce safely

Use a new output directory on the filesystem to measure. This benchmark creates only deterministic offline fixtures; it never reads or writes an existing preparation dataset. Each trial has a separate process group, a 180-second default timeout, a 4 GiB default sampled tree-RSS limit, and bounded fixture rows/input size. Owned benchmark processes are stopped on failure; logs and result artifacts remain for inspection.

```bash
uv run --no-sync python tools/data_preparation/benchmark_global.py \
  --output-root /tmp/global-benchmark-new \
  --baseline-repo /path/to/eed1208-archive \
  --rows 24000 --text-chars 4096 --trials 3
```

Omit `--baseline-repo` to compare against the current code with per-source reconstruction. For the short fixture use `--rows 60000 --text-chars 64` and another new output directory. Default cases are baseline, reuse, hash2 and hash4; hash8 is an explicit optional case. Increase limits deliberately for larger samples. The fixture is synthetic and compresses well; use representative server measurements before selecting a worker count for the 100B run.

## Deploying to an existing preparation run

Let the running process continue until ready to deploy. To adopt the code, stop preparation gracefully, wait for exit, deploy, and rerun the same command/config/dataset directory. Add `--global_hash_workers 2` or `4` only when desired; omit it for serial hashing with automatic state reuse. No redownload or output migration is needed for compatible existing progress. Restart still reconstructs the committed global filter once; generation replay can require additional reconstruction. The local implementation work did not stop or modify the remote preparation process.
