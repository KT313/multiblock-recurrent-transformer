# Preparation performance implementation and measurements

2026-09-17. See [the plan](download_performance_plan.md) and
[reproduction instructions](../tools/data_preparation/README.md).

## Implemented

- Truncation reads `len(encoding)` and one cut offset instead of copying whole ID/offset vectors.
  The pre-cut, prefix boundary and recount loop are unchanged, including an explicit offset fallback.
- Count-only batches use the installed tokenizer's `encode_batch_fast`, which omits unused offset
  tracking; versions without that API use ordinary batch encoding. Both paths return direct lengths.
- `--debug` logs live interval overviews every five seconds; `--debug SECONDS` overrides the interval.
  Reports include in-progress spans, coordinator/queue/read/shutdown waits, process CPU, source and
  thread IDs. Spawned MinHash and decontamination workers report through the parent's logger too.
- An offline replay compares frozen original logic, production Python, existing native APIs, and
  an optional small Rust adapter. The adapter remains outside production dependencies and defaults.

Remote-copy changes and new concurrency/process/writer pools remain deferred until stage measurements
justify them. Existing prefetch bounds, shard sizes, fsync behavior, dataset identity and resume rules
are retained. Chat message encoding still runs per message; batching that branch is a separate
opportunity requiring exact conversation/mask parity. Pretrain tokenization already batches 256 rows.

## Local comparisons

Four Rayon threads, saved legacy llama-32k, tokenizers 0.23.1, alternating order, three fresh-process
trials per variant, no sampling profiler. These are synthetic offline CPU/local-storage workloads,
not real server download measurements. All four variants matched exact semantic outputs.

| Workload | Original median | Python accessor change | Existing native APIs | Rust adapter |
|---|---:|---:|---:|---:|
| Mixed pipeline: 23,001 rows, one active/passive group, cap 256 | 1.291 s | 1.210 s (1.067x) | 1.250 s (1.032x) | 1.332 s (0.969x) |
| Long-text truncation: 192 documents, cap 16,384 including specials | 4.899 s | 4.282 s (1.144x) | 4.390 s (1.116x) | 4.361 s (1.123x) |
| Long-text count only | 3.205 s | 3.158 s (1.015x) | 2.619 s (1.224x) | 2.658 s (1.206x) |

The count-only native-API result motivated the additional production `count_batch` change. The table's
Python column predates that addition. Truncation still needs offsets and therefore does not use the
fast no-offset encoder. Moving the loop into Rust did not beat optimized Python in these runs;
there is no evidence here to justify making the adapter a production requirement.

A separate final production count-only comparison measured **3.275 s versus 2.660 s median
(1.231x throughput)** across three trials, again with exact counts. An earlier verification run
overlapped a type-check process and is excluded from the reported performance results.

The mixed fixture publishes two full 10,000-row active shards, one partial active shard, one full
2,500-row passive shard and one partial passive shard. Parity checks cover stored text/token rows,
schema, shard offsets and rejection/exhaustion metadata. The long fixture contains distinct seeded
multilingual word sequences; neither fixture is representative evidence for every production source.

Complete saved reports:

- [Four-way pipeline](../tools/data_preparation/results/pipeline_20260917.json).
- [Four-way truncation](../tools/data_preparation/results/truncation_20260917.json).
- [Four-way counting](../tools/data_preparation/results/count_native_comparison_20260917.json).
- [Final production counting](../tools/data_preparation/results/count_production_20260917.json).

## Separate unresolved server stall

The supplied 02:30:40–02:30:56 sample showed approximately 100% system CPU and almost no user CPU.
A later profiler launch reportedly waited about a minute and began when downloads resumed. That
regime is not explained by the measured tokenizer costs. Profiler startup versus target inspection
must be distinguished, and a delayed profile can miss the offending interval entirely.

The empty thread snapshots are later (02:32:29 onward), so they do not identify the earlier wait.
NFS counters are mount-wide, the initial nfsiostat report is historical, and the huge backlog value
remains unattributed. These improvements must not be presented as a fix for that kernel-heavy stall.

## Download-specific batching opportunities

The HTTP layer reads blocks, pretrain tokenization batches up to 256 rows, and publication writes
whole shards. The remaining per-row work is between these boundaries:

- `_json_array_batches`, `_json_lines_batches`, and `_json_gz_batches` currently yield singleton
  lists. This preserves immediate early stopping but adds generator/projection overhead per row.
- Parquet `row_group_batches` decodes up to 1,000 rows and converts them with `to_pylist`;
  `read_row_group` flattens that batch before `_parquet_rows` routes and copies individual rows.
- `_fetch` converts, updates offsets, checks targets, and calls `_TokenStep.add` per row, rebuilding
  token batches. `_store` also appends individually to an already-buffered shard writer.

Preserving bounded batches through decoding, routing and conversion is a plausible next experiment.
It must retain exact instruction-target stopping, error timing, grouped-language routing, row-group
completion and per-row durable offsets. Simply changing JSON readers to eagerly parse 1,000 rows
could process errors or fetch data beyond an intended early stop. No batching rewrite is included
in this change; the new `input_next`, `arrow_to_python` and queue timings help prioritize it.

## Validation

- Full offline preparation, tooling and chat suite: 1,064 passed, one optional Rust test skipped.
- Optional native adapter tests and replay tests, with the built adapter enabled: four passed.
- Final focused truncation/tokenizer, diagnostic and cleanup checks: 74 passed.
- Repository Ruff and scoped strict mypy (`data_preparation tools/data_preparation`) passed.
- Whole-repository basedpyright reports four pre-existing errors in `storage/test_manifest.py`
  and `model/test_cuda_paths.py`; both reproduced in an untouched HEAD snapshot.
- `mypy .` is blocked by duplicate modules in local ignored `audits/` snapshots. The same
  failure reproduces with baseline code and those existing snapshots; baseline without the local
  snapshots passes. Unrelated files were left untouched.

Server-specific tokenizer parity, NFS throughput, and the 32-CPU source mix remain unmeasured.
