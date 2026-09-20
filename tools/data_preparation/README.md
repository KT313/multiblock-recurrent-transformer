# Preparation performance tools

Dataset preparation uses the installed Python `tokenizers` package. Rust compilation is needed only
for the optional comparison experiment, not for dataset preparation.

## Capture stage timings

Use `--debug` for live five-second overviews, `--debug 2` for a different interval, and
`--debug-file PATH` for a readable copy of the main/worker reports. See
[the debug guide](../../docs/download_debug.md) for timing semantics and interpreting quiet network periods.

## Reproduce a local comparison

Use original, untruncated JSONL input. Each line contains `text` and optional `source`, `group`, and
`passive` fields. Sources sharing a group share a real token worker and separate shard writers.
The replay exercises `_fetch`, `_TokenStep`, `_TokenWorker`, `RawFolder`, and atomic Parquet/manifest
publication. It intentionally bypasses network retrieval and decoder input; it is not an NFS or
internet benchmark. It currently benchmarks pretrain rows; instruction behavior is regression-tested.

Generate synthetic fixtures:

```bash
uv run --no-sync python -m tools.data_preparation.make_fixture /tmp/preparer-mixed.jsonl --case mixed
uv run --no-sync python -m tools.data_preparation.make_fixture /tmp/preparer-long.jsonl --case long
```

Compare the benchmark variants in fresh processes with identical artifacts:

```bash
uv run --no-sync python -m tools.data_preparation.benchmark_download --tokenizer dataset/tokenizers/llama-32k --fixture /tmp/preparer-mixed.jsonl --output-root /tmp/preparer-replay --mode pipeline --threads 4 --jobs 1 --cap 256 --repeats 3
uv run --no-sync python -m tools.data_preparation.benchmark_download --tokenizer dataset/tokenizers/llama-32k --fixture /tmp/preparer-long.jsonl --output-root /tmp/preparer-replay --mode truncate --threads 4 --cap 16384 --repeats 3
uv run --no-sync python -m tools.data_preparation.benchmark_download --tokenizer dataset/tokenizers/llama-32k --fixture /tmp/preparer-long.jsonl --output-root /tmp/preparer-replay --mode count --threads 4 --repeats 3
```

The tool creates a unique output directory per comparison and per trial. It refuses overlap with
the fixture, tokenizer or local `dataset/`, and never repairs/reuses a production dataset. Inputs
default to at most 64 MiB/100,000 rows. Each child is limited to 300 seconds, monitored for 4 GiB RSS
and 1 GiB output; monitoring limits can overshoot between polls. All outputs are retained for review.
The replay uses threads, not child process pools. Run CPU benchmarks without competing workloads.

Reports include all trials, semantic results, versions, affinity, cgroup limits/counters, tokenizer
and fixture digests, tool-source hashes and the native binary digest when applicable. Compare each
child's before/after `cpu.stat` for throttling. The explicit `--threads` value configures Rayon;
the tool does not change Slurm allocation or reserve cores. Median ratios are not confidence intervals.

For pipeline mode, timing includes tokenizer setup per group, worker drain/join, full and final
partial shards, and manifest publication. For truncate/count mode, hashing is outside the timed
interval. Warm-up happens before timing. Peak RSS includes setup and warm-up. Output digest checks
are per source, preserving row order and durable progress independently of job completion order.

## Compare the existing native APIs and the experimental Rust adapter

`--variants original candidate builtin` needs no compilation. `builtin` uses
`Tokenizer.enable_truncation(cap + 1)` for the cut boundary, retaining the exact Python prefix/recount
loop, and `encode_batch_fast` for count-only work. Ordinary token-sequence truncation followed by
decode is not an equivalent replacement for preserving original text prefixes.

The optional `rust` variant moves that same prefix/recount loop into Rust. Encoding, normalization,
offsets and parallelism all come from the existing `tokenizers` crate; it implements no new tokenizer.
Only compact `(text, count)` results cross back into Python. It releases the GIL during work and
includes boundary conversion costs in measurements. This is benchmark-only tooling, pinned to
tokenizers 0.23.1 to match the Python comparison. It loads a second native library; build/dependency
differences mean the experiment is not a pure language-only comparison.

```bash
PYO3_PYTHON="$PWD/.venv/bin/python" cargo build --locked --release -j 2 --manifest-path tools/data_preparation/native/Cargo.toml --target-dir /tmp/preparer-native-target
PREPARER_NATIVE_LIBRARY=/tmp/preparer-native-target/release/libpreparer_native.so uv run --no-sync pytest tools/data_preparation/test_native_backends.py -n 0
uv run --no-sync python -m tools.data_preparation.benchmark_download --tokenizer dataset/tokenizers/llama-32k --fixture /tmp/preparer-long.jsonl --output-root /tmp/preparer-replay --mode truncate --threads 4 --cap 16384 --repeats 3 --variants original candidate builtin rust --native-library /tmp/preparer-native-target/release/libpreparer_native.so
```

Use the deployment tokenizer, representative inputs, and target filesystem when evaluating performance.
Synthetic local comparisons do not establish server throughput or tokenizer parity.

## Final cross-source admission benchmark

`benchmark_global.py` compares original per-source recovery, session reuse, and ordered
hashing with 2/4 workers on deterministic offline candidates. It creates a new disposable
output directory for each run, retains comparison artifacts, and bounds each owned trial
by time, input size and sampled process-tree RSS. It never uses the live dataset directory.

```bash
uv run --no-sync python tools/data_preparation/benchmark_global.py \
  --output-root /tmp/global-benchmark-new \
  --rows 24000 --text-chars 4096 --trials 3
```

Use `--baseline-repo /path/to/original-archive` for an old-code comparison; otherwise
baseline uses the current implementation with a fresh session per source.
