# Scheduled samples and benchmarks across training GPUs

Scheduled inference uses the model replica already resident on each training rank. Rank 0 writes artifacts and runs lm-eval's official task construction, filtering, and metric aggregation; other ranks execute inference jobs through the existing process group. No additional model copies or process groups are created.

## Configuration

```yaml
sample_batch_size: 1           # default 8; choose 1 to distribute the 11 default prompts over eight GPUs
sample_max_new_tokens: 64
sample_use_cache: true
sample_temperature: [0.0, 0.7] # scalar remains supported; repeat all recurrences per temperature
benchmark_batch_size: 8        # positive, fixed forward batch size
benchmark_limit: 8             # per leaf task; omit for a full evaluation
benchmark_apply_chat_template: false
```

The usual sample/benchmark interval and progress triggers still apply. `sample_batch_size` may change on resume; old checkpoints that omit it remain compatible. Changing sample batch size changes the stochastic latent draws and can change completions. Training samples retain seed 0. Distribution only changes which rank executes each fixed batch. With a temperature list, output is ordered by temperature, recurrence, then prompt. Every combination uses the same batch-offset seeds as its corresponding standalone scalar run; adding a temperature does not change existing completions. Each JSONL row records its actual temperature in `decoding.temperature`. Empty lists, negative values, and nonfinite temperatures are rejected.

Benchmark prompting stays explicit. Continuations use BOS plus literal text. Chat uses the checkpoint-owned role-token formatter, including literal body encoding, few-shot history, and an unfinished assistant prefix. Chat inputs that exceed context are rejected rather than silently losing role structure. System messages remain unsupported: stock MMLU descriptions currently require plain prompting. EOS token IDs stop generation; the ordinary-token spelling `</s>` does not, unless a task explicitly requests it as a text stop.

## Benchmark reproducibility

Training benchmarks use `fixed_document_jobs_v1` on both one and multiple ranks. Requests from one task/document stay together; deterministic groups target eight requests, with at most 128 requests per document. Job seeds derive from the configured seed, recurrence index, public-method call index, method, and job index. Identical text/repeats remain distinct occurrences. Each job restores the surrounding Python, NumPy, Torch CPU, and local-CUDA RNG streams.

The standalone `evaluation/evaluate.py` CLI uses whole-list HFLM batching, which differs from scheduled training's batching/seeding protocol. Compare rank counts with identical jobs and batch sizes. Different hardware/precision need not be bitwise identical. HFLM's default logits reuse allows compatible candidates to share one stochastic prefix forward. Response/request caches are disabled; no responses or KV state persist between optimizer updates.

Result JSON contains protocol, worker job/request counts, execution time, padded token counters, process-lifetime peak CUDA allocation (without resetting training counters), seed previews and rolling request digests, local HFLM settings, and evaluator/inference world sizes. Transport is bounded to 1 MiB per benchmark job/reply and 32 MiB per collective payload. The official evaluator still holds its task requests/metric state on the host. Bootstrap uses the harness's supported serial mode with 100 iterations; this does not disable standard errors.

## Failures, stopping, and artifacts

Requested inference failures propagate. Under the training CLI, the fatal-worker handler exits immediately and torchrun terminates peers. Library callers receive exceptions and own peer termination. No recovery collective or emergency checkpoint runs after an arbitrary worker failure.

Cooperative stops are checked at shared round boundaries; a slow generation batch or rank-0 task-loading/aggregation phase can delay a stop. Incomplete evaluations publish no partial success. Complete outputs replace the previous file atomically on rank 0. A later logging failure does not undo a completed file. Training's process-group timeout is four hours.

The direct `evaluation/evaluate.py` CLI is single-device and rejects a multi-rank launch. Use the checkpoint check below to run the scheduled training benchmark path immediately, including multi-GPU inference.

## Run benchmarks immediately from a training checkpoint

Run from the repository root, inside your GPU allocation. For a training config with `backend: ddp`:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=8 \
  evaluation/benchmark_checkpoint.py --config config/final_multiblock_1-4B_100B_tokens.yaml
```

Distributed execution is supported on one node; multi-node execution is outside the supported scope. For a config with `backend: single_device`, use `uv run --no-sync python evaluation/benchmark_checkpoint.py --config <run.yaml>`. To check a DDP config on one GPU without changing its settings, use `torchrun --standalone --nproc_per_node=1`.

This command selects the most recently written regular checkpoint under `<out_dir>/<run_name>/checkpoints`, using training's selection rules. It ignores the `resume` switch, `resume_checkpoint_path`, and benchmark schedule triggers; use `--checkpoint /path/to/file.pth` to select a specific checkpoint. It runs the configured tasks, recurrences, batch size, few-shot count, limit, seed and chat-template setting immediately, through the same function used by scheduled training. All normal training CLI overrides work. For a quick check, append `--benchmark_limit 8`; omit that override to use the config unchanged.

Weights and architecture come from the checkpoint, checked against the current architecture config. The current config controls precision and custom kernels. The checkpoint-owned tokenizer is validated for identity, vocabulary and template parity; `--tokenizer_dir /relocated/tokenizer` supports moving the matching artifact to another server. Legacy checkpoints use the current config's dataset tokenizer location. Paths in the config resolve from the working directory, just as in training.

The command loads model weights on each rank but creates no optimizer, data loader, DDP model wrapper or training compile graph. Scheduled inference also uses the unwrapped, uncompiled replica. Checkpoints are memory-mapped to avoid eagerly loading unused optimizer storage on every rank. No training steps, data preparation, samples or checkpoint saves run. The inference rank count may differ from the checkpoint's training rank count, including ZeRO-1 checkpoints.

Results use the normal benchmark JSON schema and are saved in a fresh `<out_dir>/<run_name>/benchmark_checks/step-XXXXXXXX-*/step-XXXXXXXX.json`. Existing scheduled results and training logs are preserved. `--output_dir` chooses a new directory, which must not already exist. Ctrl-C/SIGTERM requests a stop at the next shared benchmark boundary; incomplete work publishes no results. Fatal failures under torchrun use the same worker-exit policy as training. This verifies checkpoint inference, not resuming an optimizer or a complete training step. Benchmark datasets still need network access or an existing local cache.

## Testing

CPU tests are offline and should run serially:

```bash
uv run --no-sync pytest evaluation -n 0 -m 'not gpu'
uv run --no-sync pytest training/test_run.py -n 0 -k 'samples or benchmarks'
```

Role-token tests use an already downloaded tokenizer; set `MBRT_TEST_BASE_TOKENIZER` if it is outside this checkout. They never download it.

On a machine with free GPUs, test the tiny model/harness on one GPU, then NCCL execution on two/eight GPUs:

```bash
uv run --no-sync pytest evaluation/test_qualification.py -n 0 -m gpu
RUN_DISTRIBUTED_GPU_TESTS=1 uv run --no-sync pytest evaluation/test_distributed.py -n 0 -m gpu
```

These tests do not establish research-checkpoint accuracy. Use the [checkpoint command](#run-benchmarks-immediately-from-a-training-checkpoint)
with representative weights and production precision/kernels. Run bounded plain and chat evaluations with compatible tasks,
including generation and the intended few-shot settings. Keep outputs separate and check context limits before using
task-default few-shot counts. Single-node tests do not validate multi-node execution.
