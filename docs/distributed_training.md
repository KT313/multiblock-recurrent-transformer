# Distributed training (one machine, several GPUs)

`backend: ddp` trains on every GPU of one machine with one process per GPU under torchrun. Several machines are out
of scope.

```bash
make training-ddp config/<run>.yaml            # every visible GPU; GPUS=2 for a subset
# the same without make
uv run torchrun --standalone --nproc_per_node=gpu --shutdown-timeout=1800 training/train.py --config config/<run>.yaml
```

The run config must say `backend: ddp`. The single-device backend refuses to start under torchrun (it would train
one independent copy of the run per GPU), and the DDP backend refuses to start without torchrun's environment.

Optimizer state is replicated by default. Set `optimizer_sharding: zero1` in the top-level run configuration
to distribute it across ranks, with either FP32 or 8-bit ELLISAdam. See [optimizer-state sharding](optimizer_sharding.md)
for supported optimizers, checkpoint restrictions, memory caveats and validation limits.

## What is shared and what is per rank

Every rank runs the same `train()`; the differences are these.

- **Steps and budgets are world totals.** `micro_batches_per_step` is split evenly over the ranks
  (`Settings.micro_batches_per_rank`), so one optimizer step trains `micro_batches_per_step x tokens_per_micro_batch`
  tokens whatever the GPU count, and every `*_steps` / `*_interval` setting, stage boundary and checkpoint name means
  the same thing on 1 or 8 GPUs. `eval_iters` is split the same way. Both must be multiples of the GPU count.
- **One data reader.** Rank 0 owns the data stream (`BatchStream`: the per-source readers, the token-share picks,
  the packing pool) and scatters each rank its pack for every micro-batch (`RankBatches` in `training/step.py`).
  The other ranks have no train loaders. The checkpoint therefore holds one stream state; a resume continues the
  stream exactly as on one GPU.
- **Fresh recurrence depths per local microbatch, shared across ranks.** The sampler deterministically combines
  the optimizer step, zero-based local microbatch index and core-block index, excluding rank. Each local microbatch
  gets a fresh draw for every core; corresponding microbatches on all ranks get the same depth vector. For example,
  64 global microbatches on 8 GPUs means 8 local microbatches per GPU: all ranks might use depths `8, 4, 6` at local
  index 0 and `2, 7, 7` at index 1. Fresh draws can coincide. This balances recurrence work; other costs can still
  cause timing differences. Latent values remain different per token and rank (the global RNG uses `seed + rank`);
  DDP broadcasts rank 0's parameters at start, so the model initialization is shared.
- **Validation is sharded.** Each rank scores its share of the validation rows; the per-depth losses are the mean of
  the per-rank means, the per-source losses are summed over the ranks.
- **Scheduled inference is distributed.** Fixed sample batches and benchmark inference jobs use the resident model replicas; rank 0 retains official lm-eval aggregation and publication. See [distributed evaluation](distributed_evaluation.md) for batching, prompting, and qualification.
- **Rank 0 writes and logs.** The run lock, `run_config.json`, `model_config.json`, accepted resume history, checkpoints, samples, benchmarks,
  the export, `train.log`, wandb and the dashboard belong to rank 0. The other ranks print WARNING and above with a
  `[rank N]` prefix; `torchrun --redirects 3 --local-ranks-filter 0` silences them entirely.
- **Stopping.** Ctrl-C reaches torchrun (the ranks run in their own session) and torchrun forwards the signal to every
  rank; each rank sets its stop flag, the flags are combined after the step, so all ranks save the same checkpoint and
  stop together. torchrun then waits `--shutdown-timeout` seconds (its default is 30) before it kills the ranks, and
  a kill inside the last step or the checkpoint write loses the stop checkpoint. `make training-ddp` passes
  `--shutdown-timeout=1800` (`SHUTDOWN_TIMEOUT=n` overrides); a launch by hand needs the flag too. A second Ctrl-C
  makes torchrun signal the ranks again, which should end them right away. Ctrl-C during an `auto_prepare` build stops
  at the next shard, which can take minutes, so the window is generous on purpose. torchrun itself always exits 1 with
  a `SignalException` traceback after a Ctrl-C, even when every rank stopped cleanly; the confirmation of a clean stop
  is rank 0's summary ("stopped on request after step N") and its checkpoint.

## Concurrent training runs on shared storage

Use distinct `run_name` values and `auto_prepare: false` in each run. The exclusive output lock lives at
`<out_dir>/<run_name>/.train.lock`; the shared dataset lock lives at `<dataset_dir>/.build.lock`.
Multiple read-only training runs may share a prepared dataset. Preparation requires an exclusive dataset lock,
so it fails while any training reader is active, and training fails while preparation is active.

`auto_prepare: true` retains exclusive dataset ownership for the entire run, even when the dataset is already
prepared. It can still build missing data. With `auto_prepare: false`, missing data fails with a preparation command;
training never upgrades its lock or prepares under shared access.

Rank zero holds each run's locks through reader cleanup and the normal all-rank completion barrier. Fatal rank
loss still relies on launcher teardown of peers. Locks are nonblocking and advisory; all processes must use the
same dataset root and a filesystem that enforces locking across nodes. Test that behavior on the actual mount
before relying on concurrent jobs. Do not delete or replace lock files. Dataset lock files contain no holder list.
The old output-root lock is no longer used; avoid mixing old and new trainers targeting the same run directory.

## Resuming

A checkpoint stores one RNG state per rank and the number of ranks. Resume with the same number of GPUs; a checkpoint
written with another count is refused. The backend name itself may change between `single_device` and a one-rank
`ddp` run.

Depth sampling needs no additional checkpoint state: checkpoints occur between optimizer steps, and the restored
step starts at local microbatch index 0. Resuming with the same code and settings reproduces the depth schedule.
The per-microbatch sampler uses a new versioned seed mapping, including for index 0. Checkpoints from the earlier
per-optimizer-step sampler remain loadable, but their subsequent sampled depths and training trajectory change.
Evaluation depths, explicit depth overrides and latent RNG consumption are unchanged.

Raw `RecurrentGPT` callers own `step` and `micro_batch_index`; both default to 0. An accumulation loop outside
the native trainer must set the local index before each forward. Replaying a forward uses the same context.
The Hugging Face wrapper retains its separate per-forward depth counter; this change does not add HF resume state.

## Is the data reader keeping up?

Rank 0 tokenizes for every GPU. `data/wait_seconds` and `data/wait_fraction` (wandb, `train.log`) report the time
rank 0 blocked waiting for a tokenizer worker per log interval; above five percent of the training time a warning
names the slowest source. Worker start-ups (a source's first batch, an epoch restart) are not counted, so a warning
means the tokenizer workers are the bottleneck.

## Timeouts

The process group's collective timeout is four hours (`DDP_TIMEOUT` in `training/backend/ddp.py`): while rank 0
builds a missing dataset (`auto_prepare`), prepares benchmark tasks, or aggregates metrics, the other ranks can wait
at a shared phase boundary. Benchmark inference itself is distributed. These waits are legitimate. A rank that dies ends the whole launch through torchrun regardless.

## Testing without GPUs

The DDP backend runs on the CPU with gloo, which is how the multi-rank loop is tested (`training/backend/test_ddp.py`
in-process at world size 1, where it must reproduce the golden run bit for bit; `training/test_distributed.py`
launches two ranks through torchrun). What the CPU path does not cover: NCCL, bf16 autocast under DDP,
`torch.compile` with the DDP wrapper, and throughput.

Configuration sidecars and accepted resume records are described in [resume configuration history](resume_configuration_history.md).
## Fatal worker failures

The `torchrun` DDP CLI reports an ordinary fatal error directly to stderr, including its rank and
original traceback, then exits the worker with code 1 before run-level logger, reader and process-group
cleanup. It does not attempt a final distributed checkpoint, including after a non-finite training loss.
Resume from a regular checkpoint. `torchrun` owns termination of the remaining ranks; PyTorch loader
workers detect parent exit. A peer unable to honor SIGTERM may require the launcher's forced-kill grace.

This policy is opt-in at the CLI boundary (`backend: ddp` with `TORCHELASTIC_RUN_ID` present).
Library `train()` calls and single-device runs retain ordinary exception cleanup. Successful runs,
cooperative stop requests, lock refusals and build cancellation retain their existing behavior.
No validation collectives, batches, DDP synchronization settings or numerical policies change.
The boundary cannot intercept a native call that hangs without raising, or guarantee error output through
a blocked stderr destination. Such hangs still require external job supervision. Fatal exits can leave
incomplete disposable output; only previously completed checkpoints should be used for recovery.
