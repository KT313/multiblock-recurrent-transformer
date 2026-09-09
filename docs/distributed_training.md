# Distributed training (one machine, several GPUs)

`backend: ddp` trains on every GPU of one machine with one process per GPU under torchrun. Several machines are out
of scope.

```bash
make training-ddp config/<run>.yaml GPUS=8
# the same without make
uv run torchrun --standalone --nproc_per_node=8 training/train.py --config config/<run>.yaml
```

The run config must say `backend: ddp`. The single-device backend refuses to start under torchrun (it would train
one independent copy of the run per GPU), and the DDP backend refuses to start without torchrun's environment.

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
- **Same recurrence depth on every rank.** The depth sampler is seeded by the optimizer step, so every rank runs the
  same number of iterations and none waits for a deeper one. Only the random latent state differs per rank (the
  global RNG is seeded with `seed + rank`); DDP broadcasts rank 0's parameters at start, so the model init is shared.
- **Validation is sharded.** Each rank scores its share of the validation rows; the per-depth losses are the mean of
  the per-rank means, the per-source losses are summed over the ranks.
- **Rank 0 writes and logs.** The run lock, `run_config.json`, `model_config.json`, checkpoints, samples, benchmarks,
  the export, `train.log`, wandb and the dashboard belong to rank 0. The other ranks print WARNING and above with a
  `[rank N]` prefix; `torchrun --redirects 3 --local-ranks-filter 0` silences them entirely.
- **Stopping.** Ctrl-C (torchrun forwards it as SIGTERM) sets the stop flag on every rank; the flag is combined over
  the ranks after each step, so all ranks save the same checkpoint and stop together.

## Resuming

A checkpoint stores one RNG state per rank and the number of ranks. Resume with the same number of GPUs; a checkpoint
written with another count is refused. The backend name itself may change between `single_device` and a one-rank
`ddp` run.

## Is the data reader keeping up?

Rank 0 tokenizes for every GPU. `data/wait_seconds` and `data/wait_fraction` (wandb, `train.log`) report the time
rank 0 blocked waiting for a tokenizer worker per log interval; above five percent of the training time a warning
names the slowest source. A wait that persists beyond a source's first batch and its epoch restarts means the
tokenizer workers are the bottleneck.

## Timeouts

The process group's collective timeout is four hours (`DDP_TIMEOUT` in `training/backend/ddp.py`): while rank 0
builds a missing dataset (`auto_prepare`) or runs the lm-eval benchmarks, the other ranks wait at a barrier, and that
wait is legitimate. A rank that dies ends the whole launch through torchrun regardless.

## Testing without GPUs

The DDP backend runs on the CPU with gloo, which is how the multi-rank loop is tested (`training/backend/test_ddp.py`
in-process at world size 1, where it must reproduce the golden run bit for bit; `training/test_distributed.py`
launches two ranks through torchrun). What the CPU path does not cover: NCCL, bf16 autocast under DDP,
`torch.compile` with the DDP wrapper, and throughput.
