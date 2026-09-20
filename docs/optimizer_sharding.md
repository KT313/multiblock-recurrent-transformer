# Optimizer-state sharding

Enable ZeRO stage 1 in the **top-level training configuration**:

```yaml
backend: ddp
optimizer_sharding: zero1   # default: none
optimizer: ELLISAdam8bit    # also ELLISAdam (FP32 moments) or AdamW
```

Launch with the existing DDP launcher, for example:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 \
  training/train.py --config config/your_run.yaml
```

`zero1` is rejected unless DDP is selected. One-rank DDP works but saves no optimizer memory.
Model parameters remain FP32, and BF16 mixed forward computation remains supported. Selecting ELLISAdam8bit
retains its usual block-quantized moments and FP32 embedding/small-tensor exceptions; sharding does not change
the optimizer mathematics or precision policy. No optimizer is silently substituted.

## What it distributes

PyTorch `ZeroRedundancyOptimizer` assigns whole parameters to ranks. Each rank updates its owned parameters
and broadcasts the resulting weights. Weights and gradients remain replicated; only optimizer moments are
distributed. Accumulation, supervised-token normalization, clipping and recurrence are unchanged.

The largest state shard determines the limiting GPU. Shards need not have equal byte sizes, especially when
the embedding retains FP32 moments. Memory freed is available for larger packed microbatches, but throughput
and the maximum fitting batch must be measured on the target hardware.

Gradient and parameter metrics keep their existing global ordering. Moment-derived metrics are calculated
by their owner and merged as scalars; `avg_RMS` is weighted by the number of contributing parameter tensors.
The rank-zero representation probe still runs after the completed update. No full moments are gathered for logging.

## Checkpoints and resume

Start a fresh run when enabling or disabling sharding or changing optimizer precision. Resume requires the
same sharding mode, optimizer/group settings and number of ranks. Older checkpoints implicitly use `none`.
`allow_settings_change` does not permit changing the sharding mode. Existing unsharded runs remain compatible.

All ranks participate in optimizer consolidation before rank zero writes the usual atomic single-file
checkpoint. Consolidated snapshots are released after publication. Restore loads only owner-local live
moments and suppresses the unused outer-wrapper moment copy. Empty/lazy state is preserved without dummy updates.

**Budget memory for saving and loading as well as training.** Consolidation uses GPU serialization/receive
buffers on all ranks, and rank zero holds the full optimizer state on CPU. Each rank currently reads the full
checkpoint on CPU during resume. This feature does not provide sharded checkpoint I/O or recovery after a rank
is lost during a collective; retain the last complete checkpoint.

## Testing

Run the offline two-rank CPU tests with local multiprocessing sockets permitted:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync --offline pytest training/test_optimizer_sharding.py -n 0
```

CPU tests do not establish NCCL correctness, compiled CUDA optimizer parity, large-model memory peaks or speed.
Test those on the target hardware, including save/resume memory use. Re-run the tests when upgrading PyTorch:
the wrapper depends on its state-load order and consolidation cache lifecycle.
