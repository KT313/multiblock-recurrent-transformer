# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fixed loader batching and file-descriptor limits, independent of run settings."""

# The worker batch of the unpadded train loaders: rows tokenized per worker batch and worker batches kept ready ahead
# (torch's `prefetch_factor`). Their product is how many tokenized rows a source has waiting when `BatchStream`
# refills its packing pool (`POOL_TOKEN_FACTOR` pack lengths of documents, drawn between two micro-batches); 64 x 4 =
# 256 rows covers a refill from one source, so the pull waits for no tokenization (with the old 4 x 4 = 16 the rest was
# tokenized while the GPU idled). Neither value touches the sample order or the numerics: the one worker
# (`TRAIN_LOADER_NUM_WORKERS`) walks its range in order and the stream concatenates its batches, `WorkerBatch.rows_read`
# counts the rows of any batch size and the samples pulled ahead travel in the checkpoint (`BatchStream.state_dict`).
# Fixed here, not settings: a `Settings` field is compared on resume, and this one may differ freely.
# File descriptors: torch shares a worker's tensors with the training process through one descriptor per tensor
# (its `file_descriptor` strategy), held until the batch has been received, so the batches in flight cost up to
# TRAIN_LOADER_BATCH_ROWS x TRAIN_LOADER_PREFETCH_FACTOR x 2 tensors per source of the `ulimit -n` budget. A worker
# that runs out drops the batch with a traceback on stderr and the loader skips it; `build_run_dataloaders` lifts the
# soft limit to the hard one (`raise_open_file_limit`) and `RunDataloaders` refuses an epoch that delivered fewer
# rows than the range holds, so the loss is an error and never silent.
TRAIN_LOADER_BATCH_ROWS = 64
TRAIN_LOADER_PREFETCH_FACTOR = 4
UNLIMITED_OPEN_FILES = 1 << 20  # the soft open-file limit `raise_open_file_limit` sets under an unlimited hard limit
