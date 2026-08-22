# Pretraining Data Preparation Scripts ⚡ PARALLEL OPTIMIZED

This directory contains two scripts for preparing pretraining datasets for depth-recurrent language models (300M-1B parameters).

## Overview

The data preparation is split into two phases:

1. **Download** (`download_pretraining_data.py`) - **PARALLEL OPTIMIZED** downloads with multiprocessing filtering
2. **Process** (`process_pretraining_data.py`) - Applies comprehensive preprocessing and creates training mixtures

**NEW: Parallel Architecture (3-4x faster!)**
- Downloads 2-3 datasets concurrently (I/O bound, network limited)
- Each download: Fast streaming with NO quality filtering
- After download: Multiprocessing filtering using all CPU cores
- Raw data kept in `raw_unfiltered/` for re-filtering experiments

**Performance:**
- Old serial version: 12+ hours
- New parallel version: 3-5 hours
- Speed improvement: 3-4x faster

This architecture allows you to:
- Download once, re-filter multiple times with different parameters
- Resume interrupted downloads
- Parallel downloads maximize network throughput
- Multiprocessing filtering maximizes CPU utilization
- Keep raw data for experimentation

## Quick Start

### Step 1: Download & Filter Data ⚡ (~3-5 hours, ~250GB) - PARALLEL OPTIMIZED

```bash
# Dry run to see the plan
python download_pretraining_data.py \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --model_size 300M \
    --dry_run

# Download all datasets for 300M model (PARALLEL - 3 concurrent downloads)
python download_pretraining_data.py \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --cache_dir /path/to/fast_storage/.cache \
    --model_size 300M \
    --stages both \
    --parallel_downloads 3 \
    --num_workers 8

# Or download only Stage 1 (for testing)
python download_pretraining_data.py \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --stages stage1 \
    --model_size 300M \
    --parallel_downloads 2
```

**What happens:**
1. **Phase 1**: Fast parallel downloads (3 concurrent, no filtering) - ~1-2 hours
2. **Phase 2**: Multiprocessing filtering (8 workers per dataset) - ~2-3 hours
3. Raw data saved to `raw_unfiltered/`, filtered data to `raw_datasets/`

**Output structure:**
```
pretraining_300m/
└── raw_datasets/
    ├── raw_unfiltered/          # NEW: Raw data (kept for re-filtering)
    │   ├── stage1/
    │   │   ├── fineweb_edu/     # Raw parquet files (raw-*.parquet)
    │   │   ├── wikipedia/
    │   │   └── ...
    │   └── stage2/
    │       ├── fineweb_edu/
    │       └── ...
    ├── stage1/                  # Filtered, ready for process script
    │   ├── fineweb_edu/         # Filtered parquet files (data-*.parquet)
    │   ├── wikipedia/
    │   ├── books_gutenberg/
    │   ├── stack_v2/
    │   ├── peso/
    │   ├── arxiv/
    │   └── openwebmath/
    ├── stage2/                  # Filtered, ready for process script
    │   ├── fineweb_edu_filtered/
    │   ├── stack_v2_filtered/
    │   ├── openwebmath/
    │   ├── tinygsm/
    │   ├── algebraic_stack/
    │   ├── gsm8k_train/
    │   ├── peso/
    │   └── arxiv/
    └── download_metadata.json
```

### Step 2: Process Data (~12-24 hours, ~150GB output)

```bash
# Dry run to see the plan
python process_pretraining_data.py \
    --input_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m/processed \
    --model_size 300M \
    --dry_run

# Full processing with all preprocessing steps
python process_pretraining_data.py \
    --input_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m/processed \
    --model_size 300M \
    --stages both \
    --num_workers 8

# Quick processing (skip slow fuzzy dedup for testing)
python process_pretraining_data.py \
    --input_dir /path/to/shared_storage/recpre/datasets/pretraining_300m \
    --output_dir /path/to/shared_storage/recpre/datasets/pretraining_300m/processed \
    --model_size 300M \
    --stages stage1 \
    --skip_fuzzy_dedup \
    --skip_decontamination
```

**Output structure:**
```
pretraining_300m/processed/
├── stage1/
│   ├── data-00000.parquet
│   ├── data-00001.parquet
│   └── ...
├── stage2/
│   ├── data-00000.parquet
│   ├── data-00001.parquet
│   └── ...
├── preprocessing_stats.json
└── verification_samples.txt
```

### Step 3: Use in Training

Update your training config YAML:

```yaml
# cluster/configs/pretrain_300m_stage1.yaml

data_config:
  train_data:
    - type: hfds
      prefix: pretrain-stage1
      data_dir: /path/to/shared_storage/recpre/datasets/pretraining_300m/processed/stage1

# cluster/configs/pretrain_300m_stage2.yaml

data_config:
  train_data:
    - type: hfds
      prefix: pretrain-stage2
      data_dir: /path/to/shared_storage/recpre/datasets/pretraining_300m/processed/stage2
```

Then train:

```bash
# Train on Stage 1 (80% of total training)
./cluster/scripts/start_train.sh -r large -c pretrain_300m_stage1.yaml

# Then train on Stage 2 (20% of total training)
./cluster/scripts/start_train.sh -r large -c pretrain_300m_stage2.yaml
```

## Token Budgets

### 300M Model (30B total tokens)

**Stage 1 (24B tokens - 80% of training):**
- 65% FineWeb-Edu: 15.6B tokens
- 9% Wikipedia: 2.16B tokens (oversampled 2x)
- 6% Books: 1.44B tokens
- 12% The Stack v2: 2.88B tokens
- 3% peS2o: 0.72B tokens
- 2% arXiv: 0.48B tokens
- 3% OpenWebMath: 0.72B tokens

**Stage 2 (6B tokens - 20% of training, domain upsampling):**
- 35% FineWeb-Edu (score ≥3): 2.1B tokens
- 28% The Stack v2 (stars ≥2): 1.68B tokens
- 22% Math mixture: 1.32B tokens
  - 40% OpenWebMath: 0.528B
  - 30% TinyGSM-MIND: 0.396B
  - 20% Algebraic Stack: 0.264B
  - 10% GSM8K train: 0.132B
- 15% peS2o + arXiv: 0.9B tokens

### 1B Model (100B total tokens)

**Stage 1:** 85B tokens (same proportions, scaled up)
**Stage 2:** 15B tokens (same proportions, scaled up)

## Preprocessing Pipeline

The processing script applies these steps in order:

1. **Exact Deduplication** (~10-15% reduction)
   - MD5 hash-based
   - Fast, removes exact duplicates

2. **Fuzzy Deduplication** (~30-40% reduction)
   - MinHash + LSH with Jaccard threshold 0.8
   - 5-gram based, 256 permutations
   - **Most impactful step** but slow
   - Can be skipped for quick testing

3. **Quality Filtering** (~5-10% reduction)
   - Minimum 3 sentences
   - Max 30% ALL CAPS words
   - Max 30% duplicate 2-grams
   - Max 20% duplicate 3-grams
   - Min 25% alphanumeric characters

4. **PII Removal**
   - Emails → `[EMAIL]`
   - IP addresses → `[IP]`
   - Phone numbers → `[PHONE]`
   - API keys → `[KEY]`

5. **Benchmark Decontamination** (~0.1-1% reduction)
   - Removes documents with >10% 13-gram overlap with test sets
   - Test sets checked:
     - GSM8K, MATH, HumanEval, MBPP
     - ARC-Challenge, HellaSwag, MMLU, WinoGrande
   - **Critical** to prevent benchmark contamination

## Command-Line Options

### download_pretraining_data.py ⚡ PARALLEL OPTIMIZED

```bash
Required:
  --output_dir DIR          Output directory for raw datasets

Options:
  --model_size {300M,1B}    Model size (default: 300M)
  --stages {stage1,stage2,both}  Which stages to download (default: both)
  --cache_dir DIR           HuggingFace cache directory
  --token_buffer FLOAT      Extra tokens to download (default: 0.15 = 15%)

Parallel Options (NEW):
  --parallel_downloads INT  Number of concurrent downloads (default: 3)
  --num_workers INT         Worker processes for filtering (default: 8)

Other:
  --shard_size INT          Examples per shard (default: 10,000)
  --keep_raw                Keep raw unfiltered data (default: True)
  --resume                  Resume interrupted download
  --dry_run                 Show plan without downloading
```

**Performance Tips:**
- `--parallel_downloads 3` is optimal for most networks (100+ Mbps)
- `--num_workers 8` uses 8 CPU cores for filtering (adjust to your CPU)
- Use `--parallel_downloads 2` for slower networks (<50 Mbps)
- Use `--num_workers 16` if you have 16+ CPU cores

### process_pretraining_data.py

```bash
Required:
  --input_dir DIR           Input directory with raw_datasets/
  --output_dir DIR          Output directory for processed datasets

Options:
  --model_size {300M,1B}    Model size (default: 300M)
  --stages {stage1,stage2,both}  Which stages to process (default: both)
  --cache_dir DIR           HuggingFace cache directory
  --num_workers INT         Worker processes (default: 4)

Preprocessing:
  --skip_exact_dedup        Skip exact deduplication
  --skip_fuzzy_dedup        Skip fuzzy deduplication (saves time)
  --skip_quality_filter     Skip quality filtering
  --skip_pii_removal        Skip PII removal
  --skip_decontamination    Skip benchmark decontamination

Fuzzy Dedup Parameters:
  --fuzzy_threshold FLOAT   Jaccard threshold (default: 0.8)
  --minhash_num_perm INT    MinHash permutations (default: 256)

Output:
  --shard_size INT          Examples per shard (default: 10,000)
  --dry_run                 Show plan without processing
```

## Resource Requirements

### Download Script ⚡ PARALLEL OPTIMIZED
- **Time:** 3-5 hours (with 3 parallel downloads + 8 filter workers)
  - Old serial version: 12+ hours
  - **Speed improvement: 3-4x faster**
  - Phase 1 (parallel download): 1-2 hours
  - Phase 2 (multiprocessing filter): 2-3 hours
- **Disk:** ~500GB total (250GB raw + 250GB filtered)
  - Raw kept in `raw_unfiltered/` for re-filtering
  - Filtered data in `raw_datasets/` for processing script
- **Memory:** 8-16GB (for multiprocessing filtering with 8 workers)
  - Download phase: <2GB per concurrent download
  - Filter phase: ~1-2GB per worker process
- **Network:** ~300GB total download (parallelized across 3 streams)
- **CPU:** Uses all cores during filtering (default: 8 workers)

### Processing Script
- **Time:** 12-24 hours (CPU dependent)
- **Disk:** ~150GB processed data (after deduplication)
- **Memory:** 32-64GB RAM (for fuzzy deduplication)
  - Can run with less if `--skip_fuzzy_dedup` is used
- **CPU:** Benefits from 8+ cores

## Troubleshooting

### Download Issues

**Problem:** "Permission denied" writing to cache
**Solution:** Use `--cache_dir /path/to/fast_storage/.cache`

**Problem:** "Connection timeout"
**Solution:** Use `--resume` to continue interrupted download

**Problem:** "Dataset too small"
**Solution:** Check pass rates in metadata.json, adjust filters if needed

### Processing Issues

**Problem:** "Out of memory during fuzzy dedup"
**Solution:**
- Use `--skip_fuzzy_dedup` for testing
- Process Stage 1 and Stage 2 separately
- Increase swap space or use machine with more RAM

**Problem:** "Benchmark datasets won't load"
**Solution:**
- Use `--skip_decontamination` if benchmarks aren't available
- Check `--cache_dir` permissions

**Problem:** "Processing too slow"
**Solution:**
- Increase `--num_workers` (but watch memory usage)
- Skip fuzzy dedup for initial testing
- Process only Stage 1 first

## Data Format

Both scripts save data in **parquet format** with these columns:

- `text` (str): The cleaned document text
- `source` (str): Source dataset name
- `estimated_tokens` (int): Approximate token count (chars/4)

The training dataloader will tokenize on-the-fly using your configured tokenizer.

## Expected Performance

With optimal pretraining on 30B tokens (300M model):

| Benchmark | Expected Range |
|-----------|----------------|
| ARC-E | 58-65% |
| ARC-C | 32-38% |
| HellaSwag | 48-55% |
| MMLU | 28-34% |
| GSM8K | 2-8% |
| HumanEval | 5-12% |
| MBPP | 8-15% |

Performance improves significantly with instruction fine-tuning (use `prepare_flan_mixture.py`).

## Resuming Work

### Resume Download
```bash
python download_pretraining_data.py \
    --output_dir /path/to/output \
    --model_size 300M \
    --resume
```

The script checks for `metadata.json` in each dataset directory and skips completed downloads.

### Re-process with Different Parameters

You can re-run processing with different parameters without re-downloading:

```bash
# First run: full preprocessing
python process_pretraining_data.py --input_dir ... --output_dir .../v1 --stages both

# Second run: skip fuzzy dedup for faster iteration
python process_pretraining_data.py --input_dir ... --output_dir .../v2 --skip_fuzzy_dedup

# Third run: only Stage 1 with different threshold
python process_pretraining_data.py --input_dir ... --output_dir .../v3 --stages stage1 --fuzzy_threshold 0.9
```

## Validation

After processing, check:

1. **preprocessing_stats.json** - Review deduplication rates, filter stats
2. **verification_samples.txt** - Inspect sample documents
3. **Token counts** - Verify they match targets (may be slightly lower due to preprocessing)

Expected preprocessing losses:
- Exact dedup: 10-15%
- Fuzzy dedup: 30-40%
- Quality filter: 5-10%
- Decontamination: <1%
- **Total: 40-50% reduction** from raw to final

This is why we download with a 15% buffer.

## Credits

- FineWeb-Edu: HuggingFace (HuggingFaceFW/fineweb-edu)
- Wikipedia: Wikimedia Foundation
- The Stack v2: BigCode (bigcode/the-stack-v2)
- peS2o: Allen AI (allenai/peS2o)
- OpenWebMath: open-web-math
- TinyGSM-MIND: Aeala
- Algebraic Stack: EleutherAI

## Further Reading

- [FineWeb Paper](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- [RedPajama-Data-v2 Quality Signals](https://github.com/togethercomputer/RedPajama-Data)
- [Deduplication for LLMs](https://arxiv.org/abs/2107.06499)
- [Data Contamination in LLM Benchmarks](https://arxiv.org/abs/2311.01964)
