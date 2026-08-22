from datasets import load_dataset
from pathlib import Path
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

set_verbosity_info()
enable_progress_bar()

BASE = Path("/path/to/shared_storage/recpre/datasets/fineweb-edu")
(BASE / "train").mkdir(parents=True, exist_ok=True)
(BASE / "validation").mkdir(parents=True, exist_ok=True)

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT") # only 'train' exists

# N_TOTAL = 100_000
N_VAL = 50_000
SEED = 42

train = ds["train"]
# if train.num_rows < N_TOTAL:
#     raise ValueError(f"fineweb-edu has only {train.num_rows:,} rows locally; need {N_TOTAL:,}.")

# 1) take a deterministic subset
subset = train.shuffle(seed=SEED) # .select(range(N_TOTAL))

# 2) split 90k/10k (train/validation)
splits = subset.train_test_split(test_size=N_VAL, seed=SEED)  # returns 'train' and 'test'
splits["validation"] = splits.pop("test")  # rename 'test' -> 'validation'

# 3) save to disk
splits["train"].save_to_disk(str(BASE / "train"))
splits["validation"].save_to_disk(str(BASE / "validation"))

print(f"Saved HF datasets to: {BASE}")
print({k: v.num_rows for k, v in splits.items()})

