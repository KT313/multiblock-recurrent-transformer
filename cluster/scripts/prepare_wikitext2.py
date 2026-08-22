from datasets import load_dataset
from pathlib import Path
from datasets.utils.logging import set_verbosity_info, enable_progress_bar

set_verbosity_info()
enable_progress_bar()

BASE = Path("/path/to/shared_storage/recpre/datasets/wikitext2")
(BASE / "train").mkdir(parents=True, exist_ok=True)
(BASE / "validation").mkdir(parents=True, exist_ok=True)

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
# keep it small for a smoke test; increase later
ds["train"].shuffle(seed=42).save_to_disk(str(BASE / "train"))
ds["validation"].save_to_disk(str(BASE / "validation"))

print(f"Saved HF datasets to: {BASE}")
print({k: v.num_rows for k, v in ds.items()})

