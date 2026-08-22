import os
from pathlib import Path
from transformers import AutoTokenizer

BASE = Path("/path/to/shared_storage/recpre")
OUT = BASE / "artifacts" / "tokenizer_llama32k"
OUT.mkdir(parents=True, exist_ok=True)

name = "hf-internal-testing/llama-tokenizer"  # small, 32k, fine for a smoke run
tok = AutoTokenizer.from_pretrained(name, use_fast=True)
tok.save_pretrained(OUT)

print(f"Saved tokenizer to: {OUT}")

