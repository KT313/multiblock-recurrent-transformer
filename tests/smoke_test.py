#!/usr/bin/env python
"""Lightweight regression guard for the repo cleanup.

Run from the repo root:
    uv run python tests/smoke_test.py                 # check against golden baseline
    uv run python tests/smoke_test.py --update-golden # (re)record the baseline

Checks, in order:
  1. every recpre module imports (known-bad excluded)
  2. train.py imports at module level
  3. the standalone HF modeling/config files are byte-identical to baseline
     (they define eval behavior of every exported checkpoint)
  4. the three upstream-bug fixes are still present
  5. a seeded tiny RecurrentGPT forward pass on CPU reproduces golden logits/loss

Runs on CPU in a few seconds; no datasets, no GPU needed.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import pkgutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = REPO
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
GOLDEN_META = GOLDEN_DIR / "meta.json"
GOLDEN_LOGITS = GOLDEN_DIR / "tiny_forward.pt"

# upstream AMD/ROCm-only helper, unimportable outside ROCm environments
IMPORT_SKIP = {"recpre.attention_backends.testing"}

# byte-for-byte preservation contract (eval behavior of exported checkpoints)
FROZEN_FILES = [
    "recpre/modeling_recurrent_gpt_standalone.py",
    "recpre/configuration_recurrent_gpt_standalone.py",
]

# (file, required substring, human name) — anchors for the 3 upstream-bug fixes
FIX_ANCHORS = [
    ("recpre/monitor.py", 'if "wte" not in n])', "monitor.py name->n NameError fix"),
    ("recpre/data_loading_utils.py", "def apply_chat_template_supervise_assistant",
     "working chat-template assistant masking present"),
    ("recpre/huggingface_dataset.py", "otherwise these might run out of bounds",
     "dataset-iterator bounds check"),
]

import os  # noqa: E402
os.environ.setdefault("RECUR_ALLOW_CPU", "1")
os.environ.setdefault("REC_SAFE_HEAD", "1")
sys.path.insert(0, str(CODE))

failures: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_imports():
    print("[1/5] recpre import walk")
    import recpre
    n_ok = 0
    for m in pkgutil.walk_packages(recpre.__path__, "recpre."):
        if m.name in IMPORT_SKIP:
            continue
        try:
            importlib.import_module(m.name)
            n_ok += 1
        except Exception as e:
            check(f"import {m.name}", False, f"{type(e).__name__}: {e}")
    check("recpre import walk", not failures, f"{n_ok} modules")


def check_train_import():
    print("[2/5] train.py module import")
    try:
        spec = importlib.util.spec_from_file_location("train_module", CODE / "train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        check("train.py imports", True)
    except Exception as e:
        check("train.py imports", False, f"{type(e).__name__}: {e}")


def check_frozen_files(meta: dict, update: bool):
    print("[3/5] frozen standalone HF files (byte-for-byte)")
    hashes = meta.setdefault("frozen_sha256", {})
    for rel in FROZEN_FILES:
        h = sha256(CODE / rel)
        if update or rel not in hashes:
            hashes[rel] = h
            check(f"baseline recorded: {rel}", True, h[:12])
        else:
            check(f"unchanged: {rel}", h == hashes[rel], h[:12])


def check_fix_anchors():
    print("[4/5] upstream-bug-fix anchors")
    for rel, needle, name in FIX_ANCHORS:
        text = (CODE / rel).read_text()
        check(name, needle in text)
    # the broken upstream chat-template version contained a bare exit()
    check("no exit() in data_loading_utils.py",
          "exit()" not in (CODE / "recpre/data_loading_utils.py").read_text())


def tiny_forward():
    import torch
    import recpre.utils
    from recpre.config_dynamic import Config

    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = Config.from_name(
        "crow-300m-final",  # shrink the real final-run config; keeps 2-layer prelude
        n_embd=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=4,
        vocab_size=256, block_size=128, padding_multiple=128,
        n_layers_in_recurrent_block=[1, 1], mean_recurrence=2, mean_backprop_depth=2,
        n_layers_in_prelude=2, n_layers_in_coda=1,
    )
    objective = {"op": recpre.utils.chunked_cross_entropy, "label_smoothing": 0.0,
                 "ignore_index": -100, "z_regularization": 0.0}
    model = cfg.construct_model(objective=objective, gradient_checkpointing=False)
    model.eval()
    ids = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(123)  # pins recurrence-step sampling inside forward
    with torch.no_grad():
        out = model(ids, labels=ids.clone(), return_logits=True)
    n_params = sum(p.numel() for p in model.parameters())
    return out["logits"].float(), out["loss"].item(), n_params


def check_forward(meta: dict, update: bool):
    import torch
    print("[5/5] seeded tiny forward vs golden")
    try:
        logits, loss, n_params = tiny_forward()
    except Exception as e:
        check("tiny forward runs", False, f"{type(e).__name__}: {e}")
        return
    check("tiny forward runs", True, f"{n_params} params, loss {loss:.6f}")
    if update or not GOLDEN_LOGITS.exists():
        torch.save({"logits": logits, "loss": loss}, GOLDEN_LOGITS)
        meta["n_params"] = n_params
        meta["torch_version"] = torch.__version__
        check("golden baseline recorded", True, str(GOLDEN_LOGITS.relative_to(REPO)))
    else:
        golden = torch.load(GOLDEN_LOGITS, weights_only=True)
        if meta.get("torch_version") != torch.__version__:
            print(f"  WARN  torch {torch.__version__} != baseline {meta.get('torch_version')}"
                  " — numeric drift possible; re-record if this is intentional")
        check("param count unchanged", n_params == meta.get("n_params"),
              f"{n_params} vs {meta.get('n_params')}")
        exact = torch.equal(logits, golden["logits"])
        if exact:
            check("logits identical to golden", True)
        else:
            maxdiff = (logits - golden["logits"]).abs().max().item()
            check("logits identical to golden", False, f"max abs diff {maxdiff:.3e}")
        check("loss matches golden", abs(loss - golden["loss"]) < 1e-9,
              f"{loss:.9f} vs {golden['loss']:.9f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true",
                    help="(re)record the golden baseline instead of comparing")
    args = ap.parse_args()

    GOLDEN_DIR.mkdir(exist_ok=True)
    meta = json.loads(GOLDEN_META.read_text()) if GOLDEN_META.exists() else {}

    check_imports()
    check_train_import()
    check_frozen_files(meta, args.update_golden)
    check_fix_anchors()
    check_forward(meta, args.update_golden)

    if args.update_golden or not GOLDEN_META.exists():
        GOLDEN_META.write_text(json.dumps(meta, indent=2) + "\n")

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
