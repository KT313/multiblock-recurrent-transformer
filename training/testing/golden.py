# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The golden 20-step tiny run: the numerics oracle of the training loop (`tasks/training_pipeline_restructure.md`).

Test support, not a test module: `write_tiny_yaml` (the settings the end-to-end tests run on), `golden_run_metrics`
(a tiny run reduced to its numerics), `record_golden_run` (writes `training/golden_tiny_run.json`) and
`golden_mismatches` (the comparison `test_golden_tiny_run` in `test_run.py` and `test_golden_tiny_steps` in
`test_step.py` share). The golden is a refactor guard, not a promise about CPU training: it is recorded in fp32 on
the CPU with one thread and deterministic algorithms, so it catches a changed operation order, an extra RNG draw or
a moved forward pass; it does not exercise the bf16 autocast path of real training.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

from data_preparation.lib.build.runner import prepare
from training.backend.single_device import SingleDeviceBackend
from training.checkpoint import checkpoint_dir, find_latest_checkpoint
from training.run import train
from training.settings import parse_settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TINY_YAML = REPO_ROOT / "config" / "tiny.yaml"
TINY_DATASET_YAML = REPO_ROOT / "config" / "datasets" / "tiny.yaml"
GOLDEN_RUN_PATH = REPO_ROOT / "training" / "golden_tiny_run.json"

GOLDEN_EXACT_ENV = "GOLDEN_EXACT"  # `GOLDEN_EXACT=1`: compare every float with `==` instead of rel 1e-5
GOLDEN_RELATIVE_TOLERANCE = 1e-5
GOLDEN_PER_STEP_KEYS = ("loss", "grad_norm", "lr")
GOLDEN_ALWAYS_EXACT_KEYS = ("lr", "checkpoints", "optimizer_steps")


def write_tiny_yaml(tmp_path: Path, tiny_dataset_dir: Path, out_dir: Path, **overrides: Any) -> Path:
    """`config/tiny.yaml` with `dataset_dir` / `out_dir` rewritten and `overrides` set as plain values (a key the
    file does not have is added at the end); returns the path of the written yaml (`tmp_path / tiny.yaml`)."""
    settings: dict[str, Any] = yaml.safe_load(TINY_YAML.read_text())
    settings.update({"out_dir": str(out_dir), "dataset_dir": str(tiny_dataset_dir), **overrides})
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def golden_exact_requested() -> bool:
    """`GOLDEN_EXACT=1` in the environment: the golden tests compare floats with `==`."""
    return os.environ.get(GOLDEN_EXACT_ENV) == "1"


@contextmanager
def single_thread_deterministic() -> Iterator[None]:
    """One intra-op thread and deterministic algorithms for the block; both restored afterwards, also on failure."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def optimizer_steps_taken(optimizer_state: dict[str, Any]) -> int:
    """The number of `optimizer.step()` calls behind a checkpoint's optimizer state dict: the `step` counter of its
    first parameter (a tensor for ELLISAdam and torch AdamW alike; every parameter gets a gradient every step, so
    all counters agree)."""
    per_parameter = optimizer_state["state"]
    return int(per_parameter[min(per_parameter)]["step"])


def golden_run_metrics(tiny_dataset_dir: Path) -> dict[str, Any]:
    """The 20-step tiny run in fp32 on the CPU (one thread, deterministic algorithms), reduced to its numerics.

    `config/tiny.yaml` with `precision: "32"`, `wandb_enabled: false`, `export_to_hf: false`,
    `resume: false` and `out_dir` in a temporary directory, through
    `train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)`. Returns
    `{"steps": {"<done>": {loss, grad_norm, lr[, val_loss, val_loss_<depth>...]}}, "checkpoints": [file names],
    "optimizer_steps": number of optimizer.step() calls, "parameter_norms": {name: L2 norm in the final checkpoint}}`.
    The per-step values are `report.history`; the `optimizer.step()` calls are read from the final checkpoint's
    optimizer state (`optimizer_steps_taken`). Nothing is probed inside the run.
    """
    with tempfile.TemporaryDirectory() as tmp, single_thread_deterministic():
        tmp_path = Path(tmp)
        out_dir = tmp_path / "out"
        yaml_path = write_tiny_yaml(
            tmp_path, tiny_dataset_dir, out_dir, precision="32", wandb_enabled=False, export_to_hf=False, resume=False
        )
        settings = parse_settings(["--config", str(yaml_path)])
        report = train(settings, backend=SingleDeviceBackend(device="cpu", precision="32"), keep_history=True)

        steps: dict[str, dict[str, float]] = {}
        for done, metrics in sorted(report.history.items()):
            step_metrics = {key: metrics[key] for key in GOLDEN_PER_STEP_KEYS}
            step_metrics |= {key: value for key, value in metrics.items() if key.startswith("val_loss")}
            steps[str(done)] = step_metrics
        final_checkpoint = find_latest_checkpoint(out_dir, settings.run_name)
        assert final_checkpoint is not None
        final_state = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        return {
            "steps": steps,
            "checkpoints": sorted(p.name for p in checkpoint_dir(out_dir).glob("*.pth")),
            "optimizer_steps": optimizer_steps_taken(final_state["optimizer"]),
            "parameter_norms": {
                name: float(torch.linalg.vector_norm(tensor.float())) for name, tensor in final_state["model"].items()
            },
        }


def golden_run_json(metrics: dict[str, Any]) -> str:
    """The fixture text: sorted keys, indent 2, floats as `repr` (json's default, round-trips exactly)."""
    return json.dumps(metrics, sort_keys=True, indent=2) + "\n"


def record_golden_run() -> Path:
    """Re-record `training/golden_tiny_run.json`. ONLY do this in a commit whose purpose is a numerics change of the
    training loop, or when the tiny dataset changes (the fixture depends on `config/datasets/tiny.yaml` and the data
    pipeline: the synthetic rows, dedup, the instruct shuffle and input inversions, the 5 % validation split):

        uv run python -c "from training.testing.golden import record_golden_run; record_golden_run()"

    Builds the tiny dataset into a temporary directory first (as the `tiny_dataset_dir` fixture does). The committed
    fixture was recorded with torch 2.13.0+cu130 on the author's machine (CPU, fp32, one thread, deterministic
    algorithms); two consecutive recordings there are byte-identical.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dataset_root = Path(tmp) / "tiny_dataset"
        prepare(TINY_DATASET_YAML, dataset_root, assume_yes=False)
        metrics = golden_run_metrics(dataset_root)
    GOLDEN_RUN_PATH.write_text(golden_run_json(metrics))
    return GOLDEN_RUN_PATH


def _floats_agree(expected: float, actual: Any) -> bool:
    """`actual` within `GOLDEN_RELATIVE_TOLERANCE × |expected|` of `expected` (no absolute tolerance, like
    `pytest.approx(rel=1e-5, abs=0)`)."""
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return abs(actual - expected) <= GOLDEN_RELATIVE_TOLERANCE * abs(expected)


def golden_mismatches(expected: Any, actual: Any, *, exact: bool, path: str = "") -> list[str]:
    """Every difference between a recorded golden structure and a fresh one, as `path: expected != actual` lines.

    Floats are compared with a relative tolerance of `GOLDEN_RELATIVE_TOLERANCE` (no absolute tolerance), or with
    `==` when `exact`; values under a key in `GOLDEN_ALWAYS_EXACT_KEYS` (learning rates, checkpoint names, the
    optimizer-step count), ints, strings and key sets are always compared exactly.
    """
    key = path.rsplit("/", 1)[-1]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected a mapping, got {type(actual).__name__}"]
        mismatches = [f"{path}/{k}: missing" for k in sorted(set(expected) - set(actual))]
        mismatches += [f"{path}/{k}: unexpected" for k in sorted(set(actual) - set(expected))]
        for k in sorted(set(expected) & set(actual)):
            mismatches += golden_mismatches(expected[k], actual[k], exact=exact, path=f"{path}/{k}")
        return mismatches
    if isinstance(expected, list):
        return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, float) and not exact and key not in GOLDEN_ALWAYS_EXACT_KEYS:
        return [] if _floats_agree(expected, actual) else [f"{path}: {expected!r} != {actual!r}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]
