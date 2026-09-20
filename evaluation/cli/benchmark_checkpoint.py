# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Resolve a training config and checkpoint for immediate scheduled-style benchmarking."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import torch
from jsonargparse import ActionConfigFile, ArgumentParser  # type: ignore[attr-defined]

from data_preparation import DatasetLayout, load_dataset_config
from data_preparation.lib.abort import StopCheck
from data_preparation.lib.log import get_logger
from evaluation.cli.checkpoint import load_checkpoint_config
from evaluation.distributed_benchmarks import evaluate_configured_benchmarks
from model.config import RecurrentConfig
from model.layers.init import checkpoint_initialization
from model.model import RecurrentGPT
from tokenization.validation import check_model_vocabulary
from training.backend.base import Backend
from training.checkpoint import find_latest_checkpoint
from training.data.tokenizer import Tokenizer
from training.execution.setup import check_tokenizer_vocabulary, create_backend, get_run_directory
from training.failure import FatalHandler, fatal_errors
from training.settings import Settings
from training.stopping import StopController, complete_main_phase
from training.tokenizer_contract import check_checkpoint_tokenizer, resolve_checkpoint_tokenizer
from training.tokenizer_parity import check_template_parity

log = get_logger("training.benchmark_checkpoint")


@dataclass(frozen=True)
class BenchmarkOptions:
    checkpoint: str | None = None
    output_dir: str | None = None
    tokenizer_dir: str | None = None


def parse_benchmark_settings(argv: list[str] | None = None) -> tuple[Settings, BenchmarkOptions]:
    parser = ArgumentParser(description="Run training-config benchmarks immediately from a saved checkpoint.")
    parser.add_argument("--config", action=ActionConfigFile, required=True, help="Training YAML; paths resolve as in training")
    parser.add_class_arguments(Settings, nested_key=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override; default: newest regular checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="New result directory; default: unique benchmark_checks subdirectory")
    parser.add_argument("--tokenizer_dir", type=str, default=None, help="Relocated tokenizer; must match checkpoint identity")
    namespace = parser.parse_args(argv)
    namespace.pop("config", None)
    options = BenchmarkOptions(**{name: namespace.pop(name) for name in ("checkpoint", "output_dir", "tokenizer_dir")})
    settings = Settings(**parser.instantiate(namespace).as_dict())
    if not settings.benchmark_tasks:
        raise ValueError("benchmark_tasks must contain at least one task for an immediate benchmark check")
    return settings, options


def select_benchmark_checkpoint(settings: Settings, override: str | None) -> Path:
    """Use the training checkpoint search, regardless of resume flags or scheduled benchmark triggers."""
    path = Path(override) if override else find_latest_checkpoint(get_run_directory(settings), settings.run_name)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"no benchmark checkpoint found: {path or get_run_directory(settings) / 'checkpoints'}")
    return path.resolve()


def load_benchmark_inputs(
    settings: Settings, options: BenchmarkOptions, checkpoint: Path, device: torch.device,
) -> tuple[RecurrentGPT, Tokenizer, int]:
    # map checkpoint storage lazily so unused optimizer tensors are not read into each rank's RAM
    before = checkpoint.stat()
    state: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = load_checkpoint_config(state)
    requested = RecurrentConfig.from_yaml(
        settings.model_architecture_config, **(settings.model_overwrite | {"use_custom_kernels": settings.use_custom_kernels}),
    )
    config.use_custom_kernels = settings.use_custom_kernels  # execution choice comes from the current training config
    if config != requested:
        changed = [key for key, value in config.to_dict().items() if requested.to_dict().get(key) != value]
        raise ValueError(f"training architecture differs from checkpoint in {changed}; use its matching training config")

    # preserve checkpoint-owned token IDs and validate the same formatter used during training
    tokenizer_path = resolve_checkpoint_tokenizer(state, checkpoint, options.tokenizer_dir)
    if tokenizer_path is None:
        dataset = load_dataset_config(settings.dataset_config)
        tokenizer_path = DatasetLayout(Path(settings.dataset_dir)).tokenizer_dir(dataset.tokenizer.name)
    tokenizer = Tokenizer(tokenizer_path)
    check_checkpoint_tokenizer(state.get("tokenizer_contract"), tokenizer.contract)
    check_tokenizer_vocabulary(tokenizer, config)
    check_model_vocabulary(config, tokenizer.contract)
    check_template_parity(tokenizer)

    # restore only model weights; scheduled benchmarks also execute the unwrapped, uncompiled model
    with checkpoint_initialization():
        model = RecurrentGPT(config, gradient_checkpointing=settings.gradient_checkpointing)
    model.load_state_dict(state["model"], strict=True)
    step = int(state["step"])
    model.step = step
    model.to(device)
    check_model_vocabulary(config, tokenizer.contract, model)
    after = checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("checkpoint changed while loading; retry with a stable checkpoint")
    return model, tokenizer, step


def create_benchmark_output(settings: Settings, options: BenchmarkOptions, step: int) -> Path:
    if options.output_dir is not None:
        directory = Path(options.output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        return directory
    parent = get_run_directory(settings).resolve() / "benchmark_checks"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=f"step-{step:08d}-", dir=parent))


def run_checkpoint_benchmark_check(
    settings: Settings, options: BenchmarkOptions, *, should_stop: StopCheck | None = None,
    on_fatal_error: FatalHandler | None = None,
) -> int:
    # training owns device selection and process-group setup; inference needs no DDP wrapper or optimizer
    with fatal_errors(on_fatal_error):
        backend = create_backend(settings)
    try:
        with fatal_errors(on_fatal_error):
            return run_benchmark_check_on_backend(backend, settings, options, should_stop, on_fatal_error)
    finally:
        backend.shutdown()


def run_benchmark_check_on_backend(
    backend: Backend, settings: Settings, options: BenchmarkOptions, should_stop: StopCheck | None,
    on_fatal_error: FatalHandler | None,
) -> int:
    # choose once on rank zero so a newly published checkpoint cannot split the ranks across different steps
    selected = complete_main_phase(backend, "checkpoint selection", lambda: str(select_benchmark_checkpoint(settings, options.checkpoint)))
    checkpoint_name = backend.all_gather_object(selected)[0]
    assert checkpoint_name is not None
    checkpoint = Path(checkpoint_name)
    log.info("loading benchmark checkpoint %s on %s", checkpoint, backend.device)
    model, tokenizer, step = load_benchmark_inputs(settings, options, checkpoint, backend.device)
    identity = (step, checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    if any(other != identity for other in backend.all_gather_object(identity)):
        raise ValueError("ranks loaded different checkpoint versions")

    # publish into a fresh directory, keeping scheduled results and all training files untouched
    local_dir = complete_main_phase(backend, "benchmark output directory", lambda: str(create_benchmark_output(settings, options, step)))
    output_dir = backend.all_gather_object(local_dir)[0]
    assert output_dir is not None
    output = Path(output_dir) / f"step-{step:08d}.json"
    log.info("benchmark check: step %d, %d ranks, tasks %s, limit %s; results: %s",
             step, backend.world_size, settings.benchmark_tasks, settings.benchmark_limit, output)
    result = evaluate_configured_benchmarks(
        backend, model, tokenizer, settings, out_path=output, step=step,
        stop=StopController(backend, should_stop), on_fatal_error=on_fatal_error,
    )
    if backend.is_main:
        if result.completed:
            print(f"Benchmark results: {output}", flush=True)
            for name, value in sorted(result.metrics.items()):
                print(f"{name}: {value}", flush=True)
        else:
            log.warning("benchmark check interrupted; no partial results published")
    return 0 if result.completed else 130
