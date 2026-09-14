# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded test bootstrap: inject errors, then execute the real training CLI under torchrun."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import time
from typing import Any
from unittest.mock import patch

from training import run as run_module
from training.execution import loop as loop_helpers
from training.execution import RunState, build_run_model, save_run_checkpoint
from training.backend.ddp import DDPBackend
from training.data.collate import Batch
from training.data.loader import RunDataloaders
from training.evaluation import evaluate as original_evaluate
from training.steps import NonFiniteLossError
from training.step import run_one_optimizer_step

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    case = os.environ['FAILURE_CASE']
    rank = int(os.environ['RANK'])
    failing_rank = int(os.environ.get('FAILURE_RANK', '0'))
    destination = Path(os.environ['FAILURE_RECORDS'])

    def event(name: str, **fields: Any) -> None:
        with (destination / f'rank-{rank}.jsonl').open('a') as file:
            file.write(json.dumps(dict(event=name, rank=rank, pid=os.getpid(), time=time.monotonic(), **fields)) + '\n')

    event('worker_started')

    def evaluate(settings: Any, backend: Any, model: Any, loader: Any) -> Any:
        event('validation_enter')

        def batches() -> Any:
            if rank == failing_rank and case in ('loader_error', 'empty', 'cleanup_hang'):
                event('injected_failure')
                if case == 'empty':
                    return
                try:
                    raise OSError('FAULT_CAUSE: unreadable validation input')
                except OSError as cause:
                    raise ValueError('FAULT_ORIGIN: validation loader failed') from cause
            for index, batch in enumerate(loader):
                if rank == failing_rank and case == 'later_loader_error' and index == 1:
                    event('injected_failure')
                    raise ValueError('FAULT_ORIGIN: second validation pull failed')
                if rank == failing_rank and case == 'post_forward_metric' and index == 0:
                    event('injected_failure')
                    batch = Batch(batch.input_ids, batch.labels, [*batch.data_ids, 'extra-source'])
                yield batch

        return original_evaluate(settings, backend, model, batches())

    original_shutdown = DDPBackend.shutdown
    original_close = RunDataloaders.close

    def shutdown(self: DDPBackend) -> None:
        event('shutdown_enter')
        if case == 'cleanup_hang' and rank == failing_rank:
            time.sleep(3600)  # external parent deadline owns cleanup if the fatal policy regresses
        original_shutdown(self)

    def close(self: RunDataloaders) -> None:
        event('loaders_close_enter')
        if case == 'cleanup_hang' and rank == failing_rank:
            time.sleep(3600)
        original_close(self)

    DDPBackend.shutdown = shutdown  # type: ignore[method-assign]
    RunDataloaders.close = close  # type: ignore[method-assign]
    original_save = save_run_checkpoint

    def save(state: RunState, *args: Any, **kwargs: Any) -> Path:
        result = original_save(state, *args, **kwargs)
        if rank == 0:
            for file in state.run_directory.glob('checkpoints/*.pth'):
                event('checkpoint', file=str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        return result

    step_count = 0

    def step(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_count
        step_count += 1
        if case == 'nonfinite' and rank == failing_rank and step_count == 2:
            event('injected_failure')
            raise NonFiniteLossError('FAULT_ORIGIN: nonfinite training loss')
        result = run_one_optimizer_step(*args, **kwargs)
        if case == 'graceful_stop' and rank == failing_rank and step_count == 2:
            event('requested_sigint')
            os.kill(os.getpid(), signal.SIGINT)
        return result

    original_model = build_run_model

    def build_model(*args: Any, **kwargs: Any) -> Any:
        if case == 'setup_error' and rank == failing_rank:
            event('injected_failure')
            raise ValueError('FAULT_ORIGIN: model setup failed')
        return original_model(*args, **kwargs)

    try:
        with (
            patch.object(loop_helpers, 'save_run_checkpoint', save),
            patch.object(run_module, 'build_run_model', build_model),
            patch.object(loop_helpers, 'evaluate', evaluate),
            patch.object(loop_helpers, 'run_one_optimizer_step', step),
        ):
            runpy.run_path(str(ROOT / 'training/train.py'), run_name='__main__')
    except SystemExit as error:
        event('worker_exit', code=error.code)
        raise


if __name__ == '__main__':
    main()
