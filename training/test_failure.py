# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Fatal DDP worker exit bypasses cleanup; launcher and loader workers terminate the job."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import suppress
from typing import Any, NoReturn

import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import RunLocked
from training.data.test_ownership import _terminate_subprocess_tree
from training.failure import exit_failed_worker, fatal_errors, handle_fatal_error
from training.testing.golden import write_tiny_yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.xdist_group('module:training/test_failure.py')


@pytest.mark.parametrize('error', [BuildAborted('cancel'), RunLocked(Path('/fixture'), 'training', None),
                                 KeyboardInterrupt(), SystemExit(2)])
def test_fatal_handler_does_not_change_control_flow(error: BaseException) -> None:
    def forbidden(caught: Exception) -> NoReturn:
        pytest.fail('cancellation must not invoke fatal worker exit')
    handle_fatal_error(forbidden, error)


def test_library_failure_keeps_original_exception() -> None:
    error = ValueError('original')
    with pytest.raises(ValueError) as caught, fatal_errors(None):
        raise error
    assert caught.value is error


@pytest.mark.parametrize('broken_stderr', [False, True])
def test_fatal_exit_bypasses_logger_and_exits_even_with_broken_stderr(
    monkeypatch: pytest.MonkeyPatch, broken_stderr: bool,
) -> None:
    output = bytearray()
    def write(fd: int, value: memoryview) -> int:
        assert fd == 2
        if broken_stderr:
            raise OSError('stderr unavailable')
        output.extend(value[:7])  # partial writes must not truncate the report
        return len(value[:7])
    def exit_process(code: int) -> NoReturn:
        raise SystemExit(code)
    monkeypatch.setattr(os, 'write', write)
    monkeypatch.setattr(os, '_exit', exit_process)
    monkeypatch.setenv('RANK', '7')
    try:
        raise ValueError('original failure')
    except ValueError as error:
        with pytest.raises(SystemExit) as stopped:
            exit_failed_worker(error)
    assert stopped.value.code == 1
    if not broken_stderr:
        assert b'[rank 7] training failed' in output and b'ValueError: original failure' in output


def _identity(pid: int) -> str | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]  # Linux starttime; avoid matching a reused PID
    except (FileNotFoundError, ProcessLookupError):  # the process exited between the listing and the read (ESRCH)
        return None


def _descendants(pid: int, known: dict[int, str]) -> None:
    identity = _identity(pid)
    if identity is None:
        return
    known[pid] = identity
    try:
        children = {int(child) for task in Path(f'/proc/{pid}/task').iterdir()
                    for child in (task / 'children').read_text().split()}
    except (FileNotFoundError, ProcessLookupError):
        return
    for child in children:
        _descendants(child, known)


def _events(directory: Path) -> list[dict[str, Any]]:
    result = []
    for file in directory.glob('rank-*.jsonl'):
        for line in file.read_text().splitlines():
            with suppress(json.JSONDecodeError):  # telemetry writer may be mid-append
                result.append(json.loads(line))
    return result


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != 'linux', reason='bounded owned-process checks use Linux /proc')
@pytest.mark.timeout(150)
@pytest.mark.parametrize(('case', 'rank'), [('loader_error', 0), ('loader_error', 1), ('empty', 0),
                                          ('later_loader_error', 0), ('post_forward_metric', 0),
                                          ('cleanup_hang', 0), ('nonfinite', 1), ('setup_error', 1), ('graceful_stop', 1)])
def test_torchrun_fatal_exit_and_graceful_control(
    tiny_dataset_dir: Path, tmp_path: Path, case: str, rank: int,
) -> None:
    config = write_tiny_yaml(tmp_path, tiny_dataset_dir, tmp_path / 'out', backend='ddp', precision='32',
                             compile_model=False, resume=False, auto_prepare=False, export_to_hf=False,
                             save_step_interval=1, eval_step_interval=2, eval_iters=4, validation_batch_size=1,
                             wandb_enabled=False)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '--max-restarts=0', '--shutdown-timeout=1800', '--module', 'training.testing.failure_worker',
               '--config', str(config)]
    env = os.environ | {'FAILURE_CASE': case, 'FAILURE_RANK': str(rank), 'FAILURE_RECORDS': str(tmp_path),
                        'TRAINING_DASHBOARD': '0', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    known: dict[int, str] = {}
    started = time.monotonic()
    with (tmp_path / 'stdout.log').open('w') as stdout, (tmp_path / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, text=True)
        try:
            while process.poll() is None:
                _descendants(process.pid, known)
                triggers = [row['time'] for row in _events(tmp_path)
                            if row['event'] in ('injected_failure', 'requested_sigint')]
                deadline = min(started + 100, min(triggers) + 45) if triggers else started + 100
                if time.monotonic() > deadline:
                    pytest.fail(f'launcher did not terminate; see {tmp_path}')
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                _terminate_subprocess_tree(process)
            process.wait(timeout=10)
            # Orphaned PyTorch dataloader workers check their parent's PID periodically. Allow their
            # existing watchdog to finish, but never leave an observed child running on test failure.
            deadline = time.monotonic() + 12
            while any(_identity(pid) == identity for pid, identity in known.items()) and time.monotonic() < deadline:
                time.sleep(0.1)
            live = [pid for pid, identity in known.items() if _identity(pid) == identity]
            for pid in live:
                with suppress(ProcessLookupError):
                    os.kill(pid, 9)
            assert not live, f'owned processes outlived torchrun: {live}'
    trace = _events(tmp_path)
    triggers = [row for row in trace if row['event'] in ('injected_failure', 'requested_sigint')]
    assert triggers, (tmp_path / 'stderr.log').read_text()
    assert process.returncode != 0  # graceful worker code 130 also makes torchrun return 1
    checkpoints = [row for row in trace if row['event'] == 'checkpoint']
    assert bool(checkpoints) == (case != 'setup_error')
    assert all(hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() == row['sha256'] for row in checkpoints)
    files = list((tmp_path / 'out/tiny/checkpoints').glob('*.pth'))
    text = (tmp_path / 'stderr.log').read_text()
    if case == 'graceful_stop':
        assert any('00000002' in file.name for file in files)
        assert sorted(row['code'] for row in trace if row['event'] == 'worker_exit') == [130, 130]
        assert 'exiting worker immediately' not in text
    else:
        assert all('00000001' in file.name for file in files)
        assert f'[rank {rank}] training failed; exiting worker immediately' in text
        expected = 'validation loader yielded no batch' if case == 'empty' else 'IndexError' if case == 'post_forward_metric' else 'FAULT_ORIGIN'
        assert expected in text
        if case in ('loader_error', 'cleanup_hang'):
            assert 'FAULT_CAUSE' in text and 'direct cause' in text
        assert not any(row['rank'] == rank and row['event'] in ('loaders_close_enter', 'shutdown_enter') for row in trace)
