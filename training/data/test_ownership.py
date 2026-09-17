# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded CPU/gloo evidence for entry refusal and the all-reader completion boundary."""

from datetime import timedelta
from contextlib import suppress
import multiprocessing as mp
import os
import signal
import subprocess
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TypeVar

import pytest

from data_preparation.lib.abort import BuildAborted
from data_preparation.lib.build.lock import RunLocked, build_lock
from training.data.ownership import dataset_access, main_rank_phase, training_dataset_access

T = TypeVar("T")


class _GlooBackend:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.is_main = rank == 0
        self.world_size = 2

    def all_gather_object(self, obj: T) -> list[T]:
        import torch.distributed as dist
        values: list[T] = [obj, obj]
        dist.all_gather_object(values, obj)
        return values

    def barrier(self) -> None:
        import torch.distributed as dist
        dist.barrier()


def _rank(rank: int, root: str, rendezvous: str, connection: Connection, mode: str, shared: bool) -> None:
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=8))
        backend = _GlooBackend(rank)
        with training_dataset_access(Path(root), Path(root).parent / "output", backend, shared=shared) as lease:
            assert (lease is not None) == backend.is_main
            connection.send("entered")
            if mode in {"setup_error", "cancelled"}:
                with main_rank_phase(backend, "auto-prepare"):
                    if backend.is_main:
                        if mode == "cancelled":
                            raise BuildAborted("preparation cancelled deliberately")
                        raise ValueError("preparation failed deliberately")
            if mode == "delayed" and rank == 1:
                assert connection.recv() == "close readers"
            if mode == "peer_error" and rank == 1:
                raise ValueError("reader failed deliberately")
            connection.send("readers closed")
        connection.send("released")
    except Exception as error:
        connection.send(f"error: {type(error).__name__}: {error}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


@pytest.mark.timeout(40)
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("mode", ["delayed", "conflict", "setup_error", "peer_error", "cancelled"])
def test_gloo_dataset_lifetime(tmp_path: Path, mode: str, shared: bool) -> None:
    context = mp.get_context("spawn")
    connections = [context.Pipe() for _ in range(2)]
    root = tmp_path / "dataset"
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = [context.Process(target=_rank, args=(rank, str(root), rendezvous, pair[1], mode, shared))
                 for rank, pair in enumerate(connections)]
    held = build_lock(root) if mode == "conflict" else None
    if held is not None:
        held.__enter__()
    try:
        for process in processes:
            process.start()
        for parent, child in connections:
            child.close()
            assert parent.poll(20), "rank did not report entry"
            message = parent.recv()
            if mode == "conflict":
                assert "error:" in message and str(root) in message and "RunLocked" in message
            else:
                assert message == "entered"
        if mode == "delayed":
            assert connections[0][0].poll(10)
            assert connections[0][0].recv() == "readers closed"
            with pytest.raises(RunLocked), build_lock(root):
                pass
            connections[1][0].send("close readers")
            assert connections[1][0].poll(10)
            assert connections[1][0].recv() == "readers closed"
            for parent, _ in connections:
                assert parent.poll(10) and parent.recv() == "released"
        elif mode in {"setup_error", "peer_error", "cancelled"}:
            for rank, (parent, _) in enumerate(connections):
                assert parent.poll(15)
                message = parent.recv()
                if mode == "peer_error" and rank == 0:
                    assert message == "readers closed"
                    assert parent.poll(15)
                    message = parent.recv()
                assert message.startswith("error:")
                if mode == "setup_error":
                    assert "preparation failed deliberately" in message
                if mode == "cancelled":
                    assert message.startswith("error: BuildAborted:") and "preparation cancelled deliberately" in message
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for parent, _ in connections:
            parent.close()
        if held is not None:
            held.__exit__(None, None, None)
    with build_lock(root):
        pass


def test_single_rank_setup_exception_releases_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="setup"), dataset_access(tmp_path):
        raise ValueError("setup")
    with build_lock(tmp_path):
        pass


def _terminate_subprocess_tree(process: subprocess.Popen[str]) -> None:
    """Freeze and enumerate descendants before killing, including torchrun's separate worker sessions."""

    stopped: list[int] = []

    def freeze(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            return
        stopped.append(pid)
        # SIGSTOP stops every thread; enumerate all task children so none can escape by spawning while traversed.
        try:
            children = {int(child) for task in (Path("/proc") / str(pid) / "task").iterdir()
                        for child in (task / "children").read_text().split()}
        except FileNotFoundError:
            children = set()
        for child in children:
            freeze(child)

    freeze(process.pid)
    for pid in reversed(stopped[1:]):
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    # Let the launcher reap its children; force it down too if its normal error handling does not finish promptly.
    with suppress(ProcessLookupError):
        os.kill(process.pid, signal.SIGCONT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.timeout(15)
def test_timeout_cleanup_reaches_a_child_in_another_session() -> None:
    import sys

    command = [sys.executable, "-c", "import subprocess,sys; "
               "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
               "print(child.pid, flush=True); child.wait()"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert process.stdout is not None
        child_pid = int(process.stdout.readline())
        _terminate_subprocess_tree(process)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            _terminate_subprocess_tree(process)


@pytest.mark.slow
@pytest.mark.timeout(60)
@pytest.mark.parametrize("conflict", [False, True])
def test_torchrun_training_uses_the_dataset_lease(
    tmp_path: Path, tiny_dataset_dir: Path, conflict: bool
) -> None:
    import json
    import sys
    from contextlib import nullcontext

    from training.testing.golden import write_tiny_yaml

    output = tmp_path / "output"
    config = write_tiny_yaml(
        tmp_path, tiny_dataset_dir, output, backend="ddp", precision="32", wandb_enabled=False,
        export_to_hf=False, resume=False,
    )
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
               "--max-restarts=0", "training/train.py", "--config", str(config)]
    with build_lock(tiny_dataset_dir) if conflict else nullcontext():
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
            cwd=Path(__file__).resolve().parents[2],
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "TRAINING_DASHBOARD": "0"},
        )
        try:
            _, stderr = process.communicate(timeout=45)
        finally:
            if process.poll() is None:
                _terminate_subprocess_tree(process)
    if conflict:
        assert process.returncode != 0 and str(tiny_dataset_dir) in stderr and "RunLocked" in stderr
        assert not list(output.rglob("*.pth"))
    else:
        assert process.returncode == 0, stderr[-5000:]
        report = json.loads((output / "tiny" / "train_report.json").read_text())
        assert report["completed_steps"] == 20 and report["stopped"] is False
    with build_lock(tiny_dataset_dir):
        pass


class _CountingBackend:
    is_main = True
    world_size = 1

    def __init__(self) -> None:
        self.exchanges = 0

    def all_gather_object(self, obj: T) -> list[T]:
        self.exchanges += 1
        return [obj]

    def barrier(self) -> None:
        pass


@pytest.mark.parametrize("forced", [KeyboardInterrupt(), SystemExit(130)])
def test_forced_abort_never_starts_a_status_collective(forced: BaseException) -> None:
    backend = _CountingBackend()
    with pytest.raises(type(forced)) as caught, main_rank_phase(backend, "forced abort"):
        raise forced
    assert caught.value is forced and backend.exchanges == 0


def test_cooperative_cancellation_retains_originating_error_and_cause() -> None:
    backend = _CountingBackend()
    cause = ValueError("original reason")
    cancellation = BuildAborted("stop preparation")
    with pytest.raises(BuildAborted) as caught, main_rank_phase(backend, "auto-prepare"):
        raise cancellation from cause
    assert caught.value is cancellation and caught.value.__cause__ is cause
    assert backend.exchanges == 1
