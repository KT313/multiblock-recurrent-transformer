# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded torchrun worker for cooperative stop/phase failure tests (CPU/gloo only)."""

import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from training import run as run_module
from training.execution import loop as loop_helpers
from training.backend.ddp import DDPBackend
from training.settings import parse_settings
from training.step import run_one_optimizer_step
from training.stopping import StopController, complete_main_phase


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 15
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"peer did not publish {path}")
        time.sleep(0.01)


def assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config, root, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    (root / f"{mode}-rank-{os.environ['RANK']}.pid").write_text(str(os.getpid()))
    backend = DDPBackend(device="cpu", precision="32")
    rank = backend.rank
    if mode == "stop":
        waiting, request = root / "waiting", root / "request"
        vote = backend.any_flag

        def completion_vote(flag: bool) -> bool:
            if rank == 1:
                waiting.touch()
            return vote(flag)

        backend.any_flag = completion_vote  # type: ignore[method-assign]

        def main_work() -> None:
            wait_for(waiting)
            request.touch()  # rank 1 is already at the completion vote; its flag must be sampled afterward

        complete_main_phase(backend, "late request", main_work)
        backend.any_flag = vote  # type: ignore[method-assign]
        stop = StopController(backend, lambda: rank == 1 and request.exists())
        assert stop.poll("after samples")

        for phase in ("samples", "checkpoint publication"):
            def fail() -> None:
                raise ValueError("publication failed")

            try:
                complete_main_phase(backend, phase, fail)
            except (ValueError, RuntimeError) as error:
                assert str(error) == ("publication failed" if rank == 0 else f"rank zero failed during {phase}")
            else:
                raise AssertionError("phase failure was swallowed")
    settings = parse_settings(["--config", str(config)])
    requested = False
    step = run_one_optimizer_step
    count = 0

    def completed_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal requested, count
        result = step(*args, **kwargs)
        count += 1
        if count == 2 and rank == 1:
            requested = True
        return result

    if mode == "stop":
        with patch.object(loop_helpers, "run_one_optimizer_step", completed_step):
            stopped = run_module.train(settings, backend=backend, should_stop=lambda: requested)
        assert stopped.stopped and stopped.completed_steps == stopped.steps_this_process == 2
        checkpoint = Path(settings.out_dir) / settings.run_name / "checkpoints" / "step-00000002-tiny.pth"
        state = torch.load(checkpoint, weights_only=False)
        assert state["step"] == 2 and len(state["rng_states"]) == 2
        assert state["data_stream"]["consumed_rows"]
    elif mode == "provenance_failure":
        settings.resume = True
        directory = Path(settings.out_dir) / settings.run_name
        before = {path: path.read_bytes() for path in directory.glob("*config.json")}
        def fail_publication(*args: Any, **kwargs: Any) -> None:
            raise OSError("configuration publication refused")

        def unexpected_step(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("publication failure reached an optimizer update")

        with patch.object(run_module, "publish_configuration", fail_publication), \
                patch.object(loop_helpers, "run_one_optimizer_step", unexpected_step):
            try:
                run_module.train(settings, backend=backend)
            except (OSError, RuntimeError) as error:
                expected = "configuration publication refused" if rank == 0 else "rank zero failed during configuration publication"
                assert str(error) == expected
            else:
                raise AssertionError("configuration publication failure was swallowed")
        assert all(path.read_bytes() == content for path, content in before.items())
        assert not list((directory / "resumes").glob("*.json"))
    elif mode == "resume":
        settings.resume = True
        resumed = run_module.train(settings, backend=backend)
        assert resumed.completed_steps == 20 and resumed.steps_this_process == 18 and not resumed.stopped
    elif mode == "full":
        resumed_state = torch.load(Path(settings.out_dir) / settings.run_name / "checkpoints" / "step-00000020-tiny.pth",
                                   weights_only=False)
        settings.out_dir = str(root / "full")
        full = run_module.train(settings, backend=backend)
        assert full.completed_steps == 20 and not full.stopped
        full_state = torch.load(Path(settings.out_dir) / settings.run_name / "checkpoints" / "step-00000020-tiny.pth",
                                weights_only=False)
        for key in ("model", "optimizer", "rng_states", "data_stream"):
            assert_equal(resumed_state[key], full_state[key])
    else:
        raise ValueError(f"unknown worker mode {mode}")
    (root / f"{mode}-rank-{rank}-passed").write_text("passed")



if __name__ == "__main__":
    os.environ["TRAINING_DASHBOARD"] = "0"
    main()
