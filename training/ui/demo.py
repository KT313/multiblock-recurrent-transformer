# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""A scripted fake run for a look at the dashboard: ``uv run python -m training.ui.dashboard [seconds]``."""

from __future__ import annotations

import logging
import math
import random
import sys
import time
import warnings
from pathlib import Path

from rich.console import Console

from training.ui.common import KEEP, log
from training.ui.dashboard import training_dashboard

DEMO_STAGES = ["pretrain", "instruct"]
DEMO_STEPS = [30, 20]


def demo(
    seconds: float = 5.0, *, enabled: bool | None = None, log_file: Path | None = None, console: Console | None = None
) -> None:
    """A fake two-stage run (30 + 20 optimizer steps over ``seconds``) driving the whole API while the things a real
    run writes around it fire — a stray ``print``, a bare ``sys.stderr.write``, a ``warnings.warn``, a third-party
    logger with its own stderr handler. Piped (or ``TRAINING_DASHBOARD=0``) it shows the fallback."""
    total_steps = sum(DEMO_STEPS)
    pause = seconds / total_steps
    rng = random.Random(0)
    details = {"model": "crow-tiny", "dataset": "tiny", "device": "cpu", "precision": "32"}
    library = logging.getLogger("demo_third_party_library")  # like transformers / datasets: a StreamHandler on stderr
    library_handler = logging.StreamHandler(sys.stderr)
    library.addHandler(library_handler)
    try:
        with training_dashboard(
            "demo-run",
            DEMO_STAGES,
            DEMO_STEPS,
            total_steps,
            details=details,
            log_step_interval=5,
            log_file=log_file,
            enabled=enabled,
            console=console,
        ) as board:
            board.note_event("no checkpoint found, starting from scratch")
            board.set_status("training")
            log.info("Total training steps: %d (4 micro-batches each)", total_steps, extra=KEEP)
            loss = 6.0
            for step in range(1, total_steps + 1):
                time.sleep(pause)
                loss = loss * 0.97 + rng.uniform(-0.05, 0.05)
                stage_index = 0 if step <= DEMO_STEPS[0] else 1
                in_transition = 27 <= step <= 30
                metrics: dict[str, float] = {
                    "loss": loss,
                    "ppl": math.exp(loss),
                    "lr": 3e-4 * min(step / 10, 1.0),
                    "grad_norm": rng.uniform(0.5, 1.5),
                    "tokens/second": rng.uniform(9_000, 11_000),
                    "total_tokens": step * 8_192,
                }
                board.update_step(step, stage_index, (step - 27) / 4 if in_transition else None, metrics)
                if step == 27:
                    board.note_event("starting transition 0 -> 1 (pretrain -> instruct)")
                if step == 30:
                    board.note_event("transition complete, now in stage 1 (instruct)")
                    board.note_event("saved checkpoint outputs/demo/checkpoints/step-00000030-demo-run-stage-0_end.pth")
                if step % 10 == 0:
                    board.set_status("evaluating")
                    board.update_validation(step, {"val_loss_4": loss + 0.3, "val_loss_8": loss + 0.15, "val_loss": loss + 0.1})
                    board.set_status("training")
                if step % 25 == 0 and step != 30:
                    board.note_event(f"saved checkpoint outputs/demo/checkpoints/step-{step:08d}-demo-run.pth")
                if step % 7 == 0:
                    log.info("step %d: sample log record (grad metrics, data composition ...)", step)
                if step == 12:
                    print(f"step {step}: a stray print (lands in the log panel, not on the terminal)")
                if step == 18:
                    sys.stderr.write(f"step {step}: a bare stderr write (kept)\n")
                if step == 24:
                    warnings.warn(f"step {step}: a warnings.warn (kept)", stacklevel=1)
                if step == 33:
                    library.warning("step %d: a third-party logger with its own stderr handler (kept)", step)
                if step == 40:
                    log.warning("step %d: an example warning (kept in the scrollback)", step)
            board.set_status("exporting")
            board.note_event("exported HuggingFace model to outputs/demo/hf_export")
            board.set_status("finished")
            log.info("Training finished after %d steps", total_steps, extra={"keep": True})
            time.sleep(min(0.5, seconds / 10))
    finally:
        library.removeHandler(library_handler)


def main(argv: list[str] | None = None) -> int:
    """``python -m training.ui.dashboard [seconds]``: the demo; Ctrl-C leaves through the dashboard and exits 130."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        demo(float(arguments[0]) if arguments else 5.0)
    except KeyboardInterrupt:
        return 130
    return 0
