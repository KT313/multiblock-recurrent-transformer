# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Formatting for the dashboard and its fallback lines, the readers of the step dict, and the panel-height fit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


from training.ui.throughput import Throughput

MIN_LOG_LINES = 2  # the log panel never shrinks below this on a short terminal; the events panel goes down to one line

# metric keys of the step dict (`RunLogger.log_step`) shown in the metrics table, with labels
METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("loss", "loss"),
    ("ppl", "ppl"),
    ("lr", "lr"),
    ("grad_norm", "grad norm"),
    ("tokens/second", "tokens/s"),
    ("seconds/step", "s/step"),
    ("total_tokens", "tokens"),
)

def format_duration(seconds: float | None) -> str:
    """``h:mm:ss`` (``Nd hh:mm:ss`` from one day on); ``—`` for unknown / non-finite / negative values."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    whole = int(seconds)
    days, rest = divmod(whole, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_tokens(count: float) -> str:
    """A token count with a k / M / B / T suffix (``1.23B``); plain below 1000."""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(count) >= threshold:
            return f"{count / threshold:.2f}{suffix}"
    return f"{count:.0f}"


# format strings per metric key of the step dict; `total_tokens` goes through `format_tokens`, anything else `.4g`
METRIC_FORMATS: dict[str, str] = {
    "loss": "{:.4f}",
    "ppl": "{:.2f}",
    "lr": "{:.2e}",
    "grad_norm": "{:.3f}",
    "tokens/second": "{:,.0f}",
    "seconds/step": "{:.2f}s",
}


def format_metric(key: str, value: float) -> str:
    """The table / fallback-line rendering of one metric of the step dict."""
    if key == "total_tokens":
        return format_tokens(value)
    return METRIC_FORMATS.get(key, "{:.4g}").format(value)


def fit_panel_heights(available: int, events: int, log: int) -> tuple[int, int]:
    """Lines of the events and log panels (bodies, without the two border lines each) that fit into ``available``
    terminal lines: the log panel shrinks first (down to :data:`MIN_LOG_LINES`), then the events panel (down to one
    line). Below that the frame cannot fit and rich crops it."""
    events, log = max(events, 1), max(log, MIN_LOG_LINES)
    while events + log + 4 > available and log > MIN_LOG_LINES:
        log -= 1
    while events + log + 4 > available and events > 1:
        events -= 1
    return events, log


def as_float(value: object) -> float | None:
    """``float(value)`` for numbers and one-element tensors, None for anything that does not convert."""
    try:
        return float(value)  # type: ignore[arg-type]  # the point of the helper is to accept anything float-like
    except (TypeError, ValueError):
        return None


def floats(values: Mapping[str, object]) -> dict[str, float]:
    """The float-convertible entries of ``values`` (insertion order kept)."""
    return {key: value for key, raw in values.items() if (value := as_float(raw)) is not None}


def known_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """The metric-table keys present in ``metrics`` (in table order) as floats."""
    return floats({key: metrics[key] for key, _label in METRIC_COLUMNS if key in metrics})


def step_line(
    step: int,
    stage_index: int,
    transition: float | None,
    metrics: Mapping[str, object],
    *,
    total_steps: int,
    stage_names: Sequence[str],
    log_step_interval: int,
    throughput: Throughput,
) -> str | None:
    """The log line of ``step`` (``step 5/30 | stage 0 pretrain | loss 3.0000 | ... | ETA 0:00:50``) — None at a
    step that is not logged (every ``log_step_interval``\\ th step and the last one are). ``transition`` is the
    progress of the running stage transition, None outside one; ``throughput`` (with ``step`` already recorded)
    supplies the seconds per step when the step dict has none, the elapsed time and the ETA."""
    if step % log_step_interval and step < total_steps:
        return None
    stage_name = stage_names[stage_index] if 0 <= stage_index < len(stage_names) else "?"
    parts = [f"step {step}/{total_steps}", f"stage {stage_index} {stage_name}"]
    if transition is not None:
        parts.append(f"transition {transition:.0%}")
    known = known_metrics(metrics)
    parts += [f"{label} {format_metric(key, known[key])}" for key, label in METRIC_COLUMNS if key in known]
    if "seconds/step" not in known and throughput.seconds_per_step is not None:
        parts.append(f"s/step {throughput.seconds_per_step:.2f}s")
    parts.append(f"elapsed {format_duration(throughput.elapsed)}")
    parts.append(f"ETA {format_duration(throughput.remaining(step))}")
    return " | ".join(parts)


def validation_line(step: int, losses: Mapping[str, object]) -> str:
    """The log line of a validation: ``step 10: validation val_loss_4 3.2500, val_loss 3.1250``."""
    values = ", ".join(f"{key} {value:.4f}" for key, value in floats(losses).items())
    return f"step {step}: validation {values or '(no losses)'}"


def event_line(text: str) -> str:
    """The log line of an event: ``event: saved checkpoint ...``."""
    return f"event: {text}"


def status_line(text: str) -> str:
    """The (DEBUG) log line of a status change: ``status: evaluating``."""
    return f"status: {text}"
