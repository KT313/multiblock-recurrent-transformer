# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Formatting for the dashboard and its fallback lines, the readers of the step dict, and the panel-height fit."""

from __future__ import annotations

import math
from collections.abc import Mapping

from rich.text import Text

MIN_LOG_LINES = 2  # the log panel never shrinks below this on a short terminal; the events panel goes down to one line

# metric keys of the step dict (`RunLogger.log_step`, today's `train.py` names) shown in the metrics table, with labels
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


def line(text: str, style: str = "") -> Text:
    """One terminal row: never wraps, cropped with an ellipsis; markup in ``text`` is not interpreted."""
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")
