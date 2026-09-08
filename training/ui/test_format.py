# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for the formatting helpers, the step-dict readers and the panel-height fit.
"""

from __future__ import annotations


from training.ui.format import (
    depth_losses,
    as_float,
    event_line,
    fit_panel_heights,
    floats,
    format_duration,
    format_metric,
    format_tokens,
    known_metrics,
    status_line,
    validation_line,
)


def test_format_duration() -> None:
    assert format_duration(None) == "—"
    assert format_duration(float("inf")) == "—"
    assert format_duration(-1) == "—"
    assert format_duration(0) == "0:00:00"
    assert format_duration(3_725.9) == "1:02:05"
    assert format_duration(90_061) == "1d 01:01:01"


def test_format_tokens() -> None:
    assert format_tokens(999) == "999"
    assert format_tokens(1_500) == "1.50k"
    assert format_tokens(2_500_000) == "2.50M"
    assert format_tokens(1.234e9) == "1.23B"
    assert format_tokens(3e12) == "3.00T"


def test_format_metric() -> None:
    assert format_metric("loss", 3.14159) == "3.1416"
    assert format_metric("ppl", 23.1) == "23.10"
    assert format_metric("lr", 0.0003) == "3.00e-04"
    assert format_metric("grad_norm", 1.23456) == "1.235"
    assert format_metric("tokens/second", 12345.6) == "12,346"
    assert format_metric("seconds/step", 0.5) == "0.50s"
    assert format_metric("total_tokens", 2e9) == "2.00B"
    assert format_metric("other", 0.123456) == "0.1235"


def test_log_lines_of_the_fallback_and_the_log_file() -> None:
    assert validation_line(10, {"val_loss_4": 3.25, "val_loss": 3.125, "bad": "x"}) == "step 10: validation val_loss_4 3.2500, val_loss 3.1250"
    assert validation_line(3, {}) == "step 3: validation (no losses)"
    assert validation_line(4, {"val_loss": 3.0, "val_loss/finetune-flan": 2.5}) == "step 4: validation val_loss 3.0000, val_loss/finetune-flan 2.5000"
    assert depth_losses({"val_loss_4": 3.25, "val_loss/pretrain-a": 3.0, "val_loss": 3.125}) == {"val_loss_4": 3.25, "val_loss": 3.125}
    assert event_line("saved checkpoint x.pth") == "event: saved checkpoint x.pth"
    assert status_line("evaluating") == "status: evaluating"


def test_fit_panel_heights_shrinks_the_log_panel_first_then_the_events() -> None:
    assert fit_panel_heights(100, 6, 12) == (6, 12), "room for everything"
    assert fit_panel_heights(6 + 12 + 4, 6, 12) == (6, 12), "exactly enough (two borders per panel)"
    assert fit_panel_heights(6 + 8 + 4, 6, 12) == (6, 8), "the log panel gives way first"
    assert fit_panel_heights(3 + 2 + 4, 6, 12) == (3, 2), "then the events panel, the log panel at its minimum"
    assert fit_panel_heights(0, 6, 12) == (1, 2), "never below one event line and MIN_LOG_LINES"
    assert fit_panel_heights(100, 0, 0) == (1, 2), "an empty panel still shows its placeholder line"


def test_step_dict_readers_accept_anything_float_like() -> None:
    class Scalar:  # a one-element tensor: `float()` works through `__float__`
        def __float__(self) -> float:
            return 2.5

    assert as_float(Scalar()) == 2.5 and as_float("3") == 3.0 and as_float("x") is None and as_float(None) is None
    assert floats({"a": 1, "b": "nope", "c": Scalar()}) == {"a": 1.0, "c": 2.5}
    assert known_metrics({"loss": 3, "unknown": 1.0, "lr": "bad", "ppl": Scalar()}) == {"loss": 3.0, "ppl": 2.5}
