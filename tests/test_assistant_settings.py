"""Tests for assistant reasoning configuration helpers."""

from pi_assistant.config.assistant_settings import (
    _normalize_reasoning_effort,
    reasoning_effort_choices_for_model,
)


def test_normalize_reasoning_retains_minimal_choice() -> None:
    """`minimal` should remain selectable when allowed by the model."""

    allowed = ("minimal", "low", "medium", "high")
    assert _normalize_reasoning_effort("minimal", allowed) == "minimal"


def test_reasoning_choices_for_nano_exclude_minimal() -> None:
    choices = reasoning_effort_choices_for_model("gpt-5-nano-2025-08-07")
    assert choices == ("low", "medium", "high")
