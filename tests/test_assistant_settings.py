"""Tests for assistant reasoning configuration helpers."""

from pi_assistant.config.assistant_settings import _normalize_reasoning_effort


def test_normalize_reasoning_retains_minimal_choice() -> None:
    """`minimal` should remain selectable when allowed by the model."""

    allowed = ("minimal", "low", "medium", "high")
    assert _normalize_reasoning_effort("minimal", allowed) == "minimal"
