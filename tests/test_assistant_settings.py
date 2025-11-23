"""Tests for assistant reasoning configuration helpers."""

import pytest

from pi_assistant.config import assistant_settings as assistant_settings_module
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


def test_reasoning_choices_for_gpt41_are_empty() -> None:
    assert reasoning_effort_choices_for_model("gpt-4.1-2025-04-14") == ()


def test_resolve_reasoning_effort_returns_none_when_model_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ASSISTANT_REASONING_EFFORT", "medium")

    result = assistant_settings_module._resolve_reasoning_effort("low", ())

    assert result is None
    err = capsys.readouterr().err
    assert "ASSISTANT_REASONING_EFFORT is ignored" in err
    assert "does not support reasoning tokens" in err


def test_resolve_reasoning_effort_returns_valid_choice_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASSISTANT_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(
        assistant_settings_module, "_prompt_for_reasoning_effort", lambda *_, **__: None
    )

    result = assistant_settings_module._resolve_reasoning_effort(
        "medium", ("low", "medium", "high")
    )

    assert result == "medium"
