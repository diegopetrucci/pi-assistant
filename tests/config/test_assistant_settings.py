"""Tests for assistant reasoning configuration helpers."""

import io

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


def test_prompt_for_reasoning_effort_returns_none_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeStdin(io.StringIO):
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(assistant_settings_module.sys, "stdin", _FakeStdin())

    result = assistant_settings_module._prompt_for_reasoning_effort("low", ("low", "high"))

    assert result is None


def test_coerce_assistant_model_key_supports_aliases() -> None:
    assert assistant_settings_module._coerce_assistant_model_key("  FAST ") == "mini"
    assert assistant_settings_module._coerce_assistant_model_key("5") == "5.1"
    assert assistant_settings_module._coerce_assistant_model_key("") is None


def test_normalize_assistant_model_choice_accepts_raw_model_id() -> None:
    mini_model = str(assistant_settings_module.ASSISTANT_MODEL_REGISTRY["mini"]["id"])
    assert assistant_settings_module.normalize_assistant_model_choice(mini_model) == mini_model


def test_resolve_assistant_model_prefers_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASSISTANT_MODEL", "custom-model")

    result = assistant_settings_module._resolve_assistant_model("default-model")

    assert result == "custom-model"


def test_resolve_assistant_model_persists_prompted_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASSISTANT_MODEL", raising=False)
    monkeypatch.setattr(assistant_settings_module.sys, "stderr", io.StringIO())
    monkeypatch.setattr(
        assistant_settings_module, "_prompt_for_assistant_model", lambda default: "prompted-model"
    )
    persisted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        assistant_settings_module,
        "_persist_env_value",
        lambda key, value: persisted.append((key, value)),
    )

    result = assistant_settings_module._resolve_assistant_model("fallback-model")

    assert result == "prompted-model"
    assert persisted == [("ASSISTANT_MODEL", "prompted-model")]
    assert assistant_settings_module.os.environ["ASSISTANT_MODEL"] == "prompted-model"


def test_resolve_reasoning_effort_warns_on_invalid_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ASSISTANT_REASONING_EFFORT", "turbo")
    monkeypatch.setattr(
        assistant_settings_module, "_prompt_for_reasoning_effort", lambda *_, **__: None
    )

    result = assistant_settings_module._resolve_reasoning_effort("low", ("low", "medium"))

    assert result == "low"
    err = capsys.readouterr().err
    assert "Invalid ASSISTANT_REASONING_EFFORT" in err
