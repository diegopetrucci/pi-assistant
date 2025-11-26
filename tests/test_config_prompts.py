import importlib
import io
import os
from typing import Iterator

import pytest


def _load_config_modules():
    # Ensure interactive prompts remain dormant during module import in tests.
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    os.environ.setdefault("LOCATION_NAME", "Test City")
    os.environ.setdefault("ASSISTANT_MODEL", "gpt-5-nano-2025-08-07")
    os.environ.setdefault("ASSISTANT_REASONING_EFFORT", "low")

    assistant_module = importlib.import_module("pi_assistant.config.assistant_settings")
    base_module = importlib.import_module("pi_assistant.config.base")
    return importlib.reload(assistant_module), importlib.reload(base_module)


assistant_settings, base = _load_config_modules()


class _FakeInput:
    """Iterates over pre-seeded responses to emulate user input."""

    def __init__(self, responses: list[str]):
        self._iterator: Iterator[str] = iter(responses)

    def __call__(self, _prompt: str = "") -> str:
        try:
            return next(self._iterator)
        except StopIteration:  # pragma: no cover - defensive guard
            raise AssertionError("No more fake input values available") from None


def _patch_tty_streams(monkeypatch: pytest.MonkeyPatch, module) -> None:
    """Force stdin to look like a TTY and capture stderr output."""

    class _FakeStdin(io.StringIO):
        def isatty(self) -> bool:  # pragma: no cover - trivial
            return True

    monkeypatch.setattr(module.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(module.sys, "stderr", io.StringIO())


def test_prompt_for_assistant_model_accepts_default_on_enter(monkeypatch: pytest.MonkeyPatch):
    _patch_tty_streams(monkeypatch, assistant_settings)
    monkeypatch.setattr("builtins.input", _FakeInput([""]))

    default_model = str(assistant_settings._ASSISTANT_MODEL_CHOICES["mini"]["value"])
    selected = assistant_settings._prompt_for_assistant_model(default_model)

    assert selected == default_model


def test_prompt_for_assistant_model_accepts_non_mini_default(monkeypatch: pytest.MonkeyPatch):
    _patch_tty_streams(monkeypatch, assistant_settings)
    monkeypatch.setattr("builtins.input", _FakeInput([""]))

    default_model = str(assistant_settings._ASSISTANT_MODEL_CHOICES["5.1"]["value"])
    selected = assistant_settings._prompt_for_assistant_model(default_model)

    assert selected == default_model


def test_prompt_for_reasoning_effort_accepts_default_on_enter(monkeypatch: pytest.MonkeyPatch):
    _patch_tty_streams(monkeypatch, assistant_settings)
    monkeypatch.setattr("builtins.input", _FakeInput([""]))

    allowed = ("minimal", "low", "medium")
    selected = assistant_settings._prompt_for_reasoning_effort("low", allowed)

    assert selected == "low"


def test_prompt_for_reasoning_effort_invalid_choice_defaults_to_normalized_default(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_tty_streams(monkeypatch, assistant_settings)
    monkeypatch.setattr("builtins.input", _FakeInput(["extreme"]))

    allowed = ("minimal", "low", "high")
    selected = assistant_settings._prompt_for_reasoning_effort("minimal", allowed)

    assert selected == "minimal"


def test_prompt_for_location_name_requires_input(monkeypatch: pytest.MonkeyPatch):
    _patch_tty_streams(monkeypatch, base)
    monkeypatch.setattr("builtins.input", _FakeInput(["", "   ", "Lisbon, PT"]))

    persisted: list[tuple[str, str]] = []

    def _fake_persist(key: str, value: str) -> None:
        persisted.append((key, value))

    monkeypatch.setattr(base, "_persist_env_value", _fake_persist)
    monkeypatch.setenv("LOCATION_NAME", "Original City")

    location = base._prompt_for_location_name()

    assert location == "Lisbon, PT"
    assert persisted == [("LOCATION_NAME", "Lisbon, PT")]
    assert os.environ["LOCATION_NAME"] == "Lisbon, PT"


def test_prompt_for_location_name_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch):
    _patch_tty_streams(monkeypatch, base)

    def _raise_keyboard_interrupt(_prompt: str = "") -> str:  # pragma: no cover - trivial helper
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

    persisted: list[str] = []
    monkeypatch.setattr(base, "_persist_env_value", lambda *args: persisted.append("called"))
    original = "Original City"
    monkeypatch.setenv("LOCATION_NAME", original)

    location = base._prompt_for_location_name()

    assert location is None
    assert not persisted
    assert os.environ["LOCATION_NAME"] == original
