import io
import os
from typing import cast

import pytest

from pi_assistant.config import wake_word


def test_wake_word_label_prefers_phrase_over_label() -> None:
    config = cast(dict[str, object], {"phrase": " Hey Jarvis ", "label": "Jarvis"})

    assert wake_word._wake_word_label(config) == "Hey Jarvis"


def test_wake_word_label_falls_back_to_label() -> None:
    config = cast(dict[str, object], {"label": "  Jarvis "})

    assert wake_word._wake_word_label(config) == "Jarvis"


def test_wake_word_label_uses_aliases_when_missing() -> None:
    config = cast(dict[str, object], {"aliases": ["  hey rhasspy", ""]})

    assert wake_word._wake_word_label(config) == "hey rhasspy"


def test_normalize_wake_word_token_strips_and_lowercases() -> None:
    assert wake_word._normalize_wake_word_token(" Hey-R PI ") == "hey_r_pi"


def test_match_wake_word_name_accepts_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wake_word,
        "_WAKE_MODELS",
        {"hey_rhasspy": {"aliases": ["Jarvis"], "phrase": " Hey Rhasspy "}},
        raising=False,
    )

    assert wake_word._match_wake_word_name("jarvis") == "hey_rhasspy"


def test_default_wake_word_name_returns_first_when_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wake_word,
        "_WAKE_MODELS",
        {"hey_alpha": {}, "hey_beta": {}},
        raising=False,
    )
    monkeypatch.setattr(
        wake_word,
        "_WAKE",
        {"default_model_name": "unknown"},
        raising=False,
    )

    assert wake_word._default_wake_word_name() == "hey_alpha"


def test_resolve_wake_word_name_normalizes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wake_word,
        "_WAKE_MODELS",
        {"hey_pi": {"aliases": ["Jarvis"]}},
        raising=False,
    )
    monkeypatch.setattr(
        wake_word,
        "_WAKE",
        {"default_model_name": "hey_pi"},
        raising=False,
    )
    monkeypatch.setenv("WAKE_WORD_NAME", "jarvis")
    monkeypatch.setattr(
        wake_word,
        "_prompt_for_wake_word_choice",
        lambda default: pytest.fail("prompt should not run"),
    )

    resolved = wake_word._resolve_wake_word_name()

    assert resolved == "hey_pi"
    assert os.environ["WAKE_WORD_NAME"] == "hey_pi"


def test_resolve_wake_word_name_prompts_and_persists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wake_word,
        "_WAKE_MODELS",
        {"hey_pi": {"phrase": "Hey Pi"}},
        raising=False,
    )
    monkeypatch.setattr(
        wake_word,
        "_WAKE",
        {"default_model_name": "hey_pi"},
        raising=False,
    )
    monkeypatch.setenv("WAKE_WORD_NAME", "invalid")
    monkeypatch.setattr(wake_word.sys, "stderr", io.StringIO(), raising=False)

    def _prompt(default: str) -> str:
        return "hey_pi"

    monkeypatch.setattr(
        wake_word,
        "_prompt_for_wake_word_choice",
        _prompt,
        raising=False,
    )
    persisted: dict[str, str] = {}

    def record_choice(name: str) -> None:
        persisted["name"] = name

    monkeypatch.setattr(wake_word, "_persist_wake_word_choice", record_choice, raising=False)

    resolved = wake_word._resolve_wake_word_name()

    assert resolved == "hey_pi"
    assert persisted["name"] == "hey_pi"


def test_wake_path_from_config_prefers_selected_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wake_word,
        "_SELECTED_WAKE",
        {"model_path": "custom.tflite"},
        raising=False,
    )
    assert wake_word._wake_path_from_config("model_path", "fallback.tflite") == "custom.tflite"

    monkeypatch.setattr(wake_word, "_SELECTED_WAKE", {}, raising=False)
    assert wake_word._wake_path_from_config("model_path", "fallback.tflite") == "fallback.tflite"


def test_persist_wake_word_choice_sets_env_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WAKE_WORD_NAME", raising=False)
    recorded: dict[str, str] = {}
    stderr = io.StringIO()
    monkeypatch.setattr(wake_word.sys, "stderr", stderr, raising=False)
    monkeypatch.setattr(
        wake_word,
        "_persist_env_value",
        lambda key, value: recorded.update({key: value}),
        raising=False,
    )

    wake_word._persist_wake_word_choice("hey_pi")

    assert os.environ["WAKE_WORD_NAME"] == "hey_pi"
    assert recorded == {"WAKE_WORD_NAME": "hey_pi"}
    assert "Saved WAKE_WORD_NAME=hey_pi" in stderr.getvalue()
