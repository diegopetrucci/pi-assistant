from typing import cast

from pi_assistant.config.wake_word import _wake_word_label


def test_wake_word_label_prefers_phrase_over_label() -> None:
    config = cast(dict[str, object], {"phrase": " Hey Jarvis ", "label": "Jarvis"})

    assert _wake_word_label(config) == "Hey Jarvis"


def test_wake_word_label_falls_back_to_label() -> None:
    config = cast(dict[str, object], {"label": "  Jarvis "})

    assert _wake_word_label(config) == "Jarvis"


def test_wake_word_label_uses_aliases_when_missing() -> None:
    config = cast(dict[str, object], {"aliases": ["  hey rhasspy", ""]})

    assert _wake_word_label(config) == "hey rhasspy"
