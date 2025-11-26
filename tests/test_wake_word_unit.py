from typing import Any, cast

import pytest

from pi_assistant import wake_word
from pi_assistant.wake_word import WakeWordEngine, WakeWordEngineOptions

DEFAULT_THRESHOLD = 0.1
DEFAULT_CONSECUTIVE = 3
OVERRIDE_THRESHOLD = 0.8
OVERRIDE_CONSECUTIVE = 5


class _DummyModel:
    def __init__(self):
        self.models = {"dummy": object()}

    def predict(self, audio_bytes):
        return {"dummy": 0.0}


def _fake_model_factory(**kwargs):
    return _DummyModel()


@pytest.fixture(autouse=True)
def _stub_wake_word_dependencies(monkeypatch):
    monkeypatch.setattr(wake_word, "_require_model_factory", lambda: _fake_model_factory)
    monkeypatch.setattr(
        WakeWordEngine,
        "_load_model",
        lambda self, *args, **kwargs: _DummyModel(),
    )


def test_wake_word_engine_rejects_invalid_override_keys():
    bad_overrides = cast(Any, {"unsupported_flag": True})
    with pytest.raises(TypeError, match="Invalid wake-word override"):
        WakeWordEngine("primary-model.onnx", **bad_overrides)


def test_wake_word_engine_overrides_do_not_mutate_options():
    options = WakeWordEngineOptions(
        threshold=DEFAULT_THRESHOLD,
        consecutive_required=DEFAULT_CONSECUTIVE,
    )

    engine = WakeWordEngine(
        "primary-model.onnx",
        options=options,
        threshold=OVERRIDE_THRESHOLD,
        consecutive_required=OVERRIDE_CONSECUTIVE,
    )

    assert options.threshold == DEFAULT_THRESHOLD
    assert options.consecutive_required == DEFAULT_CONSECUTIVE
    assert engine.threshold == OVERRIDE_THRESHOLD
    assert engine.consecutive_required == OVERRIDE_CONSECUTIVE
