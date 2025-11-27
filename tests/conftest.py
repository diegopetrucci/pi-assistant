import importlib
import os
import sys
from types import ModuleType
from typing import Any, Tuple

import pytest

_TEST_ENV_DEFAULTS = {
    "OPENAI_API_KEY": "test-key",
    "LOCATION_NAME": "Test City",
    "ASSISTANT_MODEL": "gpt-5-nano-2025-08-07",
    "ASSISTANT_REASONING_EFFORT": "low",
    "VERBOSE_LOG_CAPTURE_ENABLED": "0",
}


class _AudioOpStub(ModuleType):
    def __init__(self) -> None:
        super().__init__("audioop")

    @staticmethod
    def ratecv(*ratecv_args: Any) -> Tuple[bytes, Any]:
        audio_bytes, *_, state = ratecv_args
        return audio_bytes, state


try:
    importlib.import_module("audioop")
except ModuleNotFoundError:
    sys.modules["audioop"] = _AudioOpStub()

for key, value in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


@pytest.fixture(autouse=True)
def set_test_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep critical environment variables stable across tests."""

    for key, value in _TEST_ENV_DEFAULTS.items():
        monkeypatch.setenv(key, value)
