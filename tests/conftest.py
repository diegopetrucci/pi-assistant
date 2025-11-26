import importlib
import os
import sys
from types import ModuleType
from typing import Any, Tuple


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

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")
