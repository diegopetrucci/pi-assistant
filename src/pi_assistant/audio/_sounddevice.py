"""
Thin wrapper for ``sounddevice`` that can fall back to a dummy implementation
when the native PortAudio library is unavailable (e.g. on CI runners).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

_ALLOW_DUMMY = os.getenv("PI_TRANSCRIPTION_USE_DUMMY_AUDIO") == "1"


class _DummyStream:
    """Minimal ``InputStream`` stand-in used by tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.started = False
        self._args = args
        self._kwargs = kwargs

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.started = False


class _DummySoundDevice:
    """Simple shim that emulates the subset of sounddevice used in tests."""

    def __init__(self) -> None:
        self.default = SimpleNamespace(device=(0, 1))
        self._input = {
            "index": 0,
            "name": "Dummy Input Device",
            "max_input_channels": 1,
            "default_samplerate": 16000,
        }
        self._output = {
            "index": 1,
            "name": "Dummy Output Device",
            "max_output_channels": 1,
            "default_samplerate": 24000,
        }

    def InputStream(self, *args: Any, **kwargs: Any) -> _DummyStream:
        return _DummyStream(*args, **kwargs)

    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        if kind == "output":
            return self._output
        if device is None:
            return [self._input, self._output]
        if isinstance(device, int):
            if device == self._input["index"]:
                return self._input
            if device == self._output["index"]:
                return self._output
        raise ValueError("Unknown dummy audio device")

    def check_output_settings(self, **_: Any) -> bool:
        return True

    def check_input_settings(self, **_: Any) -> bool:
        return True

    def play(self, *_: Any, **__: Any) -> None:
        return None

    def wait(self) -> None:
        return None

    def stop(self) -> None:
        return None


def _load_sounddevice():
    try:
        import sounddevice as real_sounddevice  # type: ignore

        return real_sounddevice
    except (ImportError, OSError):
        if _ALLOW_DUMMY:
            return _DummySoundDevice()
        raise


sounddevice = _load_sounddevice()

__all__ = ["sounddevice"]
