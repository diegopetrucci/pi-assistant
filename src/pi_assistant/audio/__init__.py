"""Audio capture and playback utilities with lazy imports to avoid cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["AudioCapture", "SpeechPlayer"]


def __getattr__(name: str):
    if name == "AudioCapture":
        from .capture import AudioCapture as _AudioCapture

        return _AudioCapture
    if name == "SpeechPlayer":
        from .playback import SpeechPlayer as _SpeechPlayer

        return _SpeechPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .capture import AudioCapture as AudioCapture
    from .playback import SpeechPlayer as SpeechPlayer
