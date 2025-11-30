"""Silence detection helpers for the controller."""

from __future__ import annotations

from collections.abc import Callable

from pi_assistant.audio.processing.metrics import calculate_rms
from pi_assistant.config import CHANNELS as DEFAULT_CHANNELS


class SilenceTracker:
    """Track speech activity and surface auto-stop events."""

    def __init__(
        self,
        *,
        silence_threshold: float,
        max_silence_seconds: float,
        sample_rate: int,
        channels: int = DEFAULT_CHANNELS,
        rms_func: Callable[[bytes], float] = calculate_rms,
    ):
        self._threshold = silence_threshold
        self._max_silence = max_silence_seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self._calculate_rms = rms_func
        self._heard_speech = False
        self._silence_duration = 0.0

    def reset(self) -> None:
        self._heard_speech = False
        self._silence_duration = 0.0

    def clear_silence(self) -> None:
        """Reset only the accumulated silence window."""

        self._silence_duration = 0.0

    @property
    def heard_speech(self) -> bool:
        """Return True once any speech has been detected in the current window."""

        return self._heard_speech

    @property
    def silence_duration(self) -> float:
        """Expose the accumulated silence time since the most recent speech chunk.

        Returns 0.0 if no speech has been detected yet.
        """

        return self._silence_duration if self._heard_speech else 0.0

    def has_observed_silence(self, minimum_seconds: float) -> bool:
        """Return True when silence has persisted for at least ``minimum_seconds``."""

        if minimum_seconds <= 0:
            return True
        return self.silence_duration >= minimum_seconds

    def observe(self, chunk: bytes) -> bool:
        """Return True when accumulated silence exceeds the threshold."""

        if not chunk:
            return False

        if self._channels <= 0:
            return False

        frames = len(chunk) / (2.0 * self._channels)
        chunk_duration = frames / self._sample_rate if frames and self._sample_rate > 0 else 0.0
        rms = self._calculate_rms(chunk)

        if rms >= self._threshold:
            self._heard_speech = True
            self._silence_duration = 0.0
            return False

        if not self._heard_speech:
            return False

        self._silence_duration += chunk_duration
        return self._silence_duration >= self._max_silence
