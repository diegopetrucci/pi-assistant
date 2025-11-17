"""
Streaming PCM16 resampler built on audioop.ratecv.
"""

from __future__ import annotations

import audioop

import numpy as np


class LinearResampler:
    """Incremental resampler for mono PCM16 audio."""

    def __init__(self, source_rate: int, target_rate: int, channels: int = 1):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive integers.")
        if channels <= 0:
            raise ValueError("Channel count must be positive.")

        self.source_rate = source_rate
        self.target_rate = target_rate
        self.channels = channels
        self._state = None
        self._width = 2  # PCM16

    def process(self, audio_bytes: bytes) -> np.ndarray:
        """Return resampled PCM16 samples as a numpy array."""

        if not audio_bytes:
            return np.array([], dtype=np.int16)

        if self.source_rate == self.target_rate:
            return np.frombuffer(audio_bytes, dtype=np.int16).copy()

        converted, self._state = audioop.ratecv(
            audio_bytes,
            self._width,
            self.channels,
            self.source_rate,
            self.target_rate,
            self._state,
        )
        return np.frombuffer(converted, dtype=np.int16).copy()

    def reset(self) -> None:
        """Clear accumulated state."""

        self._state = None
