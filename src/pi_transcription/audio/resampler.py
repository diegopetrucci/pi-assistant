"""
Lightweight linear-interpolation resampler for streaming PCM audio.
"""

from __future__ import annotations

import numpy as np


class LinearResampler:
    """Incremental linear-interpolation resampler for mono PCM16 audio."""

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Sample rates must be positive integers.")

        self.source_rate = source_rate
        self.target_rate = target_rate
        self._step = source_rate / target_rate if target_rate else 1.0
        self._buffer = np.array([], dtype=np.float32)
        self._position = 0.0

    def process(self, audio_bytes: bytes) -> np.ndarray:
        """Return resampled PCM16 samples as a numpy array."""

        if self.source_rate == self.target_rate:
            return np.frombuffer(audio_bytes, dtype=np.int16).copy()

        if not audio_bytes:
            return np.array([], dtype=np.int16)

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if self._buffer.size:
            self._buffer = np.concatenate((self._buffer, samples))
        else:
            self._buffer = samples.copy()

        outputs = []
        pos = self._position
        step = self._step
        buf_len = self._buffer.size

        # Need at least two samples for interpolation
        while buf_len >= 2 and pos + 1 < buf_len:
            idx = int(pos)
            frac = pos - idx
            next_idx = idx + 1
            sample = self._buffer[idx] * (1.0 - frac) + self._buffer[next_idx] * frac
            outputs.append(sample)
            pos += step
            # buf_len stays constant until we discard samples outside the loop

        consumed = int(pos)
        if consumed > 0:
            self._buffer = self._buffer[consumed:]
            pos -= consumed

        self._position = pos

        if not outputs:
            return np.array([], dtype=np.int16)

        clipped = np.clip(np.array(outputs, dtype=np.float32), -32768, 32767)
        return clipped.astype(np.int16)

    def reset(self) -> None:
        """Clear accumulated samples."""

        self._buffer = np.array([], dtype=np.float32)
        self._position = 0.0
