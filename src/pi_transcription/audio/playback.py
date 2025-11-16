"""Minimal helper for playing PCM16 assistant audio replies."""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd


class SpeechPlayer:
    """Serializes playback so assistant audio doesn't overlap itself."""

    def __init__(self, default_sample_rate: int = 24000):
        self._default_sample_rate = default_sample_rate
        self._lock = asyncio.Lock()

    async def play(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> None:
        """Play PCM16 audio without blocking the event loop."""

        if not audio_bytes:
            return

        async with self._lock:
            await asyncio.to_thread(
                self._play_blocking, audio_bytes, sample_rate or self._default_sample_rate
            )

    @staticmethod
    def _play_blocking(audio_bytes: bytes, sample_rate: int) -> None:
        """Convert PCM16 bytes into float32 samples and play them synchronously."""

        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        if samples.size == 0:
            return

        float_samples = samples.astype(np.float32) / 32768.0
        try:
            sd.play(float_samples, samplerate=sample_rate)
            sd.wait()
        except Exception:  # pragma: no cover - depends on host audio stack
            raise


__all__ = ["SpeechPlayer"]
