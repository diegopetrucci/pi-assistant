"""Minimal helper for playing PCM16 assistant audio replies."""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd


class SpeechPlayer:
    """Serializes playback so assistant audio doesn't overlap and allows stop commands."""

    def __init__(self, default_sample_rate: int = 24000):
        self._default_sample_rate = default_sample_rate
        self._play_lock = asyncio.Lock()
        self._is_playing = asyncio.Event()

    async def play(self, audio_bytes: bytes, sample_rate: Optional[int] = None) -> None:
        """Play PCM16 audio without blocking the event loop."""

        if not audio_bytes:
            return

        async with self._play_lock:
            self._is_playing.set()
            try:
                await asyncio.to_thread(
                    self._play_blocking, audio_bytes, sample_rate or self._default_sample_rate
                )
            finally:
                self._is_playing.clear()

    async def stop(self) -> bool:
        """Stop any in-progress playback, returning True if something was interrupted."""

        if not self._is_playing.is_set():
            return False
        await asyncio.to_thread(sd.stop)
        return True

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
