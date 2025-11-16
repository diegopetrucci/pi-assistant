"""Minimal helper for playing PCM16 assistant audio replies."""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd

from .resampler import LinearResampler


class SpeechPlayer:
    """Serializes playback so assistant audio doesn't overlap and allows stop commands."""

    def __init__(self, default_sample_rate: int = 24000):
        self._default_sample_rate = default_sample_rate
        self._play_lock = asyncio.Lock()
        self._is_playing = asyncio.Event()
        self._resample_warnings: set[int] = set()
        self._override_logged = False
        self._output_device = self._detect_output_device()
        self._playback_sample_rate = self._select_playback_sample_rate(default_sample_rate)

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

    def _play_blocking(self, audio_bytes: bytes, source_rate: int) -> None:
        """Convert PCM16 bytes into float32 samples and play them synchronously."""

        if source_rate <= 0:
            source_rate = self._default_sample_rate or self._playback_sample_rate

        samples = self._prepare_samples(audio_bytes, source_rate)
        if samples.size == 0:
            return

        float_samples = samples.astype(np.float32) / 32768.0
        try:
            sd.play(
                float_samples,
                samplerate=self._playback_sample_rate,
                device=self._output_device,
            )
            sd.wait()
        except Exception:  # pragma: no cover - depends on host audio stack
            raise

    def _prepare_samples(self, audio_bytes: bytes, source_rate: int) -> np.ndarray:
        if source_rate == self._playback_sample_rate:
            return np.frombuffer(audio_bytes, dtype=np.int16).copy()

        resampler = LinearResampler(source_rate, self._playback_sample_rate)
        samples = resampler.process(audio_bytes)
        self._warn_if_resampling(source_rate, self._playback_sample_rate)
        return samples

    def _detect_output_device(self) -> Optional[int]:
        device = sd.default.device
        if isinstance(device, (list, tuple)):
            candidate = device[1]
        else:
            candidate = device
        return candidate if isinstance(candidate, int) else None

    def _select_playback_sample_rate(self, preferred_rate: int) -> int:
        candidates: list[int] = []
        for rate in (
            preferred_rate,
            self._device_default_sample_rate(),
            48000,
            44100,
            32000,
        ):
            if not rate:
                continue
            rate_int = int(rate)
            if rate_int <= 0 or rate_int in candidates:
                continue
            candidates.append(rate_int)

        for rate in candidates:
            if self._is_rate_supported(rate):
                if rate != preferred_rate:
                    self._log_playback_override(preferred_rate, rate)
                return rate

        return preferred_rate

    def _device_default_sample_rate(self) -> Optional[int]:
        info = self._output_device_info()
        if not info:
            return None
        rate = info.get("default_samplerate")
        try:
            return int(rate)
        except (TypeError, ValueError):
            return None

    def _is_rate_supported(self, rate: int) -> bool:
        try:
            sd.check_output_settings(device=self._output_device, samplerate=rate, channels=1)
            return True
        except Exception:
            return False

    def _output_device_info(self) -> dict:
        try:
            if self._output_device is None:
                return sd.query_devices(kind="output")
            return sd.query_devices(self._output_device)
        except Exception:
            return {}

    def _log_playback_override(self, requested: int, selected: int) -> None:
        if self._override_logged:
            return
        self._override_logged = True
        device_info = self._output_device_info()
        device_name = device_info.get("name", "default output")
        print(
            f"[AUDIO] Playback device '{device_name}' does not support {requested} Hz; "
            f"assistant audio will play at {selected} Hz.",
            flush=True,
        )

    def _warn_if_resampling(self, source: int, target: int) -> None:
        if source in self._resample_warnings:
            return
        self._resample_warnings.add(source)
        message = (
            f"[AUDIO] Resampling assistant audio {source} Hz -> {target} Hz "
            "for playback compatibility."
        )
        print(message, flush=True)


__all__ = ["SpeechPlayer"]
