"""Minimal helper for playing PCM16 assistant audio replies."""

from __future__ import annotations

import asyncio
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ._sounddevice import sounddevice as sd
from .resampler import LinearResampler
from .utils import device_info_dict


class SpeechPlayer:
    """Serializes playback so assistant audio doesn't overlap and allows stop commands."""

    def __init__(
        self,
        default_sample_rate: int = 24000,
        *,
        debug_dump_enabled: bool = False,
        debug_dump_directory: Optional[Path | str] = None,
    ):
        self._default_sample_rate = default_sample_rate
        self._play_lock = asyncio.Lock()
        self._is_playing = asyncio.Event()
        self._resample_warnings: set[int] = set()
        self._override_logged = False
        self._output_device = self._detect_output_device()
        self._playback_sample_rate = self._select_playback_sample_rate(default_sample_rate)
        self._debug_dump_enabled = bool(debug_dump_enabled)
        self._debug_dump_dir = self._prepare_dump_directory(debug_dump_directory)

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
        aligned_bytes = self._ensure_pcm_alignment(audio_bytes)
        if not aligned_bytes:
            return np.array([], dtype=np.int16)

        if self._debug_dump_enabled:
            self._debug_dump_audio(aligned_bytes, max(source_rate, 0))
            self._log_debug_summary(aligned_bytes, source_rate)

        if source_rate == self._playback_sample_rate:
            return np.frombuffer(aligned_bytes, dtype=np.int16).copy()

        resampler = LinearResampler(source_rate, self._playback_sample_rate)
        samples = resampler.process(aligned_bytes)
        self._warn_if_resampling(source_rate, self._playback_sample_rate)
        return samples

    def _ensure_pcm_alignment(self, audio_bytes: bytes) -> bytes:
        if not audio_bytes:
            return b""
        remainder = len(audio_bytes) % 2
        if remainder == 0:
            return audio_bytes
        aligned = audio_bytes[: len(audio_bytes) - remainder]
        dropped = len(audio_bytes) - len(aligned)
        if aligned:
            print(
                (
                    "[AUDIO] Discarding "
                    f"{dropped} trailing byte(s) from assistant audio "
                    "to align PCM16 frames."
                ),
                flush=True,
            )
        else:
            print(
                "[AUDIO] Dropping assistant audio payload with incomplete PCM16 frame alignment.",
                flush=True,
            )
        return aligned

    def _prepare_dump_directory(self, directory: Optional[Path | str]) -> Optional[Path]:
        if not self._debug_dump_enabled:
            return None
        target = Path(directory) if directory else Path("logs") / "audio_dumps"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"[AUDIO] Unable to create audio dump directory '{target}': {exc}",
                flush=True,
            )
            return None
        return target

    def _debug_dump_audio(self, audio_bytes: bytes, source_rate: int) -> None:
        dump_dir = self._debug_dump_dir
        if not dump_dir:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_rate = source_rate if source_rate > 0 else self._default_sample_rate
        resolved_rate = safe_rate or self._playback_sample_rate or 24000
        filepath = dump_dir / f"assistant_{timestamp}_{resolved_rate}Hz.wav"
        try:
            with wave.open(str(filepath), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(resolved_rate)
                wav_file.writeframes(audio_bytes)
        except Exception as exc:
            print(f"[AUDIO] Failed to dump assistant audio to {filepath}: {exc}", flush=True)
        else:
            print(f"[AUDIO] Saved assistant audio dump to {filepath}", flush=True)

    def _log_debug_summary(self, audio_bytes: bytes, source_rate: int) -> None:
        if not self._debug_dump_enabled:
            return
        frame_count = len(audio_bytes) // 2
        resolved_rate = source_rate if source_rate > 0 else self._playback_sample_rate or 1
        duration = frame_count / resolved_rate if resolved_rate > 0 else 0.0
        buffer_view = np.frombuffer(audio_bytes, dtype=np.int16)
        peak = int(np.max(np.abs(buffer_view))) if buffer_view.size else 0
        print(
            (
                f"[AUDIO] Preparing {frame_count} PCM16 samples "
                f"(~{duration:.2f}s) @ {source_rate or self._default_sample_rate} Hz; "
                f"peak amplitude {peak}/32768; playback rate {self._playback_sample_rate} Hz."
            ),
            flush=True,
        )

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
        if isinstance(rate, (int, float)):
            return int(rate)
        if isinstance(rate, str):
            try:
                return int(float(rate))
            except (ValueError, TypeError):
                return None
        return None

    def _is_rate_supported(self, rate: int) -> bool:
        try:
            sd.check_output_settings(device=self._output_device, samplerate=rate, channels=1)
            return True
        except Exception:
            return False

    def _output_device_info(self) -> dict[str, object]:
        try:
            if self._output_device is None:
                raw = sd.query_devices(kind="output")
            else:
                raw = sd.query_devices(self._output_device)
        except Exception:
            return {}
        return device_info_dict(raw)

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
