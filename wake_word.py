"""
Wake word detection helpers built around the openWakeWord library.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from openwakeword.model import Model
except Exception as exc:  # pragma: no cover - handled downstream
    Model = None
    _IMPORT_ERROR = exc


class StreamState(Enum):
    """High-level states for the local audio controller."""

    LISTENING = "listening"
    STREAMING = "streaming"


@dataclass
class WakeWordDetection:
    """Container for wake word inference results."""

    score: float = 0.0
    triggered: bool = False


class PreRollBuffer:
    """Stores a rolling window of raw PCM audio for pre-trigger playback."""

    def __init__(self, max_seconds: float, sample_rate: int, sample_width: int = 2):
        self.max_bytes = int(max_seconds * sample_rate * sample_width)
        self._buffer: deque[bytes] = deque()
        self._size = 0

    def add(self, chunk: bytes) -> None:
        """Append a new audio chunk to the buffer."""

        if not chunk:
            return

        self._buffer.append(chunk)
        self._size += len(chunk)

        while self._size > self.max_bytes and self._buffer:
            removed = self._buffer.popleft()
            self._size -= len(removed)

    def flush(self) -> bytes:
        """Return and clear the buffered audio."""

        payload = b"".join(self._buffer)
        self.clear()
        return payload

    def clear(self) -> None:
        """Drop all buffered audio."""

        self._buffer.clear()
        self._size = 0


class _LinearResampler:
    """Simple linear-interpolation resampler for mono PCM16 audio."""

    def __init__(self, source_rate: int, target_rate: int):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._step = source_rate / target_rate if target_rate else 1.0
        self._buffer = np.array([], dtype=np.float32)
        self._position = 0.0

    def process(self, audio_bytes: bytes) -> np.ndarray:
        """Return resampled PCM16 audio suitable for the detector."""

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


class WakeWordEngine:
    """Thin wrapper around openWakeWord with additional gating logic."""

    def __init__(
        self,
        model_path: Path,
        *,
        fallback_model_path: Optional[Path] = None,
        melspec_model_path: Optional[Path] = None,
        embedding_model_path: Optional[Path] = None,
        source_sample_rate: int = 24000,
        target_sample_rate: int = 16000,
        threshold: float = 0.5,
        consecutive_required: int = 2,
    ):
        if Model is None:
            raise RuntimeError(
                "openwakeword is not installed. Install the dependency or use "
                "--force-always-on to bypass wake-word gating."
            ) from _IMPORT_ERROR

        self.threshold = threshold
        self.consecutive_required = max(1, consecutive_required)
        self._consecutive_hits = 0
        self._resampler = _LinearResampler(source_sample_rate, target_sample_rate)
        self._model = self._load_model(
            model_path,
            fallback_model_path,
            melspec_model_path=melspec_model_path,
            embedding_model_path=embedding_model_path,
        )
        self._model_label = next(iter(self._model.models.keys()))

    def _load_model(
        self,
        primary: Path,
        fallback: Optional[Path],
        *,
        melspec_model_path: Optional[Path],
        embedding_model_path: Optional[Path],
    ) -> Model:
        """Load a TFLite model, falling back to ONNX when needed."""

        errors = []
        for path in filter(None, [primary, fallback]):
            resolved = Path(path)
            if not resolved.exists():
                errors.append(f"Missing model file: {resolved}")
                continue

            inference = "onnx" if resolved.suffix.lower() == ".onnx" else "tflite"
            try:
                extra_kwargs = {}
                if (
                    melspec_model_path
                    and embedding_model_path
                    and inference == "onnx"
                    and melspec_model_path.suffix.lower() == ".onnx"
                ):
                    extra_kwargs["melspec_model_path"] = str(melspec_model_path)
                    extra_kwargs["embedding_model_path"] = str(embedding_model_path)
                return Model(
                    wakeword_models=[str(resolved)],
                    inference_framework=inference,
                    **extra_kwargs,
                )
            except Exception as exc:  # pragma: no cover - depends on local runtimes
                logging.getLogger(__name__).warning(
                    "Failed to load wake-word model %s via %s: %s", resolved, inference, exc
                )
                errors.append(f"{resolved} ({inference}): {exc}")

        joined = "; ".join(errors) if errors else "Unknown error"
        raise RuntimeError(f"Unable to initialize openWakeWord model: {joined}")

    def process_chunk(self, audio_bytes: bytes) -> WakeWordDetection:
        """Run detection on a chunk of 24 kHz PCM16 audio."""

        downsampled = self._resampler.process(audio_bytes)
        if downsampled.size == 0:
            return WakeWordDetection()

        predictions = self._model.predict(downsampled)
        score = float(predictions.get(self._model_label, 0.0))
        triggered = False

        if score >= self.threshold:
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.consecutive_required:
                triggered = True
                self._consecutive_hits = 0
        else:
            self._consecutive_hits = 0

        return WakeWordDetection(score=score, triggered=triggered)

    def reset_detection(self) -> None:
        """Clear any accumulated state used for the confidence guard."""

        self._consecutive_hits = 0
