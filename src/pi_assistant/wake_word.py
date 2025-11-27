"""
Wake word detection helpers built around the openWakeWord library.
"""

from __future__ import annotations

import importlib
import logging
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, TypedDict, cast

from typing_extensions import Unpack

from pi_assistant.audio.resampler import LinearResampler


class WakeWordModel(Protocol):
    models: Mapping[str, object]

    def predict(self, audio_bytes: Any) -> Mapping[str, float]: ...


ModelFactory = Callable[..., WakeWordModel]

_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - exercised via unit tests with the real dependency
    _MODEL_FACTORY = cast(
        ModelFactory,
        importlib.import_module("openwakeword.model").Model,
    )
except Exception as exc:  # pragma: no cover - handled downstream
    _MODEL_FACTORY = None
    _IMPORT_ERROR = exc

Model = _MODEL_FACTORY


def _require_model_factory() -> ModelFactory:
    factory = Model or _MODEL_FACTORY
    if factory is None:
        raise RuntimeError(
            "openwakeword is not installed. Install the dependency to enable wake-word gating."
        ) from _IMPORT_ERROR
    return factory


class StreamState(Enum):
    """High-level states for the local audio controller."""

    LISTENING = "listening"
    STREAMING = "streaming"


@dataclass
class WakeWordDetection:
    """Container for wake word inference results."""

    score: float = 0.0
    triggered: bool = False


@dataclass(slots=True)
class WakeWordEngineOptions:
    """Configuration defaults for the wake-word detection engine."""

    fallback_model_path: Optional[Path | str] = None
    melspec_model_path: Optional[Path] = None
    embedding_model_path: Optional[Path] = None
    source_sample_rate: int = 24000
    target_sample_rate: int = 16000
    threshold: float = 0.5
    consecutive_required: int = 2


WAKE_WORD_ENGINE_OPTION_FIELDS = frozenset(WakeWordEngineOptions.__dataclass_fields__.keys())


class WakeWordEngineOverrides(TypedDict, total=False):
    fallback_model_path: Optional[Path | str]
    melspec_model_path: Optional[Path]
    embedding_model_path: Optional[Path]
    source_sample_rate: int
    target_sample_rate: int
    threshold: float
    consecutive_required: int


class PreRollBuffer:
    """Stores a rolling window of raw PCM audio for pre-trigger playback."""

    def __init__(
        self,
        max_seconds: float,
        sample_rate: int,
        sample_width: int = 2,
        *,
        channels: int = 1,
    ):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if sample_width <= 0:
            raise ValueError("sample_width must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.max_bytes = int(max_seconds * sample_rate * sample_width * channels)
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


class WakeWordEngine:
    """Thin wrapper around openWakeWord with additional gating logic."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        options: Optional[WakeWordEngineOptions] = None,
        **config: Unpack[WakeWordEngineOverrides],
    ):
        resolved_options = options or WakeWordEngineOptions()
        if config:
            _validate_wake_engine_overrides(config)
            resolved_options = replace(resolved_options, **config)
        factory = _require_model_factory()
        self.threshold = resolved_options.threshold
        self.consecutive_required = max(1, resolved_options.consecutive_required)
        self._consecutive_hits = 0
        self._resampler = LinearResampler(
            resolved_options.source_sample_rate,
            resolved_options.target_sample_rate,
        )
        self._model = self._load_model(
            model_path,
            resolved_options.fallback_model_path,
            melspec_model_path=resolved_options.melspec_model_path,
            embedding_model_path=resolved_options.embedding_model_path,
            factory=factory,
        )
        self._model_label = next(iter(self._model.models.keys()))

    def _load_model(
        self,
        primary: Path | str,
        fallback: Optional[Path | str],
        *,
        melspec_model_path: Optional[Path],
        embedding_model_path: Optional[Path],
        factory: ModelFactory,
    ) -> WakeWordModel:
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
                return factory(
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


def _validate_wake_engine_overrides(overrides: Mapping[str, object]) -> None:
    if not overrides:
        return
    invalid = set(overrides) - set(WAKE_WORD_ENGINE_OPTION_FIELDS)
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise TypeError(f"Invalid wake-word override(s): {joined}")
