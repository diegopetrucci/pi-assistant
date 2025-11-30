"""Signal-processing helpers shared across modules."""

from __future__ import annotations

import numpy as np

__all__ = ["calculate_rms"]


def calculate_rms(audio_bytes: bytes) -> float:
    """Compute the root-mean-square amplitude for a PCM16 chunk."""

    if not audio_bytes:
        return 0.0

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0

    float_samples = samples.astype(np.float32)
    return float(np.sqrt(np.mean(float_samples**2)))
