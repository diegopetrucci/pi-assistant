"""Helpers for preparing microphone chunks for server streaming."""

from __future__ import annotations

from typing import Optional, Protocol, TypeVar

from pi_assistant.audio.resampler import LinearResampler

ResamplerT = TypeVar("ResamplerT", bound=LinearResampler)


class ResamplerFactory(Protocol):
    def __call__(self, source_rate: int, target_rate: int) -> LinearResampler: ...


class AudioChunkPreparer:
    """Resample capture audio to the stream sample rate when needed."""

    def __init__(
        self,
        capture_rate: int,
        stream_rate: int,
        *,
        resampler_factory: Optional[ResamplerFactory] = None,
    ):
        self._resampler: Optional[LinearResampler] = None
        factory = resampler_factory or LinearResampler
        if capture_rate != stream_rate:
            self._resampler = factory(capture_rate, stream_rate)

    @property
    def is_resampling(self) -> bool:
        return self._resampler is not None

    def prepare(self, chunk: bytes) -> bytes:
        """Return PCM16 audio ready for streaming to the server."""

        if not chunk:
            return b""
        if not self._resampler:
            return chunk
        samples = self._resampler.process(chunk)
        return samples.tobytes() if samples.size else b""

    def reset(self) -> None:
        """Clear internal buffers so the next chunk starts fresh."""

        if self._resampler:
            self._resampler.reset()
