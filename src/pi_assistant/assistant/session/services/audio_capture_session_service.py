"""Start and stop microphone capture."""

from __future__ import annotations

import asyncio

from pi_assistant.audio import AudioCapture

from .session_service import BaseSessionService


class AudioCaptureSessionService(BaseSessionService):
    """Drive the AudioCapture stream within the session lifecycle."""

    def __init__(self, audio_capture: AudioCapture):
        super().__init__("capture")
        self._audio_capture = audio_capture

    async def _start(self) -> None:
        loop = asyncio.get_running_loop()
        self._audio_capture.start_stream(loop)

    async def _stop(self) -> None:
        self._audio_capture.stop_stream()
