import asyncio
from typing import Optional, cast

import pytest

from pi_assistant.assistant.session_services.audio_capture_session_service import (
    AudioCaptureSessionService,
)
from pi_assistant.audio import AudioCapture


class _AudioCaptureStub:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def start_stream(self, loop: asyncio.AbstractEventLoop) -> None:
        self.start_calls += 1
        self.loop = loop

    def stop_stream(self) -> None:
        self.stop_calls += 1


@pytest.mark.asyncio
async def test_audio_capture_session_service_uses_running_loop() -> None:
    capture = _AudioCaptureStub()
    service = AudioCaptureSessionService(cast(AudioCapture, capture))

    await service.start()
    assert capture.loop is asyncio.get_running_loop()
    assert capture.start_calls == 1

    await service.start()
    assert capture.start_calls == 1  # idempotent start

    await service.stop()
    assert capture.stop_calls == 1


@pytest.mark.asyncio
async def test_audio_capture_session_service_stop_is_noop_before_start() -> None:
    capture = _AudioCaptureStub()
    service = AudioCaptureSessionService(cast(AudioCapture, capture))

    await service.stop()
    assert capture.stop_calls == 0
