import asyncio

import numpy as np
import pytest

from pi_transcription.cli import controller
from pi_transcription.wake_word import WakeWordDetection


class FakeCapture:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def get_audio_chunk(self):
        return await self.queue.get()


class FakeWebSocket:
    def __init__(self):
        self.sent_chunks = []

    async def send_audio_chunk(self, chunk):
        self.sent_chunks.append(chunk)


class DummyTranscriptBuffer:
    def __init__(self):
        self.started = 0
        self.cleared = []
        self.appended = []

    async def start_turn(self):
        self.started += 1

    async def clear_current_turn(self, reason):
        self.cleared.append(reason)

    async def append_transcript(self, *args, **kwargs):
        self.appended.append(args)
        return None

    async def finalize_turn(self):
        return None


async def _shutdown_task(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_run_audio_controller_force_always_on_handles_stop(monkeypatch):
    capture = FakeCapture()
    capture.queue.put_nowait(np.zeros(32, dtype=np.int16).tobytes())
    capture.queue.put_nowait(np.ones(32, dtype=np.int16).tobytes())
    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    scheduled = []

    def fake_schedule(tb, assistant, player):
        scheduled.append(True)
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,
            force_always_on=True,
            transcript_buffer=transcript_buffer,
            assistant=assistant,
            speech_player=speech_player,
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    await asyncio.sleep(0.05)
    stop_signal.set()
    capture.queue.put_nowait(np.zeros(32, dtype=np.int16).tobytes())
    await asyncio.sleep(0.05)
    await _shutdown_task(task)

    assert transcript_buffer.started >= 1
    assert "manual stop command" in transcript_buffer.cleared
    assert not scheduled  # manual stop path does not schedule a response
    assert len(ws_client.sent_chunks) >= 1


@pytest.mark.asyncio
async def test_run_audio_controller_streams_after_wake_word_and_auto_stop(monkeypatch):
    capture = FakeCapture()
    first_chunk = (np.ones(32, dtype=np.int16) * 20000).tobytes()
    second_chunk = (np.ones(32, dtype=np.int16) * 30000).tobytes()
    silence_chunk = np.zeros(32, dtype=np.int16).tobytes()
    capture.queue.put_nowait(first_chunk)
    capture.queue.put_nowait(second_chunk)
    for _ in range(40):
        capture.queue.put_nowait(silence_chunk)
    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    detections = [
        WakeWordDetection(score=0.1, triggered=False),
        WakeWordDetection(score=0.9, triggered=True),
        WakeWordDetection(score=0.2, triggered=False),
    ]

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def process_chunk(self, chunk):
            idx = min(self.calls, len(detections) - 1)
            self.calls += 1
            return detections[idx]

        def reset_detection(self):
            return None

    class DummyPreRollBuffer:
        def __init__(self, *args, **kwargs):
            self.buffer = bytearray()

        def add(self, chunk):
            self.buffer.extend(chunk)

        def flush(self):
            data = bytes(self.buffer)
            self.buffer.clear()
            return data

        def clear(self):
            self.buffer.clear()

    class DummyResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = False

        def process(self, audio_bytes):
            if isinstance(audio_bytes, bytes):
                return np.frombuffer(audio_bytes, dtype=np.int16)
            return audio_bytes

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(controller, "PreRollBuffer", DummyPreRollBuffer)
    monkeypatch.setattr(controller, "LinearResampler", DummyResampler)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", True)
    monkeypatch.setattr(controller, "AUTO_STOP_SILENCE_THRESHOLD", 5000)
    monkeypatch.setattr(controller, "AUTO_STOP_MAX_SILENCE_SECONDS", 0.01)

    scheduled = []

    def fake_schedule(tb, assistant, player):
        scheduled.append(True)
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,
            force_always_on=False,
            transcript_buffer=transcript_buffer,
            assistant=assistant,
            speech_player=speech_player,
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    await asyncio.sleep(0.2)
    await _shutdown_task(task)

    assert transcript_buffer.started >= 1
    assert ws_client.sent_chunks  # pre-roll and live chunks sent
    assert scheduled  # auto-stop transition scheduled a response
