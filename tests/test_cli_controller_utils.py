import asyncio

import numpy as np
import pytest

from pi_assistant.audio.metrics import calculate_rms
from pi_assistant.cli import controller

EXPECTED_RMS_MIN = 26000
EXPECTED_RMS_MAX = 28000


def test_calculate_rms_handles_empty_bytes():
    assert calculate_rms(b"") == 0.0
    assert calculate_rms(b"\x00\x00" * 0) == 0.0


def test_calculate_rms_returns_expected_value():
    samples = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    rms = calculate_rms(samples)

    assert EXPECTED_RMS_MIN < rms < EXPECTED_RMS_MAX


def _chunk_with_duration(seconds: float, amplitude: int = 0) -> bytes:
    frames = max(1, int(seconds * controller.SAMPLE_RATE))
    samples = np.full(frames * controller.CHANNELS, amplitude, dtype=np.int16)
    return samples.tobytes()


def test_audio_chunk_preparer_resamples_when_needed(monkeypatch):
    created = []

    class DummyResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = False
            created.append(self)

        def process(self, chunk):
            data = np.frombuffer(chunk, dtype=np.int16)
            return (data // 2).astype(np.int16)

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(controller, "LinearResampler", DummyResampler)
    preparer = controller.AudioChunkPreparer(
        48000,
        16000,
        resampler_factory=controller.LinearResampler,
    )

    assert preparer.is_resampling is True
    processed = preparer.prepare((np.ones(4, dtype=np.int16) * 4).tobytes())
    assert np.array_equal(np.frombuffer(processed, dtype=np.int16), np.ones(4, dtype=np.int16) * 2)

    preparer.reset()
    assert created[0].reset_called is True


def test_silence_tracker_requires_multiple_silence_chunks():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )

    silence = _chunk_with_duration(0.005, amplitude=0)
    speech = _chunk_with_duration(0.02, amplitude=2000)

    assert tracker.observe(silence) is False  # no speech yet
    assert tracker.observe(speech) is False  # speech resets timer
    assert tracker.observe(silence) is False
    assert tracker.observe(silence) is True


def test_silence_tracker_triggers_on_single_long_chunk():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )
    speech = _chunk_with_duration(0.02, amplitude=2000)
    long_silence = _chunk_with_duration(0.02, amplitude=0)

    tracker.observe(speech)
    assert tracker.observe(long_silence) is True


def test_silence_tracker_clear_silence_keeps_speech_flag():
    tracker = controller.SilenceTracker(
        silence_threshold=1000,
        max_silence_seconds=0.01,
        sample_rate=controller.SAMPLE_RATE,
    )
    speech = _chunk_with_duration(0.02, amplitude=2000)
    silence = _chunk_with_duration(0.005, amplitude=0)

    tracker.observe(speech)
    tracker.observe(silence)
    tracker.clear_silence()

    assert tracker.observe(silence) is False
    assert tracker.observe(silence) is True


def test_stream_state_manager_tracks_deferred_finalize():
    manager = controller.StreamStateManager()

    assert manager.state == controller.StreamState.LISTENING
    assert manager.transition_to_listening("noop") is None

    previous = manager.transition_to_streaming()
    assert previous == controller.StreamState.LISTENING
    assert manager.state == controller.StreamState.STREAMING

    manager.increment_retrigger_budget()
    assert manager.retrigger_budget == 1
    manager.reset_retrigger_budget()
    assert manager.retrigger_budget == 0

    previous = manager.transition_to_listening("silence", defer_finalize=True)
    assert previous == controller.StreamState.STREAMING
    assert manager.awaiting_server_stop is True
    assert manager.pending_finalize_reason == "silence"

    reason = manager.complete_deferred_finalize("fallback")
    assert reason == "silence"
    assert manager.awaiting_server_stop is False
    assert manager.pending_finalize_reason is None

    manager.suppress_next_server_stop_event()
    assert manager.consume_suppressed_stop_event() is True
    assert manager.consume_suppressed_stop_event() is False


@pytest.mark.asyncio
async def test_response_task_manager_schedules_and_cleanup():
    executed = []

    call_id = {"value": 0}

    async def quick_task(task_id: int):
        executed.append(task_id)

    def factory():
        call_id["value"] += 1
        task = asyncio.create_task(quick_task(call_id["value"]))
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("first")
    manager.schedule("second")

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert executed == [1, 2]


@pytest.mark.asyncio
async def test_response_task_manager_cancel_marks_tasks():
    event = asyncio.Event()
    tasks = []

    async def waiting():
        await event.wait()

    def factory():
        task = asyncio.create_task(waiting())
        tasks.append(task)
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("hold")
    manager.schedule("hold-again")
    manager.cancel("manual stop")

    await asyncio.sleep(0)

    assert all(task.cancelled() for task in tasks)


@pytest.mark.asyncio
async def test_response_task_manager_drain_cancels_pending_tasks():
    event = asyncio.Event()
    tasks = []

    def factory():
        task = asyncio.create_task(event.wait())
        tasks.append(task)
        return task

    manager = controller.ResponseTaskManager(task_factory=factory)
    manager.schedule("pending")
    await manager.drain()

    assert all(task.cancelled() for task in tasks)
