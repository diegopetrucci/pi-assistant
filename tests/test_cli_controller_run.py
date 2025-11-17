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
            if self.calls < len(detections):
                result = detections[self.calls]
            else:
                result = WakeWordDetection(score=0.0, triggered=False)
            self.calls += 1
            return result

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


@pytest.mark.asyncio
async def test_run_audio_controller_handles_server_stop_signal(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(64, dtype=np.int16) * 20000).tobytes()
    for _ in range(5):
        capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    detections = [
        WakeWordDetection(score=0.95, triggered=True),
        WakeWordDetection(score=0.1, triggered=False),
    ]

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def process_chunk(self, chunk):
            if self.calls < len(detections):
                result = detections[self.calls]
            else:
                result = WakeWordDetection(score=0.0, triggered=False)
            self.calls += 1
            return result

        def reset_detection(self):
            return None

    class DummyPreRollBuffer:
        def __init__(self, *args, **kwargs):
            self.buffer = bytearray()

        def add(self, chunk):
            self.buffer.extend(chunk)

        def flush(self):
            payload = bytes(self.buffer)
            self.buffer.clear()
            return payload

        def clear(self):
            self.buffer.clear()

    class DummyResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = False

        def process(self, audio_bytes):
            return np.frombuffer(audio_bytes, dtype=np.int16)

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(controller, "PreRollBuffer", DummyPreRollBuffer)
    monkeypatch.setattr(controller, "LinearResampler", DummyResampler)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    transitions = []

    def fake_log_transition(previous, current, reason):
        transitions.append((previous, current, reason))

    monkeypatch.setattr(controller, "log_state_transition", fake_log_transition)

    def fake_schedule(*args, **kwargs):
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

    await asyncio.sleep(0.05)
    speech_stopped_signal.set()
    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.1)
    await _shutdown_task(task)

    assert any(reason == "server speech stop event" for _, _, reason in transitions)
    assert ws_client.sent_chunks  # streaming occurred before stop


@pytest.mark.asyncio
async def test_run_audio_controller_cancellation_cleans_response_tasks(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(64, dtype=np.int16) * 20000).tobytes()
    for _ in range(5):
        capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    detections = [
        WakeWordDetection(score=0.95, triggered=True),
        WakeWordDetection(score=0.1, triggered=False),
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
            payload = bytes(self.buffer)
            self.buffer.clear()
            return payload

        def clear(self):
            self.buffer.clear()

    class DummyResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = False

        def process(self, audio_bytes):
            return np.frombuffer(audio_bytes, dtype=np.int16)

        def reset(self):
            self.reset_called = True

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(controller, "PreRollBuffer", DummyPreRollBuffer)
    monkeypatch.setattr(controller, "LinearResampler", DummyResampler)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    pending_event = asyncio.Event()
    scheduled_tasks = []

    async def _sleep_forever():
        await pending_event.wait()

    def fake_schedule(*args, **kwargs):
        task = asyncio.create_task(_sleep_forever())
        scheduled_tasks.append(task)
        return task

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

    await asyncio.sleep(0.05)
    speech_stopped_signal.set()
    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.05)

    await _shutdown_task(task)

    assert scheduled_tasks, "expected response task to be scheduled"
    assert all(t.cancelled() for t in scheduled_tasks)


@pytest.mark.asyncio
async def test_run_audio_controller_manual_stop_clears_buffers(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(64, dtype=np.int16) * 25000).tobytes()
    # Provide multiple chunks so the loop keeps running
    for _ in range(10):
        capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    detections = [
        WakeWordDetection(score=0.2, triggered=False),
        WakeWordDetection(score=0.95, triggered=True),
        WakeWordDetection(score=0.1, triggered=False),
    ]

    class TrackingWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.reset_count = 0

        def process_chunk(self, chunk):
            idx = min(self.calls, len(detections) - 1)
            self.calls += 1
            return detections[idx]

        def reset_detection(self):
            self.reset_count += 1

    class TrackingPreRollBuffer:
        def __init__(self, *args, **kwargs):
            self.buffer = bytearray()
            self.clear_count = 0
            self.flushed = False

        def add(self, chunk):
            self.buffer.extend(chunk)

        def flush(self):
            data = bytes(self.buffer)
            self.buffer.clear()
            self.flushed = True
            return data

        def clear(self):
            self.clear_count += 1
            self.buffer.clear()

    class TrackingResampler:
        def __init__(self, *args, **kwargs):
            self.reset_called = 0

        def process(self, audio_bytes):
            return np.frombuffer(audio_bytes, dtype=np.int16)

        def reset(self):
            self.reset_called += 1

    wake_engine = TrackingWakeWordEngine(None)

    def wake_factory(*args, **kwargs):
        return wake_engine

    preroll_buffer = TrackingPreRollBuffer(None, None, None)
    resampler = TrackingResampler(None, None)

    monkeypatch.setattr(controller, "WakeWordEngine", wake_factory)
    monkeypatch.setattr(controller, "PreRollBuffer", lambda *args, **kwargs: preroll_buffer)
    monkeypatch.setattr(controller, "LinearResampler", lambda *args, **kwargs: resampler)
    monkeypatch.setattr(controller, "STREAM_SAMPLE_RATE", controller.SAMPLE_RATE // 2 or 1)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    def fake_schedule(*args, **kwargs):
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

    await asyncio.sleep(0.05)  # allow wake word to trigger streaming
    stop_signal.set()
    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.05)
    await _shutdown_task(task)

    assert preroll_buffer.flushed is True
    assert preroll_buffer.clear_count >= 1
    assert resampler.reset_called >= 1
    assert wake_engine.reset_count >= 1
    assert "manual stop command" in transcript_buffer.cleared


@pytest.mark.asyncio
async def test_run_audio_controller_auto_stop_silence_transition(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 30000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()
    capture.queue.put_nowait(loud_chunk)
    for _ in range(50):
        capture.queue.put_nowait(silence_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def process_chunk(self, chunk):
            self.calls += 1
            if self.calls == 1:
                return WakeWordDetection(score=0.95, triggered=True)
            return WakeWordDetection(score=0.0, triggered=False)

        def reset_detection(self):
            return None

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(
        controller,
        "PreRollBuffer",
        lambda *args, **kwargs: type(
            "P",
            (),
            {
                "buffer": bytearray(),
                "add": lambda self, chunk: self.buffer.extend(chunk),
                "flush": lambda self: bytes(self.buffer),
                "clear": lambda self: self.buffer.clear(),
            },
        )(),
    )
    monkeypatch.setattr(
        controller,
        "LinearResampler",
        lambda *args, **kwargs: type(
            "R",
            (),
            {
                "process": lambda self, audio_bytes: np.frombuffer(audio_bytes, dtype=np.int16),
                "reset": lambda self: None,
            },
        )(),
    )
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", True)
    monkeypatch.setattr(controller, "AUTO_STOP_SILENCE_THRESHOLD", 1000)
    monkeypatch.setattr(controller, "AUTO_STOP_MAX_SILENCE_SECONDS", 0.01)

    transitions = []

    def fake_log_transition(previous, current, reason):
        transitions.append((previous, current, reason))

    monkeypatch.setattr(controller, "log_state_transition", fake_log_transition)

    scheduled = []

    def fake_schedule(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(0))
        scheduled.append(task)
        return task

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

    await asyncio.sleep(0.3)
    await _shutdown_task(task)

    assert any(reason == "silence detected" for _, _, reason in transitions)
    assert scheduled  # response scheduled on auto-stop


@pytest.mark.asyncio
async def test_run_audio_controller_retrigger_delays_auto_stop(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 32000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()

    capture.queue.put_nowait(loud_chunk)  # prime buffer
    capture.queue.put_nowait(loud_chunk)  # triggers streaming
    capture.queue.put_nowait(loud_chunk)  # retrigger during streaming
    for _ in range(3):
        capture.queue.put_nowait(silence_chunk)  # first silence window (single threshold)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    detections = [
        WakeWordDetection(score=0.2, triggered=False),
        WakeWordDetection(score=0.95, triggered=True),
        WakeWordDetection(score=0.95, triggered=True),
    ]

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def process_chunk(self, chunk):
            if self.calls < len(detections):
                result = detections[self.calls]
            else:
                result = WakeWordDetection(score=0.0, triggered=False)
            self.calls += 1
            return result

        def reset_detection(self):
            return None

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)

    def make_preroll(*args, **kwargs):
        class P:
            def __init__(self):
                self.buffer = bytearray()

            def add(self, chunk):
                self.buffer.extend(chunk)

            def flush(self):
                data = bytes(self.buffer)
                self.buffer.clear()
                return data

            def clear(self):
                self.buffer.clear()

        return P()

    monkeypatch.setattr(controller, "PreRollBuffer", make_preroll)
    monkeypatch.setattr(
        controller,
        "LinearResampler",
        lambda *args, **kwargs: type(
            "R",
            (),
            {
                "process": lambda self, audio_bytes: np.frombuffer(audio_bytes, dtype=np.int16),
                "reset": lambda self: None,
            },
        )(),
    )
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", True)
    monkeypatch.setattr(controller, "AUTO_STOP_SILENCE_THRESHOLD", 1000)
    monkeypatch.setattr(controller, "AUTO_STOP_MAX_SILENCE_SECONDS", 0.01)

    transitions = []

    def fake_log_transition(previous, current, reason):
        transitions.append((previous, current, reason))

    monkeypatch.setattr(controller, "log_state_transition", fake_log_transition)

    scheduled = []

    def fake_schedule(*args, **kwargs):
        task = asyncio.create_task(asyncio.sleep(0))
        scheduled.append(task)
        return task

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
    reasons = [reason for _, _, reason in transitions]
    assert "silence detected" not in reasons  # retrigger prevented transition

    for _ in range(20):
        capture.queue.put_nowait(silence_chunk)

    await asyncio.sleep(0.2)
    await _shutdown_task(task)

    reasons = [reason for _, _, reason in transitions]
    assert "silence detected" in reasons
    assert scheduled


@pytest.mark.asyncio
async def test_run_audio_controller_stop_signal_during_preroll_flush(monkeypatch):
    capture = FakeCapture()
    first_chunk = (np.ones(128, dtype=np.int16) * 15000).tobytes()
    next_chunk = (np.ones(128, dtype=np.int16) * 8000).tobytes()
    capture.queue.put_nowait(first_chunk)
    capture.queue.put_nowait(next_chunk)
    capture.queue.put_nowait(next_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class ImmediateWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.reset_called = 0

        def process_chunk(self, chunk):
            self.calls += 1
            if self.calls == 1:
                return WakeWordDetection(score=0.9, triggered=True)
            return WakeWordDetection(score=0.0, triggered=False)

        def reset_detection(self):
            self.reset_called += 1

    wake_engine = ImmediateWakeWordEngine(None)
    monkeypatch.setattr(controller, "WakeWordEngine", lambda *args, **kwargs: wake_engine)

    preroll_state = {"flushed": False, "cleared": 0}

    def preroll_factory(*args, **kwargs):
        class TrackingPreRoll:
            def __init__(self):
                self.buffer = bytearray()

            def add(self, chunk):
                self.buffer.extend(chunk)

            def flush(self):
                preroll_state["flushed"] = True
                stop_signal.set()
                data = bytes(self.buffer)
                self.buffer.clear()
                return data

            def clear(self):
                preroll_state["cleared"] += 1
                self.buffer.clear()

        return TrackingPreRoll()

    monkeypatch.setattr(controller, "PreRollBuffer", preroll_factory)
    monkeypatch.setattr(
        controller,
        "LinearResampler",
        lambda *args, **kwargs: type(
            "R",
            (),
            {
                "process": lambda self, audio_bytes: np.frombuffer(audio_bytes, dtype=np.int16),
                "reset": lambda self: None,
            },
        )(),
    )
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    def fake_schedule(*args, **kwargs):
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

    assert preroll_state["flushed"] is True
    assert preroll_state["cleared"] >= 1
    assert wake_engine.reset_called >= 1
    assert "manual stop command" in transcript_buffer.cleared
    assert ws_client.sent_chunks  # pre-roll payload sent before stop
