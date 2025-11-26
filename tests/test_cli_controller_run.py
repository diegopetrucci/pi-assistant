import asyncio
from typing import Any, Callable, cast

import numpy as np
import pytest

from pi_assistant.cli import controller
from pi_assistant.wake_word import WakeWordDetection


@pytest.fixture(autouse=True)
def _disable_server_stop_guard(monkeypatch):
    """Disable the server-stop guard so tests can finalize turns without waiting."""

    monkeypatch.setattr(controller, "SERVER_STOP_MIN_SILENCE_SECONDS", 0.0)


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

    async def finalize_turn(self):
        return None


def _chunk_with_duration(seconds: float, amplitude: int = 0) -> bytes:
    frames = max(1, int(seconds * controller.SAMPLE_RATE))
    samples = np.full(frames * controller.CHANNELS, amplitude, dtype=np.int16)
    return samples.tobytes()


def _make_response_scheduler(
    storage: list[asyncio.Task[None]],
    *,
    auto_complete: bool = False,
) -> Callable[..., asyncio.Task[None]]:
    """Return a schedule_turn_response stub that toggles lifecycle callbacks."""

    sleep_duration = 0 if auto_complete else 10

    def _schedule(tb, assistant, player, **kwargs):
        hooks = kwargs.get("hooks")

        async def _task():
            if hooks and hooks.on_transcript_ready:
                hooks.on_transcript_ready()
            if hooks and hooks.on_reply_start:
                hooks.on_reply_start()
            try:
                await asyncio.sleep(sleep_duration)
            finally:
                if hooks and hooks.on_reply_complete:
                    hooks.on_reply_complete()

        task = asyncio.create_task(_task())
        storage.append(task)
        return task

    return _schedule


async def _shutdown_task(task: asyncio.Task[Any]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _wait_for_condition(
    predicate: Callable[[], bool],
    *,
    timeout: float,
    poll_interval: float = 0.01,
    timeout_message: str,
) -> None:
    """Poll ``predicate`` until it returns True or fail after ``timeout`` seconds."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            pytest.fail(timeout_message)
        await asyncio.sleep(poll_interval)


async def _ensure_condition_stays_false(
    predicate: Callable[[], bool],
    *,
    duration: float,
    poll_interval: float = 0.01,
    failure_message: str,
) -> None:
    """Assert ``predicate`` remains False for ``duration`` seconds."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration
    while loop.time() < deadline:
        if predicate():
            pytest.fail(failure_message)
        await asyncio.sleep(poll_interval)


@pytest.mark.asyncio
async def test_run_audio_controller_uses_supplied_context(monkeypatch):
    capture = object()
    ws_client = cast(controller.WebSocketClient, object())
    transcript_buffer = cast(controller.TurnTranscriptAggregator, object())
    assistant = cast(controller.LLMResponder, object())
    speech_player = cast(controller.SpeechPlayer, object())
    context = controller.AudioControllerContext(
        transcript_buffer=transcript_buffer,
        assistant=assistant,
        speech_player=speech_player,
        stop_signal=asyncio.Event(),
        speech_stopped_signal=asyncio.Event(),
    )
    recorded = {}

    class DummyRunner:
        def __init__(self, audio_capture, client, passed_context):
            recorded["args"] = (audio_capture, client, passed_context)

        async def run(self):
            recorded["ran"] = True

    monkeypatch.setattr(controller, "_AudioControllerLoop", DummyRunner)

    await controller.run_audio_controller(capture, ws_client, context=context)

    assert recorded["args"] == (capture, ws_client, context)
    assert recorded.get("ran")


@pytest.mark.asyncio
async def test_run_audio_controller_rejects_context_and_kwargs():
    transcript_buffer = cast(controller.TurnTranscriptAggregator, object())
    assistant = cast(controller.LLMResponder, object())
    speech_player = cast(controller.SpeechPlayer, object())
    context = controller.AudioControllerContext(
        transcript_buffer=transcript_buffer,
        assistant=assistant,
        speech_player=speech_player,
        stop_signal=asyncio.Event(),
        speech_stopped_signal=asyncio.Event(),
    )

    with pytest.raises(TypeError, match="either `context` or individual"):
        await controller.run_audio_controller(
            cast(controller.WebSocketClient, object()),
            cast(controller.WebSocketClient, object()),
            context=context,
            transcript_buffer=transcript_buffer,
            assistant=assistant,
            speech_player=speech_player,
            stop_signal=asyncio.Event(),
            speech_stopped_signal=asyncio.Event(),
        )


@pytest.mark.asyncio
async def test_run_audio_controller_reports_invalid_config_keys():
    transcript_buffer = cast(controller.TurnTranscriptAggregator, object())
    assistant = cast(controller.LLMResponder, object())
    speech_player = cast(controller.SpeechPlayer, object())
    with pytest.raises(TypeError, match="Invalid audio controller configuration:"):
        await controller.run_audio_controller(
            object(),
            cast(controller.WebSocketClient, object()),
            transcript_buffer=transcript_buffer,
            assistant=assistant,
            speech_player=speech_player,
            stop_signal=asyncio.Event(),
            speech_stopped_signal=asyncio.Event(),
            extra_toggle=True,
        )


@pytest.mark.asyncio
async def test_run_audio_controller_wake_engine_failure(monkeypatch, capsys):
    capture = FakeCapture()
    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    def failing_wake_engine(*args, **kwargs):
        raise RuntimeError("missing model")

    monkeypatch.setattr(controller, "WakeWordEngine", failing_wake_engine)

    with pytest.raises(RuntimeError, match="missing model"):
        await controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )

    captured = capsys.readouterr()
    assert "missing model" in captured.err


@pytest.mark.asyncio
async def test_run_audio_controller_skips_misaligned_chunk(monkeypatch, capsys):
    capture = FakeCapture()
    capture.queue.put_nowait(b"\x00")
    capture.queue.put_nowait(np.zeros(32, dtype=np.int16).tobytes())
    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.processed_a_chunk = False

        def process_chunk(self, chunk):
            self.processed_a_chunk = True
            return WakeWordDetection(score=0.0, triggered=False)

        def reset_detection(self):
            return None

    dummy_wake_engine_instance = DummyWakeWordEngine()
    monkeypatch.setattr(
        controller,
        "WakeWordEngine",
        lambda *args, **kwargs: dummy_wake_engine_instance,
    )

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    await asyncio.sleep(0.05)
    await _shutdown_task(task)

    captured = capsys.readouterr()
    assert "Dropping malformed audio chunk" in captured.err
    assert dummy_wake_engine_instance.processed_a_chunk


@pytest.mark.asyncio
async def test_run_audio_controller_handles_manual_stop(monkeypatch):
    capture = FakeCapture()
    capture.queue.put_nowait(np.zeros(32, dtype=np.int16).tobytes())
    capture.queue.put_nowait(np.ones(32, dtype=np.int16).tobytes())
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
            triggered = self.calls == 1
            score = 0.9 if triggered else 0.1
            return WakeWordDetection(score=score, triggered=triggered)

        def reset_detection(self):
            return None

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)

    scheduled = []
    fake_schedule = _make_response_scheduler(scheduled, auto_complete=True)

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
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

    transitions = []

    def fake_log(previous, current, reason):
        transitions.append(reason)

    monkeypatch.setattr(controller, "log_state_transition", fake_log)

    scheduled = []
    fake_schedule = _make_response_scheduler(scheduled)

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if "silence detected" in transitions:
            break
    else:
        pytest.fail("auto-stop transition never triggered")
    assert not scheduled  # auto-stop should wait for server confirmation
    for _ in range(5):
        capture.queue.put_nowait(silence_chunk)
    capture.queue.put_nowait(silence_chunk)
    speech_stopped_signal.set()
    await asyncio.sleep(0.05)
    await _shutdown_task(task)

    assert transcript_buffer.started >= 1
    assert ws_client.sent_chunks  # pre-roll and live chunks sent
    assert scheduled  # auto-stop transition scheduled a response


@pytest.mark.asyncio
async def test_server_stop_event_respects_min_silence(monkeypatch):
    monkeypatch.setattr(controller, "SERVER_STOP_MIN_SILENCE_SECONDS", 0.3)

    capture = FakeCapture()
    loud_chunk = _chunk_with_duration(0.2, amplitude=20000)
    silence_chunk = _chunk_with_duration(0.05, amplitude=0)

    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class DummyWakeWordEngine:
        def __init__(self, *args, **kwargs):
            self.called = False

        def process_chunk(self, chunk):
            if not self.called:
                self.called = True
                return WakeWordDetection(score=0.95, triggered=True)
            return WakeWordDetection(score=0.1, triggered=False)

        def reset_detection(self):
            return None

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    scheduled = []
    monkeypatch.setattr(
        controller,
        "schedule_turn_response",
        _make_response_scheduler(scheduled, auto_complete=True),
    )

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    await _wait_for_condition(
        lambda: transcript_buffer.started >= 1,
        timeout=2.0,
        timeout_message="wake phrase never started a turn",
    )

    speech_stopped_signal.set()
    capture.queue.put_nowait(silence_chunk)
    await _ensure_condition_stays_false(
        lambda: bool(scheduled),
        duration=0.2,
        failure_message="server stop finalized before silence elapsed",
    )

    for _ in range(10):
        capture.queue.put_nowait(silence_chunk)
    await _wait_for_condition(
        lambda: capture.queue.empty(),
        timeout=2.0,
        timeout_message="controller did not drain silence queue",
    )
    speech_stopped_signal.set()
    capture.queue.put_nowait(silence_chunk)

    await _wait_for_condition(
        lambda: bool(scheduled),
        timeout=2.0,
        timeout_message="server stop never finalized after silence",
    )

    await _shutdown_task(task)


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

    monkeypatch.setattr(
        controller,
        "schedule_turn_response",
        _make_response_scheduler([], auto_complete=True),
    )

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
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

    def fake_schedule(*args, **kwargs):
        hooks = kwargs.get("hooks")

        async def _sleep_forever():
            if hooks and hooks.on_transcript_ready:
                hooks.on_transcript_ready()
            if hooks and hooks.on_reply_start:
                hooks.on_reply_start()
            try:
                await pending_event.wait()
            finally:
                if hooks and hooks.on_reply_complete:
                    hooks.on_reply_complete()

        task = asyncio.create_task(_sleep_forever())
        scheduled_tasks.append(task)
        return task

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
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
    sample_rate = controller.SAMPLE_RATE
    monkeypatch.setattr(controller, "STREAM_SAMPLE_RATE", sample_rate // 2 or 1)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    monkeypatch.setattr(
        controller,
        "schedule_turn_response",
        _make_response_scheduler([], auto_complete=True),
    )

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
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
    fake_schedule = _make_response_scheduler(scheduled)
    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(reason == "silence detected" for _, _, reason in transitions):
            break
    else:
        pytest.fail("auto-stop transition never triggered")

    assert not scheduled  # waiting for server stop
    for _ in range(5):
        capture.queue.put_nowait(silence_chunk)
    speech_stopped_signal.set()
    await asyncio.sleep(0.1)
    await _shutdown_task(task)

    assert any(reason == "silence detected" for _, _, reason in transitions)
    assert scheduled  # response scheduled on auto-stop


@pytest.mark.asyncio
async def test_run_audio_controller_resets_wake_engine_on_stream_start(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 20000).tobytes()
    for _ in range(10):
        capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class DummyWakeWordEngine:
        TRIGGER_CALL_COUNT = 2

        def __init__(self):
            self.calls = 0
            self.reset_calls = 0

        def process_chunk(self, chunk):
            self.calls += 1
            if self.calls == self.TRIGGER_CALL_COUNT:
                return WakeWordDetection(score=0.95, triggered=True)
            return WakeWordDetection(score=0.1, triggered=False)

        def reset_detection(self):
            self.reset_calls += 1

    wake_engine = DummyWakeWordEngine()
    monkeypatch.setattr(controller, "WakeWordEngine", lambda *args, **kwargs: wake_engine)

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
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if transcript_buffer.started:
            break
    else:
        pytest.fail("controller never entered streaming state")

    assert wake_engine.reset_calls == 1

    await _shutdown_task(task)


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
    fake_schedule = _make_response_scheduler(scheduled)
    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    await asyncio.sleep(0.2)
    reasons = [reason for _, _, reason in transitions]
    assert "silence detected" not in reasons  # retrigger prevented transition

    for _ in range(20):
        capture.queue.put_nowait(silence_chunk)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if any(reason == "silence detected" for _, _, reason in transitions):
            break
    else:
        pytest.fail("auto-stop transition never triggered")

    assert not scheduled
    for _ in range(5):
        capture.queue.put_nowait(silence_chunk)
    speech_stopped_signal.set()
    await asyncio.sleep(0.05)
    await _shutdown_task(task)

    reasons = [reason for _, _, reason in transitions]
    assert "silence detected" in reasons
    assert scheduled


@pytest.mark.asyncio
async def test_wake_word_ignored_while_finalizing_previous_turn(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 20000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()
    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(silence_chunk)
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
            triggered = self.calls in {0, 3}
            score = 0.95 if triggered else 0.1
            self.calls += 1
            return WakeWordDetection(score=score, triggered=triggered)

        def reset_detection(self):
            return None

    monkeypatch.setattr(controller, "WakeWordEngine", DummyWakeWordEngine)
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    release_event = asyncio.Event()
    scheduled = []
    observed_hooks: list[controller.ResponseLifecycleHooks] = []

    def fake_schedule(tb, assistant, player, **kw):
        hooks = kw.get("hooks")
        observed_hooks.clear()
        if hooks:
            observed_hooks.append(hooks)

        async def _task():
            await release_event.wait()

        task = asyncio.create_task(_task())
        scheduled.append(task)
        return task

    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if transcript_buffer.started >= 1:
            break
    else:
        pytest.fail("initial wake phrase did not start streaming")

    speech_stopped_signal.set()
    capture.queue.put_nowait(silence_chunk)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if scheduled:
            break
    else:
        pytest.fail("response task was not scheduled")

    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.05)
    assert transcript_buffer.started == 1

    if observed_hooks and observed_hooks[0].on_transcript_ready:
        observed_hooks[0].on_transcript_ready()
    release_event.set()
    await _shutdown_task(task)
    for pending in scheduled:
        pending.cancel()


@pytest.mark.asyncio
async def test_auto_stop_wait_allows_new_wake(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 20000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()

    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(loud_chunk)
    for _ in range(50):
        capture.queue.put_nowait(silence_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class ControlledWakeEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.allow_retry = False

        def process_chunk(self, chunk):
            if self.calls == 0:
                result = WakeWordDetection(score=0.95, triggered=True)
            elif self.allow_retry:
                result = WakeWordDetection(score=0.95, triggered=True)
                self.allow_retry = False
            else:
                result = WakeWordDetection(score=0.1, triggered=False)
            self.calls += 1
            return result

        def reset_detection(self):
            return None

    wake_engine = ControlledWakeEngine(None)
    monkeypatch.setattr(controller, "WakeWordEngine", lambda *args, **kwargs: wake_engine)

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

    def fake_log(previous, current, reason):
        transitions.append(reason)

    monkeypatch.setattr(controller, "log_state_transition", fake_log)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(50):
        await asyncio.sleep(0.01)
        if "silence detected" in transitions:
            break
    else:
        pytest.fail("auto-stop transition never triggered")

    await asyncio.sleep(0.05)
    wake_engine.allow_retry = True
    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.05)

    assert transcript_buffer.started == 1  # retrigger ignored until server confirms stop

    speech_stopped_signal.set()
    await asyncio.sleep(0.05)
    await _shutdown_task(task)


@pytest.mark.asyncio
async def test_wake_phrase_override_cancels_pending_responses(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 20000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()

    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(loud_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class ControlledWakeEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.allow_retry = False

        def process_chunk(self, chunk):
            if self.calls == 0:
                result = WakeWordDetection(score=0.95, triggered=True)
            elif self.allow_retry:
                self.allow_retry = False
                result = WakeWordDetection(score=0.95, triggered=True)
            else:
                result = WakeWordDetection(score=0.1, triggered=False)
            self.calls += 1
            return result

        def reset_detection(self):
            return None

    wake_engine = ControlledWakeEngine(None)
    monkeypatch.setattr(controller, "WakeWordEngine", lambda *args, **kwargs: wake_engine)

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
    monkeypatch.setattr(controller, "AUTO_STOP_ENABLED", False)

    scheduled = []
    fake_schedule = _make_response_scheduler(scheduled)
    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if transcript_buffer.started >= 1:
            break
    else:
        pytest.fail("initial wake phrase did not start a turn")

    speech_stopped_signal.set()
    capture.queue.put_nowait(silence_chunk)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if scheduled:
            break
    else:
        pytest.fail("first response never scheduled")

    wake_engine.allow_retry = True
    capture.queue.put_nowait(loud_chunk)

    try:
        required_turns = 2

        async def wait_for_cancellation() -> None:
            while not (transcript_buffer.started >= required_turns and scheduled[0].cancelled()):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_cancellation(), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("pending response was not cancelled after wake override")

    capture.queue.put_nowait(silence_chunk)
    await _shutdown_task(task)
    for pending in scheduled:
        pending.cancel()


@pytest.mark.asyncio
async def test_wake_phrase_waits_for_server_ack_before_override(monkeypatch):
    capture = FakeCapture()
    loud_chunk = (np.ones(128, dtype=np.int16) * 20000).tobytes()
    silence_chunk = np.zeros(128, dtype=np.int16).tobytes()

    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(loud_chunk)
    capture.queue.put_nowait(silence_chunk)

    ws_client = FakeWebSocket()
    transcript_buffer = DummyTranscriptBuffer()
    assistant = object()
    speech_player = object()
    stop_signal = asyncio.Event()
    speech_stopped_signal = asyncio.Event()

    class ControlledWakeEngine:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.allow_retry = False

        def process_chunk(self, chunk):
            if self.calls == 0:
                result = WakeWordDetection(score=0.95, triggered=True)
            elif self.allow_retry:
                self.allow_retry = False
                result = WakeWordDetection(score=0.95, triggered=True)
            else:
                result = WakeWordDetection(score=0.1, triggered=False)
            self.calls += 1
            return result

        def reset_detection(self):
            return None

    wake_engine = ControlledWakeEngine(None)
    monkeypatch.setattr(controller, "WakeWordEngine", lambda *args, **kwargs: wake_engine)

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
    monkeypatch.setattr(controller, "AUTO_STOP_MAX_SILENCE_SECONDS", 0.0)

    scheduled = []
    fake_schedule = _make_response_scheduler(scheduled)
    monkeypatch.setattr(controller, "schedule_turn_response", fake_schedule)

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
            stop_signal=stop_signal,
            speech_stopped_signal=speech_stopped_signal,
        )
    )

    for _ in range(100):
        await asyncio.sleep(0.01)
        if transcript_buffer.started >= 1:
            break
    else:
        pytest.fail("controller never entered streaming state")

    wake_engine.allow_retry = True
    capture.queue.put_nowait(loud_chunk)
    await asyncio.sleep(0.05)
    assert not scheduled  # wake ignored until server confirms stop

    speech_stopped_signal.set()
    capture.queue.put_nowait(silence_chunk)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if scheduled:
            break
    else:
        pytest.fail("response not scheduled after server ack")

    assert len(scheduled) == 1

    await _shutdown_task(task)
    for pending in scheduled:
        pending.cancel()


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

    monkeypatch.setattr(
        controller,
        "schedule_turn_response",
        _make_response_scheduler([], auto_complete=True),
    )

    task = asyncio.create_task(
        controller.run_audio_controller(
            capture,
            ws_client,  # pyright: ignore[reportArgumentType]
            transcript_buffer=transcript_buffer,  # pyright: ignore[reportArgumentType]
            assistant=assistant,  # pyright: ignore[reportArgumentType]
            speech_player=speech_player,  # pyright: ignore[reportArgumentType]
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
