from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import (
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    List,
    Type,
    cast,
)

import pytest

from pi_assistant.cli.controller_actors import (
    AssistantResponderActor,
    SilenceActor,
    SpeechGateActor,
    StreamUploaderActor,
    WakeWordActor,
)
from pi_assistant.cli.controller_components import StreamStateManager
from pi_assistant.cli.controller_events import (
    AudioChunkEvent,
    FinalizeTurnEvent,
    ManualStopEvent,
    PrerollReadyEvent,
    SilenceDetectedEvent,
    StateTransitionEvent,
    StateTransitionNotification,
    StreamChunkEvent,
    WakeWordTriggeredEvent,
)
from pi_assistant.wake_word import StreamState, WakeWordDetection

EXPECTED_WAKE_PUBLICATIONS = 2
PREROLL_TRIGGER_CHUNK_INDEX = 42
SILENCE_EVENT_CHUNK_INDEX = 10

EventHandler = Callable[[Any], Awaitable[None]]


class DummyBus:
    def __init__(self) -> None:
        self.handlers: DefaultDict[Type[Any], List[EventHandler]] = defaultdict(list)
        self.published: list[tuple[type[Any], Any, bool]] = []

    def subscribe(self, event_type: Type[Any], handler: EventHandler) -> None:
        self.handlers[event_type].append(handler)

    async def publish(self, event: Any, priority: bool = False) -> None:
        self.published.append((type(event), event, priority))

    async def emit(self, event: Any) -> None:
        for handler in self.handlers.get(type(event), []):
            await handler(event)


class TranscriptBufferStub:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    async def clear_current_turn(self, reason: str) -> None:
        self.cleared.append(reason)


class ResponseTasksStub:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel(self, reason: str) -> None:
        self.cancelled.append(reason)


class SilenceTrackerStub:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.observed: list[bytes] = []
        self.clears = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def observe(self, payload: bytes) -> None:
        self.observed.append(payload)

    def clear_silence(self) -> None:
        self.clears += 1


class WakeEngineStub:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_detection(self) -> None:
        self.reset_calls += 1


@pytest.mark.asyncio
async def test_wake_word_actor_publishes_trigger_and_stream_events():
    bus = DummyBus()
    silence_tracker = SilenceTrackerStub()
    pre_roll = type("PreRoll", (), {"flush": lambda self: b"", "clear": lambda self: None})()
    loop: Any = type(
        "LoopStub",
        (),
        {},
    )()
    loop.state_manager = StreamStateManager()
    loop.silence_tracker = silence_tracker
    loop.pre_roll = pre_roll
    loop.wake_engine = None
    buffered: list[bytes] = []
    detections = [
        WakeWordDetection(score=0.9, triggered=True),
        WakeWordDetection(score=0.1, triggered=False),
    ]

    def _buffer(chunk: bytes) -> None:
        buffered.append(chunk)

    def _run_detection(_: bytes) -> WakeWordDetection:
        return detections.pop(0)

    loop._buffer_preroll_if_listening = _buffer
    loop._run_wake_word_detection = _run_detection

    WakeWordActor(cast(Any, loop), cast(Any, bus))

    await bus.emit(AudioChunkEvent(chunk=b"listen", index=1))
    loop.state_manager.transition_to_streaming()
    await bus.emit(AudioChunkEvent(chunk=b"stream", index=2))

    assert buffered == [b"listen"]
    assert len(bus.published) == EXPECTED_WAKE_PUBLICATIONS
    event_type, event, priority = bus.published[0]
    assert event_type is WakeWordTriggeredEvent
    assert event.chunk == b"listen"
    assert priority is True
    event_type, event, priority = bus.published[1]
    assert event_type is StreamChunkEvent
    assert event.payload == b"stream"
    assert priority is False


@pytest.mark.asyncio
async def test_wake_word_actor_flushes_preroll_on_stream_transition():
    bus = DummyBus()
    silence_tracker = SilenceTrackerStub()
    wake_engine = WakeEngineStub()

    class PreRollStub:
        def __init__(self) -> None:
            self.flush_calls = 0

        def flush(self) -> bytes:
            self.flush_calls += 1
            return b"preroll-bytes"

    loop: Any = type("LoopStub", (), {})()
    loop.silence_tracker = silence_tracker
    loop.pre_roll = PreRollStub()
    loop.wake_engine = wake_engine

    actor = WakeWordActor(cast(Any, loop), cast(Any, bus))
    actor._last_listening_chunk = (PREROLL_TRIGGER_CHUNK_INDEX, b"last-chunk")

    event = StateTransitionEvent(
        previous=StreamState.LISTENING,
        current=StreamState.STREAMING,
        reason="wake phrase detected",
        trigger="wake_phrase",
        trigger_chunk_index=PREROLL_TRIGGER_CHUNK_INDEX,
    )

    await bus.emit(event)

    assert loop.pre_roll.flush_calls == 1
    assert silence_tracker.reset_calls == 1
    assert silence_tracker.observed == [b"last-chunk"]
    assert wake_engine.reset_calls == 1
    assert bus.published and bus.published[0][0] is PrerollReadyEvent
    preroll_event = bus.published[0][1]
    assert preroll_event.payload == b"preroll-bytes"
    assert preroll_event.chunk_index == PREROLL_TRIGGER_CHUNK_INDEX
    assert actor._last_listening_chunk is None


@pytest.mark.asyncio
async def test_silence_actor_emits_events_and_resets_on_listening_transition():
    bus = DummyBus()

    class LoopStub:
        def __init__(self) -> None:
            self._auto_stop_enabled = True
            self.state_manager = StreamStateManager()
            self.silence_tracker = SilenceTrackerStub()
            self._silence_results = [(True, True)]

        def _observe_streaming_silence(self, _: bytes) -> tuple[bool, bool]:
            return self._silence_results.pop(0)

    loop: Any = LoopStub()
    SilenceActor(cast(Any, loop), cast(Any, bus))

    await bus.emit(AudioChunkEvent(chunk=b"", index=SILENCE_EVENT_CHUNK_INDEX))
    assert bus.published
    event_type, event, priority = bus.published[0]
    assert event_type is SilenceDetectedEvent
    assert event.chunk_index == SILENCE_EVENT_CHUNK_INDEX
    assert priority is True

    state_event = StateTransitionEvent(
        previous=StreamState.STREAMING,
        current=StreamState.LISTENING,
        reason="stop",
        trigger="manual_stop",
    )
    await bus.emit(state_event)
    assert loop.silence_tracker.reset_calls == 1


@pytest.mark.asyncio
async def test_stream_uploader_actor_forwards_payloads():
    bus = DummyBus()

    class LoopStub:
        def __init__(self) -> None:
            self.preroll_payloads: list[bytes] = []
            self.stream_payloads: list[bytes] = []

        async def _send_preroll_payload(self, payload: bytes) -> None:
            self.preroll_payloads.append(payload)

        async def _forward_audio(self, payload: bytes) -> None:
            self.stream_payloads.append(payload)

    loop: Any = LoopStub()
    StreamUploaderActor(cast(Any, loop), cast(Any, bus))

    await bus.emit(PrerollReadyEvent(payload=b"pre"))
    await bus.emit(StreamChunkEvent(payload=b"stream", chunk_index=1))

    assert loop.preroll_payloads == [b"pre"]
    assert loop.stream_payloads == [b"stream"]


@pytest.mark.asyncio
async def test_assistant_responder_actor_schedules_finalize():
    bus = DummyBus()

    class LoopStub:
        def __init__(self) -> None:
            self.finalized: list[str] = []

        def _finalize_turn(self, reason: str) -> None:
            self.finalized.append(reason)

    loop: Any = LoopStub()
    AssistantResponderActor(cast(Any, loop), cast(Any, bus))

    await bus.emit(FinalizeTurnEvent(reason="server stop"))

    assert loop.finalized == ["server stop"]


@pytest.mark.asyncio
async def test_speech_gate_actor_manual_stop_transitions_and_cleans_up():
    bus = DummyBus()

    class LoopStub:
        def __init__(self) -> None:
            self.state_manager = StreamStateManager()
            self.state_manager.transition_to_streaming()
            self.context: Any = SimpleNamespace(transcript_buffer=TranscriptBufferStub())
            self.response_tasks = ResponseTasksStub()
            self.reset_calls = 0
            self.clear_calls: list[str] = []
            self.logged: list[tuple[StreamState | None, StreamState, str]] = []
            self.notified: list[tuple[Any, ...]] = []

        def _verbose_print(self, *_: Any, **__: Any) -> None:
            return

        def _log_state_transition(
            self,
            previous: StreamState | None,
            current: StreamState,
            reason: str,
        ) -> None:
            self.logged.append((previous, current, reason))

        async def _notify_state_transition(
            self,
            previous: StreamState | None,
            current: StreamState,
            *,
            notification: StateTransitionNotification,
        ) -> None:
            self.notified.append(
                (
                    previous,
                    current,
                    notification.reason,
                    notification.trigger,
                    notification.trigger_chunk_index,
                    notification.priority,
                )
            )

        def _reset_stream_resources(self) -> None:
            self.reset_calls += 1

        def _clear_server_stop_timeout(self, *, cause: str) -> None:
            self.clear_calls.append(cause)

    loop: Any = LoopStub()
    SpeechGateActor(cast(Any, loop), cast(Any, bus))

    event = ManualStopEvent(reason="manual halt")
    await bus.emit(event)

    assert loop.state_manager.state == StreamState.LISTENING
    assert loop.reset_calls == 1
    assert loop.clear_calls == ["manual_stop"]
    assert loop.context.transcript_buffer.cleared == ["manual halt"]
    assert loop.response_tasks.cancelled == ["manual halt"]
    assert loop.logged and loop.logged[-1][2] == "manual halt"
    assert loop.notified
    notified = loop.notified[-1]
    assert notified[0] == StreamState.STREAMING
    assert notified[1] == StreamState.LISTENING
    assert notified[-1] is True
