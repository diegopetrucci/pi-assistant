from __future__ import annotations

from collections import defaultdict
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
    SpeechGateActorConfig,
    SpeechGateActorDependencies,
    StreamUploaderActor,
    WakeWordActor,
    WakeWordActorDependencies,
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
        self._silence_duration = 0.0
        self._has_observed = True
        self.heard_speech = True

    def reset(self) -> None:
        self.reset_calls += 1

    def observe(self, payload: bytes) -> bool:
        self.observed.append(payload)
        return False

    def clear_silence(self) -> None:
        self.clears += 1

    @property
    def silence_duration(self) -> float:
        return self._silence_duration

    def has_observed_silence(self, _: float) -> bool:
        return self._has_observed


class PreRollStub:
    def __init__(self) -> None:
        self.added: list[bytes] = []
        self.flush_calls = 0
        self.next_flush_payload: bytes | None = None
        self._buffer: list[bytes] = []

    def add(self, chunk: bytes) -> None:
        self.added.append(chunk)
        self._buffer.append(chunk)

    def flush(self) -> bytes:
        self.flush_calls += 1
        if self.next_flush_payload is not None:
            payload = self.next_flush_payload
            self.next_flush_payload = None
            return payload
        payload = b"".join(self._buffer)
        self._buffer.clear()
        return payload

    def clear(self) -> None:
        self._buffer.clear()
        self.next_flush_payload = None


class WakeEngineStub:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_detection(self) -> None:
        self.reset_calls += 1


@pytest.mark.asyncio
async def test_wake_word_actor_publishes_trigger_and_stream_events():
    bus = DummyBus()
    silence_tracker = SilenceTrackerStub()
    pre_roll = PreRollStub()
    state_manager = StreamStateManager()
    detections = [
        WakeWordDetection(score=0.9, triggered=True),
        WakeWordDetection(score=0.1, triggered=False),
    ]

    def _run_detection(_: bytes) -> WakeWordDetection:
        return detections.pop(0)

    WakeWordActor(
        cast(Any, bus),
        WakeWordActorDependencies(
            state_manager=state_manager,
            silence_tracker=cast(Any, silence_tracker),
            pre_roll=cast(Any, pre_roll),
            run_wake_word_detection=_run_detection,
            verbose_print=lambda *_: None,
            wake_engine=None,
        ),
    )

    await bus.emit(AudioChunkEvent(chunk=b"listen", index=1))
    state_manager.transition_to_streaming()
    await bus.emit(AudioChunkEvent(chunk=b"stream", index=2))

    assert pre_roll.added == [b"listen"]
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
    pre_roll = PreRollStub()
    pre_roll.next_flush_payload = b"preroll-bytes"
    state_manager = StreamStateManager()

    actor = WakeWordActor(
        cast(Any, bus),
        WakeWordActorDependencies(
            state_manager=state_manager,
            silence_tracker=cast(Any, silence_tracker),
            pre_roll=cast(Any, pre_roll),
            run_wake_word_detection=lambda _: WakeWordDetection(),
            verbose_print=lambda *_: None,
            wake_engine=cast(Any, wake_engine),
        ),
    )
    actor._last_listening_chunk = (PREROLL_TRIGGER_CHUNK_INDEX, b"last-chunk")

    event = StateTransitionEvent(
        previous=StreamState.LISTENING,
        current=StreamState.STREAMING,
        reason="wake phrase detected",
        trigger="wake_phrase",
        trigger_chunk_index=PREROLL_TRIGGER_CHUNK_INDEX,
    )

    await bus.emit(event)

    assert pre_roll.flush_calls == 1
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
    silence_tracker = SilenceTrackerStub()
    results = [(True, True)]

    def observe(_: bytes) -> tuple[bool, bool]:
        return results.pop(0)

    SilenceActor(
        cast(Any, bus),
        auto_stop_enabled=True,
        observe_streaming_silence=observe,
        silence_tracker=cast(Any, silence_tracker),
    )

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
    assert silence_tracker.reset_calls == 1


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
    StreamUploaderActor(
        cast(Any, bus),
        send_preroll_payload=loop._send_preroll_payload,
        forward_audio=loop._forward_audio,
    )

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
    AssistantResponderActor(cast(Any, bus), finalize_turn=loop._finalize_turn)

    await bus.emit(FinalizeTurnEvent(reason="server stop"))

    assert loop.finalized == ["server stop"]


@pytest.mark.asyncio
async def test_speech_gate_actor_manual_stop_transitions_and_cleans_up():
    bus = DummyBus()
    state_manager = StreamStateManager()
    state_manager.transition_to_streaming()
    response_tasks = ResponseTasksStub()
    transcript_buffer = TranscriptBufferStub()
    silence_tracker = SilenceTrackerStub()
    logged: list[tuple[StreamState | None, StreamState, str]] = []
    notified: list[tuple[Any, ...]] = []
    reset_calls = 0
    clear_calls: list[str] = []

    async def _enter_streaming_state(**_: Any) -> bool:
        return True

    async def _notify_state_transition(
        previous: StreamState | None,
        current: StreamState,
        notification: StateTransitionNotification,
    ) -> None:
        notified.append(
            (
                previous,
                current,
                notification.reason,
                notification.trigger,
                notification.trigger_chunk_index,
                notification.priority,
            )
        )

    def _log_state_transition(
        previous: StreamState | None,
        current: StreamState,
        reason: str,
    ) -> None:
        logged.append((previous, current, reason))

    def _reset_stream_resources() -> None:
        nonlocal reset_calls
        reset_calls += 1

    def _clear_server_stop_timeout(cause: str) -> None:
        clear_calls.append(cause)

    SpeechGateActor(
        cast(Any, bus),
        SpeechGateActorDependencies(
            state_manager=state_manager,
            response_tasks=cast(Any, response_tasks),
            transcript_buffer=transcript_buffer,
            silence_tracker=cast(Any, silence_tracker),
            enter_streaming_state=_enter_streaming_state,
            notify_state_transition=_notify_state_transition,
            log_state_transition=_log_state_transition,
            reset_stream_resources=_reset_stream_resources,
            clear_server_stop_timeout=_clear_server_stop_timeout,
            schedule_server_stop_timeout=lambda: None,
            verbose_print=lambda *_: None,
        ),
        SpeechGateActorConfig(
            auto_stop_enabled=True,
            server_stop_min_silence_seconds=0.5,
        ),
    )

    event = ManualStopEvent(reason="manual halt")
    await bus.emit(event)

    assert state_manager.state == StreamState.LISTENING
    assert reset_calls == 1
    assert clear_calls == ["manual_stop"]
    assert transcript_buffer.cleared == ["manual halt"]
    assert response_tasks.cancelled == ["manual halt"]
    assert logged and logged[-1][2] == "manual halt"
    assert notified
    notified_event = notified[-1]
    assert notified_event[0] == StreamState.STREAMING
    assert notified_event[1] == StreamState.LISTENING
    assert notified_event[-1] is True
