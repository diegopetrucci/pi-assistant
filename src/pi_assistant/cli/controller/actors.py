"""Async actors that coordinate the event-driven audio controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from pi_assistant.audio.wake_word import (
    PreRollBuffer,
    StreamState,
    WakeWordDetection,
    WakeWordEngine,
)
from pi_assistant.cli.controller.components import (
    ResponseTaskManager,
    SilenceTracker,
    StreamStateManager,
)
from pi_assistant.cli.logging import CONTROL_LOG_LABEL, LOGGER, TURN_LOG_LABEL, WAKE_LOG_LABEL

from .events import (
    AudioChunkEvent,
    ControllerEventBus,
    FinalizeTurnEvent,
    ManualStopEvent,
    PrerollReadyEvent,
    ServerStopEvent,
    SilenceDetectedEvent,
    StateTransitionEvent,
    StateTransitionNotification,
    StreamChunkEvent,
    WakeWordTriggeredEvent,
)
from .helpers import should_ignore_server_stop_event


@dataclass(slots=True)
class WakeWordActorDependencies:
    """Bundle of dependencies required to construct a WakeWordActor."""

    state_manager: StreamStateManager
    silence_tracker: SilenceTracker
    pre_roll: PreRollBuffer
    run_wake_word_detection: Callable[[bytes], WakeWordDetection]
    wake_engine: Optional[WakeWordEngine] = None


class WakeWordActor:
    """Convert raw audio chunks into wake-word and streaming events."""

    def __init__(self, bus: ControllerEventBus, deps: WakeWordActorDependencies) -> None:
        self.bus = bus
        self.state_manager = deps.state_manager
        self.silence_tracker = deps.silence_tracker
        self.pre_roll = deps.pre_roll
        self._run_wake_word_detection = deps.run_wake_word_detection
        self.wake_engine = deps.wake_engine
        self._last_listening_chunk: tuple[int, bytes] | None = None
        bus.subscribe(AudioChunkEvent, self._on_audio_chunk)
        bus.subscribe(StateTransitionEvent, self._on_state_transition)

    async def _on_audio_chunk(self, event: AudioChunkEvent) -> None:
        state = self.state_manager.state
        if state == StreamState.LISTENING:
            self._last_listening_chunk = (event.index, event.chunk)
            self.pre_roll.add(event.chunk)

        detection = self._run_wake_word_detection(event.chunk)
        if detection.triggered:
            await self.bus.publish(
                WakeWordTriggeredEvent(
                    chunk=event.chunk,
                    chunk_index=event.index,
                    detection=detection,
                    state_at_detection=state,
                ),
                priority=True,
            )
        if state == StreamState.STREAMING:
            await self.bus.publish(StreamChunkEvent(payload=event.chunk, chunk_index=event.index))

    async def _on_state_transition(self, event: StateTransitionEvent) -> None:
        if event.current == StreamState.STREAMING:
            self.silence_tracker.reset()
            payload = self.pre_roll.flush()
            if payload:
                await self.bus.publish(
                    PrerollReadyEvent(payload=payload, chunk_index=event.trigger_chunk_index)
                )
            elif self._last_listening_chunk and self._should_emit_last_listening_chunk(
                event, self._last_listening_chunk
            ):
                await self.bus.publish(
                    StreamChunkEvent(
                        payload=self._last_listening_chunk[1],
                        chunk_index=self._last_listening_chunk[0],
                    )
                )
            else:
                LOGGER.verbose(WAKE_LOG_LABEL, "Triggered -> streaming (no buffered audio)")
            if self._last_listening_chunk:
                self.silence_tracker.observe(self._last_listening_chunk[1])
            self._last_listening_chunk = None
            if self.wake_engine:
                self.wake_engine.reset_detection()
        elif event.current == StreamState.LISTENING:
            self._last_listening_chunk = None

    def _should_emit_last_listening_chunk(
        self,
        event: StateTransitionEvent,
        last_chunk: tuple[int, bytes],
    ) -> bool:
        if event.trigger_chunk_index is None:
            return False
        return event.trigger_chunk_index == last_chunk[0]


class SilenceActor:
    """Surface silence events for the speech gate to evaluate."""

    def __init__(
        self,
        bus: ControllerEventBus,
        *,
        auto_stop_enabled: bool,
        observe_streaming_silence: Callable[[bytes], tuple[bool, bool]],
        silence_tracker: SilenceTracker,
    ) -> None:
        self.bus = bus
        self._auto_stop_enabled = auto_stop_enabled
        self._observe_streaming_silence = observe_streaming_silence
        self._silence_tracker = silence_tracker
        bus.subscribe(AudioChunkEvent, self._on_audio_chunk)
        bus.subscribe(StateTransitionEvent, self._on_state_transition)

    async def _on_audio_chunk(self, event: AudioChunkEvent) -> None:
        silence_reached, observed = self._observe_streaming_silence(event.chunk)
        if self._auto_stop_enabled and silence_reached and observed:
            await self.bus.publish(
                SilenceDetectedEvent(chunk_index=event.index),
                priority=True,
            )

    async def _on_state_transition(self, event: StateTransitionEvent) -> None:
        if event.current == StreamState.LISTENING:
            self._silence_tracker.reset()


class StreamUploaderActor:
    """Forward prepared audio payloads to the websocket client."""

    def __init__(
        self,
        bus: ControllerEventBus,
        *,
        send_preroll_payload: Callable[[bytes], Awaitable[None]],
        forward_audio: Callable[[bytes], Awaitable[None]],
    ) -> None:
        self._send_preroll_payload = send_preroll_payload
        self._forward_audio = forward_audio
        bus.subscribe(PrerollReadyEvent, self._on_preroll_ready)
        bus.subscribe(StreamChunkEvent, self._on_stream_chunk)

    async def _on_preroll_ready(self, event: PrerollReadyEvent) -> None:
        await self._send_preroll_payload(event.payload)

    async def _on_stream_chunk(self, event: StreamChunkEvent) -> None:
        await self._forward_audio(event.payload)


class AssistantResponderActor:
    """Schedule assistant responses whenever a turn finalizes."""

    def __init__(
        self,
        bus: ControllerEventBus,
        finalize_turn: Callable[[str], None],
    ) -> None:
        self._finalize_turn = finalize_turn
        bus.subscribe(FinalizeTurnEvent, self._on_finalize)

    async def _on_finalize(self, event: FinalizeTurnEvent) -> None:
        self._finalize_turn(event.reason)


@dataclass(slots=True)
class SpeechGateActorDependencies:
    """Bundle of orchestration helpers used by the SpeechGateActor."""

    state_manager: StreamStateManager
    response_tasks: ResponseTaskManager
    transcript_buffer: Any
    silence_tracker: SilenceTracker
    enter_streaming_state: Callable[..., Awaitable[bool]]
    notify_state_transition: Callable[
        [StreamState | None, StreamState, StateTransitionNotification], Awaitable[None]
    ]
    log_state_transition: Callable[[StreamState | None, StreamState, str], None]
    reset_stream_resources: Callable[[], None]
    clear_server_stop_timeout: Callable[[str], None]
    schedule_server_stop_timeout: Callable[[], None]


@dataclass(slots=True)
class SpeechGateActorConfig:
    """Configuration toggles for SpeechGateActor behavior."""

    auto_stop_enabled: bool
    server_stop_min_silence_seconds: float


class SpeechGateActor:
    """Coordinate state transitions based on wake events and control signals."""

    def __init__(
        self,
        bus: ControllerEventBus,
        deps: SpeechGateActorDependencies,
        config: SpeechGateActorConfig,
    ) -> None:
        self.bus = bus
        self.state_manager = deps.state_manager
        self.response_tasks = deps.response_tasks
        self.transcript_buffer = deps.transcript_buffer
        self.silence_tracker = deps.silence_tracker
        self._enter_streaming_state = deps.enter_streaming_state
        self._notify_state_transition = deps.notify_state_transition
        self._log_state_transition = deps.log_state_transition
        self._reset_stream_resources = deps.reset_stream_resources
        self._clear_server_stop_timeout = deps.clear_server_stop_timeout
        self._schedule_server_stop_timeout = deps.schedule_server_stop_timeout
        self._auto_stop_enabled = config.auto_stop_enabled
        self._server_stop_min_silence_seconds = config.server_stop_min_silence_seconds
        bus.subscribe(WakeWordTriggeredEvent, self._on_wake_word)
        bus.subscribe(ManualStopEvent, self._on_manual_stop)
        bus.subscribe(ServerStopEvent, self._on_server_stop)
        bus.subscribe(SilenceDetectedEvent, self._on_silence_detected)

    async def _on_wake_word(self, event: WakeWordTriggeredEvent) -> None:
        state = event.state_at_detection
        if self.state_manager.awaiting_server_stop:
            LOGGER.verbose(
                WAKE_LOG_LABEL,
                "Wake phrase ignored while awaiting server stop confirmation.",
            )
            return
        if state != StreamState.LISTENING:
            retrigger_count = self.state_manager.increment_retrigger_budget()
            LOGGER.verbose(
                WAKE_LOG_LABEL,
                f"Wake word retrigger detected during streaming (count={retrigger_count})",
            )
            return
        if self.state_manager.finalizing_turn:
            LOGGER.verbose(
                WAKE_LOG_LABEL,
                "Wake phrase ignored while finalizing previous turn.",
            )
            return
        if self.state_manager.awaiting_assistant_reply:
            LOGGER.verbose(TURN_LOG_LABEL, "Wake phrase overriding assistant reply.")
            self.state_manager.clear_awaiting_assistant_reply()
            self.response_tasks.cancel("wake phrase override")
        transitioned = await self._enter_streaming_state(
            trigger="wake_phrase",
            reason="wake phrase detected",
        )
        if not transitioned:
            return
        await self._notify_state_transition(
            event.state_at_detection,
            self.state_manager.state,
            StateTransitionNotification(
                reason="wake phrase detected",
                trigger="wake_phrase",
                trigger_chunk_index=event.chunk_index,
                priority=True,
            ),
        )

    async def _on_manual_stop(self, event: ManualStopEvent) -> None:
        if self.state_manager.state != StreamState.STREAMING:
            return
        LOGGER.verbose(CONTROL_LOG_LABEL, "Stop command received; returning to listening.")
        previous_state = self.state_manager.transition_to_listening(event.reason)
        if previous_state:
            self._log_state_transition(previous_state, self.state_manager.state, event.reason)
            await self._notify_state_transition(
                previous_state,
                self.state_manager.state,
                StateTransitionNotification(
                    reason=event.reason,
                    trigger="manual_stop",
                    priority=True,
                ),
            )
            self._reset_stream_resources()
        self._clear_server_stop_timeout("manual_stop")
        await self.transcript_buffer.clear_current_turn(event.reason)
        self.response_tasks.cancel(event.reason)

    async def _on_server_stop(self, event: ServerStopEvent) -> None:
        if self.state_manager.consume_suppressed_stop_event():
            LOGGER.verbose(
                TURN_LOG_LABEL,
                "Ignoring stale server speech stop acknowledgement.",
            )
            return
        ignore_reason = should_ignore_server_stop_event(
            self.state_manager,
            self.silence_tracker,
            self._server_stop_min_silence_seconds,
        )
        if ignore_reason:
            LOGGER.verbose(TURN_LOG_LABEL, f"Server speech stop ignored: {ignore_reason}")
            return
        self._clear_server_stop_timeout("server_ack")
        if self.state_manager.awaiting_server_stop:
            reason = self.state_manager.complete_deferred_finalize(event.reason)
            if reason:
                await self.bus.publish(
                    FinalizeTurnEvent(reason=reason),
                    priority=True,
                )
            return
        previous_state = self.state_manager.transition_to_listening(event.reason)
        if previous_state:
            self._log_state_transition(previous_state, self.state_manager.state, event.reason)
            await self._notify_state_transition(
                previous_state,
                self.state_manager.state,
                StateTransitionNotification(
                    reason=event.reason,
                    trigger="server_stop",
                    priority=True,
                ),
            )
            self._reset_stream_resources()
            await self.bus.publish(
                FinalizeTurnEvent(reason=event.reason),
                priority=True,
            )

    async def _on_silence_detected(self, event: SilenceDetectedEvent) -> None:
        if not self._auto_stop_enabled:
            return
        if self.state_manager.retrigger_budget == 0:
            if self.state_manager.awaiting_server_stop:
                LOGGER.verbose(
                    TURN_LOG_LABEL,
                    (
                        "Silence timer fired but awaiting server stop; skipping "
                        "duplicate close request."
                    ),
                )
                return
            previous_state = self.state_manager.transition_to_listening(
                "silence detected",
                defer_finalize=True,
            )
            if previous_state:
                self._log_state_transition(
                    previous_state, self.state_manager.state, "silence detected"
                )
                await self._notify_state_transition(
                    previous_state,
                    self.state_manager.state,
                    StateTransitionNotification(
                        reason="silence detected",
                        trigger="auto_stop",
                        trigger_chunk_index=event.chunk_index,
                        priority=True,
                    ),
                )
                self._reset_stream_resources()
                LOGGER.verbose(
                    TURN_LOG_LABEL,
                    "Awaiting server confirmation before finalizing turn.",
                )
                self._schedule_server_stop_timeout()
            return
        retrigger_count = self.state_manager.retrigger_budget
        retrigger_log = (
            f"Silence detected but {retrigger_count} retrigger(s) observed; keeping stream open"
        )
        LOGGER.verbose(TURN_LOG_LABEL, retrigger_log)
        self.state_manager.reset_retrigger_budget()
        self.silence_tracker.clear_silence()


__all__ = [
    "AssistantResponderActor",
    "SilenceActor",
    "SpeechGateActor",
    "SpeechGateActorConfig",
    "SpeechGateActorDependencies",
    "StreamUploaderActor",
    "WakeWordActor",
    "WakeWordActorDependencies",
]
