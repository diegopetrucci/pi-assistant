"""Async actors that coordinate the event-driven audio controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_assistant.cli.logging_utils import CONTROL_LOG_LABEL, TURN_LOG_LABEL, WAKE_LOG_LABEL
from pi_assistant.wake_word import StreamState

from .controller_events import (
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
from .controller_helpers import should_ignore_server_stop_event

if TYPE_CHECKING:
    from .controller_loop import _AudioControllerLoop


class WakeWordActor:
    """Convert raw audio chunks into wake-word and streaming events."""

    def __init__(self, loop: "_AudioControllerLoop", bus: ControllerEventBus) -> None:
        self.loop = loop
        self.bus = bus
        self._last_listening_chunk: tuple[int, bytes] | None = None
        bus.subscribe(AudioChunkEvent, self._on_audio_chunk)
        bus.subscribe(StateTransitionEvent, self._on_state_transition)

    async def _on_audio_chunk(self, event: AudioChunkEvent) -> None:
        state = self.loop.state_manager.state
        if state == StreamState.LISTENING:
            self._last_listening_chunk = (event.index, event.chunk)
            self.loop._buffer_preroll_if_listening(event.chunk)

        detection = self.loop._run_wake_word_detection(event.chunk)
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
            self.loop.silence_tracker.reset()
            payload = self.loop.pre_roll.flush()
            if payload:
                await self.bus.publish(
                    PrerollReadyEvent(payload=payload, chunk_index=event.trigger_chunk_index)
                )
            elif (
                event.trigger_chunk_index is not None
                and self._last_listening_chunk
                and event.trigger_chunk_index == self._last_listening_chunk[0]
            ):
                await self.bus.publish(
                    StreamChunkEvent(
                        payload=self._last_listening_chunk[1],
                        chunk_index=self._last_listening_chunk[0],
                    )
                )
            else:
                self.loop._verbose_print(
                    f"{WAKE_LOG_LABEL} Triggered -> streaming (no buffered audio)"
                )
            if self._last_listening_chunk:
                self.loop.silence_tracker.observe(self._last_listening_chunk[1])
            self._last_listening_chunk = None
            if self.loop.wake_engine:
                self.loop.wake_engine.reset_detection()
        elif event.current == StreamState.LISTENING:
            self._last_listening_chunk = None


class SilenceActor:
    """Surface silence events for the speech gate to evaluate."""

    def __init__(self, loop: "_AudioControllerLoop", bus: ControllerEventBus) -> None:
        self.loop = loop
        self.bus = bus
        bus.subscribe(AudioChunkEvent, self._on_audio_chunk)
        bus.subscribe(StateTransitionEvent, self._on_state_transition)

    async def _on_audio_chunk(self, event: AudioChunkEvent) -> None:
        silence_reached, observed = self.loop._observe_streaming_silence(event.chunk)
        if self.loop._auto_stop_enabled and silence_reached and observed:
            await self.bus.publish(
                SilenceDetectedEvent(chunk_index=event.index),
                priority=True,
            )

    async def _on_state_transition(self, event: StateTransitionEvent) -> None:
        if event.current == StreamState.LISTENING:
            self.loop.silence_tracker.reset()


class StreamUploaderActor:
    """Forward prepared audio payloads to the websocket client."""

    def __init__(self, loop: "_AudioControllerLoop", bus: ControllerEventBus) -> None:
        self.loop = loop
        bus.subscribe(PrerollReadyEvent, self._on_preroll_ready)
        bus.subscribe(StreamChunkEvent, self._on_stream_chunk)

    async def _on_preroll_ready(self, event: PrerollReadyEvent) -> None:
        await self.loop._send_preroll_payload(event.payload)

    async def _on_stream_chunk(self, event: StreamChunkEvent) -> None:
        await self.loop._forward_audio(event.payload)


class AssistantResponderActor:
    """Schedule assistant responses whenever a turn finalizes."""

    def __init__(self, loop: "_AudioControllerLoop", bus: ControllerEventBus) -> None:
        self.loop = loop
        bus.subscribe(FinalizeTurnEvent, self._on_finalize)

    async def _on_finalize(self, event: FinalizeTurnEvent) -> None:
        self.loop._finalize_turn(event.reason)


class SpeechGateActor:
    """Coordinate state transitions based on wake events and control signals."""

    def __init__(self, loop: "_AudioControllerLoop", bus: ControllerEventBus) -> None:
        self.loop = loop
        self.bus = bus
        bus.subscribe(WakeWordTriggeredEvent, self._on_wake_word)
        bus.subscribe(ManualStopEvent, self._on_manual_stop)
        bus.subscribe(ServerStopEvent, self._on_server_stop)
        bus.subscribe(SilenceDetectedEvent, self._on_silence_detected)

    async def _on_wake_word(self, event: WakeWordTriggeredEvent) -> None:
        state = event.state_at_detection
        if self.loop.state_manager.awaiting_server_stop:
            self.loop._verbose_print(
                f"{WAKE_LOG_LABEL} Wake phrase ignored while awaiting server stop confirmation."
            )
            return
        if state != StreamState.LISTENING:
            retrigger_count = self.loop.state_manager.increment_retrigger_budget()
            print(
                f"{WAKE_LOG_LABEL} Wake word retrigger detected during streaming "
                f"(count={retrigger_count})"
            )
            return
        if self.loop.state_manager.finalizing_turn:
            self.loop._verbose_print(
                f"{WAKE_LOG_LABEL} Wake phrase ignored while finalizing previous turn."
            )
            return
        if self.loop.state_manager.awaiting_assistant_reply:
            self.loop._verbose_print(f"{TURN_LOG_LABEL} Wake phrase overriding assistant reply.")
            self.loop.state_manager.clear_awaiting_assistant_reply()
            self.loop.response_tasks.cancel("wake phrase override")
        transitioned = await self.loop._enter_streaming_state(
            trigger="wake_phrase",
            reason="wake phrase detected",
        )
        if not transitioned:
            return
        await self.loop._notify_state_transition(
            StreamState.LISTENING,
            self.loop.state_manager.state,
            notification=StateTransitionNotification(
                reason="wake phrase detected",
                trigger="wake_phrase",
                trigger_chunk_index=event.chunk_index,
                priority=True,
            ),
        )

    async def _on_manual_stop(self, event: ManualStopEvent) -> None:
        if self.loop.state_manager.state != StreamState.STREAMING:
            return
        self.loop._verbose_print(
            f"{CONTROL_LOG_LABEL} Stop command received; returning to listening."
        )
        previous_state = self.loop.state_manager.transition_to_listening(event.reason)
        if previous_state:
            self.loop._log_state_transition(
                previous_state, self.loop.state_manager.state, event.reason
            )
            await self.loop._notify_state_transition(
                previous_state,
                self.loop.state_manager.state,
                notification=StateTransitionNotification(
                    reason=event.reason,
                    trigger="manual_stop",
                    priority=True,
                ),
            )
            self.loop._reset_stream_resources()
        self.loop._clear_server_stop_timeout(cause="manual_stop")
        await self.loop.context.transcript_buffer.clear_current_turn(event.reason)
        self.loop.response_tasks.cancel(event.reason)

    async def _on_server_stop(self, event: ServerStopEvent) -> None:
        if self.loop.state_manager.consume_suppressed_stop_event():
            self.loop._verbose_print(
                f"{TURN_LOG_LABEL} Ignoring stale server speech stop acknowledgement."
            )
            return
        ignore_reason = should_ignore_server_stop_event(
            self.loop.state_manager,
            self.loop.silence_tracker,
            self.loop._server_stop_min_silence_seconds,
        )
        if ignore_reason:
            self.loop._verbose_print(
                f"{TURN_LOG_LABEL} Server speech stop ignored: {ignore_reason}"
            )
            return
        self.loop._clear_server_stop_timeout(cause="server_ack")
        if self.loop.state_manager.awaiting_server_stop:
            reason = self.loop.state_manager.complete_deferred_finalize(event.reason)
            if reason:
                await self.bus.publish(
                    FinalizeTurnEvent(reason=reason),
                    priority=True,
                )
            return
        previous_state = self.loop.state_manager.transition_to_listening(event.reason)
        if previous_state:
            self.loop._log_state_transition(
                previous_state, self.loop.state_manager.state, event.reason
            )
            await self.loop._notify_state_transition(
                previous_state,
                self.loop.state_manager.state,
                notification=StateTransitionNotification(
                    reason=event.reason,
                    trigger="server_stop",
                    priority=True,
                ),
            )
            self.loop._reset_stream_resources()
            await self.bus.publish(
                FinalizeTurnEvent(reason=event.reason),
                priority=True,
            )

    async def _on_silence_detected(self, event: SilenceDetectedEvent) -> None:
        if not self.loop._auto_stop_enabled:
            return
        state_manager = self.loop.state_manager
        if state_manager.retrigger_budget == 0:
            if state_manager.awaiting_server_stop:
                self.loop._verbose_print(
                    f"{TURN_LOG_LABEL} Silence timer fired but awaiting server stop; "
                    "skipping duplicate close request."
                )
                return
            previous_state = state_manager.transition_to_listening(
                "silence detected",
                defer_finalize=True,
            )
            if previous_state:
                self.loop._log_state_transition(
                    previous_state, self.loop.state_manager.state, "silence detected"
                )
                await self.loop._notify_state_transition(
                    previous_state,
                    self.loop.state_manager.state,
                    notification=StateTransitionNotification(
                        reason="silence detected",
                        trigger="auto_stop",
                        trigger_chunk_index=event.chunk_index,
                        priority=True,
                    ),
                )
                self.loop._reset_stream_resources()
                self.loop._verbose_print(
                    f"{TURN_LOG_LABEL} Awaiting server confirmation before finalizing turn."
                )
                self.loop._schedule_server_stop_timeout()
            return
        retrigger_count = state_manager.retrigger_budget
        retrigger_log = (
            f"{TURN_LOG_LABEL} Silence detected but "
            f"{retrigger_count} retrigger(s) observed; keeping stream open"
        )
        self.loop._verbose_print(retrigger_log)
        state_manager.reset_retrigger_budget()
        self.loop.silence_tracker.clear_silence()


__all__ = [
    "AssistantResponderActor",
    "SilenceActor",
    "SpeechGateActor",
    "StreamUploaderActor",
    "WakeWordActor",
]
