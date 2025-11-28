"""Private helpers for running the CLI audio controller loop."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from typing import Any, Optional
from uuid import uuid4

from pi_assistant.cli.controller_components import (
    AudioChunkPreparer,
    ResponseTaskManager,
    SilenceTracker,
    StreamStateManager,
)
from pi_assistant.cli.controller_context import AudioControllerContext
from pi_assistant.cli.logging_utils import (
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    TURN_LOG_LABEL,
    WAKE_LOG_LABEL,
)
from pi_assistant.config import (
    CHANNELS,
    PREROLL_DURATION_SECONDS,
    SAMPLE_RATE,
    WAKE_WORD_CONSECUTIVE_FRAMES,
    WAKE_WORD_EMBEDDING_MODEL_PATH,
    WAKE_WORD_MELSPEC_MODEL_PATH,
    WAKE_WORD_MODEL_FALLBACK_PATH,
    WAKE_WORD_MODEL_PATH,
    WAKE_WORD_NAME,
    WAKE_WORD_PHRASE,
    WAKE_WORD_SCORE_THRESHOLD,
    WAKE_WORD_TARGET_SAMPLE_RATE,
)
from pi_assistant.network import WebSocketClient
from pi_assistant.wake_word import (
    StreamState,
    WakeWordDetection,
    WakeWordEngine,
)

from .controller_actors import (
    AssistantResponderActor,
    SilenceActor,
    SpeechGateActor,
    SpeechGateActorConfig,
    SpeechGateActorDependencies,
    StreamUploaderActor,
    WakeWordActor,
    WakeWordActorDependencies,
)
from .controller_events import (
    AudioChunkEvent,
    ControllerEventBus,
    ManualStopEvent,
    ServerStopEvent,
    StateTransitionEvent,
    StateTransitionNotification,
)
from .controller_helpers import ResponseLifecycleHooks
from .controller_loop_bindings import (
    ControllerLoopBindings,
    build_controller_loop_bindings,
)
from .controller_loop_server_stop import ServerStopTimeoutMixin


class _AudioControllerLoop(ServerStopTimeoutMixin):
    def __init__(
        self,
        audio_capture,
        ws_client: WebSocketClient,
        context: AudioControllerContext,
        bindings: ControllerLoopBindings | None = None,
    ) -> None:
        self.bindings = bindings or build_controller_loop_bindings()
        self._controller_module = self.bindings.controller_module
        self.audio_capture = audio_capture
        self.ws_client = ws_client
        self.context = context
        self.capture_sample_rate = SAMPLE_RATE
        self.bytes_per_frame = 2 * CHANNELS
        self.stream_sample_rate = self._controller_module.STREAM_SAMPLE_RATE
        self.pre_roll = self.bindings.pre_roll_factory(
            PREROLL_DURATION_SECONDS,
            self.capture_sample_rate,
            channels=CHANNELS,
        )
        self.chunk_preparer = AudioChunkPreparer(
            self.capture_sample_rate,
            self.stream_sample_rate,
            resampler_factory=self.bindings.resampler_factory,
        )
        silence_threshold = self._controller_module.AUTO_STOP_SILENCE_THRESHOLD
        max_silence_seconds = self._controller_module.AUTO_STOP_MAX_SILENCE_SECONDS
        self.silence_tracker = SilenceTracker(
            silence_threshold=silence_threshold,
            max_silence_seconds=max_silence_seconds,
            sample_rate=self.capture_sample_rate,
        )
        self.state_manager = StreamStateManager()
        self.lifecycle_hooks = ResponseLifecycleHooks(
            on_transcript_ready=self.state_manager.clear_finalizing_turn,
            on_reply_start=self.state_manager.mark_awaiting_assistant_reply,
            on_reply_complete=self.state_manager.clear_awaiting_assistant_reply,
        )
        self.response_tasks = ResponseTaskManager(task_factory=self._build_task_factory())
        self.wake_engine = self._create_wake_engine()
        self._server_stop_timeout_handle: asyncio.TimerHandle | None = None
        self._server_stop_timeout_task: asyncio.Task[None] | None = None
        self._server_stop_timeout_turn_id: str | None = None
        self._session_id = uuid4().hex
        self._turn_sequence = 0
        self._active_turn_id: str | None = None
        self._event_bus: ControllerEventBus | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._actors: list[Any] = []

    def _build_task_factory(self) -> Callable[[], asyncio.Task]:
        def _factory() -> asyncio.Task:
            scheduler = self._controller_module.schedule_turn_response
            return scheduler(
                self.context.transcript_buffer,
                self.context.assistant,
                self.context.speech_player,
                hooks=self.lifecycle_hooks,
            )

        return _factory

    def _create_wake_engine(self) -> Optional[WakeWordEngine]:
        try:
            engine = self.bindings.wake_engine_cls(
                WAKE_WORD_MODEL_PATH,
                fallback_model_path=WAKE_WORD_MODEL_FALLBACK_PATH,
                melspec_model_path=WAKE_WORD_MELSPEC_MODEL_PATH,
                embedding_model_path=WAKE_WORD_EMBEDDING_MODEL_PATH,
                source_sample_rate=self.capture_sample_rate,
                target_sample_rate=WAKE_WORD_TARGET_SAMPLE_RATE,
                threshold=WAKE_WORD_SCORE_THRESHOLD,
                consecutive_required=WAKE_WORD_CONSECUTIVE_FRAMES,
            )
            self._verbose_print(
                f"{WAKE_LOG_LABEL} Wake-word model: {WAKE_WORD_PHRASE} ({WAKE_WORD_NAME})"
            )
            return engine
        except RuntimeError as exc:
            print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
            raise

    async def run(self) -> None:
        self._log_startup()
        self._shutdown_event = asyncio.Event()
        self._event_bus = ControllerEventBus()
        self._install_actors()
        tasks = [
            asyncio.create_task(self._event_bus.run(), name="controller-event-bus"),
            asyncio.create_task(self._run_audio_producer(), name="controller-audio-producer"),
            asyncio.create_task(self._watch_stop_signal(), name="controller-stop-signal"),
            asyncio.create_task(
                self._watch_server_stop_signal(),
                name="controller-server-stop",
            ),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            processed_chunks = self.state_manager.chunk_count
            self._verbose_print(
                f"[INFO] Audio controller stopped ({processed_chunks} chunks processed)"
            )
            raise
        except Exception as exc:
            print(f"{ERROR_LOG_LABEL} Audio controller error: {exc}", file=sys.stderr)
            raise
        finally:
            if self._shutdown_event:
                self._shutdown_event.set()
            self._clear_server_stop_timeout(cause="shutdown")
            if self._event_bus:
                await self._event_bus.shutdown()
            for task in tasks:
                task.cancel()
            gather_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in gather_results:
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    print(
                        f"{ERROR_LOG_LABEL} Exception during task cancellation: {result}",
                        file=sys.stderr,
                    )
            await self._await_server_stop_timeout_task()
            await self.response_tasks.drain()

    def _install_actors(self) -> None:
        if not self._event_bus:
            return

        async def _notify_transition(
            previous: StreamState | None,
            current: StreamState,
            notification: StateTransitionNotification,
        ) -> None:
            await self._notify_state_transition(
                previous,
                current,
                notification=notification,
            )

        def _clear_server_stop_timeout(cause: str) -> None:
            self._clear_server_stop_timeout(cause=cause)

        self._actors = [
            WakeWordActor(
                bus=self._event_bus,
                deps=WakeWordActorDependencies(
                    state_manager=self.state_manager,
                    silence_tracker=self.silence_tracker,
                    pre_roll=self.pre_roll,
                    run_wake_word_detection=self._run_wake_word_detection,
                    verbose_print=self._verbose_print,
                    wake_engine=self.wake_engine,
                ),
            ),
            SilenceActor(
                bus=self._event_bus,
                auto_stop_enabled=self._auto_stop_enabled,
                observe_streaming_silence=self._observe_streaming_silence,
                silence_tracker=self.silence_tracker,
            ),
            StreamUploaderActor(
                bus=self._event_bus,
                send_preroll_payload=self._send_preroll_payload,
                forward_audio=self._forward_audio,
            ),
            AssistantResponderActor(
                bus=self._event_bus,
                finalize_turn=self._finalize_turn,
            ),
            SpeechGateActor(
                bus=self._event_bus,
                deps=SpeechGateActorDependencies(
                    state_manager=self.state_manager,
                    response_tasks=self.response_tasks,
                    transcript_buffer=self.context.transcript_buffer,
                    silence_tracker=self.silence_tracker,
                    enter_streaming_state=self._enter_streaming_state,
                    notify_state_transition=_notify_transition,
                    log_state_transition=self._log_state_transition,
                    reset_stream_resources=self._reset_stream_resources,
                    clear_server_stop_timeout=_clear_server_stop_timeout,
                    schedule_server_stop_timeout=self._schedule_server_stop_timeout,
                    verbose_print=self._verbose_print,
                ),
                config=SpeechGateActorConfig(
                    auto_stop_enabled=self._auto_stop_enabled,
                    server_stop_min_silence_seconds=self._server_stop_min_silence_seconds,
                ),
            ),
        ]

    async def _run_audio_producer(self) -> None:
        chunk_task: Optional[asyncio.Task[bytes]] = None
        try:
            while not (self._shutdown_event and self._shutdown_event.is_set()):
                if chunk_task is None:
                    chunk_task = asyncio.create_task(self.audio_capture.get_audio_chunk())
                done, _ = await asyncio.wait({chunk_task}, timeout=0.1)
                if not done:
                    continue
                audio_bytes = chunk_task.result()
                chunk_task = None
                if len(audio_bytes) % self.bytes_per_frame != 0:
                    print(
                        f"{ERROR_LOG_LABEL} Dropping malformed audio chunk: "
                        f"{len(audio_bytes)} bytes (expected multiple of {self.bytes_per_frame}).",
                        file=sys.stderr,
                    )
                    continue
                chunk_index = self.state_manager.increment_chunk_count()
                if self._event_bus:
                    await self._event_bus.publish(
                        AudioChunkEvent(chunk=audio_bytes, index=chunk_index)
                    )
                self._log_chunk_progress(chunk_index)
        except asyncio.CancelledError:
            raise
        finally:
            if chunk_task:
                chunk_task.cancel()
                with contextlib.suppress(Exception):
                    await chunk_task

    async def _watch_stop_signal(self) -> None:
        try:
            while True:
                await self.context.stop_signal.wait()
                if self._shutdown_event and self._shutdown_event.is_set():
                    return
                self.context.stop_signal.clear()
                if self._event_bus:
                    await self._event_bus.publish(ManualStopEvent())
        except asyncio.CancelledError:
            pass

    async def _watch_server_stop_signal(self) -> None:
        try:
            while True:
                await self.context.speech_stopped_signal.wait()
                if self._shutdown_event and self._shutdown_event.is_set():
                    return
                self.context.speech_stopped_signal.clear()
                if self._event_bus:
                    await self._event_bus.publish(ServerStopEvent())
        except asyncio.CancelledError:
            pass

    def _log_startup(self) -> None:
        self._verbose_print("[INFO] Starting audio controller...")
        if self.chunk_preparer.is_resampling:
            self._verbose_print(
                f"{CONTROL_LOG_LABEL} Resampling capture audio "
                f"{self.capture_sample_rate} Hz -> {self.stream_sample_rate} Hz for OpenAI."
            )
        else:
            self._verbose_print(
                f"{CONTROL_LOG_LABEL} Streaming audio at {self.stream_sample_rate} Hz."
            )
        self._log_state_transition(None, self.state_manager.state, "awaiting wake phrase")

    def _observe_streaming_silence(self, audio_bytes: bytes) -> tuple[bool, bool]:
        if self.state_manager.state != StreamState.STREAMING:
            return False, False
        return self.silence_tracker.observe(audio_bytes), True

    def _buffer_preroll_if_listening(self, audio_bytes: bytes) -> None:
        if self.state_manager.state == StreamState.LISTENING:
            self.pre_roll.add(audio_bytes)

    def _run_wake_word_detection(self, audio_bytes: bytes) -> WakeWordDetection:
        if not self.wake_engine:
            return WakeWordDetection()
        detection = self.wake_engine.process_chunk(audio_bytes)
        if detection.score >= WAKE_WORD_SCORE_THRESHOLD:
            score_log = (
                f"{WAKE_LOG_LABEL} score={detection.score:.2f} "
                f"(threshold {WAKE_WORD_SCORE_THRESHOLD:.2f}) "
                f"state={self.state_manager.state.value}"
            )
            self._verbose_print(score_log)
        return detection

    async def _enter_streaming_state(self, *, trigger: str, reason: str) -> bool:
        """Transition to STREAMING and start a new turn if the state actually changed."""

        previous_state = self.state_manager.transition_to_streaming()
        if not previous_state:
            return False
        self._start_new_turn(trigger)
        await self.context.transcript_buffer.start_turn()
        self._log_state_transition(previous_state, self.state_manager.state, reason)
        return True

    async def _send_preroll_payload(self, payload: bytes) -> None:
        buffered_frames = len(payload) // self.bytes_per_frame
        duration_ms = buffered_frames / self.capture_sample_rate * 1000
        self._verbose_print(
            f"{WAKE_LOG_LABEL} Triggered -> streaming (sent {duration_ms:.0f} ms of buffered audio)"
        )
        resampled_payload = self.chunk_preparer.prepare(payload)
        if resampled_payload:
            await self.ws_client.send_audio_chunk(resampled_payload)

    async def _forward_audio(self, audio_bytes: bytes) -> None:
        stream_chunk = self.chunk_preparer.prepare(audio_bytes)
        if stream_chunk:
            await self.ws_client.send_audio_chunk(stream_chunk)

    def _log_chunk_progress(self, chunk_index: int) -> None:
        if chunk_index % 100 != 0:
            return
        debug_log = (
            f"[DEBUG] Processed {chunk_index} audio chunks (state={self.state_manager.state.value})"
        )
        self._verbose_print(debug_log)

    def _reset_stream_resources(self) -> None:
        self.pre_roll.clear()
        self.silence_tracker.reset()
        self.chunk_preparer.reset()
        if self.wake_engine:
            self.wake_engine.reset_detection()

    def _finalize_turn(self, reason: str) -> None:
        self._audit_log("turn_finalize", reason=reason)
        self.state_manager.mark_finalizing_turn()
        self.response_tasks.schedule(reason)
        self._active_turn_id = None

    def _start_new_turn(self, trigger: str) -> None:
        self._turn_sequence += 1
        self._active_turn_id = f"turn-{self._turn_sequence:04d}"
        self._audit_log("turn_started", trigger=trigger)

    def _audit_log(self, action: str, **fields: Any) -> None:
        context: dict[str, Any] = {
            "session": self._session_id,
            "state": self.state_manager.state.value,
        }
        if self._active_turn_id:
            context["turn"] = self._active_turn_id
        context.update(fields)
        kv_pairs = " ".join(f"{key}={value!r}" for key, value in context.items())
        self._console_print(f"{TURN_LOG_LABEL} action={action} {kv_pairs}")

    @property
    def _auto_stop_enabled(self) -> bool:
        return self._controller_module.AUTO_STOP_ENABLED

    @property
    def _server_stop_timeout_seconds(self) -> float:
        return self._controller_module.SERVER_STOP_TIMEOUT_SECONDS

    @property
    def _server_stop_min_silence_seconds(self) -> float:
        return self._controller_module.SERVER_STOP_MIN_SILENCE_SECONDS

    def _console_print(self, message: str) -> None:
        self._controller_module.console_print(message)

    def _verbose_print(self, message: str) -> None:
        self._controller_module.verbose_print(message)

    def _log_state_transition(
        self,
        previous: StreamState | None,
        current: StreamState,
        reason: str,
    ) -> None:
        self._controller_module.log_state_transition(previous, current, reason)

    async def _notify_state_transition(
        self,
        previous: StreamState | None,
        current: StreamState,
        *,
        notification: StateTransitionNotification,
    ) -> None:
        if previous is None or not self._event_bus:
            return
        await self._event_bus.publish(
            StateTransitionEvent(
                previous=previous,
                current=current,
                reason=notification.reason,
                trigger=notification.trigger,
                trigger_chunk_index=notification.trigger_chunk_index,
            ),
            priority=notification.priority,
        )


__all__ = ["AudioControllerContext", "ControllerLoopBindings", "_AudioControllerLoop"]
