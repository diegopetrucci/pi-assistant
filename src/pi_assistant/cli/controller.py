"""Audio controller loop and assistant interaction helpers for the CLI."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional

from pi_assistant.assistant import LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer
from pi_assistant.audio.resampler import LinearResampler
from pi_assistant.cli.controller_components import (
    AudioChunkPreparer,
    ResponseTaskManager,
    SilenceTracker,
    StreamStateManager,
)
from pi_assistant.cli.logging_utils import (
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    TURN_LOG_LABEL,
    WAKE_LOG_LABEL,
    log_state_transition,
    verbose_print,
)
from pi_assistant.config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    PREROLL_DURATION_SECONDS,
    SAMPLE_RATE,
    SERVER_STOP_MIN_SILENCE_SECONDS,
    STREAM_SAMPLE_RATE,
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
    PreRollBuffer,
    StreamState,
    WakeWordDetection,
    WakeWordEngine,
)

from .controller_helpers import (
    ResponseLifecycleHooks,
    schedule_turn_response,
    should_ignore_server_stop_event,
)


@dataclass(slots=True)
class AudioControllerContext:
    transcript_buffer: TurnTranscriptAggregator
    assistant: LLMResponder
    speech_player: SpeechPlayer
    stop_signal: asyncio.Event
    speech_stopped_signal: asyncio.Event


class _AudioControllerLoop:
    def __init__(
        self,
        audio_capture,
        ws_client: WebSocketClient,
        context: AudioControllerContext,
    ) -> None:
        self.audio_capture = audio_capture
        self.ws_client = ws_client
        self.context = context
        self.capture_sample_rate = SAMPLE_RATE
        self.bytes_per_frame = 2 * CHANNELS
        self.pre_roll = PreRollBuffer(
            PREROLL_DURATION_SECONDS,
            self.capture_sample_rate,
            channels=CHANNELS,
        )
        self.chunk_preparer = AudioChunkPreparer(
            self.capture_sample_rate,
            STREAM_SAMPLE_RATE,
            resampler_factory=LinearResampler,
        )
        self.silence_tracker = SilenceTracker(
            silence_threshold=AUTO_STOP_SILENCE_THRESHOLD,
            max_silence_seconds=AUTO_STOP_MAX_SILENCE_SECONDS,
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

    def _build_task_factory(self) -> Callable[[], asyncio.Task]:
        return partial(
            schedule_turn_response,
            self.context.transcript_buffer,
            self.context.assistant,
            self.context.speech_player,
            hooks=self.lifecycle_hooks,
        )

    def _create_wake_engine(self) -> Optional[WakeWordEngine]:
        try:
            engine = WakeWordEngine(
                WAKE_WORD_MODEL_PATH,
                fallback_model_path=WAKE_WORD_MODEL_FALLBACK_PATH,
                melspec_model_path=WAKE_WORD_MELSPEC_MODEL_PATH,
                embedding_model_path=WAKE_WORD_EMBEDDING_MODEL_PATH,
                source_sample_rate=self.capture_sample_rate,
                target_sample_rate=WAKE_WORD_TARGET_SAMPLE_RATE,
                threshold=WAKE_WORD_SCORE_THRESHOLD,
                consecutive_required=WAKE_WORD_CONSECUTIVE_FRAMES,
            )
            verbose_print(
                f"{WAKE_LOG_LABEL} Wake-word model: {WAKE_WORD_PHRASE} ({WAKE_WORD_NAME})"
            )
            return engine
        except RuntimeError as exc:
            print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
            raise

    async def run(self) -> None:
        self._log_startup()
        try:
            while True:
                audio_bytes = await self.audio_capture.get_audio_chunk()
                await self._process_chunk(audio_bytes)
        except asyncio.CancelledError:
            processed_chunks = self.state_manager.chunk_count
            verbose_print(f"[INFO] Audio controller stopped ({processed_chunks} chunks processed)")
            raise
        except Exception as exc:
            print(f"{ERROR_LOG_LABEL} Audio controller error: {exc}", file=sys.stderr)
            raise
        finally:
            await self.response_tasks.drain()

    def _log_startup(self) -> None:
        verbose_print("[INFO] Starting audio controller...")
        if self.chunk_preparer.is_resampling:
            verbose_print(
                f"{CONTROL_LOG_LABEL} Resampling capture audio "
                f"{self.capture_sample_rate} Hz -> {STREAM_SAMPLE_RATE} Hz for OpenAI."
            )
        else:
            verbose_print(f"{CONTROL_LOG_LABEL} Streaming audio at {STREAM_SAMPLE_RATE} Hz.")
        log_state_transition(None, self.state_manager.state, "awaiting wake phrase")

    async def _process_chunk(self, audio_bytes: bytes) -> None:
        if len(audio_bytes) % self.bytes_per_frame != 0:
            print(
                f"{ERROR_LOG_LABEL} Dropping malformed audio chunk: "
                f"{len(audio_bytes)} bytes (expected multiple of {self.bytes_per_frame}).",
                file=sys.stderr,
            )
            return
        chunk_index = self.state_manager.increment_chunk_count()
        silence_reached, observed_silence = self._observe_streaming_silence(audio_bytes)

        if self._handle_server_stop_event():
            return

        self._buffer_preroll_if_listening(audio_bytes)

        if await self._handle_stop_signal():
            return

        detection = self._run_wake_word_detection(audio_bytes)
        skip_chunk, skip_loop = await self._handle_wake_detection(detection, audio_bytes)
        if skip_loop:
            return

        now_streaming = self.state_manager.state == StreamState.STREAMING
        if now_streaming and not observed_silence:
            silence_reached, _ = self._observe_streaming_silence(audio_bytes)

        if now_streaming and not skip_chunk:
            await self._forward_audio(audio_bytes)

        if AUTO_STOP_ENABLED and now_streaming:
            self._handle_auto_stop(silence_reached)

        self._log_chunk_progress(chunk_index)

    def _observe_streaming_silence(self, audio_bytes: bytes) -> tuple[bool, bool]:
        if self.state_manager.state != StreamState.STREAMING:
            return False, False
        return self.silence_tracker.observe(audio_bytes), True

    def _handle_server_stop_event(self) -> bool:
        signal = self.context.speech_stopped_signal
        if not signal.is_set():
            return False
        if self.state_manager.consume_suppressed_stop_event():
            signal.clear()
            verbose_print(f"{TURN_LOG_LABEL} Ignoring stale server speech stop acknowledgement.")
            return True
        ignore_reason = should_ignore_server_stop_event(
            self.state_manager,
            self.silence_tracker,
            SERVER_STOP_MIN_SILENCE_SECONDS,
        )
        if ignore_reason:
            signal.clear()
            verbose_print(f"{TURN_LOG_LABEL} Server speech stop ignored: {ignore_reason}")
            return True
        signal.clear()
        if self.state_manager.awaiting_server_stop:
            reason = self.state_manager.complete_deferred_finalize("server speech stop event")
            if reason:
                self._finalize_turn(reason)
            return False
        previous_state = self.state_manager.transition_to_listening("server speech stop event")
        if previous_state:
            log_state_transition(
                previous_state, self.state_manager.state, "server speech stop event"
            )
            self._reset_stream_resources()
            self._finalize_turn("server speech stop event")
        return False

    async def _handle_stop_signal(self) -> bool:
        signal = self.context.stop_signal
        if not signal.is_set():
            return False
        signal.clear()
        if self.state_manager.state != StreamState.STREAMING:
            return True
        verbose_print(f"{CONTROL_LOG_LABEL} Stop command received; returning to listening.")
        previous_state = self.state_manager.transition_to_listening("stop command received")
        if previous_state:
            log_state_transition(previous_state, self.state_manager.state, "stop command received")
            self._reset_stream_resources()
        await self.context.transcript_buffer.clear_current_turn("manual stop command")
        self.response_tasks.cancel("manual stop command")
        return True

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
            verbose_print(score_log)
        return detection

    async def _handle_wake_detection(
        self,
        detection: WakeWordDetection,
        audio_bytes: bytes,
    ) -> tuple[bool, bool]:
        if not (self.wake_engine and detection.triggered):
            return False, False
        if self.state_manager.awaiting_server_stop:
            verbose_print(
                f"{WAKE_LOG_LABEL} Wake phrase ignored while awaiting server stop confirmation."
            )
            return False, True
        if self.state_manager.state != StreamState.LISTENING:
            retrigger_count = self.state_manager.increment_retrigger_budget()
            print(
                f"{WAKE_LOG_LABEL} Wake word retrigger detected during streaming "
                f"(count={retrigger_count})"
            )
            return False, False
        if self.state_manager.finalizing_turn:
            verbose_print(f"{WAKE_LOG_LABEL} Wake phrase ignored while finalizing previous turn.")
            return False, True
        if self.state_manager.awaiting_assistant_reply:
            verbose_print(f"{TURN_LOG_LABEL} Wake phrase overriding assistant reply.")
            self.state_manager.clear_awaiting_assistant_reply()
            self.response_tasks.cancel("wake phrase override")
        previous_state = self.state_manager.transition_to_streaming()
        await self.context.transcript_buffer.start_turn()
        self.wake_engine.reset_detection()
        if previous_state:
            log_state_transition(previous_state, self.state_manager.state, "wake phrase detected")
        self.silence_tracker.reset()
        payload = self.pre_roll.flush()
        if payload:
            await self._send_preroll_payload(payload)
            return True, False
        verbose_print(f"{WAKE_LOG_LABEL} Triggered -> streaming (no buffered audio)")
        return False, False

    async def _send_preroll_payload(self, payload: bytes) -> None:
        buffered_frames = len(payload) // self.bytes_per_frame
        duration_ms = buffered_frames / self.capture_sample_rate * 1000
        verbose_print(
            f"{WAKE_LOG_LABEL} Triggered -> streaming (sent {duration_ms:.0f} ms of buffered audio)"
        )
        resampled_payload = self.chunk_preparer.prepare(payload)
        if resampled_payload:
            await self.ws_client.send_audio_chunk(resampled_payload)

    async def _forward_audio(self, audio_bytes: bytes) -> None:
        stream_chunk = self.chunk_preparer.prepare(audio_bytes)
        if stream_chunk:
            await self.ws_client.send_audio_chunk(stream_chunk)

    def _handle_auto_stop(self, silence_reached: bool) -> None:
        if not silence_reached:
            return
        if self.state_manager.retrigger_budget == 0:
            if self.state_manager.awaiting_server_stop:
                verbose_print(
                    f"{TURN_LOG_LABEL} Silence timer fired but awaiting server stop; "
                    "skipping duplicate close request."
                )
                return
            previous_state = self.state_manager.transition_to_listening(
                "silence detected",
                defer_finalize=True,
            )
            if previous_state:
                log_state_transition(previous_state, self.state_manager.state, "silence detected")
                self._reset_stream_resources()
                verbose_print(
                    f"{TURN_LOG_LABEL} Awaiting server confirmation before finalizing turn."
                )
            return
        retrigger_count = self.state_manager.retrigger_budget
        retrigger_log = (
            f"{TURN_LOG_LABEL} Silence detected but "
            f"{retrigger_count} retrigger(s) observed; keeping stream open"
        )
        verbose_print(retrigger_log)
        self.state_manager.reset_retrigger_budget()
        self.silence_tracker.clear_silence()

    def _log_chunk_progress(self, chunk_index: int) -> None:
        if chunk_index % 100 != 0:
            return
        debug_log = (
            f"[DEBUG] Processed {chunk_index} audio chunks (state={self.state_manager.state.value})"
        )
        verbose_print(debug_log)

    def _reset_stream_resources(self) -> None:
        self.pre_roll.clear()
        self.silence_tracker.reset()
        self.chunk_preparer.reset()
        if self.wake_engine:
            self.wake_engine.reset_detection()

    def _finalize_turn(self, reason: str) -> None:
        verbose_print(f"{TURN_LOG_LABEL} Speech ended (reason={reason}) – finalizing turn.")
        self.state_manager.mark_finalizing_turn()
        self.response_tasks.schedule(reason)


async def run_audio_controller(
    audio_capture,
    ws_client: WebSocketClient,
    *,
    context: Optional[AudioControllerContext] = None,
    **legacy_kwargs,
) -> None:
    """Multiplex microphone audio between the wake-word detector and the OpenAI stream."""

    if context and legacy_kwargs:
        raise TypeError("Provide either `context` or individual controller dependencies, not both.")
    if context is None:
        try:
            context = AudioControllerContext(**legacy_kwargs)
        except TypeError as exc:  # pragma: no cover - invalid wiring
            raise TypeError(f"Invalid audio controller configuration: {exc}") from exc
    runner = _AudioControllerLoop(audio_capture, ws_client, context)
    await runner.run()
