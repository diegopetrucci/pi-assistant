"""Audio controller loop and assistant interaction helpers for the CLI."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Optional

from pi_assistant.assistant import LLMReply, LLMResponder, TurnTranscriptAggregator
from pi_assistant.audio import SpeechPlayer
from pi_assistant.audio.resampler import LinearResampler
from pi_assistant.cli.controller_components import (
    AudioChunkPreparer,
    ResponseTaskManager,
    SilenceTracker,
    StreamStateManager,
)
from pi_assistant.cli.logging_utils import (
    ASSISTANT_LOG_LABEL,
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    TURN_LOG_LABEL,
    WAKE_LOG_LABEL,
    log_state_transition,
    verbose_print,
)
from pi_assistant.config import (
    ASSISTANT_TTS_SAMPLE_RATE,
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    CONFIRMATION_CUE_ENABLED,
    CONFIRMATION_CUE_TEXT,
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


def should_ignore_server_stop_event(
    state_manager: StreamStateManager,
    silence_tracker: SilenceTracker,
    min_silence_seconds: float,
) -> str | None:
    """Return a reason to ignore a server VAD stop, or ``None`` if it may proceed."""

    if state_manager.state != StreamState.STREAMING:
        # Defensive: the server stop acknowledgement may arrive after we've already
        # returned to listening, in which case we can safely treat it as handled.
        return None
    if not silence_tracker.heard_speech:
        return None
    if not silence_tracker.has_observed_silence(min_silence_seconds):
        current = silence_tracker.silence_duration
        return f"{current:.2f}s silence < {min_silence_seconds:.2f}s minimum"
    return None


def _maybe_schedule_confirmation_cue(
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Play a cached 'Got it' cue without blocking the main turn flow."""
    if not (assistant.tts_enabled and CONFIRMATION_CUE_ENABLED and CONFIRMATION_CUE_TEXT):
        return

    cached = assistant.peek_phrase_audio(CONFIRMATION_CUE_TEXT)
    if not cached:
        asyncio.create_task(assistant.warm_phrase_audio(CONFIRMATION_CUE_TEXT))
        return
    audio_bytes, sample_rate = cached
    task = asyncio.create_task(
        speech_player.play(audio_bytes, sample_rate=sample_rate or ASSISTANT_TTS_SAMPLE_RATE)
    )

    def _log_task_error(fut: asyncio.Task):
        try:
            fut.result()
        except Exception as exc:  # pragma: no cover - host audio failure
            verbose_print(f"{CONTROL_LOG_LABEL} Confirmation cue failed: {exc}")

    task.add_done_callback(_log_task_error)


@dataclass(slots=True)
class ResponseLifecycleHooks:
    on_transcript_ready: Optional[Callable[[], None]] = None
    on_reply_start: Optional[Callable[[], None]] = None
    on_reply_complete: Optional[Callable[[], None]] = None


async def _finalize_transcript(
    transcript_buffer: TurnTranscriptAggregator,
    hooks: Optional[ResponseLifecycleHooks],
) -> Optional[str]:
    """Return the finalized transcript while honoring lifecycle hooks."""
    try:
        return await transcript_buffer.finalize_turn()
    finally:
        if hooks and hooks.on_transcript_ready:
            hooks.on_transcript_ready()


async def _request_assistant_reply(
    transcript: str,
    assistant: LLMResponder,
    hooks: Optional[ResponseLifecycleHooks],
) -> Optional[LLMReply]:
    """Fetch an assistant reply and handle lifecycle notifications."""

    reply_started = False
    if hooks and hooks.on_reply_start:
        hooks.on_reply_start()
        reply_started = True

    reply: Optional[LLMReply] = None
    try:
        reply = await assistant.generate_reply(transcript)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        print(f"{ASSISTANT_LOG_LABEL} Error requesting assistant reply: {exc}", file=sys.stderr)
        return None
    finally:
        if reply_started and hooks and hooks.on_reply_complete:
            hooks.on_reply_complete()

    return reply


async def _play_assistant_audio(reply: LLMReply, speech_player: SpeechPlayer) -> None:
    if not reply.audio_bytes:
        return
    try:
        await speech_player.play(reply.audio_bytes, sample_rate=reply.audio_sample_rate)
    except Exception as exc:  # pragma: no cover - host audio failure
        print(f"{ASSISTANT_LOG_LABEL} Error playing audio reply: {exc}", file=sys.stderr)


async def finalize_turn_and_respond(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    *,
    hooks: Optional[ResponseLifecycleHooks] = None,
) -> None:
    """Gather a completed turn transcript and fetch an assistant reply."""

    transcript = await _finalize_transcript(transcript_buffer, hooks)
    if not transcript:
        return

    _maybe_schedule_confirmation_cue(assistant, speech_player)

    print(f"{TURN_LOG_LABEL} Transcript ready ({len(transcript)} chars); requesting assistant...")
    verbose_print(f"{TURN_LOG_LABEL} Sending transcript to assistant: {transcript}")

    reply = await _request_assistant_reply(transcript, assistant, hooks)
    if not reply:
        print(f"{ASSISTANT_LOG_LABEL} (empty response)")
        return

    if reply.text:
        print(f"{ASSISTANT_LOG_LABEL} {reply.text}")
    else:
        print(f"{ASSISTANT_LOG_LABEL} (no text content)")

    await _play_assistant_audio(reply, speech_player)


def schedule_turn_response(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    *,
    hooks: Optional[ResponseLifecycleHooks] = None,
) -> asyncio.Task:
    """Fire-and-forget helper for assistant calls with error reporting."""

    task = asyncio.create_task(
        finalize_turn_and_respond(
            transcript_buffer,
            assistant,
            speech_player,
            hooks=hooks,
        )
    )

    def _log_task_error(fut: asyncio.Task):
        try:
            fut.result()
        except asyncio.CancelledError:
            verbose_print(f"{TURN_LOG_LABEL} Assistant reply task cancelled.")
        except Exception as exc:  # pragma: no cover - unexpected
            print(f"{ASSISTANT_LOG_LABEL} Unexpected assistant error: {exc}", file=sys.stderr)

    task.add_done_callback(_log_task_error)
    return task


async def run_audio_controller(  # noqa: PLR0913, PLR0912, PLR0915
    audio_capture,
    ws_client: WebSocketClient,
    *,
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    stop_signal: asyncio.Event,
    speech_stopped_signal: asyncio.Event,
) -> None:
    """Multiplex microphone audio between the wake-word detector and the OpenAI stream."""

    verbose_print("[INFO] Starting audio controller...")
    capture_sample_rate = SAMPLE_RATE
    bytes_per_frame = 2 * CHANNELS

    wake_engine: Optional[WakeWordEngine] = None
    try:
        wake_engine = WakeWordEngine(
            WAKE_WORD_MODEL_PATH,
            fallback_model_path=WAKE_WORD_MODEL_FALLBACK_PATH,
            melspec_model_path=WAKE_WORD_MELSPEC_MODEL_PATH,
            embedding_model_path=WAKE_WORD_EMBEDDING_MODEL_PATH,
            source_sample_rate=capture_sample_rate,
            target_sample_rate=WAKE_WORD_TARGET_SAMPLE_RATE,
            threshold=WAKE_WORD_SCORE_THRESHOLD,
            consecutive_required=WAKE_WORD_CONSECUTIVE_FRAMES,
        )
        verbose_print(f"{WAKE_LOG_LABEL} Wake-word model: {WAKE_WORD_PHRASE} ({WAKE_WORD_NAME})")
    except RuntimeError as exc:
        print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
        raise
    pre_roll = PreRollBuffer(
        PREROLL_DURATION_SECONDS,
        capture_sample_rate,
        channels=CHANNELS,
    )
    chunk_preparer = AudioChunkPreparer(
        capture_sample_rate,
        STREAM_SAMPLE_RATE,
        resampler_factory=LinearResampler,
    )
    silence_tracker = SilenceTracker(
        silence_threshold=AUTO_STOP_SILENCE_THRESHOLD,
        max_silence_seconds=AUTO_STOP_MAX_SILENCE_SECONDS,
        sample_rate=capture_sample_rate,
    )
    state_manager = StreamStateManager()
    lifecycle_hooks = ResponseLifecycleHooks(
        on_transcript_ready=state_manager.clear_finalizing_turn,
        on_reply_start=state_manager.mark_awaiting_assistant_reply,
        on_reply_complete=state_manager.clear_awaiting_assistant_reply,
    )

    task_factory = partial(
        schedule_turn_response,
        transcript_buffer,
        assistant,
        speech_player,
        hooks=lifecycle_hooks,
    )
    response_tasks = ResponseTaskManager(task_factory=task_factory)

    if chunk_preparer.is_resampling:
        verbose_print(
            f"{CONTROL_LOG_LABEL} Resampling capture audio "
            f"{capture_sample_rate} Hz -> {STREAM_SAMPLE_RATE} Hz for OpenAI."
        )
    else:
        verbose_print(f"{CONTROL_LOG_LABEL} Streaming audio at {STREAM_SAMPLE_RATE} Hz.")

    log_state_transition(None, state_manager.state, "awaiting wake phrase")

    def finalize_turn(reason: str) -> None:
        verbose_print(f"{TURN_LOG_LABEL} Speech ended (reason={reason}) – finalizing turn.")
        state_manager.mark_finalizing_turn()
        response_tasks.schedule(reason)

    def reset_stream_resources() -> None:
        pre_roll.clear()
        silence_tracker.reset()
        chunk_preparer.reset()
        if wake_engine:
            wake_engine.reset_detection()

    try:
        while True:
            audio_bytes = await audio_capture.get_audio_chunk()
            chunk_index = state_manager.increment_chunk_count()

            if len(audio_bytes) % bytes_per_frame != 0:
                print(
                    f"{ERROR_LOG_LABEL} Dropping malformed audio chunk: "
                    f"{len(audio_bytes)} bytes (expected multiple of {bytes_per_frame}).",
                    file=sys.stderr,
                )
                continue

            was_streaming = state_manager.state == StreamState.STREAMING
            silence_reached = False
            observed_silence_for_chunk = False
            if was_streaming:
                # Track silence for every streaming chunk, even if we drop it later,
                # so the auto-stop timers stay accurate.
                silence_reached = silence_tracker.observe(audio_bytes)
                observed_silence_for_chunk = True

            if speech_stopped_signal.is_set():
                if state_manager.consume_suppressed_stop_event():
                    speech_stopped_signal.clear()
                    verbose_print(
                        f"{TURN_LOG_LABEL} Ignoring stale server speech stop acknowledgement."
                    )
                    continue

                ignore_reason = should_ignore_server_stop_event(
                    state_manager,
                    silence_tracker,
                    SERVER_STOP_MIN_SILENCE_SECONDS,
                )
                if ignore_reason:
                    speech_stopped_signal.clear()
                    verbose_print(f"{TURN_LOG_LABEL} Server speech stop ignored: {ignore_reason}")
                    continue

                speech_stopped_signal.clear()
                if state_manager.awaiting_server_stop:
                    reason = state_manager.complete_deferred_finalize("server speech stop event")
                    if reason:
                        finalize_turn(reason)
                else:
                    previous_state = state_manager.transition_to_listening(
                        "server speech stop event"
                    )
                    if previous_state:
                        log_state_transition(
                            previous_state, state_manager.state, "server speech stop event"
                        )
                        reset_stream_resources()
                        finalize_turn("server speech stop event")

            if state_manager.state == StreamState.LISTENING:
                pre_roll.add(audio_bytes)

            if stop_signal.is_set():
                stop_signal.clear()
                if state_manager.state == StreamState.STREAMING:
                    verbose_print(
                        f"{CONTROL_LOG_LABEL} Stop command received; returning to listening."
                    )
                    previous_state = state_manager.transition_to_listening("stop command received")
                    if previous_state:
                        log_state_transition(
                            previous_state, state_manager.state, "stop command received"
                        )
                        reset_stream_resources()
                    await transcript_buffer.clear_current_turn("manual stop command")
                    response_tasks.cancel("manual stop command")
                continue

            detection = WakeWordDetection()
            if wake_engine:
                detection = wake_engine.process_chunk(audio_bytes)
                if detection.score >= WAKE_WORD_SCORE_THRESHOLD:
                    score_log = (
                        f"{WAKE_LOG_LABEL} score={detection.score:.2f} "
                        f"(threshold {WAKE_WORD_SCORE_THRESHOLD:.2f}) "
                        f"state={state_manager.state.value}"
                    )
                    verbose_print(score_log)

            skip_current_chunk = False
            if wake_engine and detection.triggered:
                if state_manager.awaiting_server_stop:
                    verbose_print(
                        f"{WAKE_LOG_LABEL} Wake phrase ignored while awaiting server stop "
                        "confirmation."
                    )
                    continue
                if state_manager.state == StreamState.LISTENING:
                    if state_manager.finalizing_turn:
                        verbose_print(
                            f"{WAKE_LOG_LABEL} Wake phrase ignored while finalizing previous turn."
                        )
                        continue
                    if state_manager.awaiting_assistant_reply:
                        verbose_print(f"{TURN_LOG_LABEL} Wake phrase overriding assistant reply.")
                        state_manager.clear_awaiting_assistant_reply()
                        response_tasks.cancel("wake phrase override")
                    previous_state = state_manager.transition_to_streaming()
                    await transcript_buffer.start_turn()
                    if wake_engine:
                        wake_engine.reset_detection()
                    if previous_state:
                        log_state_transition(
                            previous_state, state_manager.state, "wake phrase detected"
                        )
                    silence_tracker.reset()
                    payload = pre_roll.flush()
                    if payload:
                        buffered_frames = len(payload) // bytes_per_frame
                        duration_ms = buffered_frames / capture_sample_rate * 1000
                        verbose_print(
                            f"{WAKE_LOG_LABEL} Triggered -> streaming "
                            f"(sent {duration_ms:.0f} ms of buffered audio)"
                        )
                        resampled_payload = chunk_preparer.prepare(payload)
                        if resampled_payload:
                            await ws_client.send_audio_chunk(resampled_payload)
                        skip_current_chunk = True
                    else:
                        skip_current_chunk = False
                else:
                    retrigger_count = state_manager.increment_retrigger_budget()
                    print(
                        f"{WAKE_LOG_LABEL} Wake word retrigger detected during streaming "
                        f"(count={retrigger_count})"
                    )

            now_streaming = state_manager.state == StreamState.STREAMING
            if now_streaming and not observed_silence_for_chunk:
                # Streaming just started mid-iteration; register this chunk for silence tracking.
                silence_reached = silence_tracker.observe(audio_bytes)
                observed_silence_for_chunk = True

            if now_streaming and not skip_current_chunk:
                stream_chunk = chunk_preparer.prepare(audio_bytes)
                if stream_chunk:
                    await ws_client.send_audio_chunk(stream_chunk)

            if AUTO_STOP_ENABLED and state_manager.state == StreamState.STREAMING:
                if silence_reached:
                    if state_manager.retrigger_budget == 0:
                        if state_manager.awaiting_server_stop:
                            verbose_print(
                                f"{TURN_LOG_LABEL} Silence timer fired but awaiting server "
                                "stop; skipping duplicate close request."
                            )
                        else:
                            previous_state = state_manager.transition_to_listening(
                                "silence detected", defer_finalize=True
                            )
                            if previous_state:
                                log_state_transition(
                                    previous_state, state_manager.state, "silence detected"
                                )
                                reset_stream_resources()
                                verbose_print(
                                    f"{TURN_LOG_LABEL} Awaiting server confirmation before "
                                    "finalizing turn."
                                )
                    else:
                        retrigger_count = state_manager.retrigger_budget
                        retrigger_log = (
                            f"{TURN_LOG_LABEL} Silence detected but "
                            f"{retrigger_count} retrigger(s) observed; "
                            "keeping stream open"
                        )
                        verbose_print(retrigger_log)
                        state_manager.reset_retrigger_budget()
                        silence_tracker.clear_silence()

            if chunk_index % 100 == 0:
                debug_log = (
                    f"[DEBUG] Processed {chunk_index} audio chunks "
                    f"(state={state_manager.state.value})"
                )
                verbose_print(debug_log)

    except asyncio.CancelledError:
        verbose_print(
            f"[INFO] Audio controller stopped ({state_manager.chunk_count} chunks processed)"
        )
        raise
    except Exception as exc:
        print(f"{ERROR_LOG_LABEL} Audio controller error: {exc}", file=sys.stderr)
        raise
    finally:
        await response_tasks.drain()
