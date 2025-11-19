"""Audio controller loop and assistant interaction helpers for the CLI."""

from __future__ import annotations

import asyncio
import sys
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


async def finalize_turn_and_respond(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Gather a completed turn transcript and fetch an assistant reply."""

    transcript = await transcript_buffer.finalize_turn()
    if not transcript:
        return

    _maybe_schedule_confirmation_cue(assistant, speech_player)

    print(f"{TURN_LOG_LABEL} Transcript ready ({len(transcript)} chars); requesting assistant...")
    verbose_print(f"{TURN_LOG_LABEL} Sending transcript to assistant: {transcript}")

    try:
        reply = await assistant.generate_reply(transcript)
    except Exception as exc:  # pragma: no cover - network failure
        print(f"{ASSISTANT_LOG_LABEL} Error requesting assistant reply: {exc}", file=sys.stderr)
        return

    if not reply:
        print(f"{ASSISTANT_LOG_LABEL} (empty response)")
        return

    if reply.text:
        print(f"{ASSISTANT_LOG_LABEL} {reply.text}")
    else:
        print(f"{ASSISTANT_LOG_LABEL} (no text content)")

    if reply.audio_bytes:
        try:
            await speech_player.play(reply.audio_bytes, sample_rate=reply.audio_sample_rate)
        except Exception as exc:  # pragma: no cover - host audio failure
            print(f"{ASSISTANT_LOG_LABEL} Error playing audio reply: {exc}", file=sys.stderr)


def schedule_turn_response(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> asyncio.Task:
    """Fire-and-forget helper for assistant calls with error reporting."""

    task = asyncio.create_task(
        finalize_turn_and_respond(transcript_buffer, assistant, speech_player)
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

    wake_engine: Optional[WakeWordEngine] = None
    try:
        wake_engine = WakeWordEngine(
            WAKE_WORD_MODEL_PATH,
            fallback_model_path=WAKE_WORD_MODEL_FALLBACK_PATH,
            melspec_model_path=WAKE_WORD_MELSPEC_MODEL_PATH,
            embedding_model_path=WAKE_WORD_EMBEDDING_MODEL_PATH,
            source_sample_rate=SAMPLE_RATE,
            target_sample_rate=WAKE_WORD_TARGET_SAMPLE_RATE,
            threshold=WAKE_WORD_SCORE_THRESHOLD,
            consecutive_required=WAKE_WORD_CONSECUTIVE_FRAMES,
        )
        verbose_print(f"{WAKE_LOG_LABEL} Wake-word model: {WAKE_WORD_PHRASE} ({WAKE_WORD_NAME})")
    except RuntimeError as exc:
        print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
        raise
    pre_roll = PreRollBuffer(PREROLL_DURATION_SECONDS, SAMPLE_RATE)
    chunk_preparer = AudioChunkPreparer(
        SAMPLE_RATE,
        STREAM_SAMPLE_RATE,
        resampler_factory=LinearResampler,
    )
    silence_tracker = SilenceTracker(
        silence_threshold=AUTO_STOP_SILENCE_THRESHOLD,
        max_silence_seconds=AUTO_STOP_MAX_SILENCE_SECONDS,
        sample_rate=SAMPLE_RATE,
    )
    state_manager = StreamStateManager()
    task_factory = partial(schedule_turn_response, transcript_buffer, assistant, speech_player)
    response_tasks = ResponseTaskManager(task_factory=task_factory)

    if chunk_preparer.is_resampling:
        verbose_print(
            f"{CONTROL_LOG_LABEL} Resampling capture audio "
            f"{SAMPLE_RATE} Hz -> {STREAM_SAMPLE_RATE} Hz for OpenAI."
        )
    else:
        verbose_print(f"{CONTROL_LOG_LABEL} Streaming audio at {STREAM_SAMPLE_RATE} Hz.")

    log_state_transition(None, state_manager.state, "awaiting wake phrase")

    def finalize_turn(reason: str) -> None:
        print(f"{TURN_LOG_LABEL} Speech ended (reason={reason}) – finalizing turn.")
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

            if speech_stopped_signal.is_set():
                speech_stopped_signal.clear()
                if state_manager.consume_suppressed_stop_event():
                    verbose_print(
                        f"{TURN_LOG_LABEL} Ignoring stale server speech stop acknowledgement."
                    )
                elif state_manager.awaiting_server_stop:
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
                        f"{WAKE_LOG_LABEL} Wake phrase overriding prior turn awaiting server stop."
                    )
                    reason = state_manager.complete_deferred_finalize("wake phrase override")
                    if reason:
                        finalize_turn(reason)
                    state_manager.suppress_next_server_stop_event()
                if state_manager.state == StreamState.LISTENING:
                    previous_state = state_manager.transition_to_streaming()
                    response_tasks.cancel("wake phrase override")
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
                        duration_ms = (len(payload) / (2 * CHANNELS)) / SAMPLE_RATE * 1000
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

            if state_manager.state == StreamState.STREAMING and not skip_current_chunk:
                stream_chunk = chunk_preparer.prepare(audio_bytes)
                if stream_chunk:
                    await ws_client.send_audio_chunk(stream_chunk)

            if AUTO_STOP_ENABLED and state_manager.state == StreamState.STREAMING:
                if silence_tracker.observe(audio_bytes):
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
