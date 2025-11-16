"""Audio controller loop and assistant interaction helpers for the CLI."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional, Set

import numpy as np

from pi_transcription.assistant import LLMResponder, TurnTranscriptAggregator
from pi_transcription.audio import SpeechPlayer
from pi_transcription.cli.logging_utils import (
    ASSISTANT_LOG_LABEL,
    CONTROL_LOG_LABEL,
    ERROR_LOG_LABEL,
    TURN_LOG_LABEL,
    WAKE_LOG_LABEL,
    log_state_transition,
)
from pi_transcription.config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    PREROLL_DURATION_SECONDS,
    SAMPLE_RATE,
    WAKE_WORD_CONSECUTIVE_FRAMES,
    WAKE_WORD_EMBEDDING_MODEL_PATH,
    WAKE_WORD_MELSPEC_MODEL_PATH,
    WAKE_WORD_MODEL_FALLBACK_PATH,
    WAKE_WORD_MODEL_PATH,
    WAKE_WORD_SCORE_THRESHOLD,
    WAKE_WORD_TARGET_SAMPLE_RATE,
)
from pi_transcription.network import WebSocketClient
from pi_transcription.wake_word import (
    PreRollBuffer,
    StreamState,
    WakeWordDetection,
    WakeWordEngine,
)


def calculate_rms(audio_bytes: bytes) -> float:
    """Compute the root-mean-square amplitude for a PCM16 chunk."""

    if not audio_bytes:
        return 0.0

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0

    float_samples = samples.astype(np.float32)
    return float(np.sqrt(np.mean(float_samples**2)))


async def finalize_turn_and_respond(
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
) -> None:
    """Gather a completed turn transcript and fetch an assistant reply."""

    transcript = await transcript_buffer.finalize_turn()
    if not transcript:
        return

    print(f"{TURN_LOG_LABEL} Sending transcript to assistant: {transcript}")

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
        except Exception as exc:  # pragma: no cover - unexpected
            print(f"{ASSISTANT_LOG_LABEL} Unexpected assistant error: {exc}", file=sys.stderr)

    task.add_done_callback(_log_task_error)
    return task


async def run_audio_controller(
    audio_capture,
    ws_client: WebSocketClient,
    *,
    force_always_on: bool,
    transcript_buffer: TurnTranscriptAggregator,
    assistant: LLMResponder,
    speech_player: SpeechPlayer,
    stop_signal: asyncio.Event,
) -> None:
    """Multiplex microphone audio between the wake-word detector and the OpenAI stream."""

    print("[INFO] Starting audio controller...")

    wake_engine: Optional[WakeWordEngine] = None
    if not force_always_on:
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
        except RuntimeError as exc:
            print(f"{ERROR_LOG_LABEL} {exc}", file=sys.stderr)
            raise

    pre_roll = PreRollBuffer(PREROLL_DURATION_SECONDS, SAMPLE_RATE)
    state = StreamState.STREAMING if force_always_on else StreamState.LISTENING
    initial_reason = "wake-word override" if force_always_on else "awaiting wake phrase"
    log_state_transition(None, state, initial_reason)
    if force_always_on:
        print(f"{WAKE_LOG_LABEL} Wake-word override active; streaming immediately")

    chunk_count = 0
    silence_duration = 0.0
    heard_speech = False
    retrigger_budget = 0
    response_tasks: Set[asyncio.Task] = set()

    if state == StreamState.STREAMING:
        await transcript_buffer.start_turn()

    try:
        while True:
            audio_bytes = await audio_capture.get_audio_chunk()
            chunk_count += 1

            if state == StreamState.LISTENING:
                pre_roll.add(audio_bytes)

            if stop_signal.is_set():
                stop_signal.clear()
                if state == StreamState.STREAMING:
                    print(f"{CONTROL_LOG_LABEL} Stop command received; returning to listening.")
                    previous_state = state
                    state = StreamState.LISTENING
                    log_state_transition(previous_state, state, "stop command received")
                    pre_roll.clear()
                    heard_speech = False
                    silence_duration = 0.0
                    retrigger_budget = 0
                    if wake_engine:
                        wake_engine.reset_detection()
                    await transcript_buffer.clear_current_turn()
                continue

            detection = WakeWordDetection()
            if wake_engine:
                detection = wake_engine.process_chunk(audio_bytes)
                if detection.score >= WAKE_WORD_SCORE_THRESHOLD:
                    print(
                        f"{WAKE_LOG_LABEL} score={detection.score:.2f} "
                        f"(threshold {WAKE_WORD_SCORE_THRESHOLD:.2f}) state={state.value}"
                    )

            skip_current_chunk = False
            if wake_engine and detection.triggered:
                if state == StreamState.LISTENING:
                    previous_state = state
                    state = StreamState.STREAMING
                    await transcript_buffer.start_turn()
                    log_state_transition(previous_state, state, "wake phrase detected")
                    payload = pre_roll.flush()
                    if payload:
                        duration_ms = (len(payload) / (2 * CHANNELS)) / SAMPLE_RATE * 1000
                        print(
                            f"{WAKE_LOG_LABEL} Triggered -> streaming "
                            f"(sent {duration_ms:.0f} ms of buffered audio)"
                        )
                        await ws_client.send_audio_chunk(payload)
                        skip_current_chunk = True
                    else:
                        skip_current_chunk = False

                    heard_speech = False
                    silence_duration = 0.0
                    retrigger_budget = 0
                else:
                    retrigger_budget += 1
                    print(
                        f"{WAKE_LOG_LABEL} Wake word retrigger detected during streaming "
                        f"(count={retrigger_budget})"
                    )

            if state == StreamState.STREAMING and not skip_current_chunk:
                await ws_client.send_audio_chunk(audio_bytes)

            if AUTO_STOP_ENABLED and state == StreamState.STREAMING and not force_always_on:
                frames = len(audio_bytes) / (2 * CHANNELS)
                chunk_duration = frames / SAMPLE_RATE if frames else 0.0
                rms = calculate_rms(audio_bytes)

                if rms >= AUTO_STOP_SILENCE_THRESHOLD:
                    heard_speech = True
                    silence_duration = 0.0
                elif heard_speech:
                    silence_duration += chunk_duration
                    if silence_duration >= AUTO_STOP_MAX_SILENCE_SECONDS:
                        if retrigger_budget == 0:
                            previous_state = state
                            state = StreamState.LISTENING
                            log_state_transition(previous_state, state, "silence detected")
                            pre_roll.clear()
                            heard_speech = False
                            silence_duration = 0.0
                            retrigger_budget = 0
                            if wake_engine:
                                wake_engine.reset_detection()
                            task = schedule_turn_response(
                                transcript_buffer, assistant, speech_player
                            )
                            response_tasks.add(task)
                            task.add_done_callback(lambda fut, s=response_tasks: s.discard(fut))
                        else:
                            print(
                                f"{TURN_LOG_LABEL} Silence detected but "
                                f"{retrigger_budget} retrigger(s) observed; keeping stream open"
                            )
                            retrigger_budget = 0
                            silence_duration = 0.0

            if chunk_count % 100 == 0:
                print(f"[DEBUG] Processed {chunk_count} audio chunks (state={state.value})")

    except asyncio.CancelledError:
        print(f"[INFO] Audio controller stopped ({chunk_count} chunks processed)")
        raise
    except Exception as exc:
        print(f"{ERROR_LOG_LABEL} Audio controller error: {exc}", file=sys.stderr)
        raise
    finally:
        if response_tasks:
            for task in response_tasks:
                task.cancel()
            await asyncio.gather(*response_tasks, return_exceptions=True)
