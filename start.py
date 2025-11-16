"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import argparse
import asyncio
import sys
from functools import partial
from typing import Optional

import numpy as np

from audio_capture import AudioCapture
from config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    FORCE_ALWAYS_ON,
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
from transcription_tests import test_audio_capture, test_websocket_client
from wake_word import PreRollBuffer, StreamState, WakeWordDetection, WakeWordEngine
from websocket_client import WebSocketClient

# ANSI color codes for log labels
RESET = "\033[0m"
COLOR_ORANGE = "\033[38;5;208m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"

TURN_LOG_LABEL = f"{COLOR_ORANGE}[TURN]{RESET}"
TRANSCRIPT_LOG_LABEL = f"{COLOR_GREEN}[TRANSCRIPT]{RESET}"
VAD_LOG_LABEL = f"{COLOR_YELLOW}[VAD]{RESET}"
STATE_LOG_LABEL = f"{COLOR_CYAN}[STATE]{RESET}"
WAKE_LOG_LABEL = f"{COLOR_BLUE}[WAKE]{RESET}"


def log_state_transition(previous: Optional[StreamState], new: StreamState, reason: str) -> None:
    """Emit a consistent log for controller state changes."""

    if previous == new:
        return

    if previous is None:
        print(f"{STATE_LOG_LABEL} Entered {new.value.upper()} ({reason})")
    else:
        print(f"{STATE_LOG_LABEL} {previous.value.upper()} -> {new.value.upper()} ({reason})")


def calculate_rms(audio_bytes):
    """Compute the root-mean-square amplitude for a PCM16 chunk.

    The incoming `audio_bytes` represent signed 16-bit mono samples, so the
    resulting RMS value is in the 0-32767 range and offers a simple proxy for
    perceived loudness. We convert to float before squaring to avoid overflow
    and return 0.0 when the buffer is empty.
    """

    if not audio_bytes:
        return 0.0

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    if samples.size == 0:
        return 0.0

    # Use float32 to avoid overflow while squaring
    float_samples = samples.astype(np.float32)
    return float(np.sqrt(np.mean(float_samples**2)))


def handle_transcription_event(event):
    """
    Handle transcription events from OpenAI

    Args:
        event: Parsed JSON event from server
    """
    event_type = event.get("type")

    if event_type == "conversation.item.input_audio_transcription.delta":
        # Partial transcription
        delta = event.get("delta", "")
        print(f"[PARTIAL] {delta}", end="", flush=True)

    elif event_type == "conversation.item.input_audio_transcription.completed":
        # Final transcription
        transcript = event.get("transcript", "")
        print(f"\n{TRANSCRIPT_LOG_LABEL} {transcript}")

    elif event_type == "input_audio_buffer.committed":
        # VAD detected speech
        item_id = event.get("item_id", "")
        print(f"{VAD_LOG_LABEL} Speech detected (item: {item_id})")

    elif event_type == "error":
        # Error from API
        error = event.get("error", {})
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "No message")
        error_code = error.get("code", "unknown")
        print(f"[ERROR] {error_type} ({error_code}): {error_message}", file=sys.stderr)

    elif event_type == "transcription_session.created":
        # Initial transcription session created
        print("[INFO] Transcription session created")

    elif event_type == "transcription_session.updated":
        # Session configuration acknowledged
        print("[INFO] Transcription session configuration updated")

    else:
        # Other events (for debugging)
        print(f"[DEBUG] Received event: {event_type}")


async def run_audio_controller(audio_capture, ws_client, *, force_always_on: bool):
    """
    Multiplex audio between the wake-word detector and the OpenAI stream.
    """

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
            print(f"[ERROR] {exc}", file=sys.stderr)
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

    try:
        while True:
            audio_bytes = await audio_capture.get_audio_chunk()
            chunk_count += 1

            if state == StreamState.LISTENING:
                pre_roll.add(audio_bytes)

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
    except Exception as e:
        print(f"[ERROR] Audio controller error: {e}", file=sys.stderr)
        raise


async def receive_transcription_events(ws_client):
    """
    Continuously receive and handle transcription events from WebSocket

    Args:
        ws_client: WebSocketClient instance
    """
    print("[INFO] Starting event receiver...")
    event_count = 0

    try:
        async for event in ws_client.receive_events():
            event_count += 1
            handle_transcription_event(event)

    except asyncio.CancelledError:
        print(f"[INFO] Event receiver stopped ({event_count} events received)")
        raise
    except Exception as e:
        print(f"[ERROR] Event receiver error: {e}", file=sys.stderr)
        raise


async def run_transcription(force_always_on: bool = False):
    """
    Main integration function - runs real-time transcription
    Combines audio capture and WebSocket streaming
    """
    print("\n=== Starting Real-Time Transcription ===\n")

    # Create instances
    audio_capture = AudioCapture()
    ws_client = WebSocketClient()

    # Get current event loop
    loop = asyncio.get_running_loop()

    try:
        # Connect to OpenAI via WebSocket
        await ws_client.connect()

        # Start audio stream
        audio_capture.start_stream(loop)

        print("\n✓ System ready")
        print("Listening... (Press Ctrl+C to stop)\n")

        # Create concurrent tasks for audio streaming and event receiving
        audio_task = asyncio.create_task(
            run_audio_controller(audio_capture, ws_client, force_always_on=force_always_on)
        )
        event_task = asyncio.create_task(receive_transcription_events(ws_client))

        # Wait for both tasks (they run until cancelled)
        await asyncio.gather(audio_task, event_task)

    except KeyboardInterrupt:
        print("\n\nShutdown requested...")

    except Exception as e:
        print(f"\n[ERROR] Transcription failed: {e}", file=sys.stderr)
        raise

    finally:
        # Graceful shutdown
        print("Cleaning up...")

        # Stop audio stream
        audio_capture.stop_stream()

        # Close WebSocket connection
        await ws_client.close()

        print("✓ Shutdown complete\n")


def parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="OpenAI transcription client with wake-word gating."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["run", "test-audio", "test-websocket"],
        default="run",
        help="Select an execution mode (default: run)",
    )
    parser.set_defaults(force_always_on=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--force-always-on",
        action="store_true",
        dest="force_always_on",
        help="Bypass wake-word gating and stream audio continuously.",
    )
    group.add_argument(
        "--no-force-always-on",
        action="store_false",
        dest="force_always_on",
        help="Explicitly disable the wake-word override even if the env var is set.",
    )
    return parser.parse_args()


def main():
    """Main entry point"""

    args = parse_args()
    if args.mode == "test-audio":
        run_func = test_audio_capture
    elif args.mode == "test-websocket":
        run_func = partial(test_websocket_client, handle_transcription_event)
    else:
        force_flag = FORCE_ALWAYS_ON if args.force_always_on is None else args.force_always_on
        run_func = partial(run_transcription, force_always_on=force_flag)

    try:
        asyncio.run(run_func())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
