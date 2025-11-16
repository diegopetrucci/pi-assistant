"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import asyncio
import sys
from functools import partial

import numpy as np

from audio_capture import AudioCapture
from config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    CHANNELS,
    SAMPLE_RATE,
)
from transcription_tests import test_audio_capture, test_websocket_client
from websocket_client import WebSocketClient

# ANSI color codes for log labels
RESET = "\033[0m"
COLOR_ORANGE = "\033[38;5;208m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"

TURN_LOG_LABEL = f"{COLOR_ORANGE}[TURN]{RESET}"
TRANSCRIPT_LOG_LABEL = f"{COLOR_GREEN}[TRANSCRIPT]{RESET}"
VAD_LOG_LABEL = f"{COLOR_YELLOW}[VAD]{RESET}"


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


async def stream_audio_to_websocket(audio_capture, ws_client):
    """
    Continuously read audio chunks from capture queue and stream to WebSocket

    Args:
        audio_capture: AudioCapture instance
        ws_client: WebSocketClient instance
    """
    print("[INFO] Starting audio streaming...")
    chunk_count = 0
    silence_duration = 0.0
    heard_speech = False

    try:
        while True:
            # Get audio chunk from queue
            audio_bytes = await audio_capture.get_audio_chunk()

            # Send to OpenAI via WebSocket
            await ws_client.send_audio_chunk(audio_bytes)

            chunk_count += 1

            if AUTO_STOP_ENABLED:
                # Derive chunk duration from frames (2 bytes per int16 sample)
                frames = len(audio_bytes) / (2 * CHANNELS)
                chunk_duration = frames / SAMPLE_RATE if frames else 0.0
                rms = calculate_rms(audio_bytes)

                if rms >= AUTO_STOP_SILENCE_THRESHOLD:
                    heard_speech = True
                    silence_duration = 0.0
                elif heard_speech:
                    silence_duration += chunk_duration
                    if silence_duration >= AUTO_STOP_MAX_SILENCE_SECONDS:
                        print(
                            f"{TURN_LOG_LABEL} "
                            f"Stopped after {AUTO_STOP_MAX_SILENCE_SECONDS:.1f}s of silence"
                        )
                        heard_speech = False
                        silence_duration = 0.0

            # Debug: Log every 100 chunks (~4 seconds at 24kHz with 1024 buffer)
            if chunk_count % 100 == 0:
                print(f"[DEBUG] Streamed {chunk_count} audio chunks")

    except asyncio.CancelledError:
        print(f"[INFO] Audio streaming stopped ({chunk_count} chunks sent)")
        raise
    except Exception as e:
        print(f"[ERROR] Audio streaming error: {e}", file=sys.stderr)
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


async def run_transcription():
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
        audio_task = asyncio.create_task(stream_audio_to_websocket(audio_capture, ws_client))
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


def main():
    """Main entry point"""

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test-audio":
            run_func = test_audio_capture
        elif mode == "test-websocket":
            run_func = partial(test_websocket_client, handle_transcription_event)
        else:
            print("Usage: python3 transcribe.py [test-audio|test-websocket]")
            print("  (no arguments): Run full real-time transcription")
            print("  test-audio: Test audio capture only")
            print("  test-websocket: Test WebSocket connection only")
            sys.exit(1)
    else:
        run_func = run_transcription

    try:
        asyncio.run(run_func())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
