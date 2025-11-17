"""
Helper routines for validating hardware and connectivity outside the main
transcription loop.
"""

import asyncio
import sys

from pi_assistant.audio import AudioCapture
from pi_assistant.config import BUFFER_SIZE, CHANNELS, SAMPLE_RATE
from pi_assistant.network import WebSocketClient


async def test_audio_capture():
    """Capture audio for a short window to verify microphone + PyAudio setup."""
    print("\n=== Audio Capture Test ===\n")

    capture = AudioCapture()
    loop = asyncio.get_running_loop()
    capture.start_stream(loop)

    print("\nCapturing audio for 5 seconds...")
    print("(Speak into your microphone or make some noise)\n")

    chunk_count = 0
    total_bytes = 0

    try:
        start_time = loop.time()
        print(f"Start time: {start_time}")
        while loop.time() - start_time < 5.0:
            try:
                audio_data = await asyncio.wait_for(capture.get_audio_chunk(), timeout=1.0)
                chunk_count += 1
                total_bytes += len(audio_data)

                if chunk_count % 10 == 0:
                    print(f"Captured {chunk_count} chunks, {total_bytes:,} bytes")
            except asyncio.TimeoutError:
                print("Warning: No audio data received (timeout)")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        capture.stop_stream()

        print("\n=== Test Complete ===")
        print(f"Total chunks: {chunk_count}")
        print(f"Total bytes: {total_bytes:,}")
        print(f"Expected bytes per chunk: {BUFFER_SIZE * CHANNELS * 2}")
        print(f"Audio format verified: {SAMPLE_RATE}Hz, {CHANNELS} channel(s), 16-bit PCM")


async def test_websocket_client(event_handler=None):
    """Connect to OpenAI and stream back any events for quick validation."""
    print("\n=== WebSocket Client Test ===\n")

    ws_client = WebSocketClient()

    try:
        await ws_client.connect()

        print("\n✓ Connection successful")
        print("✓ Session configuration sent")
        print("\nListening for events (up to 10 messages)...\n")

        event_count = 0
        try:
            async for event in ws_client.receive_events():
                event_count += 1
                if event_handler:
                    event_handler(event)
                else:
                    print(event)

                if event_count >= 10:
                    break

        except asyncio.TimeoutError:
            print("No events received (timeout)")

        print("\n=== Test Complete ===")
        print(f"Total events received: {event_count}")

    except KeyboardInterrupt:
        print("\nTest interrupted")

    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        raise

    finally:
        await ws_client.close()


if __name__ == "__main__":
    print("Run individual tests via `pi-assistant test-audio` or `pi-assistant test-websocket`.")
