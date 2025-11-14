"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""

import asyncio
import sys

from audio_capture import AudioCapture
from config import BUFFER_SIZE, CHANNELS, SAMPLE_RATE
from websocket_client import WebSocketClient


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
        print(f"\n[TRANSCRIPT] {transcript}")

    elif event_type == "input_audio_buffer.committed":
        # VAD detected speech
        item_id = event.get("item_id", "")
        print(f"[VAD] Speech detected (item: {item_id})")

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

    try:
        while True:
            # Get audio chunk from queue
            audio_bytes = await audio_capture.get_audio_chunk()

            # Send to OpenAI via WebSocket
            await ws_client.send_audio_chunk(audio_bytes)

            chunk_count += 1

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


async def test_audio_capture():
    """Test function to verify audio capture is working"""
    print("\n=== Audio Capture Test ===\n")

    # Create audio capture instance
    capture = AudioCapture()

    # Get current event loop
    loop = asyncio.get_running_loop()

    # Start audio stream
    capture.start_stream(loop)

    print("\nCapturing audio for 5 seconds...")
    print("(Speak into your microphone or make some noise)\n")

    chunk_count = 0
    total_bytes = 0

    try:
        # Capture for 5 seconds
        start_time = loop.time()
        print(f"Start time: {start_time}")
        while loop.time() - start_time < 5.0:
            # Get audio chunk with timeout
            try:
                audio_data = await asyncio.wait_for(capture.get_audio_chunk(), timeout=1.0)
                chunk_count += 1
                total_bytes += len(audio_data)

                # Show progress every 10 chunks
                if chunk_count % 10 == 0:
                    print(f"Captured {chunk_count} chunks, {total_bytes:,} bytes")
            except asyncio.TimeoutError:
                print("Warning: No audio data received (timeout)")
                break

    except KeyboardInterrupt:
        print("\nTest interrupted")

    finally:
        # Stop stream
        capture.stop_stream()

        print("\n=== Test Complete ===")
        print(f"Total chunks: {chunk_count}")
        print(f"Total bytes: {total_bytes:,}")
        print(f"Expected bytes per chunk: {BUFFER_SIZE * CHANNELS * 2}")  # 2 bytes per int16 sample
        print(f"Audio format verified: {SAMPLE_RATE}Hz, {CHANNELS} channel(s), 16-bit PCM")


async def test_websocket_client():
    """Test function to verify WebSocket client connection and configuration"""
    print("\n=== WebSocket Client Test ===\n")

    # Create WebSocket client instance
    ws_client = WebSocketClient()

    try:
        # Connect to OpenAI
        await ws_client.connect()

        print("\n✓ Connection successful")
        print("✓ Session configuration sent")
        print("\nListening for events for 5 seconds...")
        print("(The server may send acknowledgment or other events)\n")

        # Listen for events for 5 seconds
        event_count = 0
        try:
            async for event in ws_client.receive_events():
                event_count += 1
                handle_transcription_event(event)

                # Break after 5 seconds or 10 events (whichever comes first)
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
        # Close WebSocket connection
        await ws_client.close()


def main():
    """Main entry point"""
    import sys

    # Determine which mode to run based on command-line argument
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "test-audio":
            run_func = test_audio_capture
        elif mode == "test-websocket":
            run_func = test_websocket_client
        else:
            print("Usage: python3 transcribe.py [test-audio|test-websocket]")
            print("  (no arguments): Run full real-time transcription")
            print("  test-audio: Test audio capture only")
            print("  test-websocket: Test WebSocket connection only")
            sys.exit(1)
    else:
        # Default to full transcription
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
