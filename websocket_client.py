"""
WebSocket client module for real-time speech-to-text transcription
Handles WebSocket connection to OpenAI Realtime API
"""

import asyncio
import base64
import json
import sys

import websockets

from config import (
    OPENAI_REALTIME_ENDPOINT,
    SESSION_CONFIG,
    WEBSOCKET_HEADERS,
)


class WebSocketClient:
    """Handles WebSocket connection to OpenAI Realtime API"""

    def __init__(self):
        self.websocket = None
        self.connected = False

    async def connect(self):
        """
        Establish WebSocket connection to OpenAI Realtime API using API key
        """
        print("Connecting to OpenAI Realtime API...")
        print(f"  Endpoint: {OPENAI_REALTIME_ENDPOINT}")
        print("  Auth method: API key")

        try:
            # Connect with required headers
            self.websocket = await websockets.connect(
                OPENAI_REALTIME_ENDPOINT, additional_headers=WEBSOCKET_HEADERS
            )
            self.connected = True
            print("✓ Connected to OpenAI Realtime API")

            # Wait for initial session.created event (with timeout)
            try:
                await asyncio.wait_for(self.wait_for_session_created(), timeout=10.0)
                print("  (Initial session created)")
            except asyncio.TimeoutError:
                print("  [WARNING] No session.created event received after 10s")
                print("  [INFO] Trying to send session.update anyway...")

            await self.send_session_config()

        except Exception as e:
            print(f"Error connecting to OpenAI: {e}", file=sys.stderr)
            raise

    async def wait_for_session_created(self):
        """
        Wait for and capture the transcription_session.created event

        Returns:
            dict: The event from the server
        """
        print("Waiting for transcription_session.created event...")

        try:
            async for message in self.websocket:
                event = json.loads(message)
                event_type = event.get("type")

                # Debug: print what we received
                print(f"[DEBUG] Received event type: {event_type}")

                # Handle both transcription_session.created and error events
                if event_type == "transcription_session.created":
                    print("✓ Transcription session created")
                    return event
                elif event_type == "error":
                    error = event.get("error", {})
                    print(
                        f"[ERROR] Server error: {error.get('message', 'Unknown error')}",
                        file=sys.stderr,
                    )
                    # Don't raise - continue listening for more events
                else:
                    # Print the full event for debugging
                    print(f"[DEBUG] Full event: {json.dumps(event, indent=2)}")

        except Exception as e:
            print(f"Error waiting for session: {e}", file=sys.stderr)
            raise

    async def send_session_config(self):
        """
        Send transcription session configuration to OpenAI
        Configures audio format, model, VAD, and noise reduction
        """
        transcription = SESSION_CONFIG["input_audio_transcription"]
        vad = SESSION_CONFIG["turn_detection"]
        noise_reduction = SESSION_CONFIG["input_audio_noise_reduction"]

        print("Sending session configuration...")
        print(f"  Model: {transcription['model']}")
        print(f"  VAD: {vad['type']}")
        print(f"  Noise reduction: {noise_reduction['type']}")

        try:
            # Send the session update event with required session wrapper
            session_update = {"type": "transcription_session.update", "session": SESSION_CONFIG}
            await self.websocket.send(json.dumps(session_update))
            print("✓ Session configuration sent")

        except Exception as e:
            print(f"Error sending session config: {e}", file=sys.stderr)
            raise

    async def send_audio_chunk(self, audio_bytes):
        """
        Send audio chunk to OpenAI Realtime API

        Args:
            audio_bytes: Raw PCM16 audio data as bytes
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Create audio append message
        message = {"type": "input_audio_buffer.append", "audio": audio_base64}

        # Send to OpenAI
        await self.websocket.send(json.dumps(message))

    async def receive_events(self):
        """
        Continuously receive and handle events from OpenAI

        Yields:
            dict: Parsed JSON event from server
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")

        try:
            async for message in self.websocket:
                # Parse JSON event
                event = json.loads(message)
                yield event

        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed by server")
            self.connected = False

        except Exception as e:
            print(f"Error receiving events: {e}", file=sys.stderr)
            raise

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print("WebSocket connection closed")
