"""
WebSocket client module for real-time speech-to-text transcription
Handles WebSocket connection to OpenAI Realtime API
"""
import base64
import json
import sys
import websockets
import aiohttp
from config import (
    OPENAI_REALTIME_ENDPOINT,
    WEBSOCKET_HEADERS,
    SESSION_CONFIG,
    OPENAI_API_KEY
)


class WebSocketClient:
    """Handles WebSocket connection to OpenAI Realtime API"""

    def __init__(self):
        self.websocket = None
        self.connected = False
        self.ephemeral_token = None

    async def get_ephemeral_token(self):
        """
        Get ephemeral token from OpenAI for WebSocket authentication

        Returns:
            str: Ephemeral token (client_secret) for WebSocket connection
        """
        print('Requesting ephemeral token...')

        url = 'https://api.openai.com/v1/realtime/transcription_sessions'
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Empty payload - the endpoint may not require any body
        payload = {}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f'Failed to get ephemeral token: {response.status} - {error_text}')

                    data = await response.json()
                    self.ephemeral_token = data.get('client_secret', {}).get('value')

                    if not self.ephemeral_token:
                        raise Exception('No client_secret in response')

                    print('✓ Ephemeral token obtained')
                    return self.ephemeral_token

        except Exception as e:
            print(f'Error getting ephemeral token: {e}', file=sys.stderr)
            raise

    async def connect(self, use_ephemeral_token=True):
        """
        Establish WebSocket connection to OpenAI Realtime API

        Args:
            use_ephemeral_token: If True, get ephemeral token first (recommended for transcription)
        """
        # Get ephemeral token if requested
        if use_ephemeral_token:
            await self.get_ephemeral_token()
            # Use ephemeral token for WebSocket connection
            headers = {
                'Authorization': f'Bearer {self.ephemeral_token}',
                'OpenAI-Beta': 'realtime=v1'
            }
            # Use base endpoint without model parameter when using ephemeral token
            endpoint = 'wss://api.openai.com/v1/realtime'
        else:
            # Use API key directly
            headers = WEBSOCKET_HEADERS
            endpoint = OPENAI_REALTIME_ENDPOINT

        print('Connecting to OpenAI Realtime API...')
        print(f'  Endpoint: {endpoint}')
        print(f'  Auth method: {"Ephemeral token" if use_ephemeral_token else "API key"}')

        try:
            # Connect with required headers
            self.websocket = await websockets.connect(
                endpoint,
                additional_headers=headers
            )
            self.connected = True
            print('✓ Connected to OpenAI Realtime API')

            # Send session configuration immediately after connection
            # (skip if using ephemeral token - config already set server-side)
            if not use_ephemeral_token:
                await self.send_session_config()
            else:
                print('  (Session config already set via ephemeral token)')

        except Exception as e:
            print(f'Error connecting to OpenAI: {e}', file=sys.stderr)
            raise

    async def send_session_config(self):
        """
        Send transcription session configuration to OpenAI
        Configures audio format, model, VAD, and noise reduction
        """
        print('Sending session configuration...')
        print(f'  Model: {SESSION_CONFIG["input_audio_transcription"]["model"]}')
        print(f'  VAD: {SESSION_CONFIG["turn_detection"]["type"]}')
        print(f'  Noise reduction: {SESSION_CONFIG["input_audio_noise_reduction"]["type"]}')

        try:
            config_json = json.dumps(SESSION_CONFIG)
            await self.websocket.send(config_json)
            print('Session configuration sent')

        except Exception as e:
            print(f'Error sending session config: {e}', file=sys.stderr)
            raise

    async def send_audio_chunk(self, audio_bytes):
        """
        Send audio chunk to OpenAI Realtime API

        Args:
            audio_bytes: Raw PCM16 audio data as bytes
        """
        if not self.connected or not self.websocket:
            raise RuntimeError('WebSocket not connected')

        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Create audio append message
        message = {
            "type": "input_audio_buffer.append",
            "audio": audio_base64
        }

        # Send to OpenAI
        await self.websocket.send(json.dumps(message))

    async def receive_events(self):
        """
        Continuously receive and handle events from OpenAI

        Yields:
            dict: Parsed JSON event from server
        """
        if not self.connected or not self.websocket:
            raise RuntimeError('WebSocket not connected')

        try:
            async for message in self.websocket:
                # Parse JSON event
                event = json.loads(message)
                yield event

        except websockets.exceptions.ConnectionClosed:
            print('WebSocket connection closed by server')
            self.connected = False

        except Exception as e:
            print(f'Error receiving events: {e}', file=sys.stderr)
            raise

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print('WebSocket connection closed')
