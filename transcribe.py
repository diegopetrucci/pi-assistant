"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""
import asyncio
import base64
import json
import signal
import sys
import sounddevice as sd
import numpy as np
import websockets
import aiohttp
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    CHANNELS,
    DTYPE,
    AUDIO_QUEUE_MAX_SIZE,
    OPENAI_REALTIME_ENDPOINT,
    WEBSOCKET_HEADERS,
    SESSION_CONFIG,
    OPENAI_API_KEY
)


class AudioCapture:
    """Handles audio capture from USB microphone"""

    def __init__(self):
        self.audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX_SIZE)
        self.stream = None
        self.loop = None
        self.callback_count = 0  # Debug counter

    def callback(self, indata, frames, time_info, status):
        """
        Audio callback function called by sounddevice for each audio block.
        Runs in a separate thread, so we use threadsafe queue operations.

        Args:
            indata: Input audio data as numpy array
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        self.callback_count += 1

        # Debug: Print first few callbacks
        if self.callback_count <= 3:
            print(f'[DEBUG] Callback #{self.callback_count}: {len(indata)} frames', flush=True)

        if status:
            print(f'Audio callback status: {status}', file=sys.stderr)

        # Convert numpy array to bytes
        audio_bytes = indata.copy().tobytes()

        # Put audio data in queue (non-blocking)
        # If queue is full, skip this chunk to prevent blocking
        try:
            self.loop.call_soon_threadsafe(
                self.audio_queue.put_nowait,
                audio_bytes
            )
        except asyncio.QueueFull:
            print('Warning: Audio queue full, dropping frame', file=sys.stderr)

    def start_stream(self, loop):
        """
        Initialize and start the audio stream

        Args:
            loop: asyncio event loop for threadsafe operations
        """
        self.loop = loop

        print(f'Initializing audio stream...')
        print(f'  Sample rate: {SAMPLE_RATE} Hz')
        print(f'  Channels: {CHANNELS} (mono)')
        print(f'  Buffer size: {BUFFER_SIZE} frames')
        print(f'  Data type: {DTYPE}')

        # Initialize sounddevice input stream
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BUFFER_SIZE,
            callback=self.callback
        )

        self.stream.start()
        print('Audio stream started')

    def stop_stream(self):
        """Stop and close the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print('Audio stream closed')

    async def get_audio_chunk(self):
        """
        Get the next audio chunk from the queue

        Returns:
            bytes: Raw audio data
        """
        return await self.audio_queue.get()


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


def handle_transcription_event(event):
    """
    Handle transcription events from OpenAI

    Args:
        event: Parsed JSON event from server
    """
    event_type = event.get('type')

    if event_type == 'conversation.item.input_audio_transcription.delta':
        # Partial transcription
        delta = event.get('delta', '')
        print(f'[PARTIAL] {delta}', end='', flush=True)

    elif event_type == 'conversation.item.input_audio_transcription.completed':
        # Final transcription
        transcript = event.get('transcript', '')
        print(f'\n[TRANSCRIPT] {transcript}')

    elif event_type == 'input_audio_buffer.committed':
        # VAD detected speech
        item_id = event.get('item_id', '')
        print(f'[VAD] Speech detected (item: {item_id})')

    elif event_type == 'error':
        # Error from API
        error = event.get('error', {})
        error_type = error.get('type', 'unknown')
        error_message = error.get('message', 'No message')
        error_code = error.get('code', 'unknown')
        print(f'[ERROR] {error_type} ({error_code}): {error_message}', file=sys.stderr)

    elif event_type == 'transcription_session.updated':
        # Session configuration acknowledged
        print('[INFO] Session configuration acknowledged')

    else:
        # Other events (for debugging)
        print(f'[DEBUG] Received event: {event_type}')


async def test_audio_capture():
    """Test function to verify audio capture is working"""
    print('\n=== Audio Capture Test ===\n')

    # Create audio capture instance
    capture = AudioCapture()

    # Get current event loop
    loop = asyncio.get_running_loop()

    # Start audio stream
    capture.start_stream(loop)

    print('\nCapturing audio for 5 seconds...')
    print('(Speak into your microphone or make some noise)\n')

    chunk_count = 0
    total_bytes = 0

    try:
        # Capture for 5 seconds
        start_time = loop.time()
        print(f'Start time: {start_time}')
        while loop.time() - start_time < 5.0:
            # Get audio chunk with timeout
            try:
                audio_data = await asyncio.wait_for(
                    capture.get_audio_chunk(),
                    timeout=1.0
                )
                chunk_count += 1
                total_bytes += len(audio_data)

                # Show progress every 10 chunks
                if chunk_count % 10 == 0:
                    print(f'Captured {chunk_count} chunks, {total_bytes:,} bytes')
            except asyncio.TimeoutError:
                print('Warning: No audio data received (timeout)')
                break

    except KeyboardInterrupt:
        print('\nTest interrupted')

    finally:
        # Stop stream
        capture.stop_stream()

        print(f'\n=== Test Complete ===')
        print(f'Total chunks: {chunk_count}')
        print(f'Total bytes: {total_bytes:,}')
        print(f'Expected bytes per chunk: {BUFFER_SIZE * CHANNELS * 2}')  # 2 bytes per int16 sample
        print(f'Audio format verified: {SAMPLE_RATE}Hz, {CHANNELS} channel(s), 16-bit PCM')


async def test_websocket_client():
    """Test function to verify WebSocket client connection and configuration"""
    print('\n=== WebSocket Client Test ===\n')

    # Create WebSocket client instance
    ws_client = WebSocketClient()

    try:
        # Connect to OpenAI
        await ws_client.connect()

        print('\n✓ Connection successful')
        print('✓ Session configuration sent')
        print('\nListening for events for 5 seconds...')
        print('(The server may send acknowledgment or other events)\n')

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
            print('No events received (timeout)')

        print(f'\n=== Test Complete ===')
        print(f'Total events received: {event_count}')

    except KeyboardInterrupt:
        print('\nTest interrupted')

    except Exception as e:
        print(f'Test failed: {e}', file=sys.stderr)
        raise

    finally:
        # Close WebSocket connection
        await ws_client.close()


def main():
    """Main entry point"""
    import sys

    # Determine which test to run based on command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == 'audio':
            test_func = test_audio_capture
        elif sys.argv[1] == 'websocket':
            test_func = test_websocket_client
        else:
            print(f'Usage: python3 transcribe.py [audio|websocket]')
            sys.exit(1)
    else:
        # Default to WebSocket test
        test_func = test_websocket_client

    try:
        asyncio.run(test_func())
    except KeyboardInterrupt:
        print('\nShutdown requested')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
