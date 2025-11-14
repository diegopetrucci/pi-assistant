"""
Audio capture module for real-time speech-to-text transcription
Handles audio capture from USB microphone
"""
import asyncio
import sys
import sounddevice as sd
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    CHANNELS,
    DTYPE,
    AUDIO_QUEUE_MAX_SIZE
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
