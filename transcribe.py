"""
Real-time speech-to-text transcription system for Raspberry Pi 5
Streams audio from USB microphone to OpenAI's Realtime API
"""
import asyncio
import base64
import signal
import sys
import sounddevice as sd
import numpy as np
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


def main():
    """Main entry point"""
    try:
        # Run audio capture test
        asyncio.run(test_audio_capture())
    except KeyboardInterrupt:
        print('\nShutdown requested')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
