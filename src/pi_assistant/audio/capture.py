"""
Audio capture module for real-time speech-to-text transcription
Handles audio capture from USB microphone
"""

import asyncio
import sys

from pi_assistant.cli.logging_utils import verbose_print
from pi_assistant.config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_QUEUE_MAX_SIZE,
    BUFFER_SIZE,
    CHANNELS,
    DTYPE,
    SAMPLE_RATE,
)

from ._sounddevice import sounddevice as sd


class AudioCapture:
    """Handles audio capture from USB microphone"""

    def __init__(self):
        self.audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX_SIZE)
        self.stream = None
        self.loop = None
        self.callback_count = 0  # Debug counter
        self.input_device = None

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
            verbose_print(
                f"[DEBUG] Callback #{self.callback_count}: {len(indata)} frames", flush=True
            )

        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        # Convert numpy array to bytes
        audio_bytes = indata.copy().tobytes()

        # Put audio data in queue (non-blocking)
        # If queue is full, skip this chunk to prevent blocking
        try:
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, audio_bytes)
        except asyncio.QueueFull:
            print("Warning: Audio queue full, dropping frame", file=sys.stderr)

    def start_stream(self, loop):
        """
        Initialize and start the audio stream

        Args:
            loop: asyncio event loop for threadsafe operations
        """
        self.loop = loop

        verbose_print("Initializing audio stream...")
        verbose_print(f"  Sample rate: {SAMPLE_RATE} Hz")
        verbose_print(f"  Channels: {CHANNELS} (mono)")
        verbose_print(f"  Buffer size: {BUFFER_SIZE} frames")
        verbose_print(f"  Data type: {DTYPE}")

        device = self._select_input_device()
        self.input_device = device
        verbose_print(f"  Input device: {self._describe_device(device)}")

        # Initialize sounddevice input stream
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BUFFER_SIZE,
                callback=self.callback,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize audio input stream. "
                "Verify that a microphone is connected and available. "
                "Set AUDIO_INPUT_DEVICE to override the default device."
            ) from exc

        self.stream.start()
        verbose_print("Audio stream started")

    def stop_stream(self):
        """Stop and close the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            verbose_print("Audio stream closed")

    async def get_audio_chunk(self):
        """
        Get the next audio chunk from the queue

        Returns:
            bytes: Raw audio data
        """
        return await self.audio_queue.get()

    def _select_input_device(self):
        """
        Determine which audio input device to use.

        Prefers the explicit AUDIO_INPUT_DEVICE override, then the system default,
        then falls back to the first enumerated input device with sufficient channels.
        """

        override = self._parse_device_override(AUDIO_INPUT_DEVICE)
        if override is not None:
            self._validate_device(override)
            return override

        default_device = self._coerce_input_index(sd.default.device)
        if default_device is not None and default_device >= 0:
            if self._device_is_valid(default_device):
                return default_device

        return self._first_available_input_device()

    @staticmethod
    def _parse_device_override(value):
        if not value:
            return None

        candidate = value.strip()
        if not candidate:
            return None

        try:
            return int(candidate)
        except ValueError:
            return candidate

    @staticmethod
    def _coerce_input_index(device):
        if isinstance(device, (list, tuple)):
            candidate = device[0]
        else:
            candidate = device

        return candidate if isinstance(candidate, int) else None

    def _device_is_valid(self, device):
        try:
            sd.query_devices(device)
            return True
        except Exception:
            return False

    def _validate_device(self, device):
        try:
            sd.query_devices(device)
        except Exception as exc:
            raise RuntimeError(
                f"AUDIO_INPUT_DEVICE '{device}' is not recognized by sounddevice."
            ) from exc

    def _first_available_input_device(self):
        try:
            devices = sd.query_devices()
        except Exception as exc:
            raise RuntimeError(
                "Unable to query audio devices via PortAudio. "
                "Run `arecord -l` to verify the USB microphone is attached."
            ) from exc

        for idx, info in enumerate(devices):
            if info.get("max_input_channels", 0) >= CHANNELS:
                return idx

        raise RuntimeError(
            "No audio input devices with the required channel count were found. "
            "Connect a microphone and retry."
        )

    def _describe_device(self, device):
        if device is None:
            return "system default"

        try:
            info = sd.query_devices(device)
            name = info.get("name", "Unknown device")
            index = info.get("index", device if isinstance(device, int) else "?")
            return f"{name} (id {index})"
        except Exception:
            return str(device)
