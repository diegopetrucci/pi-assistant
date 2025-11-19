"""
Audio capture module for real-time speech-to-text transcription
Handles audio capture from USB microphone
"""

import asyncio
import os
import sys
from typing import Protocol, cast

from pi_assistant.cli.logging_utils import verbose_print
from pi_assistant.config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_QUEUE_MAX_SIZE,
    BUFFER_SIZE,
    CHANNELS,
    DTYPE,
    SAMPLE_RATE,
)
from pi_assistant.config.base import _persist_env_value

from ._sounddevice import sounddevice as sd
from .utils import device_info_dict


class _AudioQueue(Protocol):
    async def get(self) -> bytes: ...

    def put_nowait(self, item: bytes) -> None: ...


class AudioCapture:
    """Handles audio capture from USB microphone"""

    def __init__(self):
        self.audio_queue: _AudioQueue = cast(
            _AudioQueue, asyncio.Queue(maxsize=AUDIO_QUEUE_MAX_SIZE)
        )
        self.stream = None
        self.loop = None
        self.callback_count = 0  # Debug counter
        self.input_device = None
        self.sample_rate = SAMPLE_RATE

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
        if self.callback_count <= 3:  # noqa: PLR2004
            verbose_print(
                f"[DEBUG] Callback #{self.callback_count}: {len(indata)} frames", flush=True
            )

        if status:
            print(f"Audio callback status: {status}", file=sys.stderr)

        # Convert numpy array to bytes
        audio_bytes = indata.copy().tobytes()

        # Put audio data in queue from the event loop thread.
        # If the queue is full, skip this chunk to prevent blocking.
        loop = self.loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._enqueue_audio_bytes, audio_bytes)

    def start_stream(self, loop):
        """
        Initialize and start the audio stream

        Args:
            loop: asyncio event loop for threadsafe operations
        """
        self.loop = loop
        self.sample_rate = SAMPLE_RATE

        verbose_print("Initializing audio stream...")
        verbose_print(f"  Sample rate: {self.sample_rate} Hz")
        verbose_print(f"  Channels: {CHANNELS} (mono)")
        verbose_print(f"  Buffer size: {BUFFER_SIZE} frames")
        verbose_print(f"  Data type: {DTYPE}")

        device = self._select_input_device()
        self.input_device = device
        verbose_print(f"  Input device: {self._describe_device(device)}")

        self._ensure_sample_rate_supported(device)

        # Initialize sounddevice input stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=BUFFER_SIZE,
                callback=self.callback,
                device=device,
            )
        except Exception as exc:
            raise self._stream_initialization_error(exc, device) from exc

        self.stream.start()
        verbose_print("Audio stream started")

    def stop_stream(self):
        """Stop and close the audio stream"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            verbose_print("Audio stream closed")

    def _enqueue_audio_bytes(self, audio_bytes: bytes) -> None:
        """Attempt a non-blocking enqueue of audio; drop if the queue is full."""
        try:
            self.audio_queue.put_nowait(audio_bytes)
        except asyncio.QueueFull:
            print("Warning: Audio queue full, dropping frame", file=sys.stderr)

    def _ensure_sample_rate_supported(self, device) -> None:
        """Validate that the selected device accepts the configured sample rate."""

        try:
            sd.check_input_settings(
                device=device,
                channels=CHANNELS,
                dtype=DTYPE,
                samplerate=self.sample_rate,
            )
        except Exception as exc:
            fallback_rate = self._device_default_sample_rate(device)
            if fallback_rate and fallback_rate != self.sample_rate:
                self._persist_sample_rate_hint(device, fallback_rate)
            raise self._unsupported_sample_rate_error(device) from exc

    def _persist_sample_rate_hint(self, device, fallback_rate: int) -> None:
        """Save the suggested sample rate so the next launch succeeds."""

        device_label = self._describe_device(device)
        _persist_env_value("SAMPLE_RATE", str(fallback_rate))
        os.environ["SAMPLE_RATE"] = str(fallback_rate)
        print(
            "[INFO] "
            f"Detected microphone {device_label} prefers {fallback_rate} Hz. "
            "Saved SAMPLE_RATE to .env; restart the assistant to apply."
        )

    def _unsupported_sample_rate_error(self, device) -> RuntimeError:
        """Return a descriptive error when the mic rejects SAMPLE_RATE."""

        device_label = self._describe_device(device)
        return RuntimeError(
            f"Microphone {device_label} does not support SAMPLE_RATE={self.sample_rate} Hz."
        )

    def _device_default_sample_rate(self, device) -> int | None:
        try:
            info = device_info_dict(sd.query_devices(device))
        except Exception:
            return None

        candidate = info.get("default_samplerate")
        if isinstance(candidate, (int, float, str)):
            try:
                return int(float(candidate))
            except ValueError:
                return None
        return None

    def _stream_initialization_error(self, exc: Exception, device) -> RuntimeError:
        message = str(exc).lower()
        if "sample rate" in message or "painvalidsamplerate" in message:
            return self._unsupported_sample_rate_error(device)

        return RuntimeError(
            "Unable to initialize audio input stream. "
            "Verify that a microphone is connected and available. "
            "Set AUDIO_INPUT_DEVICE to override the default device."
        )

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

        for idx, entry in enumerate(self._iter_device_records(devices)):
            max_channels = entry.get("max_input_channels")
            if isinstance(max_channels, (int, float)) and int(max_channels) >= CHANNELS:
                return idx

        raise RuntimeError(
            "No audio input devices with the required channel count were found. "
            "Connect a microphone and retry."
        )

    def _describe_device(self, device):
        if device is None:
            return "system default"

        try:
            info = device_info_dict(sd.query_devices(device))
            name_obj = info.get("name")
            name = str(name_obj) if name_obj not in (None, "") else "Unknown device"
            idx_obj = info.get("index")
            index = (
                idx_obj if isinstance(idx_obj, int) else device if isinstance(device, int) else "?"
            )
            return f"{name} (id {index})"
        except Exception:
            return str(device)

    def _iter_device_records(self, devices: object) -> list[dict[str, object]]:
        if isinstance(devices, list):
            return [device_info_dict(item) for item in devices]
        if isinstance(devices, tuple):
            return [device_info_dict(item) for item in devices]
        return [device_info_dict(devices)]
