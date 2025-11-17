import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest

from pi_assistant.audio import capture as capture_module
from pi_assistant.audio.capture import AudioCapture


def make_dummy_query(devices_by_index, device_lists=None):
    """Return a fake sd.query_devices implementation."""

    device_lists = device_lists or []

    def _query(device=None, kind=None):
        if device is None and kind is None:
            return device_lists
        if kind == "output":
            return {"name": "Default output"}
        if isinstance(device, int):
            if device in devices_by_index:
                return devices_by_index[device]
            raise ValueError("unknown device")
        raise ValueError("unsupported query")

    return _query


def test_select_input_device_prefers_override(monkeypatch):
    monkeypatch.setattr(capture_module, "AUDIO_INPUT_DEVICE", "5")
    monkeypatch.setattr(
        capture_module.sd,
        "query_devices",
        make_dummy_query({5: {"name": "Mic", "index": 5, "max_input_channels": 1}}),
    )

    capture = AudioCapture()
    selected = capture._select_input_device()

    assert selected == 5


def test_select_input_device_uses_default_when_valid(monkeypatch):
    monkeypatch.setattr(capture_module, "AUDIO_INPUT_DEVICE", "")
    monkeypatch.setattr(capture_module.sd, "default", SimpleNamespace(device=(3, 0)))
    monkeypatch.setattr(
        capture_module.sd,
        "query_devices",
        make_dummy_query({3: {"name": "DefaultMic", "index": 3, "max_input_channels": 1}}),
    )

    capture = AudioCapture()

    assert capture._select_input_device() == 3


def test_select_input_device_scans_available_when_needed(monkeypatch):
    monkeypatch.setattr(capture_module, "AUDIO_INPUT_DEVICE", None)
    monkeypatch.setattr(capture_module.sd, "default", SimpleNamespace(device=(-1, -1)))

    devices = [
        {"max_input_channels": 0},
        {"max_input_channels": capture_module.CHANNELS, "index": 9, "name": "Mic9"},
    ]

    def fake_query(device=None, kind=None):
        if device is None:
            return devices
        if isinstance(device, int):
            if device == -1:
                raise ValueError("invalid default")
            return {"name": f"Device{device}", "index": device, "max_input_channels": 1}
        raise ValueError("unsupported")

    monkeypatch.setattr(capture_module.sd, "query_devices", fake_query)

    capture = AudioCapture()

    assert capture._select_input_device() == 1


def test_first_available_input_device_raises_when_none(monkeypatch):
    monkeypatch.setattr(
        capture_module.sd,
        "query_devices",
        lambda: [],
    )

    capture = AudioCapture()

    with pytest.raises(RuntimeError):
        capture._first_available_input_device()


def test_callback_warns_when_queue_full(monkeypatch):
    capture = AudioCapture()
    capture.loop = type("Loop", (), {"call_soon_threadsafe": lambda self, cb, *args: cb(*args)})()

    class FullQueue:
        def put_nowait(self, _):
            raise asyncio.QueueFull

    capture.audio_queue = FullQueue()
    audio_chunk = np.zeros((4,), dtype=np.int16)

    stderr = io.StringIO()
    monkeypatch.setattr(capture_module.sys, "stderr", stderr)

    capture.callback(audio_chunk, frames=4, time_info=None, status=None)

    assert "Audio queue full" in stderr.getvalue()


def test_callback_logs_status(monkeypatch):
    capture = AudioCapture()
    capture.loop = type("Loop", (), {"call_soon_threadsafe": lambda self, cb, *args: cb(*args)})()

    class Queue:
        def put_nowait(self, _):
            pass

    capture.audio_queue = Queue()
    audio_chunk = np.zeros((4,), dtype=np.int16)

    stderr = io.StringIO()
    monkeypatch.setattr(capture_module.sys, "stderr", stderr)

    capture.callback(audio_chunk, frames=4, time_info=None, status="OVERFLOW")

    assert "Audio callback status: OVERFLOW" in stderr.getvalue()


def test_parse_device_override_whitespace():
    assert AudioCapture._parse_device_override("   ") is None
    assert AudioCapture._parse_device_override("\n\t 3 ") == 3


def test_select_input_device_raises_when_no_options(monkeypatch):
    monkeypatch.setattr(capture_module, "AUDIO_INPUT_DEVICE", None)
    monkeypatch.setattr(capture_module.sd, "default", SimpleNamespace(device=(-1, -1)))

    def fail_query(*args, **kwargs):
        raise RuntimeError("no devices")

    monkeypatch.setattr(capture_module.sd, "query_devices", fail_query)

    capture = AudioCapture()

    with pytest.raises(RuntimeError):
        capture._select_input_device()


def test_select_input_device_invalid_override(monkeypatch):
    monkeypatch.setattr(capture_module, "AUDIO_INPUT_DEVICE", "42")

    def fail_query(device):
        raise ValueError("unknown device")

    monkeypatch.setattr(capture_module.sd, "query_devices", fail_query)

    capture = AudioCapture()

    with pytest.raises(RuntimeError, match="AUDIO_INPUT_DEVICE"):
        capture._select_input_device()


def test_first_available_input_device_query_failure(monkeypatch):
    def fail_query():
        raise ValueError("portaudio error")

    monkeypatch.setattr(capture_module.sd, "query_devices", fail_query)

    capture = AudioCapture()

    with pytest.raises(RuntimeError):
        capture._first_available_input_device()


def test_stop_stream_handles_none():
    capture = AudioCapture()

    # Should not raise
    capture.stop_stream()


class _DummyStream:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_start_stream_initialization_failure(monkeypatch):
    capture = AudioCapture()
    monkeypatch.setattr(capture, "_select_input_device", lambda: 1)

    def fail_input_stream(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(capture_module.sd, "InputStream", fail_input_stream)

    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="Unable to initialize audio input stream"):
            capture.start_stream(loop=loop)
    finally:
        loop.close()


def test_start_and_stop_stream_success(monkeypatch):
    capture = AudioCapture()
    dummy_stream = _DummyStream()

    monkeypatch.setattr(capture, "_select_input_device", lambda: 0)
    monkeypatch.setattr(capture_module.sd, "InputStream", lambda **kwargs: dummy_stream)

    loop = asyncio.new_event_loop()
    try:
        capture.start_stream(loop=loop)
    finally:
        loop.close()

    assert capture.stream is dummy_stream
    assert dummy_stream.started is True

    capture.stop_stream()

    assert dummy_stream.stopped is True
    assert dummy_stream.closed is True
