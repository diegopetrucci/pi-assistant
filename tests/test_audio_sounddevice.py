"""Unit tests for the sounddevice compatibility shim."""

from __future__ import annotations

import builtins

import pytest

from pi_assistant.audio import _sounddevice as sounddevice_mod


def test_dummy_stream_tracks_state_and_args() -> None:
    stream = sounddevice_mod._DummyStream("mic", samplerate=16000)
    assert stream.started is False
    assert stream._args == ("mic",)
    assert stream._kwargs == {"samplerate": 16000}

    stream.start()
    assert stream.started is True

    stream.stop()
    assert stream.started is False

    stream.start()
    stream.close()
    assert stream.started is False


def test_dummy_sounddevice_input_stream_and_queries() -> None:
    dummy = sounddevice_mod._DummySoundDevice()
    stream = dummy.InputStream("mic", channels=1)
    assert isinstance(stream, sounddevice_mod._DummyStream)

    devices = dummy.query_devices()
    assert devices[0]["name"] == "Dummy Input Device"
    assert devices[1]["name"] == "Dummy Output Device"

    assert dummy.query_devices(kind="output")["index"] == 1
    assert dummy.query_devices(device=0)["name"] == "Dummy Input Device"
    assert dummy.query_devices(device=1)["name"] == "Dummy Output Device"

    with pytest.raises(ValueError):
        dummy.query_devices(device=999)


def test_dummy_sounddevice_settings_and_playback_noops() -> None:
    dummy = sounddevice_mod._DummySoundDevice()

    assert dummy.check_input_settings() is True
    assert dummy.check_output_settings() is True

    dummy.play(b"1234")
    dummy.wait()
    dummy.stop()


def test_load_sounddevice_returns_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_module = object()
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "sounddevice":
            return sentinel_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert sounddevice_mod._load_sounddevice() is sentinel_module


def test_load_sounddevice_returns_dummy_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("missing sounddevice in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sounddevice_mod, "_ALLOW_DUMMY", True)

    result = sounddevice_mod._load_sounddevice()
    assert isinstance(result, sounddevice_mod._DummySoundDevice)


def test_load_sounddevice_raises_when_dummy_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("missing sounddevice in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(sounddevice_mod, "_ALLOW_DUMMY", False)

    with pytest.raises(ImportError):
        sounddevice_mod._load_sounddevice()
