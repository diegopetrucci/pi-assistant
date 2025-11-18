import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from pi_assistant.audio import playback as playback_module
from pi_assistant.audio.playback import SpeechPlayer


class DummySD:
    def __init__(self):
        self.default = SimpleNamespace(device=(None, 0))
        self.play_calls = []
        self.stop_called = False
        self.wait_called = False
        self.check_calls = []
        self.output_info = {"name": "Dummy output", "default_samplerate": 16000}

    def play(self, data, samplerate, device):
        self.play_calls.append((data.copy(), samplerate, device))

    def wait(self):
        self.wait_called = True

    def stop(self):
        self.stop_called = True

    def check_output_settings(self, device, samplerate, channels):
        self.check_calls.append((device, samplerate, channels))
        if samplerate == 44100:  # noqa: PLR2004
            raise RuntimeError("unsupported rate")
        return True

    def query_devices(self, device=None, kind=None):
        if kind == "output":
            return self.output_info
        if isinstance(device, int):
            return {"name": f"Device{device}", "default_samplerate": 22050}
        raise ValueError("unsupported query")


async def run_async(coro):
    return await coro


@pytest.mark.asyncio
async def test_play_serializes_and_converts_to_float(monkeypatch):
    dummy_sd = DummySD()
    monkeypatch.setattr(playback_module, "sd", dummy_sd)

    player = SpeechPlayer(default_sample_rate=16000)
    samples = np.array([0, 32767, -32768], dtype=np.int16).tobytes()

    await run_async(player.play(samples, sample_rate=16000))

    assert len(dummy_sd.play_calls) == 1
    data, rate, device = dummy_sd.play_calls[0]
    np.testing.assert_allclose(data, np.array([0.0, 0.9999695, -1.0], dtype=np.float32))
    assert rate == player._playback_sample_rate
    assert device == player._output_device
    assert dummy_sd.wait_called


@pytest.mark.asyncio
async def test_stop_returns_false_when_idle(monkeypatch):
    dummy_sd = DummySD()
    monkeypatch.setattr(playback_module, "sd", dummy_sd)
    player = SpeechPlayer()

    assert await run_async(player.stop()) is False


@pytest.mark.asyncio
async def test_stop_interrupts_playback(monkeypatch):
    dummy_sd = DummySD()
    monkeypatch.setattr(playback_module, "sd", dummy_sd)
    player = SpeechPlayer()

    async def run_play():
        await player.play(np.array([0, 0], dtype=np.int16).tobytes(), sample_rate=16000)

    play_task = asyncio.create_task(run_play())
    await asyncio.sleep(0)
    assert await run_async(player.stop()) is True
    await play_task
    assert dummy_sd.stop_called


def test_prepare_samples_resamples_when_needed(monkeypatch):
    dummy_sd = DummySD()
    monkeypatch.setattr(playback_module, "sd", dummy_sd)
    player = SpeechPlayer()

    tone = np.array([1000, -1000], dtype=np.int16).tobytes()
    resampled = player._prepare_samples(tone, source_rate=8000)

    assert resampled.dtype == np.int16
    assert resampled.size > 0


def test_select_playback_rate_logs_override_once(monkeypatch, capsys):
    dummy_sd = DummySD()
    dummy_sd.output_info = {"name": "Dummy output", "default_samplerate": 22050}
    monkeypatch.setattr(playback_module, "sd", dummy_sd)

    player = SpeechPlayer(default_sample_rate=44100)
    assert player._playback_sample_rate != 44100  # noqa: PLR2004

    captured = capsys.readouterr()
    assert "does not support 44100 Hz" in captured.out

    player._log_playback_override(48000, 32000)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_warn_if_resampling_only_once(monkeypatch, capsys):
    dummy_sd = DummySD()
    monkeypatch.setattr(playback_module, "sd", dummy_sd)
    player = SpeechPlayer()

    player._warn_if_resampling(16000, 24000)
    player._warn_if_resampling(16000, 24000)

    out = capsys.readouterr().out
    assert out.count("Resampling assistant audio 16000 Hz -> 24000 Hz") == 1
