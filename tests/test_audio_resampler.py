import math

import numpy as np
import pytest

from pi_transcription.audio.resampler import LinearResampler


def pcm16(values):
    return np.array(values, dtype=np.int16).tobytes()


def test_resampler_rejects_invalid_params():
    with pytest.raises(ValueError):
        LinearResampler(0, 16000)
    with pytest.raises(ValueError):
        LinearResampler(16000, 0)
    with pytest.raises(ValueError):
        LinearResampler(16000, 8000, channels=0)


def test_pass_through_when_rates_match():
    samples = pcm16([0, 1000, -1000])
    resampler = LinearResampler(16000, 16000)

    result = resampler.process(samples)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result, np.frombuffer(samples, dtype=np.int16))


def test_process_returns_empty_array_for_no_data():
    resampler = LinearResampler(24000, 16000)

    result = resampler.process(b"")

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_downsamples_and_preserves_state():
    tone = pcm16([1000, -1000] * 8)
    resampler = LinearResampler(24000, 12000)

    first = resampler.process(tone[: len(tone) // 2])
    second = resampler.process(tone[len(tone) // 2 :])

    combined = np.concatenate([first, second])
    assert combined.size > 0
    assert combined.dtype == np.int16

    resampler.reset()
    after_reset = resampler.process(tone[: len(tone) // 2])
    np.testing.assert_array_equal(first, after_reset)


def test_upsamples_simple_waveform():
    sine_wave = []
    for i in range(8):
        value = int(2000 * math.sin(i / 8 * math.pi * 2))
        sine_wave.append(value)
    resampler = LinearResampler(8000, 16000)

    result = resampler.process(pcm16(sine_wave))

    assert result.size > len(sine_wave)
    assert result.dtype == np.int16
