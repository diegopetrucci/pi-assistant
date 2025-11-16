import numpy as np

from pi_transcription.cli import controller


def test_calculate_rms_handles_empty_bytes():
    assert controller.calculate_rms(b"") == 0.0
    assert controller.calculate_rms(b"\x00\x00" * 0) == 0.0


def test_calculate_rms_returns_expected_value():
    samples = np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    rms = controller.calculate_rms(samples)

    assert 26000 < rms < 28000  # approximate RMS for the provided samples
