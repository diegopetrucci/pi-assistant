import unittest
import wave
from pathlib import Path

import wake_word
from config import (
    BUFFER_SIZE,
    SAMPLE_RATE,
    WAKE_WORD_CONSECUTIVE_FRAMES,
    WAKE_WORD_EMBEDDING_MODEL_PATH,
    WAKE_WORD_MELSPEC_MODEL_PATH,
    WAKE_WORD_MODEL_FALLBACK_PATH,
    WAKE_WORD_MODEL_PATH,
    WAKE_WORD_SCORE_THRESHOLD,
    WAKE_WORD_TARGET_SAMPLE_RATE,
)
from wake_word import WakeWordEngine

FIXTURE_PATH = Path("tests/hey_jarvis.wav")
OPENWAKEWORD_AVAILABLE = wake_word.Model is not None


@unittest.skipUnless(OPENWAKEWORD_AVAILABLE, "openwakeword is not installed")
class WakeWordEngineTest(unittest.TestCase):
    """Basic regression test to ensure the wake-word detector fires on sample audio."""

    def test_detects_hey_jarvis_once(self):
        if not FIXTURE_PATH.exists():
            self.skipTest("wake-word fixture audio missing")

        try:
            engine = WakeWordEngine(
                WAKE_WORD_MODEL_PATH,
                fallback_model_path=WAKE_WORD_MODEL_FALLBACK_PATH,
                melspec_model_path=WAKE_WORD_MELSPEC_MODEL_PATH,
                embedding_model_path=WAKE_WORD_EMBEDDING_MODEL_PATH,
                source_sample_rate=SAMPLE_RATE,
                target_sample_rate=WAKE_WORD_TARGET_SAMPLE_RATE,
                threshold=WAKE_WORD_SCORE_THRESHOLD,
                consecutive_required=WAKE_WORD_CONSECUTIVE_FRAMES,
            )
        except RuntimeError as exc:
            self.skipTest(f"WakeWordEngine unavailable: {exc}")

        trigger_count = 0
        peak_score = 0.0

        with wave.open(str(FIXTURE_PATH), "rb") as wav_file:
            self.assertEqual(
                wav_file.getframerate(),
                SAMPLE_RATE,
                "Fixture sample rate must match the capture pipeline.",
            )
            frames_per_chunk = BUFFER_SIZE
            chunk = wav_file.readframes(frames_per_chunk)
            while chunk:
                result = engine.process_chunk(chunk)
                peak_score = max(peak_score, result.score)
                if result.triggered:
                    trigger_count += 1
                chunk = wav_file.readframes(frames_per_chunk)

        self.assertGreaterEqual(
            peak_score,
            WAKE_WORD_SCORE_THRESHOLD,
            "Expected the wake-word score to cross the configured threshold.",
        )
        self.assertEqual(
            trigger_count,
            1,
            "Expected a single wake-word trigger for the fixture audio.",
        )
