"""
Configuration settings for Raspberry Pi Audio Transcription System
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Return True when the env var is set to a truthy value."""

    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent

# Audio Configuration
SAMPLE_RATE = 24000  # 24 kHz (OpenAI requirement)
BUFFER_SIZE = 1024  # frames (balanced for Pi 5 performance)
CHANNELS = 1  # Mono
DTYPE = "int16"  # 16-bit PCM

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# For transcription mode, use ?intent=transcription
OPENAI_REALTIME_ENDPOINT = "wss://api.openai.com/v1/realtime?intent=transcription"
OPENAI_MODEL = "gpt-4o-transcribe"

# WebSocket Headers
WEBSOCKET_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}

# Session Configuration for OpenAI Realtime API (Transcription mode)
# Sent directly as a transcription_session.update event
SESSION_CONFIG = {
    "input_audio_format": "pcm16",
    "input_audio_transcription": {"model": "gpt-4o-transcribe", "prompt": "", "language": "en"},
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    },
    "input_audio_noise_reduction": {"type": "near_field"},
    "include": ["item.input_audio_transcription.logprobs"],
}

# Auto-stop Configuration
AUTO_STOP_ENABLED = True
AUTO_STOP_SILENCE_THRESHOLD = 500  # RMS amplitude (0-32767 for int16)
AUTO_STOP_MAX_SILENCE_SECONDS = 2.0  # Stop after this much silence (post speech)

# Queue Configuration
AUDIO_QUEUE_MAX_SIZE = 100  # Limit queue size to prevent memory buildup


# Wake-word / gating configuration
def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


WAKE_WORD_MODEL_PATH = _env_path(
    "WAKE_WORD_MODEL_PATH", BASE_DIR / "models" / "hey_jarvis_v0.1.onnx"
)
WAKE_WORD_MODEL_FALLBACK_PATH = _env_path(
    "WAKE_WORD_MODEL_FALLBACK_PATH", BASE_DIR / "models" / "hey_jarvis_v0.1.tflite"
)
WAKE_WORD_MELSPEC_MODEL_PATH = _env_path(
    "WAKE_WORD_MELSPEC_MODEL_PATH", BASE_DIR / "models" / "melspectrogram.onnx"
)
WAKE_WORD_EMBEDDING_MODEL_PATH = _env_path(
    "WAKE_WORD_EMBEDDING_MODEL_PATH", BASE_DIR / "models" / "embedding_model.onnx"
)
WAKE_WORD_TARGET_SAMPLE_RATE = 16000
WAKE_WORD_SCORE_THRESHOLD = float(os.getenv("WAKE_WORD_SCORE_THRESHOLD", "0.5"))
WAKE_WORD_CONSECUTIVE_FRAMES = int(os.getenv("WAKE_WORD_CONSECUTIVE_FRAMES", "2"))
PREROLL_DURATION_SECONDS = float(os.getenv("PREROLL_DURATION_SECONDS", "1.0"))
FORCE_ALWAYS_ON = _env_bool("FORCE_ALWAYS_ON", default=False)

# Validation
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )
