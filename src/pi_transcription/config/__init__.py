"""
Configuration settings for the Pi transcription client.

Defaults live in ``config/defaults.toml`` and can be overridden via environment
variables or CLI flags.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10 environments
    import tomli as tomllib  # type: ignore[import]

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.toml"

if not DEFAULTS_PATH.exists():  # pragma: no cover - configuration issue
    raise FileNotFoundError(
        f"Missing configuration defaults at {DEFAULTS_PATH}. Ensure config/defaults.toml exists."
    )

with DEFAULTS_PATH.open("rb") as defaults_file:
    _DEFAULTS = tomllib.load(defaults_file)


def _env_bool(name: str, default: bool = False) -> bool:
    """Return True when the env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


# Audio Configuration
_AUDIO = _DEFAULTS["audio"]
SAMPLE_RATE = _env_int("SAMPLE_RATE", _AUDIO["sample_rate"])
BUFFER_SIZE = _env_int("BUFFER_SIZE", _AUDIO["buffer_size"])
CHANNELS = _env_int("CHANNELS", _AUDIO["channels"])
DTYPE = os.getenv("DTYPE", _AUDIO["dtype"])
AUDIO_QUEUE_MAX_SIZE = _env_int("AUDIO_QUEUE_MAX_SIZE", _AUDIO["queue_max_size"])
PREROLL_DURATION_SECONDS = _env_float(
    "PREROLL_DURATION_SECONDS", _AUDIO["preroll_duration_seconds"]
)

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", _DEFAULTS["openai"]["model"])
OPENAI_REALTIME_ENDPOINT = os.getenv(
    "OPENAI_REALTIME_ENDPOINT", _DEFAULTS["openai"]["realtime_endpoint"]
)

WEBSOCKET_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": _DEFAULTS["openai"]["beta_header"],
}

# Session Configuration for OpenAI Realtime API (Transcription mode)
SESSION_CONFIG = copy.deepcopy(_DEFAULTS["session"])
SESSION_CONFIG["input_audio_transcription"]["model"] = OPENAI_MODEL

# Assistant / LLM Configuration
_ASSISTANT = _DEFAULTS.get("assistant", {})
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", _ASSISTANT.get("model", "gpt-5.1-2025-11-13"))
ASSISTANT_WEB_SEARCH_ENABLED = _env_bool(
    "ASSISTANT_WEB_SEARCH_ENABLED", _ASSISTANT.get("web_search_enabled", True)
)

# Auto-stop Configuration
_AUTO_STOP = _DEFAULTS["auto_stop"]
AUTO_STOP_ENABLED = _env_bool("AUTO_STOP_ENABLED", _AUTO_STOP["enabled"])
AUTO_STOP_SILENCE_THRESHOLD = _env_int(
    "AUTO_STOP_SILENCE_THRESHOLD", _AUTO_STOP["silence_threshold"]
)
AUTO_STOP_MAX_SILENCE_SECONDS = _env_float(
    "AUTO_STOP_MAX_SILENCE_SECONDS", _AUTO_STOP["max_silence_seconds"]
)

# Wake-word / gating configuration
_WAKE = _DEFAULTS["wake_word"]
WAKE_WORD_MODEL_PATH = _env_path("WAKE_WORD_MODEL_PATH", _WAKE["model_path"])
WAKE_WORD_MODEL_FALLBACK_PATH = _env_path(
    "WAKE_WORD_MODEL_FALLBACK_PATH", _WAKE["fallback_model_path"]
)
WAKE_WORD_MELSPEC_MODEL_PATH = _env_path(
    "WAKE_WORD_MELSPEC_MODEL_PATH", _WAKE["melspec_model_path"]
)
WAKE_WORD_EMBEDDING_MODEL_PATH = _env_path(
    "WAKE_WORD_EMBEDDING_MODEL_PATH", _WAKE["embedding_model_path"]
)
WAKE_WORD_TARGET_SAMPLE_RATE = _env_int("WAKE_WORD_TARGET_SAMPLE_RATE", _WAKE["target_sample_rate"])
WAKE_WORD_SCORE_THRESHOLD = _env_float("WAKE_WORD_SCORE_THRESHOLD", _WAKE["score_threshold"])
WAKE_WORD_CONSECUTIVE_FRAMES = _env_int("WAKE_WORD_CONSECUTIVE_FRAMES", _WAKE["consecutive_frames"])

FORCE_ALWAYS_ON = _env_bool("FORCE_ALWAYS_ON", default=False)

# Validation
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )
