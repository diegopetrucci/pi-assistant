"""
Configuration settings for the Pi transcription client.

Defaults live in ``config/defaults.toml`` and can be overridden via environment
variables or CLI flags.
"""

from __future__ import annotations

import copy
import os
import sys
from getpass import getpass
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for 3.10 environments
    import tomli as tomllib  # type: ignore[import]

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.toml"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

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
STREAM_SAMPLE_RATE = _env_int(
    "STREAM_SAMPLE_RATE", _AUDIO.get("stream_sample_rate", _AUDIO["sample_rate"])
)
BUFFER_SIZE = _env_int("BUFFER_SIZE", _AUDIO["buffer_size"])
CHANNELS = _env_int("CHANNELS", _AUDIO["channels"])
DTYPE = os.getenv("DTYPE", _AUDIO["dtype"])
AUDIO_INPUT_DEVICE = os.getenv("AUDIO_INPUT_DEVICE")
AUDIO_QUEUE_MAX_SIZE = _env_int("AUDIO_QUEUE_MAX_SIZE", _AUDIO["queue_max_size"])
PREROLL_DURATION_SECONDS = _env_float(
    "PREROLL_DURATION_SECONDS", _AUDIO["preroll_duration_seconds"]
)


# OpenAI API Configuration
def _prompt_for_api_key() -> str | None:
    """Interactively request and persist the OpenAI API key when missing."""

    if not sys.stdin.isatty():  # Non-interactive session (CI, tests, etc.)
        return None

    sys.stderr.write(
        "\nOPENAI_API_KEY is missing. Paste your OpenAI API key to store it in .env:\n"
    )
    try:
        api_key = getpass("OpenAI API key: ").strip()
    except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive prompt
        sys.stderr.write("\nNo API key provided; aborting.\n")
        return None

    if not api_key:
        sys.stderr.write("Empty API key provided; aborting.\n")
        return None

    _persist_api_key(api_key)
    os.environ["OPENAI_API_KEY"] = api_key
    sys.stderr.write("Saved API key to .env\n\n")
    return api_key


def _persist_env_value(key: str, value: str) -> None:
    """Write or update a key=value entry in the repo's .env file."""

    existing_lines: list[str] = []
    replaced = False

    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    for line in existing_lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(f"{key}={value}")

    ENV_PATH.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _persist_api_key(api_key: str) -> None:
    """Write or update OPENAI_API_KEY in the repo's .env file."""

    _persist_env_value("OPENAI_API_KEY", api_key)


_ASSISTANT_MODEL_CHOICES: dict[str, dict[str, str]] = {
    "mini": {
        "value": "gpt-5-mini-2025-08-07",
        "description": "Mini - faster (~2s per reply), less precise (default)",
    },
    "5.1": {
        "value": "gpt-5.1-2025-11-13",
        "description": "5.1 - slower (~5s per reply), more precise",
    },
}


def _prompt_for_assistant_model(default_model: str) -> str | None:
    """Interactive assistant model selection shown on first run."""

    if not sys.stdin.isatty():
        return None

    default_key = next(
        (key for key, data in _ASSISTANT_MODEL_CHOICES.items() if data["value"] == default_model),
        "mini",
    )

    sys.stderr.write(
        "\nChoose which assistant model to use. This affects how fast and detailed replies feel:\n"
    )
    for key, data in _ASSISTANT_MODEL_CHOICES.items():
        suffix = " (default)" if key == default_key else ""
        sys.stderr.write(f"  {key}: {data['description']}{suffix}\n")

    prompt = (
        f"Assistant model [{'/'.join(_ASSISTANT_MODEL_CHOICES.keys())}] (default {default_key}): "
    )
    try:
        choice = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.stderr.write("\nNo selection provided; keeping the default.\n")
        return None

    if not choice:
        choice = default_key

    normalized = choice.replace(" ", "")
    if normalized in {"mini", "fast"}:
        choice_key = "mini"
    elif normalized in {"5.1", "5", "full"}:
        choice_key = "5.1"
    else:
        choice_key = normalized if normalized in _ASSISTANT_MODEL_CHOICES else None

    if choice_key is None:
        sys.stderr.write("Unrecognized choice; defaulting to Mini.\n")
        choice_key = "mini"

    return _ASSISTANT_MODEL_CHOICES[choice_key]["value"]


def _resolve_assistant_model(default_model: str) -> str:
    """Return the assistant model, prompting/persisting on first launch."""

    env_model = os.getenv("ASSISTANT_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    prompted_model = _prompt_for_assistant_model(default_model)
    if prompted_model:
        _persist_env_value("ASSISTANT_MODEL", prompted_model)
        os.environ["ASSISTANT_MODEL"] = prompted_model
        sys.stderr.write("Saved ASSISTANT_MODEL to .env\n\n")
        return prompted_model

    return default_model


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or _prompt_for_api_key()
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
_DEFAULT_ASSISTANT_MODEL = _ASSISTANT.get("model", _ASSISTANT_MODEL_CHOICES["mini"]["value"])
ASSISTANT_MODEL = _resolve_assistant_model(_DEFAULT_ASSISTANT_MODEL)
ASSISTANT_SYSTEM_PROMPT = os.getenv("ASSISTANT_SYSTEM_PROMPT", _ASSISTANT.get("system_prompt", ""))
ASSISTANT_WEB_SEARCH_ENABLED = _env_bool(
    "ASSISTANT_WEB_SEARCH_ENABLED", _ASSISTANT.get("web_search_enabled", True)
)
ASSISTANT_TTS_ENABLED = _env_bool("ASSISTANT_TTS_ENABLED", _ASSISTANT.get("tts_enabled", True))
ASSISTANT_TTS_RESPONSES_ENABLED = _env_bool(
    "ASSISTANT_TTS_RESPONSES_ENABLED", _ASSISTANT.get("tts_responses_enabled", True)
)
ASSISTANT_TTS_MODEL = os.getenv(
    "ASSISTANT_TTS_MODEL", _ASSISTANT.get("tts_model", "gpt-4o-mini-tts")
)
ASSISTANT_TTS_VOICE = os.getenv("ASSISTANT_TTS_VOICE", _ASSISTANT.get("tts_voice", "alloy"))
ASSISTANT_TTS_FORMAT = os.getenv("ASSISTANT_TTS_FORMAT", _ASSISTANT.get("tts_format", "pcm"))
ASSISTANT_TTS_SAMPLE_RATE = _env_int(
    "ASSISTANT_TTS_SAMPLE_RATE", _ASSISTANT.get("tts_sample_rate", 24000)
)

_DEVICE = _DEFAULTS.get("device", {})


def _prompt_for_location_name() -> str | None:
    """Request a location name (city/region) for contextual replies."""

    if not sys.stdin.isatty():
        return None

    sys.stderr.write(
        "\nLOCATION_NAME is missing. Provide a city or region so the assistant knows "
        "where this Pi is located.\n"
    )
    try:
        location = input("Device location (e.g., London, UK): ").strip()
    except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive prompt
        sys.stderr.write("\nNo location provided; leaving LOCATION_NAME empty.\n")
        return None

    if not location:
        sys.stderr.write("Empty location provided; leaving LOCATION_NAME empty.\n")
        return None

    _persist_env_value("LOCATION_NAME", location)
    os.environ["LOCATION_NAME"] = location
    sys.stderr.write("Saved LOCATION_NAME to .env\n\n")
    return location


def _resolve_location_name() -> str:
    env_location = os.getenv("LOCATION_NAME")
    if env_location and env_location.strip():
        return env_location.strip()
    default_location = _DEVICE.get("location_name", "").strip()
    if default_location:
        return default_location
    prompted = _prompt_for_location_name()
    return (prompted or "").strip()


LOCATION_NAME = _resolve_location_name()

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
        "OPENAI_API_KEY not configured. "
        "Set the variable manually or rerun in an interactive shell to supply it."
    )
