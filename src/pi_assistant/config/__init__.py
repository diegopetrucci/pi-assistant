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
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", _ASSISTANT.get("model", "gpt-5.1-2025-11-13"))
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
_WAKE_MODELS: dict[str, dict[str, object]] = _WAKE.get("models", {}) or {}


def _normalize_wake_word_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def _wake_word_label(config: dict[str, object]) -> str:
    label = config.get("phrase") or config.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return ""


def _match_wake_word_name(raw: str | None) -> str | None:
    if not raw:
        return None
    normalized = _normalize_wake_word_token(raw)
    for key, data in _WAKE_MODELS.items():
        candidates = [key, key.removeprefix("hey_"), _wake_word_label(data)]
        aliases = data.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(alias for alias in aliases if isinstance(alias, str))
        for candidate in candidates:
            if not candidate:
                continue
            if _normalize_wake_word_token(candidate) == normalized:
                return key
    return None


def _default_wake_word_name() -> str:
    configured_default = _WAKE.get("default_model_name")
    if isinstance(configured_default, str):
        normalized = _normalize_wake_word_token(configured_default)
        for key in _WAKE_MODELS:
            if _normalize_wake_word_token(key) == normalized:
                return key
    if _WAKE_MODELS:
        return next(iter(_WAKE_MODELS))
    return "hey_rhasspy"


def _prompt_for_wake_word_choice(default_name: str) -> str | None:
    if not sys.stdin.isatty() or not _WAKE_MODELS:
        return None

    options = list(_WAKE_MODELS.items())
    sys.stderr.write(
        "\nChoose a wake phrase so the Pi knows when to start listening.\n"
        "Say the phrase before speaking to the assistant.\n"
    )
    default_label = _wake_word_label(_WAKE_MODELS.get(default_name, {})) or default_name
    for idx, (key, data) in enumerate(options, start=1):
        label = _wake_word_label(data) or key.replace("_", " ").title()
        default_suffix = " (default)" if key == default_name else ""
        sys.stderr.write(f"  {idx}) {label}{default_suffix}\n")

    while True:
        try:
            choice = input(f"Wake phrase [{default_label}]: ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive prompt
            sys.stderr.write("\nNo wake phrase selected; using default.\n")
            return None

        if not choice:
            return default_name
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        match = _match_wake_word_name(choice)
        if match:
            return match
        sys.stderr.write("Invalid selection. Enter the number or name from the list above.\n")


def _persist_wake_word_choice(name: str) -> None:
    _persist_env_value("WAKE_WORD_NAME", name)
    os.environ["WAKE_WORD_NAME"] = name
    sys.stderr.write(f"Saved WAKE_WORD_NAME={name} to .env\n\n")


def _resolve_wake_word_name() -> str:
    default_choice = _default_wake_word_name()
    raw_choice = os.getenv("WAKE_WORD_NAME")
    env_choice = _match_wake_word_name(raw_choice)
    if env_choice:
        # Normalize the in-memory env var so downstream code sees the canonical key.
        os.environ["WAKE_WORD_NAME"] = env_choice
        return env_choice

    if raw_choice:
        sys.stderr.write(f"Unsupported WAKE_WORD_NAME={raw_choice} – falling back to default.\n")

    prompted = _prompt_for_wake_word_choice(default_choice)
    if prompted:
        _persist_wake_word_choice(prompted)
        return prompted
    return default_choice


WAKE_WORD_NAME = _resolve_wake_word_name()
_SELECTED_WAKE = _WAKE_MODELS.get(WAKE_WORD_NAME, {})


def _wake_path_from_config(key: str, fallback: str) -> str:
    value = _SELECTED_WAKE.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return fallback


_DEFAULT_WAKE_MODEL_PATH = _wake_path_from_config("model_path", _WAKE["model_path"])
_DEFAULT_WAKE_FALLBACK_PATH = _wake_path_from_config(
    "fallback_model_path", _WAKE["fallback_model_path"]
)
WAKE_WORD_PHRASE = _wake_word_label(_SELECTED_WAKE) or WAKE_WORD_NAME.replace("_", " ").title()

WAKE_WORD_MODEL_PATH = _env_path("WAKE_WORD_MODEL_PATH", _DEFAULT_WAKE_MODEL_PATH)
WAKE_WORD_MODEL_FALLBACK_PATH = _env_path(
    "WAKE_WORD_MODEL_FALLBACK_PATH", _DEFAULT_WAKE_FALLBACK_PATH
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
