"""
Shared configuration helpers and non-assistant settings for the Pi assistant.

Defaults live in ``config/defaults.toml`` and can be overridden via environment
variables or CLI flags.
"""

from __future__ import annotations

import copy
import os
import sys
from getpass import getpass
from pathlib import Path

import tomllib
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


def _coerce_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _env_bool(name: str, default: bool = False) -> bool:
    """Return True when the env var is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        _warn_invalid_env_value(name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        _warn_invalid_env_value(name, value, default)
        return default


def _env_path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    return _coerce_path(raw)


def _normalize_language(value: str | None, fallback: str = "en") -> str:
    """Return a normalized language tag, defaulting to ``fallback`` when empty."""

    if value is None:
        return fallback
    trimmed = value.strip()
    return trimmed or fallback


def _persist_env_value(key: str, value: str) -> bool:
    """Write or update a key=value entry in the repo's .env file, returning True on success."""

    existing_lines: list[str] = []
    replaced = False

    if ENV_PATH.exists():
        try:
            existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            sys.stderr.write(f"Unable to read {ENV_PATH}: {exc}\n")
            return False

    new_lines: list[str] = []
    for line in existing_lines:
        if line.startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(f"{key}={value}")

    contents = "\n".join(new_lines).rstrip()
    try:
        ENV_PATH.write_text((contents + "\n") if contents else "\n", encoding="utf-8")
    except OSError as exc:
        sys.stderr.write(f"Unable to write {ENV_PATH}: {exc}\n")
        return False
    return True


def _remove_env_keys(keys: tuple[str, ...]) -> set[str]:
    """Remove specified keys from .env and return the ones that existed."""

    if not keys:
        return set()

    existing_lines: list[str] = []
    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

    removed: set[str] = set()
    trimmed_lines: list[str] = []
    for line in existing_lines:
        matched = False
        for key in keys:
            if line.startswith(f"{key}="):
                removed.add(key)
                matched = True
                break
        if not matched:
            trimmed_lines.append(line)

    if existing_lines:
        new_contents = "\n".join(trimmed_lines).rstrip()
        ENV_PATH.write_text((new_contents + "\n") if new_contents else "", encoding="utf-8")

    for key in keys:
        os.environ.pop(key, None)

    return removed


def _warn_invalid_env_value(name: str, value: str | None, default: object) -> None:
    """Emit a warning when env overrides cannot be parsed."""

    sys.stderr.write(f"Invalid value for {name}={value!r}; falling back to {default!r}.\n")


_FIRST_LAUNCH_ENV_KEYS: tuple[str, ...] = (
    "ASSISTANT_MODEL",
    "ASSISTANT_REASONING_EFFORT",
    "LOCATION_NAME",
)


def reset_first_launch_choices() -> set[str]:
    """
    Clear saved selections that were captured via first-launch prompts.

    Returns the subset of keys that were removed from the environment.
    """

    return _remove_env_keys(_FIRST_LAUNCH_ENV_KEYS)


def _persist_api_key(api_key: str) -> None:
    """Write or update OPENAI_API_KEY in the repo's .env file."""

    _persist_env_value("OPENAI_API_KEY", api_key)


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
AUDIO_DEBUG_DUMP_ENABLED = _env_bool(
    "AUDIO_DEBUG_DUMP_ENABLED", _AUDIO.get("debug_dump_enabled", False)
)
_DEBUG_DUMP_DEFAULT = _AUDIO.get("debug_dump_directory", "logs/audio_dumps")
AUDIO_DEBUG_DUMP_DIRECTORY = _env_path(
    "AUDIO_DEBUG_DUMP_DIRECTORY",
    _DEBUG_DUMP_DEFAULT,
)


# OpenAI API Configuration
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
_INPUT_TRANSCRIPTION_DEFAULTS = _DEFAULTS["session"]["input_audio_transcription"]
TRANSCRIPTION_LANGUAGE = _normalize_language(
    os.getenv("TRANSCRIPTION_LANGUAGE"),
    _INPUT_TRANSCRIPTION_DEFAULTS.get("language", "en"),
)
SESSION_CONFIG["input_audio_transcription"]["language"] = TRANSCRIPTION_LANGUAGE

_DEVICE = _DEFAULTS.get("device", {})


def _prompt_for_location_name() -> str | None:
    """Request a location name (city/region) for contextual replies."""

    if not sys.stdin.isatty():
        return None

    sys.stderr.write(
        "\nLOCATION_NAME is missing. Provide a city or region so the assistant knows "
        "where this Pi is located.\n"
    )
    while True:
        try:
            location = input("Device location (e.g., London, UK): ").strip()
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive prompt
            sys.stderr.write("\nNo location provided; leaving LOCATION_NAME empty.\n")
            return None

        if location:
            break

        sys.stderr.write("LOCATION_NAME is required. Enter a city or region (Ctrl+C to cancel).\n")

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

_UX = _DEFAULTS.get("ux", {})
CONFIRMATION_CUE_ENABLED = _env_bool(
    "CONFIRMATION_CUE_ENABLED", _UX.get("confirmation_cue_enabled", True)
)
_CONFIRMATION_CUE_DEFAULT_TEXT = _UX.get("confirmation_cue_text", "Got it.")
CONFIRMATION_CUE_TEXT = (
    os.getenv("CONFIRMATION_CUE_TEXT") or _CONFIRMATION_CUE_DEFAULT_TEXT or ""
).strip()

_LOGGING = _DEFAULTS.get("logging", {})
VERBOSE_LOG_CAPTURE_ENABLED = _env_bool(
    "VERBOSE_LOG_CAPTURE_ENABLED", _LOGGING.get("verbose_capture_enabled", False)
)
if VERBOSE_LOG_CAPTURE_ENABLED:
    _DEFAULT_VERBOSE_DIR = _LOGGING.get("verbose_log_directory")
    default_verbose_dir = (
        _DEFAULT_VERBOSE_DIR.strip()
        if isinstance(_DEFAULT_VERBOSE_DIR, str) and _DEFAULT_VERBOSE_DIR.strip()
        else "logs"
    )

    VERBOSE_LOG_DIRECTORY = _env_path("VERBOSE_LOG_DIRECTORY", default_verbose_dir)
else:
    VERBOSE_LOG_DIRECTORY = None

# Auto-stop Configuration
_AUTO_STOP = _DEFAULTS["auto_stop"]
AUTO_STOP_ENABLED = _env_bool("AUTO_STOP_ENABLED", _AUTO_STOP["enabled"])
AUTO_STOP_SILENCE_THRESHOLD = _env_int(
    "AUTO_STOP_SILENCE_THRESHOLD", _AUTO_STOP["silence_threshold"]
)
AUTO_STOP_MAX_SILENCE_SECONDS = _env_float(
    "AUTO_STOP_MAX_SILENCE_SECONDS", _AUTO_STOP["max_silence_seconds"]
)
SERVER_STOP_MIN_SILENCE_SECONDS = _env_float(
    "SERVER_STOP_MIN_SILENCE_SECONDS", _AUTO_STOP.get("server_stop_min_silence_seconds", 0.75)
)

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not configured. "
        "Set the variable manually or rerun in an interactive shell to supply it."
    )

__all__ = [
    "PROJECT_ROOT",
    "DEFAULTS_PATH",
    "ENV_PATH",
    "_DEFAULTS",
    "_coerce_path",
    "_env_bool",
    "_env_int",
    "_env_float",
    "_env_path",
    "_normalize_language",
    "_persist_env_value",
    "_persist_api_key",
    "_prompt_for_api_key",
    "_prompt_for_location_name",
    "_resolve_location_name",
    "SAMPLE_RATE",
    "STREAM_SAMPLE_RATE",
    "BUFFER_SIZE",
    "CHANNELS",
    "DTYPE",
    "AUDIO_INPUT_DEVICE",
    "AUDIO_QUEUE_MAX_SIZE",
    "PREROLL_DURATION_SECONDS",
    "AUDIO_DEBUG_DUMP_ENABLED",
    "AUDIO_DEBUG_DUMP_DIRECTORY",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_REALTIME_ENDPOINT",
    "WEBSOCKET_HEADERS",
    "SESSION_CONFIG",
    "TRANSCRIPTION_LANGUAGE",
    "LOCATION_NAME",
    "CONFIRMATION_CUE_ENABLED",
    "CONFIRMATION_CUE_TEXT",
    "VERBOSE_LOG_CAPTURE_ENABLED",
    "VERBOSE_LOG_DIRECTORY",
    "AUTO_STOP_ENABLED",
    "AUTO_STOP_SILENCE_THRESHOLD",
    "AUTO_STOP_MAX_SILENCE_SECONDS",
    "SERVER_STOP_MIN_SILENCE_SECONDS",
    "reset_first_launch_choices",
]
