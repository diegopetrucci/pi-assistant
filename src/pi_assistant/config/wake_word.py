"""Wake-word configuration helpers."""

from __future__ import annotations

import os
import sys

from .base import _DEFAULTS, _env_float, _env_int, _env_path, _persist_env_value

_WAKE = _DEFAULTS["wake_word"]
_WAKE_MODELS: dict[str, dict[str, object]] = _WAKE.get("models", {}) or {}


def _normalize_wake_word_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def _wake_word_label(config: dict[str, object]) -> str:
    label = config.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    phrase = config.get("phrase")
    if isinstance(phrase, str) and phrase.strip():
        return phrase.strip()
    aliases = config.get("aliases", [])
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                return alias.strip()
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

__all__ = [
    "WAKE_WORD_CONSECUTIVE_FRAMES",
    "WAKE_WORD_EMBEDDING_MODEL_PATH",
    "WAKE_WORD_MELSPEC_MODEL_PATH",
    "WAKE_WORD_MODEL_FALLBACK_PATH",
    "WAKE_WORD_MODEL_PATH",
    "WAKE_WORD_NAME",
    "WAKE_WORD_PHRASE",
    "WAKE_WORD_SCORE_THRESHOLD",
    "WAKE_WORD_TARGET_SAMPLE_RATE",
]
