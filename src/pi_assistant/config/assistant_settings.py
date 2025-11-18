"""Assistant/LLM configuration helpers."""

from __future__ import annotations

import os
import sys

from .base import (
    _DEFAULTS,
    TRANSCRIPTION_LANGUAGE,
    _env_bool,
    _env_int,
    _normalize_language,
    _persist_env_value,
)

_ASSISTANT = _DEFAULTS.get("assistant", {})
_TESTING = _DEFAULTS.get("testing", {})


def _coerce_assistant_model_key(choice: str | None) -> str | None:
    if not choice:
        return None
    normalized = choice.strip().lower().replace(" ", "")
    if not normalized:
        return None
    alias_map = {
        "nano": "nano",
        "mini": "mini",
        "fast": "mini",
        "5.1": "5.1",
        "5": "5.1",
        "full": "5.1",
    }
    mapped = alias_map.get(normalized)
    if mapped:
        return mapped
    for key in _ASSISTANT_MODEL_CHOICES:
        if normalized == key.replace(" ", "").lower():
            return key
    return None


_ASSISTANT_MODEL_CHOICES: dict[str, dict[str, object]] = {
    "mini": {
        "value": "gpt-5-mini-2025-08-07",
        "description": "Mini - faster (~2s per reply), less precise (default)",
        "reasoning_efforts": ("minimal", "low", "medium", "high"),
    },
    "nano": {
        "value": "gpt-5-nano-2025-08-07",
        "description": "Nano - newest ultra-fast tier (experimental reasoning support)",
        "reasoning_efforts": ("low", "medium", "high"),
    },
    "5.1": {
        "value": "gpt-5.1-2025-11-13",
        "description": "5.1 - slower (~5s per reply), more precise",
        "reasoning_efforts": ("none", "low", "medium", "high"),
    },
}

ASSISTANT_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    key: {
        "id": str(data["value"]),
        "description": str(data["description"]),
    }
    for key, data in _ASSISTANT_MODEL_CHOICES.items()
}

_REASONING_EFFORT_DESCRIPTIONS: dict[str, str] = {
    "none": "None - GPT-5.1 only; zero reasoning tokens for lowest latency.",
    "minimal": "Minimal - GPT-5 Mini's lowest option; prioritizes speed.",
    "low": "Low - recommended default; balances speed with short reasoning bursts.",
    "medium": "Medium - more deliberate replies, slightly higher latency.",
    "high": "High - maximum reasoning tokens for the most thorough replies.",
}

_REASONING_EFFORT_ALIASES: dict[str, str] = {
    "fast": "none",
    "zero": "none",
    "default": "low",
    "balanced": "low",
    "med": "medium",
    "mid": "medium",
    "hi": "high",
    "max": "high",
    "minimum": "minimal",
    "mini": "minimal",
    "min": "minimal",
}


def _normalize_reasoning_effort(
    value: str | None, allowed_choices: tuple[str, ...] | None = None
) -> str | None:
    """Return a normalized reasoning effort level or None when invalid."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized)
    allowed = allowed_choices or tuple(_REASONING_EFFORT_DESCRIPTIONS.keys())
    if normalized in allowed:
        return normalized
    return None


def _prompt_for_reasoning_effort(
    default_choice: str, allowed_choices: tuple[str, ...]
) -> str | None:
    """Interactively ask which reasoning effort to use on first launch."""

    if not sys.stdin.isatty():
        return None

    normalized_default = _normalize_reasoning_effort(default_choice, allowed_choices) or (
        allowed_choices[0] if allowed_choices else "low"
    )
    sys.stderr.write(
        "\nChoose how much reasoning the assistant should use. "
        "Lower levels respond faster, while higher levels allocate more reasoning tokens.\n"
    )
    for key in allowed_choices:
        description = _REASONING_EFFORT_DESCRIPTIONS.get(key, key.capitalize())
        suffix = " (default)" if key == normalized_default else ""
        sys.stderr.write(f"  {key}: {description}{suffix}\n")

    options = "/".join(allowed_choices)
    prompt = f"Reasoning effort [{options}] (default {normalized_default}): "
    try:
        choice = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive prompt
        sys.stderr.write("\nNo selection provided; keeping the default.\n")
        return None

    normalized_choice = _normalize_reasoning_effort(choice or normalized_default, allowed_choices)
    if normalized_choice is None:
        sys.stderr.write(f"Unrecognized choice; defaulting to {normalized_default}.\n")
        normalized_choice = normalized_default
    return normalized_choice


def reasoning_effort_choices_for_model(model_name: str) -> tuple[str, ...]:
    """Return the reasoning levels supported by the selected model."""

    for entry in _ASSISTANT_MODEL_CHOICES.values():
        if entry.get("value") == model_name:
            configured = entry.get("reasoning_efforts")
            if isinstance(configured, (list, tuple)) and configured:
                return tuple(str(level) for level in configured)
    lowered = model_name.lower()
    if "mini" in lowered:
        return ("minimal", "low", "medium", "high")
    if "5.1" in lowered or "gpt-5.1" in lowered:
        return ("none", "low", "medium", "high")
    return ("low", "medium", "high")


def _resolve_reasoning_effort(default_choice: str, allowed_choices: tuple[str, ...]) -> str:
    """Return the configured reasoning effort, prompting/persisting when needed."""

    fallback = allowed_choices[0] if allowed_choices else "low"
    normalized_default = _normalize_reasoning_effort(default_choice, allowed_choices) or fallback
    env_value = os.getenv("ASSISTANT_REASONING_EFFORT")
    normalized_env = _normalize_reasoning_effort(env_value, allowed_choices)
    if normalized_env:
        return normalized_env
    if env_value:
        sys.stderr.write(
            f"Invalid ASSISTANT_REASONING_EFFORT '{env_value}'; "
            f"allowed values: {', '.join(allowed_choices)}. "
            f"Using {normalized_default} instead.\n"
        )

    prompted = _prompt_for_reasoning_effort(normalized_default, allowed_choices)
    if prompted:
        _persist_env_value("ASSISTANT_REASONING_EFFORT", prompted)
        os.environ["ASSISTANT_REASONING_EFFORT"] = prompted
        sys.stderr.write("Saved ASSISTANT_REASONING_EFFORT to .env\n\n")
        return prompted

    return normalized_default


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

    choice_key = _coerce_assistant_model_key(choice)
    if choice_key is None:
        lowered_choice = choice.strip().lower()
        for key, data in _ASSISTANT_MODEL_CHOICES.items():
            model_id = str(data["value"]).strip().lower()
            if lowered_choice == key.lower() or lowered_choice == model_id:
                choice_key = key
                break

    if choice_key is None:
        sys.stderr.write(f"Unrecognized choice; defaulting to {default_key}.\n")
        choice_key = default_key

    return str(_ASSISTANT_MODEL_CHOICES[choice_key]["value"])


def normalize_assistant_model_choice(value: str | None) -> str | None:
    """Return the canonical assistant model identifier for overrides."""

    choice_key = _coerce_assistant_model_key(value)
    if choice_key:
        return str(_ASSISTANT_MODEL_CHOICES[choice_key]["value"])
    if not value:
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    for data in _ASSISTANT_MODEL_CHOICES.values():
        model_id = str(data["value"]).strip()
        if lowered == model_id.lower():
            return model_id
    return None


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


_DEFAULT_ASSISTANT_MODEL = _ASSISTANT.get("model", _ASSISTANT_MODEL_CHOICES["mini"]["value"])
ASSISTANT_MODEL = _resolve_assistant_model(_DEFAULT_ASSISTANT_MODEL)
ASSISTANT_REASONING_CHOICES = reasoning_effort_choices_for_model(ASSISTANT_MODEL)
_DEFAULT_REASONING_EFFORT = _ASSISTANT.get("reasoning_effort", "low")
ASSISTANT_REASONING_EFFORT = _resolve_reasoning_effort(
    _DEFAULT_REASONING_EFFORT, ASSISTANT_REASONING_CHOICES
)
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
SIMULATED_QUERY_TEXT = os.getenv(
    "SIMULATED_QUERY_TEXT",
    _TESTING.get("simulate_query_text", ""),
)
ASSISTANT_LANGUAGE = _normalize_language(
    os.getenv("ASSISTANT_LANGUAGE"),
    _ASSISTANT.get("language", TRANSCRIPTION_LANGUAGE),
)

__all__ = [
    "_normalize_reasoning_effort",
    "_prompt_for_reasoning_effort",
    "ASSISTANT_LANGUAGE",
    "ASSISTANT_MODEL",
    "ASSISTANT_MODEL_REGISTRY",
    "ASSISTANT_REASONING_CHOICES",
    "ASSISTANT_REASONING_EFFORT",
    "ASSISTANT_SYSTEM_PROMPT",
    "ASSISTANT_TTS_ENABLED",
    "ASSISTANT_TTS_FORMAT",
    "ASSISTANT_TTS_MODEL",
    "ASSISTANT_TTS_RESPONSES_ENABLED",
    "ASSISTANT_TTS_SAMPLE_RATE",
    "ASSISTANT_TTS_VOICE",
    "ASSISTANT_WEB_SEARCH_ENABLED",
    "SIMULATED_QUERY_TEXT",
    "normalize_assistant_model_choice",
    "reasoning_effort_choices_for_model",
]
