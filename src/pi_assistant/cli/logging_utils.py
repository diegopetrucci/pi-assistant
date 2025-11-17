"""ANSI log labels and helpers for CLI output."""

from __future__ import annotations

from typing import Optional

from pi_assistant.wake_word import StreamState

# ANSI color codes for log labels
RESET = "\033[0m"
COLOR_ORANGE = "\033[38;5;208m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_MAGENTA = "\033[35m"
COLOR_RED = "\033[31m"

TURN_LOG_LABEL = f"{COLOR_ORANGE}[TURN]{RESET}"
TRANSCRIPT_LOG_LABEL = f"{COLOR_GREEN}[TRANSCRIPT]{RESET}"
VAD_LOG_LABEL = f"{COLOR_YELLOW}[VAD]{RESET}"
STATE_LOG_LABEL = f"{COLOR_CYAN}[STATE]{RESET}"
WAKE_LOG_LABEL = f"{COLOR_BLUE}[WAKE]{RESET}"
ASSISTANT_LOG_LABEL = f"{COLOR_MAGENTA}[ASSISTANT]{RESET}"
CONTROL_LOG_LABEL = f"{COLOR_MAGENTA}[CONTROL]{RESET}"
ERROR_LOG_LABEL = f"{COLOR_RED}[ERROR]{RESET}"

_VERBOSE_LOGGING = False


def set_verbose_logging(enabled: bool) -> None:
    """Toggle verbose logging for CLI helpers."""

    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


def is_verbose_logging_enabled() -> bool:
    """Return True if verbose logging is active."""

    return _VERBOSE_LOGGING


def verbose_print(*args, **kwargs) -> None:
    """Print only when verbose logging is enabled."""

    if not _VERBOSE_LOGGING:
        return
    print(*args, **kwargs)


def log_state_transition(previous: Optional[StreamState], new: StreamState, reason: str) -> None:
    """Emit a consistent log for controller state changes."""

    if not _VERBOSE_LOGGING:
        return

    if previous == new:
        return

    if previous is None:
        print(f"{STATE_LOG_LABEL} Entered {new.value.upper()} ({reason})")
    else:
        print(f"{STATE_LOG_LABEL} {previous.value.upper()} -> {new.value.upper()} ({reason})")
