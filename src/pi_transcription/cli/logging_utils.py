"""ANSI log labels and helpers for CLI output."""

from __future__ import annotations

from typing import Optional

from pi_transcription.wake_word import StreamState

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


def log_state_transition(previous: Optional[StreamState], new: StreamState, reason: str) -> None:
    """Emit a consistent log for controller state changes."""

    if previous == new:
        return

    if previous is None:
        print(f"{STATE_LOG_LABEL} Entered {new.value.upper()} ({reason})")
    else:
        print(f"{STATE_LOG_LABEL} {previous.value.upper()} -> {new.value.upper()} ({reason})")
