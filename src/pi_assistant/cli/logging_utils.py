"""ANSI log labels and helpers for CLI output."""

from __future__ import annotations

import atexit
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

from pi_assistant.config import VERBOSE_LOG_CAPTURE_ENABLED, VERBOSE_LOG_DIRECTORY
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
_VERBOSE_LOG_FILE: Optional[TextIO] = None
_VERBOSE_LOG_PATH: Optional[Path] = None
_VERBOSE_LOG_ERROR_REPORTED = False
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_AUTO_CONFIGURE_PENDING = bool(VERBOSE_LOG_CAPTURE_ENABLED and VERBOSE_LOG_DIRECTORY is not None)
_AUTO_CONFIGURED = False


def _close_verbose_log() -> None:
    """Close the on-disk verbose log if it is open."""

    global _VERBOSE_LOG_FILE, _VERBOSE_LOG_PATH
    if _VERBOSE_LOG_FILE is not None:
        try:
            _VERBOSE_LOG_FILE.close()
        except Exception:
            pass
        finally:
            _VERBOSE_LOG_FILE = None
            _VERBOSE_LOG_PATH = None


atexit.register(_close_verbose_log)


def configure_verbose_log_capture(
    destination: str | Path | None, *, per_session: bool = False
) -> None:
    """Enable or disable persistent capture of verbose logs."""

    global _VERBOSE_LOG_FILE, _VERBOSE_LOG_PATH, _VERBOSE_LOG_ERROR_REPORTED
    global _AUTO_CONFIGURE_PENDING, _AUTO_CONFIGURED

    _AUTO_CONFIGURE_PENDING = False
    _AUTO_CONFIGURED = destination is not None
    _close_verbose_log()
    if destination is None:
        _VERBOSE_LOG_PATH = None
        return

    candidate = Path(destination)
    try:
        if per_session:
            candidate.mkdir(parents=True, exist_ok=True)
            log_path = _build_session_log_path(candidate)
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            log_path = candidate

        _VERBOSE_LOG_FILE = log_path.open("a", encoding="utf-8")
        _VERBOSE_LOG_PATH = log_path
        _VERBOSE_LOG_ERROR_REPORTED = False
    except OSError as exc:
        _VERBOSE_LOG_PATH = None
        if not _VERBOSE_LOG_ERROR_REPORTED:
            sys.stderr.write(f"Unable to open verbose log file at {candidate}: {exc}\n")
            _VERBOSE_LOG_ERROR_REPORTED = True


def _build_session_log_path(directory: Path) -> Path:
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    safe_timestamp = timestamp.replace(":", "-")
    base_path = directory / f"{safe_timestamp}.log"

    try:
        base_path.open("x", encoding="utf-8").close()
        return base_path
    except FileExistsError:
        pass

    counter = 1
    while True:
        candidate = directory / f"{safe_timestamp}_{counter}.log"
        try:
            candidate.open("x", encoding="utf-8").close()
            return candidate
        except FileExistsError:
            counter += 1


def _log_capture_active() -> bool:
    return _VERBOSE_LOG_FILE is not None


def current_verbose_log_path() -> Optional[Path]:
    """Return the active verbose log path, if capture is enabled."""

    _ensure_auto_configured()
    return _VERBOSE_LOG_PATH


def _ensure_auto_configured() -> None:
    """Lazily configure verbose log capture when defaults request it."""

    global _AUTO_CONFIGURED
    if not _AUTO_CONFIGURE_PENDING or _AUTO_CONFIGURED:
        return

    configure_verbose_log_capture(VERBOSE_LOG_DIRECTORY, per_session=True)
    _AUTO_CONFIGURED = True


def _compose_verbose_line(timestamp: str, args: tuple[object, ...], sep: str) -> str:
    if not args:
        return f"[{timestamp}]"

    first, *rest = args
    segments = [f"[{timestamp}] {first}"]
    if rest:
        segments.extend(str(value) for value in rest)
    return sep.join(segments)


def _write_verbose_log_entry(timestamp: str, args: tuple[object, ...], sep: str, end: str) -> None:
    global _VERBOSE_LOG_ERROR_REPORTED

    if _VERBOSE_LOG_FILE is None:
        return

    line = _compose_verbose_line(timestamp, args, sep)
    clean_line = _ANSI_ESCAPE_RE.sub("", line)
    try:
        _VERBOSE_LOG_FILE.write(clean_line + end)
        _VERBOSE_LOG_FILE.flush()
    except OSError as exc:
        if not _VERBOSE_LOG_ERROR_REPORTED:
            sys.stderr.write(f"Unable to write to verbose log file: {exc}\n")
            _VERBOSE_LOG_ERROR_REPORTED = True


def _format_timestamp() -> str:
    """Return the current time formatted as mm:ss.mmm (minutes:seconds.milliseconds)."""

    now = datetime.now()
    return f"{now:%M:%S}.{now.microsecond // 1000:03d}"


def set_verbose_logging(enabled: bool) -> None:
    """Toggle verbose logging for CLI helpers."""

    global _VERBOSE_LOGGING
    _VERBOSE_LOGGING = bool(enabled)


def is_verbose_logging_enabled() -> bool:
    """Return True if verbose logging is active."""

    return _VERBOSE_LOGGING


def verbose_print(*args, **kwargs) -> None:
    """Print when verbose logging is enabled and optionally persist to disk."""

    _ensure_auto_configured()
    timestamp = _format_timestamp()
    if _log_capture_active():
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        _write_verbose_log_entry(timestamp, args, sep, end)

    if not _VERBOSE_LOGGING:
        return

    if args:
        first, *rest = args
        print(f"[{timestamp}] {first}", *rest, **kwargs)
    else:
        print(f"[{timestamp}]", **kwargs)


def log_state_transition(previous: Optional[StreamState], new: StreamState, reason: str) -> None:
    """Emit a consistent log for controller state changes."""

    _ensure_auto_configured()
    if not _VERBOSE_LOGGING and not _log_capture_active():
        return

    if previous == new:
        return

    if previous is None:
        verbose_print(f"{STATE_LOG_LABEL} Entered {new.value.upper()} ({reason})")
    else:
        verbose_print(
            f"{STATE_LOG_LABEL} {previous.value.upper()} -> {new.value.upper()} ({reason})"
        )
