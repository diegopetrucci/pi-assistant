"""Structured logging utilities for the CLI."""

from __future__ import annotations

import atexit
import re
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TextIO, TypedDict

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from pi_assistant.config import VERBOSE_LOG_CAPTURE_ENABLED, VERBOSE_LOG_DIRECTORY
from pi_assistant.wake_word import StreamState

# ANSI color codes for console labels
RESET = "\033[0m"
COLOR_ORANGE = "\033[38;5;208m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_CYAN = "\033[36m"
COLOR_MAGENTA = "\033[35m"
COLOR_RED = "\033[31m"
COLOR_WHITE = "\033[37m"

TURN_LOG_LABEL = "TURN"
TRANSCRIPT_LOG_LABEL = "TRANSCRIPT"
VAD_LOG_LABEL = "VAD"
STATE_LOG_LABEL = "STATE"
WAKE_LOG_LABEL = "WAKE"
ASSISTANT_LOG_LABEL = "ASSISTANT"
CONTROL_LOG_LABEL = "CONTROL"
ERROR_LOG_LABEL = "ERROR"
AUDIO_LOG_LABEL = "AUDIO"
WS_LOG_LABEL = "WS"

_WS_LABELS = {
    "": WS_LOG_LABEL,
    "←": "WS←",
    "→": "WS→",
}

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class LogOptions(TypedDict, total=False):
    sep: str
    end: str
    verbose: bool
    error: bool
    flush: bool
    color: str | None
    exc_info: bool | BaseException | tuple[type[BaseException], BaseException, TracebackType | None]


EXC_INFO_TUPLE_LEN = 3
MAX_SESSION_LOG_COLLISIONS = 1000


def strip_ansi_sequences(text: str) -> str:
    """Return *text* with ANSI escape codes removed."""

    return ANSI_ESCAPE_RE.sub("", text)


def _format_exc_details(exc_info: object) -> Optional[str]:
    if not exc_info:
        return None

    if exc_info is True:
        formatted = traceback.format_exc()
        return None if formatted.strip() == "NoneType: None" else formatted.rstrip()

    if isinstance(exc_info, BaseException):
        return "".join(
            traceback.format_exception(exc_info.__class__, exc_info, exc_info.__traceback__)
        ).rstrip()

    if (
        isinstance(exc_info, tuple)
        and len(exc_info) == EXC_INFO_TUPLE_LEN
        and isinstance(exc_info[0], type)
        and issubclass(exc_info[0], BaseException)
        and isinstance(exc_info[2], (TracebackType, type(None)))
    ):
        return "".join(traceback.format_exception(*exc_info)).rstrip()

    raise TypeError("exc_info must be True, an exception instance, or (type, value, traceback)")


_LABEL_COLORS = {
    TURN_LOG_LABEL: COLOR_ORANGE,
    TRANSCRIPT_LOG_LABEL: COLOR_GREEN,
    VAD_LOG_LABEL: COLOR_YELLOW,
    STATE_LOG_LABEL: COLOR_CYAN,
    WAKE_LOG_LABEL: COLOR_BLUE,
    ASSISTANT_LOG_LABEL: COLOR_MAGENTA,
    CONTROL_LOG_LABEL: COLOR_MAGENTA,
    ERROR_LOG_LABEL: COLOR_RED,
    AUDIO_LOG_LABEL: COLOR_BLUE,
    WS_LOG_LABEL: COLOR_WHITE,
    "WS←": COLOR_WHITE,
    "WS→": COLOR_WHITE,
}


class Logger:
    """Centralized logger that enforces `[timestamp] [source] message` output."""

    def __init__(self) -> None:
        self._verbose_logging = False
        self._chunk_progress_logging = False
        self._log_file: Optional[TextIO] = None
        self._log_path: Optional[Path] = None
        self._log_error_reported = False
        self._auto_configure_pending = bool(
            VERBOSE_LOG_CAPTURE_ENABLED and VERBOSE_LOG_DIRECTORY is not None
        )
        self._auto_configured = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def close(self) -> None:
        if self._log_file is None:
            return

        try:
            self._log_file.close()
        except Exception:
            pass
        finally:
            self._log_file = None
            self._log_path = None

    def configure_verbose_log_capture(
        self, destination: str | Path | None, *, per_session: bool = False
    ) -> None:
        self._auto_configure_pending = False
        self._auto_configured = destination is not None
        self.close()
        if destination is None:
            self._log_path = None
            return

        candidate = Path(destination)
        try:
            if per_session:
                candidate.mkdir(parents=True, exist_ok=True)
                log_path = self._build_session_log_path(candidate)
            else:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                log_path = candidate

            self._log_file = log_path.open("a", encoding="utf-8")
            self._log_path = log_path
            self._log_error_reported = False
        except OSError as exc:
            self._log_path = None
            if not self._log_error_reported:
                sys.stderr.write(f"Unable to open verbose log file at {candidate}: {exc}\n")
                self._log_error_reported = True

    # ------------------------------------------------------------------
    # Configuration
    def set_verbose_logging(self, enabled: bool) -> None:
        self._verbose_logging = bool(enabled)

    def is_verbose_logging_enabled(self) -> bool:
        return self._verbose_logging

    def set_chunk_progress_logging(self, enabled: bool) -> None:
        self._chunk_progress_logging = bool(enabled)

    def is_chunk_progress_logging_enabled(self) -> bool:
        return self._chunk_progress_logging

    def current_verbose_log_path(self) -> Optional[Path]:
        self._ensure_auto_configured()
        return self._log_path

    # ------------------------------------------------------------------
    # Public logging APIs
    def log(self, source: str, *message_parts: object, **options: Unpack[LogOptions]) -> None:
        if not source:
            raise ValueError("source is required")

        cleaned_source = source.strip()
        sep = options.pop("sep", " ")
        end = options.pop("end", "\n")
        exc_info = options.pop("exc_info", None)
        verbose = bool(options.pop("verbose", False))
        error = bool(options.pop("error", False))
        flush = bool(options.pop("flush", False))
        color = options.pop("color", None)
        if options:
            unexpected = ", ".join(sorted(options))
            raise TypeError(f"Unsupported log option(s): {unexpected}")

        parts = tuple(message_parts)
        line_cache: Optional[str] = None

        def _render_line() -> str:
            nonlocal line_cache
            if line_cache is not None:
                return line_cache

            timestamp = self._format_timestamp()
            line = f"[{timestamp}] [{cleaned_source}]"
            if parts:
                message = sep.join(str(part) for part in parts)
                if message:
                    line = f"{line} {message}"

            trace_details = _format_exc_details(exc_info)
            if trace_details:
                line = f"{line}\n{trace_details}"

            line_cache = line
            return line_cache

        if verbose:
            self._ensure_auto_configured()

        should_capture = verbose and self._log_file is not None
        should_emit = not verbose or self._verbose_logging
        if verbose and not should_emit and not should_capture:
            return

        if should_capture:
            self._write_verbose_log_entry(_render_line(), end)

        color_code = color if color is not None else _LABEL_COLORS.get(cleaned_source)
        console_line = _render_line()
        if color_code:
            label_token = f"[{cleaned_source}]"
            colored_label = f"{color_code}{label_token}{RESET}"
            console_line = console_line.replace(label_token, colored_label, 1)

        if should_emit:
            stream = sys.stderr if error else sys.stdout
            stream.write(console_line)
            stream.write(end)
            if flush:
                stream.flush()

    def verbose(self, source: str, *message_parts: object, **kwargs: Any) -> None:
        kwargs["verbose"] = True
        self.log(source, *message_parts, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    def _log_capture_active(self) -> bool:
        return self._log_file is not None

    def _ensure_auto_configured(self) -> None:
        if not self._auto_configure_pending or self._auto_configured:
            return

        if VERBOSE_LOG_DIRECTORY is None:
            self._auto_configure_pending = False
            return

        self.configure_verbose_log_capture(VERBOSE_LOG_DIRECTORY, per_session=True)
        self._auto_configured = True

    def _build_session_log_path(self, directory: Path) -> Path:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        safe_timestamp = timestamp.replace(":", "-")
        base_path = directory / f"{safe_timestamp}.log"

        try:
            with base_path.open("x", encoding="utf-8"):
                pass
            return base_path
        except FileExistsError:
            pass

        counter = 1
        while counter <= MAX_SESSION_LOG_COLLISIONS:
            candidate = directory / f"{safe_timestamp}_{counter}.log"
            try:
                with candidate.open("x", encoding="utf-8"):
                    pass
                return candidate
            except FileExistsError:
                counter += 1

        with tempfile.NamedTemporaryFile(
            dir=directory,
            prefix=f"{safe_timestamp}_",
            suffix=".log",
            delete=False,
        ) as handle:
            return Path(handle.name)

    def _write_verbose_log_entry(self, line: str, end: str) -> None:
        if self._log_file is None:
            return

        clean_line = strip_ansi_sequences(line)
        try:
            self._log_file.write(clean_line + end)
            self._log_file.flush()
        except OSError as exc:
            if not self._log_error_reported:
                sys.stderr.write(f"Unable to write to verbose log file: {exc}\n")
                self._log_error_reported = True

    @staticmethod
    def _format_timestamp() -> str:
        now = datetime.now()
        return f"{now:%M:%S}.{now.microsecond // 1000:03d}"


LOGGER = Logger()
atexit.register(LOGGER.close)


def configure_verbose_log_capture(
    destination: str | Path | None, *, per_session: bool = False
) -> None:
    LOGGER.configure_verbose_log_capture(destination, per_session=per_session)


def set_verbose_logging(enabled: bool) -> None:
    LOGGER.set_verbose_logging(enabled)


def is_verbose_logging_enabled() -> bool:
    return LOGGER.is_verbose_logging_enabled()


def set_chunk_progress_logging(enabled: bool) -> None:
    LOGGER.set_chunk_progress_logging(enabled)


def is_chunk_progress_logging_enabled() -> bool:
    return LOGGER.is_chunk_progress_logging_enabled()


def current_verbose_log_path() -> Optional[Path]:
    return LOGGER.current_verbose_log_path()


def ws_log_label(direction: str | None = None) -> str:
    arrow = direction if direction in ("←", "→") else ""
    return _WS_LABELS[arrow]


def log_state_transition(previous: Optional[StreamState], new: StreamState, reason: str) -> None:
    if previous == new:
        return

    if previous is None:
        LOGGER.verbose(STATE_LOG_LABEL, f"Entered {new.value.upper()} ({reason})")
    else:
        LOGGER.verbose(
            STATE_LOG_LABEL,
            f"{previous.value.upper()} -> {new.value.upper()} ({reason})",
        )
