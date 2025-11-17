import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("VERBOSE_LOG_CAPTURE_ENABLED", "0")

from pi_assistant.cli import logging_utils
from pi_assistant.wake_word import StreamState


@pytest.fixture(autouse=True)
def reset_verbose_log_capture():
    """Ensure each test starts with logging disabled and no open files."""

    logging_utils.configure_verbose_log_capture(None)
    logging_utils.set_verbose_logging(False)
    yield
    logging_utils.configure_verbose_log_capture(None)
    logging_utils.set_verbose_logging(False)


def test_verbose_print_writes_to_disk_even_when_disabled(tmp_path):
    log_file = tmp_path / "logs" / "session.log"
    logging_utils.configure_verbose_log_capture(log_file)
    logging_utils.set_verbose_logging(False)

    logging_utils.verbose_print(logging_utils.STATE_LOG_LABEL, "hello", "world", end="!")

    logging_utils.configure_verbose_log_capture(None)

    data = log_file.read_text(encoding="utf-8")
    assert "hello world" in data
    assert data.endswith("!")
    assert "\x1b" not in data  # ANSI colors stripped for readability


def test_state_transition_logs_when_capture_enabled(tmp_path):
    log_file = tmp_path / "session.log"
    logging_utils.configure_verbose_log_capture(log_file)
    logging_utils.set_verbose_logging(False)

    logging_utils.log_state_transition(None, StreamState.LISTENING, "boot")

    logging_utils.configure_verbose_log_capture(None)

    contents = log_file.read_text(encoding="utf-8")
    assert "LISTENING" in contents
    assert "boot" in contents


def test_per_session_capture_generates_iso_filename(tmp_path):
    log_dir = tmp_path / "logs"
    logging_utils.configure_verbose_log_capture(log_dir, per_session=True)
    path = logging_utils.current_verbose_log_path()
    logging_utils.configure_verbose_log_capture(None)

    assert path is not None
    assert path.parent == log_dir
    stem = path.stem
    assert "T" in stem
    assert stem[:4].isdigit()
