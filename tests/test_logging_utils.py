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


def test_verbose_print_emits_timestamp_when_enabled(capsys):
    logging_utils.set_verbose_logging(True)

    logging_utils.verbose_print("message")

    out = capsys.readouterr().out
    assert out.startswith("[")
    assert "message" in out


def test_console_print_passthrough_when_not_verbose(capsys):
    logging_utils.set_verbose_logging(False)

    logging_utils.console_print("plain output")

    out = capsys.readouterr().out
    assert out.strip() == "plain output"


def test_log_state_transition_skips_when_state_unchanged(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(logging_utils, "_AUTO_CONFIGURE_PENDING", False)
    monkeypatch.setattr(logging_utils, "_AUTO_CONFIGURED", True)
    monkeypatch.setattr(logging_utils, "verbose_print", lambda *args, **kwargs: calls.append(args))
    logging_utils.set_verbose_logging(True)

    logging_utils.log_state_transition(StreamState.LISTENING, StreamState.LISTENING, "noop")

    assert calls == []


def test_auto_configuration_enables_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils, "_AUTO_CONFIGURE_PENDING", True)
    monkeypatch.setattr(logging_utils, "_AUTO_CONFIGURED", False)
    monkeypatch.setattr(logging_utils, "VERBOSE_LOG_DIRECTORY", tmp_path)

    logging_utils._ensure_auto_configured()
    path = logging_utils.current_verbose_log_path()

    assert path is not None
    logging_utils.configure_verbose_log_capture(None)


def test_write_verbose_log_entry_reports_error_once(monkeypatch, capsys):
    class BrokenLog:
        def write(self, _value):
            raise OSError("disk full")

        def flush(self):
            pass

    monkeypatch.setattr(logging_utils, "_VERBOSE_LOG_FILE", BrokenLog())
    monkeypatch.setattr(logging_utils, "_VERBOSE_LOG_ERROR_REPORTED", False)

    logging_utils._write_verbose_log_entry("00:00.000", ("msg",), " ", "\n")
    logging_utils._write_verbose_log_entry("00:00.000", ("msg",), " ", "\n")

    err = capsys.readouterr().err
    assert err.count("Unable to write to verbose log file") == 1
