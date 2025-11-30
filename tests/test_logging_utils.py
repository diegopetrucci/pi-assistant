from typing import Any

import pytest

from pi_assistant.cli import logging_utils
from pi_assistant.wake_word import StreamState


@pytest.fixture(autouse=True)
def reset_verbose_log_capture():
    """Ensure each test starts with logging disabled and no open files."""

    logging_utils.configure_verbose_log_capture(None)
    logging_utils.set_verbose_logging(False)
    logging_utils.set_chunk_progress_logging(False)
    yield
    logging_utils.configure_verbose_log_capture(None)
    logging_utils.set_verbose_logging(False)
    logging_utils.set_chunk_progress_logging(False)


def test_logger_verbose_writes_to_disk_even_when_disabled(tmp_path):
    log_file = tmp_path / "logs" / "session.log"
    logging_utils.configure_verbose_log_capture(log_file)
    logging_utils.set_verbose_logging(False)

    logging_utils.LOGGER.verbose(
        logging_utils.STATE_LOG_LABEL,
        "hello",
        "world",
        end="!",
    )

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


def test_logger_verbose_includes_timestamp_when_enabled(capsys):
    logging_utils.set_verbose_logging(True)

    logging_utils.LOGGER.verbose("TEST", "message")

    out = capsys.readouterr().out
    assert out.startswith("[")
    assert "] [TEST] message" in out


def test_logger_log_includes_timestamp(capsys):
    logging_utils.set_verbose_logging(False)

    logging_utils.LOGGER.log("TRACE", "plain output")

    out = capsys.readouterr().out.strip()
    assert out.startswith("[")
    assert "] [TRACE] plain output" in out


def test_logger_log_with_exc_info(capsys):
    logging_utils.set_verbose_logging(True)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logging_utils.LOGGER.log(
            logging_utils.ERROR_LOG_LABEL,
            "Failure encountered",
            exc_info=True,
            error=True,
        )

    captured = capsys.readouterr()
    assert "RuntimeError: boom" in captured.err
    assert "Traceback" in captured.err
    logging_utils.set_verbose_logging(False)


def test_logger_log_exc_info_with_exception_instance(capsys):
    logging_utils.set_verbose_logging(True)

    try:
        raise ValueError("invalid")
    except ValueError as exc:
        logging_utils.LOGGER.log(
            logging_utils.ERROR_LOG_LABEL,
            "Second failure",
            exc_info=exc,
            error=True,
        )

    err = capsys.readouterr().err
    assert "ValueError: invalid" in err
    logging_utils.set_verbose_logging(False)


def test_logger_log_exc_info_with_tuple(capsys):
    logging_utils.set_verbose_logging(True)

    try:
        raise KeyError("missing")
    except KeyError as exc:
        info = (exc.__class__, exc, exc.__traceback__)
        logging_utils.LOGGER.log(
            logging_utils.ERROR_LOG_LABEL,
            "Tuple failure",
            exc_info=info,
            error=True,
        )

    err = capsys.readouterr().err
    assert "KeyError: 'missing'" in err
    logging_utils.set_verbose_logging(False)


def test_logger_log_rejects_unknown_option():
    with pytest.raises(TypeError):
        bad_kwargs: dict[str, Any] = {"unsupported": True}
        logging_utils.LOGGER.log("TRACE", "noop", **bad_kwargs)


def test_logger_applies_color_for_known_label(capsys):
    logging_utils.set_verbose_logging(True)

    logging_utils.LOGGER.log(logging_utils.TURN_LOG_LABEL, "colorful")

    out = capsys.readouterr().out
    assert "\033[" in out  # ANSI color present
    assert "[TURN]" in out
    logging_utils.set_verbose_logging(False)


def test_strip_ansi_sequences_removes_codes():
    colored = "\033[31mhello\033[0m world"
    assert logging_utils.strip_ansi_sequences(colored) == "hello world"


def test_ws_log_label_directional_variants():
    logging_utils.set_verbose_logging(False)
    base = logging_utils.ws_log_label()
    assert base == logging_utils.WS_LOG_LABEL

    inbound = logging_utils.ws_log_label("←")
    outbound = logging_utils.ws_log_label("→")
    assert inbound == "WS←"
    assert outbound == "WS→"


def test_log_state_transition_skips_when_state_unchanged(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configure_pending", False, raising=False)
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configured", True, raising=False)
    monkeypatch.setattr(
        logging_utils.LOGGER,
        "verbose",
        lambda *args, **kwargs: calls.append(args),
        raising=False,
    )
    logging_utils.set_verbose_logging(True)

    logging_utils.log_state_transition(StreamState.LISTENING, StreamState.LISTENING, "noop")

    assert calls == []


def test_log_state_transition_logs_from_pending(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configure_pending", False, raising=False)
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configured", True, raising=False)
    monkeypatch.setattr(
        logging_utils.LOGGER,
        "verbose",
        lambda source, message, **_: calls.append((source, message)),
        raising=False,
    )
    logging_utils.set_verbose_logging(True)

    logging_utils.log_state_transition(None, StreamState.LISTENING, "boot")

    assert calls and "Entered LISTENING" in calls[-1][1]


def test_log_state_transition_logs_between_states(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configure_pending", False, raising=False)
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configured", True, raising=False)
    monkeypatch.setattr(
        logging_utils.LOGGER,
        "verbose",
        lambda source, message, **_: calls.append((source, message)),
        raising=False,
    )
    logging_utils.set_verbose_logging(True)

    logging_utils.log_state_transition(StreamState.LISTENING, StreamState.STREAMING, "hotword")

    assert calls and "LISTENING -> STREAMING" in calls[-1][1]


def test_auto_configuration_enables_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configure_pending", True, raising=False)
    monkeypatch.setattr(logging_utils.LOGGER, "_auto_configured", False, raising=False)
    monkeypatch.setattr(logging_utils, "VERBOSE_LOG_DIRECTORY", tmp_path)

    logging_utils.LOGGER._ensure_auto_configured()
    path = logging_utils.current_verbose_log_path()

    assert path is not None
    logging_utils.configure_verbose_log_capture(None)


def test_write_verbose_log_entry_reports_error_once(monkeypatch, capsys):
    class BrokenLog:
        def write(self, _value):
            raise OSError("disk full")

        def flush(self):
            pass

    monkeypatch.setattr(logging_utils.LOGGER, "_log_file", BrokenLog(), raising=False)
    monkeypatch.setattr(logging_utils.LOGGER, "_log_error_reported", False, raising=False)

    logging_utils.LOGGER._write_verbose_log_entry("[00:00.000] [TEST] msg", "\n")
    logging_utils.LOGGER._write_verbose_log_entry("[00:00.000] [TEST] msg", "\n")

    err = capsys.readouterr().err
    assert err.count("Unable to write to verbose log file") == 1


def test_chunk_progress_logging_toggle():
    logging_utils.set_chunk_progress_logging(True)
    assert logging_utils.is_chunk_progress_logging_enabled() is True

    logging_utils.set_chunk_progress_logging(False)
    assert logging_utils.is_chunk_progress_logging_enabled() is False
