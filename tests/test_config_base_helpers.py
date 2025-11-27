import importlib
import os
from pathlib import Path

import pytest

EXPECTED_INT_VALUE = 42
EXPECTED_FLOAT_VALUE = 0.5

base = importlib.reload(importlib.import_module("pi_assistant.config.base"))


def test_persist_env_value_updates_existing_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(base, "ENV_PATH", env_path)

    assert base._persist_env_value("FOO", "1") is True
    assert base._persist_env_value("BAR", "2") is True
    assert base._persist_env_value("FOO", "updated") is True

    contents = env_path.read_text(encoding="utf-8")
    assert contents == "FOO=updated\nBAR=2\n"


def test_persist_env_value_returns_false_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(base, "ENV_PATH", env_path)

    original_write_text = Path.write_text

    def fail_write_text(self, *args, **kwargs):
        if self == env_path:
            raise OSError("disk full")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    success = base._persist_env_value("FOO", "1")

    assert success is False
    err = capsys.readouterr().err
    assert "Unable to write" in err
    assert "disk full" in err
    assert not env_path.exists()


def test_persist_env_value_returns_false_on_read_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING=1\n", encoding="utf-8")
    monkeypatch.setattr(base, "ENV_PATH", env_path)

    original_read_text = Path.read_text

    def fail_read_text(self, *args, **kwargs):
        if self == env_path:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    success = base._persist_env_value("FOO", "2")

    assert success is False
    err = capsys.readouterr().err
    assert "Unable to read" in err
    assert "permission denied" in err


def test_remove_env_keys_strips_requested_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("KEEP=1\nREMOVE=2\n", encoding="utf-8")
    monkeypatch.setattr(base, "ENV_PATH", env_path)
    monkeypatch.setenv("REMOVE", "2")

    removed = base._remove_env_keys(("REMOVE", "MISSING"))

    assert removed == {"REMOVE"}
    assert env_path.read_text(encoding="utf-8") == "KEEP=1\n"
    assert "REMOVE" not in os.environ


def test_reset_first_launch_choices_clears_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ASSISTANT_MODEL=mini\nASSISTANT_REASONING_EFFORT=low\nLOCATION_NAME=Lisbon\nOTHER=1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(base, "ENV_PATH", env_path)
    for key in base._FIRST_LAUNCH_ENV_KEYS:
        monkeypatch.setenv(key, "value")

    removed = base.reset_first_launch_choices()

    assert removed == set(base._FIRST_LAUNCH_ENV_KEYS)
    assert env_path.read_text(encoding="utf-8") == "OTHER=1\n"
    for key in base._FIRST_LAUNCH_ENV_KEYS:
        assert key not in os.environ


def test_env_path_coerces_relative_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CUSTOM_PATH", "logs/output.log")

    path = base._env_path("CUSTOM_PATH", "fallback.log")

    assert path == tmp_path / "logs" / "output.log"

    monkeypatch.delenv("CUSTOM_PATH", raising=False)
    fallback = base._env_path("CUSTOM_PATH", "defaults/cache")
    assert fallback == tmp_path / "defaults" / "cache"


def test_env_parsing_helpers_handle_types(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAG", "YES")
    monkeypatch.setenv("SOME_INT", str(EXPECTED_INT_VALUE))
    monkeypatch.setenv("SOME_FLOAT", str(EXPECTED_FLOAT_VALUE))

    assert base._env_bool("FLAG", False) is True
    assert base._env_int("SOME_INT", 0) == EXPECTED_INT_VALUE
    assert base._env_float("SOME_FLOAT", 1.0) == EXPECTED_FLOAT_VALUE


def test_env_parsing_helpers_fall_back_on_invalid_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLAG", "maybe")
    monkeypatch.setenv("SOME_INT", "not-a-number")
    monkeypatch.setenv("SOME_FLOAT", "invalid")

    assert base._env_bool("FLAG", True) is False
    assert base._env_int("SOME_INT", EXPECTED_INT_VALUE) == EXPECTED_INT_VALUE
    assert base._env_float("SOME_FLOAT", EXPECTED_FLOAT_VALUE) == EXPECTED_FLOAT_VALUE
    err = capsys.readouterr().err
    assert "Invalid value for SOME_INT='not-a-number';" in err
    assert "Invalid value for SOME_FLOAT='invalid';" in err


def test_verbose_log_directory_defaults_to_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify VERBOSE_LOG_DIRECTORY resolves to the configured default path."""

    monkeypatch.delenv("VERBOSE_LOG_DIRECTORY", raising=False)
    monkeypatch.delenv("VERBOSE_LOG_CAPTURE_ENABLED", raising=False)

    global base
    base = importlib.reload(importlib.import_module("pi_assistant.config.base"))

    expected = Path.home() / ".cache/pi-assistant/logs"
    assert base.VERBOSE_LOG_DIRECTORY == expected
