import importlib
import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")

EXPECTED_INT_VALUE = 42
EXPECTED_FLOAT_VALUE = 0.5

base = importlib.reload(importlib.import_module("pi_assistant.config.base"))


def test_persist_env_value_updates_existing_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(base, "ENV_PATH", env_path)

    base._persist_env_value("FOO", "1")
    base._persist_env_value("BAR", "2")
    base._persist_env_value("FOO", "updated")

    contents = env_path.read_text(encoding="utf-8")
    assert contents == "FOO=updated\nBAR=2\n"


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
