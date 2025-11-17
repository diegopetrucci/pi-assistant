import io
import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOCATION_NAME", "Test City")

from pi_assistant.config import (
    PROJECT_ROOT,
    _env_bool,
    _env_float,
    _env_int,
    _env_path,
    _persist_env_value,
    _prompt_for_api_key,
    _prompt_for_location_name,
)


def _stdin(is_tty: bool) -> io.StringIO:
    buffer = io.StringIO()
    buffer.isatty = lambda: is_tty  # type: ignore[attr-defined]
    return buffer


def test_env_bool_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_BOOL_FLAG", raising=False)

    assert _env_bool("TEST_BOOL_FLAG", default=True) is True
    assert _env_bool("TEST_BOOL_FLAG", default=False) is False


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "Yes", "on", "  YeS  "],
)
def test_env_bool_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TEST_BOOL_FLAG", value)

    assert _env_bool("TEST_BOOL_FLAG") is True


@pytest.mark.parametrize(
    "value",
    ["0", "false", "no", "off", "", "not-truthy"],
)
def test_env_bool_non_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TEST_BOOL_FLAG", value)

    assert _env_bool("TEST_BOOL_FLAG", default=True) is False


def test_env_int_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_INT", raising=False)

    assert _env_int("TEST_INT", 7) == 7


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", 0),
        ("42", 42),
        ("-5", -5),
        (" 10 ", 10),
    ],
)
def test_env_int_parses_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: int) -> None:
    monkeypatch.setenv("TEST_INT", value)

    assert _env_int("TEST_INT", 123) == expected


def test_env_int_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_INT", "not-an-int")

    with pytest.raises(ValueError):
        _env_int("TEST_INT", 0)


def test_env_float_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_FLOAT", raising=False)

    assert _env_float("TEST_FLOAT", 1.5) == 1.5


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0", 0.0),
        ("3.14", 3.14),
        ("-2.5", -2.5),
        ("1e3", 1000.0),
        ("  6.02e23  ", 6.02e23),
    ],
)
def test_env_float_parses_values(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: float
) -> None:
    monkeypatch.setenv("TEST_FLOAT", value)

    assert _env_float("TEST_FLOAT", 123.456) == expected


@pytest.mark.parametrize("value", ["not-a-number", "nan?maybe", ""])
def test_env_float_invalid(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("TEST_FLOAT", value)

    with pytest.raises(ValueError):
        _env_float("TEST_FLOAT", 0.0)


def test_env_path_uses_default_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_PATH", raising=False)

    result = _env_path("TEST_PATH", "relative/path.txt")

    assert result == (PROJECT_ROOT / "relative/path.txt").resolve()


def test_env_path_with_absolute_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    absolute = tmp_path / "data.bin"
    monkeypatch.setenv("TEST_PATH", str(absolute))

    assert _env_path("TEST_PATH", "ignored/default") == absolute


def test_env_path_expands_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PATH", "~/custom/config.toml")

    assert _env_path("TEST_PATH", "ignored") == Path("~/custom/config.toml").expanduser()


def test_env_path_resolves_relative_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_PATH", "subdir/file.txt")

    assert _env_path("TEST_PATH", "ignored") == (PROJECT_ROOT / "subdir/file.txt").resolve()


def test_persist_env_value_creates_new_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr("pi_assistant.config.ENV_PATH", env_path, raising=False)

    _persist_env_value("FOO", "bar")

    assert env_path.read_text() == "FOO=bar\n"


def test_persist_env_value_updates_existing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("FOO=old\nBAZ=keep\n", encoding="utf-8")
    monkeypatch.setattr("pi_assistant.config.ENV_PATH", env_path, raising=False)

    _persist_env_value("FOO", "new")

    assert env_path.read_text() == "FOO=new\nBAZ=keep\n"


def test_persist_env_value_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr("pi_assistant.config.ENV_PATH", env_path, raising=False)

    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):  # type: ignore[override]
        if self == env_path:
            raise PermissionError("nope")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fake_write_text, raising=False)

    with pytest.raises(PermissionError):
        _persist_env_value("FOO", "bar")


def test_prompt_for_api_key_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pi_assistant.config.sys.stdin", _stdin(False), raising=False)

    def fake_getpass(prompt: str) -> str:
        raise AssertionError("getpass should not be called for non-interactive sessions")

    monkeypatch.setattr("pi_assistant.config.getpass", fake_getpass, raising=False)

    assert _prompt_for_api_key() is None


def test_prompt_for_api_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("pi_assistant.config.sys.stdin", _stdin(True), raising=False)
    monkeypatch.setattr("pi_assistant.config.getpass", lambda prompt: "sk-test", raising=False)
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        "pi_assistant.config._persist_api_key",
        lambda value: saved.setdefault("api_key", value),
        raising=False,
    )

    assert _prompt_for_api_key() == "sk-test"
    assert saved["api_key"] == "sk-test"


def test_prompt_for_location_name_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pi_assistant.config.sys.stdin", _stdin(False), raising=False)

    assert _prompt_for_location_name() is None


def test_prompt_for_location_name_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pi_assistant.config.sys.stdin", _stdin(True), raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "Lisbon, PT", raising=False)
    saved: dict[str, str] = {}
    monkeypatch.setattr(
        "pi_assistant.config._persist_env_value",
        lambda key, value: saved.setdefault("location", value),
        raising=False,
    )
    monkeypatch.delenv("LOCATION_NAME", raising=False)

    assert _prompt_for_location_name() == "Lisbon, PT"
    assert saved["location"] == "Lisbon, PT"
