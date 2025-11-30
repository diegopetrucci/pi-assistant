from __future__ import annotations

import os

from pi_assistant.config.base import reset_first_launch_choices


def test_reset_first_launch_choices_clears_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ASSISTANT_MODEL=gpt-5-nano-2025-08-07\nLOCATION_NAME=Lisbon, PT\nOTHER=value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("pi_assistant.config.base.ENV_PATH", env_file)
    monkeypatch.setenv("ASSISTANT_MODEL", "gpt-5-nano-2025-08-07")
    monkeypatch.setenv("ASSISTANT_REASONING_EFFORT", "low")
    monkeypatch.setenv("LOCATION_NAME", "Lisbon, PT")

    cleared = reset_first_launch_choices()

    assert "ASSISTANT_MODEL" in cleared
    assert "LOCATION_NAME" in cleared
    assert "ASSISTANT_REASONING_EFFORT" not in cleared  # not persisted in file
    contents = env_file.read_text(encoding="utf-8")
    assert contents == "OTHER=value\n"
    assert os.getenv("ASSISTANT_MODEL") is None
    assert os.getenv("ASSISTANT_REASONING_EFFORT") is None
    assert os.getenv("LOCATION_NAME") is None
