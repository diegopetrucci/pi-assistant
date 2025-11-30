"""Helpers for wiring controller loop dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

from pi_assistant.audio.processing.resampler import LinearResampler
from pi_assistant.audio.wake_word import PreRollBuffer, WakeWordEngine
from pi_assistant.config import (
    AUTO_STOP_ENABLED,
    AUTO_STOP_MAX_SILENCE_SECONDS,
    AUTO_STOP_SILENCE_THRESHOLD,
    SERVER_STOP_MIN_SILENCE_SECONDS,
    SERVER_STOP_TIMEOUT_SECONDS,
    STREAM_SAMPLE_RATE,
)


@dataclass
class ControllerLoopBindings:
    """Container for controller loop factories sourced from the CLI module."""

    controller_module: ModuleType
    wake_engine_cls: type[WakeWordEngine]
    pre_roll_factory: type[PreRollBuffer]
    resampler_factory: type[LinearResampler]


def build_controller_loop_bindings() -> ControllerLoopBindings:
    """Return factories sourced from ``pi_assistant.cli.controller``.

    Some integration tests monkeypatch attributes directly on the CLI module, so we
    ensure the commonly patched names exist even when the module does not re-export
    them explicitly.
    """

    from pi_assistant.cli import controller as controller_module

    _ensure_controller_exports(controller_module)
    return ControllerLoopBindings(
        controller_module=controller_module,
        wake_engine_cls=controller_module.WakeWordEngine,
        pre_roll_factory=controller_module.PreRollBuffer,
        resampler_factory=controller_module.LinearResampler,
    )


def _ensure_controller_exports(controller_module: ModuleType) -> None:
    defaults = {
        "AUTO_STOP_ENABLED": AUTO_STOP_ENABLED,
        "AUTO_STOP_MAX_SILENCE_SECONDS": AUTO_STOP_MAX_SILENCE_SECONDS,
        "AUTO_STOP_SILENCE_THRESHOLD": AUTO_STOP_SILENCE_THRESHOLD,
        "SERVER_STOP_MIN_SILENCE_SECONDS": SERVER_STOP_MIN_SILENCE_SECONDS,
        "SERVER_STOP_TIMEOUT_SECONDS": SERVER_STOP_TIMEOUT_SECONDS,
        "STREAM_SAMPLE_RATE": STREAM_SAMPLE_RATE,
        "PreRollBuffer": PreRollBuffer,
        "WakeWordEngine": WakeWordEngine,
    }
    for attr, value in defaults.items():
        if not hasattr(controller_module, attr):
            setattr(controller_module, attr, value)


__all__ = ["ControllerLoopBindings", "build_controller_loop_bindings"]
