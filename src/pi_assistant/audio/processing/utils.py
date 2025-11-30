"""Shared helpers for audio capture/playback modules."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["device_info_dict"]


def device_info_dict(info: object) -> dict[str, object]:
    """Return a plain dict from sounddevice info objects for logging/debugging."""
    if isinstance(info, dict):
        return dict(info)
    if isinstance(info, Mapping):
        return dict(info.items())
    if hasattr(info, "__dict__"):
        return dict(vars(info))
    return {}
