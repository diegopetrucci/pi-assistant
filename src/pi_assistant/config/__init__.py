"""
Configuration settings for the Pi transcription client.

Defaults live in ``config/defaults.toml`` and can be overridden via environment
variables or CLI flags.
"""

from __future__ import annotations

import sys as _sys

from .assistant_settings import *  # noqa: F401,F403
from .base import *  # noqa: F401,F403
from .wake_word import *  # noqa: F401,F403

sys = _sys
