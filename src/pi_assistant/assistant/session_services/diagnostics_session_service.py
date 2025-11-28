"""Surface readiness logs once core services are up."""

from __future__ import annotations

import asyncio
from typing import Sequence

from .session_service import BaseSessionService


class DiagnosticsSessionService(BaseSessionService):
    """Emit a ready banner after dependent services finish starting."""

    def __init__(self, dependencies: Sequence[asyncio.Event]):
        super().__init__("diagnostics")
        self._dependencies = tuple(dependencies)

    async def _start(self) -> None:
        if self._dependencies:
            await asyncio.gather(*(event.wait() for event in self._dependencies))
        print("\n✓ System ready")
        print("Listening... (Press Ctrl+C to stop)\n")

    async def _stop(self) -> None:
        return
