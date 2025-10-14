"""Thin orchestrator facade for TB-Seed extraction.

This module provides a small `Orchestrator` class that currently wraps
the existing `ContinuousLearningLoop`. It exists to allow incremental
refactors that move functionality out of the large monolith.
"""
from typing import Any, Dict, Optional
from .continuous_learning_loop import ContinuousLearningLoop


class Orchestrator:
    def __init__(self, **kwargs):
        # Delegate to legacy ContinuousLearningLoop for now
        self._core = ContinuousLearningLoop(**kwargs)

    async def start_training(self, game_id: str, **kwargs) -> Dict[str, Any]:
        return await self._core.start_training_with_direct_control(game_id, **kwargs)

    async def get_available_games(self):
        return await self._core.get_available_games()

    def shutdown(self):
        # Provide a simple shutdown hook
        try:
            if hasattr(self._core, 'shutdown_handler'):
                self._core.shutdown_handler.request_shutdown()
        except Exception:
            pass

    def ensure_initialized(self):
        """Expose a synchronous ensure_initialized method mirroring the legacy API."""
        if hasattr(self._core, '_ensure_initialized'):
            return self._core._ensure_initialized()
        if hasattr(self._core, 'ensure_initialized'):
            return self._core.ensure_initialized()
        return None
