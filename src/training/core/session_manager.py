"""Thin session manager facade.

This module provides a `SessionManager` facade that will be expanded as
session responsibilities are extracted from the monolith.
"""
from .continuous_learning_loop import ContinuousLearningLoop
from typing import Optional


class SessionManager:
    def __init__(self, core: Optional[ContinuousLearningLoop] = None):
        self._core = core or ContinuousLearningLoop()

    def current_session(self):
        return getattr(self._core, 'current_session_id', None)

    def current_game(self):
        return getattr(self._core, 'current_game_id', None)
