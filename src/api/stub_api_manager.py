"""A minimal stub API manager for CI/testing.

This is strictly opt-in and should only be used with the explicit CLI flag `--use-stub-api-for-ci`.
It simulates the minimal subset of ARC3 used by the runner: create_scorecard, reset_game, take_action, get_game_state, close_scorecard.

The stub uses in-memory structures and deterministic behavior to keep CI stable.
"""
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class GameState:
    guid: str
    state: str
    score: float
    available_actions: List[int]
    frame: List[List[int]] = field(default_factory=lambda: [[0]*32 for _ in range(32)])

class StubAPIManager:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._initialized = False
        self._games: Dict[str, GameState] = {}
        self._scorecards: Dict[str, Dict] = {}

    async def initialize(self):
        await asyncio.sleep(0)
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    async def create_scorecard(self, name: str, description: str) -> str:
        sc_id = str(uuid.uuid4())
        self._scorecards[sc_id] = {'name': name, 'description': description}
        return sc_id

    async def reset_game(self, game_id: str, scorecard_id: Optional[str] = None) -> GameState:
        guid = str(uuid.uuid4())
        state = GameState(guid=guid, state='NOT_FINISHED', score=0.0, available_actions=[1,2,3,4,5,6])
        self._games[game_id] = state
        return state

    async def take_action(self, game_id: str, action: Dict[str, Any], scorecard_id: Optional[str] = None, guid: Optional[str] = None) -> Dict[str, Any]:
        # Deterministic simple rule: ACTION6 gives +10, others give +1; end when score >= 100
        gs = self._games.get(game_id)
        if not gs:
            return {}
        if action.get('id') == 6:
            gs.score += 10
        else:
            gs.score += 1
        if gs.score >= 100:
            gs.state = 'WIN'
        # return as dict to simulate real API
        return {'state': gs.state, 'score': gs.score, 'available_actions': gs.available_actions, 'frame': gs.frame, 'guid': gs.guid}

    async def get_game_state(self, game_id: str, card_id: Optional[str] = None, guid: Optional[str] = None) -> Dict[str, Any]:
        gs = self._games.get(game_id)
        if not gs:
            return {}
        return {'state': gs.state, 'score': gs.score, 'available_actions': gs.available_actions, 'frame': gs.frame, 'guid': gs.guid}

    async def close_scorecard(self, scorecard_id: str):
        if scorecard_id in self._scorecards:
            del self._scorecards[scorecard_id]
        return True

    async def get_available_games(self) -> List[Dict[str, Any]]:
        # Return a deterministic small list
        return [{'game_id': 'stub_game_1', 'id': 'stub_game_1'}]

    async def submit_score(self, game_id: str, score: float, level: int = 1, actions_taken: int = 0, win: bool = False):
        # No-op for stub
        return True

    def is_healthy(self) -> bool:
        return True

    def get_rate_limit_status(self) -> Dict[str, int]:
        return {'current_usage': 0, 'max_requests': 1000}

    @property
    def rate_limiter(self):
        class DummyLimiter:
            def get_usage_warning(self):
                return None
            def should_pause(self):
                return (False, 0)
        return DummyLimiter()

    async def close(self):
        return True
