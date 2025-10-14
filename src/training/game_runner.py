"""Lightweight GameRunner for minimal mode.

This runner depends only on the `APIManager` contract from the seed. It uses async methods to create/reset/play a single game using simple selection logic and the FrameProvider + DummyDetector.
"""
from typing import Any, Dict, Optional
import asyncio
from datetime import datetime

try:
    from src.vision.frame_provider import DummyFrameProvider
    from src.vision.dummy_detector import DummyDetector
except Exception:
    # Allow imports to fail gracefully in some contexts
    DummyFrameProvider = None
    DummyDetector = None

try:
    from src.database.db_facade import DBFacade
except Exception:
    DBFacade = None

class GameRunner:
    def __init__(self, api_manager: Any, frame_provider: Any = None, detector: Any = None):
        self.api_manager = api_manager
        self.frame_provider = frame_provider or (DummyFrameProvider() if DummyFrameProvider else None)
        self.detector = detector or (DummyDetector() if DummyDetector else None)
        self.db = DBFacade() if DBFacade else None

    async def run_game(self, game_id: str, max_actions: int = 100) -> Dict[str, Any]:
        """Run one game using the API manager and minimal vision.

        Returns a result dict: {'game_id', 'score', 'actions_taken', 'win'}
        """
        # Create or reset the game
        if not self.api_manager:
            raise RuntimeError("API manager required")

        # Ensure API initialized
        if hasattr(self.api_manager, 'initialize') and not getattr(self.api_manager, 'is_initialized', lambda: True)():
            await self.api_manager.initialize()

        # Create/reset
        scorecard_id = None
        try:
            scorecard_id = await self.api_manager.create_scorecard(f"Minimal run {game_id}", "Minimal mode")
        except Exception:
            scorecard_id = None

        reset_response = await self.api_manager.reset_game(game_id, scorecard_id)
        if not reset_response:
            return {'game_id': game_id, 'score': 0.0, 'actions_taken': 0, 'win': False}

        guid = getattr(reset_response, 'guid', None) or (reset_response.get('guid') if isinstance(reset_response, dict) else None)
        state = getattr(reset_response, 'state', None) or (reset_response.get('state') if isinstance(reset_response, dict) else 'NOT_FINISHED')
        score = getattr(reset_response, 'score', 0.0) or (reset_response.get('score') if isinstance(reset_response, dict) else 0.0)

        actions = 0
        win = False

        # Persist session/game start
        now = datetime.utcnow().isoformat()
        session_id = f"session_{now}_{game_id}"
        if self.db:
            try:
                self.db.upsert_session(session_id, now, status='running')
                self.db.upsert_game(game_id, guid, scorecard_id, now)
            except Exception:
                # DB is optional but recommended by seed
                pass

        while actions < max_actions and state == 'NOT_FINISHED':
            # Get frame and optional detections
            frame = None
            if self.frame_provider:
                frame = await self.frame_provider.get_frame()

            detections = []
            if self.detector and frame is not None:
                detections = await self.detector.detect_objects(frame)

            # Decide action: if detection available choose ACTION6 with coords, else random action
            if detections:
                det = detections[0]
                # choose ACTION6 with bbox center (defensive)
                bbox = None
                if isinstance(det, dict):
                    bbox = det.get('bbox')
                x = y = None
                if bbox and hasattr(bbox, '__len__') and len(bbox) >= 4:
                    try:
                        x = int(bbox[0] + bbox[2] // 2)
                        y = int(bbox[1] + bbox[3] // 2)
                    except Exception:
                        x = y = None

                if x is None or y is None:
                    # fallback to detector's actionable point or center
                    if self.detector and frame is not None and hasattr(self.detector, 'detect_actionable_point'):
                        try:
                            pt = await self.detector.detect_actionable_point(frame)
                            x = int(pt.get('x', 16))
                            y = int(pt.get('y', 16))
                        except Exception:
                            x = 16; y = 16
                    else:
                        x = 16; y = 16

                action = {'id': 6, 'x': x, 'y': y}
            else:
                # Ask detector for fallback point
                if self.detector and frame is not None and hasattr(self.detector, 'detect_actionable_point'):
                    pt = await self.detector.detect_actionable_point(frame)
                    action = {'id': 6, 'x': int(pt['x']), 'y': int(pt['y'])}
                else:
                    action = {'id': 1}  # fallback action

            result = await self.api_manager.take_action(game_id, action, scorecard_id, guid)
            if not result:
                break

            state = getattr(result, 'state', None) or (result.get('state') if isinstance(result, dict) else 'UNKNOWN')
            score = getattr(result, 'score', 0.0) or (result.get('score') if isinstance(result, dict) else score)
            actions += 1

            # Persist action
            try:
                if self.db:
                    self.db.add_action(session_id, game_id, action, result if isinstance(result, dict) else {}, datetime.utcnow().isoformat())
            except Exception:
                pass

            if state == 'WIN' or result.get('win', False):
                win = True
                break

            # small sleep to avoid tight loop
            await asyncio.sleep(0.05)

        # try to close scorecard
        try:
            if scorecard_id:
                await self.api_manager.close_scorecard(scorecard_id)
        except Exception:
            pass

        # Persist final session and game
        now_end = datetime.utcnow().isoformat()
        if self.db:
            try:
                if score is None:
                    final_score = 0.0
                else:
                    try:
                        final_score = float(score)
                    except Exception:
                        final_score = 0.0

                final_state = state if isinstance(state, str) else (str(state) if state is not None else 'UNKNOWN')

                self.db.end_game(game_id, now_end, final_score, final_state)
                self.db.end_session(session_id, now_end, status='completed' if win else 'timeout')
            except Exception:
                pass

        return {'game_id': game_id, 'score': score, 'actions_taken': actions, 'win': win}
