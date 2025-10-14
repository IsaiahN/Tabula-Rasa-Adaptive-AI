"""A tiny object detector shim for minimal runs.

Provides a stable interface that other components can call: `detect_objects(frame)` -> list of detections.
Detections are dicts: { 'label': str, 'confidence': float, 'bbox': [x,y,w,h] } or simple coordinate tuples.
"""
from typing import Any, List, Dict

class DummyDetector:
    def __init__(self):
        pass

    async def detect_objects(self, frame: Any) -> List[Dict]:
        """Return an empty detection list (no objects) or a simple heuristic detection for testing."""
        # Minimal behavior: no detections
        return []

    async def detect_actionable_point(self, frame: Any) -> Dict:
        """Return a fallback coordinate for ACTION6 when no detections are present.
        Returns a dict with 'x' and 'y' keys.
        """
        # Fallback to center
        height = len(frame) if hasattr(frame, '__len__') else 32
        width = len(frame[0]) if height and hasattr(frame[0], '__len__') else 32
        return {'x': width // 2, 'y': height // 2}
