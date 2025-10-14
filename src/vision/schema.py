"""Simple vision schema types and validators for frame and detection shapes."""
from typing import Dict, List, Any


def validate_detection(d: Dict) -> bool:
    # Expected keys: 'label', 'confidence' or 'score', and 'bbox' (x,y,w,h)
    if not isinstance(d, dict):
        return False
    if 'bbox' not in d:
        return False
    if not (isinstance(d['bbox'], (list, tuple)) and len(d['bbox']) == 4):
        return False
    if 'label' not in d:
        return False
    if 'confidence' not in d and 'score' not in d:
        return False
    return True


def validate_detections(lst: List[Dict]) -> bool:
    if not isinstance(lst, list):
        return False
    return all(validate_detection(x) for x in lst)
