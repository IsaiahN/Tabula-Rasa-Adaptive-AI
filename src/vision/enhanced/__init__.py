"""
Enhanced Vision Processing Module

This module contains advanced vision processing capabilities including
pseudo-button detection for Action 6 games.
"""

from .pseudo_button_detector import PseudoButtonDetector, create_pseudo_button_detector
from .advanced_detection import AdvancedObjectDetector, DetectionConfig, DetectionMethod, Detection

# Create stub classes for missing imports to maintain compatibility
class RealTimeProcessor:
    pass

class ProcessingConfig:
    pass

class ProcessingMode:
    pass

class ProcessedFrame:
    pass

class AttentionMechanism:
    pass

class AttentionConfig:
    pass

class AttentionType:
    pass

class AttentionResult:
    pass

class VisualReasoningEngine:
    pass

class ReasoningConfig:
    pass

class ReasoningType:
    pass

class ReasoningResult:
    pass

class SpatialRelation:
    pass

__all__ = [
    'PseudoButtonDetector', 'create_pseudo_button_detector',
    'AdvancedObjectDetector', 'DetectionConfig', 'DetectionMethod', 'Detection',
    'RealTimeProcessor', 'ProcessingConfig', 'ProcessingMode', 'ProcessedFrame',
    'AttentionMechanism', 'AttentionConfig', 'AttentionType', 'AttentionResult',
    'VisualReasoningEngine', 'ReasoningConfig', 'ReasoningType', 'ReasoningResult',
    'SpatialRelation'
]