"""
Vision Package

Modular computer vision components for ARC puzzle analysis.
"""

from .object_detection import DetectedObject, ObjectDetector
from .spatial_analysis import SpatialRelationship, SpatialAnalyzer
from .pattern_recognition import PatternInfo, PatternRecognizer
from .change_detection import ChangeInfo, ChangeDetector
from .feature_extraction import ActionableTarget, FeatureExtractor
from .position_tracking import PositionTracker
from .movement_detection import MovementDetector
from .pattern_analysis import PatternAnalyzer
from .frame_processing import FrameProcessor
from .frame_analyzer import FrameAnalyzer

__all__ = [
    'DetectedObject',
    'ObjectDetector', 
    'SpatialRelationship',
    'SpatialAnalyzer',
    'PatternInfo',
    'PatternRecognizer',
    'ChangeInfo',
    'ChangeDetector',
    'ActionableTarget',
    'FeatureExtractor',
    'PositionTracker',
    'MovementDetector',
    'PatternAnalyzer',
    'FrameProcessor',
    'FrameAnalyzer'
]
