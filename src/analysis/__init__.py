"""
Analysis Package

Modular analysis components for action traces and performance.
"""

from .pattern_analysis import PatternAnalyzer
from .sequence_detection import SequenceDetector
from .performance_tracking import PerformanceTracker
from .insight_generation import InsightGenerator

__all__ = [
    'PatternAnalyzer',
    'SequenceDetector',
    'PerformanceTracker',
    'InsightGenerator'
]
