"""
Learning Package

Modular learning components for ARC meta-learning and pattern recognition.
"""

from .pattern_recognition import ARCPatternRecognizer, ARCPattern
from .insight_extraction import ARCInsightExtractor, ARCInsight
from .knowledge_transfer import KnowledgeTransfer
from .meta_learning import ARCMetaLearningSystem

__all__ = [
    'ARCPatternRecognizer', 'ARCPattern',
    'ARCInsightExtractor', 'ARCInsight',
    'KnowledgeTransfer',
    'ARCMetaLearningSystem'
]
