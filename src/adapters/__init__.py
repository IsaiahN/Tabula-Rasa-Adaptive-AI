"""
Adapters Package

Modular adapter components for integrating different systems.
"""

from .visual_processing import ARCVisualProcessor
from .action_mapping import ARCActionMapper
from .learning_integration import AdaptiveLearningARCAgent

__all__ = [
    'ARCVisualProcessor',
    'ARCActionMapper', 
    'AdaptiveLearningARCAgent'
]
