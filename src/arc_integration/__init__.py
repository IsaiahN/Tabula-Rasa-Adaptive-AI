"""
ARC-AGI-3 Integration Package

This package provides integration between the Adaptive Learning Agent and ARC-AGI-3 testing framework.
"""

from .arc_agent_adapter import AdaptiveLearningARCAgent, ARCVisualProcessor, ARCActionMapper
# Note: ContinuousLearningLoop is imported lazily in master_arc_trainer.py to avoid circular imports

__all__ = [
    'AdaptiveLearningARCAgent',
    'ARCVisualProcessor',
    'ARCActionMapper'
]
