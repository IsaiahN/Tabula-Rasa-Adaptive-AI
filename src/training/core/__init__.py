"""
Core Training Components

Contains the main orchestrator classes for the training system.
"""

from .continuous_learning_loop import ContinuousLearningLoop
from .master_trainer import MasterARCTrainer

__all__ = [
    'ContinuousLearningLoop',
    'MasterARCTrainer'
]
