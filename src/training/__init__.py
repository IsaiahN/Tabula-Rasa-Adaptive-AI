"""
Training System Package

This package contains all training-related functionality for the ARC-AGI-3 system,
organized into modular components for better maintainability and reusability.
"""

from .core.continuous_learning_loop import ContinuousLearningLoop
from .core.master_trainer import MasterARCTrainer, MasterTrainingConfig
from src.config.centralized_config import action_limits as ActionLimits

__all__ = [
    'ContinuousLearningLoop',
    'MasterARCTrainer',
    'MasterTrainingConfig',
    'ActionLimits'
]
