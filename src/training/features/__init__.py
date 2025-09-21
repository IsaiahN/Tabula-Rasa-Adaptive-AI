"""
Advanced Features Package

Contains advanced features that leverage the clean modular architecture.
These features demonstrate the power and flexibility of the new system.
"""

from .adaptive_learning import AdaptiveLearningController, LearningStrategy
from .intelligent_scheduling import TrainingScheduler, ScheduleConfig
from .cross_validation import CrossValidationEngine, ValidationConfig
from .ensemble_learning import EnsembleTrainer, EnsembleConfig
from .hyperparameter_optimization import HyperparameterOptimizer, OptimizationConfig

__all__ = [
    'AdaptiveLearningController',
    'LearningStrategy',
    'TrainingScheduler',
    'ScheduleConfig',
    'CrossValidationEngine',
    'ValidationConfig',
    'EnsembleTrainer',
    'EnsembleConfig',
    'HyperparameterOptimizer',
    'OptimizationConfig'
]
