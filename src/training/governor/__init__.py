"""
Governor System Components

Manages meta-cognitive processes, resource allocation,
and system state management.
"""

from .governor import TrainingGovernor
from .meta_cognitive import MetaCognitiveController

__all__ = [
    'TrainingGovernor',
    'MetaCognitiveController'
]
