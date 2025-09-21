"""
Advanced Learning Algorithms

Contains state-of-the-art learning algorithms that leverage the modular architecture.
These algorithms demonstrate the power and flexibility of the new system.
"""

from .meta_learning import MetaLearningAlgorithm, MetaLearningConfig
from .transfer_learning import TransferLearningAlgorithm, TransferConfig
from .reinforcement_learning import ReinforcementLearningAlgorithm, RLConfig
from .ensemble_learning import EnsembleLearningAlgorithm, EnsembleConfig
from .online_learning import OnlineLearningAlgorithm, OnlineConfig
from .federated_learning import FederatedLearningAlgorithm, FederatedConfig

__all__ = [
    'MetaLearningAlgorithm',
    'MetaLearningConfig',
    'TransferLearningAlgorithm',
    'TransferConfig',
    'ReinforcementLearningAlgorithm',
    'RLConfig',
    'EnsembleLearningAlgorithm',
    'EnsembleConfig',
    'OnlineLearningAlgorithm',
    'OnlineConfig',
    'FederatedLearningAlgorithm',
    'FederatedConfig'
]
