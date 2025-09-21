"""
Training System Interfaces

Defines abstract base classes and interfaces for all training system components.
This ensures consistent APIs and enables proper dependency injection.
"""

from .base_interfaces import (
    ComponentInterface,
    MemoryInterface,
    SessionInterface,
    APIManagerInterface,
    PerformanceInterface,
    GovernorInterface,
    LearningInterface
)

from .training_interfaces import (
    TrainingOrchestratorInterface,
    ContinuousLearningInterface,
    MasterTrainerInterface
)

from .memory_interfaces import (
    MemoryManagerInterface,
    ActionMemoryInterface,
    PatternMemoryInterface
)

from .session_interfaces import (
    SessionManagerInterface,
    SessionConfigInterface,
    SessionTrackerInterface
)

from .api_interfaces import (
    APIManagerInterface as APIManagerInterface,
    APIClientInterface
)

from .performance_interfaces import (
    PerformanceMonitorInterface,
    MetricsCollectorInterface
)

from .governor_interfaces import (
    GovernorInterface as GovernorInterface,
    MetaCognitiveInterface
)

from .learning_interfaces import (
    LearningEngineInterface,
    PatternLearnerInterface,
    KnowledgeTransferInterface
)

__all__ = [
    # Base interfaces
    'ComponentInterface',
    'MemoryInterface',
    'SessionInterface',
    'APIManagerInterface',
    'PerformanceInterface',
    'GovernorInterface',
    'LearningInterface',
    
    # Training interfaces
    'TrainingOrchestratorInterface',
    'ContinuousLearningInterface',
    'MasterTrainerInterface',
    
    # Memory interfaces
    'MemoryManagerInterface',
    'ActionMemoryInterface',
    'PatternMemoryInterface',
    
    # Session interfaces
    'SessionManagerInterface',
    'SessionConfigInterface',
    'SessionTrackerInterface',
    
    # API interfaces
    'APIManagerInterface',
    'APIClientInterface',
    
    # Performance interfaces
    'PerformanceMonitorInterface',
    'MetricsCollectorInterface',
    
    # Governor interfaces
    'GovernorInterface',
    'MetaCognitiveInterface',
    
    # Learning interfaces
    'LearningEngineInterface',
    'PatternLearnerInterface',
    'KnowledgeTransferInterface'
]
