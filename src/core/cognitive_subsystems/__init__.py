"""
37 Cognitive Subsystems

A comprehensive monitoring and management system for all cognitive components.
Each subsystem provides real-time monitoring, health tracking, and performance analysis
with full database integration for persistent storage and API access.
"""

from .base_subsystem import BaseCognitiveSubsystem, SubsystemStatus, SubsystemHealth
from .memory_subsystems import (
    MemoryAccessMonitor,
    MemoryConsolidationMonitor,
    MemoryFragmentationMonitor,
    SalientMemoryRetrievalMonitor,
    MemoryRegularizationMonitor,
    TemporalMemoryMonitor
)
from .learning_subsystems import (
    LearningProgressMonitor,
    MetaLearningMonitor,
    KnowledgeTransferMonitor,
    PatternRecognitionMonitor,
    CurriculumLearningMonitor,
    CrossSessionLearningMonitor
)
from .action_subsystems import (
    ActionIntelligenceMonitor,
    ActionExperimentationMonitor,
    EmergencyMovementMonitor,
    PredictiveCoordinatesMonitor,
    CoordinateSuccessMonitor
)
from .exploration_subsystems import (
    ExplorationStrategyMonitor,
    BoredomDetectionMonitor,
    StagnationDetectionMonitor,
    ContrarianStrategyMonitor,
    GoalInventionMonitor
)
from .energy_subsystems import (
    EnergySystemMonitor,
    SleepCycleMonitor,
    MidGameSleepMonitor,
    DeathManagerMonitor
)
from .visual_subsystems import (
    FrameAnalysisMonitor,
    BoundaryDetectionMonitor,
    MultiModalInputMonitor,
    VisualPatternMonitor
)
from .system_subsystems import (
    ResourceUtilizationMonitor,
    GradientFlowMonitor,
    UsageTrackingMonitor,
    AntiBiasWeightingMonitor,
    ClusterFormationMonitor,
    DangerZoneAvoidanceMonitor,
    SwarmIntelligenceMonitor,
    HebbianBonusesMonitor
)
from .cognitive_coordinator import CognitiveCoordinator
from .subsystem_api import CognitiveSubsystemAPI

__all__ = [
    # Base classes
    'BaseCognitiveSubsystem',
    'SubsystemStatus', 
    'SubsystemHealth',
    
    # Memory subsystems
    'MemoryAccessMonitor',
    'MemoryConsolidationMonitor', 
    'MemoryFragmentationMonitor',
    'SalientMemoryRetrievalMonitor',
    'MemoryRegularizationMonitor',
    'TemporalMemoryMonitor',
    
    # Learning subsystems
    'LearningProgressMonitor',
    'MetaLearningMonitor',
    'KnowledgeTransferMonitor',
    'PatternRecognitionMonitor',
    'CurriculumLearningMonitor',
    'CrossSessionLearningMonitor',
    
    # Action subsystems
    'ActionIntelligenceMonitor',
    'ActionExperimentationMonitor',
    'EmergencyMovementMonitor',
    'PredictiveCoordinatesMonitor',
    'CoordinateSuccessMonitor',
    
    # Exploration subsystems
    'ExplorationStrategyMonitor',
    'BoredomDetectionMonitor',
    'StagnationDetectionMonitor',
    'ContrarianStrategyMonitor',
    'GoalInventionMonitor',
    
    # Energy subsystems
    'EnergySystemMonitor',
    'SleepCycleMonitor',
    'MidGameSleepMonitor',
    'DeathManagerMonitor',
    
    # Visual subsystems
    'FrameAnalysisMonitor',
    'BoundaryDetectionMonitor',
    'MultiModalInputMonitor',
    'VisualPatternMonitor',
    
    # System subsystems
    'ResourceUtilizationMonitor',
    'GradientFlowMonitor',
    'UsageTrackingMonitor',
    'AntiBiasWeightingMonitor',
    'ClusterFormationMonitor',
    'DangerZoneAvoidanceMonitor',
    'SwarmIntelligenceMonitor',
    'HebbianBonusesMonitor',
    
    # Coordination
    'CognitiveCoordinator',
    'CognitiveSubsystemAPI'
]
