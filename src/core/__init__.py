# Core agent components
from .architect_modular import Architect
from .system_design import SystemGenome, MutationType, MutationImpact
from .mutation_system import MutationEngine, SandboxTester, Mutation, TestResult
from .evolution_engine import EvolutionEngine, FitnessEvaluator, SelectionStrategy
from .component_coordination import ComponentCoordinator, SystemIntegration

# Enhanced Learning Paradigms
from .elastic_weight_consolidation import ElasticWeightConsolidation
from .residual_learning import ResidualLearningSystem
from .extreme_learning_machines import ExtremeLearningMachine, DirectorELMEnsemble
from .enhanced_learning_integration import EnhancedLearningIntegration, create_enhanced_learning_integration

# Unified Energy Management
from .unified_energy_system import UnifiedEnergySystem, EnergyConfig, EnergyState, EnergySystemIntegration

# Coordinate Intelligence System
from .coordinate_intelligence_system import (
    CoordinateIntelligenceSystem, SuccessZoneMapper, CoordinateZone,
    ZoneType, ZoneConfidence, CoordinateIntelligence, create_coordinate_intelligence_system
)

# Enhanced Exploration Strategies
from .enhanced_exploration_strategies import (
    ExplorationType, SearchAlgorithm, ExplorationState, ExplorationResult,
    RandomExploration, CuriosityDrivenExploration, UCBExploration,
    TreeSearchExploration, GeneticAlgorithmExploration, EnhancedExplorationSystem,
    create_enhanced_exploration_system
)

# Exploration Integration
from .exploration_integration import ExplorationIntegration, create_exploration_integration

__all__ = [
    'Architect',
    'SystemGenome', 'MutationType', 'MutationImpact',
    'MutationEngine', 'SandboxTester', 'Mutation', 'TestResult',
    'EvolutionEngine', 'FitnessEvaluator', 'SelectionStrategy',
    'ComponentCoordinator', 'SystemIntegration',
    'ElasticWeightConsolidation',
    'ResidualLearningSystem',
    'ExtremeLearningMachine', 'DirectorELMEnsemble',
    'EnhancedLearningIntegration', 'create_enhanced_learning_integration',
    'UnifiedEnergySystem', 'EnergyConfig', 'EnergyState', 'EnergySystemIntegration',
    'CoordinateIntelligenceSystem', 'SuccessZoneMapper', 'CoordinateZone',
    'ZoneType', 'ZoneConfidence', 'CoordinateIntelligence', 'create_coordinate_intelligence_system',
    'ExplorationType', 'SearchAlgorithm', 'ExplorationState', 'ExplorationResult',
    'RandomExploration', 'CuriosityDrivenExploration', 'UCBExploration',
    'TreeSearchExploration', 'GeneticAlgorithmExploration', 'EnhancedExplorationSystem',
    'create_enhanced_exploration_system',
    'ExplorationIntegration', 'create_exploration_integration'
]