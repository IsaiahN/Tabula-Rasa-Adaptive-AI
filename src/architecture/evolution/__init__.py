"""
Architectural Evolution

Advanced architectural evolution capabilities for the modular system.
"""

from .architectural_evolution import (
    ArchitecturalEvolution,
    EvolutionConfig,
    EvolutionStrategy,
    EvolutionResult
)

from .component_evolution import (
    ComponentEvolution,
    ComponentMutation,
    ComponentFitness
)

from .system_optimization import (
    SystemOptimization,
    OptimizationTarget,
    OptimizationResult
)

from .adaptive_architecture import (
    AdaptiveArchitecture,
    ArchitecturePattern,
    AdaptationRule
)

__all__ = [
    # Architectural Evolution
    'ArchitecturalEvolution',
    'EvolutionConfig',
    'EvolutionStrategy',
    'EvolutionResult',
    
    # Component Evolution
    'ComponentEvolution',
    'ComponentMutation',
    'ComponentFitness',
    
    # System Optimization
    'SystemOptimization',
    'OptimizationTarget',
    'OptimizationResult',
    
    # Adaptive Architecture
    'AdaptiveArchitecture',
    'ArchitecturePattern',
    'AdaptationRule'
]
