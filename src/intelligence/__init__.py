"""
Intelligence Module for Game-Specific Hypothesis Generation and Testing

This module provides advanced intelligence capabilities for analyzing ARC games,
generating hypotheses about winning strategies, and testing those hypotheses
systematically across different learning levels.

Key Components:
- GamePatternAnalyzer: Visual pattern detection and game mechanics analysis
- HypothesisGenerator: Automatic hypothesis generation with database integration
- HypothesisTester: Experimental framework with action reasoning
- GameTypeClassifier: Game type classification (see learning module for existing implementation)

Disable pycache by default for all modules.
"""

# Disable pycache
import sys
sys.dont_write_bytecode = True

from .game_pattern_analyzer import (
    GamePatternAnalyzer,
    GameMechanic,
    VisualPattern,
    GameMechanicsProfile,
    get_game_pattern_analyzer,
    create_game_pattern_analyzer
)

from .hypothesis_generator import (
    HypothesisGenerator,
    Hypothesis,
    HypothesisType,
    HypothesisSource,
    get_hypothesis_generator,
    create_hypothesis_generator
)

from .hypothesis_tester import (
    HypothesisTester,
    TestOutcome,
    ActionRecord,
    ExperimentResult,
    get_hypothesis_tester,
    create_hypothesis_tester
)

from .hypothesis_integration import (
    HypothesisIntegrationSystem,
    get_hypothesis_integration_system,
    create_hypothesis_integration_system
)

__all__ = [
    # Game Pattern Analyzer
    'GamePatternAnalyzer',
    'GameMechanic',
    'VisualPattern',
    'GameMechanicsProfile',
    'get_game_pattern_analyzer',
    'create_game_pattern_analyzer',

    # Hypothesis Generator
    'HypothesisGenerator',
    'Hypothesis',
    'HypothesisType',
    'HypothesisSource',
    'get_hypothesis_generator',
    'create_hypothesis_generator',

    # Hypothesis Tester
    'HypothesisTester',
    'TestOutcome',
    'ActionRecord',
    'ExperimentResult',
    'get_hypothesis_tester',
    'create_hypothesis_tester',

    # Integration System
    'HypothesisIntegrationSystem',
    'get_hypothesis_integration_system',
    'create_hypothesis_integration_system'
]