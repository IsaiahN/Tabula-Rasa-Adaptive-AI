#!/usr/bin/env python3
"""
Simulation Models for Multi-Step Intelligence

This module defines the data structures and models for the simulation-driven
intelligence system that enables multi-step planning and imagination.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
import torch
import numpy as np

class SimulationStatus(Enum):
    """Status of a simulation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    FAILED = "failed"

class HypothesisType(Enum):
    """Types of simulation hypotheses."""
    VISUAL_TARGETING = "visual_targeting"
    MEMORY_GUIDED = "memory_guided"
    EXPLORATION = "exploration"
    ENERGY_OPTIMIZATION = "energy_optimization"
    LEARNING_FOCUSED = "learning_focused"
    STRATEGY_RETRIEVAL = "strategy_retrieval"

@dataclass
class SimulationHypothesis:
    """A specific what-if scenario to simulate."""
    name: str
    description: str
    hypothesis_type: HypothesisType
    action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]  # (action, coordinates)
    simulation_depth: int  # How many steps to simulate
    priority: float  # 0.0 to 1.0
    expected_outcome: str  # What we expect to achieve
    energy_cost: float  # Predicted energy cost
    learning_potential: float  # Predicted learning progress
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

@dataclass
class SimulationStep:
    """A single step in a simulation."""
    step: int
    action: int
    coordinates: Optional[Tuple[int, int]]
    predicted_state: Dict[str, Any]
    energy_change: float
    learning_progress: float
    confidence: float
    reasoning: str = ""

@dataclass
class SimulationResult:
    """Result of a multi-step simulation."""
    hypothesis: SimulationHypothesis
    status: SimulationStatus
    final_state: Dict[str, Any]
    simulation_history: List[SimulationStep]
    success_metrics: Dict[str, float]
    total_energy_cost: float
    total_learning_gain: float
    execution_time: float
    terminated_early: bool = False
    termination_reason: str = ""

@dataclass
class SimulationEvaluation:
    """Evaluation of a simulation using affective systems."""
    simulation_result: SimulationResult
    valence: float  # Positive = good, Negative = bad (-1.0 to 1.0)
    energy_impact: float
    learning_impact: float
    boredom_impact: float
    recommendation: str
    confidence: float
    reasoning: str = ""

@dataclass
class Strategy:
    """A compressed, reusable action sequence."""
    name: str
    description: str
    action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]
    initial_conditions: Dict[str, Any]  # When to use this strategy
    success_rate: float
    energy_efficiency: float
    learning_efficiency: float
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    context_patterns: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class SimulationContext:
    """Context information for simulation generation."""
    current_state: Dict[str, Any]
    available_actions: List[int]
    frame_analysis: Optional[Dict[str, Any]] = None
    memory_patterns: Optional[Dict[str, Any]] = None
    energy_level: float = 100.0
    learning_drive: float = 0.5
    boredom_level: float = 0.0
    recent_actions: List[int] = field(default_factory=list)
    success_history: List[Dict[str, Any]] = field(default_factory=list)

class SimulationMetrics:
    """Tracks simulation performance and effectiveness."""
    
    def __init__(self):
        self.total_simulations = 0
        self.successful_simulations = 0
        self.average_valence = 0.0
        self.strategy_hits = 0
        self.strategy_misses = 0
        self.energy_efficiency = 0.0
        self.learning_efficiency = 0.0
        
    def update_simulation_result(self, evaluation: SimulationEvaluation):
        """Update metrics with a simulation evaluation."""
        self.total_simulations += 1
        
        if evaluation.valence > 0:
            self.successful_simulations += 1
            
        # Update running averages
        self.average_valence = (
            (self.average_valence * (self.total_simulations - 1) + evaluation.valence) 
            / self.total_simulations
        )
        
    def get_success_rate(self) -> float:
        """Get the success rate of simulations."""
        if self.total_simulations == 0:
            return 0.0
        return self.successful_simulations / self.total_simulations
        
    def get_strategy_hit_rate(self) -> float:
        """Get the strategy hit rate."""
        total_strategy_attempts = self.strategy_hits + self.strategy_misses
        if total_strategy_attempts == 0:
            return 0.0
        return self.strategy_hits / total_strategy_attempts

class SimulationConfig:
    """Configuration for simulation parameters."""
    
    def __init__(self):
        # Simulation parameters
        self.max_simulation_depth = 10
        self.max_hypotheses = 5
        self.simulation_timeout = 1.0  # seconds
        self.min_valence_threshold = 0.1
        
        # Hypothesis generation
        self.visual_hypothesis_weight = 0.3
        self.memory_hypothesis_weight = 0.25
        self.exploration_hypothesis_weight = 0.2
        self.energy_hypothesis_weight = 0.15
        self.learning_hypothesis_weight = 0.1
        
        # Evaluation weights
        self.energy_evaluation_weight = 0.4
        self.learning_evaluation_weight = 0.3
        self.boredom_evaluation_weight = 0.2
        self.confidence_evaluation_weight = 0.1
        
        # Strategy memory
        self.max_strategies = 100
        self.strategy_similarity_threshold = 0.7
        self.strategy_decay_rate = 0.95
        self.min_strategy_success_rate = 0.3
