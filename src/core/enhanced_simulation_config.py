#!/usr/bin/env python3
"""
Enhanced Simulation Configuration

This module provides configuration for the advanced simulation system with
adaptive depth, multiple search methods, and learning capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import time

class SearchMethod(Enum):
    """Available search methods for path generation."""
    BFS = "breadth_first_search"
    DFS = "depth_first_search"
    HYBRID = "hybrid_search"
    BAYESIAN = "bayesian_search"
    ADAPTIVE = "adaptive_search"

class LearningMode(Enum):
    """Learning modes for the simulation system."""
    CONSERVATIVE = "conservative"  # Slow, careful learning
    BALANCED = "balanced"         # Moderate learning rate
    AGGRESSIVE = "aggressive"     # Fast, experimental learning

@dataclass
class AdaptiveDepthConfig:
    """Configuration for adaptive simulation depth."""
    initial_depth: int = 5
    max_depth: int = 30
    min_depth: int = 3
    depth_increment: int = 2
    depth_decrement: int = 1
    
    # Depth adjustment triggers
    confidence_threshold_high: float = 0.8  # Increase depth when confidence is high
    confidence_threshold_low: float = 0.3   # Decrease depth when confidence is low
    success_rate_threshold: float = 0.7     # Increase depth when success rate is high
    failure_rate_threshold: float = 0.3     # Decrease depth when failure rate is high
    
    # Depth adjustment rates
    depth_increase_rate: float = 0.1        # How often to increase depth
    depth_decrease_rate: float = 0.05       # How often to decrease depth
    
    # Learning progression
    depth_learning_curve: List[int] = field(default_factory=lambda: [5, 8, 12, 18, 25, 30])
    depth_learning_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.8, 0.9, 1.0])

@dataclass
class PathGenerationConfig:
    """Configuration for path generation."""
    max_paths: int = 50
    timeout: float = 2.0
    max_depth: int = 30
    
    # Search method weights
    method_weights: Dict[SearchMethod, float] = field(default_factory=lambda: {
        SearchMethod.BFS: 0.3,
        SearchMethod.DFS: 0.3,
        SearchMethod.HYBRID: 0.2,
        SearchMethod.BAYESIAN: 0.2
    })
    
    # Method selection strategy
    selection_strategy: str = "adaptive"  # "adaptive", "best_overall", "context_aware"
    exploration_rate: float = 0.1  # Rate of exploration vs exploitation
    
    # Pattern matching
    pattern_similarity_threshold: float = 0.6
    max_similar_patterns: int = 10
    
    # Cycle detection
    enable_cycle_detection: bool = True
    max_cycle_length: int = 5

@dataclass
class BayesianScoringConfig:
    """Configuration for Bayesian success scoring."""
    learning_rate: float = 0.1
    confidence_threshold: float = 0.7
    pattern_similarity_threshold: float = 0.6
    
    # Prior parameters
    initial_alpha: float = 1.0
    initial_beta: float = 1.0
    
    # Update rates
    pattern_update_rate: float = 0.1
    context_update_rate: float = 0.05
    
    # Confidence calculation
    min_confidence: float = 0.1
    max_confidence: float = 1.0
    confidence_growth_rate: float = 0.01
    
    # Pattern database
    max_patterns: int = 1000
    pattern_decay_rate: float = 0.95

@dataclass
class MethodLearningConfig:
    """Configuration for prediction method learning."""
    learning_rate: float = 0.1
    confidence_threshold: float = 0.7
    min_samples: int = 10
    adaptation_rate: float = 0.05
    
    # Performance tracking
    performance_window: int = 100  # Number of recent predictions to track
    trend_analysis_window: int = 50  # Window for trend analysis
    
    # A/B testing
    enable_ab_testing: bool = True
    min_ab_test_samples: int = 100
    ab_test_significance_threshold: float = 0.05
    
    # Context awareness
    context_similarity_threshold: float = 0.7
    max_context_profiles: int = 500
    
    # Method selection
    selection_strategy: str = "adaptive"
    exploration_rate: float = 0.1

@dataclass
class ImaginationConfig:
    """Configuration for the imagination engine."""
    pattern_similarity_threshold: float = 0.6
    analogy_confidence_threshold: float = 0.5
    max_scenarios: int = 20
    
    # Pattern database
    max_patterns: int = 1000
    pattern_decay_rate: float = 0.95
    
    # Analogy mappings
    max_analogy_mappings: int = 100
    analogy_decay_rate: float = 0.98
    
    # Scenario generation
    max_novel_scenarios: int = 5
    max_random_scenarios: int = 3
    max_efficient_scenarios: int = 2
    
    # Learning parameters
    learning_rate: float = 0.1
    success_reward: float = 0.1
    failure_penalty: float = 0.05

@dataclass
class EnhancedSimulationConfig:
    """Enhanced configuration for the simulation system."""
    
    # Core simulation parameters
    max_simulation_depth: int = 30
    max_hypotheses: int = 10
    simulation_timeout: float = 3.0
    min_valence_threshold: float = 0.1
    
    # Component configurations
    adaptive_depth: AdaptiveDepthConfig = field(default_factory=AdaptiveDepthConfig)
    path_generation: PathGenerationConfig = field(default_factory=PathGenerationConfig)
    bayesian_scoring: BayesianScoringConfig = field(default_factory=BayesianScoringConfig)
    method_learning: MethodLearningConfig = field(default_factory=MethodLearningConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    
    # Learning mode
    learning_mode: LearningMode = LearningMode.BALANCED
    
    # Performance tracking
    enable_performance_tracking: bool = True
    performance_log_interval: int = 100  # Log performance every N simulations
    
    # Debugging
    enable_debug_logging: bool = False
    debug_log_level: str = "INFO"
    
    # Persistence
    enable_persistence: bool = True
    persistence_interval: int = 1000  # Save state every N simulations
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Adjust parameters based on learning mode
        if self.learning_mode == LearningMode.CONSERVATIVE:
            self._apply_conservative_settings()
        elif self.learning_mode == LearningMode.AGGRESSIVE:
            self._apply_aggressive_settings()
        else:  # BALANCED
            self._apply_balanced_settings()
    
    def _apply_conservative_settings(self):
        """Apply conservative learning settings."""
        self.adaptive_depth.depth_increment = 1
        self.adaptive_depth.depth_increase_rate = 0.05
        self.bayesian_scoring.learning_rate = 0.05
        self.method_learning.learning_rate = 0.05
        self.imagination.learning_rate = 0.05
        self.path_generation.exploration_rate = 0.05
        self.method_learning.exploration_rate = 0.05
    
    def _apply_aggressive_settings(self):
        """Apply aggressive learning settings."""
        self.adaptive_depth.depth_increment = 3
        self.adaptive_depth.depth_increase_rate = 0.2
        self.bayesian_scoring.learning_rate = 0.2
        self.method_learning.learning_rate = 0.2
        self.imagination.learning_rate = 0.2
        self.path_generation.exploration_rate = 0.2
        self.method_learning.exploration_rate = 0.2
    
    def _apply_balanced_settings(self):
        """Apply balanced learning settings (default)."""
        # Settings are already balanced by default
        pass
    
    def get_adaptive_depth(self, 
                          current_depth: int,
                          confidence: float,
                          success_rate: float,
                          failure_rate: float) -> int:
        """Get the next adaptive depth based on current performance."""
        
        # Check if we should increase depth
        if (confidence > self.adaptive_depth.confidence_threshold_high and
            success_rate > self.adaptive_depth.success_rate_threshold and
            current_depth < self.adaptive_depth.max_depth):
            
            # Use learning curve if available
            if self.adaptive_depth.depth_learning_curve:
                for i, threshold in enumerate(self.adaptive_depth.depth_learning_thresholds):
                    if success_rate >= threshold and i < len(self.adaptive_depth.depth_learning_curve):
                        return min(self.adaptive_depth.depth_learning_curve[i], self.adaptive_depth.max_depth)
            
            # Otherwise use increment
            return min(current_depth + self.adaptive_depth.depth_increment, self.adaptive_depth.max_depth)
        
        # Check if we should decrease depth
        elif (confidence < self.adaptive_depth.confidence_threshold_low or
              failure_rate > self.adaptive_depth.failure_rate_threshold) and current_depth > self.adaptive_depth.min_depth:
            
            return max(current_depth - self.adaptive_depth.depth_decrement, self.adaptive_depth.min_depth)
        
        # No change
        return current_depth
    
    def get_search_methods(self, context: Optional[Dict[str, Any]] = None) -> List[SearchMethod]:
        """Get the list of search methods to use based on context."""
        
        # Filter methods by weight
        methods = []
        for method, weight in self.path_generation.method_weights.items():
            if weight > 0:
                methods.append(method)
        
        # Sort by weight
        methods.sort(key=lambda m: self.path_generation.method_weights[m], reverse=True)
        
        return methods
    
    def should_enable_ab_testing(self) -> bool:
        """Check if A/B testing should be enabled."""
        return self.method_learning.enable_ab_testing
    
    def get_ab_test_config(self) -> Dict[str, Any]:
        """Get A/B testing configuration."""
        return {
            'min_samples': self.method_learning.min_ab_test_samples,
            'significance_threshold': self.method_learning.ab_test_significance_threshold,
            'timeout': self.simulation_timeout
        }
    
    def get_performance_tracking_config(self) -> Dict[str, Any]:
        """Get performance tracking configuration."""
        return {
            'enable': self.enable_performance_tracking,
            'log_interval': self.performance_log_interval,
            'window_size': self.method_learning.performance_window,
            'trend_window': self.method_learning.trend_analysis_window
        }
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration."""
        return {
            'enable': self.enable_debug_logging,
            'log_level': self.debug_log_level,
            'path_generation': {
                'enable_cycle_detection': self.path_generation.enable_cycle_detection,
                'max_cycle_length': self.path_generation.max_cycle_length
            }
        }
    
    def get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence configuration."""
        return {
            'enable': self.enable_persistence,
            'interval': self.persistence_interval,
            'components': {
                'path_generator': True,
                'bayesian_scorer': True,
                'method_learner': True,
                'imagination_engine': True
            }
        }
    
    def update_learning_mode(self, new_mode: LearningMode):
        """Update the learning mode and adjust settings accordingly."""
        self.learning_mode = new_mode
        
        if new_mode == LearningMode.CONSERVATIVE:
            self._apply_conservative_settings()
        elif new_mode == LearningMode.AGGRESSIVE:
            self._apply_aggressive_settings()
        else:  # BALANCED
            self._apply_balanced_settings()
    
    def get_component_configs(self) -> Dict[str, Any]:
        """Get configurations for all components."""
        return {
            'adaptive_depth': self.adaptive_depth,
            'path_generation': self.path_generation,
            'bayesian_scoring': self.bayesian_scoring,
            'method_learning': self.method_learning,
            'imagination': self.imagination
        }
    
    def validate_config(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []
        
        # Validate depth settings
        if self.adaptive_depth.min_depth > self.adaptive_depth.max_depth:
            issues.append("min_depth cannot be greater than max_depth")
        
        if self.adaptive_depth.initial_depth < self.adaptive_depth.min_depth:
            issues.append("initial_depth cannot be less than min_depth")
        
        if self.adaptive_depth.initial_depth > self.adaptive_depth.max_depth:
            issues.append("initial_depth cannot be greater than max_depth")
        
        # Validate thresholds
        if self.adaptive_depth.confidence_threshold_high <= self.adaptive_depth.confidence_threshold_low:
            issues.append("confidence_threshold_high must be greater than confidence_threshold_low")
        
        if self.adaptive_depth.success_rate_threshold <= self.adaptive_depth.failure_rate_threshold:
            issues.append("success_rate_threshold must be greater than failure_rate_threshold")
        
        # Validate method weights
        total_weight = sum(self.path_generation.method_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Method weights should sum to 1.0, got {total_weight}")
        
        # Validate learning rates
        for component, config in self.get_component_configs().items():
            if hasattr(config, 'learning_rate'):
                if not 0 <= config.learning_rate <= 1:
                    issues.append(f"{component}.learning_rate must be between 0 and 1")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_simulation_depth': self.max_simulation_depth,
            'max_hypotheses': self.max_hypotheses,
            'simulation_timeout': self.simulation_timeout,
            'min_valence_threshold': self.min_valence_threshold,
            'learning_mode': self.learning_mode.value,
            'adaptive_depth': {
                'initial_depth': self.adaptive_depth.initial_depth,
                'max_depth': self.adaptive_depth.max_depth,
                'min_depth': self.adaptive_depth.min_depth,
                'depth_increment': self.adaptive_depth.depth_increment,
                'depth_decrement': self.adaptive_depth.depth_decrement,
                'confidence_threshold_high': self.adaptive_depth.confidence_threshold_high,
                'confidence_threshold_low': self.adaptive_depth.confidence_threshold_low,
                'success_rate_threshold': self.adaptive_depth.success_rate_threshold,
                'failure_rate_threshold': self.adaptive_depth.failure_rate_threshold,
                'depth_increase_rate': self.adaptive_depth.depth_increase_rate,
                'depth_decrease_rate': self.adaptive_depth.depth_decrease_rate,
                'depth_learning_curve': self.adaptive_depth.depth_learning_curve,
                'depth_learning_thresholds': self.adaptive_depth.depth_learning_thresholds
            },
            'path_generation': {
                'max_paths': self.path_generation.max_paths,
                'timeout': self.path_generation.timeout,
                'max_depth': self.path_generation.max_depth,
                'method_weights': {method.value: weight for method, weight in self.path_generation.method_weights.items()},
                'selection_strategy': self.path_generation.selection_strategy,
                'exploration_rate': self.path_generation.exploration_rate,
                'pattern_similarity_threshold': self.path_generation.pattern_similarity_threshold,
                'max_similar_patterns': self.path_generation.max_similar_patterns,
                'enable_cycle_detection': self.path_generation.enable_cycle_detection,
                'max_cycle_length': self.path_generation.max_cycle_length
            },
            'bayesian_scoring': {
                'learning_rate': self.bayesian_scoring.learning_rate,
                'confidence_threshold': self.bayesian_scoring.confidence_threshold,
                'pattern_similarity_threshold': self.bayesian_scoring.pattern_similarity_threshold,
                'initial_alpha': self.bayesian_scoring.initial_alpha,
                'initial_beta': self.bayesian_scoring.initial_beta,
                'pattern_update_rate': self.bayesian_scoring.pattern_update_rate,
                'context_update_rate': self.bayesian_scoring.context_update_rate,
                'min_confidence': self.bayesian_scoring.min_confidence,
                'max_confidence': self.bayesian_scoring.max_confidence,
                'confidence_growth_rate': self.bayesian_scoring.confidence_growth_rate,
                'max_patterns': self.bayesian_scoring.max_patterns,
                'pattern_decay_rate': self.bayesian_scoring.pattern_decay_rate
            },
            'method_learning': {
                'learning_rate': self.method_learning.learning_rate,
                'confidence_threshold': self.method_learning.confidence_threshold,
                'min_samples': self.method_learning.min_samples,
                'adaptation_rate': self.method_learning.adaptation_rate,
                'performance_window': self.method_learning.performance_window,
                'trend_analysis_window': self.method_learning.trend_analysis_window,
                'enable_ab_testing': self.method_learning.enable_ab_testing,
                'min_ab_test_samples': self.method_learning.min_ab_test_samples,
                'ab_test_significance_threshold': self.method_learning.ab_test_significance_threshold,
                'context_similarity_threshold': self.method_learning.context_similarity_threshold,
                'max_context_profiles': self.method_learning.max_context_profiles,
                'selection_strategy': self.method_learning.selection_strategy,
                'exploration_rate': self.method_learning.exploration_rate
            },
            'imagination': {
                'pattern_similarity_threshold': self.imagination.pattern_similarity_threshold,
                'analogy_confidence_threshold': self.imagination.analogy_confidence_threshold,
                'max_scenarios': self.imagination.max_scenarios,
                'max_patterns': self.imagination.max_patterns,
                'pattern_decay_rate': self.imagination.pattern_decay_rate,
                'max_analogy_mappings': self.imagination.max_analogy_mappings,
                'analogy_decay_rate': self.imagination.analogy_decay_rate,
                'max_novel_scenarios': self.imagination.max_novel_scenarios,
                'max_random_scenarios': self.imagination.max_random_scenarios,
                'max_efficient_scenarios': self.imagination.max_efficient_scenarios,
                'learning_rate': self.imagination.learning_rate,
                'success_reward': self.imagination.success_reward,
                'failure_penalty': self.imagination.failure_penalty
            },
            'performance_tracking': self.get_performance_tracking_config(),
            'debug': self.get_debug_config(),
            'persistence': self.get_persistence_config()
        }
