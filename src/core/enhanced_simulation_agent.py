#!/usr/bin/env python3
"""
Enhanced Simulation Agent

This module integrates all the advanced simulation components into a unified
system that can generate probable future paths, learn from outcomes, and
adapt its prediction methods over time.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .path_generator import PathGenerator, SearchMethod, SearchPath
from .bayesian_success_scorer import BayesianSuccessScorer
from .prediction_method_learner import PredictionMethodLearner, PredictionMethod
from .imagination_engine import ImaginationEngine, ImaginationScenario
from .enhanced_simulation_config import EnhancedSimulationConfig, LearningMode
from .simulation_models import SimulationContext, SimulationHypothesis, SimulationResult
from .predictive_core import PredictiveCore

logger = logging.getLogger(__name__)

class EnhancedSimulationAgent:
    """
    Enhanced simulation agent that integrates all advanced simulation components.
    
    This agent can:
    1. Generate multiple probable future paths using different search methods
    2. Score paths using Bayesian inference
    3. Learn which prediction methods work best
    4. Use imagination to play unseen games
    5. Adapt simulation depth over time
    """
    
    def __init__(self, 
                 predictive_core: PredictiveCore,
                 config: Optional[EnhancedSimulationConfig] = None,
                 persistence_dir: str = "data/enhanced_simulation_agent"):
        
        self.predictive_core = predictive_core
        self.config = config or EnhancedSimulationConfig()
        self.persistence_dir = persistence_dir
        
        # Initialize components
        self.path_generator = PathGenerator(
            max_depth=self.config.adaptive_depth.max_depth,
            max_paths=self.config.path_generation.max_paths,
            timeout=self.config.path_generation.timeout
        )
        
        from .bayesian_success_scorer import create_bayesian_success_scorer
        self.bayesian_scorer = create_bayesian_success_scorer(
            learning_rate=self.config.bayesian_scoring.learning_rate,
            confidence_threshold=self.config.bayesian_scoring.confidence_threshold,
            pattern_similarity_threshold=self.config.bayesian_scoring.pattern_similarity_threshold
        )
        
        self.method_learner = PredictionMethodLearner(
            learning_rate=self.config.method_learning.learning_rate,
            confidence_threshold=self.config.method_learning.confidence_threshold,
            min_samples=self.config.method_learning.min_samples,
            adaptation_rate=self.config.method_learning.adaptation_rate
        )
        
        self.imagination_engine = ImaginationEngine(
            pattern_similarity_threshold=self.config.imagination.pattern_similarity_threshold,
            analogy_confidence_threshold=self.config.imagination.analogy_confidence_threshold,
            max_scenarios=self.config.imagination.max_scenarios
        )
        
        # Simulation state
        self.current_depth = self.config.adaptive_depth.initial_depth
        self.simulation_count = 0
        self.successful_simulations = 0
        self.current_context: Optional[SimulationContext] = None
        
        # Performance tracking
        self.performance_history = []
        self.method_performance_history = []
        self.depth_adjustment_history = []
        
        # Learning statistics
        self.learning_stats = {
            'total_simulations': 0,
            'successful_simulations': 0,
            'method_switches': 0,
            'depth_adjustments': 0,
            'imagination_scenarios': 0,
            'pattern_learned': 0,
            'analogies_discovered': 0
        }
        
        logger.info("Enhanced Simulation Agent initialized with advanced capabilities")
    
    def generate_action_plan(self, 
                           current_state: Dict[str, Any],
                           available_actions: List[int],
                           frame_analysis: Optional[Dict[str, Any]] = None,
                           memory_patterns: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """
        Generate an action plan using the enhanced simulation system.
        
        This is the main entry point that replaces simple action selection
        with multi-step strategic planning using all available components.
        """
        
        # Create simulation context
        context = SimulationContext(
            current_state=current_state,
            available_actions=available_actions,
            frame_analysis=frame_analysis,
            memory_patterns=memory_patterns,
            energy_level=current_state.get('energy', 100.0),
            learning_drive=current_state.get('learning_drive', 0.5),
            boredom_level=current_state.get('boredom_level', 0.0),
            recent_actions=current_state.get('recent_actions', []),
            success_history=current_state.get('success_history', [])
        )
        
        self.current_context = context
        
        # Check if this is a new/unseen game
        is_unseen_game = self._is_unseen_game(context)
        
        if is_unseen_game:
            # Use imagination engine for unseen games
            return self._generate_imagination_plan(context)
        else:
            # Use standard simulation for known games
            return self._generate_simulation_plan(context)
    
    def _is_unseen_game(self, context: SimulationContext) -> bool:
        """Check if this is an unseen game type."""
        
        game_type = context.current_state.get('game_type', 'unknown')
        
        # Check if we have patterns for this game type
        if game_type in self.imagination_engine.game_patterns:
            return len(self.imagination_engine.game_patterns[game_type]) == 0
        
        return True
    
    def _generate_imagination_plan(self, context: SimulationContext) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """Generate action plan using imagination engine for unseen games."""
        
        # Generate imagination scenarios
        scenarios = self.imagination_engine.imagine_unseen_game(
            context.current_state,
            context.available_actions,
            self.config.imagination.max_scenarios
        )
        
        if not scenarios:
            # Fallback to basic simulation
            return self._generate_simulation_plan(context)
        
        # Select best scenario
        best_scenario = max(scenarios, key=lambda s: s.success_probability * s.confidence)
        
        # Extract first action from scenario
        if best_scenario.action_sequence:
            action, coordinates = best_scenario.action_sequence[0]
            reasoning = f"Imagination: {best_scenario.description} (prob: {best_scenario.success_probability:.3f})"
            
            # Update statistics
            self.learning_stats['imagination_scenarios'] += 1
            
            return action, coordinates, reasoning
        
        # Fallback
        return self._generate_simulation_plan(context)
    
    def _generate_simulation_plan(self, context: SimulationContext) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """Generate action plan using standard simulation."""
        
        # Select best prediction method
        method = self.method_learner.select_best_method(
            context.current_state,
            self._get_available_search_methods()
        )
        
        # Generate paths using selected method
        paths = self.path_generator.generate_paths(
            context.current_state,
            context.available_actions,
            [method],
            context.current_state
        )
        
        if not paths:
            # Fallback to random action
            action = np.random.choice(context.available_actions)
            return action, None, "Fallback: Random action"
        
        # Score paths using Bayesian inference
        scored_paths = []
        for path in paths:
            success_prob = self.bayesian_scorer.score_path_success_probability(
                path, context.current_state
            )
            scored_paths.append((path, success_prob))
        
        # Sort by success probability
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        # Select best path
        best_path, best_prob = scored_paths[0]
        
        # Extract first action
        if best_path.nodes:
            first_node = best_path.nodes[0]
            action = first_node.action
            coordinates = first_node.coordinates
            reasoning = f"Simulation: {method.value} (prob: {best_prob:.3f}, depth: {best_path.depth})"
        else:
            action = np.random.choice(context.available_actions)
            coordinates = None
            reasoning = "Fallback: No valid paths"
        
        # Update statistics
        self.learning_stats['total_simulations'] += 1
        
        return action, coordinates, reasoning
    
    def _get_available_search_methods(self) -> List[SearchMethod]:
        """Get available search methods based on configuration."""
        
        methods = []
        for method, weight in self.config.path_generation.method_weights.items():
            if weight > 0:
                methods.append(method)
        
        return methods
    
    def update_with_outcome(self, 
                           action: int,
                           coordinates: Optional[Tuple[int, int]],
                           actual_outcome: bool,
                           context: Optional[Dict[str, Any]] = None):
        """Update all components with the actual outcome."""
        
        # Update Bayesian scorer
        if hasattr(self, 'last_paths') and self.last_paths:
            for path in self.last_paths:
                self.bayesian_scorer.update_with_outcome(path, actual_outcome, context)
        
        # Update method learner
        if hasattr(self, 'last_method'):
            prediction = {
                'confidence': 0.5,  # Would be calculated from actual prediction
                'depth': self.current_depth
            }
            self.method_learner.update_method_performance(
                self.last_method, prediction, actual_outcome, context
            )
        
        # Update imagination engine
        if hasattr(self, 'last_scenario') and self.last_scenario:
            self.imagination_engine.learn_from_outcome(
                self.last_scenario, actual_outcome, context or {}
            )
        
        # Update learning statistics
        if actual_outcome:
            self.learning_stats['successful_simulations'] += 1
        
        # Adjust simulation depth
        self._adjust_simulation_depth(actual_outcome, context)
        
        # Update performance history
        self._update_performance_history(actual_outcome, context)
        
        logger.debug(f"Updated simulation system with outcome: success={actual_outcome}")
    
    def _adjust_simulation_depth(self, actual_outcome: bool, context: Optional[Dict[str, Any]]):
        """Adjust simulation depth based on performance."""
        
        # Calculate current performance metrics
        recent_success_rate = self._calculate_recent_success_rate()
        confidence = self._calculate_current_confidence()
        
        # Get new depth
        new_depth = self.config.get_adaptive_depth(
            self.current_depth,
            confidence,
            recent_success_rate,
            1.0 - recent_success_rate
        )
        
        if new_depth != self.current_depth:
            old_depth = self.current_depth
            self.current_depth = new_depth
            
            # Update path generator max depth
            self.path_generator.max_depth = new_depth
            
            # Record depth adjustment
            self.depth_adjustment_history.append({
                'timestamp': time.time(),
                'old_depth': old_depth,
                'new_depth': new_depth,
                'reason': 'performance_adjustment',
                'success_rate': recent_success_rate,
                'confidence': confidence
            })
            
            self.learning_stats['depth_adjustments'] += 1
            
            logger.info(f"Adjusted simulation depth: {old_depth} -> {new_depth} "
                       f"(success_rate: {recent_success_rate:.3f}, confidence: {confidence:.3f})")
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate."""
        
        if not self.performance_history:
            return 0.5
        
        # Use last 20 outcomes
        recent_outcomes = self.performance_history[-20:]
        if not recent_outcomes:
            return 0.5
        
        return sum(recent_outcomes) / len(recent_outcomes)
    
    def _calculate_current_confidence(self) -> float:
        """Calculate current confidence level."""
        
        if not self.performance_history:
            return 0.5
        
        # Calculate confidence based on consistency of recent outcomes
        recent_outcomes = self.performance_history[-10:]
        if len(recent_outcomes) < 5:
            return 0.5
        
        # Confidence based on outcome consistency
        success_rate = sum(recent_outcomes) / len(recent_outcomes)
        consistency = 1.0 - abs(success_rate - 0.5) * 2  # Higher consistency = higher confidence
        
        return max(0.1, min(1.0, consistency))
    
    def _update_performance_history(self, actual_outcome: bool, context: Optional[Dict[str, Any]]):
        """Update performance history."""
        
        self.performance_history.append(actual_outcome)
        
        # Keep only recent history
        max_history = 1000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
    
    def learn_from_pattern(self, 
                          game_type: str,
                          action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]],
                          success_rate: float,
                          context_conditions: Dict[str, Any]):
        """Learn a new pattern from successful gameplay."""
        
        from .imagination_engine import GamePattern
        
        pattern = GamePattern(
            game_type=game_type,
            pattern_id=f"{game_type}_{int(time.time())}",
            action_sequence=action_sequence,
            success_rate=success_rate,
            context_conditions=context_conditions,
            energy_efficiency=0.5,  # Would be calculated
            learning_efficiency=0.5  # Would be calculated
        )
        
        self.imagination_engine.add_pattern(pattern)
        self.learning_stats['pattern_learned'] += 1
        
        logger.debug(f"Learned new pattern for game type {game_type}")
    
    def discover_analogy(self, 
                        source_context: str,
                        target_context: str,
                        mapping_rules: Dict[str, str],
                        confidence: float):
        """Discover a new analogy mapping between contexts."""
        
        from .imagination_engine import AnalogyMapping
        
        mapping = AnalogyMapping(
            source_context=source_context,
            target_context=target_context,
            mapping_rules=mapping_rules,
            confidence=confidence,
            success_rate=0.5  # Will be updated based on usage
        )
        
        self.imagination_engine.add_analogy_mapping(mapping)
        self.learning_stats['analogies_discovered'] += 1
        
        logger.debug(f"Discovered analogy: {source_context} -> {target_context}")
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the simulation system."""
        
        # Get component statistics
        path_stats = self.path_generator.get_statistics()
        bayesian_stats = self.bayesian_scorer.get_statistics()
        method_stats = self.method_learner.get_statistics()
        imagination_stats = self.imagination_engine.get_statistics()
        
        # Calculate overall performance
        total_simulations = self.learning_stats['total_simulations']
        success_rate = (self.learning_stats['successful_simulations'] / max(1, total_simulations))
        
        return {
            'overall_performance': {
                'total_simulations': total_simulations,
                'successful_simulations': self.learning_stats['successful_simulations'],
                'success_rate': success_rate,
                'current_depth': self.current_depth,
                'method_switches': self.learning_stats['method_switches'],
                'depth_adjustments': self.learning_stats['depth_adjustments']
            },
            'learning_stats': self.learning_stats,
            'component_stats': {
                'path_generator': path_stats,
                'bayesian_scorer': bayesian_stats,
                'method_learner': method_stats,
                'imagination_engine': imagination_stats
            },
            'depth_adjustment_history': self.depth_adjustment_history[-10:],  # Last 10 adjustments
            'performance_history': {
                'recent_success_rate': self._calculate_recent_success_rate(),
                'current_confidence': self._calculate_current_confidence(),
                'total_outcomes': len(self.performance_history)
            }
        }
    
    def get_imagination_status(self) -> Dict[str, Any]:
        """Get the current status of the imagination system."""
        
        base_status = {
            'imagination_enabled': True,
            'current_depth': self.current_depth,
            'max_depth': self.config.adaptive_depth.max_depth,
            'learning_mode': self.config.learning_mode.value,
            'total_patterns': sum(len(patterns) for patterns in self.imagination_engine.game_patterns.values()),
            'total_analogies': len(self.imagination_engine.analogy_mappings),
            'recent_success_rate': self._calculate_recent_success_rate(),
            'current_confidence': self._calculate_current_confidence()
        }
        
        # Add component-specific status
        component_status = {
            'path_generator': {
                'methods_available': len(self._get_available_search_methods()),
                'total_paths_generated': path_stats.get('generation_stats', {}).get('total_paths_generated', 0)
            },
            'bayesian_scorer': {
                'total_predictions': bayesian_stats.get('learning_stats', {}).get('total_predictions', 0),
                'accuracy': bayesian_stats.get('learning_stats', {}).get('accuracy', 0.0)
            },
            'method_learner': {
                'best_method': self.method_learner.get_best_method().value,
                'total_evaluations': method_stats.get('learning_stats', {}).get('total_evaluations', 0)
            },
            'imagination_engine': {
                'total_scenarios': imagination_stats.get('imagination_stats', {}).get('total_scenarios_generated', 0),
                'successful_scenarios': imagination_stats.get('imagination_stats', {}).get('successful_scenarios', 0)
            }
        }
        
        return {**base_status, 'components': component_status}
    
    def reset_for_new_game(self, game_id: str):
        """Reset the agent state for a new game."""
        
        # Reset simulation state
        self.simulation_count = 0
        self.current_context = None
        
        # Reset depth to initial value
        self.current_depth = self.config.adaptive_depth.initial_depth
        self.path_generator.max_depth = self.current_depth
        
        # Clear temporary data
        if hasattr(self, 'last_paths'):
            delattr(self, 'last_paths')
        if hasattr(self, 'last_method'):
            delattr(self, 'last_method')
        if hasattr(self, 'last_scenario'):
            delattr(self, 'last_scenario')
        
        logger.info(f"Reset enhanced simulation agent for new game: {game_id}")
    
    def enable_learning_mode(self, mode: LearningMode):
        """Enable a specific learning mode."""
        
        self.config.update_learning_mode(mode)
        
        # Update component learning rates
        self.bayesian_scorer.learning_rate = self.config.bayesian_scoring.learning_rate
        self.method_learner.learning_rate = self.config.method_learning.learning_rate
        self.imagination_engine.learning_rate = self.config.imagination.learning_rate
        
        logger.info(f"Switched to {mode.value} learning mode")
    
    def get_method_recommendations(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get recommendations for which methods to use."""
        
        return self.method_learner.get_method_recommendations(context)
    
    def start_ab_test(self, 
                     test_name: str,
                     method_a: SearchMethod,
                     method_b: SearchMethod,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Start an A/B test between two methods."""
        
        # Convert SearchMethod to PredictionMethod
        method_a_pred = PredictionMethod(method_a.value)
        method_b_pred = PredictionMethod(method_b.value)
        
        return self.method_learner.start_ab_test(test_name, method_a_pred, method_b_pred, context)
    
    def record_ab_test_result(self, test_id: str, method: SearchMethod, success: bool):
        """Record a result for an active A/B test."""
        
        method_pred = PredictionMethod(method.value)
        self.method_learner.record_ab_test_result(test_id, method_pred, success)
