#!/usr/bin/env python3
"""
Simulation-Driven Intelligence Agent

This module integrates all simulation components into a unified system that
enables multi-step planning and imagination. This is the "Third Brain" that
bridges reactive AI to proactive intelligence.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .simulation_models import (
    SimulationContext, SimulationHypothesis, SimulationResult,
    SimulationEvaluation, Strategy, SimulationConfig
)
from .hypothesis_generator import SimulationHypothesisGenerator
from .simulation_evaluator import SimulationEvaluator
from .strategy_memory import StrategyMemory
from .predictive_core import PredictiveCore

logger = logging.getLogger(__name__)

class SimulationAgent:
    """
    Unified simulation-driven intelligence system.
    
    This agent can:
    1. Generate "what-if" scenarios (Architect)
    2. Simulate multi-step rollouts (Predictive Core)
    3. Evaluate simulations with emotional intelligence (Governor)
    4. Store successful strategies for future use (Memory)
    
    This transforms the AI from reactive (S → A → S+1) to proactive
    (S → [A1→S+1→A2→S+2...] → Best_A).
    """
    
    def __init__(self, 
                 predictive_core: PredictiveCore,
                 config: Optional[SimulationConfig] = None,
                 persistence_dir: str = "data/simulation_agent"):
        self.predictive_core = predictive_core
        self.config = config or SimulationConfig()
        
        # Initialize components
        self.hypothesis_generator = SimulationHypothesisGenerator({
            'visual_hypothesis_weight': self.config.visual_hypothesis_weight,
            'memory_hypothesis_weight': self.config.memory_hypothesis_weight,
            'exploration_hypothesis_weight': self.config.exploration_hypothesis_weight,
            'energy_hypothesis_weight': self.config.energy_hypothesis_weight,
            'learning_hypothesis_weight': self.config.learning_hypothesis_weight
        })
        
        self.simulation_evaluator = SimulationEvaluator(self.config)
        self.strategy_memory = StrategyMemory(
            max_strategies=self.config.max_strategies,
            similarity_threshold=self.config.strategy_similarity_threshold,
            decay_rate=self.config.strategy_decay_rate,
            min_success_rate=self.config.min_strategy_success_rate,
            persistence_dir=persistence_dir
        )
        
        # Simulation state
        self.current_context: Optional[SimulationContext] = None
        self.active_simulations: List[SimulationResult] = []
        self.simulation_history: List[SimulationResult] = []
        
        # Performance tracking
        self.simulation_count = 0
        self.successful_simulations = 0
        self.strategy_hits = 0
        self.strategy_misses = 0
        
        logger.info("Simulation Agent initialized with imagination capabilities")
    
    def generate_action_plan(self, 
                           current_state: Dict[str, Any],
                           available_actions: List[int],
                           frame_analysis: Optional[Dict[str, Any]] = None,
                           memory_patterns: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """
        Generate an action plan using simulation-driven intelligence.
        
        This is the main entry point that replaces simple action selection
        with multi-step strategic planning.
        
        Returns:
            Tuple of (action, coordinates, reasoning)
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
        
        # Check for existing strategies first (autopilot mode)
        relevant_strategies = self.strategy_memory.retrieve_relevant_strategies(context, max_strategies=3)
        
        if relevant_strategies and self._should_use_strategy(relevant_strategies[0], context):
            # Use existing strategy (autopilot)
            strategy = relevant_strategies[0]
            action, coords = strategy.action_sequence[0]
            reasoning = f"Using strategy '{strategy.name}' (success rate: {strategy.success_rate:.2f})"
            
            logger.debug(f"Using strategy: {strategy.name}")
            return action, coords, reasoning
        
        # Generate and evaluate hypotheses
        hypotheses = self.hypothesis_generator.generate_simulation_hypotheses(
            context, max_hypotheses=self.config.max_hypotheses
        )
        
        if not hypotheses:
            # Fallback to simple action selection
            action, coords = self._fallback_action_selection(context)
            return action, coords, "Fallback action selection (no hypotheses generated)"
        
        # Simulate and evaluate hypotheses
        best_action, best_coords, best_reasoning = self._simulate_and_evaluate_hypotheses(
            hypotheses, context
        )
        
        return best_action, best_coords, best_reasoning
    
    def _simulate_and_evaluate_hypotheses(self, 
                                         hypotheses: List[SimulationHypothesis],
                                         context: SimulationContext) -> Tuple[int, Optional[Tuple[int, int]], str]:
        """Simulate hypotheses and select the best one."""
        
        best_evaluation = None
        best_valence = float('-inf')
        
        for hypothesis in hypotheses:
            try:
                # Run simulation
                simulation_result = self.predictive_core.simulate_rollout(
                    initial_state=context.current_state,
                    hypothesis=hypothesis,
                    max_steps=min(hypothesis.simulation_depth, self.config.max_simulation_depth),
                    timeout=self.config.simulation_timeout
                )
                
                # Evaluate simulation
                evaluation = self.simulation_evaluator.evaluate_simulation(
                    simulation_result, context
                )
                
                # Track simulation
                self.simulation_count += 1
                if evaluation.valence > 0:
                    self.successful_simulations += 1
                
                self.simulation_history.append(simulation_result)
                if len(self.simulation_history) > 100:  # Keep only recent simulations
                    self.simulation_history = self.simulation_history[-100:]
                
                # Check if this is the best evaluation so far
                if evaluation.valence > best_valence:
                    best_valence = evaluation.valence
                    best_evaluation = evaluation
                
                logger.debug(f"Hypothesis '{hypothesis.name}': valence={evaluation.valence:.3f}, "
                           f"recommendation={evaluation.recommendation}")
                
            except Exception as e:
                logger.error(f"Simulation failed for hypothesis '{hypothesis.name}': {e}")
                continue
        
        # Select best action
        if best_evaluation and best_evaluation.valence > self.config.min_valence_threshold:
            # Use the first action from the best hypothesis
            hypothesis = best_evaluation.simulation_result.hypothesis
            action, coords = hypothesis.action_sequence[0]
            reasoning = (f"Best simulation: '{hypothesis.name}' "
                        f"(valence: {best_evaluation.valence:.3f}, "
                        f"recommendation: {best_evaluation.recommendation})")
            
            # Store successful simulation as strategy
            if best_evaluation.valence > 0.5:
                self._store_successful_simulation(best_evaluation.simulation_result)
            
            return action, coords, reasoning
        else:
            # Fallback to simple action selection
            action, coords = self._fallback_action_selection(context)
            return action, coords, f"Fallback (best valence: {best_valence:.3f} below threshold)"
    
    def _should_use_strategy(self, strategy: Strategy, context: SimulationContext) -> bool:
        """Determine if we should use an existing strategy (autopilot mode)."""
        
        # Use strategy if it has high success rate and matches context well
        if strategy.success_rate < 0.7:
            return False
        
        # Check if strategy is recent enough
        time_since_last_use = time.time() - strategy.last_used
        if time_since_last_use > 3600:  # 1 hour
            return False
        
        # Check if context matches strategy requirements
        if 'energy_range' in strategy.initial_conditions:
            energy_range = strategy.initial_conditions['energy_range']
            if not (energy_range[0] <= context.energy_level <= energy_range[1]):
                return False
        
        return True
    
    def _fallback_action_selection(self, context: SimulationContext) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Fallback to simple action selection when simulation fails."""
        
        available_actions = context.available_actions
        
        # Prefer movement actions if available
        movement_actions = [a for a in [1, 2, 3, 4] if a in available_actions]
        if movement_actions:
            action = np.random.choice(movement_actions)
            return action, None
        
        # Prefer interaction actions
        if 5 in available_actions:
            return 5, None
        
        # Use coordinate action as last resort
        if 6 in available_actions:
            coords = (np.random.randint(0, 64), np.random.randint(0, 64))
            return 6, coords
        
        # Use any available action
        if available_actions:
            return available_actions[0], None
        
        # Default action
        return 1, None
    
    def _store_successful_simulation(self, simulation_result: SimulationResult):
        """Store a successful simulation as a reusable strategy."""
        
        # Create a simple real-world outcome for now
        # In a real implementation, this would come from actual execution results
        real_world_outcome = {
            'success_rate': simulation_result.success_metrics.get('success_rate', 0.0),
            'energy_efficiency': simulation_result.success_metrics.get('energy_efficiency', 0.0),
            'learning_efficiency': simulation_result.success_metrics.get('learning_efficiency', 0.0)
        }
        
        # Store in strategy memory
        success = self.strategy_memory.store_successful_simulation(
            simulation_result, real_world_outcome
        )
        
        if success:
            logger.debug(f"Stored successful simulation as strategy")
        else:
            logger.warning(f"Failed to store simulation as strategy")
    
    def update_with_real_outcome(self, 
                               action: int,
                               coordinates: Optional[Tuple[int, int]],
                               real_outcome: Dict[str, Any]):
        """Update the simulation system with real-world outcome data."""
        
        # Update strategy memory with real outcome
        if hasattr(self, 'last_simulation_result') and self.last_simulation_result:
            self.strategy_memory.store_successful_simulation(
                self.last_simulation_result, real_outcome
            )
        
        # Update hypothesis generator weights based on success
        if self.current_context and self.simulation_history:
            # Find the most recent simulation that matches this action
            for sim_result in reversed(self.simulation_history[-10:]):
                if (sim_result.hypothesis.action_sequence and 
                    sim_result.hypothesis.action_sequence[0][0] == action):
                    
                    success_rate = real_outcome.get('success', False)
                    self.hypothesis_generator.update_hypothesis_weights(
                        sim_result.hypothesis.hypothesis_type,
                        success_rate
                    )
                    break
        
        # Update simulation evaluator accuracy
        if hasattr(self, 'last_evaluation') and self.last_evaluation:
            self.simulation_evaluator.update_evaluation_accuracy(
                self.last_evaluation, real_outcome
            )
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the simulation system."""
        
        strategy_stats = self.strategy_memory.get_strategy_statistics()
        evaluation_stats = self.simulation_evaluator.get_evaluation_statistics()
        hypothesis_stats = self.hypothesis_generator.get_hypothesis_statistics()
        
        return {
            'simulation_performance': {
                'total_simulations': self.simulation_count,
                'successful_simulations': self.successful_simulations,
                'success_rate': (self.successful_simulations / max(1, self.simulation_count)),
                'active_simulations': len(self.active_simulations),
                'simulation_history_size': len(self.simulation_history)
            },
            'strategy_memory': strategy_stats,
            'evaluation_system': evaluation_stats,
            'hypothesis_generation': hypothesis_stats,
            'config': {
                'max_simulation_depth': self.config.max_simulation_depth,
                'max_hypotheses': self.config.max_hypotheses,
                'simulation_timeout': self.config.simulation_timeout,
                'min_valence_threshold': self.config.min_valence_threshold
            }
        }
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old simulation data to prevent memory bloat."""
        
        # Clean up old strategies
        self.strategy_memory.cleanup_old_strategies(max_age_days=max_age_hours // 24)
        
        # Clean up old simulation history
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        self.simulation_history = [
            sim for sim in self.simulation_history
            if current_time - sim.execution_time < max_age_seconds
        ]
        
        # Clean up active simulations
        self.active_simulations = [
            sim for sim in self.active_simulations
            if current_time - sim.execution_time < max_age_seconds
        ]
        
        logger.info(f"Cleaned up simulation data older than {max_age_hours} hours")
    
    def reset_simulation_state(self):
        """Reset the simulation state for a fresh start."""
        
        self.current_context = None
        self.active_simulations.clear()
        self.simulation_history.clear()
        
        # Reset performance counters
        self.simulation_count = 0
        self.successful_simulations = 0
        self.strategy_hits = 0
        self.strategy_misses = 0
        
        logger.info("Simulation state reset")
    
    def enable_autopilot_mode(self, enabled: bool = True):
        """Enable/disable autopilot mode (using existing strategies)."""
        
        # This would be implemented by adjusting strategy retrieval thresholds
        if enabled:
            self.strategy_memory.min_success_rate = 0.5  # Lower threshold for autopilot
            logger.info("Autopilot mode enabled")
        else:
            self.strategy_memory.min_success_rate = 0.8  # Higher threshold for manual control
            logger.info("Autopilot mode disabled")
    
    def get_imagination_status(self) -> Dict[str, Any]:
        """Get the current status of the imagination system."""
        
        return {
            'active_hypotheses': len(self.hypothesis_generator.action_patterns),
            'stored_strategies': len(self.strategy_memory.strategies),
            'recent_simulations': len(self.simulation_history),
            'evaluation_accuracy': self.simulation_evaluator.get_evaluation_statistics().get('average_valence', 0.0),
            'strategy_hit_rate': self.strategy_memory.get_strategy_statistics().get('strategy_hit_rate', 0.0),
            'imagination_active': self.simulation_count > 0
        }
