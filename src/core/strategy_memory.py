#!/usr/bin/env python3
"""
Strategy Memory System

This module implements the strategy memory system that stores and retrieves
successful action sequences as compressed, reusable strategies.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import deque
import numpy as np

from .simulation_models import Strategy, SimulationResult, SimulationContext

logger = logging.getLogger(__name__)

class StrategyMemory:
    """
    Stores and retrieves successful action sequences as compressed strategies.
    This enables "autopilot" behavior for familiar situations.
    """
    
    def __init__(self, 
                 max_strategies: int = 100,
                 similarity_threshold: float = 0.7,
                 decay_rate: float = 0.95,
                 min_success_rate: float = 0.3,
                 persistence_dir: str = "data/strategy_memory"):
        self.max_strategies = max_strategies
        self.similarity_threshold = similarity_threshold
        self.decay_rate = decay_rate
        self.min_success_rate = min_success_rate
        self.persistence_dir = Path(persistence_dir)
        
        # Strategy storage
        self.strategies: List[Strategy] = []
        self.strategy_index: Dict[str, int] = {}  # name -> index mapping
        
        # Success pattern tracking
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.pattern_frequency: Dict[str, int] = {}
        
        # Performance metrics
        self.strategy_hits = 0
        self.strategy_misses = 0
        self.total_retrievals = 0
        
        # Ensure persistence directory exists
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing strategies
        self._load_strategies()
        
        logger.info(f"Strategy Memory initialized with {len(self.strategies)} strategies")
    
    def store_successful_simulation(self, 
                                  simulation_result: SimulationResult,
                                  real_world_outcome: Dict[str, Any]) -> bool:
        """
        Store a successful simulation as a reusable strategy.
        Compress the action sequence into a high-level plan.
        """
        try:
            # Compress simulation to strategy
            strategy = self._compress_simulation_to_strategy(simulation_result, real_world_outcome)
            
            # Check if strategy already exists
            if strategy.name in self.strategy_index:
                # Update existing strategy
                existing_idx = self.strategy_index[strategy.name]
                self._update_strategy(existing_idx, strategy, real_world_outcome)
                logger.debug(f"Updated existing strategy: {strategy.name}")
            else:
                # Add new strategy
                self._add_strategy(strategy)
                logger.info(f"Added new strategy: {strategy.name}")
            
            # Update success patterns
            self._update_success_patterns(strategy, real_world_outcome)
            
            # Persist strategies
            self._save_strategies()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store strategy: {e}")
            return False
    
    def retrieve_relevant_strategies(self, 
                                   context: SimulationContext,
                                   max_strategies: int = 5) -> List[Strategy]:
        """
        Retrieve strategies relevant to current situation.
        This enables quick "autopilot" responses for familiar scenarios.
        """
        self.total_retrievals += 1
        
        relevant_strategies = []
        
        for strategy in self.strategies:
            # Skip strategies with low success rate
            if strategy.success_rate < self.min_success_rate:
                continue
                
            # Calculate similarity to current context
            similarity = self._calculate_situation_similarity(context, strategy)
            
            if similarity >= self.similarity_threshold:
                relevant_strategies.append((strategy, similarity))
        
        # Sort by success rate and similarity
        relevant_strategies.sort(
            key=lambda x: x[0].success_rate * x[1] * (1 + x[0].usage_count * 0.1), 
            reverse=True
        )
        
        # Return top strategies
        top_strategies = [strategy for strategy, _ in relevant_strategies[:max_strategies]]
        
        if top_strategies:
            self.strategy_hits += 1
            # Update usage count
            for strategy in top_strategies:
                strategy.usage_count += 1
                strategy.last_used = time.time()
        else:
            self.strategy_misses += 1
        
        return top_strategies
    
    def _compress_simulation_to_strategy(self, 
                                       simulation_result: SimulationResult,
                                       real_world_outcome: Dict[str, Any]) -> Strategy:
        """Compress a simulation result into a reusable strategy."""
        
        # Extract key information
        action_sequence = simulation_result.hypothesis.action_sequence
        final_state = simulation_result.final_state
        
        # Calculate success metrics
        success_rate = real_world_outcome.get('success_rate', 0.0)
        energy_efficiency = real_world_outcome.get('energy_efficiency', 0.0)
        learning_efficiency = real_world_outcome.get('learning_efficiency', 0.0)
        
        # Create strategy name based on action pattern
        action_pattern = self._generate_action_pattern_name(action_sequence)
        strategy_name = f"{action_pattern}_{int(time.time())}"
        
        # Extract initial conditions from simulation
        initial_conditions = {
            'energy_range': (final_state.get('energy', 50) - 20, final_state.get('energy', 50) + 20),
            'available_actions': simulation_result.hypothesis.context_requirements.get('available_actions', []),
            'context_patterns': simulation_result.hypothesis.context_requirements
        }
        
        # Create context patterns for matching
        context_patterns = self._extract_context_patterns(simulation_result)
        
        return Strategy(
            name=strategy_name,
            description=f"Strategy for {simulation_result.hypothesis.description}",
            action_sequence=action_sequence,
            initial_conditions=initial_conditions,
            success_rate=success_rate,
            energy_efficiency=energy_efficiency,
            learning_efficiency=learning_efficiency,
            context_patterns=context_patterns
        )
    
    def _generate_action_pattern_name(self, action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]) -> str:
        """Generate a descriptive name for an action sequence pattern."""
        if not action_sequence:
            return "empty"
        
        # Extract action types
        actions = [action for action, _ in action_sequence]
        
        # Count action frequencies
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Generate pattern name
        if len(action_counts) == 1:
            action_type = list(action_counts.keys())[0]
            return f"action{action_type}_x{action_counts[action_type]}"
        else:
            # Multiple action types
            sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
            pattern_parts = [f"a{action}x{count}" for action, count in sorted_actions[:3]]
            return "_".join(pattern_parts)
    
    def _extract_context_patterns(self, simulation_result: SimulationResult) -> List[Dict[str, Any]]:
        """Extract context patterns from simulation for future matching."""
        patterns = []
        
        # Energy pattern
        if 'energy' in simulation_result.final_state:
            patterns.append({
                'type': 'energy',
                'value': simulation_result.final_state['energy'],
                'tolerance': 20.0
            })
        
        # Action availability pattern
        if 'available_actions' in simulation_result.hypothesis.context_requirements:
            patterns.append({
                'type': 'available_actions',
                'value': simulation_result.hypothesis.context_requirements['available_actions'],
                'tolerance': 0.8  # 80% overlap required
            })
        
        # Visual pattern (if available)
        if 'frame_analysis' in simulation_result.hypothesis.context_requirements:
            frame_analysis = simulation_result.hypothesis.context_requirements['frame_analysis']
            if 'object_count' in frame_analysis:
                patterns.append({
                    'type': 'object_count',
                    'value': frame_analysis['object_count'],
                    'tolerance': 2  # ±2 objects
                })
        
        return patterns
    
    def _calculate_situation_similarity(self, 
                                      context: SimulationContext,
                                      strategy: Strategy) -> float:
        """Calculate similarity between current context and strategy conditions."""
        similarity_scores = []
        
        # Energy similarity
        if 'energy_range' in strategy.initial_conditions:
            energy_range = strategy.initial_conditions['energy_range']
            current_energy = context.energy_level
            
            if energy_range[0] <= current_energy <= energy_range[1]:
                energy_similarity = 1.0
            else:
                # Calculate distance-based similarity
                distance = min(abs(current_energy - energy_range[0]), 
                              abs(current_energy - energy_range[1]))
                energy_similarity = max(0, 1.0 - distance / 50.0)  # Normalize by 50 energy units
            similarity_scores.append(energy_similarity)
        
        # Action availability similarity
        if 'available_actions' in strategy.initial_conditions:
            strategy_actions = set(strategy.initial_conditions['available_actions'])
            current_actions = set(context.available_actions)
            
            if strategy_actions and current_actions:
                overlap = len(strategy_actions.intersection(current_actions))
                union = len(strategy_actions.union(current_actions))
                action_similarity = overlap / union if union > 0 else 0.0
                similarity_scores.append(action_similarity)
        
        # Context pattern similarity
        for pattern in strategy.context_patterns:
            pattern_similarity = self._match_context_pattern(context, pattern)
            if pattern_similarity is not None:
                similarity_scores.append(pattern_similarity)
        
        # Calculate weighted average
        if similarity_scores:
            return sum(similarity_scores) / len(similarity_scores)
        else:
            return 0.0
    
    def _match_context_pattern(self, context: SimulationContext, pattern: Dict[str, Any]) -> Optional[float]:
        """Match a specific context pattern."""
        pattern_type = pattern.get('type')
        pattern_value = pattern.get('value')
        tolerance = pattern.get('tolerance', 0.0)
        
        if pattern_type == 'energy':
            current_energy = context.energy_level
            if abs(current_energy - pattern_value) <= tolerance:
                return 1.0
            else:
                return max(0, 1.0 - abs(current_energy - pattern_value) / tolerance)
        
        elif pattern_type == 'available_actions':
            if not context.available_actions:
                return 0.0
            current_actions = set(context.available_actions)
            pattern_actions = set(pattern_value)
            overlap = len(current_actions.intersection(pattern_actions))
            return overlap / len(pattern_actions) if pattern_actions else 0.0
        
        elif pattern_type == 'object_count':
            if context.frame_analysis and 'object_count' in context.frame_analysis:
                current_count = context.frame_analysis['object_count']
                if abs(current_count - pattern_value) <= tolerance:
                    return 1.0
                else:
                    return max(0, 1.0 - abs(current_count - pattern_value) / tolerance)
        
        return None
    
    def _update_success_patterns(self, strategy: Strategy, real_world_outcome: Dict[str, Any]):
        """Update success patterns for future hypothesis generation."""
        pattern_key = strategy.name
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = []
            self.pattern_frequency[pattern_key] = 0
        
        # Add outcome to pattern history
        self.success_patterns[pattern_key].append({
            'timestamp': time.time(),
            'success_rate': real_world_outcome.get('success_rate', 0.0),
            'energy_efficiency': real_world_outcome.get('energy_efficiency', 0.0),
            'learning_efficiency': real_world_outcome.get('learning_efficiency', 0.0)
        })
        
        # Update frequency
        self.pattern_frequency[pattern_key] += 1
        
        # Keep only recent patterns (last 100)
        if len(self.success_patterns[pattern_key]) > 100:
            self.success_patterns[pattern_key] = self.success_patterns[pattern_key][-100:]
    
    def _add_strategy(self, strategy: Strategy):
        """Add a new strategy to memory."""
        if len(self.strategies) >= self.max_strategies:
            # Remove least successful strategy
            self._remove_least_successful_strategy()
        
        self.strategies.append(strategy)
        self.strategy_index[strategy.name] = len(self.strategies) - 1
    
    def _update_strategy(self, index: int, strategy: Strategy, real_world_outcome: Dict[str, Any]):
        """Update an existing strategy with new outcome data."""
        existing_strategy = self.strategies[index]
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        new_success_rate = real_world_outcome.get('success_rate', 0.0)
        existing_strategy.success_rate = (1 - alpha) * existing_strategy.success_rate + alpha * new_success_rate
        
        # Update efficiency metrics
        new_energy_efficiency = real_world_outcome.get('energy_efficiency', 0.0)
        existing_strategy.energy_efficiency = (1 - alpha) * existing_strategy.energy_efficiency + alpha * new_energy_efficiency
        
        new_learning_efficiency = real_world_outcome.get('learning_efficiency', 0.0)
        existing_strategy.learning_efficiency = (1 - alpha) * existing_strategy.learning_efficiency + alpha * new_learning_efficiency
        
        # Update usage count
        existing_strategy.usage_count += 1
        existing_strategy.last_used = time.time()
    
    def _remove_least_successful_strategy(self):
        """Remove the least successful strategy to make room for new ones."""
        if not self.strategies:
            return
        
        # Find strategy with lowest combined score
        worst_idx = 0
        worst_score = float('inf')
        
        for i, strategy in enumerate(self.strategies):
            # Combined score: success_rate * energy_efficiency * learning_efficiency * usage_factor
            usage_factor = 1.0 + (strategy.usage_count * 0.1)
            score = strategy.success_rate * strategy.energy_efficiency * strategy.learning_efficiency * usage_factor
            
            if score < worst_score:
                worst_score = score
                worst_idx = i
        
        # Remove strategy
        removed_strategy = self.strategies.pop(worst_idx)
        del self.strategy_index[removed_strategy.name]
        
        # Update indices
        for i in range(worst_idx, len(self.strategies)):
            strategy_name = self.strategies[i].name
            self.strategy_index[strategy_name] = i
        
        logger.info(f"Removed least successful strategy: {removed_strategy.name}")
    
    def _save_strategies(self):
        """Save strategies to disk."""
        try:
            strategies_data = []
            for strategy in self.strategies:
                strategy_dict = {
                    'name': strategy.name,
                    'description': strategy.description,
                    'action_sequence': strategy.action_sequence,
                    'initial_conditions': strategy.initial_conditions,
                    'success_rate': strategy.success_rate,
                    'energy_efficiency': strategy.energy_efficiency,
                    'learning_efficiency': strategy.learning_efficiency,
                    'usage_count': strategy.usage_count,
                    'last_used': strategy.last_used,
                    'created_at': strategy.created_at,
                    'context_patterns': strategy.context_patterns
                }
                strategies_data.append(strategy_dict)
            
            # Save to JSON file
            strategies_file = self.persistence_dir / "strategies.json"
            with open(strategies_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
            
            # Save success patterns
            patterns_file = self.persistence_dir / "success_patterns.json"
            with open(patterns_file, 'w') as f:
                json.dump(self.success_patterns, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")
    
    def _load_strategies(self):
        """Load strategies from disk."""
        try:
            strategies_file = self.persistence_dir / "strategies.json"
            if strategies_file.exists():
                with open(strategies_file, 'r') as f:
                    strategies_data = json.load(f)
                
                for strategy_dict in strategies_data:
                    strategy = Strategy(
                        name=strategy_dict['name'],
                        description=strategy_dict['description'],
                        action_sequence=strategy_dict['action_sequence'],
                        initial_conditions=strategy_dict['initial_conditions'],
                        success_rate=strategy_dict['success_rate'],
                        energy_efficiency=strategy_dict['energy_efficiency'],
                        learning_efficiency=strategy_dict['learning_efficiency'],
                        usage_count=strategy_dict.get('usage_count', 0),
                        last_used=strategy_dict.get('last_used', time.time()),
                        created_at=strategy_dict.get('created_at', time.time()),
                        context_patterns=strategy_dict.get('context_patterns', [])
                    )
                    self.strategies.append(strategy)
                    self.strategy_index[strategy.name] = len(self.strategies) - 1
            
            # Load success patterns
            patterns_file = self.persistence_dir / "success_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    self.success_patterns = json.load(f)
            
            logger.info(f"Loaded {len(self.strategies)} strategies from disk")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored strategies."""
        if not self.strategies:
            return {
                'total_strategies': 0,
                'average_success_rate': 0.0,
                'strategy_hit_rate': 0.0,
                'most_used_strategy': None
            }
        
        success_rates = [s.success_rate for s in self.strategies]
        usage_counts = [s.usage_count for s in self.strategies]
        
        most_used = max(self.strategies, key=lambda s: s.usage_count) if self.strategies else None
        
        return {
            'total_strategies': len(self.strategies),
            'average_success_rate': sum(success_rates) / len(success_rates),
            'strategy_hit_rate': self.strategy_hits / max(1, self.total_retrievals),
            'most_used_strategy': most_used.name if most_used else None,
            'total_retrievals': self.total_retrievals,
            'strategy_hits': self.strategy_hits,
            'strategy_misses': self.strategy_misses
        }
    
    def cleanup_old_strategies(self, max_age_days: int = 30):
        """Remove strategies that haven't been used recently."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        strategies_to_remove = []
        for i, strategy in enumerate(self.strategies):
            if current_time - strategy.last_used > max_age_seconds:
                strategies_to_remove.append(i)
        
        # Remove strategies in reverse order to maintain indices
        for i in reversed(strategies_to_remove):
            removed_strategy = self.strategies.pop(i)
            del self.strategy_index[removed_strategy.name]
            logger.info(f"Removed old strategy: {removed_strategy.name}")
        
        # Update remaining indices
        for i, strategy in enumerate(self.strategies):
            self.strategy_index[strategy.name] = i
        
        if strategies_to_remove:
            self._save_strategies()
            logger.info(f"Cleaned up {len(strategies_to_remove)} old strategies")
