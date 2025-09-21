"""
Exploration Integration Module

This module integrates the enhanced exploration strategies with the existing
ARC-AGI-3 system components for seamless exploration functionality.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .enhanced_exploration_strategies import (
    EnhancedExplorationSystem, ExplorationState, ExplorationResult,
    RandomExploration, CuriosityDrivenExploration, UCBExploration,
    TreeSearchExploration, GeneticAlgorithmExploration,
    create_enhanced_exploration_system
)
from .action_selection import ExplorationStrategy as LegacyExplorationStrategy
from ..database.system_integration import get_system_integration
from ..database.api import Component, LogLevel

logger = logging.getLogger(__name__)


class ExplorationIntegration:
    """Integration layer for enhanced exploration strategies."""
    
    def __init__(self, enable_database_storage: bool = True):
        self.enable_database_storage = enable_database_storage
        self.enhanced_system = create_enhanced_exploration_system()
        self.legacy_strategy = LegacyExplorationStrategy()
        
        # Database integration
        if self.enable_database_storage:
            self.integration = get_system_integration()
            self.Component = Component
            self.LogLevel = LogLevel
        else:
            self.integration = None
        
        # Performance tracking
        self.exploration_stats = {
            'total_explorations': 0,
            'successful_explorations': 0,
            'failed_explorations': 0,
            'strategy_usage': {},
            'avg_confidence': 0.0,
            'avg_exploration_value': 0.0
        }
        
        logger.info("Exploration Integration initialized")
    
    def explore(self, position: Tuple[float, float, float], 
                energy_level: float, learning_progress: float,
                visited_positions: List[Tuple[float, float, float]],
                success_history: List[bool], action_history: List[int],
                available_actions: List[int], context: Dict[str, Any] = None) -> ExplorationResult:
        """
        Perform exploration using the enhanced exploration system.
        
        Args:
            position: Current position (x, y, z)
            energy_level: Current energy level (0-100)
            learning_progress: Current learning progress (0-1)
            visited_positions: List of previously visited positions
            success_history: History of successful actions
            action_history: History of actions taken
            available_actions: List of available actions
            context: Additional context information
            
        Returns:
            ExplorationResult: The exploration result
        """
        if not available_actions:
            raise ValueError("No available actions for exploration")
        
        try:
            # Create exploration state
            state = ExplorationState(
                position=position,
                energy_level=energy_level,
                learning_progress=learning_progress,
                visited_positions=set(visited_positions),
                success_history=success_history,
                action_history=action_history,
                context=context or {}
            )
            
            # Perform exploration
            result = self.enhanced_system.explore(state, available_actions)
            
            # Update statistics
            self._update_statistics(result)
            
            # Log to database
            if self.enable_database_storage:
                self._log_exploration(result)
            
            logger.debug(f"Exploration completed: action={result.action}, "
                        f"strategy={result.strategy_used.value}, "
                        f"confidence={result.confidence:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            # Fallback to legacy strategy
            return self._fallback_exploration(position, available_actions)
    
    def update_exploration(self, result: ExplorationResult, success: bool):
        """
        Update exploration strategies based on the result.
        
        Args:
            result: The exploration result
            success: Whether the exploration was successful
        """
        try:
            # Update enhanced system
            self.enhanced_system.update(result, success)
            
            # Update statistics
            if success:
                self.exploration_stats['successful_explorations'] += 1
            else:
                self.exploration_stats['failed_explorations'] += 1
            
            # Update strategy usage
            strategy_name = result.strategy_used.value
            if strategy_name not in self.exploration_stats['strategy_usage']:
                self.exploration_stats['strategy_usage'][strategy_name] = 0
            self.exploration_stats['strategy_usage'][strategy_name] += 1
            
            # Log update to database
            if self.enable_database_storage:
                self._log_exploration_update(result, success)
            
            logger.debug(f"Exploration updated: success={success}, "
                        f"strategy={strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to update exploration: {e}")
    
    def get_exploration_bonus(self, position: Tuple[float, float, float], 
                             learning_progress: float) -> float:
        """
        Get exploration bonus for a position using legacy compatibility.
        
        Args:
            position: Position to evaluate
            learning_progress: Current learning progress
            
        Returns:
            float: Exploration bonus value
        """
        try:
            # Convert to tensor for legacy compatibility
            import torch
            position_tensor = torch.tensor(position)
            
            # Use legacy strategy
            bonus = self.legacy_strategy.get_exploration_bonus(position_tensor, learning_progress)
            
            return float(bonus)
            
        except Exception as e:
            logger.error(f"Failed to get exploration bonus: {e}")
            return 0.0
    
    def should_explore(self, learning_progress: float, energy_level: float) -> bool:
        """
        Determine if agent should explore using legacy compatibility.
        
        Args:
            learning_progress: Current learning progress
            energy_level: Current energy level
            
        Returns:
            bool: True if agent should explore
        """
        try:
            return self.legacy_strategy.should_explore(learning_progress, energy_level)
            
        except Exception as e:
            logger.error(f"Failed to determine exploration: {e}")
            return False
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive exploration statistics."""
        try:
            # Get enhanced system statistics
            enhanced_stats = self.enhanced_system.get_exploration_statistics()
            
            # Combine with integration statistics
            combined_stats = {
                **self.exploration_stats,
                'enhanced_system_stats': enhanced_stats,
                'legacy_strategy_stats': {
                    'visited_positions': len(self.legacy_strategy.visited_positions),
                    'exploration_rate': self.legacy_strategy.exploration_rate,
                    'curiosity_weight': self.legacy_strategy.curiosity_weight
                }
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Failed to get exploration statistics: {e}")
            return self.exploration_stats
    
    def add_custom_strategy(self, strategy):
        """Add a custom exploration strategy."""
        try:
            self.enhanced_system.add_strategy(strategy)
            logger.info(f"Added custom strategy: {strategy.name}")
            
        except Exception as e:
            logger.error(f"Failed to add custom strategy: {e}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove an exploration strategy."""
        try:
            self.enhanced_system.remove_strategy(strategy_name)
            logger.info(f"Removed strategy: {strategy_name}")
            
        except Exception as e:
            logger.error(f"Failed to remove strategy: {e}")
    
    def _update_statistics(self, result: ExplorationResult):
        """Update exploration statistics."""
        self.exploration_stats['total_explorations'] += 1
        
        # Update running averages
        total = self.exploration_stats['total_explorations']
        current_avg_confidence = self.exploration_stats['avg_confidence']
        current_avg_exploration_value = self.exploration_stats['avg_exploration_value']
        
        # Clamp values to reasonable ranges
        confidence = max(0.0, min(1.0, result.confidence))
        exploration_value = max(0.0, min(1.0, result.exploration_value)) if result.exploration_value != float('inf') else 1.0
        
        self.exploration_stats['avg_confidence'] = (
            (current_avg_confidence * (total - 1) + confidence) / total
        )
        self.exploration_stats['avg_exploration_value'] = (
            (current_avg_exploration_value * (total - 1) + exploration_value) / total
        )
    
    def _fallback_exploration(self, position: Tuple[float, float, float], 
                             available_actions: List[int]) -> ExplorationResult:
        """Fallback exploration using legacy strategy."""
        try:
            # Create a simple state for fallback
            state = ExplorationState(
                position=position,
                energy_level=50.0,  # Default energy
                learning_progress=0.5,  # Default progress
                visited_positions=set(),
                success_history=[],
                action_history=[],
                context={}
            )
            
            # Use random exploration as fallback
            random_strategy = RandomExploration()
            result = random_strategy.explore(state, available_actions)
            
            logger.warning("Using fallback exploration strategy")
            return result
            
        except Exception as e:
            logger.error(f"Fallback exploration failed: {e}")
            # Ultimate fallback
            import random
            return ExplorationResult(
                action=random.choice(available_actions) if available_actions else 0,
                position=position,
                reward=0.0,
                confidence=0.1,
                exploration_value=0.1,
                strategy_used=type('ExplorationType', (), {'value': 'fallback'})(),
                search_algorithm=type('SearchAlgorithm', (), {'value': 'fallback'})(),
                metadata={'fallback': True, 'error': str(e)}
            )
    
    def _log_exploration(self, result: ExplorationResult):
        """Log exploration to database."""
        if not self.enable_database_storage or not self.integration:
            return
        
        try:
            log_data = {
                "action": result.action,
                "position": result.position,
                "confidence": result.confidence,
                "exploration_value": result.exploration_value,
                "strategy_used": result.strategy_used.value,
                "search_algorithm": result.search_algorithm.value,
                "metadata": result.metadata
            }
            
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Exploration performed: {result.strategy_used.value}",
                log_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log exploration: {e}")
    
    def _log_exploration_update(self, result: ExplorationResult, success: bool):
        """Log exploration update to database."""
        if not self.enable_database_storage or not self.integration:
            return
        
        try:
            log_data = {
                "action": result.action,
                "success": success,
                "strategy_used": result.strategy_used.value,
                "confidence": result.confidence
            }
            
            self.integration.log_system_event(
                self.LogLevel.INFO,
                self.Component.LEARNING_LOOP,
                f"Exploration updated: success={success}",
                log_data
            )
            
        except Exception as e:
            logger.error(f"Failed to log exploration update: {e}")


def create_exploration_integration(enable_database_storage: bool = True) -> ExplorationIntegration:
    """Create an exploration integration instance."""
    return ExplorationIntegration(enable_database_storage)
