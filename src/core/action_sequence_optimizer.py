#!/usr/bin/env python3
"""
Action Sequence Optimizer

Coordinates between Tree Evaluation Engine and OpenCV target detection to find
optimal action sequences for ARC-AGI gameplay, avoiding wasted moves and
maximizing strategic efficiency.

Key Features:
- Tree-based path planning with O(√t log t) complexity
- Integration with OpenCV target detection
- Wasted move prevention and oscillation avoidance
- Strategic ACTION6 coordinate targeting
- Real-time sequence optimization
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ActionSequenceResult:
    """Result of action sequence optimization."""
    optimal_sequence: List[int]
    target_coordinates: Optional[Tuple[int, int]]
    sequence_value: float
    wasted_moves_avoided: int
    reasoning: str
    confidence: float
    evaluation_time: float
    targets_reached: bool

@dataclass
class OptimizationConfig:
    """Configuration for action sequence optimization."""
    max_sequence_length: int = 20
    max_evaluation_time: float = 5.0
    confidence_threshold: float = 0.7
    wasted_move_penalty: float = 0.1
    strategic_action_bonus: float = 0.2
    target_priority_weight: float = 0.4

class ActionSequenceOptimizer:
    """
    Optimizes action sequences for ARC-AGI gameplay using tree evaluation
    and computer vision target detection.
    
    This class coordinates between the Tree Evaluation Engine and OpenCV
    target detection to find optimal action sequences that avoid wasted
    moves and maximize strategic efficiency.
    """
    
    def __init__(self, 
                 tree_engine=None,
                 opencv_extractor=None,
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize the Action Sequence Optimizer.
        
        Args:
            tree_engine: Tree Evaluation Engine instance
            opencv_extractor: OpenCV Feature Extractor instance
            config: Optimization configuration
        """
        self.tree_engine = tree_engine
        self.opencv_extractor = opencv_extractor
        self.config = config or OptimizationConfig()
        
        # Statistics tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'wasted_moves_avoided_total': 0,
            'average_sequence_length': 0.0,
            'average_confidence': 0.0,
            'total_evaluation_time': 0.0
        }
        
        logger.info("Action Sequence Optimizer initialized")
    
    def optimize_action_sequence(self, 
                               current_state: Dict[str, Any],
                               available_actions: List[int],
                               grid: Optional[List[List[int]]] = None,
                               game_id: str = "unknown") -> ActionSequenceResult:
        """
        Optimize action sequence for current state and available actions.
        
        This method combines tree evaluation with target detection to find
        the optimal action sequence that avoids wasted moves and targets
        strategic objectives.
        
        Args:
            current_state: Current game state
            available_actions: Available actions for current state
            grid: Optional grid for target detection
            game_id: Identifier for the game
            
        Returns:
            ActionSequenceResult containing optimal sequence and metadata
        """
        try:
            start_time = time.time()
            self.optimization_stats['total_optimizations'] += 1
            
            # Step 1: Identify actionable targets using OpenCV
            target_goals = []
            target_coordinates = None
            
            if grid and self.opencv_extractor:
                try:
                    # Get actionable targets from OpenCV
                    actionable_targets = self.opencv_extractor.identify_actionable_targets(grid, game_id)
                    
                    # Convert to target goals for tree evaluation
                    for target in actionable_targets:
                        if target.priority > 0.5:  # High-priority targets only
                            target_goals.append({
                                'type': 'coordinate',
                                'coordinates': target.coordinates,
                                'priority': target.priority,
                                'action_type': target.action_type,
                                'description': target.description
                            })
                    
                    # Get the highest priority target for ACTION6
                    if actionable_targets:
                        best_target = max(actionable_targets, key=lambda t: t.priority)
                        target_coordinates = best_target.coordinates
                        logger.info(f"Found {len(actionable_targets)} actionable targets, "
                                  f"best target at {target_coordinates} with priority {best_target.priority}")
                    
                except Exception as e:
                    logger.warning(f"Target detection failed: {e}")
                    target_goals = []
            else:
                logger.debug("No grid or OpenCV extractor available for target detection")
            
            # Step 2: Use tree evaluation to find optimal sequence
            if self.tree_engine and target_goals:
                try:
                    # Evaluate action sequences using tree evaluation
                    tree_result = self.tree_engine.evaluate_action_sequence_tree(
                        current_state=current_state,
                        target_goals=target_goals,
                        available_actions=available_actions,
                        max_sequence_length=self.config.max_sequence_length
                    )
                    
                    optimal_sequence = tree_result.get('optimal_sequence', [])
                    sequence_value = tree_result.get('sequence_value', 0.0)
                    wasted_moves_avoided = tree_result.get('wasted_moves_avoided', 0)
                    targets_reached = tree_result.get('target_reached', False)
                    reasoning = tree_result.get('reasoning', 'Tree evaluation completed')
                    
                except Exception as e:
                    logger.warning(f"Tree evaluation failed: {e}")
                    # Fallback to basic sequence
                    optimal_sequence = self._generate_fallback_sequence(available_actions, target_coordinates)
                    sequence_value = 0.5
                    wasted_moves_avoided = 0
                    targets_reached = False
                    reasoning = f"Fallback sequence due to tree evaluation failure: {e}"
            else:
                # No tree engine or targets - generate basic sequence
                optimal_sequence = self._generate_fallback_sequence(available_actions, target_coordinates)
                sequence_value = 0.3 if optimal_sequence else 0.0
                wasted_moves_avoided = 0
                targets_reached = False
                reasoning = "Basic sequence - no tree evaluation or targets available"
            
            # Step 3: Post-process sequence to ensure validity
            optimal_sequence = self._validate_and_optimize_sequence(
                optimal_sequence, available_actions, current_state
            )
            
            # Step 4: Calculate final metrics
            evaluation_time = time.time() - start_time
            confidence = min(1.0, sequence_value)
            
            # Update statistics
            self._update_statistics(optimal_sequence, confidence, evaluation_time, wasted_moves_avoided)
            
            # Create result
            result = ActionSequenceResult(
                optimal_sequence=optimal_sequence,
                target_coordinates=target_coordinates,
                sequence_value=sequence_value,
                wasted_moves_avoided=wasted_moves_avoided,
                reasoning=reasoning,
                confidence=confidence,
                evaluation_time=evaluation_time,
                targets_reached=targets_reached
            )
            
            logger.info(f"Action sequence optimization completed: "
                       f"sequence_length={len(optimal_sequence)}, "
                       f"confidence={confidence:.3f}, "
                       f"wasted_moves_avoided={wasted_moves_avoided}, "
                       f"evaluation_time={evaluation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Action sequence optimization failed: {e}")
            return ActionSequenceResult(
                optimal_sequence=[],
                target_coordinates=None,
                sequence_value=0.0,
                wasted_moves_avoided=0,
                reasoning=f"Optimization failed: {e}",
                confidence=0.0,
                evaluation_time=time.time() - start_time,
                targets_reached=False
            )
    
    def _generate_fallback_sequence(self, 
                                  available_actions: List[int],
                                  target_coordinates: Optional[Tuple[int, int]]) -> List[int]:
        """Generate a fallback sequence when tree evaluation is not available."""
        try:
            if not available_actions:
                logger.debug("No available actions for fallback sequence")
                return []
            
            # If we have target coordinates, try to use ACTION6
            if target_coordinates and 6 in available_actions:
                logger.debug(f"Using ACTION6 fallback for target {target_coordinates}")
                return [6]  # ACTION6 to target coordinates
            
            # Otherwise, use a simple exploration sequence
            # Avoid repeating the same action
            if len(available_actions) > 1:
                # Use first two different actions
                fallback = available_actions[:2]
                logger.debug(f"Using multi-action fallback: {fallback}")
                return fallback
            else:
                # Single action available
                fallback = available_actions[:1]
                logger.debug(f"Using single-action fallback: {fallback}")
                return fallback
                
        except Exception as e:
            logger.debug(f"Fallback sequence generation failed: {e}")
            fallback = available_actions[:1] if available_actions else []
            logger.debug(f"Exception fallback: {fallback}")
            return fallback
    
    def _validate_and_optimize_sequence(self, 
                                      sequence: List[int],
                                      available_actions: List[int],
                                      current_state: Dict[str, Any]) -> List[int]:
        """Validate and optimize the action sequence."""
        try:
            if not sequence:
                return []
            
            # Remove invalid actions
            valid_sequence = [action for action in sequence if action in available_actions]
            
            # Remove redundant pairs
            optimized_sequence = self._remove_redundant_pairs(valid_sequence)
            
            # Limit sequence length
            max_length = min(len(optimized_sequence), self.config.max_sequence_length)
            optimized_sequence = optimized_sequence[:max_length]
            
            return optimized_sequence
            
        except Exception as e:
            logger.debug(f"Sequence validation failed: {e}")
            return sequence
    
    def _remove_redundant_pairs(self, sequence: List[int]) -> List[int]:
        """Remove redundant action pairs from the sequence."""
        try:
            if len(sequence) < 2:
                return sequence
            
            # For very short sequences, be conservative about removing pairs
            if len(sequence) <= 3:
                # Only remove obvious redundancies (same action repeated)
                optimized = []
                for i, action in enumerate(sequence):
                    if i == 0 or action != sequence[i-1]:
                        optimized.append(action)
                return optimized
            
            optimized = []
            i = 0
            
            while i < len(sequence):
                current_action = sequence[i]
                
                # Check if next action would be redundant
                if (i + 1 < len(sequence) and 
                    self._is_redundant_pair(current_action, sequence[i + 1])):
                    # Skip the redundant pair
                    i += 2
                else:
                    # Keep the current action
                    optimized.append(current_action)
                    i += 1
            
            return optimized
            
        except Exception as e:
            logger.debug(f"Redundant pair removal failed: {e}")
            return sequence
    
    def _is_redundant_pair(self, action1: int, action2: int) -> bool:
        """Check if two actions are redundant (one undoes the other)."""
        # Define redundant action pairs - only truly redundant pairs
        redundant_pairs = {
            (1, 2): True,  # Move up, Move down (opposite directions)
            (2, 1): True,  # Move down, Move up (opposite directions)
            (3, 4): True,  # Move left, Move right (opposite directions)
            (4, 3): True,  # Move right, Move left (opposite directions)
            (5, 5): True,  # Same action repeated
        }
        
        return redundant_pairs.get((action1, action2), False)
    
    def _update_statistics(self, 
                          sequence: List[int],
                          confidence: float,
                          evaluation_time: float,
                          wasted_moves_avoided: int):
        """Update optimization statistics."""
        try:
            self.optimization_stats['successful_optimizations'] += 1
            self.optimization_stats['wasted_moves_avoided_total'] += wasted_moves_avoided
            self.optimization_stats['total_evaluation_time'] += evaluation_time
            
            # Update averages
            total = self.optimization_stats['successful_optimizations']
            self.optimization_stats['average_sequence_length'] = (
                (self.optimization_stats['average_sequence_length'] * (total - 1) + len(sequence)) / total
            )
            self.optimization_stats['average_confidence'] = (
                (self.optimization_stats['average_confidence'] * (total - 1) + confidence) / total
            )
            
        except Exception as e:
            logger.debug(f"Statistics update failed: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        return self.optimization_stats.copy()
    
    def reset_statistics(self):
        """Reset optimization statistics."""
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'wasted_moves_avoided_total': 0,
            'average_sequence_length': 0.0,
            'average_confidence': 0.0,
            'total_evaluation_time': 0.0
        }
        logger.info("Optimization statistics reset")
    
    def optimize_for_action6(self, 
                           current_state: Dict[str, Any],
                           available_actions: List[int],
                           grid: List[List[int]],
                           game_id: str = "unknown") -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Optimize specifically for ACTION6 commands with target coordinates.
        
        This is a convenience method that returns the best action and coordinates
        for ACTION6 commands.
        
        Args:
            current_state: Current game state
            available_actions: Available actions
            grid: Grid for target detection
            game_id: Game identifier
            
        Returns:
            Tuple of (best_action, target_coordinates)
        """
        try:
            # Get optimization result
            result = self.optimize_action_sequence(
                current_state=current_state,
                available_actions=available_actions,
                grid=grid,
                game_id=game_id
            )
            
            # Return best action and coordinates
            best_action = result.optimal_sequence[0] if result.optimal_sequence else available_actions[0]
            target_coordinates = result.target_coordinates
            
            return best_action, target_coordinates
            
        except Exception as e:
            logger.error(f"ACTION6 optimization failed: {e}")
            return available_actions[0] if available_actions else 0, None
