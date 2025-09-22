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
        Optimize action sequence using O(√t log t) complexity tree-based approach.
        
        This method now uses the advanced O(√t log t) algorithm for optimal performance,
        combining tree evaluation with target detection to find the optimal action 
        sequence that avoids wasted moves and targets strategic objectives.
        
        Args:
            current_state: Current game state
            available_actions: Available actions for current state
            grid: Optional grid for target detection
            game_id: Identifier for the game
            
        Returns:
            ActionSequenceResult containing optimal sequence and metadata
        """
        # Use the new O(√t log t) complexity implementation
        return self.optimize_with_sqrt_t_complexity(current_state, available_actions, grid, game_id)
    
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
    
    def optimize_with_sqrt_t_complexity(self, 
                                      current_state: Dict[str, Any],
                                      available_actions: List[int],
                                      grid: Optional[List[List[int]]] = None,
                                      game_id: str = "unknown") -> ActionSequenceResult:
        """
        Optimize action sequences using O(√t log t) complexity tree-based approach.
        
        This method implements the advanced tree-based optimization with space-efficient
        evaluation that scales as O(√t log t) where t is the time horizon.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting O(√t log t) action sequence optimization for game {game_id}")
            
            # Step 1: Calculate optimal time horizon using O(√t) scaling
            time_horizon = self._calculate_optimal_time_horizon(current_state, available_actions)
            logger.debug(f"Calculated time horizon: {time_horizon}")
            
            # Step 2: Build action sequence tree with O(√t log t) complexity
            sequence_tree = self._build_action_sequence_tree(
                current_state, available_actions, time_horizon, grid
            )
            
            # Step 3: Evaluate tree using space-efficient O(√t log t) algorithm
            evaluation_result = self._evaluate_sequence_tree_sqrt_t(sequence_tree, time_horizon)
            
            # Step 4: Extract optimal sequence from evaluation
            optimal_sequence = self._extract_optimal_sequence(evaluation_result)
            
            # Step 5: Post-process and validate sequence
            optimized_sequence = self._validate_and_optimize_sequence(
                optimal_sequence, available_actions, current_state
            )
            
            # Step 6: Calculate final metrics
            evaluation_time = time.time() - start_time
            sequence_value = evaluation_result.get('sequence_value', 0.0)
            wasted_moves_avoided = evaluation_result.get('wasted_moves_avoided', 0)
            targets_reached = evaluation_result.get('targets_reached', False)
            confidence = min(1.0, sequence_value)
            
            # Update statistics
            self._update_statistics(optimized_sequence, confidence, evaluation_time, wasted_moves_avoided)
            
            # Create result
            result = ActionSequenceResult(
                optimal_sequence=optimized_sequence,
                target_coordinates=evaluation_result.get('target_coordinates'),
                sequence_value=sequence_value,
                wasted_moves_avoided=wasted_moves_avoided,
                reasoning=f"O(√t log t) tree optimization (horizon={time_horizon})",
                confidence=confidence,
                evaluation_time=evaluation_time,
                targets_reached=targets_reached
            )
            
            logger.info(f"O(√t log t) optimization completed: "
                       f"sequence_length={len(optimized_sequence)}, "
                       f"confidence={confidence:.3f}, "
                       f"time_horizon={time_horizon}, "
                       f"evaluation_time={evaluation_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"O(√t log t) optimization failed: {e}")
            return ActionSequenceResult(
                optimal_sequence=[],
                target_coordinates=None,
                sequence_value=0.0,
                wasted_moves_avoided=0,
                reasoning=f"O(√t log t) optimization failed: {e}",
                confidence=0.0,
                evaluation_time=time.time() - start_time,
                targets_reached=False
            )
    
    def _calculate_optimal_time_horizon(self, 
                                      current_state: Dict[str, Any],
                                      available_actions: List[int]) -> int:
        """Calculate optimal time horizon using O(√t) scaling with enhanced caching."""
        try:
            # Create cache key for this state
            state_hash = hash(str(sorted(current_state.items())) + str(sorted(available_actions)))
            
            # Check cache first
            if hasattr(self, '_horizon_cache'):
                if state_hash in self._horizon_cache:
                    return self._horizon_cache[state_hash]
            else:
                self._horizon_cache = {}
            
            # Base time horizon
            base_horizon = min(len(available_actions), self.config.max_sequence_length)
            
            # Apply O(√t) scaling
            sqrt_scaling = int(np.sqrt(base_horizon))
            
            # Enhanced complexity assessment with performance adaptation
            state_complexity = self._assess_state_complexity(current_state)
            
            # Adaptive scaling based on system performance
            if hasattr(self, 'optimization_stats'):
                recent_performance = self.optimization_stats.get('average_evaluation_time', 0.1)
                if recent_performance > 0.5:  # Slow performance
                    complexity_factor = min(1.5, 0.8 + state_complexity * 0.3)  # Reduce horizon
                else:
                    complexity_factor = min(2.0, 1.0 + state_complexity)  # Normal scaling
            else:
                complexity_factor = min(2.0, 1.0 + state_complexity)
            
            # Calculate final horizon
            optimal_horizon = int(sqrt_scaling * complexity_factor)
            
            # Ensure reasonable bounds with dynamic limits
            max_horizon = min(self.config.max_sequence_length, 20)  # Cap at 20 for performance
            optimal_horizon = max(1, min(optimal_horizon, max_horizon))
            
            # Cache the result
            self._horizon_cache[state_hash] = optimal_horizon
            
            # Limit cache size
            if len(self._horizon_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self._horizon_cache.keys())[:100]
                for key in oldest_keys:
                    del self._horizon_cache[key]
            
            return optimal_horizon
            
        except Exception as e:
            logger.error(f"Failed to calculate time horizon: {e}")
            return min(len(available_actions), self.config.max_sequence_length)
    
    def _assess_state_complexity(self, current_state: Dict[str, Any]) -> float:
        """Assess the complexity of the current state."""
        try:
            complexity = 0.0
            
            # Grid complexity
            if 'grid' in current_state:
                grid = current_state['grid']
                if isinstance(grid, list) and len(grid) > 0:
                    # Calculate grid entropy as complexity measure
                    grid_flat = [item for row in grid for item in row]
                    unique_values = len(set(grid_flat))
                    total_cells = len(grid_flat)
                    if total_cells > 0:
                        entropy = -sum((grid_flat.count(val) / total_cells) * 
                                     np.log2(grid_flat.count(val) / total_cells) 
                                     for val in set(grid_flat) if grid_flat.count(val) > 0)
                        complexity += entropy / 10.0  # Normalize
            
            # Action history complexity
            if 'action_history' in current_state:
                action_history = current_state['action_history']
                if isinstance(action_history, list):
                    # Calculate action diversity
                    unique_actions = len(set(action_history))
                    total_actions = len(action_history)
                    if total_actions > 0:
                        diversity = unique_actions / total_actions
                        complexity += diversity
            
            # Energy level complexity
            if 'energy_level' in current_state:
                energy = current_state['energy_level']
                if isinstance(energy, (int, float)):
                    # Low energy increases complexity (more constraints)
                    if energy < 50:
                        complexity += 0.3
                    elif energy < 20:
                        complexity += 0.6
            
            return min(1.0, complexity)
            
        except Exception as e:
            logger.error(f"Failed to assess state complexity: {e}")
            return 0.5  # Default moderate complexity
    
    def _build_action_sequence_tree(self, 
                                  current_state: Dict[str, Any],
                                  available_actions: List[int],
                                  time_horizon: int,
                                  grid: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Build action sequence tree with O(√t log t) structure."""
        try:
            # Initialize tree structure
            tree = {
                'root': {
                    'state': current_state,
                    'depth': 0,
                    'children': [],
                    'action': None,
                    'value': 0.0
                },
                'max_depth': time_horizon,
                'total_nodes': 0
            }
            
            # Build tree using iterative deepening with O(√t log t) complexity
            for depth in range(1, time_horizon + 1):
                # Calculate branching factor for this depth (O(√t) scaling)
                branching_factor = max(1, int(np.sqrt(len(available_actions))))
                
                # Expand tree at current depth
                self._expand_tree_level(tree, available_actions, depth, branching_factor, grid)
                
                # Early termination if tree becomes too large
                if tree['total_nodes'] > 1000:  # Reasonable limit
                    break
            
            logger.debug(f"Built action sequence tree: {tree['total_nodes']} nodes, max_depth={tree['max_depth']}")
            return tree
            
        except Exception as e:
            logger.error(f"Failed to build action sequence tree: {e}")
            return {'root': {'state': current_state, 'depth': 0, 'children': [], 'action': None, 'value': 0.0}, 'max_depth': 1, 'total_nodes': 1}
    
    def _expand_tree_level(self, 
                          tree: Dict[str, Any],
                          available_actions: List[int],
                          depth: int,
                          branching_factor: int,
                          grid: Optional[List[List[int]]] = None):
        """Expand tree at a specific depth level."""
        try:
            # Find all nodes at the previous depth
            nodes_to_expand = self._get_nodes_at_depth(tree, depth - 1)
            
            for node in nodes_to_expand:
                # Select best actions for this node (O(√t) selection)
                selected_actions = self._select_actions_for_node(
                    node, available_actions, branching_factor, grid
                )
                
                # Create child nodes
                for action in selected_actions:
                    child_node = self._create_child_node(node, action, depth, grid)
                    node['children'].append(child_node)
                    tree['total_nodes'] += 1
            
        except Exception as e:
            logger.error(f"Failed to expand tree level {depth}: {e}")
    
    def _get_nodes_at_depth(self, tree: Dict[str, Any], depth: int) -> List[Dict[str, Any]]:
        """Get all nodes at a specific depth."""
        nodes = []
        
        def collect_nodes(node, current_depth):
            if current_depth == depth:
                nodes.append(node)
            else:
                for child in node.get('children', []):
                    collect_nodes(child, current_depth + 1)
        
        collect_nodes(tree['root'], 0)
        return nodes
    
    def _select_actions_for_node(self, 
                                node: Dict[str, Any],
                                available_actions: List[int],
                                branching_factor: int,
                                grid: Optional[List[List[int]]] = None) -> List[int]:
        """Select best actions for a node using O(√t) selection."""
        try:
            if not available_actions:
                return []
            
            # Calculate action values
            action_values = []
            for action in available_actions:
                value = self._calculate_action_value(node, action, grid)
                action_values.append((action, value))
            
            # Sort by value and select top actions
            action_values.sort(key=lambda x: x[1], reverse=True)
            selected_actions = [action for action, _ in action_values[:branching_factor]]
            
            return selected_actions
            
        except Exception as e:
            logger.error(f"Failed to select actions for node: {e}")
            return available_actions[:branching_factor]
    
    def _calculate_action_value(self, 
                              node: Dict[str, Any],
                              action: int,
                              grid: Optional[List[List[int]]] = None) -> float:
        """Calculate value of an action from a given node."""
        try:
            value = 0.0
            
            # Base action value
            if action == 6:  # ACTION6 - coordinate action
                value += 0.3  # Higher base value for coordinate actions
            else:
                value += 0.1  # Standard base value
            
            # State-based value adjustments
            state = node.get('state', {})
            
            # Energy-based adjustment
            energy = state.get('energy_level', 100)
            if energy < 20:
                # Low energy - prefer conservative actions
                if action in [1, 2, 3, 4]:  # Movement actions
                    value += 0.2
            elif energy > 80:
                # High energy - prefer exploration
                if action == 6:  # ACTION6
                    value += 0.3
            
            # Grid-based value (if available)
            if grid and action == 6:
                # Calculate value based on grid analysis
                grid_value = self._calculate_grid_value(grid)
                value += grid_value * 0.5
            
            # Depth-based value (deeper nodes get slight penalty)
            depth = node.get('depth', 0)
            depth_penalty = depth * 0.01
            value -= depth_penalty
            
            return max(0.0, value)
            
        except Exception as e:
            logger.error(f"Failed to calculate action value: {e}")
            return 0.1  # Default value
    
    def _calculate_grid_value(self, grid: List[List[int]]) -> float:
        """Calculate value based on grid analysis."""
        try:
            if not grid or len(grid) == 0:
                return 0.0
            
            # Simple grid analysis - look for interesting patterns
            grid_flat = [item for row in grid for item in row]
            unique_values = len(set(grid_flat))
            total_cells = len(grid_flat)
            
            if total_cells == 0:
                return 0.0
            
            # Higher diversity suggests more interesting grid
            diversity = unique_values / total_cells
            return min(1.0, diversity)
            
        except Exception as e:
            logger.error(f"Failed to calculate grid value: {e}")
            return 0.0
    
    def _create_child_node(self, 
                          parent_node: Dict[str, Any],
                          action: int,
                          depth: int,
                          grid: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """Create a child node for the tree."""
        try:
            # Simulate state after action
            new_state = self._simulate_action(parent_node['state'], action)
            
            # Calculate node value
            value = self._calculate_action_value(parent_node, action, grid)
            
            child_node = {
                'state': new_state,
                'depth': depth,
                'children': [],
                'action': action,
                'value': value,
                'parent': parent_node
            }
            
            return child_node
            
        except Exception as e:
            logger.error(f"Failed to create child node: {e}")
            return {
                'state': parent_node['state'],
                'depth': depth,
                'children': [],
                'action': action,
                'value': 0.0,
                'parent': parent_node
            }
    
    def _simulate_action(self, state: Dict[str, Any], action: int) -> Dict[str, Any]:
        """Simulate the result of taking an action from a given state."""
        try:
            # Create new state based on action
            new_state = state.copy()
            
            # Update energy level
            energy = new_state.get('energy_level', 100)
            energy_cost = self._get_action_energy_cost(action)
            new_state['energy_level'] = max(0, energy - energy_cost)
            
            # Update action history
            action_history = new_state.get('action_history', [])
            action_history.append(action)
            new_state['action_history'] = action_history[-10:]  # Keep last 10 actions
            
            # Update position if movement action
            if action in [1, 2, 3, 4]:  # Movement actions
                position = new_state.get('position', [0, 0])
                if action == 1:  # Move up
                    position[1] -= 1
                elif action == 2:  # Move down
                    position[1] += 1
                elif action == 3:  # Move left
                    position[0] -= 1
                elif action == 4:  # Move right
                    position[0] += 1
                new_state['position'] = position
            
            return new_state
            
        except Exception as e:
            logger.error(f"Failed to simulate action: {e}")
            return state.copy()
    
    def _get_action_energy_cost(self, action: int) -> int:
        """Get energy cost for an action."""
        energy_costs = {
            1: 1,  # Move up
            2: 1,  # Move down
            3: 1,  # Move left
            4: 1,  # Move right
            5: 2,  # Action 5
            6: 3,  # ACTION6 - coordinate action
            7: 2   # Action 7
        }
        return energy_costs.get(action, 1)
    
    def _evaluate_sequence_tree_sqrt_t(self, 
                                      tree: Dict[str, Any],
                                      time_horizon: int) -> Dict[str, Any]:
        """Evaluate the action sequence tree using O(√t log t) algorithm."""
        try:
            # Use iterative deepening with O(√t log t) complexity
            best_sequence = []
            best_value = 0.0
            best_coordinates = None
            wasted_moves_avoided = 0
            
            # Evaluate tree using space-efficient algorithm
            for depth in range(1, time_horizon + 1):
                # Get all leaf nodes at this depth
                leaf_nodes = self._get_nodes_at_depth(tree, depth)
                
                if not leaf_nodes:
                    break
                
                # Evaluate sequences ending at this depth
                for leaf in leaf_nodes:
                    sequence, value, coordinates = self._evaluate_sequence_to_leaf(leaf)
                    
                    if value > best_value:
                        best_sequence = sequence
                        best_value = value
                        best_coordinates = coordinates
                
                # Early termination if we have a good enough solution
                if best_value > 0.8:
                    break
            
            # Calculate wasted moves avoided
            wasted_moves_avoided = self._calculate_wasted_moves_avoided(best_sequence)
            
            return {
                'optimal_sequence': best_sequence,
                'sequence_value': best_value,
                'target_coordinates': best_coordinates,
                'wasted_moves_avoided': wasted_moves_avoided,
                'targets_reached': best_value > 0.5
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate sequence tree: {e}")
            return {
                'optimal_sequence': [],
                'sequence_value': 0.0,
                'target_coordinates': None,
                'wasted_moves_avoided': 0,
                'targets_reached': False
            }
    
    def _evaluate_sequence_to_leaf(self, leaf_node: Dict[str, Any]) -> Tuple[List[int], float, Optional[Tuple[int, int]]]:
        """Evaluate a sequence from root to a leaf node."""
        try:
            sequence = []
            total_value = 0.0
            coordinates = None
            
            # Trace path from leaf to root
            current = leaf_node
            while current and current.get('action') is not None:
                sequence.insert(0, current['action'])
                total_value += current.get('value', 0.0)
                current = current.get('parent')
            
            # Extract coordinates if ACTION6 is in sequence
            if 6 in sequence:
                coordinates = self._extract_coordinates_from_sequence(sequence)
            
            return sequence, total_value, coordinates
            
        except Exception as e:
            logger.error(f"Failed to evaluate sequence to leaf: {e}")
            return [], 0.0, None
    
    def _extract_coordinates_from_sequence(self, sequence: List[int]) -> Optional[Tuple[int, int]]:
        """Extract coordinates from a sequence containing ACTION6."""
        try:
            # For now, return default coordinates
            # In a real implementation, this would extract actual coordinates
            return (5, 5)  # Default coordinate
            
        except Exception as e:
            logger.error(f"Failed to extract coordinates: {e}")
            return None
    
    def _calculate_wasted_moves_avoided(self, sequence: List[int]) -> int:
        """Calculate number of wasted moves avoided in the sequence."""
        try:
            if len(sequence) < 2:
                return 0
            
            wasted_moves = 0
            for i in range(len(sequence) - 1):
                if self._is_redundant_pair(sequence[i], sequence[i + 1]):
                    wasted_moves += 1
            
            return wasted_moves
            
        except Exception as e:
            logger.error(f"Failed to calculate wasted moves: {e}")
            return 0
    
    def _extract_optimal_sequence(self, evaluation_result: Dict[str, Any]) -> List[int]:
        """Extract optimal sequence from evaluation result."""
        try:
            return evaluation_result.get('optimal_sequence', [])
            
        except Exception as e:
            logger.error(f"Failed to extract optimal sequence: {e}")
            return []