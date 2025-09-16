#!/usr/bin/env python3
"""
Path Generator for Multi-Step Simulation Intelligence

This module implements tree search algorithms for generating probable future paths
in the simulation system. It supports BFS, DFS, hybrid, and Bayesian search methods
that learn which approaches work best for different game types.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class SearchMethod(Enum):
    """Available search methods for path generation."""
    BFS = "breadth_first_search"
    DFS = "depth_first_search"
    HYBRID = "hybrid_search"
    BAYESIAN = "bayesian_search"
    ADAPTIVE = "adaptive_search"

@dataclass
class PathNode:
    """A node in the search tree representing a game state."""
    state: Dict[str, Any]
    action: Optional[int] = None
    coordinates: Optional[Tuple[int, int]] = None
    parent: Optional['PathNode'] = None
    depth: int = 0
    path_cost: float = 0.0
    success_probability: float = 0.0
    confidence: float = 0.0
    children: List['PathNode'] = field(default_factory=list)
    visited: bool = False
    created_at: float = field(default_factory=time.time)

@dataclass
class SearchPath:
    """A complete path from root to leaf node."""
    nodes: List[PathNode]
    total_cost: float
    success_probability: float
    confidence: float
    search_method: SearchMethod
    depth: int
    created_at: float = field(default_factory=time.time)

class PathGenerator:
    """
    Generates probable future paths using multiple tree search algorithms.
    
    This class implements the core "imagination" capability by exploring
    multiple possible action sequences and scoring their likelihood of success.
    """
    
    def __init__(self, 
                 max_depth: int = 5,
                 max_paths: int = 20,
                 timeout: float = 0.5):
        self.max_depth = max_depth
        self.max_paths = max_paths
        self.timeout = timeout
        
        # Search method performance tracking
        self.method_performance = {
            SearchMethod.BFS: {'success_rate': 0.0, 'confidence': 0.0, 'usage_count': 0},
            SearchMethod.DFS: {'success_rate': 0.0, 'confidence': 0.0, 'usage_count': 0},
            SearchMethod.HYBRID: {'success_rate': 0.0, 'confidence': 0.0, 'usage_count': 0},
            SearchMethod.BAYESIAN: {'success_rate': 0.0, 'confidence': 0.0, 'usage_count': 0}
        }
        
        # Path generation statistics
        self.generation_stats = {
            'total_paths_generated': 0,
            'successful_paths': 0,
            'average_depth': 0.0,
            'method_usage': defaultdict(int)
        }
        
        # Pattern database for learning
        self.pattern_database = {
            'successful_patterns': [],
            'failed_patterns': [],
            'action_sequences': defaultdict(list),
            'state_transitions': defaultdict(list)
        }
        
        logger.info(f"PathGenerator initialized: max_depth={max_depth}, max_paths={max_paths}")
    
    def generate_paths(self, 
                      current_state: Dict[str, Any],
                      available_actions: List[int],
                      search_methods: Optional[List[SearchMethod]] = None,
                      context: Optional[Dict[str, Any]] = None) -> List[SearchPath]:
        """
        Generate multiple probable future paths using specified search methods.
        
        Args:
            current_state: Current game state
            available_actions: Available actions from this state
            search_methods: List of search methods to use (default: all)
            context: Additional context for path generation
            
        Returns:
            List of SearchPath objects representing probable futures
        """
        if search_methods is None:
            search_methods = [SearchMethod.BFS, SearchMethod.DFS, SearchMethod.HYBRID]
        
        start_time = time.time()
        all_paths = []
        
        # Generate paths using each method
        for method in search_methods:
            if time.time() - start_time > self.timeout:
                logger.warning(f"Path generation timeout after {time.time() - start_time:.2f}s")
                break
            
            try:
                method_paths = self._generate_paths_with_method(
                    current_state, available_actions, method, context
                )
                all_paths.extend(method_paths)
                
                # Update method usage statistics
                self.method_performance[method]['usage_count'] += 1
                self.generation_stats['method_usage'][method.value] += len(method_paths)
                
            except Exception as e:
                logger.error(f"Path generation failed with method {method}: {e}")
                # Add fallback simple paths
                fallback_paths = self._generate_simple_fallback_paths(current_state, available_actions, 5)
                all_paths.extend(fallback_paths)
                continue
        
        # If no paths generated, create simple fallback paths
        if not all_paths:
            all_paths = self._generate_simple_fallback_paths(current_state, available_actions, 10)
        
        # Sort paths by success probability and confidence
        all_paths.sort(key=lambda p: (p.success_probability, p.confidence), reverse=True)
        
        # Limit to max_paths
        if len(all_paths) > self.max_paths:
            all_paths = all_paths[:self.max_paths]
        
        # Update statistics
        self.generation_stats['total_paths_generated'] += len(all_paths)
        if all_paths:
            self.generation_stats['average_depth'] = np.mean([p.depth for p in all_paths])
        
        logger.debug(f"Generated {len(all_paths)} paths using {len(search_methods)} methods")
        return all_paths
    
    def _generate_paths_with_method(self, 
                                   current_state: Dict[str, Any],
                                   available_actions: List[int],
                                   method: SearchMethod,
                                   context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Generate paths using a specific search method."""
        
        if method == SearchMethod.BFS:
            return self._breadth_first_search(current_state, available_actions, context)
        elif method == SearchMethod.DFS:
            return self._depth_first_search(current_state, available_actions, context)
        elif method == SearchMethod.HYBRID:
            return self._hybrid_search(current_state, available_actions, context)
        elif method == SearchMethod.BAYESIAN:
            return self._bayesian_search(current_state, available_actions, context)
        else:
            raise ValueError(f"Unknown search method: {method}")
    
    def _breadth_first_search(self, 
                             current_state: Dict[str, Any],
                             available_actions: List[int],
                             context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Breadth-first search: explores all actions at each level before going deeper."""
        
        paths = []
        queue = deque()
        
        # Create root node
        root = PathNode(
            state=current_state.copy(),
            depth=0,
            path_cost=0.0,
            success_probability=1.0,
            confidence=1.0
        )
        queue.append(root)
        
        # Track visited states to avoid cycles
        visited_states = set()
        visited_states.add(self._hash_state(current_state))
        
        while queue and len(paths) < self.max_paths:
            current_node = queue.popleft()
            
            # If we've reached max depth, create a path
            if current_node.depth >= self.max_depth:
                path = self._create_path_from_node(current_node, SearchMethod.BFS)
                paths.append(path)
                continue
            
            # Generate children for this node
            children = self._generate_children(current_node, available_actions, context)
            
            for child in children:
                # Check for cycles
                state_hash = self._hash_state(child.state)
                if state_hash in visited_states:
                    continue
                
                visited_states.add(state_hash)
                current_node.children.append(child)
                queue.append(child)
        
        # Create paths from leaf nodes
        leaf_nodes = [node for node in self._get_all_nodes(root) if not node.children and node.depth > 0]
        for node in leaf_nodes:
            path = self._create_path_from_node(node, SearchMethod.BFS)
            paths.append(path)
        
        return paths[:self.max_paths]
    
    def _depth_first_search(self, 
                           current_state: Dict[str, Any],
                           available_actions: List[int],
                           context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Depth-first search: explores one path to maximum depth before backtracking."""
        
        paths = []
        stack = []
        
        # Create root node
        root = PathNode(
            state=current_state.copy(),
            depth=0,
            path_cost=0.0,
            success_probability=1.0,
            confidence=1.0
        )
        stack.append(root)
        
        # Track visited states to avoid cycles
        visited_states = set()
        visited_states.add(self._hash_state(current_state))
        
        while stack and len(paths) < self.max_paths:
            current_node = stack.pop()
            
            # If we've reached max depth, create a path
            if current_node.depth >= self.max_depth:
                path = self._create_path_from_node(current_node, SearchMethod.DFS)
                paths.append(path)
                continue
            
            # Generate children for this node
            children = self._generate_children(current_node, available_actions, context)
            
            # Add children to stack in reverse order (for left-to-right exploration)
            for child in reversed(children):
                # Check for cycles
                state_hash = self._hash_state(child.state)
                if state_hash in visited_states:
                    continue
                
                visited_states.add(state_hash)
                current_node.children.append(child)
                stack.append(child)
        
        # Create paths from leaf nodes
        leaf_nodes = [node for node in self._get_all_nodes(root) if not node.children and node.depth > 0]
        for node in leaf_nodes:
            path = self._create_path_from_node(node, SearchMethod.DFS)
            paths.append(path)
        
        return paths[:self.max_paths]
    
    def _hybrid_search(self, 
                      current_state: Dict[str, Any],
                      available_actions: List[int],
                      context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Hybrid search: combines BFS and DFS based on state characteristics."""
        
        # Determine search strategy based on state characteristics
        energy_level = current_state.get('energy', 100.0)
        learning_drive = current_state.get('learning_drive', 0.5)
        boredom_level = current_state.get('boredom_level', 0.0)
        
        # High energy + high learning drive = use DFS (explore deeply)
        # Low energy + high boredom = use BFS (explore broadly)
        if energy_level > 70 and learning_drive > 0.7:
            # Use DFS for deep exploration
            return self._depth_first_search(current_state, available_actions, context)
        elif energy_level < 30 or boredom_level > 0.8:
            # Use BFS for broad exploration
            return self._breadth_first_search(current_state, available_actions, context)
        else:
            # Use a combination: BFS for first few levels, then DFS
            return self._combined_search(current_state, available_actions, context)
    
    def _bayesian_search(self, 
                        current_state: Dict[str, Any],
                        available_actions: List[int],
                        context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Bayesian search: uses historical patterns to guide exploration."""
        
        paths = []
        
        # Get historical patterns for similar states
        similar_patterns = self._find_similar_patterns(current_state)
        
        # Generate paths based on successful patterns
        for pattern in similar_patterns[:5]:  # Top 5 most similar patterns
            path = self._generate_path_from_pattern(current_state, pattern, available_actions)
            if path:
                paths.append(path)
        
        # Fill remaining slots with random exploration
        remaining_slots = self.max_paths - len(paths)
        if remaining_slots > 0:
            random_paths = self._generate_random_paths(current_state, available_actions, remaining_slots)
            paths.extend(random_paths)
        
        return paths[:self.max_paths]
    
    def _combined_search(self, 
                        current_state: Dict[str, Any],
                        available_actions: List[int],
                        context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Combined search: BFS for first 3 levels, then DFS."""
        
        paths = []
        queue = deque()
        
        # Create root node
        root = PathNode(
            state=current_state.copy(),
            depth=0,
            path_cost=0.0,
            success_probability=1.0,
            confidence=1.0
        )
        queue.append(root)
        
        # BFS phase (first 3 levels)
        bfs_levels = 3
        visited_states = set()
        visited_states.add(self._hash_state(current_state))
        
        while queue and len(paths) < self.max_paths:
            current_node = queue.popleft()
            
            if current_node.depth >= self.max_depth:
                path = self._create_path_from_node(current_node, SearchMethod.HYBRID)
                paths.append(path)
                continue
            
            children = self._generate_children(current_node, available_actions, context)
            
            for child in children:
                state_hash = self._hash_state(child.state)
                if state_hash in visited_states:
                    continue
                
                visited_states.add(state_hash)
                current_node.children.append(child)
                
                # Use BFS for first few levels, then switch to DFS
                if current_node.depth < bfs_levels:
                    queue.append(child)
                else:
                    # Switch to DFS for deeper levels
                    dfs_paths = self._depth_first_search_from_node(child, available_actions, context)
                    paths.extend(dfs_paths)
        
        return paths[:self.max_paths]
    
    def _depth_first_search_from_node(self, 
                                    start_node: PathNode,
                                    available_actions: List[int],
                                    context: Optional[Dict[str, Any]]) -> List[SearchPath]:
        """Perform depth-first search starting from a specific node."""
        
        paths = []
        stack = [start_node]
        visited_states = set()
        visited_states.add(self._hash_state(start_node.state))
        
        while stack and len(paths) < self.max_paths:
            current_node = stack.pop()
            
            # If we've reached max depth, create a path
            if current_node.depth >= self.max_depth:
                path = self._create_path_from_node(current_node, SearchMethod.HYBRID)
                paths.append(path)
                continue
            
            # Generate children for this node
            children = self._generate_children(current_node, available_actions, context)
            
            # Add children to stack in reverse order (for left-to-right exploration)
            for child in reversed(children):
                # Check for cycles
                state_hash = self._hash_state(child.state)
                if state_hash in visited_states:
                    continue
                
                visited_states.add(state_hash)
                current_node.children.append(child)
                stack.append(child)
        
        # Create paths from leaf nodes
        leaf_nodes = [node for node in self._get_all_nodes(start_node) if not node.children and node.depth > start_node.depth]
        for node in leaf_nodes:
            path = self._create_path_from_node(node, SearchMethod.HYBRID)
            paths.append(path)
        
        return paths[:self.max_paths]
    
    def _generate_children(self, 
                          parent_node: PathNode,
                          available_actions: List[int],
                          context: Optional[Dict[str, Any]]) -> List[PathNode]:
        """Generate child nodes for a given parent node."""
        
        children = []
        
        for action in available_actions:
            # Generate coordinates for coordinate-based actions
            coordinates = None
            if action == 6:  # Coordinate action
                coordinates = self._generate_coordinates(parent_node.state, context)
            
            # Predict next state
            predicted_state = self._predict_next_state(parent_node.state, action, coordinates)
            
            # Calculate success probability and confidence
            success_prob = self._calculate_success_probability(predicted_state, action, coordinates)
            confidence = self._calculate_confidence(predicted_state, action, coordinates)
            
            # Calculate path cost
            path_cost = parent_node.path_cost + self._calculate_action_cost(action, coordinates)
            
            # Create child node
            child = PathNode(
                state=predicted_state,
                action=action,
                coordinates=coordinates,
                parent=parent_node,
                depth=parent_node.depth + 1,
                path_cost=path_cost,
                success_probability=success_prob,
                confidence=confidence
            )
            
            children.append(child)
        
        return children
    
    def _predict_next_state(self, 
                           current_state: Dict[str, Any],
                           action: int,
                           coordinates: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        """Predict the next state after taking an action."""
        
        predicted_state = current_state.copy()
        
        # Simple state prediction (this would be enhanced with the PredictiveCore)
        energy_cost = self._calculate_action_cost(action, coordinates)
        predicted_state['energy'] = max(0, current_state.get('energy', 100) - energy_cost)
        
        # Update action count
        predicted_state['action_count'] = current_state.get('action_count', 0) + 1
        
        # Update position for movement actions
        if action in [1, 2, 3, 4]:  # Movement actions
            position = predicted_state.get('position', [0.0, 0.0, 0.0])
            movement_delta = {
                1: [0, 0, -0.1],   # Up
                2: [0, 0, 0.1],    # Down
                3: [-0.1, 0, 0],   # Left
                4: [0.1, 0, 0]     # Right
            }
            
            if action in movement_delta:
                delta = movement_delta[action]
                predicted_state['position'] = [p + d for p, d in zip(position, delta)]
        
        # Update learning progress
        learning_gain = self._estimate_learning_gain(current_state, predicted_state, action)
        predicted_state['learning_progress'] = current_state.get('learning_progress', 0) + learning_gain
        
        return predicted_state
    
    def _calculate_success_probability(self, 
                                     state: Dict[str, Any],
                                     action: int,
                                     coordinates: Optional[Tuple[int, int]]) -> float:
        """Calculate the probability of success for a given state and action."""
        
        # Base probability
        base_prob = 0.5
        
        # Adjust based on energy level
        energy = state.get('energy', 100)
        if energy > 80:
            base_prob += 0.2
        elif energy < 20:
            base_prob -= 0.3
        
        # Adjust based on action type
        action_weights = {
            1: 0.8,  # Movement actions
            2: 0.8,
            3: 0.8,
            4: 0.8,
            5: 0.6,  # Interaction actions
            6: 0.4,  # Coordinate actions (more complex)
            7: 0.9   # Undo actions
        }
        
        action_weight = action_weights.get(action, 0.5)
        base_prob *= action_weight
        
        # Adjust based on coordinates for coordinate actions
        if action == 6 and coordinates:
            # Simple heuristic: center coordinates are more likely to succeed
            x, y = coordinates
            center_distance = abs(x - 32) + abs(y - 32)
            coord_bonus = max(0, 0.2 - (center_distance / 100))
            base_prob += coord_bonus
        
        return max(0.0, min(1.0, base_prob))
    
    def _calculate_confidence(self, 
                            state: Dict[str, Any],
                            action: int,
                            coordinates: Optional[Tuple[int, int]]) -> float:
        """Calculate confidence in the prediction."""
        
        # Base confidence
        confidence = 0.5
        
        # Higher confidence for simpler actions
        if action in [1, 2, 3, 4]:  # Movement actions
            confidence += 0.2
        elif action == 7:  # Undo actions
            confidence += 0.1
        elif action == 6:  # Coordinate actions
            confidence -= 0.1
        
        # Adjust based on energy level
        energy = state.get('energy', 100)
        if energy > 70:
            confidence += 0.1
        elif energy < 30:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_action_cost(self, action: int, coordinates: Optional[Tuple[int, int]]) -> float:
        """Calculate the cost of taking an action."""
        
        base_costs = {
            1: 0.5,  # Movement actions
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 1.0,  # Interaction actions
            6: 2.0,  # Coordinate actions
            7: 0.1   # Undo actions
        }
        
        cost = base_costs.get(action, 1.0)
        
        # Additional cost for coordinate actions based on distance from center
        if action == 6 and coordinates:
            x, y = coordinates
            center_distance = abs(x - 32) + abs(y - 32)
            cost += center_distance / 100
        
        return cost
    
    def _estimate_learning_gain(self, 
                               current_state: Dict[str, Any],
                               predicted_state: Dict[str, Any],
                               action: int) -> float:
        """Estimate learning gain from state transition."""
        
        gain = 0.0
        
        # Energy-based learning
        energy_change = predicted_state.get('energy', 0) - current_state.get('energy', 0)
        if energy_change > 0:
            gain += 0.1
        
        # Position-based learning
        current_pos = current_state.get('position', [0.0, 0.0, 0.0])
        predicted_pos = predicted_state.get('position', [0.0, 0.0, 0.0])
        if current_pos != predicted_pos:
            gain += 0.05
        
        # Action-specific learning
        if action == 6:  # Coordinate actions
            gain += 0.2
        
        return min(gain, 1.0)
    
    def _generate_coordinates(self, 
                            state: Dict[str, Any],
                            context: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        """Generate coordinates for coordinate-based actions."""
        
        # Simple coordinate generation (would be enhanced with pattern learning)
        if context and 'successful_coordinates' in context:
            # Use successful coordinates from context
            coords = context['successful_coordinates']
            if coords:
                return coords[0]  # Use first successful coordinate
        
        # Default: random coordinates
        return (np.random.randint(0, 64), np.random.randint(0, 64))
    
    def _create_path_from_node(self, node: PathNode, method: SearchMethod) -> SearchPath:
        """Create a SearchPath from a leaf node."""
        
        # Trace path from root to leaf
        path_nodes = []
        current = node
        while current is not None:
            path_nodes.insert(0, current)
            current = current.parent
        
        # Calculate path metrics
        total_cost = sum(node.path_cost for node in path_nodes)
        success_prob = np.prod([node.success_probability for node in path_nodes])
        confidence = np.mean([node.confidence for node in path_nodes])
        
        return SearchPath(
            nodes=path_nodes,
            total_cost=total_cost,
            success_probability=success_prob,
            confidence=confidence,
            search_method=method,
            depth=len(path_nodes) - 1
        )
    
    def _get_all_nodes(self, root: PathNode) -> List[PathNode]:
        """Get all nodes in the tree starting from root."""
        
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """Create a hash of the state for cycle detection."""
        
        # Simple state hashing (would be enhanced for better uniqueness)
        key_parts = []
        for key in sorted(state.keys()):
            if key in ['position', 'orientation', 'energy']:
                key_parts.append(f"{key}:{state[key]}")
        
        return hash(tuple(key_parts))
    
    def _find_similar_patterns(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar patterns in the pattern database."""
        
        # Simple pattern matching (would be enhanced with ML)
        similar_patterns = []
        
        for pattern in self.pattern_database['successful_patterns']:
            similarity = self._calculate_pattern_similarity(state, pattern)
            if similarity > 0.5:  # Threshold for similarity
                similar_patterns.append((pattern, similarity))
        
        # Sort by similarity
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        return [pattern for pattern, _ in similar_patterns]
    
    def _calculate_pattern_similarity(self, state: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate similarity between current state and a pattern."""
        
        # Simple similarity calculation (would be enhanced with ML)
        similarity = 0.0
        
        # Compare energy levels
        state_energy = state.get('energy', 100)
        pattern_energy = pattern.get('energy', 100)
        energy_sim = 1.0 - abs(state_energy - pattern_energy) / 100.0
        similarity += energy_sim * 0.3
        
        # Compare action counts
        state_actions = state.get('action_count', 0)
        pattern_actions = pattern.get('action_count', 0)
        action_sim = 1.0 - abs(state_actions - pattern_actions) / 100.0
        similarity += action_sim * 0.2
        
        # Compare other state features
        for key in ['learning_drive', 'boredom_level']:
            if key in state and key in pattern:
                val_sim = 1.0 - abs(state[key] - pattern[key])
                similarity += val_sim * 0.1
        
        return min(1.0, similarity)
    
    def _generate_path_from_pattern(self, 
                                  current_state: Dict[str, Any],
                                  pattern: Dict[str, Any],
                                  available_actions: List[int]) -> Optional[SearchPath]:
        """Generate a path based on a successful pattern."""
        
        # Extract action sequence from pattern
        action_sequence = pattern.get('action_sequence', [])
        if not action_sequence:
            return None
        
        # Create path nodes
        path_nodes = []
        current_state_copy = current_state.copy()
        
        for i, (action, coords) in enumerate(action_sequence):
            if action not in available_actions:
                continue
            
            # Predict next state
            predicted_state = self._predict_next_state(current_state_copy, action, coords)
            
            # Create node
            node = PathNode(
                state=predicted_state,
                action=action,
                coordinates=coords,
                parent=path_nodes[-1] if path_nodes else None,
                depth=i,
                path_cost=self._calculate_action_cost(action, coords),
                success_probability=self._calculate_success_probability(predicted_state, action, coords),
                confidence=self._calculate_confidence(predicted_state, action, coords)
            )
            
            path_nodes.append(node)
            current_state_copy = predicted_state
        
        if not path_nodes:
            return None
        
        # Create SearchPath
        total_cost = sum(node.path_cost for node in path_nodes)
        success_prob = np.prod([node.success_probability for node in path_nodes])
        confidence = np.mean([node.confidence for node in path_nodes])
        
        return SearchPath(
            nodes=path_nodes,
            total_cost=total_cost,
            success_probability=success_prob,
            confidence=confidence,
            search_method=SearchMethod.BAYESIAN,
            depth=len(path_nodes) - 1
        )
    
    def _generate_random_paths(self, 
                              current_state: Dict[str, Any],
                              available_actions: List[int],
                              count: int) -> List[SearchPath]:
        """Generate random paths for exploration."""
        
        paths = []
        
        for _ in range(count):
            path_nodes = []
            current_state_copy = current_state.copy()
            
            # Generate random path
            for depth in range(min(self.max_depth, 5)):  # Limit random paths to 5 steps
                action = np.random.choice(available_actions)
                coords = None
                
                if action == 6:
                    coords = (np.random.randint(0, 64), np.random.randint(0, 64))
                
                predicted_state = self._predict_next_state(current_state_copy, action, coords)
                
                node = PathNode(
                    state=predicted_state,
                    action=action,
                    coordinates=coords,
                    parent=path_nodes[-1] if path_nodes else None,
                    depth=depth,
                    path_cost=self._calculate_action_cost(action, coords),
                    success_probability=self._calculate_success_probability(predicted_state, action, coords),
                    confidence=self._calculate_confidence(predicted_state, action, coords)
                )
                
                path_nodes.append(node)
                current_state_copy = predicted_state
            
            if path_nodes:
                total_cost = sum(node.path_cost for node in path_nodes)
                success_prob = np.prod([node.success_probability for node in path_nodes])
                confidence = np.mean([node.confidence for node in path_nodes])
                
                path = SearchPath(
                    nodes=path_nodes,
                    total_cost=total_cost,
                    success_probability=success_prob,
                    confidence=confidence,
                    search_method=SearchMethod.BAYESIAN,
                    depth=len(path_nodes) - 1
                )
                paths.append(path)
        
        return paths
    
    def update_method_performance(self, 
                                method: SearchMethod,
                                actual_outcome: bool,
                                predicted_outcome: float):
        """Update performance statistics for a search method."""
        
        if method not in self.method_performance:
            return
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        current_success_rate = self.method_performance[method]['success_rate']
        
        if actual_outcome:
            new_success_rate = current_success_rate + alpha * (1.0 - current_success_rate)
        else:
            new_success_rate = current_success_rate + alpha * (0.0 - current_success_rate)
        
        self.method_performance[method]['success_rate'] = new_success_rate
        
        # Update confidence based on prediction accuracy
        prediction_error = abs(actual_outcome - predicted_outcome)
        current_confidence = self.method_performance[method]['confidence']
        new_confidence = current_confidence + alpha * (1.0 - prediction_error - current_confidence)
        
        self.method_performance[method]['confidence'] = max(0.0, min(1.0, new_confidence))
        
        logger.debug(f"Updated {method.value} performance: success_rate={new_success_rate:.3f}, confidence={new_confidence:.3f}")
    
    def get_best_method(self) -> SearchMethod:
        """Get the best performing search method."""
        
        best_method = SearchMethod.BFS
        best_score = 0.0
        
        for method, stats in self.method_performance.items():
            # Combined score: success_rate * confidence
            score = stats['success_rate'] * stats['confidence']
            if score > best_score:
                best_score = score
                best_method = method
        
        return best_method
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about path generation."""
        
        return {
            'generation_stats': dict(self.generation_stats),
            'method_performance': {
                method.value: stats for method, stats in self.method_performance.items()
            },
            'pattern_database_size': {
                'successful_patterns': len(self.pattern_database['successful_patterns']),
                'failed_patterns': len(self.pattern_database['failed_patterns']),
                'action_sequences': len(self.pattern_database['action_sequences']),
                'state_transitions': len(self.pattern_database['state_transitions'])
            }
        }
    
    def _generate_simple_fallback_paths(self, 
                                      current_state: Dict[str, Any],
                                      available_actions: List[int],
                                      count: int) -> List[SearchPath]:
        """Generate simple fallback paths when complex methods fail."""
        paths = []
        
        for i in range(min(count, len(available_actions))):
            action = available_actions[i % len(available_actions)]
            coords = None
            
            if action == 6:  # Coordinate action
                coords = (32, 32)  # Center coordinates
            
            # Create simple path with just one action
            predicted_state = self._predict_next_state(current_state, action, coords)
            
            node = PathNode(
                state=predicted_state,
                action=action,
                coordinates=coords,
                parent=None,
                depth=1,
                path_cost=self._calculate_action_cost(action, coords),
                success_probability=self._calculate_success_probability(predicted_state, action, coords),
                confidence=self._calculate_confidence(predicted_state, action, coords)
            )
            
            path = SearchPath(
                nodes=[node],
                total_cost=node.path_cost,
                success_probability=node.success_probability,
                confidence=node.confidence,
                search_method=SearchMethod.BFS,
                depth=1
            )
            
            paths.append(path)
        
        return paths
