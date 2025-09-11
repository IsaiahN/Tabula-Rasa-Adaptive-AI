#!/usr/bin/env python3
"""
Hypothesis Generator for Architect System

This module implements the hypothesis generation system that creates
"what-if" scenarios for the Predictive Core to simulate. This transforms
the Architect from a parameter tweaker to an imagination engine.
"""

import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .simulation_models import (
    SimulationHypothesis, SimulationContext, HypothesisType
)

logger = logging.getLogger(__name__)

class SimulationHypothesisGenerator:
    """
    Generates strategic hypotheses for multi-step simulation.
    Transforms the Architect from parameter tweaker to imagination engine.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Hypothesis generation weights
        self.visual_weight = self.config.get('visual_hypothesis_weight', 0.3)
        self.memory_weight = self.config.get('memory_hypothesis_weight', 0.25)
        self.exploration_weight = self.config.get('exploration_hypothesis_weight', 0.2)
        self.energy_weight = self.config.get('energy_hypothesis_weight', 0.15)
        self.learning_weight = self.config.get('learning_hypothesis_weight', 0.1)
        
        # Action patterns for different hypothesis types
        self.action_patterns = {
            'movement_sequence': [1, 2, 3, 4],  # Basic movement
            'interaction_sequence': [5, 6, 5],  # Interaction pattern
            'exploration_sequence': [1, 3, 2, 4],  # Systematic exploration
            'energy_conservation': [7, 1, 7, 2],  # Undo-heavy pattern
            'coordinate_focused': [6, 6, 6],  # Coordinate-heavy pattern
            'mixed_strategy': [1, 5, 2, 6, 3, 7]  # Mixed approach
        }
        
        # Coordinate strategies
        self.coordinate_strategies = {
            'corners': [(0, 0), (63, 0), (0, 63), (63, 63)],
            'center': [(32, 32)],
            'edges': [(0, 32), (63, 32), (32, 0), (32, 63)],
            'random': None  # Will be generated randomly
        }
        
        logger.info("Simulation Hypothesis Generator initialized")
    
    def generate_simulation_hypotheses(self, 
                                     context: SimulationContext,
                                     max_hypotheses: int = 5) -> List[SimulationHypothesis]:
        """
        Generate multiple "what-if" scenarios for the Predictive Core to simulate.
        
        Examples:
        - "What if we prioritize movement towards the green object for 5 steps?"
        - "What if we systematically explore the right edge for 3 steps?"
        - "What if we use ACTION6 on all corner coordinates in sequence?"
        """
        
        hypotheses = []
        
        # Generate different types of hypotheses based on context
        if context.frame_analysis:
            hypotheses.extend(self._generate_visual_hypotheses(context))
        
        if context.memory_patterns:
            hypotheses.extend(self._generate_memory_hypotheses(context))
            
        hypotheses.extend(self._generate_exploration_hypotheses(context))
        
        if context.energy_level < 50:
            hypotheses.extend(self._generate_energy_hypotheses(context))
        
        if context.learning_drive > 0.7:
            hypotheses.extend(self._generate_learning_hypotheses(context))
        
        # Add strategy-based hypotheses if we have successful patterns
        if context.success_history:
            hypotheses.extend(self._generate_strategy_hypotheses(context))
        
        # Sort by priority and return top hypotheses
        hypotheses.sort(key=lambda h: h.priority, reverse=True)
        return hypotheses[:max_hypotheses]
    
    def _generate_visual_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on visual analysis."""
        hypotheses = []
        
        if not context.frame_analysis:
            return hypotheses
        
        frame_analysis = context.frame_analysis
        
        # Object-based targeting hypothesis
        if 'object_count' in frame_analysis and frame_analysis['object_count'] > 0:
            # Generate coordinate-based interaction hypothesis
            coords = self._generate_target_coordinates(frame_analysis, context)
            if coords:
                action_sequence = [(6, coord) for coord in coords[:3]]  # ACTION6 with coordinates
                
                hypothesis = SimulationHypothesis(
                    name="visual_object_targeting",
                    description="Target visible objects with coordinate actions",
                    hypothesis_type=HypothesisType.VISUAL_TARGETING,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=self.visual_weight * 0.8,
                    expected_outcome="Interact with visible objects",
                    energy_cost=len(action_sequence) * 2.0,
                    learning_potential=0.7,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'frame_analysis': frame_analysis
                    }
                )
                hypotheses.append(hypothesis)
        
        # Edge exploration hypothesis
        if 'edge_detection' in frame_analysis:
            edge_coords = self._generate_edge_coordinates(context)
            if edge_coords:
                action_sequence = [(6, coord) for coord in edge_coords[:2]]
                
                hypothesis = SimulationHypothesis(
                    name="visual_edge_exploration",
                    description="Explore detected edges with coordinate actions",
                    hypothesis_type=HypothesisType.VISUAL_TARGETING,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=self.visual_weight * 0.6,
                    expected_outcome="Explore edge boundaries",
                    energy_cost=len(action_sequence) * 2.0,
                    learning_potential=0.5,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'frame_analysis': frame_analysis
                    }
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_memory_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on memory patterns."""
        hypotheses = []
        
        if not context.memory_patterns:
            return hypotheses
        
        memory_patterns = context.memory_patterns
        
        # Successful action pattern hypothesis
        if 'successful_actions' in memory_patterns:
            successful_actions = memory_patterns['successful_actions']
            if successful_actions:
                # Create hypothesis based on successful patterns
                action_sequence = self._create_sequence_from_pattern(successful_actions[:3])
                
                hypothesis = SimulationHypothesis(
                    name="memory_successful_pattern",
                    description="Repeat previously successful action pattern",
                    hypothesis_type=HypothesisType.MEMORY_GUIDED,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=self.memory_weight * 0.9,
                    expected_outcome="Repeat successful pattern",
                    energy_cost=len(action_sequence) * 1.5,
                    learning_potential=0.6,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'memory_patterns': memory_patterns
                    }
                )
                hypotheses.append(hypothesis)
        
        # Coordinate success zone hypothesis
        if 'successful_coordinates' in memory_patterns:
            successful_coords = memory_patterns['successful_coordinates']
            if successful_coords:
                action_sequence = [(6, coord) for coord in successful_coords[:2]]
                
                hypothesis = SimulationHypothesis(
                    name="memory_coordinate_success",
                    description="Use previously successful coordinates",
                    hypothesis_type=HypothesisType.MEMORY_GUIDED,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=self.memory_weight * 0.8,
                    expected_outcome="Use known successful coordinates",
                    energy_cost=len(action_sequence) * 2.0,
                    learning_potential=0.4,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'memory_patterns': memory_patterns
                    }
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_exploration_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate exploration-based hypotheses."""
        hypotheses = []
        
        # Systematic movement exploration
        movement_sequence = self.action_patterns['exploration_sequence']
        available_movements = [a for a in movement_sequence if a in context.available_actions]
        
        if available_movements:
            action_sequence = [(action, None) for action in available_movements[:4]]
            
            hypothesis = SimulationHypothesis(
                name="systematic_exploration",
                description="Systematic movement exploration pattern",
                hypothesis_type=HypothesisType.EXPLORATION,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.exploration_weight * 0.7,
                expected_outcome="Explore environment systematically",
                energy_cost=len(action_sequence) * 0.5,
                learning_potential=0.3,
                context_requirements={
                    'available_actions': context.available_actions
                }
            )
            hypotheses.append(hypothesis)
        
        # Coordinate exploration
        if 6 in context.available_actions:
            coords = self._generate_exploration_coordinates(context)
            action_sequence = [(6, coord) for coord in coords[:3]]
            
            hypothesis = SimulationHypothesis(
                name="coordinate_exploration",
                description="Explore different coordinate regions",
                hypothesis_type=HypothesisType.EXPLORATION,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.exploration_weight * 0.6,
                expected_outcome="Explore coordinate space",
                energy_cost=len(action_sequence) * 2.0,
                learning_potential=0.5,
                context_requirements={
                    'available_actions': context.available_actions
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_energy_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate energy conservation hypotheses."""
        hypotheses = []
        
        # Energy conservation pattern
        if 7 in context.available_actions:  # Undo action available
            action_sequence = [(7, None), (1, None), (7, None), (2, None)]  # Undo-heavy pattern
            
            hypothesis = SimulationHypothesis(
                name="energy_conservation",
                description="Energy conservation with undo actions",
                hypothesis_type=HypothesisType.ENERGY_OPTIMIZATION,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.energy_weight * 0.9,
                expected_outcome="Conserve energy with undo actions",
                energy_cost=len(action_sequence) * 0.1,  # Very low energy cost
                learning_potential=0.2,
                context_requirements={
                    'available_actions': context.available_actions,
                    'energy_level': context.energy_level
                }
            )
            hypotheses.append(hypothesis)
        
        # Minimal action pattern
        minimal_actions = [a for a in [1, 2, 3, 4] if a in context.available_actions]
        if minimal_actions:
            action_sequence = [(action, None) for action in minimal_actions[:2]]
            
            hypothesis = SimulationHypothesis(
                name="minimal_energy_usage",
                description="Minimal energy usage with simple actions",
                hypothesis_type=HypothesisType.ENERGY_OPTIMIZATION,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.energy_weight * 0.7,
                expected_outcome="Minimal energy consumption",
                energy_cost=len(action_sequence) * 0.5,
                learning_potential=0.1,
                context_requirements={
                    'available_actions': context.available_actions,
                    'energy_level': context.energy_level
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_learning_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate learning-focused hypotheses."""
        hypotheses = []
        
        # Mixed strategy for learning
        mixed_actions = [a for a in self.action_patterns['mixed_strategy'] if a in context.available_actions]
        if mixed_actions:
            action_sequence = [(action, None) for action in mixed_actions[:4]]
            
            hypothesis = SimulationHypothesis(
                name="learning_mixed_strategy",
                description="Mixed action strategy for learning",
                hypothesis_type=HypothesisType.LEARNING_FOCUSED,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.learning_weight * 0.8,
                expected_outcome="Learn from diverse actions",
                energy_cost=len(action_sequence) * 1.0,
                learning_potential=0.8,
                context_requirements={
                    'available_actions': context.available_actions,
                    'learning_drive': context.learning_drive
                }
            )
            hypotheses.append(hypothesis)
        
        # Coordinate learning hypothesis
        if 6 in context.available_actions:
            coords = self._generate_learning_coordinates(context)
            action_sequence = [(6, coord) for coord in coords[:3]]
            
            hypothesis = SimulationHypothesis(
                name="coordinate_learning",
                description="Learn coordinate interactions",
                hypothesis_type=HypothesisType.LEARNING_FOCUSED,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=self.learning_weight * 0.7,
                expected_outcome="Learn coordinate effects",
                energy_cost=len(action_sequence) * 2.0,
                learning_potential=0.9,
                context_requirements={
                    'available_actions': context.available_actions,
                    'learning_drive': context.learning_drive
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_strategy_hypotheses(self, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on successful strategies."""
        hypotheses = []
        
        if not context.success_history:
            return hypotheses
        
        # Analyze recent successes
        recent_successes = context.success_history[-5:]  # Last 5 successes
        
        # Find common patterns in successful actions
        success_patterns = {}
        for success in recent_successes:
            actions = success.get('actions', [])
            for action in actions:
                success_patterns[action] = success_patterns.get(action, 0) + 1
        
        # Create hypothesis based on most successful actions
        if success_patterns:
            top_actions = sorted(success_patterns.items(), key=lambda x: x[1], reverse=True)
            action_sequence = [(action, None) for action, _ in top_actions[:3]]
            
            hypothesis = SimulationHypothesis(
                name="strategy_success_pattern",
                description="Use recently successful action pattern",
                hypothesis_type=HypothesisType.STRATEGY_RETRIEVAL,
                action_sequence=action_sequence,
                simulation_depth=len(action_sequence),
                priority=0.8,  # High priority for strategy-based hypotheses
                expected_outcome="Repeat recent success pattern",
                energy_cost=len(action_sequence) * 1.0,
                learning_potential=0.6,
                context_requirements={
                    'available_actions': context.available_actions,
                    'success_history': recent_successes
                }
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_target_coordinates(self, frame_analysis: Dict[str, Any], context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate target coordinates based on visual analysis."""
        coords = []
        
        # Simple coordinate generation based on object detection
        if 'object_count' in frame_analysis:
            object_count = frame_analysis['object_count']
            
            # Generate coordinates in a grid pattern
            grid_size = min(8, max(2, int(np.sqrt(object_count))))
            for i in range(min(3, object_count)):
                x = (i * 64) // grid_size
                y = ((i * 64) // grid_size) % 64
                coords.append((x, y))
        
        return coords
    
    def _generate_edge_coordinates(self, context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates for edge exploration."""
        return [
            (0, 32),   # Left edge
            (63, 32),  # Right edge
            (32, 0),   # Top edge
            (32, 63)   # Bottom edge
        ]
    
    def _generate_exploration_coordinates(self, context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates for systematic exploration."""
        coords = []
        
        # Use different coordinate strategies
        strategies = ['corners', 'center', 'edges']
        for strategy in strategies:
            if strategy in self.coordinate_strategies:
                strategy_coords = self.coordinate_strategies[strategy]
                if strategy_coords:
                    coords.extend(strategy_coords[:2])  # Take first 2 from each strategy
        
        return coords
    
    def _generate_learning_coordinates(self, context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates optimized for learning."""
        coords = []
        
        # Mix of different coordinate types for learning
        coords.extend(self.coordinate_strategies['corners'][:2])
        coords.extend(self.coordinate_strategies['center'])
        coords.extend(self.coordinate_strategies['edges'][:2])
        
        return coords
    
    def _create_sequence_from_pattern(self, pattern: List[int]) -> List[Tuple[int, Optional[Tuple[int, int]]]]:
        """Create action sequence from a pattern of actions."""
        sequence = []
        
        for action in pattern:
            if action == 6:  # Coordinate action
                # Add random coordinate
                coord = (random.randint(0, 63), random.randint(0, 63))
                sequence.append((action, coord))
            else:
                sequence.append((action, None))
        
        return sequence
    
    def update_hypothesis_weights(self, 
                                hypothesis_type: HypothesisType,
                                success_rate: float,
                                learning_rate: float = 0.01):
        """Update hypothesis generation weights based on success."""
        
        # Adjust weight based on success rate
        weight_adjustment = (success_rate - 0.5) * learning_rate
        
        if hypothesis_type == HypothesisType.VISUAL_TARGETING:
            self.visual_weight = max(0.1, min(0.5, self.visual_weight + weight_adjustment))
        elif hypothesis_type == HypothesisType.MEMORY_GUIDED:
            self.memory_weight = max(0.1, min(0.5, self.memory_weight + weight_adjustment))
        elif hypothesis_type == HypothesisType.EXPLORATION:
            self.exploration_weight = max(0.1, min(0.5, self.exploration_weight + weight_adjustment))
        elif hypothesis_type == HypothesisType.ENERGY_OPTIMIZATION:
            self.energy_weight = max(0.1, min(0.5, self.energy_weight + weight_adjustment))
        elif hypothesis_type == HypothesisType.LEARNING_FOCUSED:
            self.learning_weight = max(0.1, min(0.5, self.learning_weight + weight_adjustment))
        
        # Normalize weights
        total_weight = (self.visual_weight + self.memory_weight + self.exploration_weight + 
                       self.energy_weight + self.learning_weight)
        
        if total_weight > 0:
            self.visual_weight /= total_weight
            self.memory_weight /= total_weight
            self.exploration_weight /= total_weight
            self.energy_weight /= total_weight
            self.learning_weight /= total_weight
        
        logger.debug(f"Updated hypothesis weights: visual={self.visual_weight:.3f}, "
                    f"memory={self.memory_weight:.3f}, exploration={self.exploration_weight:.3f}, "
                    f"energy={self.energy_weight:.3f}, learning={self.learning_weight:.3f}")
    
    def get_hypothesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about hypothesis generation."""
        return {
            'visual_weight': self.visual_weight,
            'memory_weight': self.memory_weight,
            'exploration_weight': self.exploration_weight,
            'energy_weight': self.energy_weight,
            'learning_weight': self.learning_weight,
            'total_weight': (self.visual_weight + self.memory_weight + self.exploration_weight + 
                           self.energy_weight + self.learning_weight)
        }
