#!/usr/bin/env python3
"""
Enhanced Governor Hypothesis Management System

This module transforms the Governor from a passive monitor to an active director
of cognition that manages hypothesis pools and directs experimentation.

Key Functions:
- Generate hypotheses based on priors instead of random exploration
- Manage hypothesis pools with intelligent prioritization
- Direct experimentation based on curiosity and boredom
- Coordinate with all subsystems for cohesive decision-making
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
from collections import deque
import heapq

from .architect_priors import ArchitectPriorsSystem, SpatialStructure, ObjectPotential, CausalPrediction
from .simulation_models import SimulationHypothesis, SimulationContext, HypothesisType

logger = logging.getLogger(__name__)


class HypothesisPriority(Enum):
    """Priority levels for hypotheses."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    EXPLORATORY = 5


@dataclass
class HypothesisPool:
    """Container for managing multiple hypotheses."""
    hypotheses: List[SimulationHypothesis]
    priorities: List[float]
    last_updated: float
    pool_id: str


@dataclass
class ExperimentationContext:
    """Context for directing experimentation."""
    curiosity_level: float
    boredom_level: float
    recent_success_rate: float
    available_energy: float
    learning_drive: float
    strategy_switch_needed: bool


class HypothesisGenerator:
    """Generates hypotheses based on priors and context."""
    
    def __init__(self, architect_priors: ArchitectPriorsSystem):
        self.architect_priors = architect_priors
        self.hypothesis_templates = self._initialize_hypothesis_templates()
        
    def _initialize_hypothesis_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for different types of hypotheses."""
        return {
            'spatial_symmetry': {
                'base_priority': 0.8,
                'action_pattern': [6, 6, 6],  # Coordinate actions
                'description': 'Explore symmetry-based coordinate actions',
                'energy_cost': 2.0,
                'learning_potential': 0.7
            },
            'object_interaction': {
                'base_priority': 0.9,
                'action_pattern': [5, 6, 5],  # Interaction pattern
                'description': 'Interact with detected objects',
                'energy_cost': 1.5,
                'learning_potential': 0.8
            },
            'path_following': {
                'base_priority': 0.7,
                'action_pattern': [1, 2, 3, 4],  # Movement sequence
                'description': 'Follow detected paths',
                'energy_cost': 1.0,
                'learning_potential': 0.6
            },
            'enclosure_exploration': {
                'base_priority': 0.6,
                'action_pattern': [6, 6],  # Coordinate actions
                'description': 'Explore enclosed areas',
                'energy_cost': 2.0,
                'learning_potential': 0.5
            },
            'pattern_matching': {
                'base_priority': 0.8,
                'action_pattern': [6, 5, 6],  # Pattern-based actions
                'description': 'Match detected patterns',
                'energy_cost': 1.8,
                'learning_potential': 0.7
            },
            'causal_exploration': {
                'base_priority': 0.9,
                'action_pattern': [5, 1, 5, 2],  # Causal testing
                'description': 'Test causal relationships',
                'energy_cost': 1.2,
                'learning_potential': 0.9
            }
        }
    
    def generate_prior_based_hypotheses(self, 
                                      spatial_priors: SpatialStructure,
                                      object_priors: ObjectPotential,
                                      causal_priors: CausalPrediction,
                                      context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on priors instead of random exploration."""
        hypotheses = []
        
        # Generate spatial-based hypotheses
        hypotheses.extend(self._generate_spatial_hypotheses(spatial_priors, context))
        
        # Generate object-based hypotheses
        hypotheses.extend(self._generate_object_hypotheses(object_priors, context))
        
        # Generate causal-based hypotheses
        hypotheses.extend(self._generate_causal_hypotheses(causal_priors, context))
        
        # Generate pattern-based hypotheses
        hypotheses.extend(self._generate_pattern_hypotheses(spatial_priors, context))
        
        return hypotheses
    
    def _generate_spatial_hypotheses(self, spatial: SpatialStructure, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on spatial priors."""
        hypotheses = []
        
        # Symmetry-based hypotheses
        if spatial.symmetries:
            for symmetry in spatial.symmetries:
                if symmetry['type'] == 'horizontal':
                    coords = self._generate_symmetry_coordinates(symmetry, 'horizontal', context)
                    if coords:
                        action_sequence = [(6, coord) for coord in coords[:3]]
                        hypothesis = SimulationHypothesis(
                            name=f"horizontal_symmetry_{len(hypotheses)}",
                            description=f"Explore horizontal symmetry (strength: {symmetry['strength']:.2f})",
                            hypothesis_type=HypothesisType.VISUAL_TARGETING,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.8 * symmetry['strength'],
                            expected_outcome="Symmetry-based coordinate exploration",
                            energy_cost=len(action_sequence) * 2.0,
                            learning_potential=0.7,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'symmetry_type': 'horizontal'
                            }
                        )
                        hypotheses.append(hypothesis)
                
                elif symmetry['type'] == 'vertical':
                    coords = self._generate_symmetry_coordinates(symmetry, 'vertical', context)
                    if coords:
                        action_sequence = [(6, coord) for coord in coords[:3]]
                        hypothesis = SimulationHypothesis(
                            name=f"vertical_symmetry_{len(hypotheses)}",
                            description=f"Explore vertical symmetry (strength: {symmetry['strength']:.2f})",
                            hypothesis_type=HypothesisType.VISUAL_TARGETING,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.8 * symmetry['strength'],
                            expected_outcome="Symmetry-based coordinate exploration",
                            energy_cost=len(action_sequence) * 2.0,
                            learning_potential=0.7,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'symmetry_type': 'vertical'
                            }
                        )
                        hypotheses.append(hypothesis)
        
        # Path-based hypotheses
        if spatial.paths:
            for path in spatial.paths:
                if path['type'] == 'linear' and path['endpoints'] == 2:
                    coords = self._generate_path_coordinates(path, context)
                    if coords:
                        action_sequence = [(6, coord) for coord in coords[:2]]
                        hypothesis = SimulationHypothesis(
                            name=f"linear_path_{len(hypotheses)}",
                            description=f"Follow linear path (length: {path['length']})",
                            hypothesis_type=HypothesisType.EXPLORATION,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.7,
                            expected_outcome="Path following exploration",
                            energy_cost=len(action_sequence) * 2.0,
                            learning_potential=0.6,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'path_type': 'linear'
                            }
                        )
                        hypotheses.append(hypothesis)
        
        # Enclosure-based hypotheses
        if spatial.enclosures:
            for enclosure in spatial.enclosures:
                coords = self._generate_enclosure_coordinates(enclosure, context)
                if coords:
                    action_sequence = [(6, coord) for coord in coords[:2]]
                    hypothesis = SimulationHypothesis(
                        name=f"enclosure_{enclosure['type']}_{len(hypotheses)}",
                        description=f"Explore {enclosure['type']} enclosure (area: {enclosure['area']})",
                        hypothesis_type=HypothesisType.EXPLORATION,
                        action_sequence=action_sequence,
                        simulation_depth=len(action_sequence),
                        priority=0.6,
                        expected_outcome="Enclosure exploration",
                        energy_cost=len(action_sequence) * 2.0,
                        learning_potential=0.5,
                        context_requirements={
                            'available_actions': context.available_actions,
                            'enclosure_type': enclosure['type']
                        }
                    )
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_object_hypotheses(self, objects: ObjectPotential, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on object priors."""
        hypotheses = []
        
        # Movable object hypotheses
        if objects.movable_objects:
            for i, obj in enumerate(objects.movable_objects[:3]):  # Limit to top 3
                coords = [obj['center']]
                action_sequence = [(6, coord) for coord in coords]
                hypothesis = SimulationHypothesis(
                    name=f"movable_object_{i}",
                    description=f"Interact with movable object (score: {obj.get('movability_score', 0):.2f})",
                    hypothesis_type=HypothesisType.VISUAL_TARGETING,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=0.9 * obj.get('movability_score', 0.5),
                    expected_outcome="Object interaction",
                    energy_cost=len(action_sequence) * 2.0,
                    learning_potential=0.8,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'object_type': 'movable'
                    }
                )
                hypotheses.append(hypothesis)
        
        # Combinable object hypotheses
        if objects.combinable_objects:
            for i, combo in enumerate(objects.combinable_objects[:2]):  # Limit to top 2
                coords = [combo['object1']['center'], combo['object2']['center']]
                action_sequence = [(6, coord) for coord in coords]
                hypothesis = SimulationHypothesis(
                    name=f"combinable_objects_{i}",
                    description=f"Combine objects (score: {combo.get('combination_score', 0):.2f})",
                    hypothesis_type=HypothesisType.VISUAL_TARGETING,
                    action_sequence=action_sequence,
                    simulation_depth=len(action_sequence),
                    priority=0.8 * combo.get('combination_score', 0.5),
                    expected_outcome="Object combination",
                    energy_cost=len(action_sequence) * 2.0,
                    learning_potential=0.9,
                    context_requirements={
                        'available_actions': context.available_actions,
                        'interaction_type': 'combination'
                    }
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_causal_hypotheses(self, causal: CausalPrediction, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on causal priors."""
        hypotheses = []
        
        if causal.predicted_effects and causal.confidence > 0.5:
            # Generate hypotheses to test predicted effects
            for effect in causal.predicted_effects:
                if effect['type'] == 'position_change':
                    # Test movement actions
                    movement_actions = [1, 2, 3, 4]  # up, down, left, right
                    available_movements = [a for a in movement_actions if a in context.available_actions]
                    
                    if available_movements:
                        action_sequence = [(action, None) for action in available_movements[:2]]
                        hypothesis = SimulationHypothesis(
                            name=f"causal_movement_{len(hypotheses)}",
                            description=f"Test movement causality (confidence: {causal.confidence:.2f})",
                            hypothesis_type=HypothesisType.LEARNING_FOCUSED,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.9 * causal.confidence,
                            expected_outcome="Causal effect validation",
                            energy_cost=len(action_sequence) * 1.0,
                            learning_potential=0.9,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'causal_type': 'movement'
                            }
                        )
                        hypotheses.append(hypothesis)
                
                elif effect['type'] == 'object_manipulation':
                    # Test interaction actions
                    if 5 in context.available_actions:
                        action_sequence = [(5, None), (6, (32, 32)), (5, None)]
                        hypothesis = SimulationHypothesis(
                            name=f"causal_interaction_{len(hypotheses)}",
                            description=f"Test interaction causality (confidence: {causal.confidence:.2f})",
                            hypothesis_type=HypothesisType.LEARNING_FOCUSED,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.9 * causal.confidence,
                            expected_outcome="Causal effect validation",
                            energy_cost=len(action_sequence) * 1.5,
                            learning_potential=0.9,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'causal_type': 'interaction'
                            }
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_pattern_hypotheses(self, spatial: SpatialStructure, context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate hypotheses based on detected patterns."""
        hypotheses = []
        
        if spatial.patterns:
            for pattern in spatial.patterns:
                if pattern['type'] == 'grid' and pattern['strength'] > 0.6:
                    # Generate grid-based coordinate hypotheses
                    coords = self._generate_grid_coordinates(pattern, context)
                    if coords:
                        action_sequence = [(6, coord) for coord in coords[:3]]
                        hypothesis = SimulationHypothesis(
                            name=f"grid_pattern_{len(hypotheses)}",
                            description=f"Explore grid pattern (strength: {pattern['strength']:.2f})",
                            hypothesis_type=HypothesisType.VISUAL_TARGETING,
                            action_sequence=action_sequence,
                            simulation_depth=len(action_sequence),
                            priority=0.8 * pattern['strength'],
                            expected_outcome="Grid pattern exploration",
                            energy_cost=len(action_sequence) * 2.0,
                            learning_potential=0.6,
                            context_requirements={
                                'available_actions': context.available_actions,
                                'pattern_type': 'grid'
                            }
                        )
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_symmetry_coordinates(self, symmetry: Dict[str, Any], symmetry_type: str, context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates based on symmetry detection."""
        coords = []
        
        if symmetry_type == 'horizontal':
            axis = symmetry.get('axis', 32)
            # Generate coordinates around the horizontal axis
            for y_offset in [-10, 0, 10]:
                y = max(0, min(63, axis + y_offset))
                coords.append((32, y))  # Center horizontally
        
        elif symmetry_type == 'vertical':
            axis = symmetry.get('axis', 32)
            # Generate coordinates around the vertical axis
            for x_offset in [-10, 0, 10]:
                x = max(0, min(63, axis + x_offset))
                coords.append((x, 32))  # Center vertically
        
        return coords
    
    def _generate_path_coordinates(self, path: Dict[str, Any], context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates based on path detection."""
        coords = []
        
        # Use path endpoints and center
        if 'endpoints' in path and path['endpoints'] >= 2:
            # Generate coordinates along the path
            center = path.get('centroid', (32, 32))
            coords.append(center)
            
            # Add some variation around the center
            for offset in [(10, 0), (-10, 0), (0, 10), (0, -10)]:
                x = max(0, min(63, center[0] + offset[0]))
                y = max(0, min(63, center[1] + offset[1]))
                coords.append((x, y))
        
        return coords
    
    def _generate_enclosure_coordinates(self, enclosure: Dict[str, Any], context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates based on enclosure detection."""
        coords = []
        
        center = enclosure.get('center', (32, 32))
        coords.append(center)
        
        # Add coordinates around the enclosure
        bbox = enclosure.get('bounding_box', (0, 0, 64, 64))
        x, y, w, h = bbox
        
        # Add corner coordinates
        coords.append((x, y))  # Top-left
        coords.append((x + w, y))  # Top-right
        coords.append((x, y + h))  # Bottom-left
        coords.append((x + w, y + h))  # Bottom-right
        
        return coords
    
    def _generate_grid_coordinates(self, pattern: Dict[str, Any], context: SimulationContext) -> List[Tuple[int, int]]:
        """Generate coordinates based on grid pattern detection."""
        coords = []
        
        spacing = pattern.get('spacing', {'horizontal': 16, 'vertical': 16})
        h_spacing = spacing.get('horizontal', 16)
        v_spacing = spacing.get('vertical', 16)
        
        # Generate coordinates based on grid spacing
        for i in range(3):
            for j in range(3):
                x = min(63, i * h_spacing + 8)
                y = min(63, j * v_spacing + 8)
                coords.append((x, y))
        
        return coords


class HypothesisPrioritizer:
    """Intelligently prioritizes hypotheses based on context and priors."""
    
    def __init__(self):
        self.priority_weights = {
            'curiosity_level': 0.3,
            'learning_potential': 0.25,
            'energy_efficiency': 0.2,
            'prior_confidence': 0.15,
            'context_match': 0.1
        }
    
    def prioritize_hypotheses(self, 
                            hypotheses: List[SimulationHypothesis],
                            context: SimulationContext,
                            experimentation_context: ExperimentationContext) -> List[SimulationHypothesis]:
        """Prioritize hypotheses based on context and experimentation needs."""
        
        # Calculate priority scores for each hypothesis
        scored_hypotheses = []
        for hypothesis in hypotheses:
            priority_score = self._calculate_priority_score(hypothesis, context, experimentation_context)
            scored_hypotheses.append((priority_score, hypothesis))
        
        # Sort by priority (higher is better)
        scored_hypotheses.sort(key=lambda x: x[0], reverse=True)
        
        # Return sorted hypotheses
        return [hypothesis for _, hypothesis in scored_hypotheses]
    
    def _calculate_priority_score(self, 
                                hypothesis: SimulationHypothesis,
                                context: SimulationContext,
                                experimentation_context: ExperimentationContext) -> float:
        """Calculate priority score for a hypothesis."""
        
        # Base priority from hypothesis
        base_priority = hypothesis.priority
        
        # Curiosity boost
        curiosity_boost = experimentation_context.curiosity_level * self.priority_weights['curiosity_level']
        
        # Learning potential boost
        learning_boost = hypothesis.learning_potential * self.priority_weights['learning_potential']
        
        # Energy efficiency consideration
        energy_penalty = (hypothesis.energy_cost / 10.0) * self.priority_weights['energy_efficiency']
        if experimentation_context.available_energy < 30:
            energy_penalty *= 2.0  # Double penalty when energy is low
        
        # Prior confidence boost
        prior_confidence = self._estimate_prior_confidence(hypothesis, context)
        prior_boost = prior_confidence * self.priority_weights['prior_confidence']
        
        # Context match boost
        context_match = self._calculate_context_match(hypothesis, context)
        context_boost = context_match * self.priority_weights['context_match']
        
        # Calculate total priority
        total_priority = (base_priority + 
                         curiosity_boost + 
                         learning_boost + 
                         prior_boost + 
                         context_boost - 
                         energy_penalty)
        
        return max(0.0, min(1.0, total_priority))  # Clamp to [0, 1]
    
    def _estimate_prior_confidence(self, hypothesis: SimulationHypothesis, context: SimulationContext) -> float:
        """Estimate confidence based on priors."""
        # This is a simplified version - in practice, this would use the actual prior analysis
        if hypothesis.hypothesis_type == HypothesisType.VISUAL_TARGETING:
            return 0.8  # High confidence for visual-based hypotheses
        elif hypothesis.hypothesis_type == HypothesisType.LEARNING_FOCUSED:
            return 0.9  # Very high confidence for learning-focused hypotheses
        elif hypothesis.hypothesis_type == HypothesisType.EXPLORATION:
            return 0.6  # Medium confidence for exploration hypotheses
        else:
            return 0.5  # Default medium confidence
    
    def _calculate_context_match(self, hypothesis: SimulationHypothesis, context: SimulationContext) -> float:
        """Calculate how well the hypothesis matches the current context."""
        match_score = 0.0
        
        # Check if hypothesis actions are available
        required_actions = set()
        for action_item in hypothesis.action_sequence:
            if isinstance(action_item, tuple):
                action = action_item[0]
            else:
                action = action_item
            required_actions.add(action)
        
        available_actions = set(context.available_actions)
        action_match = len(required_actions.intersection(available_actions)) / len(required_actions)
        match_score += action_match * 0.5
        
        # Check energy level compatibility
        if context.energy_level >= hypothesis.energy_cost:
            match_score += 0.3
        else:
            match_score += 0.1  # Partial match if energy is low
        
        # Check learning drive compatibility
        if context.learning_drive > 0.7 and hypothesis.learning_potential > 0.7:
            match_score += 0.2
        
        return min(1.0, match_score)


class DirectedExperimentationController:
    """Controls directed experimentation based on curiosity and boredom."""
    
    def __init__(self):
        self.curiosity_tracker = CuriosityTracker()
        self.boredom_detector = BoredomDetector()
        self.strategy_switcher = StrategySwitcher()
        self.experiment_history = deque(maxlen=100)
    
    def select_next_experiment(self, 
                             hypothesis_pool: List[SimulationHypothesis],
                             context: SimulationContext,
                             experimentation_context: ExperimentationContext) -> Optional[SimulationHypothesis]:
        """Select the next experiment based on curiosity and boredom levels."""
        
        if not hypothesis_pool:
            return None
        
        # High curiosity: Test most promising hypothesis
        if experimentation_context.curiosity_level > 0.8:
            return self._select_most_promising(hypothesis_pool, context)
        
        # Low curiosity (boredom): Switch strategy
        if experimentation_context.boredom_level > 0.7 or experimentation_context.strategy_switch_needed:
            return self._switch_strategy(hypothesis_pool, context)
        
        # Normal: Balanced selection
        return self._balanced_selection(hypothesis_pool, context, experimentation_context)
    
    def _select_most_promising(self, hypothesis_pool: List[SimulationHypothesis], context: SimulationContext) -> SimulationHypothesis:
        """Select the most promising hypothesis for high curiosity."""
        # Sort by priority and learning potential
        scored_hypotheses = []
        for hypothesis in hypothesis_pool:
            score = hypothesis.priority * hypothesis.learning_potential
            scored_hypotheses.append((score, hypothesis))
        
        scored_hypotheses.sort(key=lambda x: x[0], reverse=True)
        return scored_hypotheses[0][1]
    
    def _switch_strategy(self, hypothesis_pool: List[SimulationHypothesis], context: SimulationContext) -> SimulationHypothesis:
        """Switch to a different strategy when bored."""
        # Avoid recently used hypothesis types
        recent_types = [h.hypothesis_type for h in self.experiment_history[-10:]]
        
        # Find hypothesis with different type
        for hypothesis in hypothesis_pool:
            if hypothesis.hypothesis_type not in recent_types:
                return hypothesis
        
        # If all types have been used recently, select by lowest recent usage
        type_counts = {}
        for h in self.experiment_history[-20:]:
            type_counts[h.hypothesis_type] = type_counts.get(h.hypothesis_type, 0) + 1
        
        # Select hypothesis with lowest count
        best_hypothesis = None
        best_count = float('inf')
        
        for hypothesis in hypothesis_pool:
            count = type_counts.get(hypothesis.hypothesis_type, 0)
            if count < best_count:
                best_count = count
                best_hypothesis = hypothesis
        
        return best_hypothesis or hypothesis_pool[0]
    
    def _balanced_selection(self, 
                          hypothesis_pool: List[SimulationHypothesis],
                          context: SimulationContext,
                          experimentation_context: ExperimentationContext) -> SimulationHypothesis:
        """Balanced selection considering multiple factors."""
        
        # Weighted selection based on priority and context
        weights = []
        for hypothesis in hypothesis_pool:
            weight = (hypothesis.priority * 0.4 + 
                     hypothesis.learning_potential * 0.3 +
                     (1.0 - hypothesis.energy_cost / 10.0) * 0.2 +
                     np.random.random() * 0.1)  # Small random factor
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Select based on weights
        selected_index = np.random.choice(len(hypothesis_pool), p=weights)
        return hypothesis_pool[selected_index]
    
    def record_experiment_result(self, hypothesis: SimulationHypothesis, success: bool, outcome: Dict[str, Any]):
        """Record the result of an experiment."""
        self.experiment_history.append(hypothesis)
        self.curiosity_tracker.update_curiosity(success, outcome)
        self.boredom_detector.update_boredom(success, outcome)


class CuriosityTracker:
    """Tracks curiosity levels based on prediction violations and anomalies."""
    
    def __init__(self):
        self.curiosity_level = 0.5
        self.prediction_violations = deque(maxlen=50)
        self.anomalies = deque(maxlen=50)
        self.learning_rate = 0.1
    
    def update_curiosity(self, success: bool, outcome: Dict[str, Any]):
        """Update curiosity level based on experiment outcome."""
        # High curiosity when predictions are violated
        if outcome.get('prediction_violation', False):
            self.curiosity_level = min(1.0, self.curiosity_level + 0.2)
        elif outcome.get('anomaly_detected', False):
            self.curiosity_level = min(1.0, self.curiosity_level + 0.15)
        elif success:
            # Successful learning increases curiosity slightly
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
        else:
            # Failed experiments decrease curiosity
            self.curiosity_level = max(0.0, self.curiosity_level - 0.05)
        
        # Decay over time
        self.curiosity_level *= 0.99


class BoredomDetector:
    """Detects boredom based on predictability and lack of learning."""
    
    def __init__(self):
        self.boredom_level = 0.0
        self.recent_outcomes = deque(maxlen=20)
        self.predictability_threshold = 0.8
        self.learning_threshold = 0.1
    
    def update_boredom(self, success: bool, outcome: Dict[str, Any]):
        """Update boredom level based on recent outcomes."""
        self.recent_outcomes.append({
            'success': success,
            'outcome': outcome,
            'timestamp': time.time()
        })
        
        # Calculate predictability
        if len(self.recent_outcomes) >= 10:
            recent_successes = [o['success'] for o in self.recent_outcomes[-10:]]
            success_rate = sum(recent_successes) / len(recent_successes)
            
            # High predictability increases boredom
            if success_rate > self.predictability_threshold or success_rate < (1 - self.predictability_threshold):
                self.boredom_level = min(1.0, self.boredom_level + 0.1)
            else:
                self.boredom_level = max(0.0, self.boredom_level - 0.05)
        
        # Check for learning stagnation
        if len(self.recent_outcomes) >= 20:
            recent_learning = [o['outcome'].get('learning_progress', 0) for o in self.recent_outcomes[-20:]]
            avg_learning = sum(recent_learning) / len(recent_learning)
            
            if avg_learning < self.learning_threshold:
                self.boredom_level = min(1.0, self.boredom_level + 0.15)
    
    def is_bored(self) -> bool:
        """Check if the system is bored."""
        return self.boredom_level > 0.7


class StrategySwitcher:
    """Manages strategy switching when bored."""
    
    def __init__(self):
        self.strategy_history = deque(maxlen=50)
        self.strategy_types = [HypothesisType.VISUAL_TARGETING, 
                              HypothesisType.EXPLORATION, 
                              HypothesisType.LEARNING_FOCUSED,
                              HypothesisType.MEMORY_GUIDED]
    
    def switch_strategy(self) -> bool:
        """Switch to a different strategy."""
        # This would implement more sophisticated strategy switching
        # For now, just return True to indicate a switch is needed
        return True


class GovernorHypothesisManager:
    """
    Main Governor Hypothesis Management System.
    
    This transforms the Governor from a passive monitor to an active director
    of cognition that manages hypothesis pools and directs experimentation.
    """
    
    def __init__(self, architect_priors: ArchitectPriorsSystem):
        self.architect_priors = architect_priors
        self.hypothesis_generator = HypothesisGenerator(architect_priors)
        self.hypothesis_prioritizer = HypothesisPrioritizer()
        self.experimentation_controller = DirectedExperimentationController()
        
        # Hypothesis pool management
        self.active_hypothesis_pool = []
        self.hypothesis_pools = {}
        self.max_pool_size = 20
        
        logger.info("Governor Hypothesis Manager initialized")
    
    def generate_and_prioritize_hypotheses(self, 
                                         frame: np.ndarray,
                                         context: SimulationContext) -> List[SimulationHypothesis]:
        """Generate and prioritize hypotheses based on priors and context."""
        
        # Get prior analysis
        spatial_priors = self.architect_priors.analyze_spatial_structure(frame)
        object_priors = self.architect_priors.analyze_object_potential(frame)
        causal_priors = self.architect_priors.predict_action_effects(6, context.current_state)  # Default action
        
        # Generate hypotheses based on priors
        hypotheses = self.hypothesis_generator.generate_prior_based_hypotheses(
            spatial_priors, object_priors, causal_priors, context
        )
        
        # Create experimentation context
        experimentation_context = ExperimentationContext(
            curiosity_level=self.experimentation_controller.curiosity_tracker.curiosity_level,
            boredom_level=self.experimentation_controller.boredom_detector.boredom_level,
            recent_success_rate=self._calculate_recent_success_rate(),
            available_energy=context.energy_level,
            learning_drive=context.learning_drive,
            strategy_switch_needed=self.experimentation_controller.boredom_detector.is_bored()
        )
        
        # Prioritize hypotheses
        prioritized_hypotheses = self.hypothesis_prioritizer.prioritize_hypotheses(
            hypotheses, context, experimentation_context
        )
        
        # Update active hypothesis pool
        self.active_hypothesis_pool = prioritized_hypotheses[:self.max_pool_size]
        
        return self.active_hypothesis_pool
    
    def select_next_experiment(self, context: SimulationContext) -> Optional[SimulationHypothesis]:
        """Select the next experiment to run."""
        if not self.active_hypothesis_pool:
            return None
        
        # Create experimentation context
        experimentation_context = ExperimentationContext(
            curiosity_level=self.experimentation_controller.curiosity_tracker.curiosity_level,
            boredom_level=self.experimentation_controller.boredom_detector.boredom_level,
            recent_success_rate=self._calculate_recent_success_rate(),
            available_energy=context.energy_level,
            learning_drive=context.learning_drive,
            strategy_switch_needed=self.experimentation_controller.boredom_detector.is_bored()
        )
        
        # Select next experiment
        selected_hypothesis = self.experimentation_controller.select_next_experiment(
            self.active_hypothesis_pool, context, experimentation_context
        )
        
        return selected_hypothesis
    
    def record_experiment_result(self, hypothesis: SimulationHypothesis, success: bool, outcome: Dict[str, Any]):
        """Record the result of an experiment."""
        self.experimentation_controller.record_experiment_result(hypothesis, success, outcome)
    
    def get_hypothesis_pool_status(self) -> Dict[str, Any]:
        """Get status of the hypothesis pool."""
        return {
            'active_hypotheses': len(self.active_hypothesis_pool),
            'curiosity_level': self.experimentation_controller.curiosity_tracker.curiosity_level,
            'boredom_level': self.experimentation_controller.boredom_detector.boredom_level,
            'recent_experiments': len(self.experimentation_controller.experiment_history),
            'strategy_switch_needed': self.experimentation_controller.boredom_detector.is_bored()
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate recent success rate."""
        if not self.experimentation_controller.experiment_history:
            return 0.5  # Default moderate success rate
        
        # This is a simplified version - in practice, this would track actual success/failure
        recent_experiments = list(self.experimentation_controller.experiment_history)[-10:]
        return 0.7  # Placeholder - would be calculated from actual results
