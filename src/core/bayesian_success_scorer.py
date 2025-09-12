#!/usr/bin/env python3
"""
Bayesian Success Scorer for Simulation Intelligence

This module implements Bayesian inference for scoring the probability of success
of different action paths in the simulation system. It learns from historical
outcomes and updates its beliefs about what works.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class SuccessPattern:
    """A pattern that has been observed to lead to success."""
    state_features: Dict[str, Any]
    action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0
    last_updated: float = field(default_factory=time.time)
    context_patterns: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BayesianPrior:
    """Bayesian prior for success probability."""
    alpha: float = 1.0  # Success count
    beta: float = 1.0   # Failure count
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)

class BayesianSuccessScorer:
    """
    Bayesian system for scoring the probability of success of action paths.
    
    This class implements the core learning mechanism that allows the system
    to get better at predicting success over time through Bayesian updating.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 confidence_threshold: float = 0.7,
                 pattern_similarity_threshold: float = 0.6):
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        self.pattern_similarity_threshold = pattern_similarity_threshold
        
        # Bayesian priors for different action types
        self.action_priors = {
            1: BayesianPrior(),  # Movement actions
            2: BayesianPrior(),
            3: BayesianPrior(),
            4: BayesianPrior(),
            5: BayesianPrior(),  # Interaction actions
            6: BayesianPrior(),  # Coordinate actions
            7: BayesianPrior()   # Undo actions
        }
        
        # State-based priors
        self.state_priors = defaultdict(lambda: BayesianPrior())
        
        # Pattern database
        self.success_patterns: List[SuccessPattern] = []
        self.failure_patterns: List[SuccessPattern] = []
        
        # Context-based priors
        self.context_priors = defaultdict(lambda: BayesianPrior())
        
        # Learning statistics
        self.learning_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'confidence_improvement': 0.0,
            'pattern_updates': 0
        }
        
        logger.info("BayesianSuccessScorer initialized")
    
    def score_path_success_probability(self, 
                                     path: 'SearchPath',
                                     context: Optional[Dict[str, Any]] = None) -> float:
        """
        Score the probability of success for a given path using Bayesian inference.
        
        Args:
            path: SearchPath object representing the action sequence
            context: Additional context for scoring
            
        Returns:
            Probability of success (0.0 to 1.0)
        """
        
        if not path.nodes:
            return 0.0
        
        # Calculate individual action success probabilities
        action_probs = []
        for node in path.nodes:
            if node.action is not None:
                action_prob = self._calculate_action_success_probability(
                    node.action, node.state, node.coordinates, context
                )
                action_probs.append(action_prob)
        
        if not action_probs:
            return 0.0
        
        # Combine probabilities using Bayesian inference
        # For independent actions, we can multiply probabilities
        # But we also consider sequence effects
        sequence_prob = self._calculate_sequence_probability(action_probs, path)
        
        # Apply context-based adjustments
        context_adjustment = self._calculate_context_adjustment(path, context)
        
        # Final probability with context adjustment
        final_prob = sequence_prob * context_adjustment
        
        # Update learning statistics
        self.learning_stats['total_predictions'] += 1
        
        return max(0.0, min(1.0, final_prob))
    
    def _calculate_action_success_probability(self, 
                                            action: int,
                                            state: Dict[str, Any],
                                            coordinates: Optional[Tuple[int, int]],
                                            context: Optional[Dict[str, Any]]) -> float:
        """Calculate success probability for a single action using Bayesian inference."""
        
        # Get base prior for this action type
        action_prior = self.action_priors.get(action, BayesianPrior())
        
        # Calculate state-based adjustment
        state_adjustment = self._calculate_state_adjustment(state, action)
        
        # Calculate coordinate-based adjustment
        coord_adjustment = self._calculate_coordinate_adjustment(coordinates, action)
        
        # Calculate context-based adjustment
        context_adjustment = self._calculate_context_adjustment_for_action(action, context)
        
        # Combine all adjustments
        base_prob = self._beta_mean(action_prior.alpha, action_prior.beta)
        
        # Apply adjustments
        adjusted_prob = base_prob * state_adjustment * coord_adjustment * context_adjustment
        
        return max(0.0, min(1.0, adjusted_prob))
    
    def _calculate_state_adjustment(self, state: Dict[str, Any], action: int) -> float:
        """Calculate adjustment based on state features."""
        
        adjustment = 1.0
        
        # Energy level adjustment
        energy = state.get('energy', 100.0)
        if energy > 80:
            adjustment *= 1.2
        elif energy < 20:
            adjustment *= 0.6
        
        # Learning drive adjustment
        learning_drive = state.get('learning_drive', 0.5)
        if learning_drive > 0.7:
            adjustment *= 1.1
        elif learning_drive < 0.3:
            adjustment *= 0.9
        
        # Boredom level adjustment
        boredom = state.get('boredom_level', 0.0)
        if boredom > 0.8:
            adjustment *= 0.8
        
        # Action count adjustment (experience)
        action_count = state.get('action_count', 0)
        if action_count > 50:  # Experienced
            adjustment *= 1.1
        elif action_count < 10:  # Inexperienced
            adjustment *= 0.9
        
        return adjustment
    
    def _calculate_coordinate_adjustment(self, 
                                       coordinates: Optional[Tuple[int, int]], 
                                       action: int) -> float:
        """Calculate adjustment based on coordinates for coordinate actions."""
        
        if action != 6 or coordinates is None:
            return 1.0
        
        x, y = coordinates
        
        # Find similar coordinate patterns in success patterns
        coord_success_rate = self._get_coordinate_success_rate(coordinates)
        
        # Distance from center adjustment
        center_distance = math.sqrt((x - 32)**2 + (y - 32)**2)
        center_adjustment = max(0.5, 1.0 - (center_distance / 45.0))  # 45 is max distance from center
        
        # Combine coordinate success rate with center adjustment
        return coord_success_rate * center_adjustment
    
    def _calculate_context_adjustment(self, path: 'SearchPath', context: Optional[Dict[str, Any]]) -> float:
        """Calculate adjustment based on context."""
        
        if context is None:
            return 1.0
        
        adjustment = 1.0
        
        # Game type adjustment
        game_type = context.get('game_type', 'unknown')
        if game_type in context:
            game_prior = self.context_priors[game_type]
            game_success_rate = self._beta_mean(game_prior.alpha, game_prior.beta)
            adjustment *= game_success_rate
        
        # Recent success rate adjustment
        recent_success_rate = context.get('recent_success_rate', 0.5)
        if recent_success_rate > 0.7:
            adjustment *= 1.2
        elif recent_success_rate < 0.3:
            adjustment *= 0.8
        
        # Pattern similarity adjustment
        similar_patterns = self._find_similar_patterns(path, context)
        if similar_patterns:
            avg_success_rate = np.mean([p.success_count / max(1, p.success_count + p.failure_count) 
                                      for p in similar_patterns])
            adjustment *= avg_success_rate
        
        return adjustment
    
    def _calculate_context_adjustment_for_action(self, 
                                               action: int, 
                                               context: Optional[Dict[str, Any]]) -> float:
        """Calculate context adjustment for a single action."""
        
        if context is None:
            return 1.0
        
        adjustment = 1.0
        
        # Action-specific context adjustments
        if 'successful_actions' in context:
            if action in context['successful_actions']:
                adjustment *= 1.2
        
        if 'failed_actions' in context:
            if action in context['failed_actions']:
                adjustment *= 0.8
        
        # Coordinate context for coordinate actions
        if action == 6 and 'successful_coordinates' in context:
            coords = context['successful_coordinates']
            if coords:
                # Check if current coordinates are similar to successful ones
                coord_similarity = self._calculate_coordinate_similarity(coords)
                adjustment *= coord_similarity
        
        return adjustment
    
    def _calculate_sequence_probability(self, 
                                      action_probs: List[float], 
                                      path: 'SearchPath') -> float:
        """Calculate probability for a sequence of actions."""
        
        if not action_probs:
            return 0.0
        
        # Base probability is the product of individual action probabilities
        base_prob = np.prod(action_probs)
        
        # Apply sequence effects
        sequence_effect = self._calculate_sequence_effect(path)
        
        return base_prob * sequence_effect
    
    def _calculate_sequence_effect(self, path: 'SearchPath') -> float:
        """Calculate the effect of action sequencing on success probability."""
        
        if len(path.nodes) < 2:
            return 1.0
        
        # Look for successful patterns that match this sequence
        sequence_patterns = self._find_sequence_patterns(path)
        
        if sequence_patterns:
            # Use pattern-based sequence effect
            avg_success_rate = np.mean([p.success_count / max(1, p.success_count + p.failure_count) 
                                      for p in sequence_patterns])
            return avg_success_rate
        else:
            # Use heuristic sequence effects
            return self._calculate_heuristic_sequence_effect(path)
    
    def _calculate_heuristic_sequence_effect(self, path: 'SearchPath') -> float:
        """Calculate heuristic sequence effects when no patterns are available."""
        
        effect = 1.0
        
        # Check for action diversity (more diverse sequences might be better)
        actions = [node.action for node in path.nodes if node.action is not None]
        unique_actions = len(set(actions))
        diversity_effect = 1.0 + (unique_actions - 1) * 0.1
        effect *= diversity_effect
        
        # Check for coordinate actions (they might be more strategic)
        coord_actions = sum(1 for node in path.nodes if node.action == 6)
        if coord_actions > 0:
            effect *= 1.0 + coord_actions * 0.05
        
        # Check for movement patterns
        movement_actions = [a for a in actions if a in [1, 2, 3, 4]]
        if len(movement_actions) > 1:
            # Check for back-and-forth movement (might be inefficient)
            if self._has_back_and_forth_movement(movement_actions):
                effect *= 0.9
        
        return effect
    
    def _has_back_and_forth_movement(self, movement_actions: List[int]) -> bool:
        """Check if movement actions show back-and-forth patterns."""
        
        if len(movement_actions) < 4:
            return False
        
        # Look for patterns like [1, 2, 1, 2] or [3, 4, 3, 4]
        for i in range(len(movement_actions) - 3):
            if (movement_actions[i] == movement_actions[i+2] and 
                movement_actions[i+1] == movement_actions[i+3]):
                return True
        
        return False
    
    def _get_coordinate_success_rate(self, coordinates: Tuple[int, int]) -> float:
        """Get success rate for coordinates based on historical data."""
        
        x, y = coordinates
        
        # Find similar coordinates in success patterns
        similar_coords = []
        for pattern in self.success_patterns:
            for action, coords in pattern.action_sequence:
                if coords and self._coordinates_similar(coordinates, coords):
                    similar_coords.append(pattern)
        
        if not similar_coords:
            return 0.5  # Default probability
        
        # Calculate average success rate
        total_success = sum(p.success_count for p in similar_coords)
        total_attempts = sum(p.success_count + p.failure_count for p in similar_coords)
        
        if total_attempts == 0:
            return 0.5
        
        return total_success / total_attempts
    
    def _coordinates_similar(self, coords1: Tuple[int, int], coords2: Tuple[int, int]) -> bool:
        """Check if two coordinate pairs are similar."""
        
        x1, y1 = coords1
        x2, y2 = coords2
        
        # Consider coordinates similar if they're within 5 units
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return distance <= 5
    
    def _calculate_coordinate_similarity(self, successful_coords: List[Tuple[int, int]]) -> float:
        """Calculate similarity to successful coordinates."""
        
        if not successful_coords:
            return 1.0
        
        # This would be enhanced with actual coordinate comparison
        # For now, return a default similarity
        return 0.8
    
    def _find_similar_patterns(self, path: 'SearchPath', context: Optional[Dict[str, Any]]) -> List[SuccessPattern]:
        """Find patterns similar to the given path."""
        
        similar_patterns = []
        
        for pattern in self.success_patterns:
            similarity = self._calculate_pattern_similarity(path, pattern)
            if similarity > self.pattern_similarity_threshold:
                similar_patterns.append(pattern)
        
        return similar_patterns
    
    def _find_sequence_patterns(self, path: 'SearchPath') -> List[SuccessPattern]:
        """Find patterns that match the action sequence."""
        
        sequence_patterns = []
        path_actions = [node.action for node in path.nodes if node.action is not None]
        
        for pattern in self.success_patterns:
            pattern_actions = [action for action, _ in pattern.action_sequence]
            if self._sequences_match(path_actions, pattern_actions):
                sequence_patterns.append(pattern)
        
        return sequence_patterns
    
    def _sequences_match(self, seq1: List[int], seq2: List[int]) -> bool:
        """Check if two action sequences match."""
        
        if len(seq1) != len(seq2):
            return False
        
        return all(a1 == a2 for a1, a2 in zip(seq1, seq2))
    
    def _calculate_pattern_similarity(self, path: 'SearchPath', pattern: SuccessPattern) -> float:
        """Calculate similarity between a path and a pattern."""
        
        similarity = 0.0
        
        # Compare action sequences
        path_actions = [node.action for node in path.nodes if node.action is not None]
        pattern_actions = [action for action, _ in pattern.action_sequence]
        
        if path_actions and pattern_actions:
            action_similarity = self._calculate_action_sequence_similarity(path_actions, pattern_actions)
            similarity += action_similarity * 0.6
        
        # Compare state features
        if path.nodes:
            last_state = path.nodes[-1].state
            state_similarity = self._calculate_state_similarity(last_state, pattern.state_features)
            similarity += state_similarity * 0.4
        
        return similarity
    
    def _calculate_action_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate similarity between two action sequences."""
        
        if not seq1 or not seq2:
            return 0.0
        
        # Simple sequence similarity (would be enhanced with more sophisticated algorithms)
        matches = sum(1 for a1, a2 in zip(seq1, seq2) if a1 == a2)
        return matches / max(len(seq1), len(seq2))
    
    def _calculate_state_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Calculate similarity between two states."""
        
        similarity = 0.0
        total_weight = 0.0
        
        # Compare energy levels
        if 'energy' in state1 and 'energy' in state2:
            energy_sim = 1.0 - abs(state1['energy'] - state2['energy']) / 100.0
            similarity += energy_sim * 0.3
            total_weight += 0.3
        
        # Compare learning drive
        if 'learning_drive' in state1 and 'learning_drive' in state2:
            drive_sim = 1.0 - abs(state1['learning_drive'] - state2['learning_drive'])
            similarity += drive_sim * 0.2
            total_weight += 0.2
        
        # Compare action counts
        if 'action_count' in state1 and 'action_count' in state2:
            count_sim = 1.0 - abs(state1['action_count'] - state2['action_count']) / 100.0
            similarity += count_sim * 0.2
            total_weight += 0.2
        
        # Compare positions
        if 'position' in state1 and 'position' in state2:
            pos1 = state1['position']
            pos2 = state2['position']
            if len(pos1) == len(pos2) == 3:
                pos_sim = 1.0 - np.linalg.norm(np.array(pos1) - np.array(pos2)) / 10.0
                similarity += pos_sim * 0.3
                total_weight += 0.3
        
        return similarity / max(total_weight, 0.1)
    
    def _beta_mean(self, alpha: float, beta: float) -> float:
        """Calculate the mean of a Beta distribution."""
        
        if alpha + beta == 0:
            return 0.5
        
        return alpha / (alpha + beta)
    
    def update_with_outcome(self, 
                           path: 'SearchPath',
                           actual_success: bool,
                           context: Optional[Dict[str, Any]] = None):
        """Update the Bayesian model with the actual outcome of a path."""
        
        # Update action priors
        for node in path.nodes:
            if node.action is not None:
                self._update_action_prior(node.action, actual_success)
        
        # Update state priors
        if path.nodes:
            last_state = path.nodes[-1].state
            state_key = self._create_state_key(last_state)
            self._update_state_prior(state_key, actual_success)
        
        # Update context priors
        if context:
            self._update_context_priors(context, actual_success)
        
        # Update pattern database
        self._update_pattern_database(path, actual_success, context)
        
        # Update learning statistics
        self._update_learning_statistics(actual_success)
        
        logger.debug(f"Updated Bayesian model with outcome: success={actual_success}")
    
    def _update_action_prior(self, action: int, success: bool):
        """Update the prior for a specific action type."""
        
        if action not in self.action_priors:
            self.action_priors[action] = BayesianPrior()
        
        prior = self.action_priors[action]
        
        if success:
            prior.alpha += 1
        else:
            prior.beta += 1
        
        prior.last_updated = time.time()
        
        # Update confidence based on total observations
        total_obs = prior.alpha + prior.beta
        prior.confidence = min(1.0, total_obs / 100.0)  # Confidence grows with observations
    
    def _update_state_prior(self, state_key: str, success: bool):
        """Update the prior for a specific state."""
        
        prior = self.state_priors[state_key]
        
        if success:
            prior.alpha += 1
        else:
            prior.beta += 1
        
        prior.last_updated = time.time()
        prior.confidence = min(1.0, (prior.alpha + prior.beta) / 50.0)
    
    def _update_context_priors(self, context: Dict[str, Any], success: bool):
        """Update context-based priors."""
        
        # Update game type prior
        if 'game_type' in context:
            game_type = context['game_type']
            prior = self.context_priors[game_type]
            
            if success:
                prior.alpha += 1
            else:
                prior.beta += 1
            
            prior.last_updated = time.time()
            prior.confidence = min(1.0, (prior.alpha + prior.beta) / 30.0)
    
    def _create_state_key(self, state: Dict[str, Any]) -> str:
        """Create a key for state-based priors."""
        
        # Create a simplified state representation for hashing
        key_parts = []
        
        # Energy level (rounded to nearest 10)
        energy = state.get('energy', 100)
        energy_bucket = (energy // 10) * 10
        key_parts.append(f"energy_{energy_bucket}")
        
        # Learning drive (rounded to nearest 0.1)
        learning_drive = state.get('learning_drive', 0.5)
        drive_bucket = round(learning_drive, 1)
        key_parts.append(f"drive_{drive_bucket}")
        
        # Action count (rounded to nearest 10)
        action_count = state.get('action_count', 0)
        count_bucket = (action_count // 10) * 10
        key_parts.append(f"count_{count_bucket}")
        
        return "_".join(key_parts)
    
    def _update_pattern_database(self, 
                                path: 'SearchPath',
                                success: bool,
                                context: Optional[Dict[str, Any]]):
        """Update the pattern database with the new outcome."""
        
        # Create pattern from path
        pattern = SuccessPattern(
            state_features=path.nodes[-1].state if path.nodes else {},
            action_sequence=[(node.action, node.coordinates) for node in path.nodes if node.action is not None],
            context_patterns=context or {}
        )
        
        if success:
            pattern.success_count = 1
            self.success_patterns.append(pattern)
        else:
            pattern.failure_count = 1
            self.failure_patterns.append(pattern)
        
        # Limit pattern database size
        max_patterns = 1000
        if len(self.success_patterns) > max_patterns:
            # Remove oldest patterns
            self.success_patterns = sorted(self.success_patterns, key=lambda p: p.last_updated)[-max_patterns:]
        
        if len(self.failure_patterns) > max_patterns:
            self.failure_patterns = sorted(self.failure_patterns, key=lambda p: p.last_updated)[-max_patterns:]
        
        self.learning_stats['pattern_updates'] += 1
    
    def _update_learning_statistics(self, actual_success: bool):
        """Update learning statistics."""
        
        # This would be enhanced to track prediction accuracy
        # For now, just count successful outcomes
        if actual_success:
            self.learning_stats['correct_predictions'] += 1
        
        # Update accuracy
        total = self.learning_stats['total_predictions']
        if total > 0:
            self.learning_stats['accuracy'] = self.learning_stats['correct_predictions'] / total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the Bayesian scorer."""
        
        return {
            'learning_stats': self.learning_stats,
            'action_priors': {
                action: {
                    'alpha': prior.alpha,
                    'beta': prior.beta,
                    'mean': self._beta_mean(prior.alpha, prior.beta),
                    'confidence': prior.confidence
                }
                for action, prior in self.action_priors.items()
            },
            'pattern_database': {
                'success_patterns': len(self.success_patterns),
                'failure_patterns': len(self.failure_patterns)
            },
            'state_priors_count': len(self.state_priors),
            'context_priors_count': len(self.context_priors)
        }
