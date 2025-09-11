#!/usr/bin/env python3
"""
Simulation Evaluator for Governor System

This module implements the simulation evaluation system that uses affective
systems (energy, learning progress, boredom) to evaluate simulated rollouts
and provide emotional valence-based recommendations.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .simulation_models import (
    SimulationResult, SimulationEvaluation, SimulationContext,
    SimulationStatus, SimulationConfig
)

logger = logging.getLogger(__name__)

class SimulationEvaluator:
    """
    Evaluates simulated rollouts using affective systems.
    Tags simulations with emotional valence based on predicted outcomes.
    This is where "emotion" guides decision-making.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Evaluation weights
        self.energy_weight = self.config.energy_evaluation_weight
        self.learning_weight = self.config.learning_evaluation_weight
        self.boredom_weight = self.config.boredom_evaluation_weight
        self.confidence_weight = self.config.confidence_evaluation_weight
        
        # Evaluation history for learning
        self.evaluation_history: List[SimulationEvaluation] = []
        self.accuracy_tracking: Dict[str, List[float]] = {}
        
        logger.info("Simulation Evaluator initialized")
    
    def evaluate_simulation(self, 
                           simulation_result: SimulationResult,
                           context: SimulationContext) -> SimulationEvaluation:
        """
        Evaluate a simulation using the energy system and learning progress drive.
        This is where "emotion" guides decision-making.
        """
        start_time = time.time()
        
        try:
            # Energy-based evaluation
            energy_outcome = self._evaluate_energy_outcome(simulation_result, context)
            
            # Learning progress evaluation  
            learning_outcome = self._evaluate_learning_outcome(simulation_result, context)
            
            # Boredom/stagnation evaluation
            boredom_outcome = self._evaluate_boredom_outcome(simulation_result, context)
            
            # Confidence evaluation
            confidence_outcome = self._evaluate_confidence_outcome(simulation_result, context)
            
            # Calculate overall affective valence
            valence = self._calculate_affective_valence(
                energy_outcome, 
                learning_outcome, 
                boredom_outcome,
                confidence_outcome
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(valence, simulation_result)
            
            # Create evaluation
            evaluation = SimulationEvaluation(
                simulation_result=simulation_result,
                valence=valence,
                energy_impact=energy_outcome,
                learning_impact=learning_outcome,
                boredom_impact=boredom_outcome,
                recommendation=recommendation,
                confidence=confidence_outcome,
                reasoning=self._generate_reasoning(
                    energy_outcome, learning_outcome, boredom_outcome, confidence_outcome
                )
            )
            
            # Store evaluation for learning
            self.evaluation_history.append(evaluation)
            if len(self.evaluation_history) > 1000:  # Keep only recent evaluations
                self.evaluation_history = self.evaluation_history[-1000:]
            
            evaluation_time = time.time() - start_time
            logger.debug(f"Simulation evaluation completed in {evaluation_time:.3f}s, valence: {valence:.3f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Simulation evaluation failed: {e}")
            # Return neutral evaluation on error
            return SimulationEvaluation(
                simulation_result=simulation_result,
                valence=0.0,
                energy_impact=0.0,
                learning_impact=0.0,
                boredom_impact=0.0,
                recommendation="error",
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}"
            )
    
    def _evaluate_energy_outcome(self, simulation: SimulationResult, context: SimulationContext) -> float:
        """Evaluate energy impact of simulation."""
        if not simulation.simulation_history:
            return 0.0
        
        # Calculate total energy change
        total_energy_change = sum(step.energy_change for step in simulation.simulation_history)
        final_energy = context.energy_level + total_energy_change
        
        # Energy gain = positive, Energy loss = negative
        # But consider current energy level (low energy = more critical)
        energy_criticality = max(0, (50 - context.energy_level) / 50)  # 0-1 scale
        
        # Base energy score
        energy_score = total_energy_change / 100.0  # Normalize by 100 energy units
        
        # Apply criticality multiplier
        energy_score *= (1 + energy_criticality)
        
        # Penalize if energy would go below critical threshold
        if final_energy < 20:
            energy_score -= 0.5  # Strong penalty for critical energy
        
        # Reward if energy would go above optimal threshold
        if final_energy > 80:
            energy_score += 0.2  # Bonus for high energy
        
        return np.clip(energy_score, -1.0, 1.0)
    
    def _evaluate_learning_outcome(self, simulation: SimulationResult, context: SimulationContext) -> float:
        """Evaluate learning progress impact of simulation."""
        if not simulation.simulation_history:
            return 0.0
        
        # Calculate total learning progress
        total_learning = sum(step.learning_progress for step in simulation.simulation_history)
        
        # Normalize by simulation depth
        normalized_learning = total_learning / len(simulation.simulation_history)
        
        # Apply learning drive multiplier
        learning_score = normalized_learning * context.learning_drive
        
        # Bonus for consistent learning across steps
        if len(simulation.simulation_history) > 1:
            learning_consistency = self._calculate_learning_consistency(simulation.simulation_history)
            learning_score *= (1 + learning_consistency * 0.2)
        
        return np.clip(learning_score, -1.0, 1.0)
    
    def _evaluate_boredom_outcome(self, simulation: SimulationResult, context: SimulationContext) -> float:
        """Evaluate boredom/stagnation impact of simulation."""
        if not simulation.simulation_history:
            return 0.0
        
        # Check for repetitive actions (boredom indicator)
        actions = [step.action for step in simulation.simulation_history]
        action_diversity = len(set(actions)) / len(actions) if actions else 0.0
        
        # Check for lack of progress (stagnation indicator)
        progress_steps = sum(1 for step in simulation.simulation_history if step.learning_progress > 0.01)
        progress_ratio = progress_steps / len(simulation.simulation_history) if simulation.simulation_history else 0.0
        
        # Calculate boredom score (higher = more boring)
        boredom_score = 0.0
        
        # Penalize low action diversity
        if action_diversity < 0.5:
            boredom_score += (0.5 - action_diversity) * 0.5
        
        # Penalize low progress
        if progress_ratio < 0.3:
            boredom_score += (0.3 - progress_ratio) * 0.5
        
        # Apply current boredom level
        boredom_score *= (1 + context.boredom_level)
        
        # Convert to positive score (lower boredom = higher score)
        boredom_outcome = -boredom_score
        
        return np.clip(boredom_outcome, -1.0, 1.0)
    
    def _evaluate_confidence_outcome(self, simulation: SimulationResult, context: SimulationContext) -> float:
        """Evaluate confidence in simulation predictions."""
        if not simulation.simulation_history:
            return 0.0
        
        # Calculate average confidence across simulation steps
        confidences = [step.confidence for step in simulation.simulation_history]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Penalize if simulation was terminated early
        if simulation.terminated_early:
            avg_confidence *= 0.7
        
        # Penalize if simulation depth is very short
        if len(simulation.simulation_history) < 3:
            avg_confidence *= 0.8
        
        return np.clip(avg_confidence, 0.0, 1.0)
    
    def _calculate_learning_consistency(self, simulation_history: List) -> float:
        """Calculate how consistent learning progress is across simulation steps."""
        if len(simulation_history) < 2:
            return 0.0
        
        learning_values = [step.learning_progress for step in simulation_history]
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_learning = np.mean(learning_values)
        if mean_learning == 0:
            return 0.0
        
        std_learning = np.std(learning_values)
        cv = std_learning / mean_learning
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0, 1.0 - cv)
        
        return consistency
    
    def _calculate_affective_valence(self, 
                                   energy_outcome: float,
                                   learning_outcome: float, 
                                   boredom_outcome: float,
                                   confidence_outcome: float) -> float:
        """Calculate overall affective valence from individual outcomes."""
        
        # Weighted combination of outcomes
        valence = (
            self.energy_weight * energy_outcome +
            self.learning_weight * learning_outcome +
            self.boredom_weight * boredom_outcome +
            self.confidence_weight * confidence_outcome
        )
        
        # Apply confidence as a multiplier (low confidence reduces valence)
        valence *= (0.5 + 0.5 * confidence_outcome)
        
        return np.clip(valence, -1.0, 1.0)
    
    def _generate_recommendation(self, valence: float, simulation: SimulationResult) -> str:
        """Generate a recommendation based on valence and simulation details."""
        
        if valence > 0.5:
            return "proceed"  # Strong positive valence
        elif valence > 0.1:
            return "consider"  # Weak positive valence
        elif valence > -0.1:
            return "neutral"  # Neutral valence
        elif valence > -0.5:
            return "avoid"  # Weak negative valence
        else:
            return "reject"  # Strong negative valence
    
    def _generate_reasoning(self, 
                          energy_outcome: float,
                          learning_outcome: float,
                          boredom_outcome: float,
                          confidence_outcome: float) -> str:
        """Generate human-readable reasoning for the evaluation."""
        
        reasoning_parts = []
        
        if energy_outcome > 0.2:
            reasoning_parts.append(f"Energy gain predicted (+{energy_outcome:.2f})")
        elif energy_outcome < -0.2:
            reasoning_parts.append(f"Energy loss predicted ({energy_outcome:.2f})")
        
        if learning_outcome > 0.2:
            reasoning_parts.append(f"Learning potential high (+{learning_outcome:.2f})")
        elif learning_outcome < -0.2:
            reasoning_parts.append(f"Learning potential low ({learning_outcome:.2f})")
        
        if boredom_outcome > 0.2:
            reasoning_parts.append(f"Low boredom risk (+{boredom_outcome:.2f})")
        elif boredom_outcome < -0.2:
            reasoning_parts.append(f"High boredom risk ({boredom_outcome:.2f})")
        
        if confidence_outcome > 0.7:
            reasoning_parts.append(f"High confidence ({confidence_outcome:.2f})")
        elif confidence_outcome < 0.3:
            reasoning_parts.append(f"Low confidence ({confidence_outcome:.2f})")
        
        if not reasoning_parts:
            return "Neutral evaluation across all factors"
        
        return "; ".join(reasoning_parts)
    
    def update_evaluation_accuracy(self, 
                                 evaluation: SimulationEvaluation,
                                 real_world_outcome: Dict[str, Any]):
        """Update evaluation accuracy based on real-world results."""
        
        # Calculate accuracy metrics
        predicted_valence = evaluation.valence
        actual_success = real_world_outcome.get('success', False)
        actual_energy_change = real_world_outcome.get('energy_change', 0.0)
        actual_learning_gain = real_world_outcome.get('learning_gain', 0.0)
        
        # Calculate actual valence
        actual_valence = 0.0
        if actual_success:
            actual_valence += 0.5
        if actual_energy_change > 0:
            actual_valence += 0.3
        if actual_learning_gain > 0:
            actual_valence += 0.2
        
        # Track accuracy
        valence_accuracy = 1.0 - abs(predicted_valence - actual_valence)
        
        # Store accuracy metrics
        evaluation_key = f"{evaluation.simulation_result.hypothesis.hypothesis_type.value}"
        if evaluation_key not in self.accuracy_tracking:
            self.accuracy_tracking[evaluation_key] = []
        
        self.accuracy_tracking[evaluation_key].append(valence_accuracy)
        
        # Keep only recent accuracy data
        if len(self.accuracy_tracking[evaluation_key]) > 100:
            self.accuracy_tracking[evaluation_key] = self.accuracy_tracking[evaluation_key][-100:]
        
        logger.debug(f"Updated evaluation accuracy: {valence_accuracy:.3f}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluation performance."""
        
        if not self.evaluation_history:
            return {
                'total_evaluations': 0,
                'average_valence': 0.0,
                'recommendation_distribution': {},
                'accuracy_by_type': {}
            }
        
        # Calculate average valence
        valences = [e.valence for e in self.evaluation_history]
        average_valence = sum(valences) / len(valences)
        
        # Calculate recommendation distribution
        recommendations = [e.recommendation for e in self.evaluation_history]
        recommendation_dist = {}
        for rec in recommendations:
            recommendation_dist[rec] = recommendation_dist.get(rec, 0) + 1
        
        # Calculate accuracy by hypothesis type
        accuracy_by_type = {}
        for hypothesis_type, accuracies in self.accuracy_tracking.items():
            if accuracies:
                accuracy_by_type[hypothesis_type] = {
                    'average_accuracy': sum(accuracies) / len(accuracies),
                    'sample_count': len(accuracies)
                }
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_valence': average_valence,
            'recommendation_distribution': recommendation_dist,
            'accuracy_by_type': accuracy_by_type
        }
    
    def adjust_evaluation_weights(self, learning_rate: float = 0.01):
        """Adjust evaluation weights based on accuracy feedback."""
        
        if not self.accuracy_tracking:
            return
        
        # Calculate overall accuracy for each hypothesis type
        type_accuracies = {}
        for hypothesis_type, accuracies in self.accuracy_tracking.items():
            if accuracies:
                type_accuracies[hypothesis_type] = sum(accuracies) / len(accuracies)
        
        # Adjust weights based on accuracy
        # This is a simple approach - more sophisticated methods could be used
        
        # If visual targeting has high accuracy, increase its weight
        if 'visual_targeting' in type_accuracies and type_accuracies['visual_targeting'] > 0.7:
            self.energy_weight += learning_rate
            self.learning_weight -= learning_rate * 0.5
        
        # If memory guided has high accuracy, increase its weight
        if 'memory_guided' in type_accuracies and type_accuracies['memory_guided'] > 0.7:
            self.learning_weight += learning_rate
            self.boredom_weight -= learning_rate * 0.5
        
        # Normalize weights
        total_weight = self.energy_weight + self.learning_weight + self.boredom_weight + self.confidence_weight
        self.energy_weight /= total_weight
        self.learning_weight /= total_weight
        self.boredom_weight /= total_weight
        self.confidence_weight /= total_weight
        
        logger.info(f"Adjusted evaluation weights: energy={self.energy_weight:.3f}, "
                   f"learning={self.learning_weight:.3f}, boredom={self.boredom_weight:.3f}, "
                   f"confidence={self.confidence_weight:.3f}")
