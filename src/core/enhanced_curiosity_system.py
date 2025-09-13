#!/usr/bin/env python3
"""
Enhanced Curiosity and Boredom System

This module implements an advanced curiosity and boredom detection system that
monitors prediction violations, learning progress, and exploration efficiency
to guide intelligent exploration and strategy switching.

Key Features:
- High Curiosity: When the world violates predictions, flag as highly interesting
- Low Curiosity (Boredom): When actions become predictable, trigger strategy switch
- Prediction Violation Detection: Identify when reality doesn't match expectations
- Anomaly Detection: Find unexpected patterns and behaviors
- Learning Acceleration: Allocate more resources to interesting discoveries
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class CuriosityEvent:
    """Represents a curiosity-triggering event."""
    event_type: str
    intensity: float
    timestamp: float
    context: Dict[str, Any]
    learning_potential: float


@dataclass
class BoredomEvent:
    """Represents a boredom-triggering event."""
    event_type: str
    intensity: float
    timestamp: float
    context: Dict[str, Any]
    strategy_switch_needed: bool


class PredictionViolationDetector:
    """Detects when reality violates predictions."""
    
    def __init__(self):
        self.violation_threshold = 0.3
        self.prediction_history = deque(maxlen=100)
        self.violation_history = deque(maxlen=50)
    
    def detect(self, prediction: Dict[str, Any], actual_outcome: Dict[str, Any]) -> Optional[CuriosityEvent]:
        """Detect prediction violations."""
        if not prediction or not actual_outcome:
            return None
        
        # Calculate violation intensity
        violation_intensity = self._calculate_violation_intensity(prediction, actual_outcome)
        
        if violation_intensity > self.violation_threshold:
            # Record the violation
            self.violation_history.append({
                'prediction': prediction,
                'actual': actual_outcome,
                'intensity': violation_intensity,
                'timestamp': time.time()
            })
            
            # Create curiosity event
            return CuriosityEvent(
                event_type='prediction_violation',
                intensity=violation_intensity,
                timestamp=time.time(),
                context={
                    'prediction': prediction,
                    'actual_outcome': actual_outcome,
                    'violation_type': self._classify_violation_type(prediction, actual_outcome)
                },
                learning_potential=min(1.0, violation_intensity * 1.5)
            )
        
        return None
    
    def _calculate_violation_intensity(self, prediction: Dict[str, Any], actual: Dict[str, Any]) -> float:
        """Calculate how much the prediction was violated."""
        intensity = 0.0
        
        # Check different types of violations
        if 'expected_outcome' in prediction and 'actual_outcome' in actual:
            if prediction['expected_outcome'] != actual['actual_outcome']:
                intensity += 0.4
        
        if 'confidence' in prediction:
            predicted_confidence = prediction['confidence']
            if predicted_confidence > 0.8 and actual.get('success', False) == False:
                intensity += 0.3  # High confidence prediction failed
            elif predicted_confidence < 0.3 and actual.get('success', False) == True:
                intensity += 0.3  # Low confidence prediction succeeded
        
        if 'energy_cost' in prediction and 'energy_used' in actual:
            predicted_cost = prediction['energy_cost']
            actual_cost = actual['energy_used']
            if abs(predicted_cost - actual_cost) / max(predicted_cost, 0.1) > 0.5:
                intensity += 0.2  # Energy cost prediction was wrong
        
        if 'learning_potential' in prediction and 'learning_achieved' in actual:
            predicted_learning = prediction['learning_potential']
            actual_learning = actual['learning_achieved']
            if abs(predicted_learning - actual_learning) > 0.3:
                intensity += 0.1  # Learning potential prediction was wrong
        
        return min(1.0, intensity)
    
    def _classify_violation_type(self, prediction: Dict[str, Any], actual: Dict[str, Any]) -> str:
        """Classify the type of prediction violation."""
        if 'expected_outcome' in prediction and 'actual_outcome' in actual:
            if prediction['expected_outcome'] == 'success' and actual['actual_outcome'] == 'failure':
                return 'success_to_failure'
            elif prediction['expected_outcome'] == 'failure' and actual['actual_outcome'] == 'success':
                return 'failure_to_success'
        
        if 'confidence' in prediction:
            if prediction['confidence'] > 0.8 and not actual.get('success', False):
                return 'overconfident_failure'
            elif prediction['confidence'] < 0.3 and actual.get('success', False):
                return 'underconfident_success'
        
        return 'general_violation'
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get statistics about prediction violations."""
        if not self.violation_history:
            return {'total_violations': 0, 'average_intensity': 0.0}
        
        intensities = [v['intensity'] for v in self.violation_history]
        return {
            'total_violations': len(self.violation_history),
            'average_intensity': statistics.mean(intensities),
            'max_intensity': max(intensities),
            'recent_violations': len([v for v in self.violation_history if time.time() - v['timestamp'] < 300])  # Last 5 minutes
        }


class AnomalyDetector:
    """Detects anomalies in behavior and outcomes."""
    
    def __init__(self):
        self.anomaly_threshold = 0.4
        self.behavior_history = deque(maxlen=200)
        self.outcome_history = deque(maxlen=200)
        self.anomaly_history = deque(maxlen=50)
    
    def detect(self, current_behavior: Dict[str, Any], current_outcome: Dict[str, Any]) -> Optional[CuriosityEvent]:
        """Detect anomalies in current behavior or outcome."""
        
        # Add to history
        self.behavior_history.append({
            'behavior': current_behavior,
            'outcome': current_outcome,
            'timestamp': time.time()
        })
        
        # Detect different types of anomalies
        anomalies = []
        
        # Behavioral anomaly detection
        behavior_anomaly = self._detect_behavioral_anomaly(current_behavior)
        if behavior_anomaly:
            anomalies.append(behavior_anomaly)
        
        # Outcome anomaly detection
        outcome_anomaly = self._detect_outcome_anomaly(current_outcome)
        if outcome_anomaly:
            anomalies.append(outcome_anomaly)
        
        # Pattern anomaly detection
        pattern_anomaly = self._detect_pattern_anomaly()
        if pattern_anomaly:
            anomalies.append(pattern_anomaly)
        
        # Return strongest anomaly
        if anomalies:
            strongest_anomaly = max(anomalies, key=lambda x: x['intensity'])
            
            # Record the anomaly
            self.anomaly_history.append({
                'anomaly': strongest_anomaly,
                'timestamp': time.time()
            })
            
            return CuriosityEvent(
                event_type='anomaly_detected',
                intensity=strongest_anomaly['intensity'],
                timestamp=time.time(),
                context={
                    'anomaly_type': strongest_anomaly['type'],
                    'behavior': current_behavior,
                    'outcome': current_outcome
                },
                learning_potential=min(1.0, strongest_anomaly['intensity'] * 1.2)
            )
        
        return None
    
    def _detect_behavioral_anomaly(self, current_behavior: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in behavior patterns."""
        if len(self.behavior_history) < 10:
            return None
        
        # Analyze recent behavior patterns
        recent_behaviors = [b['behavior'] for b in list(self.behavior_history)[-10:]]
        
        # Check for unusual action sequences
        action_sequences = [b.get('action_sequence', []) for b in recent_behaviors]
        if self._is_unusual_sequence(action_sequences[-1], action_sequences[:-1]):
            return {
                'type': 'unusual_action_sequence',
                'intensity': 0.6,
                'description': 'Unusual action sequence detected'
            }
        
        # Check for unusual energy usage
        energy_usage = [b.get('energy_used', 0) for b in recent_behaviors if 'energy_used' in b]
        if energy_usage and len(energy_usage) >= 5:
            current_energy = current_behavior.get('energy_used', 0)
            if current_energy > 0:
                avg_energy = statistics.mean(energy_usage)
                energy_deviation = abs(current_energy - avg_energy) / max(avg_energy, 0.1)
                if energy_deviation > 1.0:  # 100% deviation
                    return {
                        'type': 'unusual_energy_usage',
                        'intensity': min(1.0, energy_deviation / 2.0),
                        'description': f'Unusual energy usage: {current_energy} vs avg {avg_energy:.1f}'
                    }
        
        return None
    
    def _detect_outcome_anomaly(self, current_outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies in outcomes."""
        if len(self.outcome_history) < 10:
            return None
        
        # Analyze recent outcomes
        recent_outcomes = [b['outcome'] for b in list(self.behavior_history)[-10:]]
        
        # Check for unusual success patterns
        success_rates = [o.get('success', False) for o in recent_outcomes]
        if len(success_rates) >= 5:
            current_success = current_outcome.get('success', False)
            recent_success_rate = sum(success_rates[:-1]) / len(success_rates[:-1])
            
            # Anomaly if current outcome is very different from recent pattern
            if recent_success_rate > 0.8 and not current_success:
                return {
                    'type': 'unexpected_failure',
                    'intensity': 0.7,
                    'description': f'Unexpected failure after {recent_success_rate:.1%} success rate'
                }
            elif recent_success_rate < 0.2 and current_success:
                return {
                    'type': 'unexpected_success',
                    'intensity': 0.7,
                    'description': f'Unexpected success after {recent_success_rate:.1%} success rate'
                }
        
        # Check for unusual learning progress
        learning_progress = [o.get('learning_progress', 0) for o in recent_outcomes if 'learning_progress' in o]
        if learning_progress and len(learning_progress) >= 5:
            current_learning = current_outcome.get('learning_progress', 0)
            if current_learning > 0:
                avg_learning = statistics.mean(learning_progress)
                learning_deviation = abs(current_learning - avg_learning) / max(avg_learning, 0.01)
                if learning_deviation > 2.0:  # 200% deviation
                    return {
                        'type': 'unusual_learning_progress',
                        'intensity': min(1.0, learning_deviation / 3.0),
                        'description': f'Unusual learning progress: {current_learning:.3f} vs avg {avg_learning:.3f}'
                    }
        
        return None
    
    def _detect_pattern_anomaly(self) -> Optional[Dict[str, Any]]:
        """Detect anomalies in overall patterns."""
        if len(self.behavior_history) < 20:
            return None
        
        # Look for sudden changes in patterns
        recent_20 = list(self.behavior_history)[-20:]
        older_20 = list(self.behavior_history)[-40:-20] if len(self.behavior_history) >= 40 else []
        
        if not older_20:
            return None
        
        # Compare success rates
        recent_success_rate = sum(1 for b in recent_20 if b['outcome'].get('success', False)) / len(recent_20)
        older_success_rate = sum(1 for b in older_20 if b['outcome'].get('success', False)) / len(older_20)
        
        success_rate_change = abs(recent_success_rate - older_success_rate)
        if success_rate_change > 0.4:  # 40% change
            return {
                'type': 'pattern_shift',
                'intensity': min(1.0, success_rate_change),
                'description': f'Pattern shift: success rate changed from {older_success_rate:.1%} to {recent_success_rate:.1%}'
            }
        
        return None
    
    def _is_unusual_sequence(self, current_sequence: List, historical_sequences: List[List]) -> bool:
        """Check if current sequence is unusual compared to historical sequences."""
        if not current_sequence or not historical_sequences:
            return False
        
        # Simple heuristic: check if current sequence is very different from recent ones
        similarities = []
        for hist_seq in historical_sequences:
            if hist_seq:
                # Calculate simple similarity
                common_actions = set(current_sequence) & set(hist_seq)
                similarity = len(common_actions) / max(len(current_sequence), len(hist_seq))
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = statistics.mean(similarities)
            return avg_similarity < 0.3  # Less than 30% similarity
        
        return False
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies."""
        if not self.anomaly_history:
            return {'total_anomalies': 0, 'average_intensity': 0.0}
        
        intensities = [a['anomaly']['intensity'] for a in self.anomaly_history]
        anomaly_types = [a['anomaly']['type'] for a in self.anomaly_history]
        
        return {
            'total_anomalies': len(self.anomaly_history),
            'average_intensity': statistics.mean(intensities),
            'max_intensity': max(intensities),
            'recent_anomalies': len([a for a in self.anomaly_history if time.time() - a['timestamp'] < 300]),
            'anomaly_types': list(set(anomaly_types))
        }


class LearningAccelerator:
    """Accelerates learning when high curiosity events are detected."""
    
    def __init__(self):
        self.acceleration_factor = 1.0
        self.base_learning_rate = 0.01
        self.max_acceleration = 3.0
        self.acceleration_decay = 0.95
        self.acceleration_history = deque(maxlen=100)
    
    def allocate_more_resources(self, context: Dict[str, Any], curiosity_event: CuriosityEvent):
        """Allocate more resources when high curiosity is detected."""
        
        # Calculate acceleration factor based on curiosity intensity
        intensity = curiosity_event.intensity
        learning_potential = curiosity_event.learning_potential
        
        # Higher intensity and learning potential = more acceleration
        new_acceleration = 1.0 + (intensity * learning_potential * 2.0)
        new_acceleration = min(self.max_acceleration, new_acceleration)
        
        # Update acceleration factor
        self.acceleration_factor = max(self.acceleration_factor, new_acceleration)
        
        # Record acceleration event
        self.acceleration_history.append({
            'timestamp': time.time(),
            'acceleration_factor': new_acceleration,
            'curiosity_intensity': intensity,
            'learning_potential': learning_potential,
            'context': context
        })
        
        logger.info(f"Learning accelerated: factor={self.acceleration_factor:.2f}, "
                   f"intensity={intensity:.2f}, potential={learning_potential:.2f}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate with acceleration applied."""
        return self.base_learning_rate * self.acceleration_factor
    
    def update_acceleration(self):
        """Update acceleration factor over time."""
        # Decay acceleration over time
        self.acceleration_factor *= self.acceleration_decay
        self.acceleration_factor = max(1.0, self.acceleration_factor)
    
    def get_acceleration_status(self) -> Dict[str, Any]:
        """Get current acceleration status."""
        return {
            'acceleration_factor': self.acceleration_factor,
            'learning_rate': self.get_learning_rate(),
            'recent_accelerations': len([a for a in self.acceleration_history if time.time() - a['timestamp'] < 300])
        }


class PredictabilityTracker:
    """Tracks predictability of actions and outcomes."""
    
    def __init__(self):
        self.action_outcome_history = deque(maxlen=100)
        self.predictability_threshold = 0.8
        self.recent_window = 20
    
    def is_highly_predictable(self, recent_actions: List, recent_outcomes: List) -> bool:
        """Check if recent actions and outcomes are highly predictable."""
        if len(recent_actions) < self.recent_window or len(recent_outcomes) < self.recent_window:
            return False
        
        # Calculate predictability metrics
        action_predictability = self._calculate_action_predictability(recent_actions)
        outcome_predictability = self._calculate_outcome_predictability(recent_outcomes)
        
        # Overall predictability
        overall_predictability = (action_predictability + outcome_predictability) / 2.0
        
        return overall_predictability > self.predictability_threshold
    
    def _calculate_action_predictability(self, actions: List) -> float:
        """Calculate how predictable the action sequence is."""
        if len(actions) < 5:
            return 0.0
        
        # Look for repeating patterns
        pattern_lengths = [2, 3, 4, 5]
        max_pattern_score = 0.0
        
        for pattern_len in pattern_lengths:
            if len(actions) >= pattern_len * 2:
                # Check if there are repeating patterns of this length
                pattern_score = self._find_repeating_patterns(actions, pattern_len)
                max_pattern_score = max(max_pattern_score, pattern_score)
        
        return max_pattern_score
    
    def _calculate_outcome_predictability(self, outcomes: List) -> float:
        """Calculate how predictable the outcomes are."""
        if len(outcomes) < 5:
            return 0.0
        
        # Check for consistent success/failure patterns
        success_pattern = [o.get('success', False) for o in outcomes]
        
        # Calculate consistency
        if len(set(success_pattern)) == 1:  # All same outcome
            return 1.0
        
        # Calculate success rate consistency
        success_rate = sum(success_pattern) / len(success_pattern)
        if success_rate > 0.9 or success_rate < 0.1:  # Very high or very low success rate
            return 0.8
        
        # Check for alternating patterns
        if self._is_alternating_pattern(success_pattern):
            return 0.6
        
        return 0.3  # Low predictability
    
    def _find_repeating_patterns(self, sequence: List, pattern_len: int) -> float:
        """Find repeating patterns of given length."""
        if len(sequence) < pattern_len * 2:
            return 0.0
        
        # Look for patterns
        pattern_count = 0
        for i in range(len(sequence) - pattern_len * 2 + 1):
            pattern = sequence[i:i + pattern_len]
            next_pattern = sequence[i + pattern_len:i + pattern_len * 2]
            
            if pattern == next_pattern:
                pattern_count += 1
        
        # Calculate pattern score
        max_possible_patterns = len(sequence) - pattern_len * 2 + 1
        if max_possible_patterns > 0:
            return pattern_count / max_possible_patterns
        
        return 0.0
    
    def _is_alternating_pattern(self, sequence: List[bool]) -> bool:
        """Check if sequence follows an alternating pattern."""
        if len(sequence) < 4:
            return False
        
        # Check for alternating True/False
        alternating_1 = all(sequence[i] == (i % 2 == 0) for i in range(len(sequence)))
        alternating_2 = all(sequence[i] == (i % 2 == 1) for i in range(len(sequence)))
        
        return alternating_1 or alternating_2


class EnhancedCuriositySystem:
    """
    Main Enhanced Curiosity System that coordinates all curiosity and boredom detection.
    
    This system monitors prediction violations, anomalies, and learning progress to
    guide intelligent exploration and strategy switching.
    """
    
    def __init__(self):
        self.prediction_violation_detector = PredictionViolationDetector()
        self.anomaly_detector = AnomalyDetector()
        self.learning_accelerator = LearningAccelerator()
        self.predictability_tracker = PredictabilityTracker()
        
        # System state
        self.curiosity_level = 0.5
        self.boredom_level = 0.0
        self.strategy_switch_needed = False
        
        # Event tracking
        self.curiosity_events = deque(maxlen=100)
        self.boredom_events = deque(maxlen=100)
        
        logger.info("Enhanced Curiosity System initialized")
    
    def process_environment_update(self, 
                                 prediction: Dict[str, Any],
                                 actual_outcome: Dict[str, Any],
                                 current_behavior: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Process an environment update and detect curiosity/boredom events."""
        
        # Detect prediction violations
        violation_event = self.prediction_violation_detector.detect(prediction, actual_outcome)
        if violation_event:
            self.curiosity_events.append(violation_event)
            self.learning_accelerator.allocate_more_resources(context, violation_event)
        
        # Detect anomalies
        anomaly_event = self.anomaly_detector.detect(current_behavior, actual_outcome)
        if anomaly_event:
            self.curiosity_events.append(anomaly_event)
            self.learning_accelerator.allocate_more_resources(context, anomaly_event)
        
        # Update curiosity level
        self._update_curiosity_level()
        
        # Check for boredom
        self._check_boredom(current_behavior, actual_outcome)
        
        # Update learning accelerator
        self.learning_accelerator.update_acceleration()
        
        return {
            'curiosity_level': self.curiosity_level,
            'boredom_level': self.boredom_level,
            'strategy_switch_needed': self.strategy_switch_needed,
            'learning_rate': self.learning_accelerator.get_learning_rate(),
            'recent_curiosity_events': len([e for e in self.curiosity_events if time.time() - e.timestamp < 300]),
            'recent_boredom_events': len([e for e in self.boredom_events if time.time() - e.timestamp < 300])
        }
    
    def _update_curiosity_level(self):
        """Update curiosity level based on recent events."""
        if not self.curiosity_events:
            # Decay curiosity over time
            self.curiosity_level *= 0.99
            return
        
        # Calculate curiosity boost from recent events
        recent_events = [e for e in self.curiosity_events if time.time() - e.timestamp < 300]  # Last 5 minutes
        
        if recent_events:
            # Boost curiosity based on recent events
            total_intensity = sum(e.intensity for e in recent_events)
            total_learning_potential = sum(e.learning_potential for e in recent_events)
            
            curiosity_boost = (total_intensity + total_learning_potential) / (len(recent_events) * 2.0)
            self.curiosity_level = min(1.0, self.curiosity_level + curiosity_boost * 0.1)
        else:
            # Decay curiosity over time
            self.curiosity_level *= 0.99
        
        # Ensure curiosity stays in valid range
        self.curiosity_level = max(0.0, min(1.0, self.curiosity_level))
    
    def _check_boredom(self, current_behavior: Dict[str, Any], current_outcome: Dict[str, Any]):
        """Check for boredom based on predictability."""
        
        # Get recent actions and outcomes
        recent_actions = current_behavior.get('recent_actions', [])
        recent_outcomes = current_behavior.get('recent_outcomes', [])
        
        # Check if highly predictable
        if self.predictability_tracker.is_highly_predictable(recent_actions, recent_outcomes):
            # Increase boredom level
            self.boredom_level = min(1.0, self.boredom_level + 0.1)
            
            # Create boredom event
            boredom_event = BoredomEvent(
                event_type='high_predictability',
                intensity=self.boredom_level,
                timestamp=time.time(),
                context={
                    'recent_actions': recent_actions,
                    'recent_outcomes': recent_outcomes
                },
                strategy_switch_needed=self.boredom_level > 0.7
            )
            
            self.boredom_events.append(boredom_event)
            
            # Set strategy switch flag
            if self.boredom_level > 0.7:
                self.strategy_switch_needed = True
        else:
            # Decrease boredom level
            self.boredom_level = max(0.0, self.boredom_level - 0.05)
            
            # Reset strategy switch flag if boredom is low
            if self.boredom_level < 0.3:
                self.strategy_switch_needed = False
    
    def is_high_curiosity(self) -> bool:
        """Check if system is in high curiosity state."""
        return self.curiosity_level > 0.8
    
    def is_bored(self) -> bool:
        """Check if system is bored."""
        return self.boredom_level > 0.7
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'curiosity_level': self.curiosity_level,
            'boredom_level': self.boredom_level,
            'strategy_switch_needed': self.strategy_switch_needed,
            'learning_acceleration': self.learning_accelerator.get_acceleration_status(),
            'prediction_violations': self.prediction_violation_detector.get_violation_statistics(),
            'anomalies': self.anomaly_detector.get_anomaly_statistics(),
            'recent_curiosity_events': len([e for e in self.curiosity_events if time.time() - e.timestamp < 300]),
            'recent_boredom_events': len([e for e in self.boredom_events if time.time() - e.timestamp < 300])
        }
    
    def reset_strategy_switch_flag(self):
        """Reset the strategy switch flag."""
        self.strategy_switch_needed = False
    
    def force_curiosity_boost(self, intensity: float = 0.5):
        """Force a curiosity boost (for testing or manual intervention)."""
        self.curiosity_level = min(1.0, self.curiosity_level + intensity)
        logger.info(f"Curiosity boosted by {intensity:.2f}, new level: {self.curiosity_level:.2f}")
    
    def force_boredom_reset(self):
        """Force boredom reset (for testing or manual intervention)."""
        self.boredom_level = 0.0
        self.strategy_switch_needed = False
        logger.info("Boredom reset to 0.0")
