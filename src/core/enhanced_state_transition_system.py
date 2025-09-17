#!/usr/bin/env python3
"""
Enhanced State Transition System for Tabula Rasa

Implements dynamic state transitions with Hidden Markov Models (HMMs) and
entropy-based insight detection, building on existing dual-pathway processing.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import math

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """Enhanced cognitive states including insight moments."""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    INSIGHT = "insight"
    IMPASSE = "impasse"
    RESTRUCTURING = "restructuring"

class StateTransitionTrigger(Enum):
    """Triggers for state transitions."""
    PERFORMANCE_DECLINE = "performance_decline"
    HIGH_ENTROPY = "high_entropy"
    IMPASSE_DETECTED = "impasse_detected"
    INSIGHT_MOMENT = "insight_moment"
    CONFIDENCE_SPIKE = "confidence_spike"
    PATTERN_BREAKTHROUGH = "pattern_breakthrough"

@dataclass
class StateTransition:
    """Represents a state transition with context."""
    from_state: CognitiveState
    to_state: CognitiveState
    trigger: StateTransitionTrigger
    confidence: float
    entropy_level: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InsightMoment:
    """Represents an insight moment with restructuring information."""
    timestamp: float
    entropy_before: float
    entropy_after: float
    restructuring_type: str
    confidence_gain: float
    problem_representation: Dict[str, Any]
    solution_representation: Dict[str, Any]

class HiddenMarkovModel:
    """Simple Hidden Markov Model for state transitions."""
    
    def __init__(self, states: List[CognitiveState]):
        self.states = states
        self.n_states = len(states)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        
        # Initialize transition probabilities
        self.transition_matrix = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)  # Higher self-transition probability
        
        # Initialize emission probabilities (for observations)
        self.emission_matrix = np.random.rand(self.n_states, 10)  # 10 observation features
        self.emission_matrix = self.emission_matrix / self.emission_matrix.sum(axis=1, keepdims=True)
        
        # Initial state probabilities
        self.initial_probs = np.ones(self.n_states) / self.n_states
        
        logger.info(f"HMM initialized with {self.n_states} states: {[s.value for s in states]}")
    
    def predict_next_state(self, current_state: CognitiveState, observations: np.ndarray) -> Tuple[CognitiveState, float]:
        """Predict next state based on current state and observations."""
        current_idx = self.state_to_idx[current_state]
        
        # Calculate state probabilities
        state_probs = self.transition_matrix[current_idx] * self._calculate_emission_probs(observations)
        state_probs = state_probs / state_probs.sum()
        
        # Get most likely next state
        next_state_idx = np.argmax(state_probs)
        next_state = self.states[next_state_idx]
        confidence = state_probs[next_state_idx]
        
        return next_state, confidence
    
    def _calculate_emission_probs(self, observations: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for observations."""
        if len(observations) != self.emission_matrix.shape[1]:
            # Pad or truncate observations to match emission matrix
            padded_obs = np.zeros(self.emission_matrix.shape[1])
            min_len = min(len(observations), len(padded_obs))
            padded_obs[:min_len] = observations[:min_len]
            observations = padded_obs
        
        # Calculate probabilities for each state
        probs = np.zeros(self.n_states)
        for i in range(self.n_states):
            probs[i] = np.prod(self.emission_matrix[i] ** observations)
        
        return probs
    
    def update_transition_probabilities(self, transition: StateTransition):
        """Update transition probabilities based on observed transition."""
        from_idx = self.state_to_idx[transition.from_state]
        to_idx = self.state_to_idx[transition.to_state]
        
        # Update transition probability with learning rate
        learning_rate = 0.1
        self.transition_matrix[from_idx, to_idx] += learning_rate * transition.confidence
        self.transition_matrix[from_idx] = self.transition_matrix[from_idx] / self.transition_matrix[from_idx].sum()

class EntropyTracker:
    """Tracks entropy in performance and decision patterns."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.decision_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)
        
    def add_observation(self, performance: float, decision: Dict[str, Any]):
        """Add new performance and decision observation."""
        self.performance_history.append(performance)
        self.decision_history.append(decision)
        
        # Calculate current entropy
        current_entropy = self.calculate_entropy()
        self.entropy_history.append(current_entropy)
        
        return current_entropy
    
    def calculate_entropy(self) -> float:
        """Calculate entropy of recent performance and decision patterns."""
        if len(self.performance_history) < 3:
            return 0.0
        
        # Calculate entropy of performance variance
        performance_array = np.array(list(self.performance_history))
        performance_variance = np.var(performance_array)
        
        # Calculate entropy of decision diversity
        decision_diversity = self._calculate_decision_diversity()
        
        # Combine performance and decision entropy
        total_entropy = performance_variance + decision_diversity
        
        return total_entropy
    
    def _calculate_decision_diversity(self) -> float:
        """Calculate diversity of recent decisions."""
        if len(self.decision_history) < 2:
            return 0.0
        
        # Count unique decision types
        decision_types = [d.get('type', 'unknown') for d in self.decision_history]
        unique_types = len(set(decision_types))
        total_decisions = len(decision_types)
        
        # Calculate entropy based on decision diversity
        if total_decisions == 0:
            return 0.0
        
        diversity_ratio = unique_types / total_decisions
        return -diversity_ratio * math.log(diversity_ratio + 1e-10)
    
    def detect_high_entropy_period(self, threshold: float = 0.5) -> bool:
        """Detect if we're in a high entropy period."""
        if len(self.entropy_history) < 5:
            return False
        
        recent_entropy = list(self.entropy_history)[-5:]
        avg_entropy = np.mean(recent_entropy)
        
        return avg_entropy > threshold

class EnhancedStateTransitionSystem:
    """Enhanced state transition system with HMM and insight detection."""
    
    def __init__(self, 
                 insight_threshold: float = 0.7,
                 entropy_threshold: float = 0.5,
                 impasse_detection_window: int = 10):
        self.insight_threshold = insight_threshold
        self.entropy_threshold = entropy_threshold
        self.impasse_detection_window = impasse_detection_window
        
        # Initialize HMM
        self.hmm = HiddenMarkovModel(list(CognitiveState))
        
        # Initialize entropy tracker
        self.entropy_tracker = EntropyTracker()
        
        # State tracking
        self.current_state = CognitiveState.ANALYTICAL
        self.state_history = deque(maxlen=100)
        self.transition_history = deque(maxlen=50)
        self.insight_moments = deque(maxlen=20)
        
        # Performance tracking for impasse detection
        self.performance_history = deque(maxlen=impasse_detection_window)
        self.confidence_history = deque(maxlen=impasse_detection_window)
        
        logger.info("Enhanced State Transition System initialized")
    
    def process_decision(self, 
                        decision: Dict[str, Any], 
                        performance: float, 
                        confidence: float,
                        context: Dict[str, Any]) -> Tuple[CognitiveState, Optional[InsightMoment]]:
        """Process a decision and determine state transitions."""
        
        # Add observation to entropy tracker
        entropy = self.entropy_tracker.add_observation(performance, decision)
        
        # Update performance and confidence history
        self.performance_history.append(performance)
        self.confidence_history.append(confidence)
        
        # Detect potential triggers
        triggers = self._detect_triggers(entropy, performance, confidence, context)
        
        # Predict next state using HMM
        observations = self._create_observation_vector(performance, confidence, entropy, context)
        predicted_state, state_confidence = self.hmm.predict_next_state(self.current_state, observations)
        
        # Determine if we should transition
        should_transition, transition_trigger = self._should_transition(
            predicted_state, state_confidence, triggers
        )
        
        insight_moment = None
        if should_transition:
            # Perform state transition
            old_state = self.current_state
            self.current_state = predicted_state
            
            # Create transition record
            transition = StateTransition(
                from_state=old_state,
                to_state=self.current_state,
                trigger=transition_trigger,
                confidence=state_confidence,
                entropy_level=entropy,
                timestamp=time.time(),
                context=context
            )
            
            # Update HMM with observed transition
            self.hmm.update_transition_probabilities(transition)
            
            # Record transition
            self.transition_history.append(transition)
            
            # Check for insight moment
            insight_moment = self._detect_insight_moment(transition, entropy, performance, confidence)
            if insight_moment:
                self.insight_moments.append(insight_moment)
                logger.info(f"Insight moment detected: {insight_moment.restructuring_type}")
        
        # Record current state
        self.state_history.append(self.current_state)
        
        return self.current_state, insight_moment
    
    def _detect_triggers(self, 
                        entropy: float, 
                        performance: float, 
                        confidence: float,
                        context: Dict[str, Any]) -> List[StateTransitionTrigger]:
        """Detect potential state transition triggers."""
        triggers = []
        
        # High entropy trigger
        if entropy > self.entropy_threshold:
            triggers.append(StateTransitionTrigger.HIGH_ENTROPY)
        
        # Performance decline trigger
        if len(self.performance_history) >= 3:
            recent_performance = list(self.performance_history)[-3:]
            if all(recent_performance[i] < recent_performance[i-1] for i in range(1, len(recent_performance))):
                triggers.append(StateTransitionTrigger.PERFORMANCE_DECLINE)
        
        # Impasse detection
        if self._detect_impasse():
            triggers.append(StateTransitionTrigger.IMPASSE_DETECTED)
        
        # Confidence spike
        if len(self.confidence_history) >= 2:
            if confidence > max(self.confidence_history) * 1.5:
                triggers.append(StateTransitionTrigger.CONFIDENCE_SPIKE)
        
        # Pattern breakthrough
        if context.get('pattern_breakthrough', False):
            triggers.append(StateTransitionTrigger.PATTERN_BREAKTHROUGH)
        
        return triggers
    
    def _detect_impasse(self) -> bool:
        """Detect if the system is in an impasse."""
        if len(self.performance_history) < self.impasse_detection_window:
            return False
        
        # Check for stagnation in performance
        recent_performance = list(self.performance_history)
        performance_variance = np.var(recent_performance)
        
        # Check for low confidence
        recent_confidence = list(self.confidence_history)
        avg_confidence = np.mean(recent_confidence)
        
        # Impasse if low performance variance and low confidence
        return performance_variance < 0.01 and avg_confidence < 0.3
    
    def _should_transition(self, 
                          predicted_state: CognitiveState, 
                          state_confidence: float,
                          triggers: List[StateTransitionTrigger]) -> Tuple[bool, Optional[StateTransitionTrigger]]:
        """Determine if we should transition to the predicted state."""
        
        # Don't transition if already in predicted state
        if predicted_state == self.current_state:
            return False, None
        
        # Transition if high confidence and relevant triggers
        if state_confidence > 0.6:
            # Find most relevant trigger
            if StateTransitionTrigger.IMPASSE_DETECTED in triggers:
                return True, StateTransitionTrigger.IMPASSE_DETECTED
            elif StateTransitionTrigger.HIGH_ENTROPY in triggers:
                return True, StateTransitionTrigger.HIGH_ENTROPY
            elif StateTransitionTrigger.CONFIDENCE_SPIKE in triggers:
                return True, StateTransitionTrigger.CONFIDENCE_SPIKE
            elif StateTransitionTrigger.PATTERN_BREAKTHROUGH in triggers:
                return True, StateTransitionTrigger.PATTERN_BREAKTHROUGH
            elif StateTransitionTrigger.PERFORMANCE_DECLINE in triggers:
                return True, StateTransitionTrigger.PERFORMANCE_DECLINE
        
        return False, None
    
    def _detect_insight_moment(self, 
                              transition: StateTransition, 
                              entropy: float, 
                              performance: float, 
                              confidence: float) -> Optional[InsightMoment]:
        """Detect if this transition represents an insight moment."""
        
        # Insight moments typically involve:
        # 1. High entropy before transition
        # 2. Significant confidence gain
        # 3. Transition to insight or restructuring state
        # 4. Performance improvement
        
        if transition.to_state not in [CognitiveState.INSIGHT, CognitiveState.RESTRUCTURING]:
            return None
        
        # Check for high entropy before transition
        if entropy < self.entropy_threshold:
            return None
        
        # Check for confidence gain
        if len(self.confidence_history) >= 2:
            confidence_gain = confidence - self.confidence_history[-2]
            if confidence_gain < 0.2:  # Significant confidence gain
                return None
        else:
            confidence_gain = 0.0
        
        # Create insight moment
        insight_moment = InsightMoment(
            timestamp=time.time(),
            entropy_before=entropy,
            entropy_after=entropy * 0.5,  # Entropy typically decreases after insight
            restructuring_type=self._classify_restructuring_type(transition, context),
            confidence_gain=confidence_gain,
            problem_representation=transition.context.get('problem_representation', {}),
            solution_representation=transition.context.get('solution_representation', {})
        )
        
        return insight_moment
    
    def _classify_restructuring_type(self, transition: StateTransition, context: Dict[str, Any]) -> str:
        """Classify the type of restructuring that occurred."""
        if transition.to_state == CognitiveState.INSIGHT:
            return "insight_breakthrough"
        elif transition.to_state == CognitiveState.RESTRUCTURING:
            return "representation_restructuring"
        else:
            return "unknown"
    
    def _create_observation_vector(self, 
                                  performance: float, 
                                  confidence: float, 
                                  entropy: float,
                                  context: Dict[str, Any]) -> np.ndarray:
        """Create observation vector for HMM."""
        # Create 10-dimensional observation vector
        observations = np.zeros(10)
        
        # Performance features
        observations[0] = performance
        observations[1] = confidence
        observations[2] = entropy
        
        # Context features
        observations[3] = context.get('difficulty', 0.5)
        observations[4] = context.get('novelty', 0.5)
        observations[5] = context.get('complexity', 0.5)
        
        # Historical features
        if len(self.performance_history) > 0:
            observations[6] = np.mean(list(self.performance_history))
            observations[7] = np.std(list(self.performance_history))
        
        if len(self.confidence_history) > 0:
            observations[8] = np.mean(list(self.confidence_history))
            observations[9] = np.std(list(self.confidence_history))
        
        return observations
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Get statistics about state transitions and insights."""
        if not self.state_history:
            return {}
        
        # State distribution
        state_counts = {}
        for state in self.state_history:
            state_counts[state.value] = state_counts.get(state.value, 0) + 1
        
        # Transition statistics
        transition_counts = {}
        for transition in self.transition_history:
            key = f"{transition.from_state.value}->{transition.to_state.value}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Insight statistics
        insight_count = len(self.insight_moments)
        avg_entropy = np.mean([im.entropy_before for im in self.insight_moments]) if self.insight_moments else 0.0
        
        return {
            'current_state': self.current_state.value,
            'state_distribution': state_counts,
            'transition_counts': transition_counts,
            'insight_moments': insight_count,
            'avg_insight_entropy': avg_entropy,
            'total_transitions': len(self.transition_history),
            'entropy_trend': list(self.entropy_tracker.entropy_history)[-10:] if self.entropy_tracker.entropy_history else []
        }

# Factory function for easy integration
def create_enhanced_state_transition_system(**kwargs) -> EnhancedStateTransitionSystem:
    """Create a configured enhanced state transition system."""
    return EnhancedStateTransitionSystem(**kwargs)
