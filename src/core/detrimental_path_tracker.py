#!/usr/bin/env python3
"""
Detrimental Path Tracking System

This module implements comprehensive tracking and learning from failed actions,
negative outcomes, and detrimental patterns. This enables the AI to learn from
failures, not just successes, creating a more robust learning system.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures that can be tracked."""
    ZERO_PROGRESS = "zero_progress"
    ENERGY_LOSS = "energy_loss"
    COORDINATE_STUCK = "coordinate_stuck"
    PREDICTION_ERROR = "prediction_error"
    LEARNING_REGRESSION = "learning_regression"
    STRATEGY_FAILURE = "strategy_failure"
    SIMULATION_FAILURE = "simulation_failure"

class SeverityLevel(Enum):
    """Severity levels for detrimental patterns."""
    LOW = "low"           # Minor setbacks
    MEDIUM = "medium"     # Noticeable failures
    HIGH = "high"         # Significant failures
    CRITICAL = "critical" # System-breaking failures

@dataclass
class DetrimentalPattern:
    """A pattern of actions that consistently leads to negative outcomes."""
    pattern_id: str
    action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]  # (action, coordinates)
    failure_type: FailureType
    severity: SeverityLevel
    
    # Failure metrics
    failure_count: int = 0
    total_attempts: int = 0
    failure_rate: float = 0.0
    average_energy_loss: float = 0.0
    average_score_loss: float = 0.0
    
    # Context information
    game_contexts: Set[str] = field(default_factory=set)
    environmental_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal tracking
    first_detected: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    
    # Learning metrics
    avoidance_attempts: int = 0
    avoidance_success_rate: float = 0.0
    confidence: float = 0.0  # How confident we are this is truly detrimental
    
    def update_failure(self, energy_loss: float = 0.0, score_loss: float = 0.0, game_id: str = ""):
        """Update failure metrics when this pattern occurs."""
        self.failure_count += 1
        self.total_attempts += 1
        self.last_occurrence = time.time()
        self.consecutive_failures += 1
        
        if game_id:
            self.game_contexts.add(game_id)
        
        # Update running averages
        self.average_energy_loss = (
            (self.average_energy_loss * (self.failure_count - 1) + energy_loss) / self.failure_count
        )
        self.average_score_loss = (
            (self.average_score_loss * (self.failure_count - 1) + score_loss) / self.failure_count
        )
        
        # Update failure rate
        self.failure_rate = self.failure_count / self.total_attempts
        
        # Update confidence based on consistency and sample size
        self.confidence = min(0.95, 
                             (self.total_attempts / 10.0) * 0.4 +
                             self.failure_rate * 0.6)
        
        # Update severity based on failure rate and impact
        if self.failure_rate > 0.8 and self.average_energy_loss > 5.0:
            self.severity = SeverityLevel.CRITICAL
        elif self.failure_rate > 0.6 and self.average_energy_loss > 2.0:
            self.severity = SeverityLevel.HIGH
        elif self.failure_rate > 0.4:
            self.severity = SeverityLevel.MEDIUM
        else:
            self.severity = SeverityLevel.LOW
    
    def update_success(self):
        """Update when this pattern succeeds (resets consecutive failures)."""
        self.total_attempts += 1
        self.consecutive_failures = 0
        self.failure_rate = self.failure_count / self.total_attempts
    
    def should_avoid(self, current_context: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Determine if this pattern should be avoided in current context.
        
        Returns:
            (should_avoid, confidence)
        """
        # High confidence detrimental patterns should always be avoided
        if self.confidence > 0.8 and self.failure_rate > 0.7:
            return True, self.confidence
        
        # Critical severity patterns should be avoided
        if self.severity == SeverityLevel.CRITICAL:
            return True, 0.9
        
        # Recent consecutive failures
        if self.consecutive_failures > 5:
            return True, 0.8
        
        # Context-specific avoidance
        if self._matches_context(current_context):
            return True, self.confidence * 0.8
        
        return False, 0.0
    
    def _matches_context(self, context: Dict[str, Any]) -> bool:
        """Check if current context matches this pattern's failure conditions."""
        # Simple context matching - can be enhanced
        if not self.environmental_conditions:
            return False
        
        for key, value in self.environmental_conditions.items():
            if context.get(key) != value:
                return False
        
        return True

class DetrimentalPathTracker:
    """
    Tracks and learns from detrimental action patterns and negative outcomes.
    
    This system enables the AI to:
    1. Identify patterns that consistently lead to failures
    2. Rank detrimental patterns by severity and frequency
    3. Avoid known bad patterns in future decisions
    4. Learn from negative feedback to improve predictions
    """
    
    def __init__(self, 
                 max_patterns: int = 1000,
                 min_failure_rate: float = 0.3,
                 persistence_dir: str = "data/detrimental_patterns"):
        self.max_patterns = max_patterns
        self.min_failure_rate = min_failure_rate
        # Database-only mode: No file-based persistence
        self.persistence_dir = None  # Disabled for database-only mode
        # self.persistence_dir.mkdir(parents=True, exist_ok=True)  # Database-only mode: No file creation
        
        # Pattern storage
        self.detrimental_patterns: Dict[str, DetrimentalPattern] = {}
        self.pattern_sequence_map: Dict[Tuple, str] = {}  # Maps action sequences to pattern IDs
        
        # Failure tracking
        self.recent_failures: deque = deque(maxlen=1000)
        self.failure_history: List[Dict[str, Any]] = []
        
        # Learning metrics
        self.total_patterns_detected = 0
        self.patterns_avoided = 0
        self.successful_avoidances = 0
        
        # Load existing patterns
        self._load_patterns()
        
        logger.info(f"Detrimental Path Tracker initialized with {len(self.detrimental_patterns)} existing patterns")
    
    def record_failure(self, 
                      action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]],
                      failure_type: FailureType,
                      energy_loss: float = 0.0,
                      score_loss: float = 0.0,
                      game_id: str = "",
                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Record a failure and update detrimental pattern tracking.
        
        Args:
            action_sequence: The sequence of actions that led to failure
            failure_type: Type of failure that occurred
            energy_loss: Amount of energy lost
            score_loss: Amount of score lost
            game_id: Current game identifier
            context: Environmental context when failure occurred
            
        Returns:
            Pattern ID of the detrimental pattern
        """
        # Create pattern key from action sequence
        pattern_key = self._create_pattern_key(action_sequence)
        
        # Check if we already have this pattern
        if pattern_key in self.pattern_sequence_map:
            pattern_id = self.pattern_sequence_map[pattern_key]
            pattern = self.detrimental_patterns[pattern_id]
        else:
            # Create new detrimental pattern
            pattern_id = f"detrimental_{int(time.time() * 1000)}_{len(self.detrimental_patterns)}"
            pattern = DetrimentalPattern(
                pattern_id=pattern_id,
                action_sequence=action_sequence,
                failure_type=failure_type,
                severity=SeverityLevel.LOW
            )
            self.detrimental_patterns[pattern_id] = pattern
            self.pattern_sequence_map[pattern_key] = pattern_id
            self.total_patterns_detected += 1
        
        # Update pattern with failure
        pattern.update_failure(energy_loss, score_loss, game_id)
        
        # Store context if provided
        if context:
            pattern.environmental_conditions.update(context)
        
        # Record failure in history
        failure_record = {
            "timestamp": time.time(),
            "pattern_id": pattern_id,
            "failure_type": failure_type.value,
            "energy_loss": energy_loss,
            "score_loss": score_loss,
            "game_id": game_id,
            "action_sequence": action_sequence
        }
        self.recent_failures.append(failure_record)
        self.failure_history.append(failure_record)
        
        # Clean up old patterns if we have too many
        if len(self.detrimental_patterns) > self.max_patterns:
            self._cleanup_old_patterns()
        
        logger.debug(f"Recorded failure for pattern {pattern_id}: {failure_type.value}")
        return pattern_id
    
    def record_success(self, action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]):
        """Record a success for an action sequence (may reduce detrimental pattern confidence)."""
        pattern_key = self._create_pattern_key(action_sequence)
        
        if pattern_key in self.pattern_sequence_map:
            pattern_id = self.pattern_sequence_map[pattern_key]
            pattern = self.detrimental_patterns[pattern_id]
            pattern.update_success()
    
    def should_avoid_sequence(self, 
                            action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]],
                            context: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, str]:
        """
        Check if an action sequence should be avoided based on detrimental patterns.
        
        Special handling for ACTION5 (interact/select/rotate/attach/detach/execute):
        - ACTION5 sequences are only avoided if they consistently fail WITHOUT enabling other actions
        - ACTION5 followed by successful actions gets reduced penalty
        - Context-dependent ACTION5 usage is preserved
        
        Returns:
            (should_avoid, confidence, reason)
        """
        pattern_key = self._create_pattern_key(action_sequence)
        
        if pattern_key not in self.pattern_sequence_map:
            return False, 0.0, "No known detrimental pattern"
        
        pattern_id = self.pattern_sequence_map[pattern_key]
        pattern = self.detrimental_patterns[pattern_id]
        
        # Special handling for ACTION5 sequences
        if self._is_action5_sequence(action_sequence):
            should_avoid, confidence, reason = self._evaluate_action5_sequence(pattern, context or {})
            if should_avoid:
                self.patterns_avoided += 1
            return should_avoid, confidence, reason
        
        # Standard evaluation for non-ACTION5 sequences
        should_avoid, confidence = pattern.should_avoid(context or {})
        
        if should_avoid:
            self.patterns_avoided += 1
            reason = f"Detrimental pattern {pattern_id}: {pattern.failure_rate:.2f} failure rate, {pattern.severity.value} severity"
            return True, confidence, reason
        
        return False, 0.0, "Pattern not sufficiently detrimental"
    
    def get_detrimental_patterns(self, 
                               min_confidence: float = 0.5,
                               max_patterns: int = 10) -> List[DetrimentalPattern]:
        """Get the most detrimental patterns sorted by severity and confidence."""
        patterns = list(self.detrimental_patterns.values())
        
        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # Sort by severity (critical first) then confidence
        severity_order = {SeverityLevel.CRITICAL: 4, SeverityLevel.HIGH: 3, 
                         SeverityLevel.MEDIUM: 2, SeverityLevel.LOW: 1}
        
        patterns.sort(key=lambda p: (severity_order[p.severity], p.confidence), reverse=True)
        
        return patterns[:max_patterns]
    
    def get_avoidance_recommendations(self, 
                                    current_context: Dict[str, Any],
                                    max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for avoiding detrimental patterns in current context."""
        recommendations = []
        
        for pattern in self.get_detrimental_patterns():
            should_avoid, confidence, reason = self.should_avoid_sequence(
                pattern.action_sequence, current_context
            )
            
            if should_avoid:
                recommendations.append({
                    "pattern_id": pattern.pattern_id,
                    "action_sequence": pattern.action_sequence,
                    "reason": reason,
                    "confidence": confidence,
                    "severity": pattern.severity.value,
                    "failure_rate": pattern.failure_rate,
                    "avoidance_strategy": self._generate_avoidance_strategy(pattern)
                })
        
        return recommendations[:max_recommendations]
    
    def update_avoidance_success(self, pattern_id: str, was_successful: bool):
        """Update whether avoiding a pattern was successful."""
        if pattern_id in self.detrimental_patterns:
            pattern = self.detrimental_patterns[pattern_id]
            pattern.avoidance_attempts += 1
            
            if was_successful:
                self.successful_avoidances += 1
                pattern.avoidance_success_rate = (
                    (pattern.avoidance_success_rate * (pattern.avoidance_attempts - 1) + 1.0) / 
                    pattern.avoidance_attempts
                )
            else:
                pattern.avoidance_success_rate = (
                    (pattern.avoidance_success_rate * (pattern.avoidance_attempts - 1)) / 
                    pattern.avoidance_attempts
                )
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get metrics about detrimental pattern learning."""
        total_patterns = len(self.detrimental_patterns)
        high_confidence_patterns = len([p for p in self.detrimental_patterns.values() if p.confidence > 0.7])
        critical_patterns = len([p for p in self.detrimental_patterns.values() if p.severity == SeverityLevel.CRITICAL])
        
        return {
            "total_patterns": total_patterns,
            "high_confidence_patterns": high_confidence_patterns,
            "critical_patterns": critical_patterns,
            "patterns_avoided": self.patterns_avoided,
            "successful_avoidances": self.successful_avoidances,
            "avoidance_success_rate": (
                self.successful_avoidances / max(self.patterns_avoided, 1)
            ),
            "recent_failures": len(self.recent_failures),
            "total_failures_recorded": len(self.failure_history)
        }
    
    def _create_pattern_key(self, action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]) -> Tuple:
        """Create a hashable key for an action sequence."""
        # Normalize coordinates to reduce noise
        normalized_sequence = []
        for action, coords in action_sequence:
            if coords:
                # Round coordinates to reduce precision
                normalized_coords = (round(coords[0] / 5) * 5, round(coords[1] / 5) * 5)
                normalized_sequence.append((action, normalized_coords))
            else:
                normalized_sequence.append((action, None))
        
        return tuple(normalized_sequence)
    
    def _is_action5_sequence(self, action_sequence: List[Tuple[int, Optional[Tuple[int, int]]]]) -> bool:
        """Check if the sequence contains ACTION5 (interact/select/rotate/attach/detach/execute)."""
        return any(action == 5 for action, _ in action_sequence)
    
    def _evaluate_action5_sequence(self, pattern: DetrimentalPattern, context: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Special evaluation for ACTION5 sequences that considers their enabling nature.
        
        ACTION5 can enable other actions, so we need to be more careful about avoiding it.
        """
        # Check if this is a pure ACTION5 sequence (only ACTION5s)
        action5_only = all(action == 5 for action, _ in pattern.action_sequence)
        
        if action5_only:
            # Pure ACTION5 sequences are only avoided if they have very high failure rate
            # and no evidence of enabling other actions
            if pattern.failure_rate > 0.9 and pattern.confidence > 0.8:
                return True, pattern.confidence * 0.7, f"Pure ACTION5 sequence with {pattern.failure_rate:.2f} failure rate"
            else:
                return False, 0.0, "ACTION5 sequence may be enabling other actions"
        
        # Mixed sequences with ACTION5 - check if ACTION5 is followed by successful actions
        action5_position = next((i for i, (action, _) in enumerate(pattern.action_sequence) if action == 5), -1)
        
        if action5_position >= 0:
            # Check if this is a setup sequence (ACTION5 followed by other actions)
            if action5_position < len(pattern.action_sequence) - 1:
                # ACTION5 is not the last action - it might be enabling
                # Only avoid if the entire sequence consistently fails
                if pattern.failure_rate > 0.8 and pattern.severity == SeverityLevel.CRITICAL:
                    return True, pattern.confidence * 0.6, f"ACTION5 setup sequence with {pattern.failure_rate:.2f} failure rate"
                else:
                    return False, 0.0, "ACTION5 may be enabling subsequent actions"
            else:
                # ACTION5 is the last action - more likely to be detrimental if it fails
                if pattern.failure_rate > 0.7:
                    return True, pattern.confidence * 0.8, f"ACTION5 final action with {pattern.failure_rate:.2f} failure rate"
                else:
                    return False, 0.0, "ACTION5 final action not sufficiently detrimental"
        
        # Fallback to standard evaluation
        should_avoid, confidence = pattern.should_avoid(context)
        reason = f"ACTION5 sequence: {pattern.failure_rate:.2f} failure rate, {pattern.severity.value} severity"
        return should_avoid, confidence, reason
    
    def _generate_avoidance_strategy(self, pattern: DetrimentalPattern) -> str:
        """Generate a strategy for avoiding a detrimental pattern."""
        # Check for ACTION5 sequences first (most specific)
        if self._is_action5_sequence(pattern.action_sequence):
            return "Consider if ACTION5 is enabling other actions before avoiding"
        elif pattern.failure_type == FailureType.COORDINATE_STUCK:
            return "Avoid coordinates in similar regions"
        elif pattern.failure_type == FailureType.ENERGY_LOSS:
            return "Use energy-conserving actions instead"
        elif pattern.failure_type == FailureType.ZERO_PROGRESS:
            return "Try different action sequences or exploration strategies"
        else:
            return "Avoid this specific action sequence"
    
    def _cleanup_old_patterns(self):
        """Remove old, low-confidence patterns to make room for new ones."""
        patterns = list(self.detrimental_patterns.items())
        
        # Sort by confidence and recency
        patterns.sort(key=lambda x: (x[1].confidence, x[1].last_occurrence), reverse=True)
        
        # Keep top patterns
        keep_count = self.max_patterns // 2
        patterns_to_keep = patterns[:keep_count]
        
        # Rebuild dictionaries
        self.detrimental_patterns = dict(patterns_to_keep)
        self.pattern_sequence_map = {}
        
        for pattern_id, pattern in self.detrimental_patterns.items():
            pattern_key = self._create_pattern_key(pattern.action_sequence)
            self.pattern_sequence_map[pattern_key] = pattern_id
    
    def _load_patterns(self):
        """Load detrimental patterns from persistence."""
        # Database-only mode: Skip file-based pattern loading
        if self.persistence_dir is None:
            return
            
        pattern_file = self.persistence_dir / "detrimental_patterns.json"
        
        if not pattern_file.exists():
            return
        
        try:
            with open(pattern_file, 'r') as f:
                data = json.load(f)
            
            for pattern_data in data.get('patterns', []):
                pattern = DetrimentalPattern(
                    pattern_id=pattern_data['pattern_id'],
                    action_sequence=pattern_data['action_sequence'],
                    failure_type=FailureType(pattern_data['failure_type']),
                    severity=SeverityLevel(pattern_data['severity']),
                    failure_count=pattern_data.get('failure_count', 0),
                    total_attempts=pattern_data.get('total_attempts', 0),
                    failure_rate=pattern_data.get('failure_rate', 0.0),
                    average_energy_loss=pattern_data.get('average_energy_loss', 0.0),
                    average_score_loss=pattern_data.get('average_score_loss', 0.0),
                    game_contexts=set(pattern_data.get('game_contexts', [])),
                    environmental_conditions=pattern_data.get('environmental_conditions', {}),
                    first_detected=pattern_data.get('first_detected', time.time()),
                    last_occurrence=pattern_data.get('last_occurrence', time.time()),
                    consecutive_failures=pattern_data.get('consecutive_failures', 0),
                    avoidance_attempts=pattern_data.get('avoidance_attempts', 0),
                    avoidance_success_rate=pattern_data.get('avoidance_success_rate', 0.0),
                    confidence=pattern_data.get('confidence', 0.0)
                )
                
                self.detrimental_patterns[pattern.pattern_id] = pattern
                
                # Rebuild sequence map
                pattern_key = self._create_pattern_key(pattern.action_sequence)
                self.pattern_sequence_map[pattern_key] = pattern.pattern_id
            
            logger.info(f"Loaded {len(self.detrimental_patterns)} detrimental patterns")
            
        except Exception as e:
            logger.error(f"Failed to load detrimental patterns: {e}")
    
    def save_patterns(self):
        """Save detrimental patterns to persistence."""
        pattern_file = self.persistence_dir / "detrimental_patterns.json"
        
        try:
            data = {
                "patterns": [],
                "metadata": {
                    "total_patterns": len(self.detrimental_patterns),
                    "last_updated": time.time(),
                    "version": "1.0"
                }
            }
            
            for pattern in self.detrimental_patterns.values():
                pattern_data = {
                    "pattern_id": pattern.pattern_id,
                    "action_sequence": pattern.action_sequence,
                    "failure_type": pattern.failure_type.value,
                    "severity": pattern.severity.value,
                    "failure_count": pattern.failure_count,
                    "total_attempts": pattern.total_attempts,
                    "failure_rate": pattern.failure_rate,
                    "average_energy_loss": pattern.average_energy_loss,
                    "average_score_loss": pattern.average_score_loss,
                    "game_contexts": list(pattern.game_contexts),
                    "environmental_conditions": pattern.environmental_conditions,
                    "first_detected": pattern.first_detected,
                    "last_occurrence": pattern.last_occurrence,
                    "consecutive_failures": pattern.consecutive_failures,
                    "avoidance_attempts": pattern.avoidance_attempts,
                    "avoidance_success_rate": pattern.avoidance_success_rate,
                    "confidence": pattern.confidence
                }
                data["patterns"].append(pattern_data)
            
            with open(pattern_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.detrimental_patterns)} detrimental patterns")
            
        except Exception as e:
            logger.error(f"Failed to save detrimental patterns: {e}")
