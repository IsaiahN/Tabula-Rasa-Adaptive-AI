#!/usr/bin/env python3
"""
Outcome Tracking System for Meta-Cognitive Components

This module provides comprehensive tracking of the effectiveness of
Governor recommendations and Architect mutations, enabling the 
meta-cognitive systems to learn and adapt based on actual performance.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import deque, defaultdict
import statistics

class OutcomeType(Enum):
    """Types of outcomes that can be tracked."""
    GOVERNOR_RECOMMENDATION = "governor_recommendation"
    ARCHITECT_MUTATION = "architect_mutation"
    SYSTEM_PERFORMANCE = "system_performance"

class OutcomeStatus(Enum):
    """Status of an outcome measurement."""
    PENDING = "pending"        # Outcome measurement in progress
    SUCCESS = "success"        # Positive improvement observed
    FAILURE = "failure"        # No improvement or regression
    PARTIAL = "partial"        # Mixed results
    INCONCLUSIVE = "inconclusive"  # Not enough data

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for outcome tracking."""
    win_rate: float = 0.0
    average_score: float = 0.0
    learning_efficiency: float = 0.0
    knowledge_transfer: float = 0.0
    computational_efficiency: float = 0.0
    memory_usage: float = 0.0
    inference_speed: float = 0.0
    
    def total_performance_score(self) -> float:
        """Calculate weighted total performance score."""
        return (
            self.win_rate * 0.35 +
            (self.average_score / 100.0) * 0.25 +  # Normalize score
            self.learning_efficiency * 0.20 +
            self.knowledge_transfer * 0.10 +
            self.computational_efficiency * 0.05 +
            (1.0 - self.memory_usage) * 0.03 +  # Lower memory usage is better
            self.inference_speed * 0.02
        )
    
    def compare_to(self, baseline: 'PerformanceMetrics') -> Dict[str, float]:
        """Compare this metrics to a baseline."""
        return {
            'win_rate_delta': self.win_rate - baseline.win_rate,
            'score_delta': self.average_score - baseline.average_score,
            'learning_efficiency_delta': self.learning_efficiency - baseline.learning_efficiency,
            'knowledge_transfer_delta': self.knowledge_transfer - baseline.knowledge_transfer,
            'computational_efficiency_delta': self.computational_efficiency - baseline.computational_efficiency,
            'memory_usage_delta': self.memory_usage - baseline.memory_usage,
            'inference_speed_delta': self.inference_speed - baseline.inference_speed,
            'total_performance_delta': self.total_performance_score() - baseline.total_performance_score()
        }

@dataclass
class OutcomeRecord:
    """Detailed record of an outcome measurement."""
    outcome_id: str
    outcome_type: OutcomeType
    decision_id: str  # Links to original decision
    
    # What was changed
    intervention_type: str
    intervention_details: Dict[str, Any]
    
    # Performance measurement
    baseline_metrics: PerformanceMetrics
    post_intervention_metrics: PerformanceMetrics
    performance_deltas: Dict[str, float]
    
    # Outcome assessment
    status: OutcomeStatus
    success_score: float  # 0.0 to 1.0
    confidence: float     # How confident are we in this assessment
    
    # Context
    measurement_duration: float  # How long we measured
    sample_size: int            # Number of test cases
    context_factors: Dict[str, Any]  # Environmental factors
    
    # Metadata
    timestamp: float
    measured_by: str
    notes: str

class OutcomeTracker:
    """Tracks and analyzes outcomes of meta-cognitive decisions."""
    
    def __init__(self, data_dir: Path, logger: Optional[logging.Logger] = None):
        # Database-only mode: No file-based data directory
        self.data_dir = None  # Disabled for database-only mode
        # self.data_dir.mkdir(exist_ok=True)  # Database-only mode: No file creation
        self.logger = logger or logging.getLogger(f"{__name__}.OutcomeTracker")
        
        # Outcome storage
        self.outcome_history = deque(maxlen=10000)
        self.pending_outcomes = {}  # outcome_id -> OutcomeRecord
        
        # Analysis caches
        self.success_rates = defaultdict(lambda: deque(maxlen=100))
        self.performance_trends = defaultdict(lambda: deque(maxlen=200))
        
        # Load existing data
        self._load_existing_data()
    
    def start_outcome_measurement(self, decision_id: str, intervention_type: str, 
                                 intervention_details: Dict[str, Any],
                                 baseline_metrics: PerformanceMetrics) -> str:
        """Start tracking an outcome for a specific intervention."""
        outcome_id = f"outcome_{int(time.time())}_{hash(decision_id) % 10000}"
        
        outcome_record = OutcomeRecord(
            outcome_id=outcome_id,
            outcome_type=OutcomeType.GOVERNOR_RECOMMENDATION if 'governor' in intervention_type.lower() 
                        else OutcomeType.ARCHITECT_MUTATION if 'architect' in intervention_type.lower()
                        else OutcomeType.SYSTEM_PERFORMANCE,
            decision_id=decision_id,
            intervention_type=intervention_type,
            intervention_details=intervention_details,
            baseline_metrics=baseline_metrics,
            post_intervention_metrics=PerformanceMetrics(),  # Will be filled later
            performance_deltas={},
            status=OutcomeStatus.PENDING,
            success_score=0.0,
            confidence=0.0,
            measurement_duration=0.0,
            sample_size=0,
            context_factors={},
            timestamp=time.time(),
            measured_by="OutcomeTracker",
            notes=""
        )
        
        self.pending_outcomes[outcome_id] = outcome_record
        self.logger.info(f"Started outcome measurement {outcome_id} for {intervention_type}")
        return outcome_id
    
    def complete_outcome_measurement(self, outcome_id: str, 
                                   post_metrics: PerformanceMetrics,
                                   sample_size: int,
                                   context_factors: Dict[str, Any] = None,
                                   notes: str = "") -> OutcomeRecord:
        """Complete an outcome measurement and assess results."""
        if outcome_id not in self.pending_outcomes:
            raise ValueError(f"Outcome ID {outcome_id} not found in pending outcomes")
        
        outcome = self.pending_outcomes[outcome_id]
        outcome.post_intervention_metrics = post_metrics
        outcome.performance_deltas = post_metrics.compare_to(outcome.baseline_metrics)
        outcome.measurement_duration = time.time() - outcome.timestamp
        outcome.sample_size = sample_size
        outcome.context_factors = context_factors or {}
        outcome.notes = notes
        
        # Assess the outcome
        outcome.status, outcome.success_score, outcome.confidence = self._assess_outcome(outcome)
        
        # Move to history
        self.outcome_history.append(outcome)
        del self.pending_outcomes[outcome_id]
        
        # Update analysis caches
        self._update_caches(outcome)
        
        # Persist to disk
        self._save_outcome(outcome)
        
        self.logger.info(f"Completed outcome measurement {outcome_id}: {outcome.status.value} "
                        f"(score: {outcome.success_score:.3f}, confidence: {outcome.confidence:.3f})")
        
        return outcome
    
    def _assess_outcome(self, outcome: OutcomeRecord) -> Tuple[OutcomeStatus, float, float]:
        """Assess whether an outcome was successful."""
        deltas = outcome.performance_deltas
        
        # Key performance indicators
        win_rate_improved = deltas['win_rate_delta'] > 0.01
        score_improved = deltas['score_delta'] > 0.5
        efficiency_improved = deltas['learning_efficiency_delta'] > 0.02
        overall_improved = deltas['total_performance_delta'] > 0.01
        
        # Count positive and negative changes
        positive_changes = sum(1 for delta in deltas.values() if delta > 0.001)
        negative_changes = sum(1 for delta in deltas.values() if delta < -0.001)
        total_changes = positive_changes + negative_changes
        
        # Calculate success score
        success_score = 0.0
        if overall_improved:
            success_score += 0.4
        if win_rate_improved:
            success_score += 0.3
        if score_improved:
            success_score += 0.2
        if efficiency_improved:
            success_score += 0.1
        
        # Adjust based on consistency
        if total_changes > 0:
            consistency_bonus = (positive_changes / total_changes - 0.5) * 0.2
            success_score = max(0.0, min(1.0, success_score + consistency_bonus))
        
        # Determine status
        if success_score >= 0.7:
            status = OutcomeStatus.SUCCESS
        elif success_score >= 0.4:
            status = OutcomeStatus.PARTIAL
        elif success_score <= 0.2:
            status = OutcomeStatus.FAILURE
        else:
            status = OutcomeStatus.INCONCLUSIVE
        
        # Calculate confidence based on sample size and measurement duration
        confidence = min(1.0, 
                        (outcome.sample_size / 50.0) * 0.5 +
                        min(outcome.measurement_duration / 300.0, 1.0) * 0.3 +
                        (abs(deltas['total_performance_delta']) * 10) * 0.2)
        
        return status, success_score, confidence
    
    def _update_caches(self, outcome: OutcomeRecord):
        """Update analysis caches with new outcome."""
        intervention_type = outcome.intervention_type
        
        # Update success rates
        self.success_rates[intervention_type].append(outcome.success_score)
        
        # Update performance trends
        self.performance_trends[intervention_type].append({
            'timestamp': outcome.timestamp,
            'performance_delta': outcome.performance_deltas['total_performance_delta'],
            'win_rate_delta': outcome.performance_deltas['win_rate_delta'],
            'success_score': outcome.success_score
        })
    
    def get_intervention_effectiveness(self, intervention_type: str) -> Dict[str, float]:
        """Get effectiveness statistics for a specific intervention type."""
        if intervention_type not in self.success_rates:
            return {'success_rate': 0.0, 'average_score': 0.0, 'confidence': 0.0}
        
        scores = list(self.success_rates[intervention_type])
        if not scores:
            return {'success_rate': 0.0, 'average_score': 0.0, 'confidence': 0.0}
        
        successful = sum(1 for score in scores if score >= 0.4)
        success_rate = successful / len(scores)
        average_score = statistics.mean(scores)
        confidence = min(len(scores) / 20.0, 1.0)  # More data = higher confidence
        
        return {
            'success_rate': success_rate,
            'average_score': average_score,
            'confidence': confidence,
            'sample_count': len(scores)
        }
    
    def get_performance_trends(self, intervention_type: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent performance trends for an intervention type."""
        if intervention_type not in self.performance_trends:
            return []
        
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_trends = [
            trend for trend in self.performance_trends[intervention_type]
            if trend['timestamp'] > cutoff_time
        ]
        
        return list(recent_trends)
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Generate insights from accumulated outcome data."""
        insights = {
            'most_effective_interventions': [],
            'least_effective_interventions': [],
            'performance_correlations': {},
            'recommendations': []
        }
        
        # Analyze effectiveness by intervention type
        effectiveness_data = []
        for intervention_type in self.success_rates.keys():
            stats = self.get_intervention_effectiveness(intervention_type)
            if stats['sample_count'] >= 3:  # Need minimum sample
                effectiveness_data.append((intervention_type, stats))
        
        # Sort by effectiveness
        effectiveness_data.sort(key=lambda x: x[1]['average_score'], reverse=True)
        
        insights['most_effective_interventions'] = effectiveness_data[:3]
        insights['least_effective_interventions'] = effectiveness_data[-3:]
        
        # Generate recommendations
        if effectiveness_data:
            best_intervention, best_stats = effectiveness_data[0]
            if best_stats['success_rate'] > 0.6:
                insights['recommendations'].append(
                    f"Continue using {best_intervention}: {best_stats['success_rate']:.1%} success rate"
                )
            
            if len(effectiveness_data) > 1:
                worst_intervention, worst_stats = effectiveness_data[-1]
                if worst_stats['success_rate'] < 0.3:
                    insights['recommendations'].append(
                        f"Reduce use of {worst_intervention}: only {worst_stats['success_rate']:.1%} success rate"
                    )
        
        return insights
    
    def _load_existing_data(self):
        """Load existing outcome data from disk."""
        outcome_file = self.data_dir / "outcome_history.jsonl"
        if outcome_file.exists():
            try:
                with open(outcome_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            outcome = self._dict_to_outcome_record(data)
                            self.outcome_history.append(outcome)
                            self._update_caches(outcome)
                
                self.logger.info(f"Loaded {len(self.outcome_history)} outcome records from disk")
            except Exception as e:
                self.logger.error(f"Failed to load outcome history: {e}")
    
    def _save_outcome(self, outcome: OutcomeRecord):
        """Save an outcome record to disk."""
        outcome_file = self.data_dir / "outcome_history.jsonl"
        try:
            with open(outcome_file, 'a') as f:
                f.write(json.dumps(asdict(outcome), default=str) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save outcome record: {e}")
    
    def _dict_to_outcome_record(self, data: Dict[str, Any]) -> OutcomeRecord:
        """Convert dictionary back to OutcomeRecord."""
        # Handle enum conversions
        data['outcome_type'] = OutcomeType(data['outcome_type'])
        data['status'] = OutcomeStatus(data['status'])
        
        # Handle nested dataclass
        baseline_data = data['baseline_metrics']
        data['baseline_metrics'] = PerformanceMetrics(**baseline_data)
        
        post_data = data['post_intervention_metrics']
        data['post_intervention_metrics'] = PerformanceMetrics(**post_data)
        
        return OutcomeRecord(**data)
