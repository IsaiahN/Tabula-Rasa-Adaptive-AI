#!/usr/bin/env python3
"""
Governor Session Reporter - Phase 1 of Symbiosis Protocol

Implements structured session reporting from the Governor to the Architect,
enabling the recursive self-improvement loop described in the symbiosis document.

This module handles:
1. Session data collection and tracking
2. Performance metrics aggregation
3. Anomaly detection and logging
4. Structured report generation for Architect analysis
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class SessionStatus(Enum):
    """Status of a training session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    PLATEAUED = "plateaued"

class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ENERGY_CRISIS = "energy_crisis"
    MEMORY_PRESSURE = "memory_pressure"
    LEARNING_STAGNATION = "learning_stagnation"
    COORDINATE_STUCK = "coordinate_stuck"
    ACTION_FAILURE = "action_failure"
    SYSTEM_ERROR = "system_error"

@dataclass
class SessionObjective:
    """Objective for a training session."""
    objective_id: str
    description: str
    target_metric: str
    target_value: float
    priority: float
    achieved: bool = False
    actual_value: Optional[float] = None

@dataclass
class DecisionPoint:
    """A key decision point during the session."""
    timestamp: float
    decision_type: str
    decision_data: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    confidence: float
    energy_cost: float
    learning_gain: float

@dataclass
class PerformanceMetrics:
    """Performance metrics for the session."""
    energy_consumption: float
    energy_efficiency: float
    learning_progress: float
    prediction_accuracy: float
    memory_usage: float
    memory_efficiency: float
    action_success_rate: float
    coordinate_accuracy: float
    curiosity_level: float
    boredom_level: float
    system_health: float
    cognitive_load: float

@dataclass
class Anomaly:
    """An anomaly detected during the session."""
    anomaly_id: str
    anomaly_type: AnomalyType
    description: str
    severity: float  # 0.0 to 1.0
    timestamp: float
    context: Dict[str, Any]
    resolution_attempted: bool = False
    resolution_successful: bool = False

@dataclass
class GovernorSessionReport:
    """Complete session report from Governor to Architect."""
    session_id: str
    session_status: SessionStatus
    start_time: float
    end_time: float
    duration: float
    
    # Session objectives and outcomes
    objectives: List[SessionObjective]
    outcomes: Dict[str, Any]  # success/failure, score, etc.
    
    # Decision tracking
    decision_log: List[DecisionPoint]
    key_decisions: List[DecisionPoint]  # Most important decisions
    
    # Performance metrics
    performance_metrics: PerformanceMetrics
    system_health_history: List[Dict[str, float]]
    
    # Anomalies and issues
    anomalies: List[Anomaly]
    unresolved_challenges: List[str]
    
    # Learning insights
    learning_insights: List[str]
    breakthrough_moments: List[Dict[str, Any]]
    
    # Resource utilization
    resource_utilization: Dict[str, float]
    cognitive_costs: Dict[str, float]
    
    # Cross-session context
    previous_session_id: Optional[str] = None
    next_session_recommendations: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'session_status': self.session_status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'objectives': [asdict(obj) for obj in self.objectives],
            'outcomes': self.outcomes,
            'decision_log': [asdict(decision) for decision in self.decision_log],
            'key_decisions': [asdict(decision) for decision in self.key_decisions],
            'performance_metrics': asdict(self.performance_metrics),
            'system_health_history': self.system_health_history,
            'anomalies': [asdict(anomaly) for anomaly in self.anomalies],
            'unresolved_challenges': self.unresolved_challenges,
            'learning_insights': self.learning_insights,
            'breakthrough_moments': self.breakthrough_moments,
            'resource_utilization': self.resource_utilization,
            'cognitive_costs': self.cognitive_costs,
            'previous_session_id': self.previous_session_id,
            'next_session_recommendations': self.next_session_recommendations or []
        }

class AnomalyDetector:
    """Detects anomalies during session execution."""
    
    def __init__(self):
        self.anomaly_thresholds = {
            AnomalyType.PERFORMANCE_DEGRADATION: 0.3,
            AnomalyType.ENERGY_CRISIS: 0.2,
            AnomalyType.MEMORY_PRESSURE: 0.8,
            AnomalyType.LEARNING_STAGNATION: 0.1,
            AnomalyType.COORDINATE_STUCK: 0.9,
            AnomalyType.ACTION_FAILURE: 0.7,
            AnomalyType.SYSTEM_ERROR: 0.5
        }
        self.anomaly_history = deque(maxlen=1000)
    
    def detect_anomalies(self, 
                        current_metrics: PerformanceMetrics,
                        decision_history: List[DecisionPoint],
                        system_state: Dict[str, Any]) -> List[Anomaly]:
        """Detect anomalies based on current metrics and state."""
        anomalies = []
        
        # Performance degradation
        if current_metrics.learning_progress < self.anomaly_thresholds[AnomalyType.PERFORMANCE_DEGRADATION]:
            anomalies.append(Anomaly(
                anomaly_id=f"perf_degradation_{int(time.time())}",
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                description=f"Learning progress below threshold: {current_metrics.learning_progress:.3f}",
                severity=1.0 - current_metrics.learning_progress,
                timestamp=time.time(),
                context={'learning_progress': current_metrics.learning_progress}
            ))
        
        # Energy crisis
        if current_metrics.energy_efficiency < self.anomaly_thresholds[AnomalyType.ENERGY_CRISIS]:
            anomalies.append(Anomaly(
                anomaly_id=f"energy_crisis_{int(time.time())}",
                anomaly_type=AnomalyType.ENERGY_CRISIS,
                description=f"Energy efficiency critically low: {current_metrics.energy_efficiency:.3f}",
                severity=1.0 - current_metrics.energy_efficiency,
                timestamp=time.time(),
                context={'energy_efficiency': current_metrics.energy_efficiency}
            ))
        
        # Memory pressure
        if current_metrics.memory_usage > self.anomaly_thresholds[AnomalyType.MEMORY_PRESSURE]:
            anomalies.append(Anomaly(
                anomaly_id=f"memory_pressure_{int(time.time())}",
                anomaly_type=AnomalyType.MEMORY_PRESSURE,
                description=f"Memory usage critically high: {current_metrics.memory_usage:.3f}",
                severity=current_metrics.memory_usage,
                timestamp=time.time(),
                context={'memory_usage': current_metrics.memory_usage}
            ))
        
        # Learning stagnation
        if current_metrics.learning_progress < self.anomaly_thresholds[AnomalyType.LEARNING_STAGNATION]:
            anomalies.append(Anomaly(
                anomaly_id=f"learning_stagnation_{int(time.time())}",
                anomaly_type=AnomalyType.LEARNING_STAGNATION,
                description=f"Learning progress stagnated: {current_metrics.learning_progress:.3f}",
                severity=1.0 - current_metrics.learning_progress,
                timestamp=time.time(),
                context={'learning_progress': current_metrics.learning_progress}
            ))
        
        # Coordinate stuck (from existing system)
        if 'stuck_coordinates' in system_state:
            stuck_count = system_state['stuck_coordinates']
            if stuck_count > 1000:  # Threshold for stuck coordinates
                anomalies.append(Anomaly(
                    anomaly_id=f"coordinate_stuck_{int(time.time())}",
                    anomaly_type=AnomalyType.COORDINATE_STUCK,
                    description=f"Coordinate stuck for {stuck_count} attempts",
                    severity=min(stuck_count / 10000, 1.0),
                    timestamp=time.time(),
                    context={'stuck_count': stuck_count}
                ))
        
        return anomalies

class GovernorSessionReporter:
    """Main class for tracking and reporting Governor sessions."""
    
    def __init__(self, governor=None):
        self.governor = governor
        self.anomaly_detector = AnomalyDetector()
        self.session_data = {}
        self.current_session_id = None
        self.decision_counter = 0
        self.performance_history = deque(maxlen=1000)
        
    def start_session(self, 
                     session_id: str, 
                     objectives: List[Dict[str, Any]],
                     previous_session_id: Optional[str] = None) -> None:
        """Start tracking a new session."""
        self.current_session_id = session_id
        self.session_data = {
            'session_id': session_id,
            'start_time': time.time(),
            'objectives': [SessionObjective(
                objective_id=obj.get('id', f"obj_{i}"),
                description=obj.get('description', ''),
                target_metric=obj.get('target_metric', 'score'),
                target_value=obj.get('target_value', 0.8),
                priority=obj.get('priority', 1.0)
            ) for i, obj in enumerate(objectives)],
            'decisions': [],
            'key_decisions': [],
            'performance_snapshots': [],
            'anomalies': [],
            'learning_insights': [],
            'breakthrough_moments': [],
            'system_health_history': [],
            'resource_utilization': defaultdict(float),
            'cognitive_costs': defaultdict(float),
            'previous_session_id': previous_session_id
        }
        
        logger.info(f"Started session tracking for {session_id}")
    
    def log_decision(self, 
                    decision_type: str,
                    decision_data: Dict[str, Any],
                    result: Dict[str, Any],
                    success: bool,
                    confidence: float,
                    energy_cost: float = 0.0,
                    learning_gain: float = 0.0) -> None:
        """Log a key decision and its result."""
        if not self.current_session_id:
            logger.warning("No active session to log decision to")
            return
        
        decision = DecisionPoint(
            timestamp=time.time(),
            decision_type=decision_type,
            decision_data=decision_data,
            result=result,
            success=success,
            confidence=confidence,
            energy_cost=energy_cost,
            learning_gain=learning_gain
        )
        
        self.session_data['decisions'].append(decision)
        self.decision_counter += 1
        
        # Track key decisions (high impact or high confidence)
        if success and (confidence > 0.8 or abs(learning_gain) > 0.1):
            self.session_data['key_decisions'].append(decision)
        
        logger.debug(f"Logged decision {self.decision_counter}: {decision_type}")
    
    def log_performance_snapshot(self, 
                                performance_metrics: Dict[str, float],
                                system_state: Dict[str, Any]) -> None:
        """Log a performance snapshot."""
        if not self.current_session_id:
            return
        
        snapshot = {
            'timestamp': time.time(),
            'metrics': performance_metrics,
            'system_state': system_state
        }
        
        self.session_data['performance_snapshots'].append(snapshot)
        self.session_data['system_health_history'].append(performance_metrics)
        
        # Detect anomalies
        current_metrics = PerformanceMetrics(**performance_metrics)
        anomalies = self.anomaly_detector.detect_anomalies(
            current_metrics, 
            self.session_data['decisions'],
            system_state
        )
        
        for anomaly in anomalies:
            self.session_data['anomalies'].append(anomaly)
            logger.warning(f"Anomaly detected: {anomaly.anomaly_type.value} - {anomaly.description}")
    
    def log_learning_insight(self, insight: str, context: Dict[str, Any] = None) -> None:
        """Log a learning insight."""
        if not self.current_session_id:
            return
        
        self.session_data['learning_insights'].append({
            'timestamp': time.time(),
            'insight': insight,
            'context': context or {}
        })
        
        logger.info(f"Learning insight: {insight}")
    
    def log_breakthrough(self, 
                        breakthrough_type: str,
                        description: str,
                        impact: float,
                        context: Dict[str, Any] = None) -> None:
        """Log a breakthrough moment."""
        if not self.current_session_id:
            return
        
        breakthrough = {
            'timestamp': time.time(),
            'type': breakthrough_type,
            'description': description,
            'impact': impact,
            'context': context or {}
        }
        
        self.session_data['breakthrough_moments'].append(breakthrough)
        logger.info(f"Breakthrough: {breakthrough_type} - {description}")
    
    def update_objective_progress(self, 
                                objective_id: str, 
                                achieved: bool, 
                                actual_value: float = None) -> None:
        """Update progress on a session objective."""
        if not self.current_session_id:
            return
        
        for objective in self.session_data['objectives']:
            if objective.objective_id == objective_id:
                objective.achieved = achieved
                if actual_value is not None:
                    objective.actual_value = actual_value
                break
    
    def calculate_session_outcomes(self) -> Dict[str, Any]:
        """Calculate final session outcomes."""
        if not self.session_data:
            return {}
        
        objectives = self.session_data['objectives']
        decisions = self.session_data['decisions']
        anomalies = self.session_data['anomalies']
        
        # Calculate success metrics
        objectives_achieved = sum(1 for obj in objectives if obj.achieved)
        total_objectives = len(objectives)
        success_rate = objectives_achieved / total_objectives if total_objectives > 0 else 0.0
        
        # Calculate decision success rate
        successful_decisions = sum(1 for decision in decisions if decision.success)
        total_decisions = len(decisions)
        decision_success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0.0
        
        # Calculate learning progress
        learning_gains = [decision.learning_gain for decision in decisions]
        total_learning_gain = sum(learning_gains)
        avg_learning_gain = total_learning_gain / len(learning_gains) if learning_gains else 0.0
        
        # Calculate anomaly severity
        total_anomaly_severity = sum(anomaly.severity for anomaly in anomalies)
        avg_anomaly_severity = total_anomaly_severity / len(anomalies) if anomalies else 0.0
        
        return {
            'success_rate': success_rate,
            'objectives_achieved': objectives_achieved,
            'total_objectives': total_objectives,
            'decision_success_rate': decision_success_rate,
            'total_decisions': total_decisions,
            'learning_gain': total_learning_gain,
            'avg_learning_gain': avg_learning_gain,
            'anomaly_count': len(anomalies),
            'avg_anomaly_severity': avg_anomaly_severity,
            'session_quality': (success_rate + decision_success_rate + (1.0 - avg_anomaly_severity)) / 3.0
        }
    
    def generate_session_report(self, 
                              session_status: SessionStatus = SessionStatus.COMPLETED,
                              next_session_recommendations: List[str] = None) -> GovernorSessionReport:
        """Generate final session report for Architect analysis."""
        if not self.current_session_id:
            raise ValueError("No active session to generate report for")
        
        end_time = time.time()
        start_time = self.session_data['start_time']
        duration = end_time - start_time
        
        # Calculate final performance metrics
        if self.session_data['performance_snapshots']:
            latest_snapshot = self.session_data['performance_snapshots'][-1]
            performance_metrics = PerformanceMetrics(**latest_snapshot['metrics'])
        else:
            # Default metrics if no snapshots
            performance_metrics = PerformanceMetrics(
                energy_consumption=0.0,
                energy_efficiency=1.0,
                learning_progress=0.0,
                prediction_accuracy=0.0,
                memory_usage=0.0,
                memory_efficiency=1.0,
                action_success_rate=0.0,
                coordinate_accuracy=0.0,
                curiosity_level=0.0,
                boredom_level=0.0,
                system_health=1.0,
                cognitive_load=0.0
            )
        
        # Calculate outcomes
        outcomes = self.calculate_session_outcomes()
        
        # Identify unresolved challenges
        unresolved_challenges = []
        for anomaly in self.session_data['anomalies']:
            if not anomaly.resolution_successful and anomaly.severity > 0.5:
                unresolved_challenges.append(f"{anomaly.anomaly_type.value}: {anomaly.description}")
        
        # Create report
        report = GovernorSessionReport(
            session_id=self.current_session_id,
            session_status=session_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            objectives=self.session_data['objectives'],
            outcomes=outcomes,
            decision_log=self.session_data['decisions'],
            key_decisions=self.session_data.get('key_decisions', []),
            performance_metrics=performance_metrics,
            system_health_history=self.session_data['system_health_history'],
            anomalies=self.session_data['anomalies'],
            unresolved_challenges=unresolved_challenges,
            learning_insights=[insight['insight'] for insight in self.session_data['learning_insights']],
            breakthrough_moments=self.session_data['breakthrough_moments'],
            resource_utilization=dict(self.session_data['resource_utilization']),
            cognitive_costs=dict(self.session_data['cognitive_costs']),
            previous_session_id=self.session_data.get('previous_session_id'),
            next_session_recommendations=next_session_recommendations or []
        )
        
        logger.info(f"Generated session report for {self.current_session_id}: "
                   f"Success rate: {outcomes['success_rate']:.2f}, "
                   f"Decisions: {outcomes['total_decisions']}, "
                   f"Anomalies: {outcomes['anomaly_count']}")
        
        # Reset for next session
        self.current_session_id = None
        self.session_data = {}
        
        return report
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status."""
        if not self.current_session_id:
            return {'active': False}
        
        return {
            'active': True,
            'session_id': self.current_session_id,
            'duration': time.time() - self.session_data['start_time'],
            'decisions_logged': len(self.session_data['decisions']),
            'anomalies_detected': len(self.session_data['anomalies']),
            'objectives_count': len(self.session_data['objectives'])
        }
