#!/usr/bin/env python3
"""
Recursive Self-Improvement System - Phase 3 of Symbiosis Protocol

Implements the complete recursive self-improvement loop that connects
the Governor and Architect systems for continuous evolution.

This module handles:
1. Automatic triggering of improvement cycles
2. Coordination between Governor and Architect
3. Performance tracking across evolution cycles
4. Integration with existing cohesive system
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import deque

# Import existing components
try:
    from .governor_session_reporter import GovernorSessionReporter, GovernorSessionReport, SessionStatus
    from .architect_directive_system import ArchitectDirectiveSystem, EvolutionaryDirective, DirectiveStatus
    from .meta_cognitive_governor import MetaCognitiveGovernor
    from .architect import Architect
except ImportError:
    # Fallback for direct execution
    from governor_session_reporter import GovernorSessionReporter, GovernorSessionReport, SessionStatus
    from architect_directive_system import ArchitectDirectiveSystem, EvolutionaryDirective, DirectiveStatus

logger = logging.getLogger(__name__)

class ImprovementCycleStatus(Enum):
    """Status of an improvement cycle."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class TriggerType(Enum):
    """Types of triggers for improvement cycles."""
    SESSION_COMPLETION = "session_completion"
    PERFORMANCE_PLATEAU = "performance_plateau"
    CRITICAL_ANOMALY = "critical_anomaly"
    ENERGY_DEPLETION = "energy_depletion"
    LEARNING_STAGNATION = "learning_stagnation"
    MANUAL = "manual"
    SCHEDULED = "scheduled"

@dataclass
class ImprovementCycle:
    """A single improvement cycle in the recursive self-improvement loop."""
    cycle_id: str
    trigger_type: TriggerType
    start_time: float
    end_time: Optional[float] = None
    status: ImprovementCycleStatus = ImprovementCycleStatus.PENDING
    session_report: Optional[GovernorSessionReport] = None
    directives_generated: List[EvolutionaryDirective] = None
    directives_executed: List[EvolutionaryDirective] = None
    performance_improvement: float = 0.0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.directives_generated is None:
            self.directives_generated = []
        if self.directives_executed is None:
            self.directives_executed = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cycle_id': self.cycle_id,
            'trigger_type': self.trigger_type.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status.value,
            'session_report': self.session_report.to_dict() if self.session_report else None,
            'directives_generated': [d.to_dict() for d in self.directives_generated],
            'directives_executed': [d.to_dict() for d in self.directives_executed],
            'performance_improvement': self.performance_improvement,
            'error_message': self.error_message
        }

@dataclass
class SystemEvolutionMetrics:
    """Metrics tracking system evolution over time."""
    total_cycles: int = 0
    successful_cycles: int = 0
    total_directives_generated: int = 0
    total_directives_executed: int = 0
    cumulative_performance_improvement: float = 0.0
    average_cycle_duration: float = 0.0
    success_rate: float = 0.0
    directive_success_rate: float = 0.0
    performance_trend: float = 0.0
    
    def update(self, cycle: ImprovementCycle):
        """Update metrics with a completed cycle."""
        self.total_cycles += 1
        if cycle.status == ImprovementCycleStatus.COMPLETED:
            self.successful_cycles += 1
        
        self.total_directives_generated += len(cycle.directives_generated)
        self.total_directives_executed += len(cycle.directives_executed)
        self.cumulative_performance_improvement += cycle.performance_improvement
        
        if cycle.end_time:
            cycle_duration = cycle.end_time - cycle.start_time
            self.average_cycle_duration = (
                (self.average_cycle_duration * (self.total_cycles - 1) + cycle_duration) / self.total_cycles
            )
        
        self.success_rate = self.successful_cycles / self.total_cycles if self.total_cycles > 0 else 0.0
        self.directive_success_rate = (
            self.total_directives_executed / self.total_directives_generated 
            if self.total_directives_generated > 0 else 0.0
        )

class ImprovementTrigger:
    """Detects when improvement cycles should be triggered."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=50)
        self.energy_history = deque(maxlen=100)
        self.learning_history = deque(maxlen=100)
        
    def should_trigger_cycle(self, 
                           current_state: Dict[str, Any],
                           session_reporter: GovernorSessionReporter) -> Tuple[bool, TriggerType, str]:
        """Determine if an improvement cycle should be triggered."""
        
        # Check for session completion
        if session_reporter.current_session_id is None:
            return False, None, "No active session"
        
        # Check for performance plateau
        if self._detect_performance_plateau(current_state):
            return True, TriggerType.PERFORMANCE_PLATEAU, "Performance plateau detected"
        
        # Check for critical anomalies
        if self._detect_critical_anomaly(current_state):
            return True, TriggerType.CRITICAL_ANOMALY, "Critical anomaly detected"
        
        # Check for energy depletion
        if self._detect_energy_depletion(current_state):
            return True, TriggerType.ENERGY_DEPLETION, "Energy depletion detected"
        
        # Check for learning stagnation
        if self._detect_learning_stagnation(current_state):
            return True, TriggerType.LEARNING_STAGNATION, "Learning stagnation detected"
        
        # Check for session completion (if session is ending)
        if self._detect_session_completion(session_reporter):
            return True, TriggerType.SESSION_COMPLETION, "Session completion detected"
        
        return False, None, "No trigger conditions met"
    
    def _detect_performance_plateau(self, current_state: Dict[str, Any]) -> bool:
        """Detect if performance has plateaued."""
        if 'performance_metrics' not in current_state:
            return False
        
        performance = current_state['performance_metrics']
        self.performance_history.append(performance.get('session_quality', 0.0))
        
        if len(self.performance_history) < 10:
            return False
        
        # Check if performance has been stable for the last 10 measurements
        recent_performance = list(self.performance_history)[-10:]
        avg_performance = sum(recent_performance) / len(recent_performance)
        variance = sum((p - avg_performance) ** 2 for p in recent_performance) / len(recent_performance)
        
        # If variance is very low and performance is below threshold, consider it plateaued
        return variance < 0.01 and avg_performance < 0.6
    
    def _detect_critical_anomaly(self, current_state: Dict[str, Any]) -> bool:
        """Detect if there are critical anomalies."""
        if 'anomalies' not in current_state:
            return False
        
        anomalies = current_state['anomalies']
        if not anomalies:
            return False
        
        # Check for high-severity anomalies
        critical_anomalies = [a for a in anomalies if a.get('severity', 0.0) > 0.8]
        return len(critical_anomalies) > 0
    
    def _detect_energy_depletion(self, current_state: Dict[str, Any]) -> bool:
        """Detect if energy is critically low."""
        if 'performance_metrics' not in current_state:
            return False
        
        energy_efficiency = current_state['performance_metrics'].get('energy_efficiency', 1.0)
        self.energy_history.append(energy_efficiency)
        
        if len(self.energy_history) < 5:
            return False
        
        # Check if energy efficiency has been consistently low
        recent_energy = list(self.energy_history)[-5:]
        avg_energy = sum(recent_energy) / len(recent_energy)
        return avg_energy < 0.3
    
    def _detect_learning_stagnation(self, current_state: Dict[str, Any]) -> bool:
        """Detect if learning has stagnated."""
        if 'performance_metrics' not in current_state:
            return False
        
        learning_progress = current_state['performance_metrics'].get('learning_progress', 0.0)
        self.learning_history.append(learning_progress)
        
        if len(self.learning_history) < 10:
            return False
        
        # Check if learning progress has been consistently low
        recent_learning = list(self.learning_history)[-10:]
        avg_learning = sum(recent_learning) / len(recent_learning)
        return avg_learning < 0.1
    
    def _detect_session_completion(self, session_reporter: GovernorSessionReporter) -> bool:
        """Detect if the current session is completing."""
        # This would be implemented based on the specific session completion logic
        # For now, return False as sessions are managed externally
        return False

class RecursiveSelfImprovementSystem:
    """Main class for the recursive self-improvement system."""
    
    def __init__(self, 
                 governor: Optional[MetaCognitiveGovernor] = None,
                 architect: Optional[Architect] = None,
                 cohesive_system: Optional[Any] = None):
        self.governor = governor
        self.architect = architect
        self.cohesive_system = cohesive_system
        
        # Initialize subsystems
        self.session_reporter = GovernorSessionReporter(governor)
        self.directive_system = ArchitectDirectiveSystem(architect, governor)
        self.improvement_trigger = ImprovementTrigger()
        
        # Track improvement cycles
        self.improvement_cycles = []
        self.evolution_metrics = SystemEvolutionMetrics()
        self.cycle_counter = 0
        
        # Configuration
        self.max_concurrent_directives = 5
        self.cycle_timeout = 300.0  # 5 minutes
        self.enable_automatic_triggering = True
        
        logger.info("Recursive Self-Improvement System initialized")
    
    def start_session(self, 
                     session_id: str, 
                     objectives: List[Dict[str, Any]],
                     previous_session_id: Optional[str] = None) -> None:
        """Start a new training session with improvement tracking."""
        logger.info(f"Starting session {session_id} with improvement tracking")
        
        # Start session reporting
        self.session_reporter.start_session(session_id, objectives, previous_session_id)
        
        # Log session start
        logger.info(f"Session {session_id} started with {len(objectives)} objectives")
    
    def log_decision(self, 
                    decision_type: str,
                    decision_data: Dict[str, Any],
                    result: Dict[str, Any],
                    success: bool,
                    confidence: float,
                    energy_cost: float = 0.0,
                    learning_gain: float = 0.0) -> None:
        """Log a decision during the session."""
        self.session_reporter.log_decision(
            decision_type, decision_data, result, success, 
            confidence, energy_cost, learning_gain
        )
    
    def log_performance_snapshot(self, 
                                performance_metrics: Dict[str, float],
                                system_state: Dict[str, Any]) -> None:
        """Log a performance snapshot during the session."""
        self.session_reporter.log_performance_snapshot(performance_metrics, system_state)
        
        # Check for improvement cycle triggers
        if self.enable_automatic_triggering:
            should_trigger, trigger_type, reason = self.improvement_trigger.should_trigger_cycle(
                system_state, self.session_reporter
            )
            
            if should_trigger:
                logger.info(f"Improvement cycle triggered: {reason}")
                self.run_improvement_cycle(trigger_type, system_state)
    
    def run_improvement_cycle(self, 
                            trigger_type: TriggerType,
                            current_state: Dict[str, Any]) -> ImprovementCycle:
        """Run one complete improvement cycle."""
        self.cycle_counter += 1
        cycle_id = f"cycle_{self.cycle_counter}_{int(time.time())}"
        
        logger.info(f"Starting improvement cycle {cycle_id} (trigger: {trigger_type.value})")
        
        # Create cycle record
        cycle = ImprovementCycle(
            cycle_id=cycle_id,
            trigger_type=trigger_type,
            start_time=time.time(),
            status=ImprovementCycleStatus.RUNNING
        )
        
        try:
            # Phase 1: Generate session report
            if trigger_type == TriggerType.SESSION_COMPLETION:
                session_report = self.session_reporter.generate_session_report(
                    SessionStatus.COMPLETED
                )
            else:
                # Generate interim report for other triggers
                session_report = self.session_reporter.generate_session_report(
                    SessionStatus.ACTIVE
                )
            
            cycle.session_report = session_report
            logger.info(f"Generated session report for cycle {cycle_id}")
            
            # Phase 2: Analyze report and generate directives
            directives = self.directive_system.analyze_governor_report(session_report)
            cycle.directives_generated = directives
            logger.info(f"Generated {len(directives)} directives for cycle {cycle_id}")
            
            # Phase 3: Execute directives
            executed_directives = []
            for directive in directives[:self.max_concurrent_directives]:
                try:
                    result = self.directive_system.execute_directive(directive)
                    if result.success:
                        executed_directives.append(directive)
                        logger.info(f"Successfully executed directive {directive.directive_id}")
                    else:
                        logger.warning(f"Failed to execute directive {directive.directive_id}: {result.error_message}")
                except Exception as e:
                    logger.error(f"Error executing directive {directive.directive_id}: {e}")
            
            cycle.directives_executed = executed_directives
            
            # Calculate performance improvement
            cycle.performance_improvement = self._calculate_performance_improvement(
                session_report, executed_directives
            )
            
            # Update cycle status
            cycle.end_time = time.time()
            cycle.status = ImprovementCycleStatus.COMPLETED
            
            logger.info(f"Improvement cycle {cycle_id} completed successfully")
            
        except Exception as e:
            cycle.end_time = time.time()
            cycle.status = ImprovementCycleStatus.FAILED
            cycle.error_message = str(e)
            logger.error(f"Improvement cycle {cycle_id} failed: {e}")
        
        # Store cycle and update metrics
        self.improvement_cycles.append(cycle)
        self.evolution_metrics.update(cycle)
        
        return cycle
    
    def _calculate_performance_improvement(self, 
                                         session_report: GovernorSessionReport,
                                         executed_directives: List[EvolutionaryDirective]) -> float:
        """Calculate the performance improvement from executed directives."""
        if not executed_directives:
            return 0.0
        
        # Calculate improvement based on directive benefits
        total_expected_benefit = sum(directive.expected_benefit for directive in executed_directives)
        total_actual_benefit = sum(directive.expected_benefit * 0.8 for directive in executed_directives)  # Assume 80% of expected
        
        # Also consider session quality improvement
        session_quality = session_report.outcomes.get('session_quality', 0.0)
        quality_improvement = max(0.0, session_quality - 0.5)  # Improvement above baseline
        
        return total_actual_benefit + quality_improvement
    
    def end_session(self, 
                   session_status: SessionStatus = SessionStatus.COMPLETED,
                   next_session_recommendations: List[str] = None) -> Optional[GovernorSessionReport]:
        """End the current session and generate final report."""
        logger.info(f"Ending session with status: {session_status.value}")
        
        # Check if there's an active session
        if not self.session_reporter.current_session_id:
            logger.warning("No active session to end")
            return None
        
        # Generate final session report
        final_report = self.session_reporter.generate_session_report(
            session_status, next_session_recommendations
        )
        
        # Run final improvement cycle if there are unresolved issues
        if final_report.unresolved_challenges:
            logger.info("Running final improvement cycle for unresolved challenges")
            self.run_improvement_cycle(TriggerType.SESSION_COMPLETION, {})
        
        return final_report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and evolution metrics."""
        return {
            'evolution_metrics': asdict(self.evolution_metrics),
            'current_session': self.session_reporter.get_session_status(),
            'directive_status': self.directive_system.get_directive_status(),
            'directive_performance': self.directive_system.get_performance_summary(),
            'recent_cycles': [cycle.to_dict() for cycle in self.improvement_cycles[-5:]],
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score."""
        if not self.evolution_metrics.total_cycles:
            return 1.0
        
        # Factors in success rate, performance improvement, and directive effectiveness
        success_rate = self.evolution_metrics.success_rate
        performance_improvement = min(self.evolution_metrics.cumulative_performance_improvement, 1.0)
        directive_success_rate = self.evolution_metrics.directive_success_rate
        
        # Weighted average
        health_score = (
            success_rate * 0.4 +
            performance_improvement * 0.3 +
            directive_success_rate * 0.3
        )
        
        return min(max(health_score, 0.0), 1.0)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of system evolution."""
        return {
            'total_improvement_cycles': self.evolution_metrics.total_cycles,
            'successful_cycles': self.evolution_metrics.successful_cycles,
            'success_rate': self.evolution_metrics.success_rate,
            'total_directives_generated': self.evolution_metrics.total_directives_generated,
            'total_directives_executed': self.evolution_metrics.total_directives_executed,
            'directive_success_rate': self.evolution_metrics.directive_success_rate,
            'cumulative_performance_improvement': self.evolution_metrics.cumulative_performance_improvement,
            'average_cycle_duration': self.evolution_metrics.average_cycle_duration,
            'system_health': self._calculate_system_health(),
            'recent_improvements': [
                {
                    'cycle_id': cycle.cycle_id,
                    'trigger': cycle.trigger_type.value,
                    'directives_generated': len(cycle.directives_generated),
                    'directives_executed': len(cycle.directives_executed),
                    'performance_improvement': cycle.performance_improvement
                }
                for cycle in self.improvement_cycles[-10:]
            ]
        }
    
    def save_evolution_state(self, filepath: str) -> None:
        """Save the current evolution state to a file."""
        state = {
            'evolution_metrics': asdict(self.evolution_metrics),
            'improvement_cycles': [cycle.to_dict() for cycle in self.improvement_cycles],
            'cycle_counter': self.cycle_counter,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Evolution state saved to {filepath}")
    
    def load_evolution_state(self, filepath: str) -> None:
        """Load evolution state from a file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore evolution metrics
            self.evolution_metrics = SystemEvolutionMetrics(**state['evolution_metrics'])
            
            # Restore improvement cycles
            self.improvement_cycles = []
            for cycle_data in state['improvement_cycles']:
                cycle = ImprovementCycle(**cycle_data)
                self.improvement_cycles.append(cycle)
            
            # Restore cycle counter
            self.cycle_counter = state.get('cycle_counter', 0)
            
            logger.info(f"Evolution state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load evolution state from {filepath}: {e}")
    
    def enable_automatic_improvement(self, enabled: bool = True) -> None:
        """Enable or disable automatic improvement cycle triggering."""
        self.enable_automatic_triggering = enabled
        logger.info(f"Automatic improvement triggering {'enabled' if enabled else 'disabled'}")
    
    def force_improvement_cycle(self, trigger_reason: str = "manual") -> ImprovementCycle:
        """Force an improvement cycle to run."""
        logger.info(f"Forcing improvement cycle: {trigger_reason}")
        return self.run_improvement_cycle(TriggerType.MANUAL, {})
