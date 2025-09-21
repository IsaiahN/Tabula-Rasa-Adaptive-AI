"""
Adaptive Architecture

Advanced adaptive architecture system for dynamically adjusting
the system architecture based on runtime conditions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from ...training.interfaces import ComponentInterface
from ...training.caching import CacheManager, CacheConfig
from ...training.monitoring import PerformanceMonitor


class ArchitecturePattern(Enum):
    """Available architecture patterns."""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    LAYERED = "layered"
    EVENT_DRIVEN = "event_driven"
    PIPE_AND_FILTER = "pipe_and_filter"
    CLIENT_SERVER = "client_server"
    PEER_TO_PEER = "peer_to_peer"
    HYBRID = "hybrid"


@dataclass
class AdaptationRule:
    """Adaptation rule data structure."""
    rule_id: str
    condition: str
    action: str
    priority: int
    enabled: bool
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class AdaptationResult:
    """Adaptation result data structure."""
    adaptation_id: str
    rule_triggered: str
    action_taken: str
    success: bool
    timestamp: datetime
    metadata: Dict[str, Any]


class AdaptiveArchitecture(ComponentInterface):
    """
    Advanced adaptive architecture system for dynamically adjusting
    the system architecture based on runtime conditions.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        """Initialize the adaptive architecture system."""
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Adaptation state
        self.adaptation_rules: List[AdaptationRule] = []
        self.adaptation_results: List[AdaptationResult] = []
        self.current_architecture: ArchitecturePattern = ArchitecturePattern.MONOLITHIC
        
        # Performance tracking
        self.adaptation_times: List[float] = []
        self.rule_evaluation_times: List[float] = []
        
        # Adaptation components
        self.condition_evaluators: Dict[str, Callable] = {}
        self.action_executors: Dict[str, Callable] = {}
        self.architecture_transitions: Dict[Tuple[ArchitecturePattern, ArchitecturePattern], Callable] = {}
        
        # System state
        self.system_metrics: Dict[str, float] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        self._initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize the adaptive architecture system."""
        try:
            self.cache.initialize()
            self.performance_monitor.start_monitoring()
            
            # Initialize adaptation components
            self._initialize_condition_evaluators()
            self._initialize_action_executors()
            self._initialize_architecture_transitions()
            
            # Load adaptation rules
            self._load_adaptation_rules()
            
            self._initialized = True
            self.logger.info("Adaptive architecture system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive architecture: {e}")
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return {
            'name': 'AdaptiveArchitecture',
            'status': 'running' if self._initialized else 'stopped',
            'last_updated': datetime.now(),
            'metadata': {
                'adaptation_rules_count': len(self.adaptation_rules),
                'adaptation_results_count': len(self.adaptation_results),
                'current_architecture': self.current_architecture.value,
                'condition_evaluators_count': len(self.condition_evaluators),
                'action_executors_count': len(self.action_executors),
                'architecture_transitions_count': len(self.architecture_transitions),
                'average_adaptation_time': np.mean(self.adaptation_times) if self.adaptation_times else 0.0
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.performance_monitor.stop_monitoring()
            self.cache.clear()
            self._initialized = False
            self.logger.info("Adaptive architecture system cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self._initialized and self.cache.is_healthy()
    
    def evaluate_adaptation_rules(self) -> List[AdaptationResult]:
        """Evaluate all adaptation rules and execute actions."""
        try:
            start_time = datetime.now()
            
            # Get current system metrics
            self.system_metrics = self._get_current_system_metrics()
            
            # Evaluate rules
            triggered_rules = []
            for rule in self.adaptation_rules:
                if rule.enabled:
                    try:
                        if self._evaluate_condition(rule.condition):
                            triggered_rules.append(rule)
                    except Exception as e:
                        self.logger.warning(f"Error evaluating rule {rule.rule_id}: {e}")
                        continue
            
            # Sort by priority
            triggered_rules.sort(key=lambda r: r.priority, reverse=True)
            
            # Execute actions
            results = []
            for rule in triggered_rules:
                try:
                    result = self._execute_action(rule)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error executing action for rule {rule.rule_id}: {e}")
                    continue
            
            # Update performance metrics
            adaptation_time = (datetime.now() - start_time).total_seconds()
            self.adaptation_times.append(adaptation_time)
            
            # Store results
            self.adaptation_results.extend(results)
            
            # Cache results
            cache_key = f"adaptation_results_{datetime.now().timestamp()}"
            self.cache.set(cache_key, results, ttl=3600)
            
            self.logger.info(f"Evaluated {len(triggered_rules)} rules in {adaptation_time:.3f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating adaptation rules: {e}")
            return []
    
    def add_adaptation_rule(self, rule: AdaptationRule) -> None:
        """Add a new adaptation rule."""
        try:
            # Validate rule
            if self._validate_adaptation_rule(rule):
                self.adaptation_rules.append(rule)
                self._save_adaptation_rules()
                self.logger.info(f"Added adaptation rule: {rule.rule_id}")
            else:
                self.logger.warning("Invalid adaptation rule rejected")
                
        except Exception as e:
            self.logger.error(f"Error adding adaptation rule: {e}")
    
    def remove_adaptation_rule(self, rule_id: str) -> bool:
        """Remove an adaptation rule."""
        try:
            original_count = len(self.adaptation_rules)
            self.adaptation_rules = [r for r in self.adaptation_rules if r.rule_id != rule_id]
            
            if len(self.adaptation_rules) < original_count:
                self._save_adaptation_rules()
                self.logger.info(f"Removed adaptation rule: {rule_id}")
                return True
            else:
                self.logger.warning(f"Adaptation rule not found: {rule_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing adaptation rule: {e}")
            return False
    
    def get_adaptation_rules(self) -> List[AdaptationRule]:
        """Get all adaptation rules."""
        return self.adaptation_rules.copy()
    
    def get_adaptation_results(self, rule_id: Optional[str] = None) -> List[AdaptationResult]:
        """Get adaptation results."""
        try:
            if rule_id:
                return [r for r in self.adaptation_results if r.rule_triggered == rule_id]
            return self.adaptation_results.copy()
        except Exception as e:
            self.logger.error(f"Error getting adaptation results: {e}")
            return []
    
    def get_current_architecture(self) -> ArchitecturePattern:
        """Get current architecture pattern."""
        return self.current_architecture
    
    def set_architecture_pattern(self, pattern: ArchitecturePattern) -> bool:
        """Set architecture pattern."""
        try:
            if pattern != self.current_architecture:
                # Check if transition is possible
                transition_key = (self.current_architecture, pattern)
                if transition_key in self.architecture_transitions:
                    # Execute transition
                    transition_func = self.architecture_transitions[transition_key]
                    success = transition_func()
                    
                    if success:
                        self.current_architecture = pattern
                        self.logger.info(f"Architecture changed to {pattern.value}")
                        return True
                    else:
                        self.logger.warning(f"Failed to transition to {pattern.value}")
                        return False
                else:
                    self.logger.warning(f"No transition available from {self.current_architecture.value} to {pattern.value}")
                    return False
            else:
                self.logger.info(f"Architecture already set to {pattern.value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting architecture pattern: {e}")
            return False
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        try:
            if not self.adaptation_results:
                return {'error': 'No adaptations performed yet'}
            
            # Calculate statistics
            total_adaptations = len(self.adaptation_results)
            successful_adaptations = len([r for r in self.adaptation_results if r.success])
            
            # Calculate statistics by rule
            rule_stats = {}
            for rule in self.adaptation_rules:
                rule_results = [r for r in self.adaptation_results if r.rule_triggered == rule.rule_id]
                if rule_results:
                    rule_successes = len([r for r in rule_results if r.success])
                    rule_stats[rule.rule_id] = {
                        'count': len(rule_results),
                        'success_count': rule_successes,
                        'success_rate': rule_successes / len(rule_results)
                    }
            
            return {
                'total_adaptations': total_adaptations,
                'successful_adaptations': successful_adaptations,
                'success_rate': successful_adaptations / total_adaptations,
                'current_architecture': self.current_architecture.value,
                'adaptation_rules_count': len(self.adaptation_rules),
                'rule_statistics': rule_stats,
                'average_adaptation_time': np.mean(self.adaptation_times) if self.adaptation_times else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting adaptation statistics: {e}")
            return {'error': str(e)}
    
    def _initialize_condition_evaluators(self) -> None:
        """Initialize condition evaluators."""
        try:
            # Add common condition evaluators
            self.condition_evaluators['cpu_usage_high'] = self._evaluate_cpu_usage_high
            self.condition_evaluators['memory_usage_high'] = self._evaluate_memory_usage_high
            self.condition_evaluators['response_time_slow'] = self._evaluate_response_time_slow
            self.condition_evaluators['error_rate_high'] = self._evaluate_error_rate_high
            self.condition_evaluators['load_high'] = self._evaluate_load_high
            self.condition_evaluators['scalability_needed'] = self._evaluate_scalability_needed
            
            self.logger.info(f"Initialized {len(self.condition_evaluators)} condition evaluators")
            
        except Exception as e:
            self.logger.error(f"Error initializing condition evaluators: {e}")
    
    def _initialize_action_executors(self) -> None:
        """Initialize action executors."""
        try:
            # Add common action executors
            self.action_executors['scale_horizontal'] = self._execute_scale_horizontal
            self.action_executors['scale_vertical'] = self._execute_scale_vertical
            self.action_executors['enable_caching'] = self._execute_enable_caching
            self.action_executors['enable_parallelism'] = self._execute_enable_parallelism
            self.action_executors['switch_to_microservices'] = self._execute_switch_to_microservices
            self.action_executors['switch_to_monolithic'] = self._execute_switch_to_monolithic
            self.action_executors['optimize_algorithms'] = self._execute_optimize_algorithms
            
            self.logger.info(f"Initialized {len(self.action_executors)} action executors")
            
        except Exception as e:
            self.logger.error(f"Error initializing action executors: {e}")
    
    def _initialize_architecture_transitions(self) -> None:
        """Initialize architecture transitions."""
        try:
            # Add architecture transitions
            self.architecture_transitions[(ArchitecturePattern.MONOLITHIC, ArchitecturePattern.MICROSERVICES)] = self._transition_to_microservices
            self.architecture_transitions[(ArchitecturePattern.MICROSERVICES, ArchitecturePattern.MONOLITHIC)] = self._transition_to_monolithic
            self.architecture_transitions[(ArchitecturePattern.MONOLITHIC, ArchitecturePattern.LAYERED)] = self._transition_to_layered
            self.architecture_transitions[(ArchitecturePattern.LAYERED, ArchitecturePattern.MONOLITHIC)] = self._transition_to_monolithic
            self.architecture_transitions[(ArchitecturePattern.MONOLITHIC, ArchitecturePattern.EVENT_DRIVEN)] = self._transition_to_event_driven
            self.architecture_transitions[(ArchitecturePattern.EVENT_DRIVEN, ArchitecturePattern.MONOLITHIC)] = self._transition_to_monolithic
            
            self.logger.info(f"Initialized {len(self.architecture_transitions)} architecture transitions")
            
        except Exception as e:
            self.logger.error(f"Error initializing architecture transitions: {e}")
    
    def _load_adaptation_rules(self) -> None:
        """Load adaptation rules from cache or create default."""
        try:
            # Try to load from cache
            cache_key = "adaptation_rules"
            cached_rules = self.cache.get(cache_key)
            
            if cached_rules:
                self.adaptation_rules = cached_rules
            else:
                # Create default rules
                self._create_default_rules()
                self._save_adaptation_rules()
            
            self.logger.info("Adaptation rules loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading adaptation rules: {e}")
            self._create_default_rules()
    
    def _save_adaptation_rules(self) -> None:
        """Save adaptation rules to cache."""
        try:
            cache_key = "adaptation_rules"
            self.cache.set(cache_key, self.adaptation_rules, ttl=3600)
        except Exception as e:
            self.logger.error(f"Error saving adaptation rules: {e}")
    
    def _create_default_rules(self) -> None:
        """Create default adaptation rules."""
        try:
            default_rules = [
                AdaptationRule(
                    rule_id="cpu_high_scale_horizontal",
                    condition="cpu_usage_high",
                    action="scale_horizontal",
                    priority=1,
                    enabled=True,
                    created_at=datetime.now(),
                    metadata={'threshold': 0.8}
                ),
                AdaptationRule(
                    rule_id="memory_high_scale_vertical",
                    condition="memory_usage_high",
                    action="scale_vertical",
                    priority=2,
                    enabled=True,
                    created_at=datetime.now(),
                    metadata={'threshold': 0.9}
                ),
                AdaptationRule(
                    rule_id="response_slow_enable_caching",
                    condition="response_time_slow",
                    action="enable_caching",
                    priority=3,
                    enabled=True,
                    created_at=datetime.now(),
                    metadata={'threshold': 2.0}
                ),
                AdaptationRule(
                    rule_id="load_high_switch_microservices",
                    condition="load_high",
                    action="switch_to_microservices",
                    priority=4,
                    enabled=True,
                    created_at=datetime.now(),
                    metadata={'threshold': 0.8}
                )
            ]
            
            self.adaptation_rules.extend(default_rules)
            
        except Exception as e:
            self.logger.error(f"Error creating default rules: {e}")
    
    def _get_current_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            # This is a simplified implementation
            # In a real system, you would collect actual metrics
            return {
                'cpu_usage': np.random.uniform(0.1, 0.9),
                'memory_usage': np.random.uniform(0.1, 0.9),
                'response_time': np.random.uniform(0.1, 5.0),
                'error_rate': np.random.uniform(0, 0.1),
                'load': np.random.uniform(0.1, 4.0),
                'throughput': np.random.uniform(100, 1000)
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition."""
        try:
            start_time = datetime.now()
            
            # Check if condition evaluator exists
            if condition in self.condition_evaluators:
                result = self.condition_evaluators[condition]()
            else:
                # Try to parse as simple expression
                result = self._evaluate_simple_condition(condition)
            
            # Update performance metrics
            evaluation_time = (datetime.now() - start_time).total_seconds()
            self.rule_evaluation_times.append(evaluation_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition}: {e}")
            return False
    
    def _evaluate_simple_condition(self, condition: str) -> bool:
        """Evaluate simple condition expressions."""
        try:
            # Simple condition parsing
            if '>' in condition:
                parts = condition.split('>')
                if len(parts) == 2:
                    metric = parts[0].strip()
                    threshold = float(parts[1].strip())
                    return self.system_metrics.get(metric, 0) > threshold
            elif '<' in condition:
                parts = condition.split('<')
                if len(parts) == 2:
                    metric = parts[0].strip()
                    threshold = float(parts[1].strip())
                    return self.system_metrics.get(metric, 0) < threshold
            elif '==' in condition:
                parts = condition.split('==')
                if len(parts) == 2:
                    metric = parts[0].strip()
                    value = float(parts[1].strip())
                    return self.system_metrics.get(metric, 0) == value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating simple condition {condition}: {e}")
            return False
    
    def _execute_action(self, rule: AdaptationRule) -> AdaptationResult:
        """Execute an action for a rule."""
        try:
            start_time = datetime.now()
            
            # Check if action executor exists
            if rule.action in self.action_executors:
                success = self.action_executors[rule.action]()
            else:
                success = False
                self.logger.warning(f"Unknown action: {rule.action}")
            
            # Create result
            result = AdaptationResult(
                adaptation_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_triggered=rule.rule_id,
                action_taken=rule.action,
                success=success,
                timestamp=datetime.now(),
                metadata={
                    'rule_priority': rule.priority,
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'system_metrics': self.system_metrics.copy()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action {rule.action}: {e}")
            return AdaptationResult(
                adaptation_id=f"adapt_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rule_triggered=rule.rule_id,
                action_taken=rule.action,
                success=False,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _validate_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """Validate adaptation rule."""
        try:
            # Check required fields
            if not rule.rule_id or not rule.condition or not rule.action:
                return False
            
            # Check if condition evaluator exists or is parseable
            if rule.condition not in self.condition_evaluators:
                try:
                    self._evaluate_simple_condition(rule.condition)
                except:
                    return False
            
            # Check if action executor exists
            if rule.action not in self.action_executors:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating adaptation rule: {e}")
            return False
    
    # Condition evaluators
    def _evaluate_cpu_usage_high(self) -> bool:
        """Evaluate if CPU usage is high."""
        try:
            return self.system_metrics.get('cpu_usage', 0) > 0.8
        except Exception as e:
            self.logger.error(f"Error evaluating CPU usage: {e}")
            return False
    
    def _evaluate_memory_usage_high(self) -> bool:
        """Evaluate if memory usage is high."""
        try:
            return self.system_metrics.get('memory_usage', 0) > 0.9
        except Exception as e:
            self.logger.error(f"Error evaluating memory usage: {e}")
            return False
    
    def _evaluate_response_time_slow(self) -> bool:
        """Evaluate if response time is slow."""
        try:
            return self.system_metrics.get('response_time', 0) > 2.0
        except Exception as e:
            self.logger.error(f"Error evaluating response time: {e}")
            return False
    
    def _evaluate_error_rate_high(self) -> bool:
        """Evaluate if error rate is high."""
        try:
            return self.system_metrics.get('error_rate', 0) > 0.05
        except Exception as e:
            self.logger.error(f"Error evaluating error rate: {e}")
            return False
    
    def _evaluate_load_high(self) -> bool:
        """Evaluate if system load is high."""
        try:
            return self.system_metrics.get('load', 0) > 0.8
        except Exception as e:
            self.logger.error(f"Error evaluating load: {e}")
            return False
    
    def _evaluate_scalability_needed(self) -> bool:
        """Evaluate if scalability is needed."""
        try:
            cpu_usage = self.system_metrics.get('cpu_usage', 0)
            memory_usage = self.system_metrics.get('memory_usage', 0)
            load = self.system_metrics.get('load', 0)
            
            return (cpu_usage > 0.7 or memory_usage > 0.8 or load > 0.7)
        except Exception as e:
            self.logger.error(f"Error evaluating scalability need: {e}")
            return False
    
    # Action executors
    def _execute_scale_horizontal(self) -> bool:
        """Execute horizontal scaling."""
        try:
            self.logger.info("Executing horizontal scaling")
            # In a real system, this would actually scale horizontally
            return True
        except Exception as e:
            self.logger.error(f"Error executing horizontal scaling: {e}")
            return False
    
    def _execute_scale_vertical(self) -> bool:
        """Execute vertical scaling."""
        try:
            self.logger.info("Executing vertical scaling")
            # In a real system, this would actually scale vertically
            return True
        except Exception as e:
            self.logger.error(f"Error executing vertical scaling: {e}")
            return False
    
    def _execute_enable_caching(self) -> bool:
        """Execute enable caching."""
        try:
            self.logger.info("Executing enable caching")
            # In a real system, this would actually enable caching
            return True
        except Exception as e:
            self.logger.error(f"Error executing enable caching: {e}")
            return False
    
    def _execute_enable_parallelism(self) -> bool:
        """Execute enable parallelism."""
        try:
            self.logger.info("Executing enable parallelism")
            # In a real system, this would actually enable parallelism
            return True
        except Exception as e:
            self.logger.error(f"Error executing enable parallelism: {e}")
            return False
    
    def _execute_switch_to_microservices(self) -> bool:
        """Execute switch to microservices."""
        try:
            self.logger.info("Executing switch to microservices")
            return self.set_architecture_pattern(ArchitecturePattern.MICROSERVICES)
        except Exception as e:
            self.logger.error(f"Error executing switch to microservices: {e}")
            return False
    
    def _execute_switch_to_monolithic(self) -> bool:
        """Execute switch to monolithic."""
        try:
            self.logger.info("Executing switch to monolithic")
            return self.set_architecture_pattern(ArchitecturePattern.MONOLITHIC)
        except Exception as e:
            self.logger.error(f"Error executing switch to monolithic: {e}")
            return False
    
    def _execute_optimize_algorithms(self) -> bool:
        """Execute optimize algorithms."""
        try:
            self.logger.info("Executing optimize algorithms")
            # In a real system, this would actually optimize algorithms
            return True
        except Exception as e:
            self.logger.error(f"Error executing optimize algorithms: {e}")
            return False
    
    # Architecture transitions
    def _transition_to_microservices(self) -> bool:
        """Transition to microservices architecture."""
        try:
            self.logger.info("Transitioning to microservices architecture")
            # In a real system, this would actually transition the architecture
            return True
        except Exception as e:
            self.logger.error(f"Error transitioning to microservices: {e}")
            return False
    
    def _transition_to_monolithic(self) -> bool:
        """Transition to monolithic architecture."""
        try:
            self.logger.info("Transitioning to monolithic architecture")
            # In a real system, this would actually transition the architecture
            return True
        except Exception as e:
            self.logger.error(f"Error transitioning to monolithic: {e}")
            return False
    
    def _transition_to_layered(self) -> bool:
        """Transition to layered architecture."""
        try:
            self.logger.info("Transitioning to layered architecture")
            # In a real system, this would actually transition the architecture
            return True
        except Exception as e:
            self.logger.error(f"Error transitioning to layered: {e}")
            return False
    
    def _transition_to_event_driven(self) -> bool:
        """Transition to event-driven architecture."""
        try:
            self.logger.info("Transitioning to event-driven architecture")
            # In a real system, this would actually transition the architecture
            return True
        except Exception as e:
            self.logger.error(f"Error transitioning to event-driven: {e}")
            return False
