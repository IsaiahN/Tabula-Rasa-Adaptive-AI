#!/usr/bin/env python3
"""
Space-Time Aware Governor

Enhances the MetaCognitiveGovernor with space-time awareness for dynamic
optimization of tree evaluation parameters (d, b, h) based on available
resources and problem complexity.

Based on Tree Evaluation principles for optimal resource allocation.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ResourceLevel(Enum):
    """Resource availability levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProblemComplexity(Enum):
    """Problem complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXTREME = "extreme"

@dataclass
class SpaceTimeParameters:
    """Space-time optimization parameters."""
    branching_factor: int = 5      # d - number of actions per step
    state_bits: int = 64          # b - state representation size
    max_depth: int = 10           # h - maximum simulation depth
    memory_limit_mb: float = 100.0
    timeout_seconds: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'branching_factor': self.branching_factor,
            'state_bits': self.state_bits,
            'max_depth': self.max_depth,
            'memory_limit_mb': self.memory_limit_mb,
            'timeout_seconds': self.timeout_seconds
        }

@dataclass
class ResourceProfile:
    """Current system resource profile."""
    available_memory_mb: float
    cpu_utilization: float
    active_processes: int
    problem_complexity: ProblemComplexity
    time_constraint: float  # seconds available for decision
    
    def get_resource_level(self) -> ResourceLevel:
        """Determine resource level based on available resources."""
        if self.available_memory_mb < 50 or self.cpu_utilization > 0.9:
            return ResourceLevel.CRITICAL
        elif self.available_memory_mb < 100 or self.cpu_utilization > 0.7:
            return ResourceLevel.LOW
        elif self.available_memory_mb < 200 or self.cpu_utilization > 0.5:
            return ResourceLevel.MEDIUM
        else:
            return ResourceLevel.HIGH

class TreeParameterOptimizer:
    """
    Optimizes tree evaluation parameters based on resource constraints
    and problem complexity using learned patterns and heuristics.
    """
    
    def __init__(self):
        self.optimization_history = []
        self.performance_patterns = {}
        self.parameter_effectiveness = {}
        
        # Default parameter sets for different scenarios
        self.parameter_templates = {
            (ResourceLevel.CRITICAL, ProblemComplexity.SIMPLE): SpaceTimeParameters(
                branching_factor=2, state_bits=32, max_depth=3, memory_limit_mb=25.0
            ),
            (ResourceLevel.CRITICAL, ProblemComplexity.MODERATE): SpaceTimeParameters(
                branching_factor=2, state_bits=32, max_depth=2, memory_limit_mb=25.0
            ),
            (ResourceLevel.LOW, ProblemComplexity.SIMPLE): SpaceTimeParameters(
                branching_factor=3, state_bits=64, max_depth=5, memory_limit_mb=50.0
            ),
            (ResourceLevel.LOW, ProblemComplexity.MODERATE): SpaceTimeParameters(
                branching_factor=3, state_bits=64, max_depth=4, memory_limit_mb=50.0
            ),
            (ResourceLevel.MEDIUM, ProblemComplexity.SIMPLE): SpaceTimeParameters(
                branching_factor=4, state_bits=128, max_depth=8, memory_limit_mb=100.0
            ),
            (ResourceLevel.MEDIUM, ProblemComplexity.MODERATE): SpaceTimeParameters(
                branching_factor=4, state_bits=128, max_depth=6, memory_limit_mb=100.0
            ),
            (ResourceLevel.MEDIUM, ProblemComplexity.COMPLEX): SpaceTimeParameters(
                branching_factor=3, state_bits=128, max_depth=5, memory_limit_mb=100.0
            ),
            (ResourceLevel.HIGH, ProblemComplexity.SIMPLE): SpaceTimeParameters(
                branching_factor=6, state_bits=256, max_depth=12, memory_limit_mb=200.0
            ),
            (ResourceLevel.HIGH, ProblemComplexity.MODERATE): SpaceTimeParameters(
                branching_factor=5, state_bits=256, max_depth=10, memory_limit_mb=200.0
            ),
            (ResourceLevel.HIGH, ProblemComplexity.COMPLEX): SpaceTimeParameters(
                branching_factor=4, state_bits=256, max_depth=8, memory_limit_mb=200.0
            ),
            (ResourceLevel.HIGH, ProblemComplexity.EXTREME): SpaceTimeParameters(
                branching_factor=3, state_bits=256, max_depth=6, memory_limit_mb=200.0
            )
        }
    
    def find_optimal_parameters(self, 
                              resource_profile: ResourceProfile,
                              performance_history: List[Dict[str, Any]],
                              problem_type: str = "unknown") -> SpaceTimeParameters:
        """
        Find optimal tree evaluation parameters based on current resources
        and historical performance.
        """
        resource_level = resource_profile.get_resource_level()
        problem_complexity = resource_profile.problem_complexity
        
        # Get base parameters from template
        base_params = self.parameter_templates.get(
            (resource_level, problem_complexity),
            SpaceTimeParameters()  # Default fallback
        )
        
        # Create optimized parameters
        optimized_params = SpaceTimeParameters(
            branching_factor=base_params.branching_factor,
            state_bits=base_params.state_bits,
            max_depth=base_params.max_depth,
            memory_limit_mb=min(base_params.memory_limit_mb, resource_profile.available_memory_mb * 0.8),
            timeout_seconds=min(30.0, resource_profile.time_constraint)
        )
        
        # Apply learned optimizations
        optimized_params = self._apply_learned_optimizations(
            optimized_params, performance_history, problem_type
        )
        
        # Ensure parameters are within resource constraints
        optimized_params = self._enforce_resource_constraints(
            optimized_params, resource_profile
        )
        
        # Record this optimization for learning
        self._record_optimization(optimized_params, resource_profile, problem_type)
        
        logger.info(f"Optimized parameters: d={optimized_params.branching_factor}, "
                   f"b={optimized_params.state_bits}, h={optimized_params.max_depth}, "
                   f"memory={optimized_params.memory_limit_mb}MB")
        
        return optimized_params
    
    def _apply_learned_optimizations(self, 
                                   params: SpaceTimeParameters,
                                   performance_history: List[Dict[str, Any]],
                                   problem_type: str) -> SpaceTimeParameters:
        """Apply learned optimizations based on historical performance."""
        if not performance_history:
            return params
        
        # Analyze recent performance
        recent_performance = performance_history[-10:] if len(performance_history) >= 10 else performance_history
        
        # Calculate performance metrics
        avg_confidence = np.mean([p.get('confidence', 0.0) for p in recent_performance])
        avg_depth = np.mean([p.get('evaluation_depth', 0) for p in recent_performance])
        avg_memory_usage = np.mean([p.get('memory_usage_mb', 0) for p in recent_performance])
        
        # Adjust parameters based on performance
        if avg_confidence < 0.5:
            # Low confidence - try deeper simulation
            params.max_depth = min(params.max_depth + 2, 15)
        elif avg_confidence > 0.8:
            # High confidence - can reduce depth for efficiency
            params.max_depth = max(params.max_depth - 1, 3)
        
        if avg_memory_usage < params.memory_limit_mb * 0.5:
            # Underutilizing memory - can increase branching or depth
            params.branching_factor = min(params.branching_factor + 1, 8)
        elif avg_memory_usage > params.memory_limit_mb * 0.9:
            # Overutilizing memory - reduce parameters
            params.branching_factor = max(params.branching_factor - 1, 2)
            params.max_depth = max(params.max_depth - 1, 2)
        
        # Problem-specific optimizations
        if problem_type in self.parameter_effectiveness:
            effectiveness = self.parameter_effectiveness[problem_type]
            if effectiveness.get('high_branching_effective', False):
                params.branching_factor = min(params.branching_factor + 1, 8)
            if effectiveness.get('high_depth_effective', False):
                params.max_depth = min(params.max_depth + 2, 15)
        
        return params
    
    def _enforce_resource_constraints(self, 
                                    params: SpaceTimeParameters,
                                    resource_profile: ResourceProfile) -> SpaceTimeParameters:
        """Ensure parameters don't exceed resource constraints."""
        # Memory constraint
        max_memory = resource_profile.available_memory_mb * 0.8
        params.memory_limit_mb = min(params.memory_limit_mb, max_memory)
        
        # Time constraint
        params.timeout_seconds = min(params.timeout_seconds, resource_profile.time_constraint)
        
        # CPU constraint (affects branching factor)
        if resource_profile.cpu_utilization > 0.8:
            params.branching_factor = min(params.branching_factor, 3)
        elif resource_profile.cpu_utilization > 0.6:
            params.branching_factor = min(params.branching_factor, 3)
        
        # Process constraint (affects depth)
        if resource_profile.active_processes > 10:
            params.max_depth = min(params.max_depth, 6)
        elif resource_profile.active_processes > 5:
            params.max_depth = min(params.max_depth, 6)
        
        return params
    
    def _record_optimization(self, 
                           params: SpaceTimeParameters,
                           resource_profile: ResourceProfile,
                           problem_type: str):
        """Record optimization for future learning."""
        optimization_record = {
            'timestamp': time.time(),
            'parameters': params.to_dict(),
            'resource_profile': {
                'memory_mb': resource_profile.available_memory_mb,
                'cpu_utilization': resource_profile.cpu_utilization,
                'active_processes': resource_profile.active_processes,
                'resource_level': resource_profile.get_resource_level().value,
                'problem_complexity': resource_profile.problem_complexity.value
            },
            'problem_type': problem_type
        }
        
        self.optimization_history.append(optimization_record)
        
        # Keep only recent history
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-500:]
    
    def update_effectiveness(self, 
                           problem_type: str,
                           parameters: SpaceTimeParameters,
                           performance_result: Dict[str, Any]):
        """Update effectiveness tracking for a problem type."""
        if problem_type not in self.parameter_effectiveness:
            self.parameter_effectiveness[problem_type] = {
                'high_branching_effective': False,
                'high_depth_effective': False,
                'high_memory_effective': False,
                'sample_count': 0,
                'avg_performance': 0.0
            }
        
        effectiveness = self.parameter_effectiveness[problem_type]
        effectiveness['sample_count'] += 1
        
        # Update performance tracking
        current_avg = effectiveness['avg_performance']
        new_performance = performance_result.get('confidence', 0.0)
        effectiveness['avg_performance'] = (
            (current_avg * (effectiveness['sample_count'] - 1) + new_performance) / 
            effectiveness['sample_count']
        )
        
        # Update effectiveness flags based on performance
        if new_performance > 0.7:
            if parameters.branching_factor >= 5:
                effectiveness['high_branching_effective'] = True
            if parameters.max_depth >= 8:
                effectiveness['high_depth_effective'] = True
            if parameters.memory_limit_mb >= 100:
                effectiveness['high_memory_effective'] = True
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.optimization_history:
            return {'total_optimizations': 0}
        
        recent_optimizations = self.optimization_history[-100:]
        
        # Calculate statistics
        avg_branching = np.mean([opt['parameters']['branching_factor'] for opt in recent_optimizations])
        avg_depth = np.mean([opt['parameters']['max_depth'] for opt in recent_optimizations])
        avg_memory = np.mean([opt['parameters']['memory_limit_mb'] for opt in recent_optimizations])
        
        resource_levels = [opt['resource_profile']['resource_level'] for opt in recent_optimizations]
        resource_distribution = {level: resource_levels.count(level) for level in set(resource_levels)}
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': len(recent_optimizations),
            'average_parameters': {
                'branching_factor': avg_branching,
                'max_depth': avg_depth,
                'memory_limit_mb': avg_memory
            },
            'resource_distribution': resource_distribution,
            'problem_type_effectiveness': self.parameter_effectiveness
        }


class SpaceTimeAwareGovernor:
    """
    Enhanced Governor with space-time awareness for dynamic parameter optimization.
    
    This extends the existing MetaCognitiveGovernor with the ability to
    dynamically adjust tree evaluation parameters based on available resources
    and problem complexity.
    """
    
    def __init__(self, base_governor=None):
        self.base_governor = base_governor
        self.parameter_optimizer = TreeParameterOptimizer()
        self.current_parameters = SpaceTimeParameters()
        self.resource_monitor = ResourceMonitor()
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'parameter_adjustments': 0
        }
        
        logger.info("Space-Time Aware Governor initialized")
    
    def make_decision_with_space_time_awareness(self, 
                                              available_actions: List[int], 
                                              context: Dict[str, Any], 
                                              performance_history: List[Dict[str, Any]], 
                                              current_energy: float) -> Dict[str, Any]:
        """
        Make decision with space-time awareness, optimizing tree evaluation parameters.
        """
        try:
            # Get current resource profile
            resource_profile = self.resource_monitor.get_current_profile(context)
            
            # Determine problem complexity
            problem_complexity = self._assess_problem_complexity(context, performance_history)
            resource_profile.problem_complexity = problem_complexity
            
            # Find optimal parameters
            optimal_params = self.parameter_optimizer.find_optimal_parameters(
                resource_profile, performance_history, context.get('problem_type', 'unknown')
            )
            
            # Update current parameters
            self.current_parameters = optimal_params
            self.optimization_stats['total_optimizations'] += 1
            
            # Make decision using base governor if available
            if self.base_governor:
                decision = self.base_governor.make_decision(
                    available_actions, context, performance_history, current_energy
                )
            else:
                # Fallback decision making
                decision = self._make_fallback_decision(available_actions, context)
            
            # Enhance decision with space-time information
            decision['space_time_parameters'] = optimal_params.to_dict()
            decision['resource_profile'] = {
                'memory_mb': resource_profile.available_memory_mb,
                'cpu_utilization': resource_profile.cpu_utilization,
                'resource_level': resource_profile.get_resource_level().value,
                'problem_complexity': problem_complexity.value
            }
            decision['optimization_reasoning'] = self._generate_optimization_reasoning(
                optimal_params, resource_profile
            )
            
            self.optimization_stats['successful_optimizations'] += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Space-time aware decision making failed: {e}")
            # Fallback to basic decision
            return self._make_fallback_decision(available_actions, context)
    
    def _assess_problem_complexity(self, 
                                 context: Dict[str, Any], 
                                 performance_history: List[Dict[str, Any]]) -> ProblemComplexity:
        """Assess the complexity of the current problem."""
        # Analyze context complexity
        context_complexity = 0
        
        if 'frame_analysis' in context:
            frame_analysis = context['frame_analysis']
            if frame_analysis.get('object_count', 0) > 10:
                context_complexity += 2
            elif frame_analysis.get('object_count', 0) > 5:
                context_complexity += 1
        
        if 'available_actions' in context:
            action_count = len(context['available_actions'])
            if action_count > 8:
                context_complexity += 2
            elif action_count > 5:
                context_complexity += 1
        
        # Analyze performance history complexity
        if performance_history:
            recent_performance = performance_history[-5:]
            avg_confidence = np.mean([p.get('confidence', 0.0) for p in recent_performance])
            if avg_confidence < 0.3:
                context_complexity += 2
            elif avg_confidence < 0.5:
                context_complexity += 1
        
        # Determine complexity level
        if context_complexity >= 4:
            return ProblemComplexity.EXTREME
        elif context_complexity >= 3:
            return ProblemComplexity.COMPLEX
        elif context_complexity >= 1:
            return ProblemComplexity.MODERATE
        else:
            return ProblemComplexity.SIMPLE
    
    def _make_fallback_decision(self, available_actions: List[int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a fallback decision when base governor is not available."""
        recommended_action = available_actions[0] if available_actions else 1
        
        return {
            'recommended_action': recommended_action,
            'confidence': 0.5,
            'reasoning': 'Space-time aware fallback decision',
            'space_time_parameters': self.current_parameters.to_dict()
        }
    
    def _generate_optimization_reasoning(self, 
                                       params: SpaceTimeParameters,
                                       resource_profile: ResourceProfile) -> str:
        """Generate reasoning for parameter optimization."""
        reasoning_parts = [
            f"Resource level: {resource_profile.get_resource_level().value}",
            f"Problem complexity: {resource_profile.problem_complexity.value}",
            f"Optimized for d={params.branching_factor}, b={params.state_bits}, h={params.max_depth}",
            f"Memory limit: {params.memory_limit_mb:.1f}MB"
        ]
        
        return "; ".join(reasoning_parts)
    
    def update_performance_feedback(self, 
                                  problem_type: str,
                                  performance_result: Dict[str, Any]):
        """Update performance feedback for learning."""
        self.parameter_optimizer.update_effectiveness(
            problem_type, self.current_parameters, performance_result
        )
    
    def get_space_time_stats(self) -> Dict[str, Any]:
        """Get space-time optimization statistics."""
        base_stats = self.optimization_stats.copy()
        optimization_stats = self.parameter_optimizer.get_optimization_stats()
        
        return {
            **base_stats,
            'parameter_optimizer': optimization_stats,
            'current_parameters': self.current_parameters.to_dict()
        }


class ResourceMonitor:
    """Monitors system resources for space-time optimization."""
    
    def __init__(self):
        self.resource_history = []
    
    def get_current_profile(self, context: Dict[str, Any]) -> ResourceProfile:
        """Get current resource profile."""
        # In a real implementation, this would query actual system resources
        # For now, we'll simulate based on context and heuristics
        
        # Simulate memory usage based on context
        memory_usage = 50.0  # Base memory
        if 'frame_analysis' in context:
            memory_usage += context['frame_analysis'].get('object_count', 0) * 2
        
        available_memory = max(100.0, 500.0 - memory_usage)  # Simulate 500MB total
        
        # Simulate CPU utilization
        cpu_utilization = 0.3 + (len(context.get('available_actions', [])) * 0.05)
        cpu_utilization = min(1.0, cpu_utilization)
        
        # Simulate active processes
        active_processes = 3 + len(context.get('available_actions', [])) // 2
        
        # Time constraint based on context
        time_constraint = 30.0  # Default 30 seconds
        if 'urgency' in context:
            time_constraint = context['urgency'] * 10.0
        
        return ResourceProfile(
            available_memory_mb=available_memory,
            cpu_utilization=cpu_utilization,
            active_processes=active_processes,
            problem_complexity=ProblemComplexity.SIMPLE,  # Will be updated by caller
            time_constraint=time_constraint
        )


# Integration helper
def create_space_time_aware_governor(base_governor=None) -> SpaceTimeAwareGovernor:
    """Create a space-time aware governor."""
    return SpaceTimeAwareGovernor(base_governor)
