#!/usr/bin/env python3
"""
Enhanced Space-Time Aware Governor

Consolidates all functionality from the old MetaCognitiveGovernor with
the new space-time aware capabilities, providing a unified governor system.

This replaces the old governor entirely while maintaining all existing functionality.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

# Import space-time components
from .space_time_governor import (
    SpaceTimeAwareGovernor,
    TreeParameterOptimizer,
    SpaceTimeParameters,
    ResourceProfile,
    ResourceLevel,
    ProblemComplexity,
    create_space_time_aware_governor
)

# Import legacy components that are still needed
try:
    from .salience_system import SalienceMode
except ImportError:
    class SalienceMode(Enum):
        LOSSLESS = "lossless"
        DECAY_COMPRESSION = "decay_compression"

logger = logging.getLogger(__name__)

class GovernorRecommendationType(Enum):
    """Types of recommendations the Governor can make."""
    MODE_SWITCH = "mode_switch"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    CONSOLIDATION_TRIGGER = "consolidation_trigger"
    RESOURCE_REALLOCATION = "resource_reallocation"
    ARCHITECT_REQUEST = "architect_request"

@dataclass
class CognitiveCost:
    """Abstract cost model for cognitive operations."""
    compute_units: float
    memory_operations: int
    decision_complexity: float
    coordination_overhead: float
    
    def total_cost(self) -> float:
        """Calculate total cognitive cost."""
        return (self.compute_units + 
                self.memory_operations * 0.1 + 
                self.decision_complexity + 
                self.coordination_overhead)

@dataclass
class CognitiveBenefit:
    """Abstract benefit model for cognitive operations."""
    learning_gain: float
    performance_improvement: float
    efficiency_gain: float
    knowledge_consolidation: float
    
    def total_benefit(self) -> float:
        """Calculate total cognitive benefit."""
        return (self.learning_gain + 
                self.performance_improvement + 
                self.efficiency_gain + 
                self.knowledge_consolidation)

@dataclass
class GovernorRecommendation:
    """Recommendation from the Governor."""
    type: GovernorRecommendationType
    configuration_changes: Dict[str, Any]
    confidence: float
    reasoning: str
    expected_benefit: float = 0.0
    implementation_cost: float = 0.0

@dataclass
class ArchitectRequest:
    """Request to the Architect for architectural changes."""
    issue_type: str
    persistent_problem: str
    context_data: Dict[str, Any]
    urgency_level: str = "medium"
    expected_impact: str = "moderate"

class SystemMonitor:
    """Monitors system performance and efficiency."""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.activation_history = deque(maxlen=100)
        self.cost_history = deque(maxlen=100)
        self.benefit_history = deque(maxlen=100)
    
    def record_activation(self, cost: CognitiveCost, benefit: CognitiveBenefit):
        """Record system activation with cost and benefit."""
        self.activation_history.append(time.time())
        self.cost_history.append(cost.total_cost())
        self.benefit_history.append(benefit.total_benefit())
    
    def get_efficiency_ratio(self) -> float:
        """Calculate efficiency ratio (benefit/cost)."""
        if not self.cost_history or not self.benefit_history:
            return 1.0
        
        avg_cost = np.mean(list(self.cost_history))
        avg_benefit = np.mean(list(self.benefit_history))
        
        return avg_benefit / max(avg_cost, 0.001)
    
    def get_recent_trend(self, window_size: int = 10) -> str:
        """Get recent trend in efficiency."""
        if len(self.benefit_history) < window_size:
            return "insufficient_data"
        
        recent_costs = list(self.cost_history)[-window_size:]
        recent_benefits = list(self.benefit_history)[-window_size:]
        
        if len(recent_costs) < 2:
            return "insufficient_data"
        
        cost_trend = np.polyfit(range(len(recent_costs)), recent_costs, 1)[0]
        benefit_trend = np.polyfit(range(len(recent_benefits)), recent_benefits, 1)[0]
        
        if benefit_trend > cost_trend * 1.1:
            return "improving"
        elif benefit_trend < cost_trend * 0.9:
            return "declining"
        else:
            return "stable"

class EnhancedSpaceTimeGovernor:
    """
    Enhanced Space-Time Aware Governor that consolidates all governor functionality.
    
    This replaces the old MetaCognitiveGovernor while adding space-time awareness
    and maintaining all existing functionality for backward compatibility.
    """
    
    def __init__(self, 
                 memory_capacity: int = 1000,
                 decision_threshold: float = 0.7,
                 adaptation_rate: float = 0.1,
                 persistence_dir: Optional[Path] = None):
        
        # Initialize space-time aware components
        self.space_time_governor = SpaceTimeAwareGovernor()
        
        # Legacy governor state
        self.memory_capacity = memory_capacity
        self.decision_threshold = decision_threshold
        self.adaptation_rate = adaptation_rate
        self.persistence_dir = persistence_dir or Path(".")
        
        # System monitoring
        self.system_monitors = {}
        self.decision_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # Action limits (legacy functionality)
        self.action_limits = {
            'max_actions_per_game': 1000,
            'max_actions_per_session': 5000,
            'max_actions_per_scorecard': 1000
        }
        
        # Learning and pattern recognition
        self.learning_manager = None
        self.pattern_analyzer = None
        
        # Initialize learning components
        self._initialize_learning_components()
        
        logger.info("Enhanced Space-Time Aware Governor initialized")
    
    def _initialize_learning_components(self):
        """Initialize learning and pattern recognition components."""
        try:
            from .meta_learning import MetaLearningSystem
            # Try different initialization methods
            try:
                self.learning_manager = MetaLearningSystem(
                    persistence_dir=self.persistence_dir
                )
            except TypeError:
                # Fallback to default initialization
                self.learning_manager = MetaLearningSystem()
            logger.info("Learning manager initialized")
        except ImportError as e:
            logger.warning(f"Learning manager not available: {e}")
        
        try:
            from .pattern_analyzer import PatternAnalyzer
            # Try different initialization methods
            try:
                self.pattern_analyzer = PatternAnalyzer(
                    persistence_dir=self.persistence_dir
                )
            except TypeError:
                # Fallback to default initialization
                self.pattern_analyzer = PatternAnalyzer()
            logger.info("Pattern analyzer initialized")
        except ImportError:
            # Pattern analyzer is optional - create a dummy implementation
            self.pattern_analyzer = None
            logger.debug("Pattern analyzer not available - using dummy implementation")
    
    def make_decision(self, 
                     available_actions: List[int], 
                     context: Dict[str, Any], 
                     performance_history: List[Dict[str, Any]], 
                     current_energy: float) -> Dict[str, Any]:
        """
        Make a decision using enhanced space-time awareness.
        
        This method combines the space-time aware decision making with
        legacy governor functionality for backward compatibility.
        """
        try:
            # Use space-time aware decision making
            space_time_decision = self.space_time_governor.make_decision_with_space_time_awareness(
                available_actions=available_actions,
                context=context,
                performance_history=performance_history,
                current_energy=current_energy
            )
            
            # Enhance with legacy governor analysis
            enhanced_decision = self._enhance_with_legacy_analysis(
                space_time_decision, available_actions, context, performance_history
            )
            
            # Record decision for learning
            self._record_decision(enhanced_decision, context, performance_history)
            
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Enhanced decision making failed: {e}")
            # Fallback to basic space-time decision
            return self.space_time_governor.make_decision_with_space_time_awareness(
                available_actions, context, performance_history, current_energy
            )
    
    def _enhance_with_legacy_analysis(self, 
                                    space_time_decision: Dict[str, Any],
                                    available_actions: List[int],
                                    context: Dict[str, Any],
                                    performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance space-time decision with legacy governor analysis."""
        
        # Add legacy analysis
        enhanced_decision = space_time_decision.copy()
        
        # Add cognitive cost analysis
        cognitive_cost = self._calculate_cognitive_cost(available_actions, context)
        enhanced_decision['cognitive_cost'] = cognitive_cost.total_cost()
        
        # Add efficiency analysis
        efficiency_ratio = self._calculate_efficiency_ratio(performance_history)
        enhanced_decision['efficiency_ratio'] = efficiency_ratio
        
        # Add pattern-based recommendations
        if self.pattern_analyzer:
            pattern_recommendations = self._get_pattern_recommendations(context, performance_history)
            enhanced_decision['pattern_recommendations'] = pattern_recommendations
        
        # Add learning insights
        if self.learning_manager:
            learning_insights = self._get_learning_insights(performance_history)
            enhanced_decision['learning_insights'] = learning_insights
        
        # Enhance reasoning with legacy analysis
        legacy_reasoning = self._generate_legacy_reasoning(cognitive_cost, efficiency_ratio)
        enhanced_decision['reasoning'] = f"{enhanced_decision['reasoning']} | {legacy_reasoning}"
        
        return enhanced_decision
    
    def _calculate_cognitive_cost(self, available_actions: List[int], context: Dict[str, Any]) -> CognitiveCost:
        """Calculate cognitive cost for decision making."""
        # Base cost on number of actions and context complexity
        action_complexity = len(available_actions) * 0.1
        context_complexity = len(str(context)) * 0.001
        
        return CognitiveCost(
            compute_units=action_complexity + context_complexity,
            memory_operations=len(available_actions),
            decision_complexity=action_complexity,
            coordination_overhead=0.1
        )
    
    def _calculate_efficiency_ratio(self, performance_history: List[Dict[str, Any]]) -> float:
        """Calculate efficiency ratio from performance history."""
        if not performance_history:
            return 1.0
        
        recent_performance = performance_history[-10:] if len(performance_history) >= 10 else performance_history
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in recent_performance])
        avg_efficiency = np.mean([p.get('efficiency', 1.0) for p in recent_performance])
        
        return avg_confidence * avg_efficiency
    
    def _get_pattern_recommendations(self, context: Dict[str, Any], performance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get pattern-based recommendations."""
        if not self.pattern_analyzer:
            return []
        
        try:
            # Try different method names that might exist
            if hasattr(self.pattern_analyzer, 'analyze_patterns'):
                return self.pattern_analyzer.analyze_patterns(context, performance_history)
            elif hasattr(self.pattern_analyzer, 'get_recommendations'):
                return self.pattern_analyzer.get_recommendations(context, performance_history)
            elif hasattr(self.pattern_analyzer, 'analyze_context'):
                return self.pattern_analyzer.analyze_context(context, performance_history)
            else:
                # Return basic recommendations if no specific method is available
                return [{
                    'type': 'basic_pattern',
                    'confidence': 0.5,
                    'description': 'Basic pattern analysis not available'
                }]
        except Exception as e:
            logger.debug(f"Pattern analysis failed: {e}")
            return []
    
    def _get_learning_insights(self, performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get learning insights from performance history."""
        if not self.learning_manager:
            return {}
        
        try:
            # Try different method names that might exist
            if hasattr(self.learning_manager, 'analyze_learning_progress'):
                return self.learning_manager.analyze_learning_progress(performance_history)
            elif hasattr(self.learning_manager, 'get_learning_insights'):
                return self.learning_manager.get_learning_insights(performance_history)
            elif hasattr(self.learning_manager, 'analyze_performance'):
                return self.learning_manager.analyze_performance(performance_history)
            else:
                # Return basic insights if no specific method is available
                return {
                    'learning_available': True,
                    'performance_count': len(performance_history),
                    'method': 'basic_analysis'
                }
        except Exception as e:
            logger.debug(f"Learning analysis failed: {e}")
            return {
                'learning_available': False,
                'error': str(e),
                'performance_count': len(performance_history)
            }
    
    def _generate_legacy_reasoning(self, cognitive_cost: CognitiveCost, efficiency_ratio: float) -> str:
        """Generate reasoning based on legacy governor analysis."""
        cost_level = "low" if cognitive_cost.total_cost() < 1.0 else "high"
        efficiency_level = "good" if efficiency_ratio > 0.7 else "poor"
        
        return f"Cost: {cost_level}, Efficiency: {efficiency_level}"
    
    def _record_decision(self, decision: Dict[str, Any], context: Dict[str, Any], performance_history: List[Dict[str, Any]]):
        """Record decision for learning and analysis."""
        decision_record = {
            'timestamp': time.time(),
            'decision': decision,
            'context': context,
            'performance_history_length': len(performance_history)
        }
        
        self.decision_history.append(decision_record)
        
        # Update performance history
        if performance_history:
            self.performance_history.extend(performance_history[-5:])  # Keep last 5 entries
    
    # Legacy methods for backward compatibility
    def set_action_limit_maximums(self, **kwargs):
        """Set action limit maximums (legacy compatibility)."""
        for key, value in kwargs.items():
            if key in self.action_limits:
                self.action_limits[key] = value
                logger.info(f"Updated action limit {key} to {value}")
    
    def get_dynamic_action_limit(self, limit_type: str, 
                                current_performance: Dict[str, Any],
                                context: Dict[str, Any]) -> int:
        """Get dynamic action limit based on performance (legacy compatibility)."""
        base_limit = self.action_limits.get(limit_type, 1000)
        
        # Adjust based on performance
        if current_performance.get('confidence', 0.5) > 0.8:
            return int(base_limit * 1.2)  # Increase limit for good performance
        elif current_performance.get('confidence', 0.5) < 0.3:
            return int(base_limit * 0.8)  # Decrease limit for poor performance
        
        return base_limit
    
    def get_action_limits_status(self) -> Dict[str, Any]:
        """Get current action limits status (legacy compatibility)."""
        return {
            'action_limits': self.action_limits.copy(),
            'space_time_parameters': self.space_time_governor.current_parameters.to_dict(),
            'optimization_stats': self.space_time_governor.get_space_time_stats()
        }
    
    # New methods for enhanced functionality
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including both space-time and legacy metrics."""
        space_time_stats = self.space_time_governor.get_space_time_stats()
        
        return {
            'space_time_governor': space_time_stats,
            'legacy_metrics': {
                'decision_history_length': len(self.decision_history),
                'performance_history_length': len(self.performance_history),
                'action_limits': self.action_limits,
                'learning_manager_available': self.learning_manager is not None,
                'pattern_analyzer_available': self.pattern_analyzer is not None
            },
            'system_monitors': {
                name: {
                    'efficiency_ratio': monitor.get_efficiency_ratio(),
                    'recent_trend': monitor.get_recent_trend()
                }
                for name, monitor in self.system_monitors.items()
            }
        }
    
    def update_performance_feedback(self, problem_type: str, performance_result: Dict[str, Any]):
        """Update performance feedback for learning (enhanced version)."""
        # Update space-time governor
        self.space_time_governor.update_performance_feedback(problem_type, performance_result)
        
        # Update legacy learning components
        if self.learning_manager:
            try:
                self.learning_manager.update_performance(problem_type, performance_result)
            except Exception as e:
                logger.warning(f"Learning manager update failed: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.space_time_governor, 'cleanup'):
            self.space_time_governor.cleanup()
        
        self.decision_history.clear()
        self.performance_history.clear()
        self.system_monitors.clear()
        
        logger.info("Enhanced Space-Time Governor cleaned up")


# Factory function for easy integration
def create_enhanced_space_time_governor(memory_capacity: int = 1000,
                                       decision_threshold: float = 0.7,
                                       adaptation_rate: float = 0.1,
                                       persistence_dir: Optional[Path] = None) -> EnhancedSpaceTimeGovernor:
    """Create an enhanced space-time aware governor with all functionality."""
    return EnhancedSpaceTimeGovernor(
        memory_capacity=memory_capacity,
        decision_threshold=decision_threshold,
        adaptation_rate=adaptation_rate,
        persistence_dir=persistence_dir
    )
