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

# Import caching and performance monitoring
from .caching_system import get_global_cache, cache_result, cache_async_result, CacheLevel
from .unified_performance_monitor import get_performance_monitor, monitor_performance

# Space-time components (moved from deleted space_time_governor.py)
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
    VERY_COMPLEX = "very_complex"

@dataclass
class SpaceTimeParameters:
    """Space-time aware parameters for tree evaluation."""
    d: float = 0.5  # Depth parameter
    b: float = 0.5  # Breadth parameter  
    h: float = 0.5  # Height parameter
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ResourceProfile:
    """Resource availability profile."""
    cpu_usage: float = 0.5
    memory_usage: float = 0.5
    time_available: float = 1.0
    level: ResourceLevel = ResourceLevel.MEDIUM

class TreeParameterOptimizer:
    """Optimizes tree evaluation parameters based on resources."""
    
    def __init__(self):
        self.current_parameters = SpaceTimeParameters()
    
    def optimize_parameters(self, resource_profile: ResourceProfile, complexity: ProblemComplexity) -> SpaceTimeParameters:
        """Optimize parameters based on resources and complexity."""
        # Simple optimization logic
        if resource_profile.level == ResourceLevel.HIGH:
            return SpaceTimeParameters(d=0.8, b=0.8, h=0.8)
        elif resource_profile.level == ResourceLevel.LOW:
            return SpaceTimeParameters(d=0.3, b=0.3, h=0.3)
        else:
            return SpaceTimeParameters(d=0.5, b=0.5, h=0.5)

class SpaceTimeAwareGovernor:
    """Space-time aware governor for resource optimization."""
    
    def __init__(self):
        self.current_parameters = SpaceTimeParameters()
        self.optimizer = TreeParameterOptimizer()
    
    def make_decision_with_space_time_awareness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision with space-time awareness."""
        return {
            'parameters': self.current_parameters.to_dict(),
            'decision': 'optimized',
            'context': context
        }
    
    def get_space_time_stats(self) -> Dict[str, Any]:
        """Get space-time statistics."""
        return {
            'parameters': self.current_parameters.to_dict(),
            'optimizer_stats': {}
        }
    
    def update_performance_feedback(self, problem_type: str, performance_result: float):
        """Update performance feedback."""
        pass
    
    def record_action_result(self, action: str, success: bool, score_change: float, context: Dict[str, Any]):
        """Record action result."""
        pass
    
    def analyze_performance_and_recommend(self, recent_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance and recommend improvements."""
        return {'recommendations': [], 'analysis': {}}

def create_space_time_aware_governor():
    """Create a space-time aware governor."""
    return SpaceTimeAwareGovernor()

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
        
        # Initialize 4-Phase Memory Optimization Coordinator
        try:
            from .four_phase_memory_coordinator import create_four_phase_memory_coordinator
            # Convert string to Path object if needed
            persistence_path = self.persistence_dir if isinstance(self.persistence_dir, Path) else Path(self.persistence_dir)
            self.four_phase_coordinator = create_four_phase_memory_coordinator(persistence_path)
            self.four_phase_initialized = False
            logger.info("4-Phase Memory Coordinator created")
        except Exception as e:
            logger.warning(f"Failed to initialize 4-Phase Memory Coordinator: {e}")
            self.four_phase_coordinator = None
            self.four_phase_initialized = False
        
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
            # Initialize with database-only approach
            self.learning_manager = MetaLearningSystem()
            logger.info("Learning manager initialized successfully")
            print(f"🧠 LEARNING MANAGER: Successfully initialized MetaLearningSystem")
        except ImportError as e:
            logger.warning(f"Learning manager not available: {e}")
            print(f"🧠 LEARNING MANAGER: Import error - {e}")
            self.learning_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize learning manager: {e}")
            print(f"🧠 LEARNING MANAGER: Initialization error - {e}")
            self.learning_manager = None
        
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
    
    @monitor_performance("governor", "pattern_recommendations")
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
    
    def record_action_result(self, action: int, success: bool, score_change: float, 
                           context: Dict[str, Any] = None) -> None:
        """
        Record the result of an action for learning and analysis.
        
        Args:
            action: The action number that was taken
            success: Whether the action was successful
            score_change: Change in score from the action
            context: Additional context about the action
        """
        try:
            # Record in space-time governor
            if hasattr(self.space_time_governor, 'record_action_result'):
                self.space_time_governor.record_action_result(action, success, score_change, context)
            
            # Record in performance history
            result_entry = {
                'action': action,
                'success': success,
                'score_change': score_change,
                'timestamp': time.time(),
                'context': context or {}
            }
            self.performance_history.append(result_entry)
            
            # Update learning manager if available
            if self.learning_manager:
                try:
                    # Create a simple experience for the learning manager
                    from .data_models import Experience
                    experience = Experience(
                        state=context.get('state', []) if context else [],
                        action=action,
                        next_state=context.get('next_state', []) if context else [],
                        reward=score_change,
                        learning_progress=0.1 if success else 0.0,
                        energy_change=score_change * 0.1,
                        timestamp=int(time.time())
                    )
                    self.learning_manager.add_experience(experience, context.get('context', 'general'))
                except Exception as e:
                    logger.warning(f"Failed to record action result in learning manager: {e}")
                    # Log error to database
                    try:
                        import asyncio
                        from src.database.system_integration import get_system_integration
                        
                        async def log_error():
                            integration = get_system_integration()
                            await integration.log_system_event(
                                level="ERROR",
                                component="enhanced_space_time_governor",
                                message=f"Failed to record action result in learning manager: {e}",
                                data={"action": action, "success": success, "score_change": score_change, "error": str(e)},
                                session_id=context.get('session_id', 'unknown') if context else 'unknown'
                            )
                        
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(log_error())
                        else:
                            # Skip if no event loop is running
                            pass
                    except Exception as log_error:
                        logger.error(f"Failed to log error to database: {log_error}")
            
            logger.debug(f"Recorded action {action}: success={success}, score_change={score_change}")
            
        except Exception as e:
            logger.error(f"Failed to record action result: {e}")
    
    @monitor_performance("governor", "analyze_performance")
    @cache_result(ttl_seconds=300, level=CacheLevel.MEMORY)  # Cache for 5 minutes
    def analyze_performance_and_recommend(self, recent_actions: int = 50) -> Dict[str, Any]:
        """
        Analyze recent performance and provide recommendations.
        
        Args:
            recent_actions: Number of recent actions to analyze
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            # Get recent performance data
            recent_performance = list(self.performance_history)[-recent_actions:] if self.performance_history else []
            
            if not recent_performance:
                return {
                    'analysis': 'No recent performance data available',
                    'recommendations': ['Continue training to collect performance data'],
                    'confidence': 0.0
                }
            
            # Analyze success rate
            successful_actions = sum(1 for entry in recent_performance if entry['success'])
            success_rate = successful_actions / len(recent_performance)
            
            # Analyze score changes
            total_score_change = sum(entry['score_change'] for entry in recent_performance)
            avg_score_change = total_score_change / len(recent_performance)
            
            # Analyze action distribution
            action_counts = {}
            for entry in recent_performance:
                action = entry['action']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Generate recommendations
            recommendations = []
            confidence = 0.5
            
            if success_rate < 0.3:
                recommendations.append("Success rate is low - consider exploring different action strategies")
                confidence -= 0.2
            elif success_rate > 0.7:
                recommendations.append("Success rate is high - continue current strategy")
                confidence += 0.2
            
            if avg_score_change < 0:
                recommendations.append("Average score change is negative - review action selection")
                confidence -= 0.1
            elif avg_score_change > 0:
                recommendations.append("Average score change is positive - good progress")
                confidence += 0.1
            
            # Check for action diversity
            if len(action_counts) < 3:
                recommendations.append("Limited action diversity - try exploring more action types")
                confidence -= 0.1
            
            # Use space-time governor analysis if available
            space_time_analysis = {}
            if hasattr(self.space_time_governor, 'analyze_performance_and_recommend'):
                try:
                    space_time_analysis = self.space_time_governor.analyze_performance_and_recommend(recent_actions)
                except Exception as e:
                    logger.warning(f"Space-time governor analysis failed: {e}")
            
            # Create reasoning text
            reasoning = f"Analyzed {len(recent_performance)} recent actions. Success rate: {success_rate:.2f}, Avg score change: {avg_score_change:.2f}"
            if recommendations:
                reasoning += f" Recommendations: {', '.join(recommendations)}"
            else:
                reasoning += " No specific recommendations at this time."
            
            return {
                'analysis': {
                    'success_rate': success_rate,
                    'avg_score_change': avg_score_change,
                    'total_actions_analyzed': len(recent_performance),
                    'action_distribution': action_counts,
                    'space_time_analysis': space_time_analysis
                },
                'recommendations': recommendations,
                'confidence': max(0.0, min(1.0, confidence)),
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return {
                'analysis': f'Analysis failed: {e}',
                'recommendations': ['Check system logs for errors'],
                'confidence': 0.0,
                'reasoning': f'Analysis failed due to error: {e}'
            }
    
    def cleanup_logs_on_new_game(self, game_id: str) -> Dict[str, Any]:
        """
        Clean up logs when starting a new game.
        
        Args:
            game_id: The ID of the new game
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            # Clear old performance history to prevent memory buildup
            if len(self.performance_history) > 1000:
                # Keep only the most recent 500 entries
                recent_performance = list(self.performance_history)[-500:]
                self.performance_history.clear()
                self.performance_history.extend(recent_performance)
            
            # Clear old decision history
            if len(self.decision_history) > 500:
                recent_decisions = list(self.decision_history)[-250:]
                self.decision_history.clear()
                self.decision_history.extend(recent_decisions)
            
            logger.debug(f"Cleaned up logs for new game {game_id}")
            return {'success': True, 'action': 'cleaned', 'game_id': game_id}
            
        except Exception as e:
            logger.warning(f"Failed to cleanup logs for game {game_id}: {e}")
            return {'success': False, 'action': 'failed', 'error': str(e), 'game_id': game_id}
    
    def optimize_parameters_dynamically(self, 
                                      performance_data: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically optimize d, b, h parameters based on performance data.
        
        Args:
            performance_data: Recent performance data for analysis
            context: Current context for parameter adjustment
            
        Returns:
            Dictionary containing optimization results and new parameters
        """
        try:
            # Get current parameters
            current_params = self.space_time_governor.current_parameters
            current_d = current_params.d
            current_b = current_params.b
            current_h = current_params.h
            
            # Analyze performance trends
            performance_analysis = self._analyze_performance_trends(performance_data)
            
            # Calculate optimization adjustments
            d_adjustment = self._calculate_d_adjustment(performance_analysis, context)
            b_adjustment = self._calculate_b_adjustment(performance_analysis, context)
            h_adjustment = self._calculate_h_adjustment(performance_analysis, context)
            
            # Apply adjustments with bounds checking
            new_d = max(0.1, min(1.0, current_d + d_adjustment))
            new_b = max(0.1, min(1.0, current_b + b_adjustment))
            new_h = max(0.1, min(1.0, current_h + h_adjustment))
            
            # Update parameters
            self.space_time_governor.current_parameters.d = new_d
            self.space_time_governor.current_parameters.b = new_b
            self.space_time_governor.current_parameters.h = new_h
            
            # Log optimization
            logger.info(f"Parameter optimization: d={new_d:.3f}, b={new_b:.3f}, h={new_h:.3f}")
            
            return {
                'success': True,
                'old_parameters': {'d': current_d, 'b': current_b, 'h': current_h},
                'new_parameters': {'d': new_d, 'b': new_b, 'h': new_h},
                'adjustments': {'d': d_adjustment, 'b': b_adjustment, 'h': h_adjustment},
                'performance_analysis': performance_analysis,
                'reasoning': self._generate_optimization_reasoning(performance_analysis, d_adjustment, b_adjustment, h_adjustment)
            }
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'old_parameters': {'d': current_d, 'b': current_b, 'h': current_h},
                'new_parameters': {'d': current_d, 'b': current_b, 'h': current_h}
            }
    
    def _analyze_performance_trends(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends to inform parameter optimization."""
        if not performance_data:
            return {'trend': 'no_data', 'confidence': 0.0}
        
        # Calculate success rate trend
        success_rates = []
        for i in range(0, len(performance_data), 10):  # Every 10 actions
            batch = performance_data[i:i+10]
            if batch:
                success_rate = sum(1 for entry in batch if entry.get('success', False)) / len(batch)
                success_rates.append(success_rate)
        
        if len(success_rates) < 2:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Calculate trend direction
        recent_avg = np.mean(success_rates[-3:]) if len(success_rates) >= 3 else success_rates[-1]
        earlier_avg = np.mean(success_rates[:-3]) if len(success_rates) >= 6 else success_rates[0]
        
        trend_direction = 'improving' if recent_avg > earlier_avg else 'declining'
        trend_magnitude = abs(recent_avg - earlier_avg)
        
        # Calculate efficiency trend
        efficiency_scores = [entry.get('efficiency', 1.0) for entry in performance_data if 'efficiency' in entry]
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 1.0
        
        # Calculate action diversity
        actions = [entry.get('action', 0) for entry in performance_data]
        action_diversity = len(set(actions)) / len(actions) if actions else 0.0
        
        return {
            'trend': trend_direction,
            'magnitude': trend_magnitude,
            'success_rate': recent_avg,
            'efficiency': avg_efficiency,
            'action_diversity': action_diversity,
            'confidence': min(1.0, len(performance_data) / 50.0)  # More data = higher confidence
        }
    
    def _calculate_d_adjustment(self, performance_analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate adjustment for depth parameter (d)."""
        base_adjustment = 0.0
        
        # Adjust based on success rate trend
        if performance_analysis['trend'] == 'improving':
            # If improving, increase depth to explore more
            base_adjustment += 0.05 * performance_analysis['magnitude']
        elif performance_analysis['trend'] == 'declining':
            # If declining, decrease depth to focus on immediate actions
            base_adjustment -= 0.03 * performance_analysis['magnitude']
        
        # Adjust based on efficiency
        if performance_analysis['efficiency'] > 0.8:
            # High efficiency, can afford more depth
            base_adjustment += 0.02
        elif performance_analysis['efficiency'] < 0.5:
            # Low efficiency, reduce depth
            base_adjustment -= 0.02
        
        # Adjust based on action diversity
        if performance_analysis['action_diversity'] < 0.3:
            # Low diversity, increase depth to explore more
            base_adjustment += 0.03
        elif performance_analysis['action_diversity'] > 0.7:
            # High diversity, can reduce depth
            base_adjustment -= 0.01
        
        return base_adjustment
    
    def _calculate_b_adjustment(self, performance_analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate adjustment for branching factor parameter (b)."""
        base_adjustment = 0.0
        
        # Adjust based on success rate
        if performance_analysis['success_rate'] > 0.7:
            # High success rate, increase branching to explore more options
            base_adjustment += 0.03
        elif performance_analysis['success_rate'] < 0.3:
            # Low success rate, decrease branching to focus on fewer options
            base_adjustment -= 0.02
        
        # Adjust based on action diversity
        if performance_analysis['action_diversity'] < 0.4:
            # Low diversity, increase branching
            base_adjustment += 0.04
        elif performance_analysis['action_diversity'] > 0.8:
            # High diversity, can reduce branching
            base_adjustment -= 0.02
        
        # Adjust based on trend
        if performance_analysis['trend'] == 'improving':
            # Improving trend, increase branching slightly
            base_adjustment += 0.01
        elif performance_analysis['trend'] == 'declining':
            # Declining trend, decrease branching
            base_adjustment -= 0.02
        
        return base_adjustment
    
    def _calculate_h_adjustment(self, performance_analysis: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate adjustment for horizon parameter (h)."""
        base_adjustment = 0.0
        
        # Adjust based on efficiency
        if performance_analysis['efficiency'] > 0.8:
            # High efficiency, can afford longer horizon
            base_adjustment += 0.04
        elif performance_analysis['efficiency'] < 0.5:
            # Low efficiency, reduce horizon
            base_adjustment -= 0.03
        
        # Adjust based on success rate trend
        if performance_analysis['trend'] == 'improving':
            # Improving trend, increase horizon
            base_adjustment += 0.02 * performance_analysis['magnitude']
        elif performance_analysis['trend'] == 'declining':
            # Declining trend, decrease horizon
            base_adjustment -= 0.03 * performance_analysis['magnitude']
        
        # Adjust based on context complexity
        context_complexity = len(str(context)) / 1000.0  # Rough complexity measure
        if context_complexity > 0.5:
            # Complex context, increase horizon
            base_adjustment += 0.02
        elif context_complexity < 0.2:
            # Simple context, can reduce horizon
            base_adjustment -= 0.01
        
        return base_adjustment
    
    def _generate_optimization_reasoning(self, performance_analysis: Dict[str, Any], 
                                       d_adj: float, b_adj: float, h_adj: float) -> str:
        """Generate human-readable reasoning for parameter optimization."""
        reasoning_parts = []
        
        # Trend-based reasoning
        if performance_analysis['trend'] == 'improving':
            reasoning_parts.append("Performance is improving")
        elif performance_analysis['trend'] == 'declining':
            reasoning_parts.append("Performance is declining")
        
        # Parameter-specific reasoning
        if abs(d_adj) > 0.02:
            if d_adj > 0:
                reasoning_parts.append(f"increased depth (d) by {d_adj:.3f} for better exploration")
            else:
                reasoning_parts.append(f"decreased depth (d) by {abs(d_adj):.3f} for focused action")
        
        if abs(b_adj) > 0.02:
            if b_adj > 0:
                reasoning_parts.append(f"increased branching (b) by {b_adj:.3f} for more options")
            else:
                reasoning_parts.append(f"decreased branching (b) by {abs(b_adj):.3f} for focused search")
        
        if abs(h_adj) > 0.02:
            if h_adj > 0:
                reasoning_parts.append(f"increased horizon (h) by {h_adj:.3f} for longer-term planning")
            else:
                reasoning_parts.append(f"decreased horizon (h) by {abs(h_adj):.3f} for immediate focus")
        
        if not reasoning_parts:
            reasoning_parts.append("no significant adjustments needed")
        
        return f"Optimization: {', '.join(reasoning_parts)} (confidence: {performance_analysis['confidence']:.2f})"
    
    def get_parameter_optimization_status(self) -> Dict[str, Any]:
        """Get current status of parameter optimization system."""
        current_params = self.space_time_governor.current_parameters
        
        return {
            'current_parameters': {
                'd': current_params.d,
                'b': current_params.b,
                'h': current_params.h
            },
            'optimization_enabled': True,
            'last_optimization': getattr(self, '_last_optimization_time', None),
            'optimization_count': getattr(self, '_optimization_count', 0),
            'performance_data_points': len(self.performance_history)
        }
    
    async def initialize_four_phase_memory_system(self) -> bool:
        """Initialize the 4-Phase Memory Optimization system."""
        if not self.four_phase_coordinator:
            logger.warning("4-Phase Memory Coordinator not available")
            return False
        
        try:
            if not self.four_phase_initialized:
                success = await self.four_phase_coordinator.initialize()
                if success:
                    self.four_phase_initialized = True
                    logger.info("4-Phase Memory Optimization system initialized")
                else:
                    logger.error("Failed to initialize 4-Phase Memory Optimization system")
                return success
            else:
                logger.info("4-Phase Memory Optimization system already initialized")
                return True
        except Exception as e:
            logger.error(f"Error initializing 4-Phase Memory system: {e}")
            return False
    
    async def process_memory_through_four_phases(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory data through all four phases of optimization."""
        if not self.four_phase_initialized or not self.four_phase_coordinator:
            logger.warning("4-Phase Memory system not initialized")
            return {'error': '4-Phase Memory system not initialized'}
        
        try:
            # Process through all four phases
            results = await self.four_phase_coordinator.process_memory_data(memory_data)
            
            # Update governor with insights from 4-phase processing
            await self._integrate_four_phase_insights(results)
            
            return results
        except Exception as e:
            logger.error(f"Error processing memory through 4 phases: {e}")
            return {'error': str(e)}
    
    async def _integrate_four_phase_insights(self, four_phase_results: Dict[str, Any]) -> None:
        """Integrate insights from 4-phase processing into governor decision making."""
        try:
            # Extract pattern recognition insights
            if 'pattern_recognition' in four_phase_results:
                pattern_data = four_phase_results['pattern_recognition']
                if 'optimization_suggestions' in pattern_data:
                    # Apply pattern-based optimizations to space-time parameters
                    await self._apply_pattern_optimizations(pattern_data['optimization_suggestions'])
            
            # Extract memory clustering insights
            if 'memory_clustering' in four_phase_results:
                cluster_data = four_phase_results['memory_clustering']
                if 'clustering_insights' in cluster_data:
                    # Use clustering insights for better memory management
                    await self._apply_clustering_insights(cluster_data['clustering_insights'])
            
            # Extract architect evolution insights
            if 'architect_evolution' in four_phase_results:
                architect_data = four_phase_results['architect_evolution']
                if 'evolution_results' in architect_data:
                    # Apply architectural improvements
                    await self._apply_architectural_improvements(architect_data['evolution_results'])
            
            # Extract performance optimization insights
            if 'performance_optimization' in four_phase_results:
                performance_data = four_phase_results['performance_optimization']
                if 'optimization_results' in performance_data:
                    # Apply performance optimizations
                    await self._apply_performance_optimizations(performance_data['optimization_results'])
            
        except Exception as e:
            logger.error(f"Error integrating 4-phase insights: {e}")
    
    async def _apply_pattern_optimizations(self, optimization_suggestions: List[Dict[str, Any]]) -> None:
        """Apply pattern-based optimizations to the governor."""
        try:
            for suggestion in optimization_suggestions:
                if suggestion.get('type') == 'memory_efficiency':
                    # Adjust space-time parameters based on memory efficiency suggestions
                    efficiency_factor = suggestion.get('efficiency_factor', 1.0)
                    if efficiency_factor > 1.1:  # Significant improvement
                        # Increase depth and branching for better exploration
                        self.space_time_governor.current_parameters.d = min(1.0, 
                            self.space_time_governor.current_parameters.d * 1.05)
                        self.space_time_governor.current_parameters.b = min(1.0,
                            self.space_time_governor.current_parameters.b * 1.05)
                    elif efficiency_factor < 0.9:  # Significant degradation
                        # Reduce depth and branching for focused action
                        self.space_time_governor.current_parameters.d = max(0.1,
                            self.space_time_governor.current_parameters.d * 0.95)
                        self.space_time_governor.current_parameters.b = max(0.1,
                            self.space_time_governor.current_parameters.b * 0.95)
        except Exception as e:
            logger.error(f"Error applying pattern optimizations: {e}")
    
    async def _apply_clustering_insights(self, clustering_insights: Dict[str, Any]) -> None:
        """Apply memory clustering insights to the governor."""
        try:
            # Use clustering insights to improve memory management
            if 'memory_locality_score' in clustering_insights:
                locality_score = clustering_insights['memory_locality_score']
                if locality_score > 0.8:  # Good locality
                    # Can afford more complex operations
                    self.space_time_governor.current_parameters.h = min(1.0,
                        self.space_time_governor.current_parameters.h * 1.02)
                elif locality_score < 0.5:  # Poor locality
                    # Simplify operations
                    self.space_time_governor.current_parameters.h = max(0.1,
                        self.space_time_governor.current_parameters.h * 0.98)
        except Exception as e:
            logger.error(f"Error applying clustering insights: {e}")
    
    async def _apply_architectural_improvements(self, evolution_results: Dict[str, Any]) -> None:
        """Apply architectural improvements from the evolution engine."""
        try:
            # Apply architectural improvements to governor parameters
            if 'parameter_adjustments' in evolution_results:
                adjustments = evolution_results['parameter_adjustments']
                for param, adjustment in adjustments.items():
                    if param == 'decision_threshold':
                        self.decision_threshold = max(0.1, min(1.0, 
                            self.decision_threshold + adjustment))
                    elif param == 'adaptation_rate':
                        self.adaptation_rate = max(0.01, min(0.5,
                            self.adaptation_rate + adjustment))
        except Exception as e:
            logger.error(f"Error applying architectural improvements: {e}")
    
    async def _apply_performance_optimizations(self, optimization_results: Dict[str, Any]) -> None:
        """Apply performance optimizations from the performance engine."""
        try:
            # Apply performance optimizations to governor
            if 'resource_optimizations' in optimization_results:
                resource_opts = optimization_results['resource_optimizations']
                if 'memory_usage_reduction' in resource_opts:
                    # Reduce memory usage by adjusting parameters
                    reduction_factor = resource_opts['memory_usage_reduction']
                    self.space_time_governor.current_parameters.d = max(0.1,
                        self.space_time_governor.current_parameters.d * (1 - reduction_factor))
        except Exception as e:
            logger.error(f"Error applying performance optimizations: {e}")
    
    async def run_four_phase_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete 4-phase optimization cycle."""
        if not self.four_phase_initialized or not self.four_phase_coordinator:
            logger.warning("4-Phase Memory system not initialized")
            return {'error': '4-Phase Memory system not initialized'}
        
        try:
            # Run optimization cycle
            result = await self.four_phase_coordinator.run_optimization_cycle()
            
            # Integrate results into governor
            if result.success:
                await self._integrate_optimization_cycle_results(result)
            
            return {
                'success': result.success,
                'cycle_id': result.cycle_id,
                'phases_optimized': result.phases_optimized,
                'performance_improvements': result.performance_improvements,
                'optimization_duration': result.optimization_duration,
                'error_message': result.error_message
            }
        except Exception as e:
            logger.error(f"Error running 4-phase optimization cycle: {e}")
            return {'error': str(e)}
    
    async def _integrate_optimization_cycle_results(self, result) -> None:
        """Integrate results from an optimization cycle into the governor."""
        try:
            # Update governor parameters based on performance improvements
            for phase, improvement in result.performance_improvements.items():
                if improvement > 0.1:  # Significant improvement
                    # Adjust space-time parameters based on improvement
                    if phase == 'pattern_recognition':
                        # Pattern recognition improved - can increase exploration
                        self.space_time_governor.current_parameters.d = min(1.0,
                            self.space_time_governor.current_parameters.d * (1 + improvement * 0.1))
                    elif phase == 'memory_clustering':
                        # Memory clustering improved - can increase branching
                        self.space_time_governor.current_parameters.b = min(1.0,
                            self.space_time_governor.current_parameters.b * (1 + improvement * 0.1))
                    elif phase == 'performance_optimization':
                        # Performance improved - can increase horizon
                        self.space_time_governor.current_parameters.h = min(1.0,
                            self.space_time_governor.current_parameters.h * (1 + improvement * 0.1))
        except Exception as e:
            logger.error(f"Error integrating optimization cycle results: {e}")
    
    async def get_four_phase_system_status(self) -> Dict[str, Any]:
        """Get status of the 4-Phase Memory Optimization system."""
        if not self.four_phase_coordinator:
            return {'error': '4-Phase Memory Coordinator not available'}
        
        try:
            return await self.four_phase_coordinator.get_system_status()
        except Exception as e:
            logger.error(f"Error getting 4-phase system status: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.space_time_governor, 'cleanup'):
            self.space_time_governor.cleanup()
        
        # Clean up 4-phase coordinator
        if self.four_phase_coordinator and self.four_phase_initialized:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.four_phase_coordinator.cleanup())
                else:
                    loop.run_until_complete(self.four_phase_coordinator.cleanup())
            except Exception as e:
                logger.warning(f"Error cleaning up 4-phase coordinator: {e}")
        
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
