#!/usr/bin/env python3
"""
MetaCognitiveGovernor - The "Third Brain"

A runtime supervisor that dynamically manages the AI's "cognitive economy."
Makes high-level decisions to switch training modes, allocate "attention" 
(compute to algorithms), and trigger consolidation based on real-time 
efficacy analysis of its own software processes.

This module is completely hardware-agnostic and reasons in terms of 
abstract computational cycles and algorithmic efficiency.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque

# Import existing system components for integration
try:
    from src.core.salience_system import SalienceMode
except ImportError:
    # Fallback for direct execution
    class SalienceMode(Enum):
        LOSSLESS = "lossless"
        DECAY_COMPRESSION = "decay_compression"

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
    compute_units: float  # Abstract compute cost
    memory_operations: int  # Memory read/write operations
    decision_complexity: float  # Complexity of decisions made
    coordination_overhead: float  # Inter-system coordination cost
    
    def total_cost(self) -> float:
        """Calculate total abstract cost."""
        return (self.compute_units + 
                self.memory_operations * 0.1 + 
                self.decision_complexity + 
                self.coordination_overhead)

@dataclass
class CognitiveBenefit:
    """Benefit measurement for cognitive operations."""
    win_rate_improvement: float  # Improvement in win rate
    score_improvement: float  # Improvement in average score
    learning_efficiency: float  # Learning speed improvement
    knowledge_transfer: float  # Cross-domain learning benefit
    
    def total_benefit(self) -> float:
        """Calculate total benefit score."""
        return (self.win_rate_improvement * 100 + 
                self.score_improvement + 
                self.learning_efficiency * 10 + 
                self.knowledge_transfer * 5)

@dataclass
class GovernorRecommendation:
    """Recommendation from the Governor."""
    type: GovernorRecommendationType
    configuration_changes: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    expected_benefit: CognitiveBenefit
    rationale: str
    urgency: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/transmission."""
        return {
            'type': self.type.value,
            'configuration_changes': self.configuration_changes,
            'confidence': self.confidence,
            'expected_benefit': asdict(self.expected_benefit),
            'rationale': self.rationale,
            'urgency': self.urgency,
            'timestamp': time.time()
        }

@dataclass
class ArchitectRequest:
    """Request to the Architect for architectural changes."""
    issue_type: str
    persistent_problem: str
    failed_solutions: List[Dict[str, Any]]
    performance_data: Dict[str, Any]
    suggested_research_directions: List[str]
    priority: float  # 0.0 to 1.0

class CognitiveSystemMonitor:
    """Monitors individual cognitive systems for cost-benefit analysis."""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.cost_history = deque(maxlen=100)
        self.benefit_history = deque(maxlen=100)
        self.activation_count = 0
        self.total_runtime = 0.0
        self.last_activation = None
        self.performance_impact = {}
        
    def record_activation(self, cost: CognitiveCost, benefit: CognitiveBenefit):
        """Record an activation of this cognitive system."""
        self.cost_history.append(cost)
        self.benefit_history.append(benefit)
        self.activation_count += 1
        self.last_activation = time.time()
        
    def get_efficiency_ratio(self) -> float:
        """Get benefit/cost ratio for this system."""
        if not self.cost_history or not self.benefit_history:
            return 1.0  # Default neutral efficiency
            
        avg_cost = sum(c.total_cost() for c in self.cost_history) / len(self.cost_history)
        avg_benefit = sum(b.total_benefit() for b in self.benefit_history) / len(self.benefit_history)
        
        if avg_cost == 0:
            return float('inf') if avg_benefit > 0 else 1.0
        return avg_benefit / avg_cost
    
    def get_recent_trend(self, window_size: int = 10) -> str:
        """Get trend of recent efficiency changes."""
        if len(self.benefit_history) < window_size:
            return "insufficient_data"
            
        recent_benefits = list(self.benefit_history)[-window_size:]
        earlier_benefits = list(self.benefit_history)[-window_size*2:-window_size]
        
        if not earlier_benefits:
            return "insufficient_data"
            
        recent_avg = sum(b.total_benefit() for b in recent_benefits) / len(recent_benefits)
        earlier_avg = sum(b.total_benefit() for b in earlier_benefits) / len(earlier_benefits)
        
        if recent_avg > earlier_avg * 1.1:
            return "improving"
        elif recent_avg < earlier_avg * 0.9:
            return "declining"
        else:
            return "stable"

class MetaCognitiveGovernor:
    """
    The "Third Brain" - Meta-Cognitive Resource Allocator
    
    Acts as an internal superintendent of cognitive processes, making dynamic
    high-level decisions about resource allocation between software components.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.Governor")
        self.log_file = log_file
        
        # Cognitive system monitors
        self.system_monitors = {}
        self.initialize_system_monitors()
        
        # Decision history
        self.decision_history = deque(maxlen=1000)
        self.performance_baseline = {}
        self.current_config = None
        
        # Governor state
        self.total_decisions_made = 0
        self.successful_recommendations = 0
        self.start_time = time.time()
        
        # Architect communication
        self.pending_architect_requests = []
        self.architect_response_history = []
        
        self.logger.info("🧠 MetaCognitiveGovernor initialized - Third Brain online")
    
    def initialize_system_monitors(self):
        """Initialize monitors for all cognitive systems."""
        # Core cognitive systems from the existing architecture
        cognitive_systems = [
            "swarm_intelligence",
            "dnc_memory", 
            "meta_learning_system",
            "energy_management",
            "sleep_cycles",
            "coordinate_intelligence",
            "frame_analysis",
            "boundary_detection",
            "memory_consolidation",
            "action_intelligence",
            "goal_invention",
            "learning_progress_drive",
            "death_manager",
            "exploration_strategies",
            "pattern_recognition",
            "knowledge_transfer",
            "boredom_detection",
            "mid_game_sleep",
            "action_experimentation",
            "reset_decisions",
            "curriculum_learning",
            "multi_modal_input",
            "temporal_memory",
            "hebbian_bonuses",
            "memory_regularization",
            "gradient_flow_monitoring",
            "usage_tracking",
            "salient_memory_retrieval",
            "anti_bias_weighting",
            "stagnation_detection",
            "emergency_movement",
            "cluster_formation",
            "danger_zone_avoidance",
            "predictive_coordinates",
            "rate_limiting_management",
            "contrarian_strategy",
            "salience_system"
        ]
        
        for system in cognitive_systems:
            self.system_monitors[system] = CognitiveSystemMonitor(system)
            
        self.logger.info(f"🔍 Initialized {len(cognitive_systems)} cognitive system monitors")
    
    def get_recommended_configuration(self, 
                                    puzzle_type: str,
                                    current_performance: Dict[str, Any],
                                    current_config: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """
        Main API: Get recommended configuration changes based on current state.
        
        Args:
            puzzle_type: Type of puzzle being solved
            current_performance: Current performance metrics
            current_config: Current system configuration
            
        Returns:
            GovernorRecommendation or None if no changes recommended
        """
        self.current_config = current_config
        start_time = time.time()
        
        try:
            # Analyze current system state
            system_analysis = self._analyze_cognitive_systems()
            performance_analysis = self._analyze_performance_trends(current_performance)
            resource_analysis = self._analyze_resource_utilization()
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # Check for mode switching opportunities
            mode_rec = self._evaluate_mode_switching(puzzle_type, performance_analysis)
            if mode_rec:
                recommendations.append(mode_rec)
            
            # Check for parameter adjustments
            param_rec = self._evaluate_parameter_adjustments(system_analysis)
            if param_rec:
                recommendations.append(param_rec)
            
            # Check for consolidation needs
            consolidation_rec = self._evaluate_consolidation_trigger(resource_analysis)
            if consolidation_rec:
                recommendations.append(consolidation_rec)
            
            # Check if Architect intervention is needed
            architect_rec = self._evaluate_architect_request(system_analysis, performance_analysis)
            if architect_rec:
                recommendations.append(architect_rec)
            
            # Select best recommendation
            best_recommendation = self._select_best_recommendation(recommendations)
            
            # Record decision
            decision_time = time.time() - start_time
            self._record_decision(best_recommendation, decision_time, system_analysis)
            
            self.total_decisions_made += 1
            
            if best_recommendation:
                self.logger.info(f"🎯 Governor recommendation: {best_recommendation.type.value} "
                               f"(confidence: {best_recommendation.confidence:.2f})")
                return best_recommendation
            else:
                self.logger.debug("📊 No configuration changes recommended")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in Governor decision-making: {e}")
            return None
    
    def record_system_activation(self, system_name: str, 
                               cost: CognitiveCost, 
                               benefit: CognitiveBenefit):
        """Record activation of a cognitive system for monitoring."""
        if system_name in self.system_monitors:
            self.system_monitors[system_name].record_activation(cost, benefit)
            self.logger.debug(f"📈 Recorded {system_name} activation: "
                            f"cost={cost.total_cost():.2f}, benefit={benefit.total_benefit():.2f}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all monitored systems."""
        status = {
            'total_decisions': self.total_decisions_made,
            'successful_recommendations': self.successful_recommendations,
            'success_rate': (self.successful_recommendations / max(1, self.total_decisions_made)),
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'system_efficiencies': {},
            'pending_architect_requests': len(self.pending_architect_requests),
            'top_performers': [],
            'underperformers': []
        }
        
        # Calculate system efficiencies
        efficiencies = {}
        for name, monitor in self.system_monitors.items():
            efficiencies[name] = {
                'efficiency_ratio': monitor.get_efficiency_ratio(),
                'activation_count': monitor.activation_count,
                'trend': monitor.get_recent_trend()
            }
        
        status['system_efficiencies'] = efficiencies
        
        # Identify top performers and underperformers
        sorted_systems = sorted(efficiencies.items(), 
                              key=lambda x: x[1]['efficiency_ratio'], 
                              reverse=True)
        
        status['top_performers'] = [name for name, data in sorted_systems[:5] 
                                  if data['activation_count'] > 0]
        status['underperformers'] = [name for name, data in sorted_systems[-5:] 
                                   if data['activation_count'] > 0 and data['efficiency_ratio'] < 0.5]
        
        return status
    
    def create_architect_request(self, issue_type: str, 
                               problem_description: str,
                               performance_data: Dict[str, Any]) -> ArchitectRequest:
        """Create a request for the Architect to address systemic issues."""
        request = ArchitectRequest(
            issue_type=issue_type,
            persistent_problem=problem_description,
            failed_solutions=self._get_failed_solutions_history(issue_type),
            performance_data=performance_data,
            suggested_research_directions=self._suggest_research_directions(issue_type),
            priority=self._calculate_issue_priority(issue_type, performance_data)
        )
        
        self.pending_architect_requests.append(request)
        self.logger.warning(f"🔬 Architect request created: {issue_type}")
        
        return request
    
    def has_persistent_issues(self) -> bool:
        """Check if there are persistent issues that require architectural changes."""
        return len(self.pending_architect_requests) > 0
    
    # Private methods for internal analysis
    
    def _analyze_cognitive_systems(self) -> Dict[str, Any]:
        """Analyze the current state of all cognitive systems."""
        analysis = {
            'total_systems': len(self.system_monitors),
            'active_systems': 0,
            'high_efficiency_systems': [],
            'low_efficiency_systems': [],
            'trending_up': [],
            'trending_down': [],
            'average_efficiency': 0.0
        }
        
        total_efficiency = 0.0
        active_count = 0
        
        for name, monitor in self.system_monitors.items():
            if monitor.activation_count > 0:
                active_count += 1
                efficiency = monitor.get_efficiency_ratio()
                total_efficiency += efficiency
                trend = monitor.get_recent_trend()
                
                if efficiency > 2.0:
                    analysis['high_efficiency_systems'].append(name)
                elif efficiency < 0.5:
                    analysis['low_efficiency_systems'].append(name)
                
                if trend == "improving":
                    analysis['trending_up'].append(name)
                elif trend == "declining":
                    analysis['trending_down'].append(name)
        
        analysis['active_systems'] = active_count
        analysis['average_efficiency'] = total_efficiency / max(1, active_count)
        
        return analysis
    
    def _analyze_performance_trends(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # This would integrate with existing performance tracking
        return {
            'current_win_rate': current_performance.get('win_rate', 0.0),
            'current_score': current_performance.get('average_score', 0.0),
            'trend': 'stable',  # Would calculate from history
            'stagnation_detected': False,
            'breakthrough_potential': 0.5
        }
    
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze abstract resource utilization across systems."""
        return {
            'memory_pressure': 0.3,  # Would calculate from actual memory usage
            'compute_utilization': 0.7,
            'coordination_overhead': 0.2,
            'consolidation_needed': False
        }
    
    def _evaluate_mode_switching(self, puzzle_type: str, performance: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if mode switching would be beneficial."""
        # Example logic for mode switching
        if performance.get('current_win_rate', 0) < 0.3:
            return GovernorRecommendation(
                type=GovernorRecommendationType.MODE_SWITCH,
                configuration_changes={'enable_contrarian_strategy': True},
                confidence=0.8,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.2,
                    score_improvement=10.0,
                    learning_efficiency=0.1,
                    knowledge_transfer=0.0
                ),
                rationale="Low win rate suggests need for contrarian strategies",
                urgency=0.7
            )
        return None
    
    def _evaluate_parameter_adjustments(self, system_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if parameter adjustments would be beneficial."""
        if system_analysis['average_efficiency'] < 1.0:
            return GovernorRecommendation(
                type=GovernorRecommendationType.PARAMETER_ADJUSTMENT,
                configuration_changes={'max_actions_per_game': 750},
                confidence=0.6,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.05,
                    score_improvement=2.0,
                    learning_efficiency=0.1,
                    knowledge_transfer=0.0
                ),
                rationale="Low system efficiency suggests need for more exploration",
                urgency=0.4
            )
        return None
    
    def _evaluate_consolidation_trigger(self, resource_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if consolidation should be triggered."""
        if resource_analysis.get('memory_pressure', 0) > 0.8:
            return GovernorRecommendation(
                type=GovernorRecommendationType.CONSOLIDATION_TRIGGER,
                configuration_changes={'trigger_consolidation': True},
                confidence=0.9,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.0,
                    score_improvement=0.0,
                    learning_efficiency=0.2,
                    knowledge_transfer=0.1
                ),
                rationale="High memory pressure requires consolidation",
                urgency=0.8
            )
        return None
    
    def _evaluate_architect_request(self, system_analysis: Dict[str, Any], 
                                  performance_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if Architect intervention is needed."""
        # Check for persistent low performance across multiple systems
        if (len(system_analysis['low_efficiency_systems']) > 5 and 
            performance_analysis.get('stagnation_detected', False)):
            
            return GovernorRecommendation(
                type=GovernorRecommendationType.ARCHITECT_REQUEST,
                configuration_changes={'request_architectural_review': True},
                confidence=0.7,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.1,
                    score_improvement=5.0,
                    learning_efficiency=0.3,
                    knowledge_transfer=0.2
                ),
                rationale="Multiple low-efficiency systems suggest architectural issues",
                urgency=0.6
            )
        return None
    
    def _select_best_recommendation(self, recommendations: List[GovernorRecommendation]) -> Optional[GovernorRecommendation]:
        """Select the best recommendation from available options."""
        if not recommendations:
            return None
        
        # Sort by urgency * confidence * expected benefit
        def score_recommendation(rec):
            return rec.urgency * rec.confidence * rec.expected_benefit.total_benefit()
        
        return max(recommendations, key=score_recommendation)
    
    def _record_decision(self, recommendation: Optional[GovernorRecommendation], 
                        decision_time: float, system_analysis: Dict[str, Any]):
        """Record a decision for future analysis."""
        decision_record = {
            'timestamp': time.time(),
            'recommendation': recommendation.to_dict() if recommendation else None,
            'decision_time_ms': decision_time * 1000,
            'system_state': system_analysis,
            'decision_id': self.total_decisions_made
        }
        
        self.decision_history.append(decision_record)
        
        # Log to file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(decision_record) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to log decision: {e}")
    
    def _get_failed_solutions_history(self, issue_type: str) -> List[Dict[str, Any]]:
        """Get history of failed solutions for this issue type."""
        # Would implement based on decision history
        return []
    
    def _suggest_research_directions(self, issue_type: str) -> List[str]:
        """Suggest research directions for architectural improvements."""
        suggestions = {
            'low_efficiency': [
                "Investigate memory allocation optimization",
                "Research cognitive load balancing",
                "Explore dynamic system activation patterns"
            ],
            'stagnation': [
                "Design new exploration strategies", 
                "Research meta-learning enhancements",
                "Investigate curriculum learning improvements"
            ]
        }
        return suggestions.get(issue_type, ["General architectural review"])
    
    def _calculate_issue_priority(self, issue_type: str, performance_data: Dict[str, Any]) -> float:
        """Calculate priority level for an issue."""
        base_priority = {
            'low_efficiency': 0.6,
            'stagnation': 0.8,
            'resource_exhaustion': 0.9,
            'system_failure': 1.0
        }.get(issue_type, 0.5)
        
        # Adjust based on performance impact
        performance_impact = 1.0 - performance_data.get('win_rate', 0.5)
        
        return min(1.0, base_priority * (1 + performance_impact))


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Governor instance
    governor = MetaCognitiveGovernor("governor_decisions.log")
    
    # Simulate some system activations
    test_cost = CognitiveCost(
        compute_units=10.0,
        memory_operations=5,
        decision_complexity=2.0,
        coordination_overhead=1.0
    )
    
    test_benefit = CognitiveBenefit(
        win_rate_improvement=0.1,
        score_improvement=5.0,
        learning_efficiency=0.2,
        knowledge_transfer=0.1
    )
    
    # Record some activations
    governor.record_system_activation("swarm_intelligence", test_cost, test_benefit)
    governor.record_system_activation("meta_learning_system", test_cost, test_benefit)
    
    # Test recommendation system
    current_performance = {
        'win_rate': 0.6,
        'average_score': 45.2,
        'learning_speed': 0.3
    }
    
    current_config = {
        'enable_swarm': True,
        'salience_mode': 'decay_compression',
        'max_actions_per_game': 500
    }
    
    recommendation = governor.get_recommended_configuration(
        puzzle_type="spatial_reasoning",
        current_performance=current_performance,
        current_config=current_config
    )
    
    if recommendation:
        print(f"🎯 Governor recommendation: {recommendation.rationale}")
        print(f"   Changes: {recommendation.configuration_changes}")
        print(f"   Confidence: {recommendation.confidence:.2f}")
    else:
        print("📊 No changes recommended")
    
    # Show system status
    status = governor.get_system_status()
    print(f"\n📈 Governor Status:")
    print(f"   Decisions made: {status['total_decisions']}")
    print(f"   Success rate: {status['success_rate']:.1%}")
    print(f"   Active systems: {len(status['system_efficiencies'])}")
    print(f"   Top performers: {status['top_performers']}")
