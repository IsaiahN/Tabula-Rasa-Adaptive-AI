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
    
    def __init__(self, log_file: Optional[str] = None, outcome_tracking_dir: Optional[str] = None,
                 persistence_dir: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.Governor")
        self.log_file = log_file
        
        # Outcome tracking integration
        self.outcome_tracker = None
        if outcome_tracking_dir:
            try:
                from src.core.outcome_tracker import OutcomeTracker, PerformanceMetrics
                self.outcome_tracker = OutcomeTracker(Path(outcome_tracking_dir), self.logger)
                self.logger.info("Outcome tracking enabled")
            except ImportError:
                self.logger.warning("Outcome tracker not available")
        
        # Cross-session learning integration
        self.learning_manager = None
        if persistence_dir:
            try:
                from src.core.cross_session_learning import CrossSessionLearningManager, KnowledgeType
                self.learning_manager = CrossSessionLearningManager(Path(persistence_dir), self.logger)
                self.logger.info("Cross-session learning enabled")
            except ImportError:
                self.logger.warning("Cross-session learning not available")
        
        # Meta-cognitive memory management integration
        self.memory_manager = None
        try:
            from src.core.meta_cognitive_memory_manager import MetaCognitiveMemoryManager
            base_path = Path(persistence_dir) if persistence_dir else Path(".")
            self.memory_manager = MetaCognitiveMemoryManager(base_path, self.logger)
            self.logger.info("Meta-cognitive memory management enabled")
        except ImportError:
            self.logger.warning("Meta-cognitive memory manager not available")
        
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
        
        # Tracking for outcome measurement
        self.pending_outcome_measurements = {}  # decision_id -> outcome_id
        
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
            
            # First, check for learned patterns that might apply
            learned_recs = self.get_learned_recommendations(puzzle_type, current_performance)
            for learned_rec in learned_recs:
                # Convert learned recommendation to GovernorRecommendation if confidence is high enough
                if learned_rec['confidence'] >= 0.6 and learned_rec['success_rate'] >= 0.5:
                    rec_type_str = learned_rec['type']
                    if rec_type_str in [t.value for t in GovernorRecommendationType]:
                        rec_type = GovernorRecommendationType(rec_type_str)
                        
                        learned_recommendation = GovernorRecommendation(
                            type=rec_type,
                            configuration_changes=learned_rec['configuration_changes'],
                            confidence=min(0.95, learned_rec['confidence'] + 0.1),  # Boost confidence slightly
                            expected_benefit=CognitiveBenefit(
                                win_rate_improvement=0.1 * learned_rec['success_rate'],
                                score_improvement=5.0 * learned_rec['success_rate'],
                                learning_efficiency=0.05 * learned_rec['success_rate'],
                                knowledge_transfer=0.05 * learned_rec['success_rate']
                            ),
                            rationale=learned_rec['rationale'],
                            urgency=0.5
                        )
                        recommendations.append(learned_recommendation)
            
            # Then generate standard recommendations
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
            decision_id = self._record_decision(best_recommendation, decision_time, system_analysis, current_performance)
            
            self.total_decisions_made += 1
            
            if best_recommendation:
                # Start outcome tracking for this decision
                self.start_outcome_measurement(decision_id, best_recommendation, current_performance)
                
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
        current_win_rate = performance.get('current_win_rate', 0)
        
        # Get history of recent mode switches to avoid oscillation
        recent_mode_switches = [
            decision for decision in list(self.decision_history)[-5:]
            if decision['recommendation_type'] == GovernorRecommendationType.MODE_SWITCH.value
        ]
        
        # Avoid too frequent mode switching
        if len(recent_mode_switches) >= 2:
            return None
            
        # Calculate adaptive confidence based on past mode switch success
        success_rate = self._calculate_mode_switch_success_rate(recent_mode_switches)
        base_confidence = 0.8
        adaptive_confidence = min(0.95, max(0.4, base_confidence + (success_rate - 0.5) * 0.3))
        
        # Different thresholds and strategies based on puzzle type and performance
        if current_win_rate < 0.3:
            strategy_config, rationale = self._select_mode_switch_strategy(
                puzzle_type, current_win_rate, performance
            )
            
            return GovernorRecommendation(
                type=GovernorRecommendationType.MODE_SWITCH,
                configuration_changes=strategy_config,
                confidence=adaptive_confidence,
                expected_benefit=CognitiveBenefit(
                    win_rate_improvement=0.2 * adaptive_confidence,
                    score_improvement=10.0 * adaptive_confidence,
                    learning_efficiency=0.1,
                    knowledge_transfer=0.05
                ),
                rationale=rationale,
                urgency=0.7 if current_win_rate < 0.2 else 0.5
            )
        return None
    
    def _select_mode_switch_strategy(self, puzzle_type: str, win_rate: float, performance: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Select appropriate mode switch strategy based on context."""
        if win_rate < 0.1:
            # Emergency mode - try contrarian approaches
            return (
                {'enable_contrarian_strategy': True, 'increase_exploration': True},
                f"Critical win rate {win_rate:.2f} requires contrarian exploration"
            )
        elif win_rate < 0.2:
            # Try different cognitive approach
            return (
                {'switch_to_pattern_mode': True, 'reduce_noise_threshold': 0.1},
                f"Very low win rate {win_rate:.2f} suggests pattern recognition focus needed"
            )
        elif win_rate < 0.3:
            # Moderate adjustment
            avg_score = performance.get('average_score', 0)
            if avg_score < 5:
                return (
                    {'enable_systematic_search': True, 'increase_depth_limit': 50},
                    f"Low scores with {win_rate:.2f} win rate suggest systematic search needed"
                )
            else:
                return (
                    {'enable_intuitive_mode': True, 'reduce_analysis_time': 0.8},
                    f"Decent scores but {win_rate:.2f} win rate suggest faster intuitive decisions"
                )
        
        # Fallback
        return (
            {'enable_balanced_mode': True},
            f"Balanced approach for {win_rate:.2f} win rate"
        )
    
    def _calculate_mode_switch_success_rate(self, recent_switches: List[Dict]) -> float:
        """Calculate success rate of recent mode switches."""
        if not recent_switches:
            return 0.5  # Neutral baseline
        
        successful = 0
        for switch in recent_switches:
            outcome = switch.get('outcome_metrics', {})
            if outcome.get('win_rate_change', 0) > 0.05:  # Meaningful improvement
                successful += 1
        
        return successful / len(recent_switches) if recent_switches else 0.5
    
    def _evaluate_parameter_adjustments(self, system_analysis: Dict[str, Any]) -> Optional[GovernorRecommendation]:
        """Evaluate if parameter adjustments would be beneficial."""
        if system_analysis['average_efficiency'] < 1.0:
            # Get history of recent parameter adjustments
            recent_param_adjustments = [
                decision for decision in list(self.decision_history)[-10:]
                if decision['recommendation_type'] == GovernorRecommendationType.PARAMETER_ADJUSTMENT.value
            ]
            
            # Determine which parameter to adjust based on history and system state
            config_changes, confidence, rationale = self._select_parameter_adjustment(
                system_analysis, recent_param_adjustments
            )
            
            if config_changes:
                return GovernorRecommendation(
                    type=GovernorRecommendationType.PARAMETER_ADJUSTMENT,
                    configuration_changes=config_changes,
                    confidence=confidence,
                    expected_benefit=CognitiveBenefit(
                        win_rate_improvement=0.05 * confidence,
                        score_improvement=2.0 * confidence,
                        learning_efficiency=0.1 * confidence,
                        knowledge_transfer=0.0
                    ),
                    rationale=rationale,
                    urgency=0.4
                )
        return None
    
    def _select_parameter_adjustment(self, system_analysis: Dict[str, Any], recent_adjustments: List[Dict]) -> Tuple[Dict[str, Any], float, str]:
        """Select intelligent parameter adjustment based on system state and history."""
        efficiency = system_analysis['average_efficiency']
        win_rate = system_analysis.get('win_rate', 0.0)
        
        # Calculate adaptive confidence based on recent adjustment success
        base_confidence = 0.6
        success_rate = self._calculate_adjustment_success_rate(recent_adjustments)
        adaptive_confidence = min(0.95, max(0.3, base_confidence + (success_rate - 0.5) * 0.4))
        
        # Check what parameters were recently adjusted to avoid repetition
        recent_param_types = set()
        for adj in recent_adjustments[-3:]:  # Last 3 adjustments
            config = adj.get('configuration_changes', {})
            recent_param_types.update(config.keys())
        
        # Parameter adjustment strategies based on system state
        if efficiency < 0.7 and 'max_actions_per_game' not in recent_param_types:
            return (
                {'max_actions_per_game': 750},
                adaptive_confidence,
                "Low efficiency suggests need for more exploration actions"
            )
        elif efficiency < 0.8 and win_rate < 0.4 and 'learning_rate' not in recent_param_types:
            return (
                {'learning_rate': 0.001},
                adaptive_confidence * 0.9,
                "Poor win rate indicates learning rate adjustment needed"
            )
        elif efficiency > 0.6 and efficiency < 0.9 and 'batch_size' not in recent_param_types:
            return (
                {'batch_size': 64},
                adaptive_confidence * 0.8,
                "Moderate efficiency suggests batch size optimization"
            )
        elif 'temperature' not in recent_param_types:
            # Exploration vs exploitation balance
            temp_value = 0.8 if win_rate < 0.5 else 0.3
            return (
                {'temperature': temp_value},
                adaptive_confidence * 0.7,
                f"Adjusting exploration temperature based on win rate {win_rate:.2f}"
            )
        else:
            # All main parameters recently adjusted, suggest multi-parameter fine-tuning
            return (
                {
                    'max_actions_per_game': min(1000, int(750 * (1 + (1 - efficiency)))),
                    'exploration_bonus': 0.1 if efficiency < 0.8 else 0.05
                },
                adaptive_confidence * 0.6,
                "Multi-parameter fine-tuning after recent individual adjustments"
            )
    
    def _calculate_adjustment_success_rate(self, recent_adjustments: List[Dict]) -> float:
        """Calculate success rate of recent parameter adjustments."""
        if not recent_adjustments:
            return 0.5  # Neutral baseline
        
        successful = 0
        for adj in recent_adjustments:
            # Consider adjustment successful if it led to improvement
            # This is a simplified heuristic - in practice would track actual outcomes
            outcome = adj.get('outcome_metrics', {})
            if outcome.get('win_rate_change', 0) > 0 or outcome.get('efficiency_change', 0) > 0:
                successful += 1
        
        return successful / len(recent_adjustments) if recent_adjustments else 0.5
    
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
                        decision_time: float, system_analysis: Dict[str, Any],
                        current_performance: Dict[str, Any] = None) -> str:
        """Record a decision for future analysis."""
        decision_id = f"decision_{self.total_decisions_made}_{int(time.time())}"
        
        decision_record = {
            'timestamp': time.time(),
            'recommendation': recommendation.to_dict() if recommendation else None,
            'recommendation_type': recommendation.type.value if recommendation else 'no_action',
            'decision_time_ms': decision_time * 1000,
            'system_state': system_analysis,
            'decision_id': decision_id,
            'current_performance': current_performance or {}
        }
        
        self.decision_history.append(decision_record)
        
        # Log to file if specified
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(decision_record) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to log decision: {e}")
        
        return decision_id
    
    def start_outcome_measurement(self, decision_id: str, recommendation: GovernorRecommendation,
                                 current_performance: Dict[str, Any]) -> Optional[str]:
        """Start tracking outcomes for a recommendation."""
        if not self.outcome_tracker:
            return None
        
        try:
            from src.core.outcome_tracker import PerformanceMetrics
            
            # Convert current performance to PerformanceMetrics
            baseline_metrics = PerformanceMetrics(
                win_rate=current_performance.get('win_rate', 0.0),
                average_score=current_performance.get('average_score', 0.0),
                learning_efficiency=current_performance.get('learning_efficiency', 0.0),
                knowledge_transfer=current_performance.get('knowledge_transfer', 0.0),
                computational_efficiency=current_performance.get('computational_efficiency', 1.0),
                memory_usage=current_performance.get('memory_usage', 0.5),
                inference_speed=current_performance.get('inference_speed', 1.0)
            )
            
            outcome_id = self.outcome_tracker.start_outcome_measurement(
                decision_id=decision_id,
                intervention_type=f"governor_{recommendation.type.value}",
                intervention_details=recommendation.configuration_changes,
                baseline_metrics=baseline_metrics
            )
            
            self.pending_outcome_measurements[decision_id] = outcome_id
            return outcome_id
            
        except Exception as e:
            self.logger.error(f"Failed to start outcome measurement: {e}")
            return None
    
    def complete_outcome_measurement(self, decision_id: str, 
                                   post_performance: Dict[str, Any],
                                   sample_size: int = 1,
                                   notes: str = ""):
        """Complete outcome measurement for a decision."""
        if not self.outcome_tracker or decision_id not in self.pending_outcome_measurements:
            return
        
        try:
            from src.core.outcome_tracker import PerformanceMetrics
            
            # Convert post performance to PerformanceMetrics
            post_metrics = PerformanceMetrics(
                win_rate=post_performance.get('win_rate', 0.0),
                average_score=post_performance.get('average_score', 0.0),
                learning_efficiency=post_performance.get('learning_efficiency', 0.0),
                knowledge_transfer=post_performance.get('knowledge_transfer', 0.0),
                computational_efficiency=post_performance.get('computational_efficiency', 1.0),
                memory_usage=post_performance.get('memory_usage', 0.5),
                inference_speed=post_performance.get('inference_speed', 1.0)
            )
            
            outcome_id = self.pending_outcome_measurements[decision_id]
            outcome_record = self.outcome_tracker.complete_outcome_measurement(
                outcome_id=outcome_id,
                post_metrics=post_metrics,
                sample_size=sample_size,
                notes=notes
            )
            
            # Update decision history with outcome
            for decision in reversed(self.decision_history):
                if decision.get('decision_id') == decision_id:
                    decision['outcome_metrics'] = {
                        'success_score': outcome_record.success_score,
                        'status': outcome_record.status.value,
                        'win_rate_change': outcome_record.performance_deltas.get('win_rate_delta', 0),
                        'score_change': outcome_record.performance_deltas.get('score_delta', 0),
                        'efficiency_change': outcome_record.performance_deltas.get('learning_efficiency_delta', 0)
                    }
                    break
            
            # Update success counter
            if outcome_record.success_score >= 0.4:
                self.successful_recommendations += 1
            
            # Clean up pending measurements
            del self.pending_outcome_measurements[decision_id]
            
        except Exception as e:
            self.logger.error(f"Failed to complete outcome measurement: {e}")
    
    def get_effectiveness_insights(self) -> Dict[str, Any]:
        """Get insights about Governor effectiveness from outcome tracking."""
        if not self.outcome_tracker:
            return {'insights_available': False}
        
        insights = self.outcome_tracker.get_learning_insights()
        
        # Add Governor-specific insights
        governor_insights = {
            'total_decisions': self.total_decisions_made,
            'tracked_outcomes': len(self.outcome_tracker.outcome_history),
            'pending_measurements': len(self.pending_outcome_measurements),
            'estimated_success_rate': self.successful_recommendations / max(self.total_decisions_made, 1)
        }
        
        # Get effectiveness by recommendation type
        recommendation_types = [
            'governor_mode_switch',
            'governor_parameter_adjustment', 
            'governor_consolidation_trigger'
        ]
        
        for rec_type in recommendation_types:
            stats = self.outcome_tracker.get_intervention_effectiveness(rec_type)
            governor_insights[f'{rec_type}_effectiveness'] = stats
        
        insights['governor_specific'] = governor_insights
        insights['insights_available'] = True
        
        return insights
    
    def start_learning_session(self, session_context: Dict[str, Any] = None) -> Optional[str]:
        """Start a cross-session learning session."""
        if not self.learning_manager:
            return None
        
        session_id = self.learning_manager.start_session(session_context)
        self.logger.info(f"Started cross-session learning: {session_id}")
        return session_id
    
    def end_learning_session(self, performance_summary: Dict[str, Any] = None):
        """End the current learning session."""
        if not self.learning_manager:
            return
        
        if not performance_summary:
            performance_summary = {
                'total_decisions': self.total_decisions_made,
                'successful_decisions': self.successful_recommendations,
                'avg_improvement': 0.1 if self.successful_recommendations > 0 else 0.0
            }
        
        self.learning_manager.end_session(performance_summary)
    
    def learn_from_recommendation_outcome(self, recommendation: GovernorRecommendation,
                                        context: Dict[str, Any], success_metrics: Dict[str, float]):
        """Learn from the outcome of a recommendation."""
        if not self.learning_manager:
            return
        
        from src.core.cross_session_learning import KnowledgeType, PersistenceLevel
        
        # Learn strategy pattern
        strategy_data = {
            'recommendation_type': recommendation.type.value,
            'configuration_changes': recommendation.configuration_changes,
            'original_confidence': recommendation.confidence,
            'expected_benefit': asdict(recommendation.expected_benefit)
        }
        
        # Determine success score
        success_score = (
            success_metrics.get('win_rate_improvement', 0) * 0.4 +
            success_metrics.get('score_improvement', 0) / 20.0 * 0.3 +
            success_metrics.get('efficiency_improvement', 0) * 0.3
        )
        
        # Determine persistence level based on success
        persistence_level = PersistenceLevel.PERMANENT if success_score > 0.6 else PersistenceLevel.SESSION
        
        pattern_id = self.learning_manager.learn_pattern(
            KnowledgeType.STRATEGY_PATTERN,
            strategy_data,
            context,
            success_score,
            persistence_level
        )
        
        self.logger.debug(f"Learned strategy pattern {pattern_id} with success score {success_score:.3f}")
        
        # Also learn parameter optimization patterns
        if recommendation.type.value == 'parameter_adjustment':
            param_data = {
                'parameter_changes': recommendation.configuration_changes,
                'system_state': context.get('system_state', {}),
                'performance_improvement': success_score
            }
            
            self.learning_manager.learn_pattern(
                KnowledgeType.PARAMETER_OPTIMIZATION,
                param_data,
                context,
                success_score,
                PersistenceLevel.PERMANENT if success_score > 0.7 else PersistenceLevel.SESSION
            )
    
    def get_learned_recommendations(self, puzzle_type: str, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns."""
        if not self.learning_manager:
            return []
        
        from src.core.cross_session_learning import KnowledgeType
        
        context = {
            'puzzle_type': puzzle_type,
            'current_performance': current_performance
        }
        
        # Get applicable strategy patterns
        strategy_patterns = self.learning_manager.retrieve_applicable_patterns(
            KnowledgeType.STRATEGY_PATTERN,
            context,
            min_confidence=0.4,
            max_results=3
        )
        
        recommendations = []
        for pattern in strategy_patterns:
            pattern_data = pattern.pattern_data
            
            rec_data = {
                'type': pattern_data.get('recommendation_type', 'unknown'),
                'configuration_changes': pattern_data.get('configuration_changes', {}),
                'confidence': pattern.confidence,
                'success_rate': pattern.success_rate,
                'applications': pattern.total_applications,
                'rationale': f"Learned strategy (success rate: {pattern.success_rate:.1%})"
            }
            
            recommendations.append(rec_data)
        
        return recommendations
    
    def get_best_configuration_for_context(self, puzzle_type: str, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Get the best known configuration for the current context."""
        if not self.learning_manager:
            return {}
        
        context = {
            'puzzle_type': puzzle_type,
            'current_performance': current_performance
        }
        
        return self.learning_manager.get_best_configuration_for_context(context)
    
    def get_cross_session_insights(self) -> Dict[str, Any]:
        """Get insights from cross-session learning."""
        if not self.learning_manager:
            return {'learning_available': False}
        
        insights = self.learning_manager.get_performance_insights()
        insights['learning_available'] = True
        
        return insights
    
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

    def perform_memory_management(self, emergency_cleanup: bool = False, 
                                 target_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform intelligent memory management with meta-cognitive awareness.
        
        Args:
            emergency_cleanup: If True, perform aggressive cleanup
            target_size_mb: Target size for emergency cleanup
        
        Returns:
            Dictionary with memory management results
        """
        if not self.memory_manager:
            self.logger.warning("Memory manager not available")
            return {"status": "unavailable"}
        
        try:
            if emergency_cleanup and target_size_mb:
                self.logger.info(f"Performing emergency memory cleanup (target: {target_size_mb} MB)")
                results = self.memory_manager.emergency_cleanup(target_size_mb)
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "emergency_memory_cleanup",
                    "target_size_mb": target_size_mb,
                    "files_deleted": results["files_deleted"],
                    "bytes_freed": results["bytes_freed"],
                    "critical_files_protected": results["critical_files_protected"]
                })
                
            else:
                self.logger.info("Performing routine memory management")
                results = self.memory_manager.perform_garbage_collection(dry_run=False)
                
                # Log Governor decision
                self.log_governor_decision({
                    "decision_type": "routine_memory_management",
                    "files_processed": results["files_processed"],
                    "files_deleted": results["files_deleted"],
                    "bytes_freed": results["bytes_freed"],
                    "critical_files_protected": results["critical_files_protected"]
                })
            
            # Update system health metrics
            memory_status = self.memory_manager.get_memory_status()
            self.system_monitors["memory_system"] = {
                "total_files": memory_status["total_files"],
                "total_size_mb": memory_status["total_size_mb"],
                "status": "healthy" if memory_status["total_size_mb"] < 1000 else "attention_needed",
                "last_cleanup": time.time()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Memory management failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status with Governor analysis."""
        if not self.memory_manager:
            return {"status": "unavailable"}
        
        try:
            base_status = self.memory_manager.get_memory_status()
            
            # Add Governor analysis
            analysis = {
                "health_status": "healthy",
                "recommendations": [],
                "critical_files_count": base_status["classifications"].get("critical_lossless", {}).get("file_count", 0),
                "cleanup_needed": False
            }
            
            # Analyze memory health
            total_size = base_status["total_size_mb"]
            if total_size > 2000:  # Over 2GB
                analysis["health_status"] = "critical"
                analysis["cleanup_needed"] = True
                analysis["recommendations"].append("Immediate cleanup recommended")
            elif total_size > 1000:  # Over 1GB
                analysis["health_status"] = "attention_needed"
                analysis["recommendations"].append("Consider cleanup soon")
            
            # Check for too many temporary files
            temp_files = base_status["classifications"].get("temporary_purge", {}).get("file_count", 0)
            if temp_files > 100:
                analysis["recommendations"].append("High number of temporary files detected")
            
            base_status["governor_analysis"] = analysis
            return base_status
            
        except Exception as e:
            self.logger.error(f"Memory status check failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def schedule_memory_maintenance(self, interval_hours: int = 24) -> bool:
        """Schedule regular memory maintenance."""
        try:
            # This would integrate with a scheduler in a full implementation
            self.logger.info(f"Memory maintenance scheduled every {interval_hours} hours")
            
            # Log the scheduling decision
            self.log_governor_decision({
                "decision_type": "schedule_memory_maintenance",
                "interval_hours": interval_hours,
                "next_maintenance": time.time() + (interval_hours * 3600)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to schedule memory maintenance: {e}")
            return False


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
