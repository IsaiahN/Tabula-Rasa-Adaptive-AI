"""
Context-Dependent Fitness Evolution Engine (Tier 2)

This system evolves success criteria beyond just winning levels, adapting the fitness
function based on the context of what's being learned, current game situation, and
performance patterns. It integrates with the attention system to dynamically adjust
optimization priorities based on what needs the most improvement.

Key Features:
- Dynamic fitness criteria that evolve during gameplay
- Context-aware fitness evaluation
- Multi-dimensional optimization with pareto frontier analysis
- Integration with attention system for resource allocation
- Automatic adaptation based on performance feedback
"""

import asyncio
import json
import time
import uuid
import math
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import attention coordination components
try:
    from .central_attention_controller import SubsystemDemand
    from .weighted_communication_system import MessagePriority
    ATTENTION_COORDINATION_AVAILABLE = True
except ImportError:
    ATTENTION_COORDINATION_AVAILABLE = False
    SubsystemDemand = None
    MessagePriority = None

logger = logging.getLogger(__name__)

class LearningPhase(Enum):
    """Learning phases that influence fitness criteria."""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"
    MASTERY = "mastery"

class ContextType(Enum):
    """Types of learning contexts."""
    EARLY_EXPLORATION = "early_exploration"
    PATTERN_LEARNING = "pattern_learning"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    MASTERY_PHASE = "mastery_phase"

class FitnessCriteriaType(Enum):
    """Types of fitness criteria that can evolve."""
    EXPLORATION_DEPTH = "exploration_depth"
    PATTERN_DISCOVERY = "pattern_discovery"
    STRATEGY_INNOVATION = "strategy_innovation"
    LEARNING_EFFICIENCY = "learning_efficiency"
    ADAPTATION_SPEED = "adaptation_speed"
    GOAL_ACHIEVEMENT = "goal_achievement"
    KNOWLEDGE_RETENTION = "knowledge_retention"
    TRANSFER_LEARNING = "transfer_learning"

@dataclass
class FitnessCriteria:
    """A dynamic fitness criteria that can evolve."""
    criteria_id: str
    criteria_type: FitnessCriteriaType
    criteria_name: str
    criteria_definition: Dict[str, Any]
    base_weight: float = 1.0
    current_weight: float = 1.0
    context_modifiers: Dict[str, float] = None
    evolution_history: List[Dict[str, Any]] = None
    performance_correlation: float = 0.0
    criteria_effectiveness: float = 0.5
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.05
    last_weight_update: float = 0.0

    def __post_init__(self):
        if self.context_modifiers is None:
            self.context_modifiers = {}
        if self.evolution_history is None:
            self.evolution_history = []

@dataclass
class ContextualFitnessEvaluation:
    """Results of a context-aware fitness evaluation."""
    evaluation_id: str
    context_snapshot: Dict[str, Any]
    fitness_criteria_used: Dict[str, float]
    individual_scores: Dict[str, float]
    composite_fitness_score: float
    context_type: ContextType
    learning_phase: LearningPhase
    performance_indicators: Dict[str, Any]
    predicted_improvement_areas: List[str]
    fitness_trend_analysis: Dict[str, Any]
    evaluation_confidence: float = 0.5

@dataclass
class FitnessEvolutionTrigger:
    """A trigger that can cause fitness criteria to evolve."""
    trigger_id: str
    trigger_type: str
    trigger_context: Dict[str, Any]
    affected_criteria: List[str]
    suggested_adjustments: Dict[str, float]
    trigger_strength: float = 0.5
    automatic_adjustment: bool = False
    manual_review_required: bool = False

class ContextDependentFitnessEvolution:
    """
    Main engine for context-dependent fitness evolution.

    This system dynamically adapts fitness criteria based on learning context,
    performance patterns, and integration with the attention system.
    """

    def __init__(self, db_manager):
        self.db = db_manager

        # Core fitness criteria registry
        self.active_criteria: Dict[str, FitnessCriteria] = {}
        self.fitness_functions: Dict[str, callable] = {}

        # Context tracking
        self.current_context: Optional[Dict[str, Any]] = None
        self.current_learning_phase: LearningPhase = LearningPhase.INITIALIZATION
        self.context_history: List[Dict[str, Any]] = []

        # Attention system integration
        self.attention_controller = None
        self.communication_system = None
        self.attention_integration_enabled = ATTENTION_COORDINATION_AVAILABLE

        # Evolution tracking
        self.evolution_triggers: List[FitnessEvolutionTrigger] = []
        self.adaptation_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.fitness_evaluation_history: List[ContextualFitnessEvaluation] = []
        self.performance_correlation_matrix: Optional[np.ndarray] = None

        # Configuration
        self.config = {
            "max_criteria_count": 8,
            "min_criteria_weight": 0.1,
            "max_criteria_weight": 2.0,
            "adaptation_sensitivity": 0.1,
            "evolution_trigger_threshold": 0.7,
            "stability_window_size": 10,
            "pareto_frontier_analysis_enabled": True,
            "automatic_adaptation_enabled": True,
            "attention_integration_priority": 0.8
        }

        # Initialize default fitness criteria
        self._initialize_default_criteria()
        self._initialize_fitness_functions()

        logger.info("Context-Dependent Fitness Evolution engine initialized")

    def set_attention_coordination(self, attention_controller, communication_system):
        """Set attention coordination systems for enhanced integration."""
        self.attention_controller = attention_controller
        self.communication_system = communication_system
        if attention_controller and communication_system:
            logger.info("Fitness evolution enhanced with attention coordination")
        else:
            logger.warning("Attention coordination systems not fully available for fitness evolution")

    async def evaluate_contextual_fitness(self,
                                         game_id: str,
                                         session_id: str,
                                         context: Dict[str, Any],
                                         performance_data: Dict[str, Any],
                                         learning_phase: Optional[LearningPhase] = None) -> ContextualFitnessEvaluation:
        """
        Perform context-aware fitness evaluation using current criteria and weights.
        """
        try:
            evaluation_id = f"fitness_eval_{game_id}_{int(time.time() * 1000)}"

            # Determine learning phase and context type
            if learning_phase is None:
                learning_phase = self._infer_learning_phase(context, performance_data)

            context_type = self._classify_context_type(context, performance_data, learning_phase)

            # Apply context-based weight adjustments to criteria
            adjusted_criteria = self._apply_context_adjustments(context, learning_phase)

            # Evaluate each fitness criteria
            individual_scores = {}
            for criteria_id, criteria in adjusted_criteria.items():
                if criteria.criteria_type.value in self.fitness_functions:
                    fitness_func = self.fitness_functions[criteria.criteria_type.value]
                    score = await fitness_func(context, performance_data, criteria)
                    individual_scores[criteria_id] = score
                else:
                    # Default scoring if no specific function available
                    individual_scores[criteria_id] = 0.5

            # Calculate composite fitness score
            composite_score = self._calculate_composite_fitness(individual_scores, adjusted_criteria)

            # Analyze performance indicators and predict improvement areas
            performance_indicators = self._analyze_performance_indicators(
                context, performance_data, individual_scores
            )
            improvement_areas = self._predict_improvement_areas(
                individual_scores, adjusted_criteria, performance_data
            )

            # Perform fitness trend analysis
            fitness_trend = self._analyze_fitness_trend(individual_scores, composite_score)

            # Create evaluation result
            evaluation = ContextualFitnessEvaluation(
                evaluation_id=evaluation_id,
                context_snapshot=context.copy(),
                fitness_criteria_used={cid: c.current_weight for cid, c in adjusted_criteria.items()},
                individual_scores=individual_scores,
                composite_fitness_score=composite_score,
                context_type=context_type,
                learning_phase=learning_phase,
                performance_indicators=performance_indicators,
                predicted_improvement_areas=improvement_areas,
                fitness_trend_analysis=fitness_trend,
                evaluation_confidence=self._calculate_evaluation_confidence(context, performance_data)
            )

            # Store evaluation in database
            await self._store_fitness_evaluation(game_id, session_id, evaluation)

            # Add to history
            self.fitness_evaluation_history.append(evaluation)

            # Check for evolution triggers
            await self._check_evolution_triggers(game_id, session_id, evaluation)

            logger.debug(f"Contextual fitness evaluation completed: {composite_score:.3f} (confidence: {evaluation.evaluation_confidence:.3f})")

            return evaluation

        except Exception as e:
            logger.error(f"Error in contextual fitness evaluation: {e}")
            # Return a default evaluation in case of error
            return ContextualFitnessEvaluation(
                evaluation_id=f"error_eval_{int(time.time() * 1000)}",
                context_snapshot={},
                fitness_criteria_used={},
                individual_scores={},
                composite_fitness_score=0.0,
                context_type=ContextType.EARLY_EXPLORATION,
                learning_phase=LearningPhase.INITIALIZATION,
                performance_indicators={},
                predicted_improvement_areas=[],
                fitness_trend_analysis={},
                evaluation_confidence=0.0
            )

    async def evolve_fitness_criteria(self,
                                    game_id: str,
                                    session_id: str,
                                    trigger: FitnessEvolutionTrigger) -> Dict[str, Any]:
        """
        Evolve fitness criteria based on a trigger event.
        """
        try:
            evolution_results = {
                "trigger_id": trigger.trigger_id,
                "adjustments_made": [],
                "criteria_affected": [],
                "performance_impact": {},
                "success": False
            }

            # Store current state for comparison
            pre_adaptation_context = {
                "criteria_weights": {cid: c.current_weight for cid, c in self.active_criteria.items()},
                "composite_fitness": self.fitness_evaluation_history[-1].composite_fitness_score if self.fitness_evaluation_history else 0.0
            }

            # Apply suggested adjustments
            for criteria_id in trigger.affected_criteria:
                if criteria_id in self.active_criteria:
                    criteria = self.active_criteria[criteria_id]

                    if criteria_id in trigger.suggested_adjustments:
                        adjustment = trigger.suggested_adjustments[criteria_id]
                        old_weight = criteria.current_weight

                        # Apply adjustment with bounds checking
                        new_weight = max(
                            self.config["min_criteria_weight"],
                            min(self.config["max_criteria_weight"], old_weight + adjustment)
                        )

                        # Only apply if change exceeds stability threshold
                        if abs(new_weight - old_weight) >= criteria.stability_threshold:
                            criteria.current_weight = new_weight
                            criteria.last_weight_update = time.time()

                            # Record evolution in history
                            evolution_record = {
                                "timestamp": time.time(),
                                "trigger_type": trigger.trigger_type,
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                                "adjustment": adjustment,
                                "trigger_strength": trigger.trigger_strength
                            }
                            criteria.evolution_history.append(evolution_record)

                            evolution_results["adjustments_made"].append({
                                "criteria_id": criteria_id,
                                "old_weight": old_weight,
                                "new_weight": new_weight,
                                "adjustment": adjustment
                            })
                            evolution_results["criteria_affected"].append(criteria_id)

                            logger.info(f"Evolved fitness criteria {criteria_id}: {old_weight:.3f} -> {new_weight:.3f}")

            # Store adaptation learning in database
            post_adaptation_context = {
                "criteria_weights": {cid: c.current_weight for cid, c in self.active_criteria.items()},
                "composite_fitness": 0.0  # Will be updated after next evaluation
            }

            await self._store_fitness_adaptation_learning(
                game_id, session_id, trigger, pre_adaptation_context, post_adaptation_context
            )

            evolution_results["success"] = len(evolution_results["adjustments_made"]) > 0

            return evolution_results

        except Exception as e:
            logger.error(f"Error evolving fitness criteria: {e}")
            return {"success": False, "error": str(e)}

    async def request_attention_allocation(self,
                                         game_id: str,
                                         session_id: str,
                                         fitness_priorities: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Request attention allocation based on current fitness priorities.
        """
        if not (self.attention_integration_enabled and self.attention_controller and
                ATTENTION_COORDINATION_AVAILABLE):
            return None

        try:
            # Analyze current fitness gaps to determine attention needs
            improvement_areas = self._identify_attention_demanding_areas(fitness_priorities)

            # Create subsystem demands based on fitness analysis
            subsystem_demands = []

            for area, priority in improvement_areas.items():
                demand = SubsystemDemand(
                    subsystem_name=f"fitness_evolution_{area}",
                    requested_priority=min(0.9, priority),
                    current_load=0.5,  # Moderate computational load
                    processing_complexity=0.6,  # Moderate complexity
                    urgency_level=min(5, int(priority * 5) + 1),
                    justification=f"Fitness evolution needs attention for {area} improvement",
                    context_data={
                        "fitness_area": area,
                        "priority_score": priority,
                        "improvement_needed": True
                    }
                )
                subsystem_demands.append(demand)

            # Request attention allocation
            attention_allocation = await self.attention_controller.allocate_attention_resources(
                game_id, subsystem_demands, {"fitness_evolution_request": True}
            )

            # Store integration record
            await self._store_fitness_attention_integration(
                game_id, session_id, fitness_priorities, attention_allocation
            )

            return attention_allocation.__dict__ if attention_allocation else None

        except Exception as e:
            logger.error(f"Error requesting attention allocation for fitness evolution: {e}")
            return None

    def _initialize_default_criteria(self):
        """Initialize default fitness criteria."""
        default_criteria = [
            {
                "type": FitnessCriteriaType.EXPLORATION_DEPTH,
                "name": "Exploration Depth",
                "definition": {"measure": "coordinate_coverage", "weight_factor": 1.0}
            },
            {
                "type": FitnessCriteriaType.PATTERN_DISCOVERY,
                "name": "Pattern Discovery",
                "definition": {"measure": "pattern_detection_rate", "weight_factor": 1.2}
            },
            {
                "type": FitnessCriteriaType.STRATEGY_INNOVATION,
                "name": "Strategy Innovation",
                "definition": {"measure": "novel_strategy_generation", "weight_factor": 0.8}
            },
            {
                "type": FitnessCriteriaType.LEARNING_EFFICIENCY,
                "name": "Learning Efficiency",
                "definition": {"measure": "improvement_per_action", "weight_factor": 1.5}
            },
            {
                "type": FitnessCriteriaType.ADAPTATION_SPEED,
                "name": "Adaptation Speed",
                "definition": {"measure": "context_adaptation_rate", "weight_factor": 1.0}
            }
        ]

        for criteria_def in default_criteria:
            criteria_id = f"default_{criteria_def['type'].value}_{int(time.time() * 1000)}"
            criteria = FitnessCriteria(
                criteria_id=criteria_id,
                criteria_type=criteria_def["type"],
                criteria_name=criteria_def["name"],
                criteria_definition=criteria_def["definition"],
                last_weight_update=time.time()
            )
            self.active_criteria[criteria_id] = criteria

    def _initialize_fitness_functions(self):
        """Initialize fitness evaluation functions for each criteria type."""
        self.fitness_functions = {
            FitnessCriteriaType.EXPLORATION_DEPTH.value: self._evaluate_exploration_depth,
            FitnessCriteriaType.PATTERN_DISCOVERY.value: self._evaluate_pattern_discovery,
            FitnessCriteriaType.STRATEGY_INNOVATION.value: self._evaluate_strategy_innovation,
            FitnessCriteriaType.LEARNING_EFFICIENCY.value: self._evaluate_learning_efficiency,
            FitnessCriteriaType.ADAPTATION_SPEED.value: self._evaluate_adaptation_speed,
            FitnessCriteriaType.GOAL_ACHIEVEMENT.value: self._evaluate_goal_achievement,
            FitnessCriteriaType.KNOWLEDGE_RETENTION.value: self._evaluate_knowledge_retention,
            FitnessCriteriaType.TRANSFER_LEARNING.value: self._evaluate_transfer_learning
        }

    async def _evaluate_exploration_depth(self, context: Dict[str, Any],
                                        performance_data: Dict[str, Any],
                                        criteria: FitnessCriteria) -> float:
        """Evaluate exploration depth fitness."""
        try:
            # Extract exploration metrics from context
            coordinates_tried = context.get("coordinates_tried", [])
            total_coordinates = context.get("total_possible_coordinates", 100)
            unique_actions = context.get("unique_actions_tried", set())

            # Calculate coverage metrics
            coordinate_coverage = len(set(coordinates_tried)) / max(1, total_coordinates)
            action_diversity = len(unique_actions) / max(1, 10)  # Assume 10 possible actions

            # Combine metrics with emphasis on systematic exploration
            exploration_score = (coordinate_coverage * 0.6 + action_diversity * 0.4)

            return min(1.0, exploration_score)

        except Exception as e:
            logger.error(f"Error evaluating exploration depth: {e}")
            return 0.5

    async def _evaluate_pattern_discovery(self, context: Dict[str, Any],
                                        performance_data: Dict[str, Any],
                                        criteria: FitnessCriteria) -> float:
        """Evaluate pattern discovery fitness."""
        try:
            patterns_discovered = performance_data.get("patterns_detected", 0)
            patterns_validated = performance_data.get("patterns_validated", 0)
            pattern_confidence = performance_data.get("avg_pattern_confidence", 0.0)

            # Rate of discovery normalized by actions taken
            actions_taken = context.get("action_count", 1)
            discovery_rate = patterns_discovered / max(1, actions_taken / 10)  # Per 10 actions

            # Quality of patterns (validation rate and confidence)
            pattern_quality = 0.5
            if patterns_discovered > 0:
                validation_rate = patterns_validated / patterns_discovered
                pattern_quality = (validation_rate * 0.7 + pattern_confidence * 0.3)

            # Combine rate and quality
            pattern_score = min(1.0, discovery_rate * 0.6 + pattern_quality * 0.4)

            return pattern_score

        except Exception as e:
            logger.error(f"Error evaluating pattern discovery: {e}")
            return 0.5

    async def _evaluate_strategy_innovation(self, context: Dict[str, Any],
                                          performance_data: Dict[str, Any],
                                          criteria: FitnessCriteria) -> float:
        """Evaluate strategy innovation fitness."""
        try:
            strategies_discovered = performance_data.get("strategies_discovered", 0)
            strategy_effectiveness = performance_data.get("avg_strategy_effectiveness", 0.0)
            novel_approaches = performance_data.get("novel_approaches_tried", 0)

            # Innovation rate
            actions_taken = context.get("action_count", 1)
            innovation_rate = (strategies_discovered + novel_approaches) / max(1, actions_taken / 20)

            # Effectiveness of innovations
            effectiveness_score = strategy_effectiveness

            # Novelty bonus (trying untested approaches)
            novelty_bonus = min(0.3, novel_approaches * 0.1)

            innovation_score = min(1.0, innovation_rate * 0.5 + effectiveness_score * 0.4 + novelty_bonus)

            return innovation_score

        except Exception as e:
            logger.error(f"Error evaluating strategy innovation: {e}")
            return 0.5

    async def _evaluate_learning_efficiency(self, context: Dict[str, Any],
                                          performance_data: Dict[str, Any],
                                          criteria: FitnessCriteria) -> float:
        """Evaluate learning efficiency fitness."""
        try:
            score_improvement = performance_data.get("total_score_improvement", 0.0)
            actions_taken = context.get("action_count", 1)
            learning_events = performance_data.get("learning_events_triggered", 0)

            # Improvement per action
            improvement_efficiency = score_improvement / max(1, actions_taken)

            # Learning event frequency
            learning_frequency = learning_events / max(1, actions_taken / 5)  # Per 5 actions

            # Knowledge retention (if available)
            retention_score = performance_data.get("knowledge_retention_rate", 0.5)

            efficiency_score = min(1.0, improvement_efficiency * 0.5 + learning_frequency * 0.3 + retention_score * 0.2)

            return efficiency_score

        except Exception as e:
            logger.error(f"Error evaluating learning efficiency: {e}")
            return 0.5

    async def _evaluate_adaptation_speed(self, context: Dict[str, Any],
                                       performance_data: Dict[str, Any],
                                       criteria: FitnessCriteria) -> float:
        """Evaluate adaptation speed fitness."""
        try:
            context_changes = context.get("context_changes_detected", 0)
            adaptation_responses = performance_data.get("adaptations_made", 0)
            adaptation_latency = performance_data.get("avg_adaptation_latency", 10.0)  # Actions

            # Response rate to context changes
            response_rate = adaptation_responses / max(1, context_changes) if context_changes > 0 else 0.5

            # Speed of adaptation (lower latency is better)
            speed_score = max(0.0, 1.0 - (adaptation_latency / 20.0))  # Normalize to 20 actions

            # Adaptation effectiveness
            adaptation_effectiveness = performance_data.get("adaptation_effectiveness", 0.5)

            adaptation_score = min(1.0, response_rate * 0.4 + speed_score * 0.3 + adaptation_effectiveness * 0.3)

            return adaptation_score

        except Exception as e:
            logger.error(f"Error evaluating adaptation speed: {e}")
            return 0.5

    async def _evaluate_goal_achievement(self, context: Dict[str, Any],
                                       performance_data: Dict[str, Any],
                                       criteria: FitnessCriteria) -> float:
        """Evaluate goal achievement fitness."""
        try:
            goals_achieved = performance_data.get("goals_achieved", 0)
            goals_attempted = performance_data.get("goals_attempted", 1)
            level_completions = performance_data.get("level_completions", 0)

            # Goal completion rate
            completion_rate = goals_achieved / max(1, goals_attempted)

            # Level progression
            level_score = min(1.0, level_completions * 0.2)

            # Overall score achievement
            current_score = context.get("current_score", 0.0)
            target_score = context.get("target_score", 100.0)
            score_achievement = min(1.0, current_score / max(1, target_score))

            goal_score = completion_rate * 0.5 + level_score * 0.3 + score_achievement * 0.2

            return goal_score

        except Exception as e:
            logger.error(f"Error evaluating goal achievement: {e}")
            return 0.5

    async def _evaluate_knowledge_retention(self, context: Dict[str, Any],
                                          performance_data: Dict[str, Any],
                                          criteria: FitnessCriteria) -> float:
        """Evaluate knowledge retention fitness."""
        try:
            # This would typically require tracking performance over time
            # For now, use available proxies

            pattern_reuse = performance_data.get("pattern_reuse_count", 0)
            strategy_reapplication = performance_data.get("strategy_reapplication_count", 0)
            performance_consistency = performance_data.get("performance_consistency", 0.5)

            # Retention indicators
            reuse_score = min(1.0, (pattern_reuse + strategy_reapplication) * 0.1)
            consistency_score = performance_consistency

            retention_score = reuse_score * 0.6 + consistency_score * 0.4

            return retention_score

        except Exception as e:
            logger.error(f"Error evaluating knowledge retention: {e}")
            return 0.5

    async def _evaluate_transfer_learning(self, context: Dict[str, Any],
                                        performance_data: Dict[str, Any],
                                        criteria: FitnessCriteria) -> float:
        """Evaluate transfer learning fitness."""
        try:
            cross_game_applications = performance_data.get("cross_game_applications", 0)
            generalization_success = performance_data.get("generalization_success_rate", 0.0)
            novel_context_performance = performance_data.get("novel_context_performance", 0.5)

            # Transfer application rate
            transfer_rate = min(1.0, cross_game_applications * 0.2)

            # Success of generalizations
            generalization_score = generalization_success

            # Performance in novel contexts
            novelty_score = novel_context_performance

            transfer_score = transfer_rate * 0.4 + generalization_score * 0.4 + novelty_score * 0.2

            return transfer_score

        except Exception as e:
            logger.error(f"Error evaluating transfer learning: {e}")
            return 0.5

    def _infer_learning_phase(self, context: Dict[str, Any], performance_data: Dict[str, Any]) -> LearningPhase:
        """Infer the current learning phase from context and performance data."""
        action_count = context.get("action_count", 0)
        patterns_discovered = performance_data.get("patterns_detected", 0)
        strategies_discovered = performance_data.get("strategies_discovered", 0)
        performance_stability = performance_data.get("performance_stability", 0.0)

        # Simple heuristic-based phase inference
        if action_count < 50:
            return LearningPhase.INITIALIZATION
        elif patterns_discovered < 3 or strategies_discovered < 2:
            return LearningPhase.EXPLORATION
        elif performance_stability < 0.7:
            return LearningPhase.EXPLOITATION
        elif performance_stability < 0.9:
            return LearningPhase.REFINEMENT
        else:
            return LearningPhase.MASTERY

    def _classify_context_type(self, context: Dict[str, Any],
                             performance_data: Dict[str, Any],
                             learning_phase: LearningPhase) -> ContextType:
        """Classify the type of learning context."""
        if learning_phase in [LearningPhase.INITIALIZATION, LearningPhase.EXPLORATION]:
            return ContextType.EARLY_EXPLORATION
        elif learning_phase == LearningPhase.EXPLOITATION:
            if performance_data.get("patterns_detected", 0) > 0:
                return ContextType.PATTERN_LEARNING
            else:
                return ContextType.EARLY_EXPLORATION
        elif learning_phase == LearningPhase.REFINEMENT:
            return ContextType.STRATEGY_OPTIMIZATION
        else:  # MASTERY
            return ContextType.MASTERY_PHASE

    def _apply_context_adjustments(self, context: Dict[str, Any],
                                 learning_phase: LearningPhase) -> Dict[str, FitnessCriteria]:
        """Apply context-based adjustments to fitness criteria weights."""
        adjusted_criteria = {}

        for criteria_id, criteria in self.active_criteria.items():
            # Create a copy with potentially adjusted weight
            adjusted_criteria[criteria_id] = FitnessCriteria(
                criteria_id=criteria.criteria_id,
                criteria_type=criteria.criteria_type,
                criteria_name=criteria.criteria_name,
                criteria_definition=criteria.criteria_definition.copy(),
                base_weight=criteria.base_weight,
                current_weight=criteria.current_weight,
                context_modifiers=criteria.context_modifiers.copy(),
                evolution_history=criteria.evolution_history.copy(),
                performance_correlation=criteria.performance_correlation,
                criteria_effectiveness=criteria.criteria_effectiveness,
                adaptation_rate=criteria.adaptation_rate,
                stability_threshold=criteria.stability_threshold,
                last_weight_update=criteria.last_weight_update
            )

            # Apply learning phase adjustments
            phase_modifier = self._get_phase_modifier(criteria.criteria_type, learning_phase)
            adjusted_criteria[criteria_id].current_weight *= phase_modifier

            # Apply context-specific modifiers if any
            if hasattr(criteria, 'context_modifiers') and criteria.context_modifiers:
                for context_key, modifier in criteria.context_modifiers.items():
                    if context_key in context:
                        adjusted_criteria[criteria_id].current_weight *= modifier

        return adjusted_criteria

    def _get_phase_modifier(self, criteria_type: FitnessCriteriaType, learning_phase: LearningPhase) -> float:
        """Get phase-specific modifier for a criteria type."""
        phase_modifiers = {
            LearningPhase.INITIALIZATION: {
                FitnessCriteriaType.EXPLORATION_DEPTH: 1.5,
                FitnessCriteriaType.PATTERN_DISCOVERY: 1.2,
                FitnessCriteriaType.STRATEGY_INNOVATION: 0.8,
                FitnessCriteriaType.LEARNING_EFFICIENCY: 1.0,
                FitnessCriteriaType.ADAPTATION_SPEED: 1.1
            },
            LearningPhase.EXPLORATION: {
                FitnessCriteriaType.EXPLORATION_DEPTH: 1.3,
                FitnessCriteriaType.PATTERN_DISCOVERY: 1.4,
                FitnessCriteriaType.STRATEGY_INNOVATION: 1.2,
                FitnessCriteriaType.LEARNING_EFFICIENCY: 1.1,
                FitnessCriteriaType.ADAPTATION_SPEED: 1.0
            },
            LearningPhase.EXPLOITATION: {
                FitnessCriteriaType.EXPLORATION_DEPTH: 0.8,
                FitnessCriteriaType.PATTERN_DISCOVERY: 1.1,
                FitnessCriteriaType.STRATEGY_INNOVATION: 1.3,
                FitnessCriteriaType.LEARNING_EFFICIENCY: 1.4,
                FitnessCriteriaType.ADAPTATION_SPEED: 1.2
            },
            LearningPhase.REFINEMENT: {
                FitnessCriteriaType.EXPLORATION_DEPTH: 0.6,
                FitnessCriteriaType.PATTERN_DISCOVERY: 0.9,
                FitnessCriteriaType.STRATEGY_INNOVATION: 1.1,
                FitnessCriteriaType.LEARNING_EFFICIENCY: 1.5,
                FitnessCriteriaType.ADAPTATION_SPEED: 1.3
            },
            LearningPhase.MASTERY: {
                FitnessCriteriaType.EXPLORATION_DEPTH: 0.5,
                FitnessCriteriaType.PATTERN_DISCOVERY: 0.8,
                FitnessCriteriaType.STRATEGY_INNOVATION: 0.9,
                FitnessCriteriaType.LEARNING_EFFICIENCY: 1.2,
                FitnessCriteriaType.ADAPTATION_SPEED: 1.4
            }
        }

        return phase_modifiers.get(learning_phase, {}).get(criteria_type, 1.0)

    def _calculate_composite_fitness(self, individual_scores: Dict[str, float],
                                   criteria: Dict[str, FitnessCriteria]) -> float:
        """Calculate weighted composite fitness score."""
        if not individual_scores or not criteria:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for criteria_id, score in individual_scores.items():
            if criteria_id in criteria:
                weight = criteria[criteria_id].current_weight
                weighted_sum += score * weight
                total_weight += weight

        return weighted_sum / max(1.0, total_weight)

    def _analyze_performance_indicators(self, context: Dict[str, Any],
                                      performance_data: Dict[str, Any],
                                      individual_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance indicators from current evaluation."""
        return {
            "avg_fitness_score": np.mean(list(individual_scores.values())) if individual_scores else 0.0,
            "fitness_variance": np.var(list(individual_scores.values())) if individual_scores else 0.0,
            "best_performing_criteria": max(individual_scores, key=individual_scores.get) if individual_scores else None,
            "worst_performing_criteria": min(individual_scores, key=individual_scores.get) if individual_scores else None,
            "performance_balance": 1.0 - (np.var(list(individual_scores.values())) if individual_scores else 0.0),
            "context_complexity": len(context),
            "performance_data_richness": len(performance_data)
        }

    def _predict_improvement_areas(self, individual_scores: Dict[str, float],
                                 criteria: Dict[str, FitnessCriteria],
                                 performance_data: Dict[str, Any]) -> List[str]:
        """Predict areas that need the most improvement."""
        improvement_areas = []

        # Find criteria with low scores
        for criteria_id, score in individual_scores.items():
            if score < 0.5 and criteria_id in criteria:
                improvement_areas.append(criteria[criteria_id].criteria_type.value)

        # Add areas based on performance data patterns
        if performance_data.get("patterns_detected", 0) < 2:
            improvement_areas.append("pattern_recognition")

        if performance_data.get("strategies_discovered", 0) < 1:
            improvement_areas.append("strategy_development")

        return improvement_areas[:5]  # Limit to top 5 areas

    def _analyze_fitness_trend(self, individual_scores: Dict[str, float],
                             composite_score: float) -> Dict[str, Any]:
        """Analyze fitness evolution trends."""
        trend_analysis = {
            "current_composite_score": composite_score,
            "score_distribution": individual_scores.copy(),
            "trend_direction": "stable",
            "improvement_velocity": 0.0,
            "volatility": 0.0
        }

        if len(self.fitness_evaluation_history) >= 2:
            recent_scores = [eval.composite_fitness_score for eval in self.fitness_evaluation_history[-5:]]

            # Calculate trend direction
            if len(recent_scores) >= 2:
                slope = (recent_scores[-1] - recent_scores[0]) / max(1, len(recent_scores) - 1)
                if slope > 0.02:
                    trend_analysis["trend_direction"] = "improving"
                elif slope < -0.02:
                    trend_analysis["trend_direction"] = "declining"

                trend_analysis["improvement_velocity"] = slope
                trend_analysis["volatility"] = np.std(recent_scores) if len(recent_scores) > 1 else 0.0

        return trend_analysis

    def _calculate_evaluation_confidence(self, context: Dict[str, Any],
                                       performance_data: Dict[str, Any]) -> float:
        """Calculate confidence in the fitness evaluation."""
        confidence_factors = []

        # Data completeness
        context_completeness = min(1.0, len(context) / 10.0)  # Assume 10 is good completeness
        performance_completeness = min(1.0, len(performance_data) / 8.0)  # Assume 8 is good completeness
        confidence_factors.extend([context_completeness, performance_completeness])

        # Historical data availability
        history_factor = min(1.0, len(self.fitness_evaluation_history) / 10.0)
        confidence_factors.append(history_factor)

        # Criteria coverage
        criteria_coverage = len(self.active_criteria) / self.config["max_criteria_count"]
        confidence_factors.append(criteria_coverage)

        return np.mean(confidence_factors)

    async def _check_evolution_triggers(self, game_id: str, session_id: str,
                                      evaluation: ContextualFitnessEvaluation):
        """Check if any evolution triggers should fire based on current evaluation."""
        if not self.config["automatic_adaptation_enabled"]:
            return

        try:
            # Performance plateau trigger
            if len(self.fitness_evaluation_history) >= 5:
                recent_scores = [eval.composite_fitness_score for eval in self.fitness_evaluation_history[-5:]]
                score_variance = np.var(recent_scores)

                if score_variance < 0.01:  # Very low variance indicates plateau
                    trigger = FitnessEvolutionTrigger(
                        trigger_id=f"plateau_{game_id}_{int(time.time() * 1000)}",
                        trigger_type="performance_plateau",
                        trigger_context={"recent_scores": recent_scores, "variance": score_variance},
                        affected_criteria=list(self.active_criteria.keys()),
                        suggested_adjustments=self._generate_plateau_adjustments(),
                        trigger_strength=0.8,
                        automatic_adjustment=True
                    )

                    await self._store_evolution_trigger(game_id, session_id, trigger)

                    if trigger.automatic_adjustment:
                        await self.evolve_fitness_criteria(game_id, session_id, trigger)

            # Low performance trigger
            if evaluation.composite_fitness_score < 0.3:
                low_performing_criteria = [
                    cid for cid, score in evaluation.individual_scores.items() if score < 0.4
                ]

                if low_performing_criteria:
                    trigger = FitnessEvolutionTrigger(
                        trigger_id=f"low_perf_{game_id}_{int(time.time() * 1000)}",
                        trigger_type="low_performance",
                        trigger_context={"composite_score": evaluation.composite_fitness_score},
                        affected_criteria=low_performing_criteria,
                        suggested_adjustments=self._generate_low_performance_adjustments(low_performing_criteria),
                        trigger_strength=0.9,
                        automatic_adjustment=True
                    )

                    await self._store_evolution_trigger(game_id, session_id, trigger)

                    if trigger.automatic_adjustment:
                        await self.evolve_fitness_criteria(game_id, session_id, trigger)

        except Exception as e:
            logger.error(f"Error checking evolution triggers: {e}")

    def _generate_plateau_adjustments(self) -> Dict[str, float]:
        """Generate adjustments to break out of performance plateau."""
        adjustments = {}

        # Slightly increase exploration and innovation criteria
        for criteria_id, criteria in self.active_criteria.items():
            if criteria.criteria_type in [FitnessCriteriaType.EXPLORATION_DEPTH,
                                        FitnessCriteriaType.STRATEGY_INNOVATION]:
                adjustments[criteria_id] = 0.1
            elif criteria.criteria_type == FitnessCriteriaType.ADAPTATION_SPEED:
                adjustments[criteria_id] = 0.15
            else:
                adjustments[criteria_id] = -0.05  # Slightly reduce other criteria

        return adjustments

    def _generate_low_performance_adjustments(self, low_performing_criteria: List[str]) -> Dict[str, float]:
        """Generate adjustments for low-performing criteria."""
        adjustments = {}

        for criteria_id in low_performing_criteria:
            # Increase weight for low-performing criteria
            adjustments[criteria_id] = 0.2

        # Slightly decrease weight for other criteria to maintain balance
        for criteria_id in self.active_criteria:
            if criteria_id not in low_performing_criteria:
                adjustments[criteria_id] = -0.05

        return adjustments

    def _identify_attention_demanding_areas(self, fitness_priorities: Dict[str, float]) -> Dict[str, float]:
        """Identify areas that need attention allocation based on fitness analysis."""
        attention_demands = {}

        # Find criteria with low performance but high importance
        for criteria_id, priority in fitness_priorities.items():
            if criteria_id in self.active_criteria:
                criteria = self.active_criteria[criteria_id]

                # High importance but potentially low performance
                if criteria.current_weight > 1.0 and priority < 0.5:
                    attention_demands[criteria.criteria_type.value] = criteria.current_weight * (1.0 - priority)

        # Normalize demands
        if attention_demands:
            max_demand = max(attention_demands.values())
            attention_demands = {k: v / max_demand for k, v in attention_demands.items()}

        return attention_demands

    async def _store_fitness_evaluation(self, game_id: str, session_id: str,
                                      evaluation: ContextualFitnessEvaluation):
        """Store fitness evaluation in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO contextual_fitness_evaluations
                   (evaluation_id, game_id, session_id, evaluation_timestamp, context_snapshot,
                    fitness_criteria_used, individual_scores, composite_fitness_score, context_type,
                    learning_phase, performance_indicators, predicted_improvement_areas,
                    fitness_trend_analysis, evaluation_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evaluation.evaluation_id, game_id, session_id, time.time(),
                 json.dumps(evaluation.context_snapshot), json.dumps(evaluation.fitness_criteria_used),
                 json.dumps(evaluation.individual_scores), evaluation.composite_fitness_score,
                 evaluation.context_type.value, evaluation.learning_phase.value,
                 json.dumps(evaluation.performance_indicators), json.dumps(evaluation.predicted_improvement_areas),
                 json.dumps(evaluation.fitness_trend_analysis), evaluation.evaluation_confidence)
            )
        except Exception as e:
            logger.error(f"Error storing fitness evaluation: {e}")

    async def _store_evolution_trigger(self, game_id: str, session_id: str,
                                     trigger: FitnessEvolutionTrigger):
        """Store evolution trigger in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO fitness_evolution_triggers
                   (trigger_id, game_id, session_id, trigger_type, trigger_timestamp, trigger_context,
                    affected_criteria, suggested_adjustments, trigger_strength, automatic_adjustment,
                    manual_review_required, adjustment_applied)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trigger.trigger_id, game_id, session_id, trigger.trigger_type, time.time(),
                 json.dumps(trigger.trigger_context), json.dumps(trigger.affected_criteria),
                 json.dumps(trigger.suggested_adjustments), trigger.trigger_strength,
                 trigger.automatic_adjustment, trigger.manual_review_required, False)
            )
        except Exception as e:
            logger.error(f"Error storing evolution trigger: {e}")

    async def _store_fitness_adaptation_learning(self, game_id: str, session_id: str,
                                               trigger: FitnessEvolutionTrigger,
                                               pre_context: Dict[str, Any],
                                               post_context: Dict[str, Any]):
        """Store fitness adaptation learning in database."""
        try:
            learning_id = f"adapt_learn_{game_id}_{int(time.time() * 1000)}"

            await self.db.execute_query(
                """INSERT INTO fitness_adaptation_learning
                   (learning_id, game_id, session_id, adaptation_timestamp, pre_adaptation_context,
                    post_adaptation_context, criteria_adjustments, performance_before, performance_after,
                    adaptation_type, improvement_detected, adaptation_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (learning_id, game_id, session_id, time.time(), json.dumps(pre_context),
                 json.dumps(post_context), json.dumps(trigger.suggested_adjustments),
                 json.dumps({}), json.dumps({}), trigger.trigger_type, False, 0.7)
            )
        except Exception as e:
            logger.error(f"Error storing fitness adaptation learning: {e}")

    async def _store_fitness_attention_integration(self, game_id: str, session_id: str,
                                                 fitness_priorities: Dict[str, float],
                                                 attention_allocation: Optional[Any]):
        """Store fitness-attention integration record in database."""
        try:
            integration_id = f"fitness_attn_{game_id}_{int(time.time() * 1000)}"

            allocation_received = {}
            if attention_allocation:
                allocation_received = {
                    "allocation_id": getattr(attention_allocation, 'allocation_id', ''),
                    "allocations": getattr(attention_allocation, 'allocations', {}),
                    "reasoning": getattr(attention_allocation, 'allocation_reasoning', '')
                }

            await self.db.execute_query(
                """INSERT INTO fitness_attention_integration
                   (integration_id, game_id, session_id, integration_timestamp,
                    current_fitness_priorities, attention_allocation_request, attention_allocation_received,
                    fitness_improvement_targets, integration_effectiveness)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (integration_id, game_id, session_id, time.time(),
                 json.dumps(fitness_priorities), json.dumps({"requested": True}),
                 json.dumps(allocation_received), json.dumps(self._identify_attention_demanding_areas(fitness_priorities)),
                 0.5)  # Default effectiveness, will be updated later
            )
        except Exception as e:
            logger.error(f"Error storing fitness attention integration: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics for the fitness evolution system."""
        return {
            "active_criteria_count": len(self.active_criteria),
            "fitness_evaluations_count": len(self.fitness_evaluation_history),
            "evolution_triggers_count": len(self.evolution_triggers),
            "adaptation_history_count": len(self.adaptation_history),
            "current_learning_phase": self.current_learning_phase.value,
            "attention_integration_enabled": self.attention_integration_enabled,
            "average_composite_fitness": np.mean([eval.composite_fitness_score for eval in self.fitness_evaluation_history]) if self.fitness_evaluation_history else 0.0,
            "fitness_trend": self._analyze_fitness_trend({}, 0.0) if self.fitness_evaluation_history else {},
            "config": self.config.copy()
        }