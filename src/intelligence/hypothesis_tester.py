"""
Experimental Framework for Hypothesis Testing with Action Reasoning

This module provides systematic testing of game strategy hypotheses with explicit reasoning
for each action taken. Tracks evidence, measures outcomes, and provides detailed explanations
for learning and adaptation.

Key Features:
- Systematic hypothesis testing with controlled experiments
- Explicit action reasoning and justification
- Evidence collection and confidence scoring
- Integration with Action6Coordinator for action execution
- Multi-level learning support (micro, meso, macro)
- Detailed logging and analysis of test results
"""

import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from datetime import datetime, timedelta
import threading
import time

# Disable pycache
import sys
sys.dont_write_bytecode = True

# Import hypothesis generator components
from .hypothesis_generator import Hypothesis, HypothesisType, HypothesisSource
from .game_pattern_analyzer import GameMechanic, GameMechanicsProfile

logger = logging.getLogger(__name__)

class TestOutcome(Enum):
    """Possible outcomes of hypothesis testing."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

class ActionReason(Enum):
    """Reasons for taking specific actions."""
    HYPOTHESIS_PREDICTION = "hypothesis_prediction"
    EVIDENCE_GATHERING = "evidence_gathering"
    PATTERN_EXPLORATION = "pattern_exploration"
    ERROR_RECOVERY = "error_recovery"
    LEARNING_OPTIMIZATION = "learning_optimization"
    DATABASE_GUIDANCE = "database_guidance"

@dataclass
class ActionRecord:
    """Record of a single action taken during testing."""
    action_id: str
    action_type: str
    coordinates: Tuple[int, int]
    timestamp: datetime
    reasoning: str
    reason_category: ActionReason
    hypothesis_context: str
    expected_outcome: str
    actual_outcome: str
    confidence_before: float
    confidence_after: float
    score_change: float
    success: bool
    evidence_collected: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Complete result of a hypothesis test experiment."""
    experiment_id: str
    hypothesis: Hypothesis
    outcome: TestOutcome
    actions_taken: List[ActionRecord]
    total_score_change: float
    test_duration: float
    evidence_collected: Dict[str, Any]
    confidence_progression: List[float]
    learning_insights: List[str]
    failure_reasons: List[str]
    success_factors: List[str]
    recommendations: List[str]
    timestamp: datetime

class HypothesisTester:
    """
    Systematic testing framework for game strategy hypotheses.

    Provides controlled experimental testing with detailed action reasoning,
    evidence collection, and learning insights.
    """

    def __init__(self, db_connection: Optional[sqlite3.Connection] = None,
                 action_executor: Optional[Callable] = None):
        self.db_connection = db_connection
        self.action_executor = action_executor  # Function to execute actions in game
        self.test_cache: Dict[str, ExperimentResult] = {}
        self.active_experiments: Dict[str, bool] = {}
        self.reasoning_templates = self._initialize_reasoning_templates()
        self.evidence_collectors = self._initialize_evidence_collectors()

        logger.info("Hypothesis Tester initialized")

    def _initialize_reasoning_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize templates for action reasoning based on different contexts."""
        return {
            HypothesisType.PATTERN_COMPLETION.value: {
                "coordinate_click": "Clicking coordinate ({x}, {y}) to complete detected pattern based on {evidence}. "
                                  "Expected outcome: {expected}. Confidence: {confidence:.1%}",
                "sequence_action": "Performing sequence action to extend pattern from {current} to {target}. "
                                 "Pattern analysis indicates {pattern_type} with {strength} strength.",
                "gap_filling": "Filling gap at ({x}, {y}) to maintain pattern symmetry. "
                             "Symmetry analysis shows {symmetry_type} pattern requiring completion."
            },
            HypothesisType.OBJECT_MANIPULATION.value: {
                "object_selection": "Selecting object at ({x}, {y}) for manipulation. Object identified as {object_type} "
                                  "with {boundary_clarity} boundaries. Movement potential: {movement_potential}",
                "object_movement": "Moving object from ({from_x}, {from_y}) to ({to_x}, {to_y}). "
                                 "Strategic reasoning: {movement_strategy}. Expected interaction: {interaction}",
                "transformation": "Applying transformation to object at ({x}, {y}). "
                                "Transformation type: {transform_type}. Expected result: {expected_result}"
            },
            HypothesisType.PHYSICS_SIMULATION.value: {
                "gravity_test": "Testing gravity effects by releasing object at ({x}, {y}). "
                              "Predicted trajectory: {trajectory}. Expected landing: ({land_x}, {land_y})",
                "collision_setup": "Setting up collision test between objects at ({x1}, {y1}) and ({x2}, {y2}). "
                                  "Collision angle: {angle}. Predicted outcome: {collision_result}",
                "momentum_action": "Applying momentum to object at ({x}, {y}) with force {force}. "
                                 "Expected movement: {movement_vector}. Interaction prediction: {interaction}"
            },
            HypothesisType.SPATIAL_REASONING.value: {
                "spatial_alignment": "Aligning object at ({x}, {y}) based on spatial relationship analysis. "
                                   "Reference points: {references}. Alignment type: {alignment_type}",
                "geometric_fitting": "Attempting geometric fit at ({x}, {y}). Shape analysis: {shape_data}. "
                                    "Fit probability: {fit_probability:.1%}. Alternative positions: {alternatives}",
                "rotation_test": "Testing rotation of object at ({x}, {y}) by {degrees} degrees. "
                               "Geometric analysis suggests {rotation_reason}. Expected fit: {fit_quality}"
            },
            HypothesisType.COLOR_MATCHING.value: {
                "color_grouping": "Grouping colors at ({x}, {y}) based on {color_strategy}. "
                                "Color analysis: {color_data}. Expected group: {target_group}",
                "pattern_matching": "Matching color pattern at ({x}, {y}) to reference pattern {pattern_ref}. "
                                  "Similarity score: {similarity:.1%}. Pattern completion: {completion_status}",
                "sequence_continuation": "Continuing color sequence by placing {color} at ({x}, {y}). "
                                      "Sequence pattern: {sequence_pattern}. Next predicted: {next_color}"
            },
            HypothesisType.NAVIGATION_PATH.value: {
                "pathfinding": "Finding path from ({start_x}, {start_y}) to ({end_x}, {end_y}). "
                             "Obstacles detected: {obstacles}. Optimal route: {route_description}",
                "waypoint_selection": "Selecting waypoint at ({x}, {y}) for path optimization. "
                                    "Path efficiency: {efficiency:.1%}. Alternative routes: {alternatives}",
                "obstacle_avoidance": "Avoiding obstacle at ({obs_x}, {obs_y}) by moving to ({x}, {y}). "
                                    "Obstacle type: {obstacle_type}. Avoidance strategy: {strategy}"
            }
        }

    def _initialize_evidence_collectors(self) -> Dict[str, Callable]:
        """Initialize evidence collection functions for different action types."""
        return {
            "screenshot_analysis": self._collect_screenshot_evidence,
            "score_tracking": self._collect_score_evidence,
            "pattern_changes": self._collect_pattern_evidence,
            "object_states": self._collect_object_evidence,
            "game_state": self._collect_game_state_evidence
        }

    def test_hypothesis(self, hypothesis: Hypothesis, game_context: Dict[str, Any],
                       max_actions: int = 10, timeout_seconds: float = 60.0) -> ExperimentResult:
        """
        Test a hypothesis through systematic experimentation.

        Args:
            hypothesis: The hypothesis to test
            game_context: Current game state and context information
            max_actions: Maximum number of actions to attempt
            timeout_seconds: Maximum time to spend testing

        Returns:
            ExperimentResult with detailed test outcomes and analysis
        """
        experiment_id = f"exp_{hypothesis.hypothesis_id}_{int(time.time())}"
        logger.info(f"Starting hypothesis test: {experiment_id}")

        # Mark experiment as active
        self.active_experiments[experiment_id] = True

        start_time = time.time()
        actions_taken = []
        confidence_progression = [hypothesis.confidence]
        evidence_collected = {}
        total_score_change = 0.0
        current_confidence = hypothesis.confidence

        try:
            # Initial evidence collection
            initial_evidence = self._collect_initial_evidence(game_context)
            evidence_collected.update(initial_evidence)

            # Execute hypothesis actions
            for action_index, (coord, expected_action) in enumerate(
                zip(hypothesis.predicted_coordinates[:max_actions],
                    hypothesis.expected_actions * max_actions)[:max_actions]
            ):

                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Test timeout reached for {experiment_id}")
                    break

                # Check if experiment was cancelled
                if not self.active_experiments.get(experiment_id, False):
                    logger.info(f"Test cancelled for {experiment_id}")
                    break

                # Generate action reasoning
                action_reasoning = self._generate_action_reasoning(
                    hypothesis, coord, expected_action, action_index, evidence_collected
                )

                # Execute action with reasoning
                action_result = self._execute_reasoned_action(
                    coord, expected_action, action_reasoning, hypothesis, game_context
                )

                # Record action
                actions_taken.append(action_result)

                # Update confidence based on action outcome
                confidence_update = self._update_confidence(
                    current_confidence, action_result, hypothesis
                )
                current_confidence = confidence_update
                confidence_progression.append(current_confidence)

                # Track score changes
                total_score_change += action_result.score_change

                # Collect evidence after action
                post_action_evidence = self._collect_post_action_evidence(
                    action_result, game_context
                )
                evidence_collected.update(post_action_evidence)

                # Early termination if clear success or failure
                if action_result.success and action_result.score_change > 50:
                    logger.info(f"Early success detected for {experiment_id}")
                    break
                elif action_result.score_change < -100:
                    logger.info(f"Early failure detected for {experiment_id}")
                    break

            # Determine overall outcome
            outcome = self._determine_experiment_outcome(actions_taken, total_score_change)

            # Generate learning insights
            learning_insights = self._generate_learning_insights(
                hypothesis, actions_taken, evidence_collected, outcome
            )

            # Generate failure/success analysis
            failure_reasons = self._analyze_failure_reasons(actions_taken, outcome)
            success_factors = self._analyze_success_factors(actions_taken, outcome)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                hypothesis, actions_taken, evidence_collected, outcome
            )

            # Create experiment result
            result = ExperimentResult(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                outcome=outcome,
                actions_taken=actions_taken,
                total_score_change=total_score_change,
                test_duration=time.time() - start_time,
                evidence_collected=evidence_collected,
                confidence_progression=confidence_progression,
                learning_insights=learning_insights,
                failure_reasons=failure_reasons,
                success_factors=success_factors,
                recommendations=recommendations,
                timestamp=datetime.now()
            )

            # Cache and save result
            self.test_cache[experiment_id] = result
            if self.db_connection:
                self._save_experiment_result(result)

            logger.info(f"Hypothesis test completed: {experiment_id}, Outcome: {outcome.value}, "
                       f"Score change: {total_score_change:.1f}")

            return result

        except Exception as e:
            logger.error(f"Error during hypothesis testing {experiment_id}: {e}")

            # Create error result
            return ExperimentResult(
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                outcome=TestOutcome.ERROR,
                actions_taken=actions_taken,
                total_score_change=total_score_change,
                test_duration=time.time() - start_time,
                evidence_collected=evidence_collected,
                confidence_progression=confidence_progression,
                learning_insights=[f"Test error: {str(e)}"],
                failure_reasons=[f"Technical error: {str(e)}"],
                success_factors=[],
                recommendations=["Retry test with error handling"],
                timestamp=datetime.now()
            )

        finally:
            # Mark experiment as inactive
            self.active_experiments[experiment_id] = False

    def _generate_action_reasoning(self, hypothesis: Hypothesis, coordinate: Tuple[int, int],
                                 action: str, action_index: int,
                                 evidence: Dict[str, Any]) -> str:
        """Generate detailed reasoning for why an action is being taken."""

        hypothesis_type = hypothesis.hypothesis_type.value
        x, y = coordinate

        # Get reasoning template for hypothesis type and action
        templates = self.reasoning_templates.get(hypothesis_type, {})

        # Choose appropriate template based on action type
        template_key = "coordinate_click"  # default
        if "move" in action.lower():
            template_key = "object_movement"
        elif "transform" in action.lower() or "rotate" in action.lower():
            template_key = "transformation"
        elif "sequence" in action.lower():
            template_key = "sequence_action"
        elif "path" in action.lower():
            template_key = "pathfinding"
        elif "group" in action.lower():
            template_key = "color_grouping"

        template = templates.get(template_key,
            "Taking action '{action}' at coordinate ({x}, {y}) based on hypothesis {hypothesis_type}. "
            "Action index: {action_index}. Confidence: {confidence:.1%}")

        # Prepare template variables
        template_vars = {
            "x": x,
            "y": y,
            "action": action,
            "hypothesis_type": hypothesis_type,
            "action_index": action_index,
            "confidence": hypothesis.confidence,
            "evidence": self._summarize_evidence(evidence),
            "expected": self._predict_action_outcome(hypothesis, coordinate, action),
            "pattern_type": self._identify_pattern_type(evidence),
            "strength": self._calculate_pattern_strength(evidence),
            "object_type": self._identify_object_type(evidence, coordinate),
            "boundary_clarity": self._assess_boundary_clarity(evidence, coordinate),
            "movement_potential": self._assess_movement_potential(evidence, coordinate)
        }

        try:
            return template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            return f"Taking action '{action}' at ({x}, {y}) for hypothesis {hypothesis_type} " \
                   f"(action {action_index+1}/{len(hypothesis.predicted_coordinates)})"

    def _execute_reasoned_action(self, coordinate: Tuple[int, int], action: str,
                               reasoning: str, hypothesis: Hypothesis,
                               game_context: Dict[str, Any]) -> ActionRecord:
        """Execute an action with full reasoning and evidence collection."""

        action_id = f"action_{int(time.time() * 1000)}"
        timestamp = datetime.now()

        logger.info(f"Executing action {action_id}: {reasoning}")

        # Collect pre-action evidence
        pre_evidence = self._collect_pre_action_evidence(coordinate, game_context)
        confidence_before = hypothesis.confidence

        # Execute the actual action
        try:
            if self.action_executor:
                # Use provided action executor (integration with game system)
                execution_result = self.action_executor(coordinate, action, reasoning)
                actual_outcome = execution_result.get("outcome", "unknown")
                score_change = execution_result.get("score_change", 0.0)
                success = execution_result.get("success", False)
            else:
                # Simulate action execution for testing
                actual_outcome, score_change, success = self._simulate_action_execution(
                    coordinate, action, hypothesis, game_context
                )

        except Exception as e:
            logger.error(f"Error executing action {action_id}: {e}")
            actual_outcome = f"error: {str(e)}"
            score_change = -10.0
            success = False

        # Collect post-action evidence
        post_evidence = self._collect_post_action_evidence_detailed(coordinate, game_context)

        # Calculate confidence after action
        confidence_after = self._calculate_post_action_confidence(
            confidence_before, success, score_change, hypothesis
        )

        # Determine reason category
        reason_category = self._categorize_action_reason(action, hypothesis, game_context)

        # Create action record
        action_record = ActionRecord(
            action_id=action_id,
            action_type=action,
            coordinates=coordinate,
            timestamp=timestamp,
            reasoning=reasoning,
            reason_category=reason_category,
            hypothesis_context=f"{hypothesis.hypothesis_type.value}:{hypothesis.description}",
            expected_outcome=self._predict_action_outcome(hypothesis, coordinate, action),
            actual_outcome=actual_outcome,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            score_change=score_change,
            success=success,
            evidence_collected={
                "pre_action": pre_evidence,
                "post_action": post_evidence,
                "execution_details": {
                    "coordinate": coordinate,
                    "action": action,
                    "timestamp": timestamp.isoformat()
                }
            }
        )

        logger.info(f"Action {action_id} completed: Success={success}, Score change={score_change:.1f}")

        return action_record

    def _simulate_action_execution(self, coordinate: Tuple[int, int], action: str,
                                 hypothesis: Hypothesis, game_context: Dict[str, Any]) -> Tuple[str, float, bool]:
        """Simulate action execution when no real executor is available."""

        # Simple simulation based on hypothesis confidence and action type
        base_success_prob = hypothesis.confidence

        # Adjust success probability based on action type
        action_modifiers = {
            "click": 0.0,
            "drag": -0.1,
            "move": -0.1,
            "rotate": -0.2,
            "transform": -0.2
        }

        modifier = action_modifiers.get(action.lower(), 0.0)
        success_prob = max(0.1, base_success_prob + modifier)

        # Random outcome based on probability
        success = np.random.random() < success_prob

        if success:
            outcome = f"successful_{action}_at_{coordinate}"
            score_change = np.random.uniform(10, 50)
        else:
            outcome = f"failed_{action}_at_{coordinate}"
            score_change = np.random.uniform(-20, -5)

        return outcome, score_change, success

    def _collect_initial_evidence(self, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect initial evidence before starting hypothesis test."""
        evidence = {
            "initial_screenshot": game_context.get("screenshot_data", {}),
            "initial_score": game_context.get("current_score", 0),
            "game_state": game_context.get("game_state", {}),
            "visible_objects": self._analyze_visible_objects(game_context),
            "grid_analysis": self._analyze_grid_state(game_context),
            "collection_timestamp": datetime.now().isoformat()
        }

        return evidence

    def _collect_pre_action_evidence(self, coordinate: Tuple[int, int],
                                   game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence immediately before taking an action."""
        return {
            "target_coordinate": coordinate,
            "surrounding_area": self._analyze_coordinate_area(coordinate, game_context),
            "cursor_position": game_context.get("cursor_position", (0, 0)),
            "pre_action_score": game_context.get("current_score", 0),
            "objects_near_target": self._find_objects_near_coordinate(coordinate, game_context),
            "timestamp": datetime.now().isoformat()
        }

    def _collect_post_action_evidence(self, action_result: ActionRecord,
                                    game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence after an action is taken."""
        return {
            "action_id": action_result.action_id,
            "post_action_score": game_context.get("current_score", 0),
            "score_delta": action_result.score_change,
            "visual_changes": self._detect_visual_changes(game_context),
            "object_state_changes": self._detect_object_changes(game_context),
            "new_objects_visible": self._detect_new_objects(game_context),
            "timestamp": datetime.now().isoformat()
        }

    def _collect_post_action_evidence_detailed(self, coordinate: Tuple[int, int],
                                             game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect detailed evidence after action execution."""
        return {
            "coordinate_state": self._analyze_coordinate_state(coordinate, game_context),
            "area_effects": self._analyze_area_effects(coordinate, game_context),
            "pattern_changes": self._detect_pattern_changes(game_context),
            "interaction_results": self._analyze_interaction_results(coordinate, game_context),
            "timestamp": datetime.now().isoformat()
        }

    def _update_confidence(self, current_confidence: float, action_result: ActionRecord,
                         hypothesis: Hypothesis) -> float:
        """Update confidence based on action result."""

        if action_result.success:
            # Boost confidence for successful actions
            confidence_boost = min(0.2, action_result.score_change / 100.0)
            new_confidence = min(1.0, current_confidence + confidence_boost)
        else:
            # Reduce confidence for failed actions
            confidence_reduction = min(0.3, abs(action_result.score_change) / 50.0)
            new_confidence = max(0.1, current_confidence - confidence_reduction)

        return new_confidence

    def _determine_experiment_outcome(self, actions: List[ActionRecord],
                                    total_score_change: float) -> TestOutcome:
        """Determine overall outcome of experiment based on actions and results."""

        if not actions:
            return TestOutcome.INCONCLUSIVE

        success_count = sum(1 for action in actions if action.success)
        success_rate = success_count / len(actions)

        # Determine outcome based on multiple factors
        if total_score_change > 100 and success_rate > 0.7:
            return TestOutcome.SUCCESS
        elif total_score_change > 20 and success_rate > 0.5:
            return TestOutcome.PARTIAL_SUCCESS
        elif total_score_change < -50 or success_rate < 0.2:
            return TestOutcome.FAILURE
        elif any("error" in action.actual_outcome for action in actions):
            return TestOutcome.ERROR
        else:
            return TestOutcome.INCONCLUSIVE

    def _generate_learning_insights(self, hypothesis: Hypothesis, actions: List[ActionRecord],
                                  evidence: Dict[str, Any], outcome: TestOutcome) -> List[str]:
        """Generate learning insights from experiment results."""
        insights = []

        # Analyze action patterns
        if len(actions) > 0:
            success_rate = sum(1 for a in actions if a.success) / len(actions)
            insights.append(f"Action success rate: {success_rate:.1%} across {len(actions)} actions")

            # Analyze confidence progression
            confidence_changes = [a.confidence_after - a.confidence_before for a in actions]
            avg_confidence_change = np.mean(confidence_changes)
            insights.append(f"Average confidence change per action: {avg_confidence_change:+.3f}")

        # Hypothesis-specific insights
        if hypothesis.hypothesis_type == HypothesisType.PATTERN_COMPLETION:
            insights.append(f"Pattern completion hypothesis {outcome.value}. "
                          f"Pattern recognition accuracy affects success rate.")
        elif hypothesis.hypothesis_type == HypothesisType.OBJECT_MANIPULATION:
            insights.append(f"Object manipulation approach {outcome.value}. "
                          f"Object boundary detection is critical for success.")

        # Evidence-based insights
        if "pattern_changes" in evidence:
            insights.append("Pattern changes detected during testing indicate dynamic game state")

        # Outcome-specific insights
        if outcome == TestOutcome.SUCCESS:
            insights.append("Hypothesis validated successfully. Strategy can be reused for similar games.")
        elif outcome == TestOutcome.FAILURE:
            insights.append("Hypothesis failed. Alternative approaches should be explored.")
        elif outcome == TestOutcome.PARTIAL_SUCCESS:
            insights.append("Hypothesis partially successful. Refinement may improve performance.")

        return insights

    def _analyze_failure_reasons(self, actions: List[ActionRecord],
                               outcome: TestOutcome) -> List[str]:
        """Analyze reasons for test failure."""
        reasons = []

        if outcome in [TestOutcome.FAILURE, TestOutcome.ERROR]:
            # Analyze action failures
            failed_actions = [a for a in actions if not a.success]
            if failed_actions:
                failure_rate = len(failed_actions) / len(actions)
                reasons.append(f"High action failure rate: {failure_rate:.1%}")

                # Common failure patterns
                common_failures = {}
                for action in failed_actions:
                    failure_type = action.actual_outcome.split("_")[0] if "_" in action.actual_outcome else "unknown"
                    common_failures[failure_type] = common_failures.get(failure_type, 0) + 1

                for failure_type, count in common_failures.items():
                    if count > 1:
                        reasons.append(f"Repeated {failure_type} failures ({count} times)")

            # Low score performance
            negative_scores = [a for a in actions if a.score_change < 0]
            if len(negative_scores) > len(actions) / 2:
                reasons.append("Majority of actions resulted in negative score changes")

            # Confidence degradation
            confidence_drops = [a for a in actions if a.confidence_after < a.confidence_before]
            if len(confidence_drops) > len(actions) / 2:
                reasons.append("Confidence consistently decreased during testing")

        return reasons

    def _analyze_success_factors(self, actions: List[ActionRecord],
                               outcome: TestOutcome) -> List[str]:
        """Analyze factors contributing to test success."""
        factors = []

        if outcome in [TestOutcome.SUCCESS, TestOutcome.PARTIAL_SUCCESS]:
            # High-performing actions
            successful_actions = [a for a in actions if a.success and a.score_change > 0]
            if successful_actions:
                success_rate = len(successful_actions) / len(actions)
                factors.append(f"High action success rate: {success_rate:.1%}")

                # Analyze successful action types
                successful_types = {}
                for action in successful_actions:
                    action_type = action.action_type
                    successful_types[action_type] = successful_types.get(action_type, 0) + 1

                for action_type, count in successful_types.items():
                    if count > 1:
                        factors.append(f"Effective action type: {action_type} (succeeded {count} times)")

            # Score performance
            high_scoring = [a for a in actions if a.score_change > 20]
            if high_scoring:
                factors.append(f"High-scoring actions: {len(high_scoring)} actions with significant positive impact")

            # Confidence building
            confidence_gains = [a for a in actions if a.confidence_after > a.confidence_before]
            if len(confidence_gains) > len(actions) / 2:
                factors.append("Confidence consistently increased during testing")

        return factors

    def _generate_recommendations(self, hypothesis: Hypothesis, actions: List[ActionRecord],
                                evidence: Dict[str, Any], outcome: TestOutcome) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        if outcome == TestOutcome.SUCCESS:
            recommendations.append("Reuse this hypothesis for similar game types and patterns")
            recommendations.append("Store successful action sequences for future reference")

        elif outcome == TestOutcome.PARTIAL_SUCCESS:
            recommendations.append("Refine hypothesis parameters to improve success rate")
            recommendations.append("Focus on successful action patterns and avoid failed approaches")

        elif outcome == TestOutcome.FAILURE:
            recommendations.append("Explore alternative hypothesis types for this game")
            recommendations.append("Analyze failed actions to understand game mechanics better")

        elif outcome == TestOutcome.INCONCLUSIVE:
            recommendations.append("Extend testing with more actions or different conditions")
            recommendations.append("Gather additional evidence before making conclusions")

        # Action-specific recommendations
        if len(actions) > 0:
            avg_score = np.mean([a.score_change for a in actions])
            if avg_score < 0:
                recommendations.append("Consider more conservative action selection")
            elif avg_score > 20:
                recommendations.append("Action selection strategy is effective, maintain approach")

        # Hypothesis-type-specific recommendations
        if hypothesis.hypothesis_type == HypothesisType.PATTERN_COMPLETION:
            recommendations.append("Improve pattern recognition accuracy for better results")
        elif hypothesis.hypothesis_type == HypothesisType.OBJECT_MANIPULATION:
            recommendations.append("Enhance object boundary detection capabilities")

        return recommendations

    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to database."""
        if not self.db_connection:
            return

        try:
            # Prepare data for database storage
            actions_data = [asdict(action) for action in result.actions_taken]

            # Convert datetime objects to ISO strings for JSON serialization
            for action_data in actions_data:
                action_data['timestamp'] = action_data['timestamp'].isoformat()

            experiment_data = {
                "hypothesis_id": result.hypothesis.hypothesis_id,
                "hypothesis_type": result.hypothesis.hypothesis_type.value,
                "outcome": result.outcome.value,
                "actions_taken": actions_data,
                "total_score_change": result.total_score_change,
                "test_duration": result.test_duration,
                "evidence_collected": result.evidence_collected,
                "confidence_progression": result.confidence_progression,
                "learning_insights": result.learning_insights,
                "failure_reasons": result.failure_reasons,
                "success_factors": result.success_factors,
                "recommendations": result.recommendations
            }

            self.db_connection.execute("""
                INSERT INTO hypothesis_test_results
                (experiment_id, hypothesis_id, outcome, experiment_data,
                 total_score_change, test_duration, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id,
                result.hypothesis.hypothesis_id,
                result.outcome.value,
                json.dumps(experiment_data),
                result.total_score_change,
                result.test_duration,
                result.timestamp.isoformat()
            ))

            # Save individual action records for detailed analysis
            for action in result.actions_taken:
                self.db_connection.execute("""
                    INSERT INTO action_reasoning_log
                    (action_id, experiment_id, hypothesis_id, coordinate_x, coordinate_y,
                     action_type, reasoning, reason_category, expected_outcome, actual_outcome,
                     confidence_before, confidence_after, score_change, success, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    action.action_id,
                    result.experiment_id,
                    result.hypothesis.hypothesis_id,
                    action.coordinates[0],
                    action.coordinates[1],
                    action.action_type,
                    action.reasoning,
                    action.reason_category.value,
                    action.expected_outcome,
                    action.actual_outcome,
                    action.confidence_before,
                    action.confidence_after,
                    action.score_change,
                    action.success,
                    action.timestamp.isoformat()
                ))

            self.db_connection.commit()
            logger.info(f"Saved experiment result {result.experiment_id} to database")

        except Exception as e:
            logger.error(f"Error saving experiment result to database: {e}")

    # Helper methods for evidence collection and analysis
    def _collect_screenshot_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from screenshot analysis."""
        return {"screenshot_timestamp": datetime.now().isoformat()}

    def _collect_score_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from score tracking."""
        return {"score": context.get("current_score", 0)}

    def _collect_pattern_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from pattern analysis."""
        return {"patterns_detected": context.get("patterns", [])}

    def _collect_object_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from object state analysis."""
        return {"objects": context.get("objects", [])}

    def _collect_game_state_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evidence from overall game state."""
        return {"game_state": context.get("game_state", {})}

    def _summarize_evidence(self, evidence: Dict[str, Any]) -> str:
        """Create a summary of collected evidence."""
        if not evidence:
            return "no evidence available"

        summary_parts = []
        if "patterns_detected" in evidence:
            summary_parts.append(f"{len(evidence['patterns_detected'])} patterns")
        if "objects" in evidence:
            summary_parts.append(f"{len(evidence['objects'])} objects")
        if "score" in evidence:
            summary_parts.append(f"score: {evidence['score']}")

        return ", ".join(summary_parts) if summary_parts else "general evidence"

    def _predict_action_outcome(self, hypothesis: Hypothesis, coordinate: Tuple[int, int], action: str) -> str:
        """Predict the expected outcome of an action."""
        return f"Expected positive interaction at {coordinate} for {hypothesis.hypothesis_type.value}"

    def _identify_pattern_type(self, evidence: Dict[str, Any]) -> str:
        """Identify the primary pattern type from evidence."""
        return evidence.get("primary_pattern", "unknown pattern")

    def _calculate_pattern_strength(self, evidence: Dict[str, Any]) -> str:
        """Calculate pattern strength description."""
        strength = evidence.get("pattern_strength", 0.5)
        if strength > 0.8:
            return "strong"
        elif strength > 0.5:
            return "moderate"
        else:
            return "weak"

    def _identify_object_type(self, evidence: Dict[str, Any], coordinate: Tuple[int, int]) -> str:
        """Identify object type at coordinate."""
        return evidence.get("object_type", "unknown object")

    def _assess_boundary_clarity(self, evidence: Dict[str, Any], coordinate: Tuple[int, int]) -> str:
        """Assess clarity of object boundaries."""
        clarity = evidence.get("boundary_clarity", 0.5)
        return "clear" if clarity > 0.7 else "unclear"

    def _assess_movement_potential(self, evidence: Dict[str, Any], coordinate: Tuple[int, int]) -> str:
        """Assess movement potential of object."""
        potential = evidence.get("movement_potential", 0.5)
        return "high" if potential > 0.7 else "low"

    def _categorize_action_reason(self, action: str, hypothesis: Hypothesis, context: Dict[str, Any]) -> ActionReason:
        """Categorize the reason for taking an action."""
        if "explore" in action.lower():
            return ActionReason.PATTERN_EXPLORATION
        elif hypothesis.source.value == "database_retrieval":
            return ActionReason.DATABASE_GUIDANCE
        elif "error" in action.lower():
            return ActionReason.ERROR_RECOVERY
        else:
            return ActionReason.HYPOTHESIS_PREDICTION

    def _calculate_post_action_confidence(self, confidence_before: float, success: bool,
                                        score_change: float, hypothesis: Hypothesis) -> float:
        """Calculate confidence after action execution."""
        if success and score_change > 0:
            boost = min(0.2, score_change / 100.0)
            return min(1.0, confidence_before + boost)
        elif not success:
            reduction = min(0.3, abs(score_change) / 50.0)
            return max(0.1, confidence_before - reduction)
        else:
            return confidence_before

    def _analyze_visible_objects(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze visible objects in the game."""
        return context.get("visible_objects", [])

    def _analyze_grid_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current grid state."""
        return context.get("grid_state", {})

    def _analyze_coordinate_area(self, coordinate: Tuple[int, int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze area around specific coordinate."""
        return {"coordinate": coordinate, "analysis": "area_analysis_placeholder"}

    def _find_objects_near_coordinate(self, coordinate: Tuple[int, int], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find objects near a specific coordinate."""
        return []  # Placeholder

    def _detect_visual_changes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect visual changes in the game."""
        return {"changes_detected": False}

    def _detect_object_changes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in object states."""
        return {"object_changes": []}

    def _detect_new_objects(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect newly visible objects."""
        return []

    def _analyze_coordinate_state(self, coordinate: Tuple[int, int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze state at specific coordinate."""
        return {"coordinate": coordinate, "state": "analysis_placeholder"}

    def _analyze_area_effects(self, coordinate: Tuple[int, int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effects in area around coordinate."""
        return {"effects": []}

    def _detect_pattern_changes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in patterns."""
        return {"pattern_changes": []}

    def _analyze_interaction_results(self, coordinate: Tuple[int, int], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results of interaction at coordinate."""
        return {"interaction_results": []}

# Module functions
def create_hypothesis_tester(db_connection: Optional[sqlite3.Connection] = None,
                           action_executor: Optional[Callable] = None) -> HypothesisTester:
    """Factory function to create a HypothesisTester instance."""
    return HypothesisTester(db_connection, action_executor)

# Singleton instance for global use
_tester_instance = None

def get_hypothesis_tester(db_connection: Optional[sqlite3.Connection] = None,
                        action_executor: Optional[Callable] = None) -> HypothesisTester:
    """Get the singleton HypothesisTester instance."""
    global _tester_instance
    if _tester_instance is None:
        _tester_instance = create_hypothesis_tester(db_connection, action_executor)
    return _tester_instance