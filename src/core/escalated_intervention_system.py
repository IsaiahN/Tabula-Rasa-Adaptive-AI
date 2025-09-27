"""
Escalated Intervention System

This module implements escalating intervention strategies to help break losing streaks
by applying increasingly aggressive techniques when standard approaches fail.
"""

import json
import time
import random
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .losing_streak_detector import EscalationLevel, LosingStreakData
from .anti_pattern_learner import AntiPatternLearner

logger = logging.getLogger(__name__)

class InterventionType(Enum):
    """Types of interventions that can be applied."""
    RANDOMIZATION = "randomization"              # Inject randomness into decision making
    EXPLORATION_BOOST = "exploration_boost"     # Increase exploration factor
    STRATEGY_OVERRIDE = "strategy_override"     # Override normal strategy selection
    PATTERN_AVOIDANCE = "pattern_avoidance"     # Actively avoid known anti-patterns
    COORDINATE_DIVERSIFICATION = "coordinate_diversification"  # Force coordinate diversity
    ACTION_SEQUENCE_SHUFFLE = "action_sequence_shuffle"        # Randomize action sequences
    TIMING_ADJUSTMENT = "timing_adjustment"     # Modify action timing
    COMPLETE_RESET = "complete_reset"           # Reset all learned preferences

@dataclass
class InterventionConfig:
    """Configuration for a specific intervention type."""
    intervention_type: InterventionType
    intensity: float  # 0.0 to 1.0
    duration_actions: int  # How many actions to apply intervention
    success_threshold: float  # What constitutes success for this intervention
    cooldown_seconds: int  # Minimum time between applications

@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    intervention_id: str
    intervention_type: InterventionType
    applied_timestamp: float
    success: bool
    outcome_data: Dict[str, Any]
    duration_seconds: float
    recovery_actions: int

class EscalatedInterventionSystem:
    """
    Applies escalating intervention strategies to break losing streaks.

    This system coordinates with the LosingStreakDetector and AntiPatternLearner
    to apply increasingly aggressive interventions when standard approaches fail.
    """

    def __init__(self, db_connection, anti_pattern_learner: AntiPatternLearner):
        """
        Initialize the escalated intervention system.

        Args:
            db_connection: Database connection for persistence
            anti_pattern_learner: Anti-pattern learning system for avoidance strategies
        """
        self.db = db_connection
        self.anti_pattern_learner = anti_pattern_learner

        # Active interventions tracking
        self.active_interventions: Dict[str, Dict[str, Any]] = {}

        # Intervention configurations for different escalation levels
        self.intervention_configs = self._initialize_intervention_configs()

        # Success tracking for intervention effectiveness
        self.intervention_success_rates: Dict[InterventionType, float] = {}

    def _initialize_intervention_configs(self) -> Dict[EscalationLevel, List[InterventionConfig]]:
        """Initialize intervention configurations for each escalation level."""
        return {
            EscalationLevel.MILD: [
                InterventionConfig(
                    intervention_type=InterventionType.RANDOMIZATION,
                    intensity=0.3,
                    duration_actions=10,
                    success_threshold=0.1,  # Any progress
                    cooldown_seconds=120
                ),
                InterventionConfig(
                    intervention_type=InterventionType.EXPLORATION_BOOST,
                    intensity=0.4,
                    duration_actions=15,
                    success_threshold=0.15,
                    cooldown_seconds=180
                )
            ],
            EscalationLevel.MODERATE: [
                InterventionConfig(
                    intervention_type=InterventionType.PATTERN_AVOIDANCE,
                    intensity=0.6,
                    duration_actions=20,
                    success_threshold=0.2,
                    cooldown_seconds=240
                ),
                InterventionConfig(
                    intervention_type=InterventionType.COORDINATE_DIVERSIFICATION,
                    intensity=0.7,
                    duration_actions=25,
                    success_threshold=0.25,
                    cooldown_seconds=300
                ),
                InterventionConfig(
                    intervention_type=InterventionType.ACTION_SEQUENCE_SHUFFLE,
                    intensity=0.5,
                    duration_actions=15,
                    success_threshold=0.2,
                    cooldown_seconds=200
                )
            ],
            EscalationLevel.AGGRESSIVE: [
                InterventionConfig(
                    intervention_type=InterventionType.STRATEGY_OVERRIDE,
                    intensity=0.8,
                    duration_actions=30,
                    success_threshold=0.3,
                    cooldown_seconds=360
                ),
                InterventionConfig(
                    intervention_type=InterventionType.TIMING_ADJUSTMENT,
                    intensity=0.7,
                    duration_actions=20,
                    success_threshold=0.25,
                    cooldown_seconds=240
                ),
                InterventionConfig(
                    intervention_type=InterventionType.COMPLETE_RESET,
                    intensity=1.0,
                    duration_actions=50,
                    success_threshold=0.4,
                    cooldown_seconds=600
                )
            ]
        }

    def apply_intervention(self,
                          streak_data: LosingStreakData,
                          game_context: Dict[str, Any]) -> Optional[InterventionResult]:
        """
        Apply an appropriate intervention for the given losing streak.

        Args:
            streak_data: Information about the losing streak
            game_context: Current game context and state

        Returns:
            Intervention result if applied, None if no intervention needed
        """
        try:
            # Get available interventions for this escalation level
            available_configs = self.intervention_configs.get(streak_data.escalation_level, [])

            if not available_configs:
                logger.warning(f"No interventions configured for escalation level {streak_data.escalation_level}")
                return None

            # Filter interventions based on cooldown and success rates
            viable_configs = self._filter_viable_interventions(available_configs, streak_data, game_context)

            if not viable_configs:
                logger.info(f"No viable interventions available for streak {streak_data.streak_id}")
                return None

            # Select the most appropriate intervention
            selected_config = self._select_intervention(viable_configs, streak_data, game_context)

            # Apply the intervention
            intervention_result = self._execute_intervention(selected_config, streak_data, game_context)

            if intervention_result:
                # Record the intervention
                self._record_intervention(intervention_result, streak_data)
                logger.info(f"Applied {selected_config.intervention_type.value} intervention for streak {streak_data.streak_id}")

            return intervention_result

        except Exception as e:
            logger.error(f"Error applying intervention: {e}")
            return None

    def _filter_viable_interventions(self,
                                   configs: List[InterventionConfig],
                                   streak_data: LosingStreakData,
                                   game_context: Dict[str, Any]) -> List[InterventionConfig]:
        """Filter intervention configurations based on viability criteria."""
        viable = []
        current_time = time.time()

        for config in configs:
            # Check cooldown
            if self._is_intervention_on_cooldown(config.intervention_type, current_time):
                continue

            # Check success rate threshold
            success_rate = self.intervention_success_rates.get(config.intervention_type, 0.5)
            if success_rate < 0.2:  # Don't use interventions with very low success rates
                continue

            # Check if intervention is applicable to current context
            if self._is_intervention_applicable(config, game_context):
                viable.append(config)

        return viable

    def _is_intervention_on_cooldown(self, intervention_type: InterventionType, current_time: float) -> bool:
        """Check if an intervention type is on cooldown."""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT MAX(applied_timestamp)
                FROM escalated_interventions
                WHERE intervention_type = ?
            """, (intervention_type.value,))

            result = cursor.fetchone()
            if result and result[0]:
                last_applied = result[0]
                config = next((c for configs in self.intervention_configs.values()
                             for c in configs if c.intervention_type == intervention_type), None)
                if config:
                    return current_time - last_applied < config.cooldown_seconds

            return False

        except Exception as e:
            logger.error(f"Error checking intervention cooldown: {e}")
            return False

    def _is_intervention_applicable(self, config: InterventionConfig, game_context: Dict[str, Any]) -> bool:
        """Check if an intervention is applicable to the current game context."""
        # Pattern avoidance requires known anti-patterns
        if config.intervention_type == InterventionType.PATTERN_AVOIDANCE:
            return len(self.anti_pattern_learner.known_patterns) > 0

        # Coordinate diversification requires coordinate-based game
        if config.intervention_type == InterventionType.COORDINATE_DIVERSIFICATION:
            return game_context.get("has_coordinates", False)

        # Action sequence shuffle requires action history
        if config.intervention_type == InterventionType.ACTION_SEQUENCE_SHUFFLE:
            return len(game_context.get("recent_actions", [])) >= 3

        # Most interventions are generally applicable
        return True

    def _select_intervention(self,
                           viable_configs: List[InterventionConfig],
                           streak_data: LosingStreakData,
                           game_context: Dict[str, Any]) -> InterventionConfig:
        """Select the most appropriate intervention from viable options."""
        # Weight interventions by their historical success rate and intensity
        weights = []
        for config in viable_configs:
            success_rate = self.intervention_success_rates.get(config.intervention_type, 0.5)
            # Higher weight for higher success rate and appropriate intensity for streak severity
            weight = success_rate * (1.0 + config.intensity * streak_data.consecutive_failures / 10.0)
            weights.append(weight)

        # Select intervention using weighted random choice
        if weights:
            total_weight = sum(weights)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for i, weight in enumerate(weights):
                cumulative += weight
                if r <= cumulative:
                    return viable_configs[i]

        # Fallback to first viable intervention
        return viable_configs[0]

    def _execute_intervention(self,
                            config: InterventionConfig,
                            streak_data: LosingStreakData,
                            game_context: Dict[str, Any]) -> Optional[InterventionResult]:
        """Execute the specified intervention."""
        intervention_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            outcome_data = {}

            if config.intervention_type == InterventionType.RANDOMIZATION:
                outcome_data = self._apply_randomization(config, game_context)

            elif config.intervention_type == InterventionType.EXPLORATION_BOOST:
                outcome_data = self._apply_exploration_boost(config, game_context)

            elif config.intervention_type == InterventionType.PATTERN_AVOIDANCE:
                outcome_data = self._apply_pattern_avoidance(config, game_context)

            elif config.intervention_type == InterventionType.COORDINATE_DIVERSIFICATION:
                outcome_data = self._apply_coordinate_diversification(config, game_context)

            elif config.intervention_type == InterventionType.ACTION_SEQUENCE_SHUFFLE:
                outcome_data = self._apply_action_sequence_shuffle(config, game_context)

            elif config.intervention_type == InterventionType.STRATEGY_OVERRIDE:
                outcome_data = self._apply_strategy_override(config, game_context)

            elif config.intervention_type == InterventionType.TIMING_ADJUSTMENT:
                outcome_data = self._apply_timing_adjustment(config, game_context)

            elif config.intervention_type == InterventionType.COMPLETE_RESET:
                outcome_data = self._apply_complete_reset(config, game_context)

            # Store active intervention for monitoring
            self.active_interventions[intervention_id] = {
                "config": config,
                "streak_data": streak_data,
                "start_time": start_time,
                "remaining_actions": config.duration_actions,
                "outcome_data": outcome_data
            }

            return InterventionResult(
                intervention_id=intervention_id,
                intervention_type=config.intervention_type,
                applied_timestamp=start_time,
                success=False,  # Will be determined later
                outcome_data=outcome_data,
                duration_seconds=0,  # Will be updated when intervention completes
                recovery_actions=0
            )

        except Exception as e:
            logger.error(f"Error executing intervention {config.intervention_type.value}: {e}")
            return None

    def _apply_randomization(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply randomization intervention."""
        return {
            "type": "randomization",
            "intensity": config.intensity,
            "randomization_factor": config.intensity,
            "affect_action_selection": True,
            "affect_coordinate_selection": True,
            "description": f"Inject {config.intensity:.1%} randomness into decision making"
        }

    def _apply_exploration_boost(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exploration boost intervention."""
        return {
            "type": "exploration_boost",
            "intensity": config.intensity,
            "exploration_multiplier": 1.0 + config.intensity,
            "encourage_new_coordinates": True,
            "encourage_new_actions": True,
            "description": f"Boost exploration factor by {config.intensity:.1%}"
        }

    def _apply_pattern_avoidance(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pattern avoidance intervention."""
        # Get anti-patterns for current game
        game_type = game_context.get("game_type", "unknown")
        high_risk_patterns = [
            pattern for pattern in self.anti_pattern_learner.known_patterns.values()
            if pattern.game_type == game_type and pattern.failure_rate >= 0.7
        ]

        return {
            "type": "pattern_avoidance",
            "intensity": config.intensity,
            "patterns_to_avoid": len(high_risk_patterns),
            "avoidance_strength": config.intensity,
            "use_alternative_suggestions": True,
            "description": f"Actively avoid {len(high_risk_patterns)} known anti-patterns"
        }

    def _apply_coordinate_diversification(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply coordinate diversification intervention."""
        # Generate diverse coordinate suggestions
        screen_regions = [
            (160, 120), (480, 120),  # Top corners
            (160, 360), (480, 360),  # Bottom corners
            (320, 240),              # Center
            (80, 240), (560, 240),   # Left/right edges
            (320, 80), (320, 400)    # Top/bottom edges
        ]

        return {
            "type": "coordinate_diversification",
            "intensity": config.intensity,
            "force_diverse_coordinates": True,
            "suggested_regions": screen_regions,
            "minimum_distance": int(50 * config.intensity),
            "description": f"Force coordinate diversity with {int(50 * config.intensity)}px minimum distance"
        }

    def _apply_action_sequence_shuffle(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action sequence shuffle intervention."""
        recent_actions = game_context.get("recent_actions", [])

        # Create shuffled variations of recent action patterns
        shuffled_sequences = []
        if len(recent_actions) >= 3:
            for i in range(min(3, len(recent_actions) - 2)):
                subseq = recent_actions[i:i+3].copy()
                random.shuffle(subseq)
                shuffled_sequences.append(subseq)

        return {
            "type": "action_sequence_shuffle",
            "intensity": config.intensity,
            "shuffle_probability": config.intensity,
            "shuffled_sequences": shuffled_sequences,
            "avoid_recent_patterns": True,
            "description": f"Shuffle action sequences with {config.intensity:.1%} probability"
        }

    def _apply_strategy_override(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strategy override intervention."""
        return {
            "type": "strategy_override",
            "intensity": config.intensity,
            "override_normal_strategy": True,
            "force_experimental_approach": True,
            "ignore_learned_preferences": config.intensity > 0.7,
            "description": f"Override normal strategy with experimental approach"
        }

    def _apply_timing_adjustment(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply timing adjustment intervention."""
        # Analyze recent timing patterns to determine adjustment
        base_delay = 1.0  # Base 1 second delay
        adjustment_factor = config.intensity * 2.0  # Up to 2x multiplier

        return {
            "type": "timing_adjustment",
            "intensity": config.intensity,
            "minimum_action_delay": base_delay * (1.0 + adjustment_factor),
            "vary_timing": True,
            "avoid_rapid_sequences": True,
            "description": f"Add {base_delay * adjustment_factor:.1f}s delays between actions"
        }

    def _apply_complete_reset(self, config: InterventionConfig, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply complete reset intervention."""
        return {
            "type": "complete_reset",
            "intensity": config.intensity,
            "reset_action_preferences": True,
            "reset_coordinate_preferences": True,
            "reset_learned_patterns": True,
            "start_fresh_exploration": True,
            "description": "Complete reset of all learned preferences"
        }

    def update_intervention_progress(self,
                                   intervention_id: str,
                                   action_taken: bool = False,
                                   progress_made: float = 0.0) -> bool:
        """
        Update the progress of an active intervention.

        Args:
            intervention_id: ID of the intervention to update
            action_taken: Whether an action was taken
            progress_made: Amount of progress made (0.0 to 1.0)

        Returns:
            True if intervention should continue, False if it should end
        """
        if intervention_id not in self.active_interventions:
            return False

        try:
            intervention = self.active_interventions[intervention_id]

            if action_taken:
                intervention["remaining_actions"] -= 1

            # Check if intervention should end
            if intervention["remaining_actions"] <= 0:
                self._complete_intervention(intervention_id, progress_made >= intervention["config"].success_threshold)
                return False

            return True

        except Exception as e:
            logger.error(f"Error updating intervention progress: {e}")
            return False

    def _complete_intervention(self, intervention_id: str, success: bool):
        """Complete an active intervention and record results."""
        if intervention_id not in self.active_interventions:
            return

        try:
            intervention = self.active_interventions[intervention_id]
            duration = time.time() - intervention["start_time"]

            # Update success rate tracking
            intervention_type = intervention["config"].intervention_type
            current_rate = self.intervention_success_rates.get(intervention_type, 0.5)
            # Simple moving average with 90% weight on history
            new_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            self.intervention_success_rates[intervention_type] = new_rate

            # Create final result
            result = InterventionResult(
                intervention_id=intervention_id,
                intervention_type=intervention_type,
                applied_timestamp=intervention["start_time"],
                success=success,
                outcome_data=intervention["outcome_data"],
                duration_seconds=duration,
                recovery_actions=intervention["config"].duration_actions - intervention["remaining_actions"]
            )

            # Update database record
            self._update_intervention_record(result)

            # Remove from active interventions
            del self.active_interventions[intervention_id]

            logger.info(f"Completed intervention {intervention_type.value}: success={success}, duration={duration:.1f}s")

        except Exception as e:
            logger.error(f"Error completing intervention: {e}")

    def _record_intervention(self, result: InterventionResult, streak_data: LosingStreakData):
        """Record an intervention in the database."""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO escalated_interventions (
                    intervention_id, streak_id, game_id, escalation_level,
                    intervention_type, intervention_data, applied_timestamp,
                    success, outcome_data, duration_seconds, recovery_actions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.intervention_id, streak_data.streak_id, streak_data.game_id,
                streak_data.escalation_level.value, result.intervention_type.value,
                json.dumps(result.outcome_data), result.applied_timestamp,
                result.success, json.dumps(result.outcome_data),
                result.duration_seconds, result.recovery_actions
            ))
            self.db.commit()

        except Exception as e:
            logger.error(f"Error recording intervention: {e}")

    def _update_intervention_record(self, result: InterventionResult):
        """Update an intervention record with final results."""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                UPDATE escalated_interventions
                SET success = ?, duration_seconds = ?, recovery_actions = ?, outcome_data = ?
                WHERE intervention_id = ?
            """, (
                result.success, result.duration_seconds, result.recovery_actions,
                json.dumps(result.outcome_data), result.intervention_id
            ))
            self.db.commit()

        except Exception as e:
            logger.error(f"Error updating intervention record: {e}")

    def get_intervention_guidance(self, intervention_id: str) -> Optional[Dict[str, Any]]:
        """Get guidance for applying an active intervention."""
        if intervention_id not in self.active_interventions:
            return None

        intervention = self.active_interventions[intervention_id]
        return intervention["outcome_data"]

    def get_active_interventions(self) -> List[Dict[str, Any]]:
        """Get list of all active interventions."""
        return [
            {
                "intervention_id": iid,
                "type": intervention["config"].intervention_type.value,
                "remaining_actions": intervention["remaining_actions"],
                "elapsed_time": time.time() - intervention["start_time"],
                "guidance": intervention["outcome_data"]
            }
            for iid, intervention in self.active_interventions.items()
        ]

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about interventions."""
        try:
            cursor = self.db.cursor()

            # Overall intervention stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_interventions,
                    COUNT(CASE WHEN success = TRUE THEN 1 END) as successful_interventions,
                    AVG(duration_seconds) as avg_duration,
                    AVG(recovery_actions) as avg_recovery_actions
                FROM escalated_interventions
            """)

            overall = cursor.fetchone()

            # Success rate by intervention type
            cursor.execute("""
                SELECT intervention_type,
                       COUNT(*) as total,
                       COUNT(CASE WHEN success = TRUE THEN 1 END) as successful,
                       AVG(duration_seconds) as avg_duration
                FROM escalated_interventions
                GROUP BY intervention_type
            """)

            type_stats = {}
            for row in cursor.fetchall():
                success_rate = (row[2] / row[1]) if row[1] > 0 else 0.0
                type_stats[row[0]] = {
                    "total_applications": row[1],
                    "success_rate": success_rate,
                    "average_duration": row[3] or 0.0
                }

            return {
                "total_interventions": overall[0] or 0,
                "overall_success_rate": (overall[1] / overall[0]) if overall[0] > 0 else 0.0,
                "average_duration_seconds": overall[2] or 0.0,
                "average_recovery_actions": overall[3] or 0.0,
                "active_interventions": len(self.active_interventions),
                "intervention_type_stats": type_stats,
                "cached_success_rates": dict(self.intervention_success_rates)
            }

        except Exception as e:
            logger.error(f"Error getting intervention statistics: {e}")
            return {"error": str(e)}