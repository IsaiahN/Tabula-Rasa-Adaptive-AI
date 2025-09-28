"""
Dynamic Strategy Adjuster for Real-Time Learning Engine

Adjusts strategy in real-time based on detected patterns and immediate feedback.
Makes mid-game adjustments to action priorities, focus areas, and exploration strategies.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class StrategyAdjustment:
    """A strategy adjustment to be applied during gameplay."""
    adjustment_id: str
    adjustment_type: str
    adjustment_data: Dict[str, Any]
    trigger_pattern_id: Optional[str]
    confidence: float
    priority: int
    duration_actions: int
    applied_at_action: int
    immediate_effect: Dict[str, Any]

class DynamicStrategyAdjuster:
    """
    Adjusts strategy in real-time based on detected patterns and feedback.

    Monitors pattern detection results and game state to make immediate
    adjustments to action selection, coordinate focus, and exploration behavior.
    """

    def __init__(self, db_manager, game_type_classifier=None):
        self.db = db_manager
        self.game_type_classifier = game_type_classifier

        # Strategy adjustment state by game_id
        self.game_states: Dict[str, Dict[str, Any]] = {}

        # Configuration for strategy adjustment
        self.config = {
            "adjustment_confidence_threshold": 0.6,
            "max_concurrent_adjustments": 5,
            "adjustment_duration_base": 10,  # Base number of actions for adjustment duration
            "effectiveness_threshold": 0.7,
            "pattern_strength_threshold": 0.5,
            "score_improvement_threshold": 0.1,
            "coordinate_focus_radius": 75,
            "action_priority_boost": 0.3,
            "exploration_boost_factor": 1.5
        }

        # Active adjustments by game_id
        self.active_adjustments: Dict[str, List[StrategyAdjustment]] = {}

        # Performance metrics
        self.metrics = {
            "adjustments_made": 0,
            "successful_adjustments": 0,
            "action_priority_changes": 0,
            "coordinate_focus_shifts": 0,
            "exploration_boosts": 0,
            "pattern_avoidances": 0
        }

    async def initialize_game(self, game_id: str, session_id: str):
        """Initialize strategy adjustment for a new game."""
        try:
            self.game_states[game_id] = {
                "session_id": session_id,
                "current_strategy": {
                    "action_priorities": {},  # action_number -> priority_multiplier
                    "coordinate_focus": None,  # (x, y, radius) or None
                    "exploration_mode": "normal",  # "normal", "boosted", "focused"
                    "avoided_patterns": []  # List of pattern_ids to avoid
                },
                "adjustment_history": deque(maxlen=20),
                "effectiveness_history": deque(maxlen=10),
                "last_adjustment": time.time(),
                "action_count": 0
            }

            self.active_adjustments[game_id] = []

            logger.info(f"Initialized dynamic strategy adjustment for game {game_id}")

        except Exception as e:
            logger.error(f"Failed to initialize strategy adjustment for game {game_id}: {e}")
            raise

    async def evaluate_strategy_adjustments(self,
                                          game_id: str,
                                          session_id: str,
                                          active_patterns: List[str],
                                          last_score_change: float,
                                          recent_actions: List[int],
                                          action_count: int,
                                          game_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate and apply strategy adjustments based on current game state.

        Returns list of adjustments made.
        """
        if game_id not in self.game_states:
            await self.initialize_game(game_id, session_id)

        state = self.game_states[game_id]
        adjustments_made = []

        try:
            state["action_count"] = action_count

            # Update effectiveness history
            if last_score_change != 0:
                effectiveness = "positive" if last_score_change > 0 else "negative"
                state["effectiveness_history"].append({
                    "action_count": action_count,
                    "score_change": last_score_change,
                    "effectiveness": effectiveness
                })

            # Clean up expired adjustments
            await self._clean_expired_adjustments(game_id, action_count)

            # Evaluate different types of adjustments
            adjustment_results = await asyncio.gather(
                self._evaluate_action_priority_adjustments(game_id, active_patterns, recent_actions, game_context),
                self._evaluate_coordinate_focus_adjustments(game_id, active_patterns, game_context),
                self._evaluate_exploration_adjustments(game_id, state["effectiveness_history"], game_context),
                self._evaluate_pattern_avoidance_adjustments(game_id, active_patterns, game_context),
                return_exceptions=True
            )

            # Collect successful adjustments
            for i, result in enumerate(adjustment_results):
                if isinstance(result, Exception):
                    adjustment_types = ["action_priority", "coordinate_focus", "exploration", "pattern_avoidance"]
                    logger.warning(f"Error evaluating {adjustment_types[i]} adjustments: {result}")
                    continue

                if result:
                    adjustments_made.extend(result)

            # Apply adjustments and store in database
            applied_adjustments = []
            for adjustment in adjustments_made:
                if await self._apply_adjustment(game_id, session_id, adjustment, action_count):
                    applied_adjustments.append(adjustment)

            self.metrics["adjustments_made"] += len(applied_adjustments)
            state["last_adjustment"] = time.time()

            return applied_adjustments

        except Exception as e:
            logger.error(f"Error evaluating strategy adjustments for game {game_id}: {e}")
            return []

    async def _evaluate_action_priority_adjustments(self,
                                                  game_id: str,
                                                  active_patterns: List[str],
                                                  recent_actions: List[int],
                                                  game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate adjustments to action priorities based on effectiveness patterns."""
        adjustments = []

        try:
            # Get effectiveness patterns from active patterns
            effectiveness_patterns = await self.db.fetch_all(
                """SELECT pattern_id, pattern_data, confidence, pattern_strength
                   FROM real_time_patterns
                   WHERE game_id = ? AND pattern_type = 'action_effectiveness' AND is_active = 1""",
                (game_id,)
            )

            for pattern in effectiveness_patterns:
                pattern_data = json.loads(pattern["pattern_data"])
                action_number = pattern_data.get("action_number")
                effectiveness_rate = pattern_data.get("effectiveness_rate", 0)

                if (action_number is not None and
                    effectiveness_rate >= self.config["effectiveness_threshold"] and
                    pattern["confidence"] >= self.config["adjustment_confidence_threshold"]):

                    # Create action priority boost adjustment
                    priority_boost = self.config["action_priority_boost"] * effectiveness_rate

                    adjustment = {
                        "adjustment_type": "action_priority_change",
                        "adjustment_data": {
                            "action_number": action_number,
                            "priority_change": priority_boost,
                            "change_type": "boost",
                            "reason": f"High effectiveness rate: {effectiveness_rate:.1%}"
                        },
                        "trigger_pattern_id": pattern["pattern_id"],
                        "confidence": pattern["confidence"],
                        "priority": 2,  # Medium priority
                        "duration_actions": self.config["adjustment_duration_base"],
                        "immediate_effect": {
                            "action_prioritized": action_number,
                            "boost_amount": priority_boost
                        }
                    }

                    adjustments.append(adjustment)
                    self.metrics["action_priority_changes"] += 1

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating action priority adjustments: {e}")
            return []

    async def _evaluate_coordinate_focus_adjustments(self,
                                                   game_id: str,
                                                   active_patterns: List[str],
                                                   game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate adjustments to coordinate focus based on cluster patterns."""
        adjustments = []

        try:
            # Get coordinate cluster patterns
            cluster_patterns = await self.db.fetch_all(
                """SELECT pattern_id, pattern_data, confidence, pattern_strength
                   FROM real_time_patterns
                   WHERE game_id = ? AND pattern_type = 'coordinate_cluster' AND is_active = 1
                   ORDER BY confidence DESC LIMIT 1""",  # Focus on strongest cluster
                (game_id,)
            )

            for pattern in cluster_patterns:
                pattern_data = json.loads(pattern["pattern_data"])
                cluster_center = pattern_data.get("cluster_center")
                cluster_radius = pattern_data.get("cluster_radius", 50)

                if (cluster_center and
                    pattern["confidence"] >= self.config["adjustment_confidence_threshold"] and
                    pattern["pattern_strength"] >= self.config["pattern_strength_threshold"]):

                    # Create coordinate focus adjustment
                    focus_radius = min(self.config["coordinate_focus_radius"], cluster_radius * 1.5)

                    adjustment = {
                        "adjustment_type": "coordinate_focus_shift",
                        "adjustment_data": {
                            "focus_center": cluster_center,
                            "focus_radius": focus_radius,
                            "cluster_density": pattern_data.get("coordinate_count", 0),
                            "reason": "Strong coordinate clustering detected"
                        },
                        "trigger_pattern_id": pattern["pattern_id"],
                        "confidence": pattern["confidence"],
                        "priority": 1,  # High priority
                        "duration_actions": self.config["adjustment_duration_base"] * 2,
                        "immediate_effect": {
                            "focus_area": f"({cluster_center[0]:.0f}, {cluster_center[1]:.0f}) radius {focus_radius:.0f}",
                            "cluster_strength": pattern["pattern_strength"]
                        }
                    }

                    adjustments.append(adjustment)
                    self.metrics["coordinate_focus_shifts"] += 1

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating coordinate focus adjustments: {e}")
            return []

    async def _evaluate_exploration_adjustments(self,
                                              game_id: str,
                                              effectiveness_history: deque,
                                              game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate adjustments to exploration behavior based on recent effectiveness."""
        adjustments = []

        try:
            if len(effectiveness_history) < 3:
                return adjustments

            # Analyze recent effectiveness
            recent_effectiveness = list(effectiveness_history)[-5:]  # Last 5 effectiveness records
            negative_count = sum(1 for e in recent_effectiveness if e["effectiveness"] == "negative")
            total_count = len(recent_effectiveness)

            # If too many negative outcomes, boost exploration
            if negative_count / total_count >= 0.6:  # 60% or more negative outcomes
                confidence = min(0.9, negative_count / total_count)

                adjustment = {
                    "adjustment_type": "exploration_boost",
                    "adjustment_data": {
                        "boost_factor": self.config["exploration_boost_factor"],
                        "boost_type": "diversify_actions",
                        "reason": f"High negative effectiveness rate: {negative_count}/{total_count}"
                    },
                    "trigger_pattern_id": None,  # Not triggered by specific pattern
                    "confidence": confidence,
                    "priority": 2,  # Medium priority
                    "duration_actions": self.config["adjustment_duration_base"],
                    "immediate_effect": {
                        "exploration_mode": "boosted",
                        "diversification_factor": self.config["exploration_boost_factor"]
                    }
                }

                adjustments.append(adjustment)
                self.metrics["exploration_boosts"] += 1

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating exploration adjustments: {e}")
            return []

    async def _evaluate_pattern_avoidance_adjustments(self,
                                                    game_id: str,
                                                    active_patterns: List[str],
                                                    game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate adjustments to avoid harmful patterns."""
        adjustments = []

        try:
            # Get any negative score momentum patterns
            negative_patterns = await self.db.fetch_all(
                """SELECT pattern_id, pattern_data, confidence, pattern_strength
                   FROM real_time_patterns
                   WHERE game_id = ? AND pattern_type = 'score_momentum' AND is_active = 1""",
                (game_id,)
            )

            for pattern in negative_patterns:
                pattern_data = json.loads(pattern["pattern_data"])
                momentum_type = pattern_data.get("momentum_type")

                if (momentum_type == "negative" and
                    pattern["confidence"] >= self.config["adjustment_confidence_threshold"]):

                    # Create pattern avoidance adjustment
                    adjustment = {
                        "adjustment_type": "pattern_avoidance",
                        "adjustment_data": {
                            "avoided_pattern_id": pattern["pattern_id"],
                            "pattern_type": "negative_momentum",
                            "avoidance_strategy": "diversify_immediately",
                            "reason": "Negative score momentum detected"
                        },
                        "trigger_pattern_id": pattern["pattern_id"],
                        "confidence": pattern["confidence"],
                        "priority": 1,  # High priority
                        "duration_actions": self.config["adjustment_duration_base"] // 2,  # Shorter duration
                        "immediate_effect": {
                            "pattern_avoided": pattern["pattern_id"],
                            "avoidance_action": "diversify"
                        }
                    }

                    adjustments.append(adjustment)
                    self.metrics["pattern_avoidances"] += 1

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating pattern avoidance adjustments: {e}")
            return []

    async def _apply_adjustment(self,
                              game_id: str,
                              session_id: str,
                              adjustment: Dict[str, Any],
                              action_count: int) -> bool:
        """Apply a strategy adjustment and store it in the database."""
        try:
            # Generate adjustment ID
            adjustment_id = f"adj_{game_id}_{int(time.time() * 1000)}"
            adjustment["adjustment_id"] = adjustment_id

            # Update game state with adjustment
            state = self.game_states[game_id]
            current_strategy = state["current_strategy"]

            if adjustment["adjustment_type"] == "action_priority_change":
                action_num = adjustment["adjustment_data"]["action_number"]
                priority_change = adjustment["adjustment_data"]["priority_change"]
                current_strategy["action_priorities"][action_num] = priority_change

            elif adjustment["adjustment_type"] == "coordinate_focus_shift":
                focus_center = adjustment["adjustment_data"]["focus_center"]
                focus_radius = adjustment["adjustment_data"]["focus_radius"]
                current_strategy["coordinate_focus"] = (focus_center[0], focus_center[1], focus_radius)

            elif adjustment["adjustment_type"] == "exploration_boost":
                current_strategy["exploration_mode"] = "boosted"

            elif adjustment["adjustment_type"] == "pattern_avoidance":
                pattern_id = adjustment["adjustment_data"]["avoided_pattern_id"]
                current_strategy["avoided_patterns"].append(pattern_id)

            # Store adjustment in database
            await self.db.execute_query(
                """INSERT INTO real_time_strategy_adjustments
                   (adjustment_id, game_id, session_id, trigger_pattern_id, adjustment_type,
                    adjustment_data, applied_at_action, immediate_effect, effectiveness_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (adjustment_id, game_id, session_id,
                 adjustment.get("trigger_pattern_id"),
                 adjustment["adjustment_type"],
                 json.dumps(adjustment["adjustment_data"]),
                 action_count,
                 json.dumps(adjustment["immediate_effect"]),
                 0.0)  # Will be updated based on results
            )

            # Add to active adjustments
            strategy_adjustment = StrategyAdjustment(
                adjustment_id=adjustment_id,
                adjustment_type=adjustment["adjustment_type"],
                adjustment_data=adjustment["adjustment_data"],
                trigger_pattern_id=adjustment.get("trigger_pattern_id"),
                confidence=adjustment["confidence"],
                priority=adjustment["priority"],
                duration_actions=adjustment["duration_actions"],
                applied_at_action=action_count,
                immediate_effect=adjustment["immediate_effect"]
            )

            self.active_adjustments[game_id].append(strategy_adjustment)

            # Keep active adjustments list manageable
            if len(self.active_adjustments[game_id]) > self.config["max_concurrent_adjustments"]:
                self.active_adjustments[game_id] = self.active_adjustments[game_id][-self.config["max_concurrent_adjustments"]:]

            logger.info(f"Applied strategy adjustment {adjustment_id} of type {adjustment['adjustment_type']}")
            return True

        except Exception as e:
            logger.error(f"Error applying strategy adjustment: {e}")
            return False

    async def _clean_expired_adjustments(self, game_id: str, current_action_count: int):
        """Remove expired adjustments and update strategy state."""
        try:
            if game_id not in self.active_adjustments:
                return

            active_list = self.active_adjustments[game_id]
            expired_adjustments = []

            # Find expired adjustments
            for adjustment in active_list:
                if current_action_count >= adjustment.applied_at_action + adjustment.duration_actions:
                    expired_adjustments.append(adjustment)

            # Remove expired adjustments from active list
            for expired in expired_adjustments:
                active_list.remove(expired)

                # Update strategy state by removing the adjustment
                state = self.game_states[game_id]
                current_strategy = state["current_strategy"]

                if expired.adjustment_type == "action_priority_change":
                    action_num = expired.adjustment_data["action_number"]
                    if action_num in current_strategy["action_priorities"]:
                        del current_strategy["action_priorities"][action_num]

                elif expired.adjustment_type == "coordinate_focus_shift":
                    current_strategy["coordinate_focus"] = None

                elif expired.adjustment_type == "exploration_boost":
                    current_strategy["exploration_mode"] = "normal"

                elif expired.adjustment_type == "pattern_avoidance":
                    pattern_id = expired.adjustment_data["avoided_pattern_id"]
                    if pattern_id in current_strategy["avoided_patterns"]:
                        current_strategy["avoided_patterns"].remove(pattern_id)

                # Update database with duration
                await self.db.execute_query(
                    """UPDATE real_time_strategy_adjustments
                       SET duration_actions = ?
                       WHERE adjustment_id = ?""",
                    (current_action_count - expired.applied_at_action, expired.adjustment_id)
                )

        except Exception as e:
            logger.error(f"Error cleaning expired adjustments: {e}")

    async def get_current_strategy(self, game_id: str) -> Dict[str, Any]:
        """Get the current strategy state for a game."""
        if game_id not in self.game_states:
            return {}

        try:
            state = self.game_states[game_id]
            active_list = self.active_adjustments.get(game_id, [])

            return {
                "current_strategy": state["current_strategy"].copy(),
                "active_adjustments": [
                    {
                        "adjustment_id": adj.adjustment_id,
                        "adjustment_type": adj.adjustment_type,
                        "confidence": adj.confidence,
                        "applied_at_action": adj.applied_at_action,
                        "duration_actions": adj.duration_actions,
                        "immediate_effect": adj.immediate_effect
                    }
                    for adj in active_list
                ],
                "adjustment_count": len(active_list)
            }

        except Exception as e:
            logger.error(f"Error getting current strategy: {e}")
            return {}

    async def evaluate_adjustment_effectiveness(self,
                                              game_id: str,
                                              adjustment_id: str,
                                              score_before: float,
                                              score_after: float,
                                              actions_since_adjustment: int):
        """Evaluate the effectiveness of a specific adjustment."""
        try:
            score_improvement = score_after - score_before
            effectiveness_score = min(1.0, max(0.0, score_improvement / 10.0))  # Normalize to 0-1

            success = effectiveness_score >= self.config["score_improvement_threshold"]

            await self.db.execute_query(
                """UPDATE real_time_strategy_adjustments
                   SET effectiveness_score = ?, success = ?
                   WHERE adjustment_id = ?""",
                (effectiveness_score, success, adjustment_id)
            )

            if success:
                self.metrics["successful_adjustments"] += 1

        except Exception as e:
            logger.error(f"Error evaluating adjustment effectiveness: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy adjustment performance metrics."""
        return {
            "total_adjustments": self.metrics["adjustments_made"],
            "successful_adjustments": self.metrics["successful_adjustments"],
            "success_rate": (self.metrics["successful_adjustments"] / max(1, self.metrics["adjustments_made"])),
            "action_priority_changes": self.metrics["action_priority_changes"],
            "coordinate_focus_shifts": self.metrics["coordinate_focus_shifts"],
            "exploration_boosts": self.metrics["exploration_boosts"],
            "pattern_avoidances": self.metrics["pattern_avoidances"],
            "active_games": len(self.game_states),
            "config": self.config.copy()
        }