"""
Action Outcome Tracker for Real-Time Learning Engine

Tracks immediate consequences of actions for real-time learning and adaptation.
Analyzes score changes, frame changes, movement detection, and effectiveness classification.
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
class ActionOutcome:
    """An immediate outcome of an action taken during gameplay."""
    outcome_id: str
    game_id: str
    session_id: str
    action_number: int
    coordinates: Optional[Tuple[int, int]]
    action_timestamp: float
    score_before: float
    score_after: float
    score_delta: float
    frame_changes_detected: bool
    movement_detected: bool
    new_elements_detected: bool
    immediate_classification: str
    confidence_level: float
    context_data: Dict[str, Any]
    learning_triggers: List[str]

class ActionOutcomeTracker:
    """
    Tracks immediate outcomes of actions for real-time learning.

    Analyzes each action's immediate effects and classifies effectiveness
    to enable real-time strategy adjustments and pattern learning.
    """

    def __init__(self, db_manager, game_type_classifier=None):
        self.db = db_manager
        self.game_type_classifier = game_type_classifier

        # Outcome tracking state by game_id
        self.game_states: Dict[str, Dict[str, Any]] = {}

        # Configuration for outcome tracking
        self.config = {
            "score_change_thresholds": {
                "highly_effective": 5.0,
                "effective": 1.0,
                "neutral": 0.0,
                "negative": -1.0,
                "harmful": -5.0
            },
            "frame_change_weight": 0.3,
            "movement_weight": 0.2,
            "score_weight": 0.5,
            "min_confidence": 0.1,
            "max_confidence": 0.95,
            "learning_trigger_threshold": 0.7,
            "outcome_memory_size": 100
        }

        # Learning trigger types
        self.learning_triggers = {
            "breakthrough": "High positive score change",
            "discovery": "New elements detected",
            "movement": "Significant movement detected",
            "stagnation": "No progress detected",
            "regression": "Negative score change",
            "consistency": "Consistent effectiveness pattern"
        }

        # Performance metrics
        self.metrics = {
            "outcomes_tracked": 0,
            "highly_effective_actions": 0,
            "effective_actions": 0,
            "neutral_actions": 0,
            "negative_actions": 0,
            "harmful_actions": 0,
            "learning_triggers_fired": 0
        }

    async def initialize_game(self, game_id: str, session_id: str):
        """Initialize outcome tracking for a new game."""
        try:
            self.game_states[game_id] = {
                "session_id": session_id,
                "outcome_history": deque(maxlen=self.config["outcome_memory_size"]),
                "action_effectiveness": defaultdict(list),  # action_number -> list of effectiveness scores
                "coordinate_effectiveness": defaultdict(list),  # (x,y) -> list of effectiveness scores
                "recent_patterns": deque(maxlen=20),
                "baseline_score": 0.0,
                "last_significant_change": time.time(),
                "action_count": 0
            }

            logger.info(f"Initialized action outcome tracking for game {game_id}")

        except Exception as e:
            logger.error(f"Failed to initialize action outcome tracking for game {game_id}: {e}")
            raise

    async def track_action_outcome(self,
                                  game_id: str,
                                  session_id: str,
                                  action_number: int,
                                  coordinates: Optional[Tuple[int, int]],
                                  score_before: float,
                                  score_after: float,
                                  frame_changes: bool,
                                  movement_detected: bool,
                                  action_count: int,
                                  game_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Track the immediate outcome of an action and classify its effectiveness.

        Returns the outcome data with classification and learning triggers.
        """
        if game_id not in self.game_states:
            await self.initialize_game(game_id, session_id)

        state = self.game_states[game_id]
        outcome_id = f"outcome_{game_id}_{int(time.time() * 1000)}"

        try:
            # Calculate outcome metrics
            score_delta = score_after - score_before
            action_timestamp = time.time()

            # Detect new elements (simplified heuristic)
            new_elements_detected = self._detect_new_elements(frame_changes, movement_detected, score_delta)

            # Classify action effectiveness
            classification, confidence = self._classify_action_effectiveness(
                score_delta, frame_changes, movement_detected, new_elements_detected, game_context
            )

            # Determine learning triggers
            learning_triggers = self._identify_learning_triggers(
                classification, score_delta, frame_changes, movement_detected, new_elements_detected, state
            )

            # Create outcome object
            outcome = ActionOutcome(
                outcome_id=outcome_id,
                game_id=game_id,
                session_id=session_id,
                action_number=action_number,
                coordinates=coordinates,
                action_timestamp=action_timestamp,
                score_before=score_before,
                score_after=score_after,
                score_delta=score_delta,
                frame_changes_detected=frame_changes,
                movement_detected=movement_detected,
                new_elements_detected=new_elements_detected,
                immediate_classification=classification,
                confidence_level=confidence,
                context_data=game_context or {},
                learning_triggers=learning_triggers
            )

            # Update game state
            state["outcome_history"].append(outcome)
            state["action_effectiveness"][action_number].append(self._classification_to_score(classification))
            if coordinates:
                state["coordinate_effectiveness"][coordinates].append(self._classification_to_score(classification))
            state["action_count"] = action_count

            # Store outcome in database
            await self._store_outcome(outcome)

            # Update metrics
            self.metrics["outcomes_tracked"] += 1
            self.metrics[f"{classification.replace('-', '_')}_actions"] += 1
            self.metrics["learning_triggers_fired"] += len(learning_triggers)

            # Process learning triggers
            await self._process_learning_triggers(outcome, learning_triggers)

            # Return outcome data for immediate use
            return {
                "outcome_id": outcome_id,
                "action_number": action_number,
                "coordinates": coordinates,
                "score_delta": score_delta,
                "immediate_classification": classification,
                "confidence_level": confidence,
                "frame_changes": frame_changes,
                "movement_detected": movement_detected,
                "new_elements": new_elements_detected,
                "learning_triggers": learning_triggers,
                "effectiveness_score": self._classification_to_score(classification)
            }

        except Exception as e:
            logger.error(f"Error tracking action outcome for game {game_id}: {e}")
            return None

    def _detect_new_elements(self, frame_changes: bool, movement_detected: bool, score_delta: float) -> bool:
        """Simple heuristic to detect if new elements appeared."""
        # If there are frame changes with score improvement, likely new elements
        return frame_changes and score_delta > 0

    def _classify_action_effectiveness(self,
                                     score_delta: float,
                                     frame_changes: bool,
                                     movement_detected: bool,
                                     new_elements_detected: bool,
                                     game_context: Optional[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Classify the effectiveness of an action based on its immediate outcomes.

        Returns (classification, confidence_level).
        """
        try:
            thresholds = self.config["score_change_thresholds"]

            # Base classification from score change
            if score_delta >= thresholds["highly_effective"]:
                base_classification = "highly_effective"
                base_confidence = 0.8
            elif score_delta >= thresholds["effective"]:
                base_classification = "effective"
                base_confidence = 0.7
            elif score_delta >= thresholds["neutral"]:
                base_classification = "neutral"
                base_confidence = 0.6
            elif score_delta >= thresholds["negative"]:
                base_classification = "negative"
                base_confidence = 0.7
            else:
                base_classification = "harmful"
                base_confidence = 0.8

            # Adjust confidence based on additional signals
            confidence_adjustments = 0.0

            # Frame changes add confidence to positive classifications
            if frame_changes:
                if base_classification in ["highly_effective", "effective"]:
                    confidence_adjustments += self.config["frame_change_weight"]
                elif base_classification in ["negative", "harmful"]:
                    confidence_adjustments -= self.config["frame_change_weight"] * 0.5

            # Movement detection adds confidence
            if movement_detected:
                confidence_adjustments += self.config["movement_weight"]

            # New elements strongly support positive classification
            if new_elements_detected:
                if base_classification in ["highly_effective", "effective"]:
                    confidence_adjustments += 0.2
                elif base_classification == "neutral":
                    # Upgrade neutral to effective if new elements detected
                    base_classification = "effective"
                    confidence_adjustments += 0.1

            # Calculate final confidence
            final_confidence = base_confidence + confidence_adjustments
            final_confidence = max(self.config["min_confidence"],
                                 min(self.config["max_confidence"], final_confidence))

            return base_classification, final_confidence

        except Exception as e:
            logger.error(f"Error classifying action effectiveness: {e}")
            return "neutral", 0.5

    def _identify_learning_triggers(self,
                                  classification: str,
                                  score_delta: float,
                                  frame_changes: bool,
                                  movement_detected: bool,
                                  new_elements_detected: bool,
                                  state: Dict[str, Any]) -> List[str]:
        """Identify what learning events this outcome should trigger."""
        triggers = []

        try:
            # Breakthrough: Highly effective action
            if classification == "highly_effective":
                triggers.append("breakthrough")

            # Discovery: New elements detected
            if new_elements_detected:
                triggers.append("discovery")

            # Movement: Significant visual changes
            if movement_detected and frame_changes:
                triggers.append("movement")

            # Stagnation: Multiple neutral/negative actions in a row
            recent_outcomes = list(state["outcome_history"])[-3:]
            if len(recent_outcomes) >= 3:
                recent_classifications = [o.immediate_classification for o in recent_outcomes]
                if all(c in ["neutral", "negative"] for c in recent_classifications):
                    triggers.append("stagnation")

            # Regression: Harmful action
            if classification == "harmful":
                triggers.append("regression")

            # Consistency: Same effectiveness repeated
            if len(recent_outcomes) >= 2:
                last_classification = recent_outcomes[-1].immediate_classification
                if last_classification == classification and classification in ["highly_effective", "effective"]:
                    triggers.append("consistency")

            return triggers

        except Exception as e:
            logger.error(f"Error identifying learning triggers: {e}")
            return []

    def _classification_to_score(self, classification: str) -> float:
        """Convert classification to numerical score for analysis."""
        score_map = {
            "highly_effective": 1.0,
            "effective": 0.7,
            "neutral": 0.5,
            "negative": 0.3,
            "harmful": 0.0
        }
        return score_map.get(classification, 0.5)

    async def _store_outcome(self, outcome: ActionOutcome):
        """Store action outcome in database."""
        try:
            await self.db.execute_query(
                """INSERT INTO action_outcome_tracking
                   (outcome_id, game_id, session_id, action_number, coordinates_x, coordinates_y,
                    action_timestamp, score_before, score_after, score_delta, frame_changes_detected,
                    movement_detected, new_elements_detected, immediate_classification, confidence_level,
                    context_data, learning_triggers)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (outcome.outcome_id, outcome.game_id, outcome.session_id, outcome.action_number,
                 outcome.coordinates[0] if outcome.coordinates else None,
                 outcome.coordinates[1] if outcome.coordinates else None,
                 outcome.action_timestamp, outcome.score_before, outcome.score_after, outcome.score_delta,
                 outcome.frame_changes_detected, outcome.movement_detected, outcome.new_elements_detected,
                 outcome.immediate_classification, outcome.confidence_level,
                 json.dumps(outcome.context_data), json.dumps(outcome.learning_triggers))
            )

        except Exception as e:
            logger.error(f"Error storing action outcome: {e}")

    async def _process_learning_triggers(self, outcome: ActionOutcome, triggers: List[str]):
        """Process learning triggers and create learning events."""
        try:
            for trigger in triggers:
                if trigger in self.learning_triggers:
                    # Create a mid-game learning event for high-value triggers
                    if outcome.confidence_level >= self.config["learning_trigger_threshold"]:
                        event_id = f"trigger_{outcome.game_id}_{int(time.time() * 1000)}"

                        event_data = {
                            "trigger_type": trigger,
                            "trigger_description": self.learning_triggers[trigger],
                            "outcome_id": outcome.outcome_id,
                            "action_number": outcome.action_number,
                            "coordinates": outcome.coordinates,
                            "score_delta": outcome.score_delta,
                            "classification": outcome.immediate_classification,
                            "confidence": outcome.confidence_level
                        }

                        await self.db.execute_query(
                            """INSERT INTO mid_game_learning_events
                               (event_id, game_id, session_id, event_type, event_data, trigger_action,
                                confidence, immediate_application)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (event_id, outcome.game_id, outcome.session_id, f"outcome_trigger_{trigger}",
                             json.dumps(event_data), outcome.action_number, outcome.confidence_level, True)
                        )

        except Exception as e:
            logger.error(f"Error processing learning triggers: {e}")

    async def get_action_effectiveness_summary(self, game_id: str) -> Dict[str, Any]:
        """Get effectiveness summary for all actions in a game."""
        if game_id not in self.game_states:
            return {}

        try:
            state = self.game_states[game_id]

            # Calculate effectiveness summaries
            action_summaries = {}
            for action_num, scores in state["action_effectiveness"].items():
                if scores:
                    avg_effectiveness = sum(scores) / len(scores)
                    action_summaries[action_num] = {
                        "average_effectiveness": avg_effectiveness,
                        "sample_size": len(scores),
                        "consistency": 1.0 - (max(scores) - min(scores)) if len(scores) > 1 else 1.0,
                        "recommendation": "prioritize" if avg_effectiveness >= 0.8 else
                                       "consider" if avg_effectiveness >= 0.6 else "avoid"
                    }

            # Get coordinate effectiveness
            coordinate_summaries = {}
            for coords, scores in state["coordinate_effectiveness"].items():
                if scores:
                    avg_effectiveness = sum(scores) / len(scores)
                    coordinate_summaries[f"{coords[0]},{coords[1]}"] = {
                        "average_effectiveness": avg_effectiveness,
                        "sample_size": len(scores),
                        "coordinates": coords
                    }

            # Get recent trend
            recent_outcomes = list(state["outcome_history"])[-10:]
            recent_effectiveness = [self._classification_to_score(o.immediate_classification) for o in recent_outcomes]
            trend = "improving" if len(recent_effectiveness) >= 2 and recent_effectiveness[-1] > recent_effectiveness[0] else "declining"

            return {
                "game_id": game_id,
                "total_outcomes": len(state["outcome_history"]),
                "action_effectiveness": action_summaries,
                "coordinate_effectiveness": coordinate_summaries,
                "recent_trend": trend,
                "recent_average_effectiveness": sum(recent_effectiveness) / len(recent_effectiveness) if recent_effectiveness else 0.5
            }

        except Exception as e:
            logger.error(f"Error getting action effectiveness summary: {e}")
            return {}

    async def get_learning_insights(self, game_id: str) -> Dict[str, Any]:
        """Get learning insights based on tracked outcomes."""
        if game_id not in self.game_states:
            return {}

        try:
            # Get recent learning events triggered by outcomes
            learning_events = await self.db.fetch_all(
                """SELECT event_type, event_data, confidence, trigger_action
                   FROM mid_game_learning_events
                   WHERE game_id = ? AND event_type LIKE 'outcome_trigger_%'
                   ORDER BY trigger_action DESC LIMIT 10""",
                (game_id,)
            )

            # Get effectiveness distribution
            effectiveness_distribution = await self.db.fetch_all(
                """SELECT immediate_classification, COUNT(*) as count
                   FROM action_outcome_tracking
                   WHERE game_id = ?
                   GROUP BY immediate_classification""",
                (game_id,)
            )

            insights = {
                "game_id": game_id,
                "recent_learning_events": [dict(row) for row in learning_events] if learning_events else [],
                "effectiveness_distribution": {row["immediate_classification"]: row["count"]
                                             for row in effectiveness_distribution} if effectiveness_distribution else {},
                "learning_triggers_fired": len(learning_events) if learning_events else 0
            }

            return insights

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get action outcome tracking performance metrics."""
        total_actions = self.metrics["outcomes_tracked"]
        if total_actions == 0:
            return {"message": "No actions tracked yet"}

        return {
            "total_outcomes_tracked": total_actions,
            "effectiveness_distribution": {
                "highly_effective": f"{(self.metrics['highly_effective_actions'] / total_actions) * 100:.1f}%",
                "effective": f"{(self.metrics['effective_actions'] / total_actions) * 100:.1f}%",
                "neutral": f"{(self.metrics['neutral_actions'] / total_actions) * 100:.1f}%",
                "negative": f"{(self.metrics['negative_actions'] / total_actions) * 100:.1f}%",
                "harmful": f"{(self.metrics['harmful_actions'] / total_actions) * 100:.1f}%"
            },
            "learning_triggers_fired": self.metrics["learning_triggers_fired"],
            "active_games": len(self.game_states),
            "config": self.config.copy()
        }