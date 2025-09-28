"""
Mid-Game Pattern Detector for Real-Time Learning Engine

Detects emerging patterns during gameplay for immediate learning and adaptation.
Analyzes action sequences, coordinate clusters, score momentum, and effectiveness patterns.
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
class PatternCandidate:
    """A potential pattern detected during gameplay."""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    strength: float
    first_observed: float
    last_observed: float
    observation_count: int
    context_data: Dict[str, Any]

class MidGamePatternDetector:
    """
    Detects emerging patterns during gameplay for real-time learning.

    Monitors action sequences, coordinate usage, score changes, and effectiveness
    to identify patterns that can be used for immediate strategy adjustment.
    """

    def __init__(self, db_manager, game_type_classifier=None):
        self.db = db_manager
        self.game_type_classifier = game_type_classifier

        # Pattern detection state by game_id
        self.game_states: Dict[str, Dict[str, Any]] = {}

        # Configuration for pattern detection
        self.config = {
            "min_pattern_length": 3,
            "max_pattern_length": 15,
            "confidence_threshold": 0.6,
            "strength_threshold": 0.5,
            "coordinate_cluster_radius": 50,
            "score_momentum_window": 5,
            "action_sequence_window": 10,
            "pattern_memory_size": 50,
            "min_observations": 2,
            "decay_factor": 0.9
        }

        # Pattern candidates by game_id
        self.pattern_candidates: Dict[str, List[PatternCandidate]] = {}

        # Performance metrics
        self.metrics = {
            "patterns_detected": 0,
            "sequence_patterns": 0,
            "coordinate_patterns": 0,
            "score_patterns": 0,
            "effectiveness_patterns": 0
        }

    async def initialize_game(self, game_id: str, session_id: str):
        """Initialize pattern detection for a new game."""
        try:
            self.game_states[game_id] = {
                "session_id": session_id,
                "action_history": deque(maxlen=self.config["action_sequence_window"] * 2),
                "coordinate_history": deque(maxlen=self.config["action_sequence_window"] * 2),
                "score_history": deque(maxlen=self.config["score_momentum_window"] * 2),
                "effectiveness_history": deque(maxlen=self.config["action_sequence_window"]),
                "pattern_memory": deque(maxlen=self.config["pattern_memory_size"]),
                "last_pattern_check": time.time(),
                "action_count": 0
            }

            self.pattern_candidates[game_id] = []

            logger.info(f"Initialized pattern detection for game {game_id}")

        except Exception as e:
            logger.error(f"Failed to initialize pattern detection for game {game_id}: {e}")
            raise

    async def detect_emerging_patterns(self,
                                      game_id: str,
                                      session_id: str,
                                      recent_actions: List[int],
                                      recent_coordinates: List[Tuple[int, int]],
                                      current_score: float,
                                      action_count: int,
                                      game_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect emerging patterns based on recent gameplay data.

        Returns list of detected patterns with confidence scores.
        """
        if game_id not in self.game_states:
            await self.initialize_game(game_id, session_id)

        state = self.game_states[game_id]
        detected_patterns = []

        try:
            # Update game state with new data
            state["action_count"] = action_count
            if recent_actions:
                state["action_history"].extend(recent_actions)
            if recent_coordinates:
                state["coordinate_history"].extend(recent_coordinates)
            state["score_history"].append(current_score)

            # Parallel pattern detection
            pattern_results = await asyncio.gather(
                self._detect_action_sequence_patterns(game_id, state, game_context),
                self._detect_coordinate_cluster_patterns(game_id, state, game_context),
                self._detect_score_momentum_patterns(game_id, state, game_context),
                self._detect_action_effectiveness_patterns(game_id, state, game_context),
                return_exceptions=True
            )

            # Collect successful pattern detections
            for i, result in enumerate(pattern_results):
                if isinstance(result, Exception):
                    pattern_types = ["sequence", "coordinate", "score", "effectiveness"]
                    logger.warning(f"Error detecting {pattern_types[i]} patterns: {result}")
                    continue

                if result:
                    detected_patterns.extend(result)

            # Filter and store high-confidence patterns
            filtered_patterns = []
            for pattern in detected_patterns:
                if (pattern.get("confidence", 0) >= self.config["confidence_threshold"] and
                    pattern.get("pattern_strength", 0) >= self.config["strength_threshold"]):

                    # Store pattern in database
                    pattern_id = await self._store_pattern(game_id, session_id, pattern, action_count)
                    if pattern_id:
                        pattern["pattern_id"] = pattern_id
                        filtered_patterns.append(pattern)

            self.metrics["patterns_detected"] += len(filtered_patterns)
            state["last_pattern_check"] = time.time()

            return filtered_patterns

        except Exception as e:
            logger.error(f"Error detecting patterns for game {game_id}: {e}")
            return []

    async def _detect_action_sequence_patterns(self,
                                             game_id: str,
                                             state: Dict[str, Any],
                                             game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect repeating action sequences."""
        patterns = []

        try:
            actions = list(state["action_history"])
            if len(actions) < self.config["min_pattern_length"]:
                return patterns

            # Look for repeating subsequences
            for length in range(self.config["min_pattern_length"],
                              min(len(actions) // 2 + 1, self.config["max_pattern_length"] + 1)):

                for start in range(len(actions) - length * 2 + 1):
                    sequence = actions[start:start + length]

                    # Check for repetitions
                    repetitions = 1
                    for check_start in range(start + length, len(actions) - length + 1):
                        if actions[check_start:check_start + length] == sequence:
                            repetitions += 1

                    if repetitions >= 2:  # Found a repeating pattern
                        confidence = min(0.9, repetitions / 5.0)  # More repetitions = higher confidence
                        strength = min(0.9, length / self.config["max_pattern_length"])

                        pattern = {
                            "pattern_type": "emerging_sequence",
                            "pattern_data": {
                                "sequence": sequence,
                                "length": length,
                                "repetitions": repetitions,
                                "positions": [start + i * length for i in range(repetitions)]
                            },
                            "confidence": confidence,
                            "pattern_strength": strength,
                            "immediate_feedback": {
                                "sequence_frequency": repetitions,
                                "pattern_consistency": 1.0 if repetitions >= 3 else 0.7
                            }
                        }

                        patterns.append(pattern)
                        self.metrics["sequence_patterns"] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error detecting action sequence patterns: {e}")
            return []

    async def _detect_coordinate_cluster_patterns(self,
                                                game_id: str,
                                                state: Dict[str, Any],
                                                game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect coordinate clustering patterns."""
        patterns = []

        try:
            coordinates = list(state["coordinate_history"])
            if len(coordinates) < 3:
                return patterns

            # Analyze coordinate clusters
            clusters = self._find_coordinate_clusters(coordinates)

            for cluster in clusters:
                if len(cluster["coordinates"]) >= 3:  # Minimum cluster size
                    # Calculate cluster metrics
                    center = self._calculate_cluster_center(cluster["coordinates"])
                    radius = self._calculate_cluster_radius(cluster["coordinates"], center)

                    confidence = min(0.9, len(cluster["coordinates"]) / 10.0)
                    strength = min(0.9, 1.0 - (radius / 200.0))  # Smaller radius = stronger pattern

                    pattern = {
                        "pattern_type": "coordinate_cluster",
                        "pattern_data": {
                            "cluster_center": center,
                            "cluster_radius": radius,
                            "coordinate_count": len(cluster["coordinates"]),
                            "coordinates": cluster["coordinates"][-5:]  # Last 5 coordinates for reference
                        },
                        "confidence": confidence,
                        "pattern_strength": strength,
                        "immediate_feedback": {
                            "cluster_density": len(cluster["coordinates"]) / max(1, radius),
                            "recent_focus": True
                        }
                    }

                    patterns.append(pattern)
                    self.metrics["coordinate_patterns"] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error detecting coordinate cluster patterns: {e}")
            return []

    async def _detect_score_momentum_patterns(self,
                                            game_id: str,
                                            state: Dict[str, Any],
                                            game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect score momentum and progression patterns."""
        patterns = []

        try:
            scores = list(state["score_history"])
            if len(scores) < self.config["score_momentum_window"]:
                return patterns

            # Analyze recent score momentum
            recent_scores = scores[-self.config["score_momentum_window"]:]
            momentum = self._calculate_score_momentum(recent_scores)

            if abs(momentum) > 0.1:  # Significant momentum
                momentum_type = "positive" if momentum > 0 else "negative"

                confidence = min(0.9, abs(momentum))
                strength = min(0.9, abs(momentum) * 2)

                pattern = {
                    "pattern_type": "score_momentum",
                    "pattern_data": {
                        "momentum_type": momentum_type,
                        "momentum_value": momentum,
                        "score_trend": recent_scores,
                        "window_size": len(recent_scores)
                    },
                    "confidence": confidence,
                    "pattern_strength": strength,
                    "immediate_feedback": {
                        "trend_direction": momentum_type,
                        "trend_strength": abs(momentum)
                    }
                }

                patterns.append(pattern)
                self.metrics["score_patterns"] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error detecting score momentum patterns: {e}")
            return []

    async def _detect_action_effectiveness_patterns(self,
                                                  game_id: str,
                                                  state: Dict[str, Any],
                                                  game_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in action effectiveness."""
        patterns = []

        try:
            # Get recent action effectiveness data from database
            recent_outcomes = await self.db.fetch_all(
                """SELECT action_number, immediate_classification, confidence_level, score_delta
                   FROM action_outcome_tracking
                   WHERE game_id = ?
                   ORDER BY action_timestamp DESC LIMIT ?""",
                (game_id, self.config["action_sequence_window"])
            )

            if not recent_outcomes or len(recent_outcomes) < 3:
                return patterns

            # Analyze effectiveness patterns
            effectiveness_by_action = defaultdict(list)
            for outcome in recent_outcomes:
                action_num = outcome["action_number"]
                classification = outcome["immediate_classification"]
                effectiveness_by_action[action_num].append(classification)

            # Find consistently effective actions
            for action_num, classifications in effectiveness_by_action.items():
                if len(classifications) >= 2:
                    effectiveness_rate = sum(1 for c in classifications if c in ["highly_effective", "effective"]) / len(classifications)

                    if effectiveness_rate >= 0.7:  # 70% effectiveness or higher
                        confidence = min(0.9, effectiveness_rate)
                        strength = min(0.9, len(classifications) / 5.0)

                        pattern = {
                            "pattern_type": "action_effectiveness",
                            "pattern_data": {
                                "action_number": action_num,
                                "effectiveness_rate": effectiveness_rate,
                                "sample_size": len(classifications),
                                "recent_classifications": classifications
                            },
                            "confidence": confidence,
                            "pattern_strength": strength,
                            "immediate_feedback": {
                                "action_reliability": effectiveness_rate,
                                "recommendation": "prioritize" if effectiveness_rate >= 0.8 else "consider"
                            }
                        }

                        patterns.append(pattern)
                        self.metrics["effectiveness_patterns"] += 1

            return patterns

        except Exception as e:
            logger.error(f"Error detecting action effectiveness patterns: {e}")
            return []

    def _find_coordinate_clusters(self, coordinates: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Find clusters of coordinates using simple distance-based clustering."""
        if not coordinates:
            return []

        clusters = []
        used_coordinates = set()

        for i, coord in enumerate(coordinates):
            if coord in used_coordinates:
                continue

            cluster_coords = [coord]
            used_coordinates.add(coord)

            # Find nearby coordinates
            for j, other_coord in enumerate(coordinates):
                if other_coord in used_coordinates:
                    continue

                distance = ((coord[0] - other_coord[0]) ** 2 + (coord[1] - other_coord[1]) ** 2) ** 0.5
                if distance <= self.config["coordinate_cluster_radius"]:
                    cluster_coords.append(other_coord)
                    used_coordinates.add(other_coord)

            if len(cluster_coords) >= 2:  # Minimum cluster size
                clusters.append({"coordinates": cluster_coords})

        return clusters

    def _calculate_cluster_center(self, coordinates: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate the center of a coordinate cluster."""
        if not coordinates:
            return (0.0, 0.0)

        x_sum = sum(coord[0] for coord in coordinates)
        y_sum = sum(coord[1] for coord in coordinates)
        count = len(coordinates)

        return (x_sum / count, y_sum / count)

    def _calculate_cluster_radius(self, coordinates: List[Tuple[int, int]], center: Tuple[float, float]) -> float:
        """Calculate the radius of a coordinate cluster."""
        if not coordinates:
            return 0.0

        max_distance = 0.0
        for coord in coordinates:
            distance = ((coord[0] - center[0]) ** 2 + (coord[1] - center[1]) ** 2) ** 0.5
            max_distance = max(max_distance, distance)

        return max_distance

    def _calculate_score_momentum(self, scores: List[float]) -> float:
        """Calculate score momentum based on recent score changes."""
        if len(scores) < 2:
            return 0.0

        changes = []
        for i in range(1, len(scores)):
            change = scores[i] - scores[i-1]
            changes.append(change)

        # Simple momentum calculation
        if not changes:
            return 0.0

        # Weight recent changes more heavily
        weighted_sum = 0.0
        weight_sum = 0.0

        for i, change in enumerate(changes):
            weight = (i + 1) / len(changes)  # Linear weighting favoring recent changes
            weighted_sum += change * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    async def _store_pattern(self,
                           game_id: str,
                           session_id: str,
                           pattern: Dict[str, Any],
                           action_count: int) -> Optional[str]:
        """Store detected pattern in database."""
        try:
            pattern_id = f"pattern_{game_id}_{int(time.time() * 1000)}"
            current_time = time.time()

            await self.db.execute_query(
                """INSERT INTO real_time_patterns
                   (pattern_id, game_id, session_id, pattern_type, pattern_data,
                    confidence, detection_timestamp, game_action_count, pattern_strength,
                    immediate_feedback, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (pattern_id, game_id, session_id, pattern["pattern_type"],
                 json.dumps(pattern["pattern_data"]), pattern["confidence"],
                 current_time, action_count, pattern["pattern_strength"],
                 json.dumps(pattern.get("immediate_feedback", {})))
            )

            return pattern_id

        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return None

    async def get_active_patterns(self, game_id: str) -> List[Dict[str, Any]]:
        """Get currently active patterns for a game."""
        try:
            patterns = await self.db.fetch_all(
                """SELECT pattern_id, pattern_type, pattern_data, confidence, pattern_strength, detection_timestamp
                   FROM real_time_patterns
                   WHERE game_id = ? AND is_active = 1
                   ORDER BY detection_timestamp DESC""",
                (game_id,)
            )

            return [dict(row) for row in patterns] if patterns else []

        except Exception as e:
            logger.error(f"Error getting active patterns: {e}")
            return []

    async def update_pattern_evolution(self,
                                     pattern_id: str,
                                     evolution_data: Dict[str, Any]):
        """Update pattern evolution as it develops over time."""
        try:
            await self.db.execute_query(
                """UPDATE real_time_patterns
                   SET pattern_evolution = ?, last_updated = ?
                   WHERE pattern_id = ?""",
                (json.dumps(evolution_data), time.time(), pattern_id)
            )

        except Exception as e:
            logger.error(f"Error updating pattern evolution: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pattern detection performance metrics."""
        return {
            "total_patterns_detected": self.metrics["patterns_detected"],
            "sequence_patterns": self.metrics["sequence_patterns"],
            "coordinate_patterns": self.metrics["coordinate_patterns"],
            "score_patterns": self.metrics["score_patterns"],
            "effectiveness_patterns": self.metrics["effectiveness_patterns"],
            "active_games": len(self.game_states),
            "config": self.config.copy()
        }