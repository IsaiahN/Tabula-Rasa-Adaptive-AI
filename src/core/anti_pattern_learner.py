"""
Anti-Pattern Learning System

This module identifies and learns from patterns that consistently lead to failure,
helping the system avoid repeating ineffective approaches and suggesting better alternatives.
"""

import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class AntiPatternType(Enum):
    """Types of anti-patterns that can be detected."""
    ACTION_SEQUENCE = "action_sequence"        # Sequences of actions that fail
    COORDINATE_CLUSTER = "coordinate_cluster"  # Areas of the screen that don't work
    TIMING_PATTERN = "timing_pattern"         # Timing-based failures
    SCORE_APPROACH = "score_approach"         # Score-based approaches that fail

@dataclass
class AntiPatternData:
    """Data structure for storing anti-pattern information."""
    pattern_id: str
    game_type: str
    game_id: Optional[str]
    pattern_type: AntiPatternType
    pattern_data: Dict[str, Any]
    failure_count: int
    total_encounters: int
    failure_rate: float
    first_observed: float
    last_observed: float
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context_data: Dict[str, Any]
    alternative_suggestions: List[Dict[str, Any]]

class AntiPatternLearner:
    """
    Learns from consistently failing patterns to help avoid repeating ineffective approaches.

    This system identifies patterns in failed attempts and provides alternatives
    to help break out of losing streaks more effectively.
    """

    def __init__(self, db_connection):
        """
        Initialize the anti-pattern learner.

        Args:
            db_connection: Database connection for persistence
        """
        self.db = db_connection
        self.known_patterns: Dict[str, AntiPatternData] = {}

        # Configuration thresholds
        self.min_encounters_for_pattern = 3    # Minimum encounters to establish pattern
        self.high_failure_rate_threshold = 0.8 # 80% failure rate to consider anti-pattern
        self.confidence_threshold = 0.7       # Minimum confidence to act on pattern
        self.severity_threshold = 0.6         # Minimum severity to avoid pattern

        # Pattern analysis parameters
        self.max_action_sequence_length = 5   # Maximum sequence length to analyze
        self.coordinate_cluster_radius = 3    # Pixel radius for coordinate clustering
        self.timing_window_seconds = 10       # Time window for timing pattern analysis

        self._load_known_patterns()

    def _generate_pattern_id(self, pattern_type: AntiPatternType, pattern_data: Dict[str, Any]) -> str:
        """Generate a unique pattern ID based on type and data."""
        data_str = json.dumps(pattern_data, sort_keys=True)
        hash_obj = hashlib.md5(f"{pattern_type.value}:{data_str}".encode())
        return f"{pattern_type.value}_{hash_obj.hexdigest()[:8]}"

    def _load_known_patterns(self):
        """Load known anti-patterns from database."""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT pattern_id, game_type, game_id, pattern_type, pattern_data,
                       failure_count, total_encounters, failure_rate, first_observed,
                       last_observed, severity, confidence, context_data, alternative_suggestions
                FROM anti_patterns
                WHERE failure_rate >= ? AND confidence >= ?
            """, (self.high_failure_rate_threshold, self.confidence_threshold))

            for row in cursor.fetchall():
                pattern = AntiPatternData(
                    pattern_id=row[0],
                    game_type=row[1],
                    game_id=row[2],
                    pattern_type=AntiPatternType(row[3]),
                    pattern_data=json.loads(row[4]),
                    failure_count=row[5],
                    total_encounters=row[6],
                    failure_rate=row[7],
                    first_observed=row[8],
                    last_observed=row[9],
                    severity=row[10],
                    confidence=row[11],
                    context_data=json.loads(row[12] or "{}"),
                    alternative_suggestions=json.loads(row[13] or "[]")
                )
                self.known_patterns[pattern.pattern_id] = pattern

            logger.info(f"Loaded {len(self.known_patterns)} known anti-patterns")

        except Exception as e:
            logger.error(f"Error loading known patterns: {e}")

    def analyze_failure(self,
                       game_type: str,
                       game_id: str,
                       action_sequence: List[int],
                       coordinates_used: List[Tuple[int, int]],
                       failure_context: Dict[str, Any]) -> List[AntiPatternData]:
        """
        Analyze a failure to extract potential anti-patterns.

        Args:
            game_type: Type of game
            game_id: Unique identifier for the game
            action_sequence: Sequence of actions that led to failure
            coordinates_used: List of coordinates that were tried
            failure_context: Additional context about the failure

        Returns:
            List of identified or updated anti-patterns
        """
        identified_patterns = []

        try:
            # Analyze action sequences
            action_patterns = self._analyze_action_sequences(
                game_type, game_id, action_sequence, failure_context
            )
            identified_patterns.extend(action_patterns)

            # Analyze coordinate clusters
            coordinate_patterns = self._analyze_coordinate_clusters(
                game_type, game_id, coordinates_used, failure_context
            )
            identified_patterns.extend(coordinate_patterns)

            # Analyze timing patterns
            timing_patterns = self._analyze_timing_patterns(
                game_type, game_id, action_sequence, failure_context
            )
            identified_patterns.extend(timing_patterns)

            # Analyze score approaches
            score_patterns = self._analyze_score_approaches(
                game_type, game_id, failure_context
            )
            identified_patterns.extend(score_patterns)

            # Update patterns in database
            for pattern in identified_patterns:
                self._save_pattern(pattern)

            return identified_patterns

        except Exception as e:
            logger.error(f"Error analyzing failure: {e}")
            return []

    def _analyze_action_sequences(self,
                                 game_type: str,
                                 game_id: str,
                                 action_sequence: List[int],
                                 context: Dict[str, Any]) -> List[AntiPatternData]:
        """Analyze action sequences for failure patterns."""
        patterns = []

        try:
            # Generate subsequences of different lengths
            for length in range(2, min(len(action_sequence) + 1, self.max_action_sequence_length + 1)):
                for i in range(len(action_sequence) - length + 1):
                    subsequence = action_sequence[i:i + length]

                    pattern_data = {
                        "sequence": subsequence,
                        "length": length,
                        "position": i  # Position within the full sequence
                    }

                    pattern_id = self._generate_pattern_id(AntiPatternType.ACTION_SEQUENCE, pattern_data)

                    # Check if we've seen this pattern before
                    if pattern_id in self.known_patterns:
                        pattern = self.known_patterns[pattern_id]
                        pattern.failure_count += 1
                        pattern.total_encounters += 1
                        pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                        pattern.last_observed = time.time()
                        pattern.confidence = min(1.0, pattern.total_encounters / self.min_encounters_for_pattern)
                    else:
                        # Create new pattern
                        pattern = AntiPatternData(
                            pattern_id=pattern_id,
                            game_type=game_type,
                            game_id=game_id,
                            pattern_type=AntiPatternType.ACTION_SEQUENCE,
                            pattern_data=pattern_data,
                            failure_count=1,
                            total_encounters=1,
                            failure_rate=1.0,
                            first_observed=time.time(),
                            last_observed=time.time(),
                            severity=0.5,  # Will be refined over time
                            confidence=1.0 / self.min_encounters_for_pattern,
                            context_data=context,
                            alternative_suggestions=self._generate_action_alternatives(subsequence)
                        )
                        self.known_patterns[pattern_id] = pattern

                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error analyzing action sequences: {e}")

        return patterns

    def _analyze_coordinate_clusters(self,
                                   game_type: str,
                                   game_id: str,
                                   coordinates: List[Tuple[int, int]],
                                   context: Dict[str, Any]) -> List[AntiPatternData]:
        """Analyze coordinate usage for failure patterns."""
        patterns = []

        try:
            # Group coordinates into clusters
            clusters = self._cluster_coordinates(coordinates)

            for cluster in clusters:
                if len(cluster) >= 2:  # Only consider clusters with multiple points
                    # Calculate cluster center and bounds
                    center_x = sum(x for x, y in cluster) / len(cluster)
                    center_y = sum(y for x, y in cluster) / len(cluster)

                    pattern_data = {
                        "center": [round(center_x), round(center_y)],
                        "coordinates": cluster,
                        "cluster_size": len(cluster),
                        "bounds": {
                            "min_x": min(x for x, y in cluster),
                            "max_x": max(x for x, y in cluster),
                            "min_y": min(y for x, y in cluster),
                            "max_y": max(y for x, y in cluster)
                        }
                    }

                    pattern_id = self._generate_pattern_id(AntiPatternType.COORDINATE_CLUSTER, pattern_data)

                    if pattern_id in self.known_patterns:
                        pattern = self.known_patterns[pattern_id]
                        pattern.failure_count += 1
                        pattern.total_encounters += 1
                        pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                        pattern.last_observed = time.time()
                        pattern.confidence = min(1.0, pattern.total_encounters / self.min_encounters_for_pattern)
                    else:
                        pattern = AntiPatternData(
                            pattern_id=pattern_id,
                            game_type=game_type,
                            game_id=game_id,
                            pattern_type=AntiPatternType.COORDINATE_CLUSTER,
                            pattern_data=pattern_data,
                            failure_count=1,
                            total_encounters=1,
                            failure_rate=1.0,
                            first_observed=time.time(),
                            last_observed=time.time(),
                            severity=0.5,
                            confidence=1.0 / self.min_encounters_for_pattern,
                            context_data=context,
                            alternative_suggestions=self._generate_coordinate_alternatives(center_x, center_y)
                        )
                        self.known_patterns[pattern_id] = pattern

                    patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error analyzing coordinate clusters: {e}")

        return patterns

    def _analyze_timing_patterns(self,
                               game_type: str,
                               game_id: str,
                               action_sequence: List[int],
                               context: Dict[str, Any]) -> List[AntiPatternData]:
        """Analyze timing-based failure patterns."""
        patterns = []

        try:
            # Extract timing information from context
            if "action_timings" not in context:
                return patterns

            timings = context["action_timings"]
            if len(timings) != len(action_sequence):
                return patterns

            # Analyze rapid action sequences
            rapid_sequences = []
            for i in range(1, len(timings)):
                time_diff = timings[i] - timings[i-1]
                if time_diff < 0.5:  # Less than 500ms between actions
                    rapid_sequences.append({
                        "actions": [action_sequence[i-1], action_sequence[i]],
                        "time_diff": time_diff,
                        "position": i-1
                    })

            if rapid_sequences:
                pattern_data = {
                    "rapid_sequences": rapid_sequences,
                    "total_rapid_actions": len(rapid_sequences),
                    "average_time_diff": sum(seq["time_diff"] for seq in rapid_sequences) / len(rapid_sequences)
                }

                pattern_id = self._generate_pattern_id(AntiPatternType.TIMING_PATTERN, pattern_data)

                if pattern_id in self.known_patterns:
                    pattern = self.known_patterns[pattern_id]
                    pattern.failure_count += 1
                    pattern.total_encounters += 1
                    pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                    pattern.last_observed = time.time()
                else:
                    pattern = AntiPatternData(
                        pattern_id=pattern_id,
                        game_type=game_type,
                        game_id=game_id,
                        pattern_type=AntiPatternType.TIMING_PATTERN,
                        pattern_data=pattern_data,
                        failure_count=1,
                        total_encounters=1,
                        failure_rate=1.0,
                        first_observed=time.time(),
                        last_observed=time.time(),
                        severity=0.6,  # Timing issues can be quite problematic
                        confidence=1.0 / self.min_encounters_for_pattern,
                        context_data=context,
                        alternative_suggestions=[{
                            "type": "timing_adjustment",
                            "suggestion": "Add delays between actions",
                            "min_delay": 1.0
                        }]
                    )
                    self.known_patterns[pattern_id] = pattern

                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error analyzing timing patterns: {e}")

        return patterns

    def _analyze_score_approaches(self,
                                game_type: str,
                                game_id: str,
                                context: Dict[str, Any]) -> List[AntiPatternData]:
        """Analyze score-based approach patterns that lead to failure."""
        patterns = []

        try:
            if "score_progression" not in context:
                return patterns

            score_progression = context["score_progression"]
            if len(score_progression) < 3:
                return patterns

            # Analyze score stagnation patterns
            stagnant_periods = 0
            declining_periods = 0

            for i in range(1, len(score_progression)):
                if score_progression[i] == score_progression[i-1]:
                    stagnant_periods += 1
                elif score_progression[i] < score_progression[i-1]:
                    declining_periods += 1

            total_periods = len(score_progression) - 1
            stagnation_rate = stagnant_periods / total_periods if total_periods > 0 else 0
            decline_rate = declining_periods / total_periods if total_periods > 0 else 0

            if stagnation_rate > 0.6 or decline_rate > 0.4:  # High stagnation or decline
                pattern_data = {
                    "stagnation_rate": stagnation_rate,
                    "decline_rate": decline_rate,
                    "final_score": score_progression[-1],
                    "initial_score": score_progression[0],
                    "score_variance": self._calculate_variance(score_progression)
                }

                pattern_id = self._generate_pattern_id(AntiPatternType.SCORE_APPROACH, pattern_data)

                if pattern_id in self.known_patterns:
                    pattern = self.known_patterns[pattern_id]
                    pattern.failure_count += 1
                    pattern.total_encounters += 1
                    pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                    pattern.last_observed = time.time()
                else:
                    pattern = AntiPatternData(
                        pattern_id=pattern_id,
                        game_type=game_type,
                        game_id=game_id,
                        pattern_type=AntiPatternType.SCORE_APPROACH,
                        pattern_data=pattern_data,
                        failure_count=1,
                        total_encounters=1,
                        failure_rate=1.0,
                        first_observed=time.time(),
                        last_observed=time.time(),
                        severity=0.7,
                        confidence=1.0 / self.min_encounters_for_pattern,
                        context_data=context,
                        alternative_suggestions=[{
                            "type": "exploration_boost",
                            "suggestion": "Increase exploration to find score-improving actions"
                        }]
                    )
                    self.known_patterns[pattern_id] = pattern

                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Error analyzing score approaches: {e}")

        return patterns

    def _cluster_coordinates(self, coordinates: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Group coordinates into clusters based on proximity."""
        if not coordinates:
            return []

        clusters = []
        remaining = set(coordinates)

        while remaining:
            # Start a new cluster with an arbitrary coordinate
            current = remaining.pop()
            cluster = [current]
            to_check = [current]

            # Find all coordinates within radius
            while to_check:
                center = to_check.pop()
                for coord in list(remaining):
                    distance = ((coord[0] - center[0]) ** 2 + (coord[1] - center[1]) ** 2) ** 0.5
                    if distance <= self.coordinate_cluster_radius:
                        cluster.append(coord)
                        to_check.append(coord)
                        remaining.remove(coord)

            clusters.append(cluster)

        return clusters

    def _generate_action_alternatives(self, failed_sequence: List[int]) -> List[Dict[str, Any]]:
        """Generate alternative action suggestions for a failed sequence."""
        alternatives = []

        # Suggest different action orders
        if len(failed_sequence) > 1:
            alternatives.append({
                "type": "reorder",
                "suggestion": f"Try reversing the order: {failed_sequence[::-1]}",
                "actions": failed_sequence[::-1]
            })

        # Suggest action substitutions
        for i, action in enumerate(failed_sequence):
            if action == 6:  # Action 6 (coordinate-based)
                alternatives.append({
                    "type": "action_substitution",
                    "suggestion": f"Try actions 1-5 instead of Action 6 at position {i}",
                    "position": i,
                    "suggested_actions": [1, 2, 3, 4, 5]
                })
            else:
                alternatives.append({
                    "type": "action_substitution",
                    "suggestion": f"Try Action 6 (coordinate-based) instead of Action {action}",
                    "position": i,
                    "suggested_actions": [6]
                })

        return alternatives

    def _generate_coordinate_alternatives(self, center_x: float, center_y: float) -> List[Dict[str, Any]]:
        """Generate alternative coordinate suggestions for a failed cluster."""
        alternatives = []

        # Suggest exploring areas outside the failed cluster
        offsets = [
            (-50, 0), (50, 0), (0, -50), (0, 50),  # Cardinal directions
            (-35, -35), (35, -35), (-35, 35), (35, 35)  # Diagonals
        ]

        for dx, dy in offsets:
            new_x = max(0, min(639, int(center_x + dx)))  # Assume 640x480 screen
            new_y = max(0, min(479, int(center_y + dy)))

            alternatives.append({
                "type": "coordinate_offset",
                "suggestion": f"Try coordinates away from failed cluster",
                "coordinates": [new_x, new_y],
                "offset": [dx, dy]
            })

        return alternatives

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _save_pattern(self, pattern: AntiPatternData):
        """Save anti-pattern to database."""
        try:
            cursor = self.db.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO anti_patterns (
                    pattern_id, game_type, game_id, pattern_type, pattern_data,
                    failure_count, total_encounters, failure_rate, first_observed,
                    last_observed, severity, confidence, context_data, alternative_suggestions,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id, pattern.game_type, pattern.game_id,
                pattern.pattern_type.value, json.dumps(pattern.pattern_data),
                pattern.failure_count, pattern.total_encounters, pattern.failure_rate,
                pattern.first_observed, pattern.last_observed, pattern.severity,
                pattern.confidence, json.dumps(pattern.context_data),
                json.dumps(pattern.alternative_suggestions), time.time()
            ))

            self.db.commit()

        except Exception as e:
            logger.error(f"Error saving anti-pattern: {e}")

    def get_pattern_suggestions(self,
                              game_type: str,
                              proposed_action_sequence: List[int],
                              proposed_coordinates: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Get suggestions to avoid known anti-patterns.

        Args:
            game_type: Type of game being played
            proposed_action_sequence: Sequence of actions being considered
            proposed_coordinates: Coordinates being considered

        Returns:
            Dictionary containing warnings and alternative suggestions
        """
        warnings = []
        suggestions = []
        risk_score = 0.0

        try:
            # Check action sequence patterns
            for pattern in self.known_patterns.values():
                if (pattern.game_type == game_type and
                    pattern.confidence >= self.confidence_threshold and
                    pattern.failure_rate >= self.high_failure_rate_threshold):

                    if pattern.pattern_type == AntiPatternType.ACTION_SEQUENCE:
                        seq_data = pattern.pattern_data
                        failed_seq = seq_data["sequence"]

                        # Check if proposed sequence contains this failed pattern
                        if self._sequence_contains_pattern(proposed_action_sequence, failed_seq):
                            warnings.append({
                                "type": "action_sequence",
                                "message": f"Proposed sequence contains known failing pattern: {failed_seq}",
                                "pattern_id": pattern.pattern_id,
                                "failure_rate": pattern.failure_rate,
                                "severity": pattern.severity
                            })
                            suggestions.extend(pattern.alternative_suggestions)
                            risk_score += pattern.severity * pattern.confidence

                    elif pattern.pattern_type == AntiPatternType.COORDINATE_CLUSTER:
                        cluster_data = pattern.pattern_data
                        center = cluster_data["center"]

                        # Check if any proposed coordinates are near this failed cluster
                        for coord in proposed_coordinates:
                            distance = ((coord[0] - center[0]) ** 2 + (coord[1] - center[1]) ** 2) ** 0.5
                            if distance <= self.coordinate_cluster_radius:
                                warnings.append({
                                    "type": "coordinate_cluster",
                                    "message": f"Coordinate {coord} is near failed cluster at {center}",
                                    "pattern_id": pattern.pattern_id,
                                    "failure_rate": pattern.failure_rate,
                                    "severity": pattern.severity
                                })
                                suggestions.extend(pattern.alternative_suggestions)
                                risk_score += pattern.severity * pattern.confidence * 0.5  # Lower weight for coordinates

            return {
                "warnings": warnings,
                "suggestions": suggestions,
                "risk_score": min(1.0, risk_score),
                "recommendation": "avoid" if risk_score > 0.7 else "caution" if risk_score > 0.4 else "proceed"
            }

        except Exception as e:
            logger.error(f"Error getting pattern suggestions: {e}")
            return {"error": str(e)}

    def _sequence_contains_pattern(self, sequence: List[int], pattern: List[int]) -> bool:
        """Check if a sequence contains a specific pattern as a subsequence."""
        if len(pattern) > len(sequence):
            return False

        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True

        return False

    def record_pattern_success(self,
                             game_type: str,
                             action_sequence: List[int],
                             coordinates_used: List[Tuple[int, int]]):
        """
        Record when a pattern that was previously failing now succeeds.
        This helps update confidence and failure rates.
        """
        try:
            current_time = time.time()

            # Check action sequences
            for length in range(2, min(len(action_sequence) + 1, self.max_action_sequence_length + 1)):
                for i in range(len(action_sequence) - length + 1):
                    subsequence = action_sequence[i:i + length]
                    pattern_data = {"sequence": subsequence, "length": length, "position": i}
                    pattern_id = self._generate_pattern_id(AntiPatternType.ACTION_SEQUENCE, pattern_data)

                    if pattern_id in self.known_patterns:
                        pattern = self.known_patterns[pattern_id]
                        pattern.total_encounters += 1
                        pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                        pattern.last_observed = current_time
                        self._save_pattern(pattern)

            # Check coordinate clusters
            clusters = self._cluster_coordinates(coordinates_used)
            for cluster in clusters:
                if len(cluster) >= 2:
                    center_x = sum(x for x, y in cluster) / len(cluster)
                    center_y = sum(y for x, y in cluster) / len(cluster)

                    pattern_data = {
                        "center": [round(center_x), round(center_y)],
                        "coordinates": cluster,
                        "cluster_size": len(cluster)
                    }
                    pattern_id = self._generate_pattern_id(AntiPatternType.COORDINATE_CLUSTER, pattern_data)

                    if pattern_id in self.known_patterns:
                        pattern = self.known_patterns[pattern_id]
                        pattern.total_encounters += 1
                        pattern.failure_rate = pattern.failure_count / pattern.total_encounters
                        pattern.last_observed = current_time
                        self._save_pattern(pattern)

        except Exception as e:
            logger.error(f"Error recording pattern success: {e}")

    def get_anti_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about anti-patterns."""
        try:
            cursor = self.db.cursor()

            # Overall statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_patterns,
                    AVG(failure_rate) as avg_failure_rate,
                    AVG(confidence) as avg_confidence,
                    AVG(severity) as avg_severity,
                    COUNT(CASE WHEN failure_rate >= ? THEN 1 END) as high_risk_patterns
                FROM anti_patterns
            """, (self.high_failure_rate_threshold,))

            overall_stats = cursor.fetchone()

            # Pattern type breakdown
            cursor.execute("""
                SELECT pattern_type, COUNT(*), AVG(failure_rate), AVG(severity)
                FROM anti_patterns
                GROUP BY pattern_type
            """)

            type_breakdown = {}
            for row in cursor.fetchall():
                type_breakdown[row[0]] = {
                    "count": row[1],
                    "avg_failure_rate": row[2],
                    "avg_severity": row[3]
                }

            return {
                "total_patterns": overall_stats[0] or 0,
                "average_failure_rate": overall_stats[1] or 0.0,
                "average_confidence": overall_stats[2] or 0.0,
                "average_severity": overall_stats[3] or 0.0,
                "high_risk_patterns": overall_stats[4] or 0,
                "active_patterns": len(self.known_patterns),
                "pattern_type_breakdown": type_breakdown
            }

        except Exception as e:
            logger.error(f"Error getting anti-pattern statistics: {e}")
            return {"error": str(e)}