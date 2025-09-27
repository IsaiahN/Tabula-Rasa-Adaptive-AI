"""
Losing Streak Detection System

This module provides comprehensive tracking and analysis of consecutive failures
across multiple game attempts, enabling the system to detect when it's stuck
and needs escalated intervention strategies.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of game failure that contribute to losing streaks."""
    TIMEOUT = "timeout"
    SCORE_STAGNATION = "score_stagnation"
    ZERO_PROGRESS = "zero_progress"
    LOW_PROGRESS = "low_progress"
    ACTION_LOOP = "action_loop"
    COORDINATE_STUCK = "coordinate_stuck"
    GENERAL_FAILURE = "general_failure"

class EscalationLevel(Enum):
    """Escalation levels for intervention strategies."""
    NONE = 0
    MILD = 1      # Basic randomization, slight exploration boost
    MODERATE = 2  # Strategy override, pattern avoidance
    AGGRESSIVE = 3 # Complete strategy reset, maximum exploration

@dataclass
class LosingStreakData:
    """Data structure for tracking a losing streak."""
    streak_id: str
    game_type: str
    game_id: str
    level_identifier: Optional[str]
    consecutive_failures: int
    total_attempts: int
    first_failure_timestamp: float
    last_failure_timestamp: float
    failure_types: List[FailureType]
    escalation_level: EscalationLevel
    last_escalation_timestamp: Optional[float]
    intervention_attempts: int
    successful_intervention: bool
    streak_broken: bool
    break_timestamp: Optional[float]
    break_method: Optional[str]

class LosingStreakDetector:
    """
    Detects and tracks losing streaks across multiple game attempts.

    This system monitors consecutive failures for specific game/level combinations
    and triggers escalating intervention strategies when thresholds are reached.
    """

    def __init__(self, db_connection):
        """
        Initialize the losing streak detector.

        Args:
            db_connection: Database connection for persistence
        """
        self.db = db_connection
        self.active_streaks: Dict[str, LosingStreakData] = {}

        # Configuration thresholds
        self.mild_escalation_threshold = 3      # 3 consecutive failures
        self.moderate_escalation_threshold = 6  # 6 consecutive failures
        self.aggressive_escalation_threshold = 10 # 10 consecutive failures

        # Time thresholds (in seconds)
        self.streak_timeout = 3600  # 1 hour - consider streak stale after this
        self.escalation_cooldown = 300  # 5 minutes between escalations

        self._load_active_streaks()

    def _generate_streak_key(self, game_type: str, game_id: str, level_identifier: Optional[str] = None) -> str:
        """Generate a unique key for tracking streaks."""
        base_key = f"{game_type}:{game_id}"
        if level_identifier:
            base_key += f":{level_identifier}"
        return base_key

    def _generate_streak_id(self, game_type: str, game_id: str, level_identifier: Optional[str] = None) -> str:
        """Generate a unique streak ID for database storage."""
        timestamp = int(time.time())
        base_id = f"streak_{game_type}_{game_id}"
        if level_identifier:
            base_id += f"_{level_identifier}"
        return f"{base_id}_{timestamp}"

    def _load_active_streaks(self):
        """Load active (unbroken) streaks from database."""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT streak_id, game_type, game_id, level_identifier,
                       consecutive_failures, total_attempts, first_failure_timestamp,
                       last_failure_timestamp, failure_types, escalation_level,
                       last_escalation_timestamp, intervention_attempts,
                       successful_intervention, streak_broken, break_timestamp, break_method
                FROM losing_streaks
                WHERE streak_broken = FALSE
                AND last_failure_timestamp > ?
            """, (time.time() - self.streak_timeout,))

            for row in cursor.fetchall():
                streak_data = LosingStreakData(
                    streak_id=row[0],
                    game_type=row[1],
                    game_id=row[2],
                    level_identifier=row[3],
                    consecutive_failures=row[4],
                    total_attempts=row[5],
                    first_failure_timestamp=row[6],
                    last_failure_timestamp=row[7],
                    failure_types=[FailureType(ft) for ft in json.loads(row[8] or "[]")],
                    escalation_level=EscalationLevel(row[9]),
                    last_escalation_timestamp=row[10],
                    intervention_attempts=row[11],
                    successful_intervention=bool(row[12]),
                    streak_broken=bool(row[13]),
                    break_timestamp=row[14],
                    break_method=row[15]
                )

                streak_key = self._generate_streak_key(
                    streak_data.game_type,
                    streak_data.game_id,
                    streak_data.level_identifier
                )
                self.active_streaks[streak_key] = streak_data

            logger.info(f"Loaded {len(self.active_streaks)} active losing streaks")

        except Exception as e:
            logger.error(f"Error loading active streaks: {e}")

    def record_failure(self,
                      game_type: str,
                      game_id: str,
                      failure_type: FailureType,
                      level_identifier: Optional[str] = None,
                      context_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[LosingStreakData]]:
        """
        Record a game failure and update streak tracking.

        Args:
            game_type: Type of game (e.g., extracted from game_id)
            game_id: Unique identifier for the game
            failure_type: Type of failure that occurred
            level_identifier: Optional level identifier for games with levels
            context_data: Additional context about the failure

        Returns:
            Tuple of (streak_detected, streak_data)
        """
        try:
            streak_key = self._generate_streak_key(game_type, game_id, level_identifier)
            current_time = time.time()

            # Get or create streak data
            if streak_key in self.active_streaks:
                streak = self.active_streaks[streak_key]
                streak.consecutive_failures += 1
                streak.total_attempts += 1
                streak.last_failure_timestamp = current_time
                streak.failure_types.append(failure_type)
            else:
                # Create new streak
                streak_id = self._generate_streak_id(game_type, game_id, level_identifier)
                streak = LosingStreakData(
                    streak_id=streak_id,
                    game_type=game_type,
                    game_id=game_id,
                    level_identifier=level_identifier,
                    consecutive_failures=1,
                    total_attempts=1,
                    first_failure_timestamp=current_time,
                    last_failure_timestamp=current_time,
                    failure_types=[failure_type],
                    escalation_level=EscalationLevel.NONE,
                    last_escalation_timestamp=None,
                    intervention_attempts=0,
                    successful_intervention=False,
                    streak_broken=False,
                    break_timestamp=None,
                    break_method=None
                )
                self.active_streaks[streak_key] = streak

            # Update escalation level if needed
            self._update_escalation_level(streak)

            # Persist to database
            self._save_streak(streak)

            # Check if this constitutes a significant streak
            streak_detected = streak.consecutive_failures >= self.mild_escalation_threshold

            logger.info(f"Recorded failure for {streak_key}: {streak.consecutive_failures} consecutive failures, escalation level: {streak.escalation_level.name}")

            return streak_detected, streak

        except Exception as e:
            logger.error(f"Error recording failure: {e}")
            return False, None

    def record_success(self,
                      game_type: str,
                      game_id: str,
                      level_identifier: Optional[str] = None,
                      break_method: Optional[str] = None) -> bool:
        """
        Record a successful game completion, breaking any active streak.

        Args:
            game_type: Type of game
            game_id: Unique identifier for the game
            level_identifier: Optional level identifier
            break_method: Description of what finally worked

        Returns:
            True if a streak was broken, False otherwise
        """
        try:
            streak_key = self._generate_streak_key(game_type, game_id, level_identifier)

            if streak_key in self.active_streaks:
                streak = self.active_streaks[streak_key]
                streak.streak_broken = True
                streak.break_timestamp = time.time()
                streak.break_method = break_method or "successful_completion"

                # Save the broken streak
                self._save_streak(streak)

                # Remove from active streaks
                del self.active_streaks[streak_key]

                logger.info(f"Broke losing streak for {streak_key} after {streak.consecutive_failures} failures using method: {break_method}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error recording success: {e}")
            return False

    def _update_escalation_level(self, streak: LosingStreakData):
        """Update the escalation level based on consecutive failures."""
        current_time = time.time()

        # Check if we can escalate (respect cooldown)
        if (streak.last_escalation_timestamp and
            current_time - streak.last_escalation_timestamp < self.escalation_cooldown):
            return

        new_level = streak.escalation_level

        if streak.consecutive_failures >= self.aggressive_escalation_threshold:
            new_level = EscalationLevel.AGGRESSIVE
        elif streak.consecutive_failures >= self.moderate_escalation_threshold:
            new_level = EscalationLevel.MODERATE
        elif streak.consecutive_failures >= self.mild_escalation_threshold:
            new_level = EscalationLevel.MILD

        if new_level.value > streak.escalation_level.value:
            streak.escalation_level = new_level
            streak.last_escalation_timestamp = current_time
            logger.info(f"Escalated streak {streak.streak_id} to level {new_level.name}")

    def _save_streak(self, streak: LosingStreakData):
        """Save streak data to database."""
        try:
            cursor = self.db.cursor()

            # Convert failure types to JSON
            failure_types_json = json.dumps([ft.value for ft in streak.failure_types])

            cursor.execute("""
                INSERT OR REPLACE INTO losing_streaks (
                    streak_id, game_type, game_id, level_identifier,
                    consecutive_failures, total_attempts, first_failure_timestamp,
                    last_failure_timestamp, failure_types, escalation_level,
                    last_escalation_timestamp, intervention_attempts,
                    successful_intervention, streak_broken, break_timestamp, break_method,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                streak.streak_id, streak.game_type, streak.game_id, streak.level_identifier,
                streak.consecutive_failures, streak.total_attempts, streak.first_failure_timestamp,
                streak.last_failure_timestamp, failure_types_json, streak.escalation_level.value,
                streak.last_escalation_timestamp, streak.intervention_attempts,
                streak.successful_intervention, streak.streak_broken,
                streak.break_timestamp, streak.break_method, time.time()
            ))

            self.db.commit()

        except Exception as e:
            logger.error(f"Error saving streak: {e}")

    def get_active_streaks(self, escalation_level: Optional[EscalationLevel] = None) -> List[LosingStreakData]:
        """
        Get all active losing streaks, optionally filtered by escalation level.

        Args:
            escalation_level: Optional filter for specific escalation level

        Returns:
            List of active losing streaks
        """
        streaks = list(self.active_streaks.values())

        if escalation_level is not None:
            streaks = [s for s in streaks if s.escalation_level == escalation_level]

        return streaks

    def get_streak_for_game(self,
                           game_type: str,
                           game_id: str,
                           level_identifier: Optional[str] = None) -> Optional[LosingStreakData]:
        """
        Get the active streak for a specific game/level combination.

        Args:
            game_type: Type of game
            game_id: Unique identifier for the game
            level_identifier: Optional level identifier

        Returns:
            Active streak data or None
        """
        streak_key = self._generate_streak_key(game_type, game_id, level_identifier)
        return self.active_streaks.get(streak_key)

    def record_intervention_attempt(self,
                                  game_type: str,
                                  game_id: str,
                                  level_identifier: Optional[str] = None,
                                  success: bool = False):
        """
        Record an intervention attempt for a losing streak.

        Args:
            game_type: Type of game
            game_id: Unique identifier for the game
            level_identifier: Optional level identifier
            success: Whether the intervention was successful
        """
        try:
            streak_key = self._generate_streak_key(game_type, game_id, level_identifier)

            if streak_key in self.active_streaks:
                streak = self.active_streaks[streak_key]
                streak.intervention_attempts += 1
                if success:
                    streak.successful_intervention = True

                self._save_streak(streak)
                logger.info(f"Recorded intervention attempt for {streak_key}, success: {success}")

        except Exception as e:
            logger.error(f"Error recording intervention attempt: {e}")

    def cleanup_stale_streaks(self):
        """Remove stale streaks that haven't been updated recently."""
        try:
            current_time = time.time()
            stale_keys = []

            for key, streak in self.active_streaks.items():
                if current_time - streak.last_failure_timestamp > self.streak_timeout:
                    stale_keys.append(key)

            for key in stale_keys:
                del self.active_streaks[key]
                logger.info(f"Removed stale streak: {key}")

        except Exception as e:
            logger.error(f"Error cleaning up stale streaks: {e}")

    def get_streak_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about losing streaks."""
        try:
            cursor = self.db.cursor()

            # Active streaks stats
            active_count = len(self.active_streaks)
            active_by_level = {}
            for streak in self.active_streaks.values():
                level = streak.escalation_level.name
                active_by_level[level] = active_by_level.get(level, 0) + 1

            # Historical stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_streaks,
                    AVG(consecutive_failures) as avg_failures,
                    MAX(consecutive_failures) as max_failures,
                    COUNT(CASE WHEN streak_broken = TRUE THEN 1 END) as broken_streaks,
                    COUNT(CASE WHEN successful_intervention = TRUE THEN 1 END) as successful_interventions
                FROM losing_streaks
            """)

            historical = cursor.fetchone()

            return {
                "active_streaks": active_count,
                "active_by_escalation": active_by_level,
                "total_historical_streaks": historical[0] or 0,
                "average_failures_per_streak": historical[1] or 0.0,
                "maximum_failures_recorded": historical[2] or 0,
                "broken_streaks": historical[3] or 0,
                "successful_interventions": historical[4] or 0
            }

        except Exception as e:
            logger.error(f"Error getting streak statistics: {e}")
            return {"error": str(e)}