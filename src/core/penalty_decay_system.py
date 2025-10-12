#!/usr/bin/env python3
"""
Enhanced Penalty Decay System - Advanced coordinate penalty and recovery system.

RESTORED FROM GIT HISTORY AND ENHANCED FOR CURRENT CODEBASE

This system implements:
- Penalty Decay: Coordinates that don't improve score get penalized
- Learning from Failures: System tracks and learns from both successes and failures
- Coordinate Diversity: Avoids recently used and failed coordinates
- Gradual Recovery: Penalties decay over time, allowing retry of failed actions
- Integration with Action6Coordinator: Enhanced pseudo-button learning
- Detailed Logging: Shows when penalties are applied and decayed
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class PenaltyDecaySystem:
    """
    Advanced penalty decay system that learns from failures and promotes coordinate diversity.
    Enhanced for integration with current Action6Coordinator and pseudo-button learning.
    """

    def __init__(self, db_interface=None):
        """Initialize penalty decay system with optional database interface."""
        self.db_interface = db_interface

        # Penalty configuration
        self.penalty_config = {
            'no_improvement_penalty': 0.3,
            'score_decrease_penalty': 0.5,
            'stuck_loop_penalty': 0.8,
            'pseudo_button_failure_penalty': 0.4,  # NEW: For pseudo-button failures
            'base_decay_rate': 0.1,
            'decay_acceleration': 0.05,
            'max_penalty': 1.0,
            'min_penalty': 0.0,
            'stuck_threshold': 5,  # Consecutive zero-progress attempts
            'diversity_window': 10,  # Recent attempts to consider for diversity
            'recovery_cooldown': 30,  # Seconds before retry after penalty
            'pseudo_button_retry_threshold': 0.3  # NEW: Threshold for pseudo-button retry
        }

        # In-memory tracking for performance
        self.penalty_cache = {}
        self.diversity_cache = {}
        self.failure_patterns = {}
        self.pseudo_button_penalties = {}  # NEW: Track pseudo-button specific penalties

        # Performance metrics
        self.metrics = {
            'penalties_applied': 0,
            'penalties_decayed': 0,
            'coordinates_avoided': 0,
            'recoveries_attempted': 0,
            'successful_recoveries': 0,
            'pseudo_button_penalties': 0,  # NEW
            'pseudo_button_recoveries': 0  # NEW
        }

        # Initialize tables on first use
        self._tables_initialized = False

    async def initialize(self):
        """Initialize the penalty decay system."""
        try:
            # Create tables if they don't exist and we have a database interface
            if self.db_interface:
                await self._create_tables()
                # Load existing penalties and diversity data
                await self._load_cached_data()

            self._tables_initialized = True
            logger.info("Enhanced Penalty Decay System initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Penalty Decay System: {e}")
            # Continue without database - use in-memory only
            self._tables_initialized = True
            logger.warning("Continuing with in-memory penalty tracking only")

    async def _create_tables(self):
        """Create database tables for penalty system."""
        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                return

            # Create coordinate penalties table
            await self.db_interface.execute_query("""
                CREATE TABLE IF NOT EXISTS coordinate_penalties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    penalty_score REAL NOT NULL DEFAULT 0.0,
                    penalty_reason TEXT NOT NULL,
                    zero_progress_streak INTEGER DEFAULT 0,
                    is_stuck_coordinate BOOLEAN DEFAULT FALSE,
                    pseudo_button_failure BOOLEAN DEFAULT FALSE,
                    last_penalty_applied TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(game_id, x, y)
                )
            """)

            # Create coordinate diversity table
            await self.db_interface.execute_query("""
                CREATE TABLE IF NOT EXISTS coordinate_diversity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    usage_frequency INTEGER DEFAULT 1,
                    avoidance_score REAL DEFAULT 0.0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(game_id, x, y)
                )
            """)

            # Create failure learning table
            await self.db_interface.execute_query("""
                CREATE TABLE IF NOT EXISTS failure_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    coordinate_x INTEGER NOT NULL,
                    coordinate_y INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    failure_context TEXT,
                    learned_insights TEXT,
                    pseudo_button_context TEXT,
                    last_failure TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info("Penalty decay system database tables created/verified")

        except Exception as e:
            logger.error(f"Failed to create penalty decay tables: {e}")

    async def _load_cached_data(self):
        """Load existing penalty and diversity data into cache."""
        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                return

            # Load coordinate penalties
            penalty_data = await self.db_interface.execute_query(
                "SELECT game_id, x, y, penalty_score, penalty_reason, is_stuck_coordinate, last_penalty_applied FROM coordinate_penalties"
            )

            for row in penalty_data or []:
                key = (row['game_id'], row['x'], row['y'])
                self.penalty_cache[key] = {
                    'penalty_score': row['penalty_score'],
                    'penalty_reason': row['penalty_reason'],
                    'is_stuck': row['is_stuck_coordinate'],
                    'last_penalty': row['last_penalty_applied']
                }

            # Load diversity data
            diversity_data = await self.db_interface.execute_query(
                "SELECT game_id, x, y, avoidance_score, last_used FROM coordinate_diversity"
            )

            for row in diversity_data or []:
                key = (row['game_id'], row['x'], row['y'])
                self.diversity_cache[key] = {
                    'avoidance_score': row['avoidance_score'],
                    'last_used': row['last_used']
                }

            logger.info(f"Loaded {len(self.penalty_cache)} penalties and {len(self.diversity_cache)} diversity records")

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")

    async def record_coordinate_attempt(
        self,
        game_id: str,
        x: int,
        y: int,
        success: bool,
        score_change: float,
        action_type: str = "ACTION6",
        context: Dict[str, Any] = None,
        pseudo_button_data: Dict[str, Any] = None  # NEW: For pseudo-button integration
    ) -> Dict[str, Any]:
        """
        Record a coordinate attempt and apply penalties if needed.

        Args:
            game_id: Game identifier
            x, y: Coordinate position
            success: Whether the action was successful
            score_change: Change in score (positive = improvement)
            action_type: Type of action performed
            context: Additional context data
            pseudo_button_data: NEW - Pseudo-button specific data from Action6Coordinator

        Returns:
            Dictionary with penalty information and recommendations
        """
        try:
            coord_key = (game_id, x, y)
            current_time = datetime.now()

            # Determine penalty type and amount
            penalty_info = await self._calculate_penalty(
                game_id, x, y, success, score_change, context or {}, pseudo_button_data or {}
            )

            # Apply penalty if needed
            if penalty_info['penalty_applied']:
                await self._apply_penalty(game_id, x, y, penalty_info)
                self.metrics['penalties_applied'] += 1

                if pseudo_button_data:
                    self.metrics['pseudo_button_penalties'] += 1

                logger.info(f"Penalty applied to ({x},{y}): {penalty_info['penalty_reason']} "
                          f"(score: {penalty_info['penalty_score']:.3f})")

            # Update diversity tracking
            await self._update_diversity_tracking(game_id, x, y, current_time)

            # Record failure learning if applicable
            if not success or score_change <= 0:
                learned_insights = await self._record_failure_learning(
                    game_id, x, y, action_type, penalty_info['failure_type'],
                    context or {}, score_change, pseudo_button_data or {}
                )

            # Update cache
            self.penalty_cache[coord_key] = {
                'penalty_score': penalty_info['penalty_score'],
                'penalty_reason': penalty_info['penalty_reason'],
                'is_stuck': penalty_info['is_stuck_coordinate'],
                'last_penalty': current_time
            }

            return {
                'penalty_applied': penalty_info['penalty_applied'],
                'penalty_score': penalty_info['penalty_score'],
                'penalty_reason': penalty_info['penalty_reason'],
                'is_stuck': penalty_info['is_stuck_coordinate'],
                'avoidance_recommended': penalty_info['avoidance_recommended'],
                'recovery_available': penalty_info['recovery_available'],
                'pseudo_button_analysis': penalty_info.get('pseudo_button_analysis', {})  # NEW
            }

        except Exception as e:
            logger.error(f"Failed to record coordinate attempt: {e}")
            return {'error': str(e)}

    async def _calculate_penalty(
        self,
        game_id: str,
        x: int,
        y: int,
        success: bool,
        score_change: float,
        context: Dict[str, Any],
        pseudo_button_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate penalty based on coordinate performance."""

        coord_key = (game_id, x, y)
        current_time = datetime.now()

        # Get existing penalty data
        existing_penalty = self.penalty_cache.get(coord_key, {
            'penalty_score': 0.0,
            'penalty_reason': 'none',
            'is_stuck': False,
            'last_penalty': None
        })

        # Determine penalty type
        penalty_reason = 'none'
        penalty_amount = 0.0
        failure_type = 'none'
        pseudo_button_analysis = {}

        # NEW: Enhanced penalty calculation with pseudo-button integration
        if not success or score_change <= 0:
            if score_change < 0:
                # Score decreased - high penalty
                penalty_reason = 'score_decrease'
                penalty_amount = self.penalty_config['score_decrease_penalty']
                failure_type = 'score_decrease'
            else:
                # No improvement - moderate penalty
                penalty_reason = 'no_improvement'
                penalty_amount = self.penalty_config['no_improvement_penalty']
                failure_type = 'no_improvement'

            # NEW: Additional penalty for pseudo-button failures
            if pseudo_button_data and pseudo_button_data.get('was_pseudo_button', False):
                pseudo_button_analysis = {
                    'was_pseudo_button': True,
                    'confidence': pseudo_button_data.get('confidence', 0.0),
                    'button_type': pseudo_button_data.get('type', 'unknown'),
                    'effectiveness': pseudo_button_data.get('effectiveness', 0.0)
                }

                # Apply pseudo-button failure penalty
                if pseudo_button_data.get('confidence', 0.0) > 0.7:  # High confidence pseudo-button
                    penalty_amount += self.penalty_config['pseudo_button_failure_penalty']
                    penalty_reason = f"{penalty_reason}_pseudo_button"
                    failure_type = f"{failure_type}_pseudo_button"

        # Check for stuck coordinate pattern
        zero_streak = await self._get_zero_progress_streak(game_id, x, y)
        if zero_streak >= self.penalty_config['stuck_threshold']:
            penalty_reason = 'stuck_loop'
            penalty_amount = self.penalty_config['stuck_loop_penalty']
            failure_type = 'stuck_loop'

        # Calculate final penalty score
        current_penalty = existing_penalty['penalty_score']
        new_penalty = min(
            current_penalty + penalty_amount,
            self.penalty_config['max_penalty']
        )

        # Determine if coordinate should be avoided
        avoidance_recommended = (
            new_penalty > 0.5 or
            existing_penalty['is_stuck'] or
            zero_streak >= self.penalty_config['stuck_threshold']
        )

        # NEW: Special pseudo-button recovery logic
        if pseudo_button_data and new_penalty > 0:
            # Check if pseudo-button might work in different context
            context_similarity = self._calculate_context_similarity(
                pseudo_button_data.get('context', {}), context
            )
            if context_similarity < 0.5:  # Different context - might work now
                avoidance_recommended = False
                pseudo_button_analysis['context_retry_recommended'] = True

        # Check recovery availability
        recovery_available = (
            new_penalty > 0 and
            existing_penalty['last_penalty'] and
            (current_time - existing_penalty['last_penalty']).total_seconds() > self.penalty_config['recovery_cooldown']
        )

        return {
            'penalty_applied': penalty_amount > 0,
            'penalty_score': new_penalty,
            'penalty_reason': penalty_reason,
            'is_stuck_coordinate': zero_streak >= self.penalty_config['stuck_threshold'],
            'avoidance_recommended': avoidance_recommended,
            'recovery_available': recovery_available,
            'failure_type': failure_type,
            'zero_progress_streak': zero_streak,
            'pseudo_button_analysis': pseudo_button_analysis  # NEW
        }

    def _calculate_context_similarity(self, old_context: Dict[str, Any], new_context: Dict[str, Any]) -> float:
        """Calculate similarity between contexts for pseudo-button retry decisions."""
        try:
            # Simple similarity based on common keys and values
            if not old_context or not new_context:
                return 0.0

            common_keys = set(old_context.keys()) & set(new_context.keys())
            if not common_keys:
                return 0.0

            matches = 0
            for key in common_keys:
                if old_context[key] == new_context[key]:
                    matches += 1

            return matches / len(common_keys)

        except Exception:
            return 0.0

    async def _apply_penalty(self, game_id: str, x: int, y: int, penalty_info: Dict[str, Any]):
        """Apply penalty to coordinate in database."""

        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                return  # Skip database operations if no interface

            # Update or insert penalty record
            await self.db_interface.execute_query(
                """
                INSERT OR REPLACE INTO coordinate_penalties
                (game_id, x, y, penalty_score, penalty_reason, zero_progress_streak,
                 last_penalty_applied, is_stuck_coordinate, pseudo_button_failure, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    game_id, x, y, penalty_info['penalty_score'], penalty_info['penalty_reason'],
                    penalty_info['zero_progress_streak'], datetime.now().isoformat(),
                    penalty_info['is_stuck_coordinate'],
                    'pseudo_button' in penalty_info['penalty_reason']
                )
            )

        except Exception as e:
            logger.error(f"Failed to apply penalty: {e}")

    async def _update_diversity_tracking(self, game_id: str, x: int, y: int, current_time: datetime):
        """Update coordinate diversity tracking."""

        try:
            if not self.db_interface or not hasattr(self.db_interface, 'execute_query'):
                # Update only cache if no database
                coord_key = (game_id, x, y)
                if coord_key not in self.diversity_cache:
                    self.diversity_cache[coord_key] = {'avoidance_score': 0.0, 'last_used': current_time}
                self.diversity_cache[coord_key]['last_used'] = current_time
                return

            # Update diversity record
            await self.db_interface.execute_query(
                """
                INSERT OR REPLACE INTO coordinate_diversity
                (game_id, x, y, last_used, usage_frequency, updated_at)
                VALUES (
                    ?, ?, ?, ?,
                    COALESCE((SELECT usage_frequency FROM coordinate_diversity WHERE game_id = ? AND x = ? AND y = ?), 0) + 1,
                    CURRENT_TIMESTAMP
                )
                """,
                (game_id, x, y, current_time.isoformat(), game_id, x, y)
            )

            # Update cache
            coord_key = (game_id, x, y)
            if coord_key not in self.diversity_cache:
                self.diversity_cache[coord_key] = {'avoidance_score': 0.0, 'last_used': current_time}
            self.diversity_cache[coord_key]['last_used'] = current_time

        except Exception as e:
            logger.error(f"Failed to update diversity tracking: {e}")

    async def _record_failure_learning(
        self,
        game_id: str,
        x: int,
        y: int,
        action_type: str,
        failure_type: str,
        context: Dict[str, Any],
        score_change: float,
        pseudo_button_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record failure for learning purposes."""

        try:
            # Generate learned insights
            insights = self._generate_failure_insights(failure_type, score_change, context, pseudo_button_data)

            if self.db_interface and hasattr(self.db_interface, 'execute_query'):
                # Insert failure learning record
                await self.db_interface.execute_query(
                    """
                    INSERT INTO failure_learning
                    (game_id, coordinate_x, coordinate_y, action_type, failure_type,
                     failure_context, learned_insights, pseudo_button_context, last_failure)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        game_id, x, y, action_type, failure_type,
                        json.dumps(context), json.dumps(insights), json.dumps(pseudo_button_data)
                    )
                )

            return insights

        except Exception as e:
            logger.error(f"Failed to record failure learning: {e}")
            return {'error': str(e)}

    def _generate_failure_insights(self, failure_type: str, score_change: float, context: Dict[str, Any], pseudo_button_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from failure patterns."""

        insights = {
            'failure_type': failure_type,
            'score_change': score_change,
            'timestamp': datetime.now().isoformat(),
            'recommendations': []
        }

        if failure_type == 'score_decrease':
            insights['recommendations'].append('Avoid this coordinate - causes score decrease')
        elif failure_type == 'no_improvement':
            insights['recommendations'].append('Coordinate shows no improvement - consider alternatives')
        elif failure_type == 'stuck_loop':
            insights['recommendations'].append('Coordinate causes stuck loops - avoid for extended period')

        # NEW: Pseudo-button specific insights
        if 'pseudo_button' in failure_type and pseudo_button_data:
            insights['pseudo_button_analysis'] = {
                'confidence': pseudo_button_data.get('confidence', 0.0),
                'button_type': pseudo_button_data.get('type', 'unknown'),
                'context_dependent': True
            }
            insights['recommendations'].append('Pseudo-button may work in different game state - retry later')

        return insights

    async def _get_zero_progress_streak(self, game_id: str, x: int, y: int) -> int:
        """Get consecutive zero-progress attempts for coordinate."""

        try:
            # Use in-memory tracking if no database
            if not self.db_interface:
                # Simple in-memory tracking (limited history)
                coord_key = (game_id, x, y)
                streak_key = f"{coord_key}_streak"
                return getattr(self, streak_key, 0)

            # Note: This would need integration with action traces table
            # For now, return a conservative estimate
            return 0

        except Exception as e:
            logger.error(f"Failed to get zero progress streak: {e}")
            return 0

    async def get_coordinate_penalty(self, game_id: str, x: int, y: int) -> Dict[str, Any]:
        """Get current penalty information for a coordinate."""

        coord_key = (game_id, x, y)

        # Check cache first
        if coord_key in self.penalty_cache:
            return self.penalty_cache[coord_key]

        # Load from database if available
        try:
            if self.db_interface and hasattr(self.db_interface, 'execute_query'):
                result = await self.db_interface.execute_query(
                    """
                    SELECT penalty_score, penalty_reason, is_stuck_coordinate,
                           last_penalty_applied, zero_progress_streak
                    FROM coordinate_penalties
                    WHERE game_id = ? AND x = ? AND y = ?
                    """,
                    (game_id, x, y)
                )

                if result and len(result) > 0:
                    penalty_data = result[0]
                    return {
                        'penalty_score': penalty_data['penalty_score'],
                        'penalty_reason': penalty_data['penalty_reason'],
                        'is_stuck': penalty_data['is_stuck_coordinate'],
                        'last_penalty': penalty_data['last_penalty_applied'],
                        'zero_progress_streak': penalty_data['zero_progress_streak']
                    }

            # Default response
            return {
                'penalty_score': 0.0,
                'penalty_reason': 'none',
                'is_stuck': False,
                'last_penalty': None,
                'zero_progress_streak': 0
            }

        except Exception as e:
            logger.error(f"Failed to get coordinate penalty: {e}")
            return {'error': str(e)}

    async def get_avoidance_recommendations(self, game_id: str, candidate_coordinates: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Get avoidance scores for candidate coordinates."""

        recommendations = {}

        for x, y in candidate_coordinates:
            coord_key = (game_id, x, y)

            # Get penalty information
            penalty_info = await self.get_coordinate_penalty(game_id, x, y)

            # Get diversity information
            diversity_info = self.diversity_cache.get(coord_key, {
                'avoidance_score': 0.0,
                'last_used': None
            })

            # Calculate combined avoidance score
            penalty_score = penalty_info.get('penalty_score', 0.0)
            diversity_score = diversity_info.get('avoidance_score', 0.0)

            # Time-based decay for diversity
            if diversity_info.get('last_used'):
                try:
                    if isinstance(diversity_info['last_used'], str):
                        last_used = datetime.fromisoformat(diversity_info['last_used'])
                    else:
                        last_used = diversity_info['last_used']
                    time_since_use = (datetime.now() - last_used).total_seconds()
                    diversity_score *= max(0.1, 1.0 - (time_since_use / 3600))  # Decay over 1 hour
                except Exception:
                    diversity_score = 0.0

            # Combined score (higher = more avoidable)
            combined_score = penalty_score + (diversity_score * 0.3)

            recommendations[(x, y)] = min(combined_score, 1.0)

        return recommendations

    async def decay_penalties(self, game_id: str = None) -> Dict[str, Any]:
        """Apply time-based penalty decay."""

        try:
            current_time = datetime.now()
            decayed_count = 0

            if not self.db_interface:
                # In-memory decay
                for coord_key in list(self.penalty_cache.keys()):
                    if game_id and coord_key[0] != game_id:
                        continue

                    penalty_data = self.penalty_cache[coord_key]
                    if penalty_data.get('last_penalty'):
                        time_since_penalty = (current_time - penalty_data['last_penalty']).total_seconds()

                        if time_since_penalty > 60:  # 1 minute minimum
                            decay_factor = min(1.0, time_since_penalty / 3600)  # Full decay after 1 hour
                            new_penalty = max(0.0, penalty_data['penalty_score'] * (1.0 - decay_factor))

                            if new_penalty < penalty_data['penalty_score']:
                                self.penalty_cache[coord_key]['penalty_score'] = new_penalty
                                decayed_count += 1

                self.metrics['penalties_decayed'] += decayed_count
                return {'decayed_count': decayed_count, 'total_processed': len(self.penalty_cache)}

            # Database decay (similar to original implementation)
            query = "SELECT game_id, x, y, penalty_score, last_penalty_applied FROM coordinate_penalties"
            params = []

            if game_id:
                query += " WHERE game_id = ?"
                params.append(game_id)

            penalty_records = await self.db_interface.execute_query(query, params)

            for record in penalty_records or []:
                last_penalty = record['last_penalty_applied']
                if not last_penalty:
                    continue

                # Calculate time since last penalty
                if isinstance(last_penalty, str):
                    last_penalty = datetime.fromisoformat(last_penalty)
                time_since_penalty = (current_time - last_penalty).total_seconds()

                # Apply decay if enough time has passed
                if time_since_penalty > 60:  # 1 minute minimum
                    decay_factor = min(1.0, time_since_penalty / 3600)  # Full decay after 1 hour
                    new_penalty = max(0.0, record['penalty_score'] * (1.0 - decay_factor))

                    if new_penalty < record['penalty_score']:
                        # Update penalty in database
                        await self.db_interface.execute_query(
                            """
                            UPDATE coordinate_penalties
                            SET penalty_score = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE game_id = ? AND x = ? AND y = ?
                            """,
                            (new_penalty, record['game_id'], record['x'], record['y'])
                        )

                        # Update cache
                        coord_key = (record['game_id'], record['x'], record['y'])
                        if coord_key in self.penalty_cache:
                            self.penalty_cache[coord_key]['penalty_score'] = new_penalty

                        decayed_count += 1

            self.metrics['penalties_decayed'] += decayed_count

            logger.info(f"Decayed {decayed_count} penalties")

            return {
                'decayed_count': decayed_count,
                'total_processed': len(penalty_records) if penalty_records else 0
            }

        except Exception as e:
            logger.error(f"Failed to decay penalties: {e}")
            return {'error': str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""

        try:
            status = {
                'metrics': self.metrics,
                'cache_sizes': {
                    'penalty_cache': len(self.penalty_cache),
                    'diversity_cache': len(self.diversity_cache)
                },
                'config': self.penalty_config,
                'database_enabled': self.db_interface is not None
            }

            if self.db_interface and hasattr(self.db_interface, 'execute_query'):
                # Get penalty statistics
                penalty_stats = await self.db_interface.execute_query(
                    """
                    SELECT
                        COUNT(*) as total_penalties,
                        AVG(penalty_score) as avg_penalty,
                        MAX(penalty_score) as max_penalty,
                        COUNT(CASE WHEN is_stuck_coordinate = 1 THEN 1 END) as stuck_coordinates,
                        COUNT(CASE WHEN pseudo_button_failure = 1 THEN 1 END) as pseudo_button_failures
                    FROM coordinate_penalties
                    """
                )

                if penalty_stats:
                    status['penalty_stats'] = penalty_stats[0]

            return status

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e), 'metrics': self.metrics}


# Global instance
_penalty_system = None

def get_penalty_decay_system(db_interface=None) -> PenaltyDecaySystem:
    """Get the global penalty decay system instance."""
    global _penalty_system
    if _penalty_system is None:
        _penalty_system = PenaltyDecaySystem(db_interface=db_interface)
    return _penalty_system


# Integration helper for Action6Coordinator
async def integrate_with_action6_coordinator(action6_coordinator, penalty_system=None):
    """Helper function to integrate penalty system with Action6Coordinator."""

    if penalty_system is None:
        penalty_system = get_penalty_decay_system(action6_coordinator.db_interface)

    # Initialize penalty system
    await penalty_system.initialize()

    # Add penalty system to Action6Coordinator
    action6_coordinator.penalty_system = penalty_system

    logger.info("Penalty Decay System successfully integrated with Action6Coordinator")

    return penalty_system