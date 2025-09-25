#!/usr/bin/env python3
"""
Penalty Decay System - Advanced coordinate penalty and recovery system.

This system implements:
- Penalty Decay: Coordinates that don't improve score get penalized
- Learning from Failures: System tracks and learns from both successes and failures
- Coordinate Diversity: Avoids recently used and failed coordinates
- Gradual Recovery: Penalties decay over time, allowing retry of failed actions
- Detailed Logging: Shows when penalties are applied and decayed
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import asyncio
from src.database.api import LogLevel, Component

from src.database.system_integration import get_system_integration
from src.core.penalty_logging_system import get_penalty_logging_system

logger = logging.getLogger(__name__)


class PenaltyDecaySystem:
    """
    Advanced penalty decay system that learns from failures and promotes coordinate diversity.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.logging_system = get_penalty_logging_system()
        
        # Penalty configuration
        self.penalty_config = {
            'no_improvement_penalty': 0.3,
            'score_decrease_penalty': 0.5,
            'stuck_loop_penalty': 0.8,
            'base_decay_rate': 0.1,
            'decay_acceleration': 0.05,
            'max_penalty': 1.0,
            'min_penalty': 0.0,
            'stuck_threshold': 5,  # Consecutive zero-progress attempts
            'diversity_window': 10,  # Recent attempts to consider for diversity
            'recovery_cooldown': 30  # Seconds before retry after penalty
        }
        
        # In-memory tracking for performance
        self.penalty_cache = {}
        self.diversity_cache = {}
        self.failure_patterns = {}
        
        # Performance metrics
        self.metrics = {
            'penalties_applied': 0,
            'penalties_decayed': 0,
            'coordinates_avoided': 0,
            'recoveries_attempted': 0,
            'successful_recoveries': 0
        }
    
    async def initialize(self):
        """Initialize the penalty decay system."""
        try:
            # Create tables if they don't exist
            await self._create_tables()
            
            # Load existing penalties and diversity data
            await self._load_cached_data()
            
            logger.info("Penalty Decay System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Penalty Decay System: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables for penalty system."""
        # Tables are created via schema.sql, but we can verify they exist
        pass
    
    async def _load_cached_data(self):
        """Load existing penalty and diversity data into cache."""
        try:
            # Load coordinate penalties
            penalty_data = await self.integration.db.fetch_all(
                "SELECT game_id, x, y, penalty_score, penalty_reason, is_stuck_coordinate, last_penalty_applied FROM coordinate_penalties"
            )
            
            for row in penalty_data:
                key = (row['game_id'], row['x'], row['y'])
                self.penalty_cache[key] = {
                    'penalty_score': row['penalty_score'],
                    'penalty_reason': row['penalty_reason'],
                    'is_stuck': row['is_stuck_coordinate'],
                    'last_penalty': row['last_penalty_applied']
                }
            
            # Load diversity data
            diversity_data = await self.integration.db.fetch_all(
                "SELECT game_id, x, y, avoidance_score, last_used FROM coordinate_diversity"
            )
            
            for row in diversity_data:
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
        context: Dict[str, Any] = None
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
            
        Returns:
            Dictionary with penalty information and recommendations
        """
        try:
            coord_key = (game_id, x, y)
            current_time = datetime.now()
            
            # Update coordinate intelligence
            await self._update_coordinate_intelligence(game_id, x, y, success, score_change)
            
            # Determine penalty type and amount
            penalty_info = await self._calculate_penalty(
                game_id, x, y, success, score_change, context or {}
            )
            
            # Apply penalty if needed
            if penalty_info['penalty_applied']:
                await self._apply_penalty(game_id, x, y, penalty_info)
                self.metrics['penalties_applied'] += 1
                
                # Log penalty application
                await self.logging_system.log_penalty_applied(
                    game_id=game_id,
                    coordinate=(x, y),
                    penalty_score=penalty_info['penalty_score'],
                    penalty_reason=penalty_info['penalty_reason'],
                    context=context
                )
                
                logger.info(f"Penalty applied to ({x},{y}): {penalty_info['penalty_reason']} "
                          f"(score: {penalty_info['penalty_score']:.3f})")
            
            # Update diversity tracking
            await self._update_diversity_tracking(game_id, x, y, current_time)
            
            # Record failure learning if applicable
            if not success or score_change <= 0:
                learned_insights = await self._record_failure_learning(
                    game_id, x, y, action_type, penalty_info['failure_type'], 
                    context or {}, score_change
                )
                
                # Log failure learning
                await self.logging_system.log_failure_learned(
                    game_id=game_id,
                    coordinate=(x, y),
                    failure_type=penalty_info['failure_type'],
                    learned_insights=learned_insights,
                    context=context
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
                'recovery_available': penalty_info['recovery_available']
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
        context: Dict[str, Any]
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
            'zero_progress_streak': zero_streak
        }
    
    async def _apply_penalty(self, game_id: str, x: int, y: int, penalty_info: Dict[str, Any]):
        """Apply penalty to coordinate in database."""
        
        try:
            # Update or insert penalty record
            try:
                await self.integration.db.execute(
                    """
                    INSERT OR REPLACE INTO coordinate_penalties 
                    (game_id, x, y, penalty_score, penalty_reason, zero_progress_streak, 
                     last_penalty_applied, is_stuck_coordinate, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        game_id, x, y, penalty_info['penalty_score'], penalty_info['penalty_reason'],
                        penalty_info['zero_progress_streak'], datetime.now(), 
                        penalty_info['is_stuck_coordinate'], datetime.now()
                    )
                )
            except Exception as e:
                self.logger.error(f" DEBUG: Error in coordinate_penalties insert: {e}")
                self.logger.error(f" DEBUG: Provided 9 values for 9 columns - this should be correct")
                raise
            
            # Log penalty application
            await self.integration.db.log_system_event(
                LogLevel.INFO, Component.LEARNING_LOOP, 
                f"Penalty applied to coordinate ({x},{y}): {penalty_info['penalty_reason']}",
                {
                    'game_id': game_id,
                    'coordinate': (x, y),
                    'penalty_score': penalty_info['penalty_score'],
                    'penalty_reason': penalty_info['penalty_reason'],
                    'is_stuck': penalty_info['is_stuck_coordinate']
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to apply penalty: {e}")
    
    async def _update_coordinate_intelligence(self, game_id: str, x: int, y: int, success: bool, score_change: float):
        """Update coordinate intelligence tracking."""
        
        try:
            # Update coordinate intelligence table
            await self.integration.db.execute(
                """
                INSERT OR REPLACE INTO coordinate_intelligence 
                (game_id, x, y, attempts, successes, success_rate, last_used, created_at, updated_at)
                VALUES (
                    ?, ?, ?, 
                    COALESCE((SELECT attempts FROM coordinate_intelligence WHERE game_id = ? AND x = ? AND y = ?), 0) + 1,
                    COALESCE((SELECT successes FROM coordinate_intelligence WHERE game_id = ? AND x = ? AND y = ?), 0) + ?,
                    (COALESCE((SELECT successes FROM coordinate_intelligence WHERE game_id = ? AND x = ? AND y = ?), 0) + ?) / 
                    (COALESCE((SELECT attempts FROM coordinate_intelligence WHERE game_id = ? AND x = ? AND y = ?), 0) + 1),
                    ?, ?, ?
                )
                """,
                (game_id, x, y, game_id, x, y, game_id, x, y, 1 if success else 0,
                 game_id, x, y, 1 if success else 0, game_id, x, y, 1 if success else 0,
                 datetime.now(), datetime.now())
            )
            
        except Exception as e:
            logger.error(f"Failed to update coordinate intelligence: {e}")
    
    async def _update_diversity_tracking(self, game_id: str, x: int, y: int, current_time: datetime):
        """Update coordinate diversity tracking."""
        
        try:
            # Update diversity record
            await self.integration.db.execute(
                """
                INSERT OR REPLACE INTO coordinate_diversity 
                (game_id, x, y, last_used, usage_frequency, updated_at)
                VALUES (
                    ?, ?, ?, ?,
                    COALESCE((SELECT usage_frequency FROM coordinate_diversity WHERE game_id = ? AND x = ? AND y = ?), 0) + 1,
                    ?
                )
                """,
                (game_id, x, y, current_time, game_id, x, y, current_time)
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
        score_change: float
    ) -> Dict[str, Any]:
        """Record failure for learning purposes."""
        
        try:
            # Generate learned insights
            insights = self._generate_failure_insights(failure_type, score_change, context)
            
            # Insert failure learning record
            await self.integration.db.execute(
                """
                INSERT INTO failure_learning 
                (game_id, coordinate_x, coordinate_y, action_type, failure_type, 
                 failure_context, learned_insights, last_failure)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id, x, y, action_type, failure_type,
                    json.dumps(context), json.dumps(insights), datetime.now()
                )
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to record failure learning: {e}")
            return {'error': str(e)}
    
    def _generate_failure_insights(self, failure_type: str, score_change: float, context: Dict[str, Any]) -> Dict[str, Any]:
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
        
        return insights
    
    async def _get_zero_progress_streak(self, game_id: str, x: int, y: int) -> int:
        """Get consecutive zero-progress attempts for coordinate."""
        
        try:
            # Get recent attempts from action traces
            recent_attempts = await self.integration.db.fetch_all(
                """
                SELECT score_change FROM action_traces 
                WHERE game_id = ? AND coordinates = ? 
                ORDER BY timestamp DESC LIMIT 10
                """,
                (game_id, json.dumps([x, y]))
            )
            
            streak = 0
            for attempt in recent_attempts:
                if attempt['score_change'] == 0:
                    streak += 1
                else:
                    break
            
            return streak
            
        except Exception as e:
            logger.error(f"Failed to get zero progress streak: {e}")
            return 0
    
    async def get_coordinate_penalty(self, game_id: str, x: int, y: int) -> Dict[str, Any]:
        """Get current penalty information for a coordinate."""
        
        coord_key = (game_id, x, y)
        
        # Check cache first
        if coord_key in self.penalty_cache:
            return self.penalty_cache[coord_key]
        
        # Load from database
        try:
            result = await self.integration.db.fetch_one(
                """
                SELECT penalty_score, penalty_reason, is_stuck_coordinate, 
                       last_penalty_applied, zero_progress_streak
                FROM coordinate_penalties 
                WHERE game_id = ? AND x = ? AND y = ?
                """,
                (game_id, x, y)
            )
            
            if result:
                penalty_data = result[0]
                return {
                    'penalty_score': penalty_data['penalty_score'],
                    'penalty_reason': penalty_data['penalty_reason'],
                    'is_stuck': penalty_data['is_stuck_coordinate'],
                    'last_penalty': penalty_data['last_penalty_applied'],
                    'zero_progress_streak': penalty_data['zero_progress_streak']
                }
            else:
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
                time_since_use = (datetime.now() - diversity_info['last_used']).total_seconds()
                diversity_score *= max(0.1, 1.0 - (time_since_use / 3600))  # Decay over 1 hour
            
            # Combined score (higher = more avoidable)
            combined_score = penalty_score + (diversity_score * 0.3)
            
            recommendations[(x, y)] = min(combined_score, 1.0)
        
        return recommendations
    
    async def decay_penalties(self, game_id: str = None) -> Dict[str, Any]:
        """Apply time-based penalty decay."""
        
        try:
            current_time = datetime.now()
            decayed_count = 0
            
            # Get coordinates with penalties
            query = "SELECT game_id, x, y, penalty_score, last_penalty_applied FROM coordinate_penalties"
            params = []
            
            if game_id:
                query += " WHERE game_id = ?"
                params.append(game_id)
            
            penalty_records = await self.integration.db.fetch_all(query, params)
            
            for record in penalty_records:
                last_penalty = record['last_penalty_applied']
                if not last_penalty:
                    continue
                
                # Calculate time since last penalty
                time_since_penalty = (current_time - last_penalty).total_seconds()
                
                # Apply decay if enough time has passed
                if time_since_penalty > 60:  # 1 minute minimum
                    decay_factor = min(1.0, time_since_penalty / 3600)  # Full decay after 1 hour
                    new_penalty = max(0.0, record['penalty_score'] * (1.0 - decay_factor))
                    
                    if new_penalty < record['penalty_score']:
                        # Update penalty in database
                        await self.integration.db.execute(
                            """
                            UPDATE coordinate_penalties 
                            SET penalty_score = ?, updated_at = ?
                            WHERE game_id = ? AND x = ? AND y = ?
                            """,
                            (new_penalty, current_time, record['game_id'], record['x'], record['y'])
                        )
                        
                        # Update cache
                        coord_key = (record['game_id'], record['x'], record['y'])
                        if coord_key in self.penalty_cache:
                            self.penalty_cache[coord_key]['penalty_score'] = new_penalty
                        
                        decayed_count += 1
                        
                        # Log decay event
                        await self.logging_system.log_penalty_decayed(
                            game_id=record['game_id'],
                            coordinate=(record['x'], record['y']),
                            old_penalty=record['penalty_score'],
                            new_penalty=new_penalty,
                            decay_factor=decay_factor,
                            context={'time_since_penalty': time_since_penalty}
                        )
                        
                        await self.integration.log_system_event(
                            level=LogLevel.INFO,
                            component=Component.LEARNING_LOOP,
                            message=f"Penalty decayed for coordinate ({record['x']},{record['y']}): {record['penalty_score']:.3f} -> {new_penalty:.3f}",
                            data={
                                'game_id': record['game_id'],
                                'coordinate': (record['x'], record['y']),
                                'old_penalty': record['penalty_score'],
                                'new_penalty': new_penalty,
                                'decay_factor': decay_factor
                            }
                        )
            
            self.metrics['penalties_decayed'] += decayed_count
            
            logger.info(f"Decayed {decayed_count} penalties")
            
            return {
                'decayed_count': decayed_count,
                'total_processed': len(penalty_records)
            }
            
        except Exception as e:
            logger.error(f"Failed to decay penalties: {e}")
            return {'error': str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics."""
        
        try:
            # Get penalty statistics
            penalty_stats = await self.integration.db.fetch_one(
                """
                SELECT 
                    COUNT(*) as total_penalties,
                    AVG(penalty_score) as avg_penalty,
                    MAX(penalty_score) as max_penalty,
                    COUNT(CASE WHEN is_stuck_coordinate = 1 THEN 1 END) as stuck_coordinates
                FROM coordinate_penalties
                """
            )
            
            # Get failure learning statistics
            failure_stats = await self.integration.db.fetch_all(
                """
                SELECT 
                    failure_type,
                    COUNT(*) as count,
                    AVG(failure_count) as avg_failures
                FROM failure_learning
                GROUP BY failure_type
                """
            )
            
            return {
                'metrics': self.metrics,
                'penalty_stats': penalty_stats[0] if penalty_stats else {},
                'failure_stats': {row['failure_type']: row for row in failure_stats},
                'cache_sizes': {
                    'penalty_cache': len(self.penalty_cache),
                    'diversity_cache': len(self.diversity_cache)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}


# Global instance
_penalty_system = None

def get_penalty_decay_system() -> PenaltyDecaySystem:
    """Get the global penalty decay system instance."""
    global _penalty_system
    if _penalty_system is None:
        _penalty_system = PenaltyDecaySystem()
    return _penalty_system
