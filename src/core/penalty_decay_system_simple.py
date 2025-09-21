#!/usr/bin/env python3
"""
Simplified Penalty Decay System - Works with current database API.

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

logger = logging.getLogger(__name__)


class SimplePenaltyDecaySystem:
    """
    Simplified penalty decay system that works with in-memory storage.
    """
    
    def __init__(self):
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
        
        # In-memory storage
        self.penalty_cache = {}
        self.diversity_cache = {}
        self.failure_patterns = {}
        self.coordinate_attempts = {}  # Track recent attempts for streak calculation
        
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
        logger.info("Simple Penalty Decay System initialized successfully")
    
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
        """
        try:
            coord_key = (game_id, x, y)
            current_time = datetime.now()
            
            # Track coordinate attempts for streak calculation
            if coord_key not in self.coordinate_attempts:
                self.coordinate_attempts[coord_key] = []
            
            self.coordinate_attempts[coord_key].append({
                'success': success,
                'score_change': score_change,
                'timestamp': current_time
            })
            
            # Keep only recent attempts
            if len(self.coordinate_attempts[coord_key]) > 10:
                self.coordinate_attempts[coord_key] = self.coordinate_attempts[coord_key][-10:]
            
            # Determine penalty type and amount
            penalty_info = await self._calculate_penalty(
                game_id, x, y, success, score_change, context or {}
            )
            
            # Apply penalty if needed
            if penalty_info['penalty_applied']:
                await self._apply_penalty(game_id, x, y, penalty_info)
                self.metrics['penalties_applied'] += 1
                
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
                
                logger.info(f"Failure learned for ({x},{y}): {penalty_info['failure_type']}")
            
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
        """Apply penalty to coordinate in memory."""
        coord_key = (game_id, x, y)
        
        self.penalty_cache[coord_key] = {
            'penalty_score': penalty_info['penalty_score'],
            'penalty_reason': penalty_info['penalty_reason'],
            'is_stuck': penalty_info['is_stuck_coordinate'],
            'last_penalty': datetime.now(),
            'zero_progress_streak': penalty_info['zero_progress_streak']
        }
        
        logger.info(f"Penalty applied to coordinate ({x},{y}): {penalty_info['penalty_reason']} "
                  f"(score: {penalty_info['penalty_score']:.3f})")
    
    async def _update_diversity_tracking(self, game_id: str, x: int, y: int, current_time: datetime):
        """Update coordinate diversity tracking."""
        coord_key = (game_id, x, y)
        
        if coord_key not in self.diversity_cache:
            self.diversity_cache[coord_key] = {
                'avoidance_score': 0.0,
                'last_used': current_time,
                'usage_frequency': 0
            }
        
        self.diversity_cache[coord_key]['last_used'] = current_time
        self.diversity_cache[coord_key]['usage_frequency'] += 1
    
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
        
        # Generate learned insights
        insights = self._generate_failure_insights(failure_type, score_change, context)
        
        # Store failure pattern
        coord_key = (game_id, x, y)
        if coord_key not in self.failure_patterns:
            self.failure_patterns[coord_key] = []
        
        self.failure_patterns[coord_key].append({
            'failure_type': failure_type,
            'action_type': action_type,
            'score_change': score_change,
            'timestamp': datetime.now(),
            'insights': insights
        })
        
        return insights
    
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
        coord_key = (game_id, x, y)
        
        if coord_key not in self.coordinate_attempts:
            return 0
        
        attempts = self.coordinate_attempts[coord_key]
        streak = 0
        
        # Count backwards from most recent attempts
        for attempt in reversed(attempts):
            if attempt['score_change'] == 0:
                streak += 1
            else:
                break
        
        return streak
    
    async def get_coordinate_penalty(self, game_id: str, x: int, y: int) -> Dict[str, Any]:
        """Get current penalty information for a coordinate."""
        coord_key = (game_id, x, y)
        
        if coord_key in self.penalty_cache:
            return self.penalty_cache[coord_key]
        else:
            return {
                'penalty_score': 0.0,
                'penalty_reason': 'none',
                'is_stuck': False,
                'last_penalty': None,
                'zero_progress_streak': 0
            }
    
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
            penalty_records = []
            for coord_key, penalty_data in self.penalty_cache.items():
                if game_id is None or coord_key[0] == game_id:
                    penalty_records.append({
                        'game_id': coord_key[0],
                        'x': coord_key[1],
                        'y': coord_key[2],
                        'penalty_score': penalty_data['penalty_score'],
                        'last_penalty_applied': penalty_data['last_penalty']
                    })
            
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
                        # Update penalty in cache
                        coord_key = (record['game_id'], record['x'], record['y'])
                        if coord_key in self.penalty_cache:
                            self.penalty_cache[coord_key]['penalty_score'] = new_penalty
                        
                        decayed_count += 1
                        
                        logger.info(f"Penalty decayed for coordinate ({record['x']},{record['y']}): "
                                  f"{record['penalty_score']:.3f} -> {new_penalty:.3f}")
            
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
            # Calculate penalty statistics
            penalty_scores = [data['penalty_score'] for data in self.penalty_cache.values()]
            
            penalty_stats = {
                'total_penalties': len(self.penalty_cache),
                'avg_penalty': sum(penalty_scores) / len(penalty_scores) if penalty_scores else 0.0,
                'max_penalty': max(penalty_scores) if penalty_scores else 0.0,
                'min_penalty': min(penalty_scores) if penalty_scores else 0.0,
                'stuck_coordinates': sum(1 for data in self.penalty_cache.values() if data.get('is_stuck', False))
            }
            
            # Calculate failure statistics
            failure_stats = {}
            for coord_key, failures in self.failure_patterns.items():
                for failure in failures:
                    failure_type = failure['failure_type']
                    if failure_type not in failure_stats:
                        failure_stats[failure_type] = {'count': 0, 'avg_failures': 0}
                    failure_stats[failure_type]['count'] += 1
            
            # Calculate average failures per coordinate
            for failure_type in failure_stats:
                coords_with_failures = sum(1 for failures in self.failure_patterns.values() 
                                         if any(f['failure_type'] == failure_type for f in failures))
                failure_stats[failure_type]['avg_failures'] = (
                    failure_stats[failure_type]['count'] / coords_with_failures if coords_with_failures > 0 else 0
                )
            
            return {
                'metrics': self.metrics,
                'penalty_stats': penalty_stats,
                'failure_stats': failure_stats,
                'cache_sizes': {
                    'penalty_cache': len(self.penalty_cache),
                    'diversity_cache': len(self.diversity_cache),
                    'failure_patterns': len(self.failure_patterns)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}


# Global instance
_simple_penalty_system = None

def get_simple_penalty_decay_system() -> SimplePenaltyDecaySystem:
    """Get the global simple penalty decay system instance."""
    global _simple_penalty_system
    if _simple_penalty_system is None:
        _simple_penalty_system = SimplePenaltyDecaySystem()
    return _simple_penalty_system
