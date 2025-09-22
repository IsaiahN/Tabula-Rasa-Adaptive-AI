#!/usr/bin/env python3
"""
Emergency Override Systems

This module implements multiple levels of stuck detection and recovery mechanisms
for when the system gets stuck in loops or fails to make progress.
"""

import logging
import json
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class OverrideType(Enum):
    """Types of emergency overrides that can be triggered."""
    ACTION_LOOP_BREAK = "action_loop_break"
    COORDINATE_STUCK_BREAK = "coordinate_stuck_break"
    STAGNATION_BREAK = "stagnation_break"
    EMERGENCY_RESET = "emergency_reset"

@dataclass
class EmergencyOverride:
    """Represents an emergency override event."""
    game_id: str
    session_id: str
    override_type: OverrideType
    trigger_reason: str
    actions_before_override: int
    override_action: int
    override_successful: bool
    override_timestamp: float

class EmergencyOverrideSystem:
    """
    Emergency Override Systems
    
    Implements multiple levels of stuck detection and recovery mechanisms
    to prevent the system from getting permanently stuck.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.override_history: Dict[str, List[EmergencyOverride]] = {}
        self.override_triggers: Dict[str, Dict[str, Any]] = {}
        
    async def check_emergency_override(self, 
                                     game_id: str,
                                     session_id: str,
                                     current_state: Dict[str, Any],
                                     action_history: List[int],
                                     performance_history: List[Dict[str, Any]],
                                     available_actions: List[int]) -> Optional[EmergencyOverride]:
        """
        Check if emergency override should be triggered.
        
        Args:
            game_id: Game identifier
            session_id: Session identifier
            current_state: Current game state
            action_history: Recent action history
            performance_history: Recent performance data
            available_actions: Available actions
            
        Returns:
            EmergencyOverride if override should be triggered, None otherwise
        """
        try:
            # Check different types of emergency conditions
            override_checks = [
                await self._check_action_loop_break(game_id, action_history, available_actions),
                await self._check_coordinate_stuck_break(game_id, current_state),
                await self._check_stagnation_break(game_id, performance_history),
                await self._check_emergency_reset(game_id, action_history, performance_history)
            ]
            
            # Find the most critical override
            valid_overrides = [o for o in override_checks if o is not None]
            if not valid_overrides:
                return None
            
            # Select the most critical override (highest priority)
            override_priority = {
                OverrideType.EMERGENCY_RESET: 4,
                OverrideType.STAGNATION_BREAK: 3,
                OverrideType.ACTION_LOOP_BREAK: 2,
                OverrideType.COORDINATE_STUCK_BREAK: 1
            }
            
            most_critical = max(valid_overrides, key=lambda o: override_priority.get(o.override_type, 0))
            
            # Store override in database
            await self._store_emergency_override(most_critical)
            
            # Update local history
            if game_id not in self.override_history:
                self.override_history[game_id] = []
            self.override_history[game_id].append(most_critical)
            
            logger.warning(f"🚨 EMERGENCY OVERRIDE TRIGGERED: {most_critical.override_type.value} - "
                          f"{most_critical.trigger_reason}")
            
            return most_critical
            
        except Exception as e:
            logger.error(f"Error checking emergency override: {e}")
            return None
    
    async def _check_action_loop_break(self, 
                                     game_id: str,
                                     action_history: List[int],
                                     available_actions: List[int]) -> Optional[EmergencyOverride]:
        """Check for action loop that needs breaking."""
        try:
            if len(action_history) < 8:
                return None
            
            recent_actions = action_history[-8:]  # Last 8 actions
            
            # Extract action IDs from history (handle both int and dict formats)
            action_ids = []
            for action in recent_actions:
                if isinstance(action, dict):
                    action_ids.append(action.get('action', action.get('id', 0)))
                else:
                    action_ids.append(action)
            
            # Check for repeated action patterns
            action_counts = {}
            for action in action_ids:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Check for single action dominating
            max_count = max(action_counts.values())
            if max_count >= 6:  # Same action 6+ times in last 8
                dominant_action = max(action_counts.items(), key=lambda x: x[1])[0]
                
                # Find alternative action
                alternative_actions = [a for a in available_actions if a != dominant_action]
                if not alternative_actions:
                    alternative_actions = available_actions
                
                override_action = random.choice(alternative_actions)
                
                return EmergencyOverride(
                    game_id=game_id,
                    session_id="",  # Will be set by caller
                    override_type=OverrideType.ACTION_LOOP_BREAK,
                    trigger_reason=f"Action {dominant_action} repeated {max_count} times in last 8 actions",
                    actions_before_override=len(action_history),
                    override_action=override_action,
                    override_successful=False,
                    override_timestamp=time.time()
                )
            
            # Check for limited action diversity
            unique_actions = len(set(action_ids))
            if unique_actions <= 2 and len(action_ids) >= 6:
                # Force action diversification
                override_action = random.choice(available_actions)
                
                return EmergencyOverride(
                    game_id=game_id,
                    session_id="",  # Will be set by caller
                    override_type=OverrideType.ACTION_LOOP_BREAK,
                    trigger_reason=f"Only {unique_actions} unique actions in last {len(recent_actions)} actions",
                    actions_before_override=len(action_history),
                    override_action=override_action,
                    override_successful=False,
                    override_timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking action loop break: {e}")
            return None
    
    async def _check_coordinate_stuck_break(self, 
                                          game_id: str,
                                          current_state: Dict[str, Any]) -> Optional[EmergencyOverride]:
        """Check for coordinate-based stuck situations."""
        try:
            # Load recent coordinate attempts
            recent_coordinates = await self._get_recent_coordinates(game_id)
            if len(recent_coordinates) < 5:
                return None
            
            # Check for repeated coordinate attempts
            coordinate_counts = {}
            for coord in recent_coordinates[-10:]:  # Last 10 coordinates
                coord_key = (coord['x'], coord['y'])
                coordinate_counts[coord_key] = coordinate_counts.get(coord_key, 0) + 1
            
            # Find most repeated coordinate
            max_repetitions = max(coordinate_counts.values()) if coordinate_counts else 0
            
            if max_repetitions >= 4:  # Same coordinate 4+ times
                most_repeated = max(coordinate_counts.items(), key=lambda x: x[1])[0]
                
                # Force coordinate diversification
                # Generate random coordinate away from repeated one
                x, y = most_repeated
                offset_x = random.randint(-20, 20)
                offset_y = random.randint(-20, 20)
                new_x = max(0, min(63, x + offset_x))
                new_y = max(0, min(63, y + offset_y))
                
                return EmergencyOverride(
                    game_id=game_id,
                    session_id="",  # Will be set by caller
                    override_type=OverrideType.COORDINATE_STUCK_BREAK,
                    trigger_reason=f"Coordinate ({x}, {y}) repeated {max_repetitions} times",
                    actions_before_override=len(recent_coordinates),
                    override_action=6,  # Action 6 for coordinate
                    override_successful=False,
                    override_timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking coordinate stuck break: {e}")
            return None
    
    async def _check_stagnation_break(self, 
                                    game_id: str,
                                    performance_history: List[Dict[str, Any]]) -> Optional[EmergencyOverride]:
        """Check for general stagnation that needs breaking."""
        try:
            if len(performance_history) < 10:
                return None
            
            # Check for lack of progress
            recent_scores = [p.get('score', 0) for p in performance_history[-10:]]
            score_variance = np.var(recent_scores)
            score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # Check for stagnation indicators
            stagnation_score = 0.0
            
            # Low score variance indicates stagnation
            if score_variance < 1.0:
                stagnation_score += 0.3
            
            # Flat or declining trend
            if score_trend <= 0:
                stagnation_score += 0.4
            
            # Check for recent stagnation events
            recent_stagnation = await self._get_recent_stagnation_events(game_id)
            if len(recent_stagnation) >= 2:
                stagnation_score += 0.3
            
            if stagnation_score >= 0.7:  # High stagnation score
                # Force Action 5 as emergency recovery
                return EmergencyOverride(
                    game_id=game_id,
                    session_id="",  # Will be set by caller
                    override_type=OverrideType.STAGNATION_BREAK,
                    trigger_reason=f"Stagnation detected (score: {stagnation_score:.2f})",
                    actions_before_override=len(performance_history),
                    override_action=5,  # Action 5 for emergency recovery
                    override_successful=False,
                    override_timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking stagnation break: {e}")
            return None
    
    async def _check_emergency_reset(self, 
                                   game_id: str,
                                   action_history: List[int],
                                   performance_history: List[Dict[str, Any]]) -> Optional[EmergencyOverride]:
        """Check for emergency reset conditions."""
        try:
            if len(action_history) < 15 or len(performance_history) < 15:
                return None
            
            # Check for multiple failed recovery attempts
            recent_overrides = await self._get_recent_overrides(game_id)
            failed_overrides = [o for o in recent_overrides if not o.get('override_successful', False)]
            
            if len(failed_overrides) >= 3:  # 3+ failed overrides
                return EmergencyOverride(
                    game_id=game_id,
                    session_id="",  # Will be set by caller
                    override_type=OverrideType.EMERGENCY_RESET,
                    trigger_reason=f"{len(failed_overrides)} failed recovery attempts",
                    actions_before_override=len(action_history),
                    override_action=5,  # Action 5 for emergency reset
                    override_successful=False,
                    override_timestamp=time.time()
                )
            
            # Check for extreme stagnation
            recent_scores = [p.get('score', 0) for p in performance_history[-15:]]
            if len(recent_scores) >= 10:
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                if score_trend < -5.0:  # Severe negative trend
                    return EmergencyOverride(
                        game_id=game_id,
                        session_id="",  # Will be set by caller
                        override_type=OverrideType.EMERGENCY_RESET,
                        trigger_reason=f"Severe score regression (trend: {score_trend:.2f})",
                        actions_before_override=len(action_history),
                        override_action=5,  # Action 5 for emergency reset
                        override_successful=False,
                        override_timestamp=time.time()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking emergency reset: {e}")
            return None
    
    async def _get_recent_coordinates(self, game_id: str) -> List[Dict[str, Any]]:
        """Get recent coordinate attempts for a game."""
        try:
            query = """
                SELECT x, y, last_penalty_applied
                FROM coordinate_penalties
                WHERE game_id = ?
                ORDER BY last_penalty_applied DESC
                LIMIT 20
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting recent coordinates: {e}")
            return []
    
    async def _get_recent_stagnation_events(self, game_id: str) -> List[Dict[str, Any]]:
        """Get recent stagnation events for a game."""
        try:
            query = """
                SELECT stagnation_type, severity, detection_timestamp
                FROM stagnation_events
                WHERE game_id = ?
                ORDER BY detection_timestamp DESC
                LIMIT 10
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting recent stagnation events: {e}")
            return []
    
    async def _get_recent_overrides(self, game_id: str) -> List[Dict[str, Any]]:
        """Get recent override events for a game."""
        try:
            query = """
                SELECT override_type, override_successful, override_timestamp
                FROM emergency_overrides
                WHERE game_id = ?
                ORDER BY override_timestamp DESC
                LIMIT 10
            """
            
            results = await self.integration.db.fetch_all(query, (game_id,))
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error getting recent overrides: {e}")
            return []
    
    async def _store_emergency_override(self, override: EmergencyOverride):
        """Store emergency override in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO emergency_overrides
                (game_id, session_id, override_type, trigger_reason, actions_before_override,
                 override_action, override_successful, override_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                override.game_id, override.session_id, override.override_type.value,
                override.trigger_reason, override.actions_before_override,
                override.override_action, override.override_successful,
                override.override_timestamp
            ))
            
        except Exception as e:
            logger.error(f"Error storing emergency override: {e}")
    
    async def record_override_result(self, 
                                   override: EmergencyOverride,
                                   success: bool,
                                   frame_changes: bool = False,
                                   score_impact: float = 0.0):
        """Record the result of an emergency override."""
        try:
            override.override_successful = success
            
            # Update in database
            await self.integration.db.execute("""
                UPDATE emergency_overrides
                SET override_successful = ?
                WHERE game_id = ? AND override_timestamp = ?
            """, (success, override.game_id, override.override_timestamp))
            
            # Update local history
            if override.game_id in self.override_history:
                for hist_override in self.override_history[override.game_id]:
                    if (hist_override.game_id == override.game_id and 
                        hist_override.override_timestamp == override.override_timestamp):
                        hist_override.override_successful = success
                        break
            
            if success:
                logger.info(f"✅ Emergency override successful: {override.override_type.value}")
            else:
                logger.warning(f"❌ Emergency override failed: {override.override_type.value}")
            
        except Exception as e:
            logger.error(f"Error recording override result: {e}")
    
    async def get_override_statistics(self, game_id: str) -> Dict[str, Any]:
        """Get emergency override statistics for a game."""
        try:
            if game_id not in self.override_history:
                return {
                    'total_overrides': 0,
                    'successful_overrides': 0,
                    'override_types': {},
                    'success_rate': 0.0
                }
            
            overrides = self.override_history[game_id]
            
            # Count by type and success
            type_counts = {}
            successful_count = 0
            
            for override in overrides:
                type_name = override.override_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                
                if override.override_successful:
                    successful_count += 1
            
            success_rate = (successful_count / len(overrides)) if overrides else 0.0
            
            return {
                'total_overrides': len(overrides),
                'successful_overrides': successful_count,
                'override_types': type_counts,
                'success_rate': success_rate,
                'recent_overrides': [
                    {
                        'type': override.override_type.value,
                        'reason': override.trigger_reason,
                        'action': override.override_action,
                        'successful': override.override_successful,
                        'timestamp': override.override_timestamp
                    } for override in overrides[-5:]  # Last 5 overrides
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting override statistics: {e}")
            return {}
    
    async def should_trigger_emergency_reset(self, game_id: str) -> bool:
        """Check if emergency reset should be triggered."""
        try:
            if game_id not in self.override_history:
                return False
            
            overrides = self.override_history[game_id]
            if len(overrides) < 3:
                return False
            
            # Check recent overrides
            recent_overrides = overrides[-3:]
            failed_overrides = [o for o in recent_overrides if not o.override_successful]
            
            # Trigger reset if multiple recent overrides failed
            if len(failed_overrides) >= 2:
                return True
            
            # Check for emergency reset type
            emergency_resets = [o for o in recent_overrides if o.override_type == OverrideType.EMERGENCY_RESET]
            if emergency_resets:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency reset trigger: {e}")
            return False
    
    async def get_override_recommendations(self, 
                                         game_id: str,
                                         available_actions: List[int]) -> List[Dict[str, Any]]:
        """Get override action recommendations based on history."""
        try:
            if game_id not in self.override_history:
                return []
            
            overrides = self.override_history[game_id]
            
            # Find successful override actions
            successful_actions = {}
            for override in overrides:
                if override.override_successful:
                    action = override.override_action
                    if action in available_actions:
                        successful_actions[action] = successful_actions.get(action, 0) + 1
            
            # Sort by success count
            recommendations = []
            for action, count in sorted(successful_actions.items(), key=lambda x: x[1], reverse=True):
                recommendations.append({
                    'action': action,
                    'success_count': count,
                    'confidence': min(1.0, count / 3.0)  # Normalize confidence
                })
            
            return recommendations[:3]  # Top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error getting override recommendations: {e}")
            return []
