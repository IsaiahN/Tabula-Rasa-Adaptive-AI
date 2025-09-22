#!/usr/bin/env python3
"""
Advanced Stagnation Detection System

This module implements multi-layered stagnation detection with score regression,
action effectiveness analysis, and recovery strategies.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.database.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class StagnationType(Enum):
    """Types of stagnation that can be detected."""
    SCORE_REGRESSION = "score_regression"
    ACTION_REPETITION = "action_repetition"
    NO_FRAME_CHANGES = "no_frame_changes"
    COORDINATE_STUCK = "coordinate_stuck"
    GENERAL_STAGNATION = "general_stagnation"

@dataclass
class StagnationEvent:
    """Represents a detected stagnation event."""
    game_id: str
    session_id: str
    stagnation_type: StagnationType
    severity: float  # 0.0 to 1.0
    consecutive_count: int
    context_data: Dict[str, Any]
    recovery_action: Optional[str] = None
    recovery_successful: bool = False
    detection_timestamp: float = 0.0

class AdvancedStagnationSystem:
    """
    Advanced Stagnation Detection System
    
    Implements multi-layered stagnation detection with sophisticated analysis
    and recovery strategies.
    """
    
    def __init__(self):
        self.integration = get_system_integration()
        self.stagnation_history: Dict[str, List[StagnationEvent]] = {}
        self.recovery_strategies: Dict[StagnationType, List[str]] = {
            StagnationType.SCORE_REGRESSION: ['switch_action_type', 'try_action5', 'coordinate_exploration'],
            StagnationType.ACTION_REPETITION: ['force_action_diversification', 'emergency_override', 'try_action5'],
            StagnationType.NO_FRAME_CHANGES: ['try_action5', 'coordinate_exploration', 'action_diversification'],
            StagnationType.COORDINATE_STUCK: ['coordinate_diversification', 'try_action5', 'action_switch'],
            StagnationType.GENERAL_STAGNATION: ['comprehensive_reset', 'try_action5', 'strategy_replication']
        }
        
    async def detect_stagnation(self, 
                              game_id: str,
                              session_id: str,
                              current_state: Dict[str, Any],
                              performance_history: List[Dict[str, Any]],
                              action_history: List[int],
                              frame_change_history: List[bool]) -> Optional[StagnationEvent]:
        """
        Detect stagnation using multi-layered analysis.
        
        Args:
            game_id: Current game identifier
            session_id: Current session identifier
            current_state: Current game state
            performance_history: Recent performance data
            action_history: Recent action history
            frame_change_history: Recent frame change history
            
        Returns:
            StagnationEvent if stagnation detected, None otherwise
        """
        try:
            # Check for different types of stagnation
            stagnation_checks = [
                await self._check_score_regression(performance_history),
                await self._check_action_repetition(action_history),
                await self._check_no_frame_changes(frame_change_history),
                await self._check_coordinate_stuck(game_id, current_state),
                await self._check_general_stagnation(performance_history, action_history)
            ]
            
            # Find the most severe stagnation
            valid_stagnations = [s for s in stagnation_checks if s is not None]
            if not valid_stagnations:
                return None
            
            # Select the most severe stagnation
            most_severe = max(valid_stagnations, key=lambda s: s.severity)
            
            # Create stagnation event
            stagnation_event = StagnationEvent(
                game_id=game_id,
                session_id=session_id,
                stagnation_type=most_severe.stagnation_type,
                severity=most_severe.severity,
                consecutive_count=most_severe.consecutive_count,
                context_data=most_severe.context_data,
                detection_timestamp=time.time()
            )
            
            # Store in database
            await self._store_stagnation_event(stagnation_event)
            
            # Update local history
            if game_id not in self.stagnation_history:
                self.stagnation_history[game_id] = []
            self.stagnation_history[game_id].append(stagnation_event)
            
            logger.warning(f"Stagnation detected: {stagnation_event.stagnation_type.value} "
                          f"(severity: {stagnation_event.severity:.2f}, count: {stagnation_event.consecutive_count})")
            
            return stagnation_event
            
        except Exception as e:
            logger.error(f"Error detecting stagnation: {e}")
            return None
    
    async def _check_score_regression(self, performance_history: List[Dict[str, Any]]) -> Optional[StagnationEvent]:
        """Check for score regression over time."""
        try:
            if len(performance_history) < 10:
                return None
            
            # Extract recent scores
            recent_scores = [p.get('score', 0) for p in performance_history[-10:]]
            if len(recent_scores) < 5:
                return None
            
            # Calculate score trend
            x = np.arange(len(recent_scores))
            trend = np.polyfit(x, recent_scores, 1)[0]
            
            # Check for significant negative trend
            if trend < -2.0:  # Significant negative trend
                severity = min(1.0, abs(trend) / 5.0)
                consecutive_count = self._count_consecutive_decreases(recent_scores)
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.SCORE_REGRESSION,
                    severity=severity,
                    consecutive_count=consecutive_count,
                    context_data={
                        'trend': trend,
                        'recent_scores': recent_scores[-5:],
                        'score_variance': np.var(recent_scores)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking score regression: {e}")
            return None
    
    async def _check_action_repetition(self, action_history: List[Any]) -> Optional[StagnationEvent]:
        """Check for excessive action repetition."""
        try:
            if len(action_history) < 5:
                return None
            
            # Extract action IDs from history (handle both int and dict formats)
            recent_actions = []
            for action in action_history[-10:]:  # Last 10 actions
                if isinstance(action, dict):
                    recent_actions.append(action.get('action', action.get('id', 0)))
                else:
                    recent_actions.append(action)
            
            # Count consecutive repetitions
            consecutive_count = 1
            current_action = recent_actions[-1]
            
            for i in range(len(recent_actions) - 2, -1, -1):
                if recent_actions[i] == current_action:
                    consecutive_count += 1
                else:
                    break
            
            # Check for same action repeated too many times
            if consecutive_count >= 3:
                severity = min(1.0, consecutive_count / 5.0)
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.ACTION_REPETITION,
                    severity=severity,
                    consecutive_count=consecutive_count,
                    context_data={
                        'repeated_action': current_action,
                        'recent_actions': recent_actions,
                        'action_diversity': len(set(recent_actions))
                    }
                )
            
            # Check for limited action diversity
            unique_actions = len(set(recent_actions))
            if unique_actions <= 2 and len(recent_actions) >= 6:
                severity = 0.7  # High severity for low diversity
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.ACTION_REPETITION,
                    severity=severity,
                    consecutive_count=len(recent_actions),
                    context_data={
                        'action_diversity': unique_actions,
                        'recent_actions': recent_actions,
                        'diversity_ratio': unique_actions / len(recent_actions)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking action repetition: {e}")
            return None
    
    async def _check_no_frame_changes(self, frame_change_history: List[bool]) -> Optional[StagnationEvent]:
        """Check for lack of frame changes."""
        try:
            if len(frame_change_history) < 5:
                return None
            
            # Count consecutive actions without frame changes
            consecutive_no_changes = 0
            for i in range(len(frame_change_history) - 1, -1, -1):
                if not frame_change_history[i]:
                    consecutive_no_changes += 1
                else:
                    break
            
            if consecutive_no_changes >= 5:
                severity = min(1.0, consecutive_no_changes / 10.0)
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.NO_FRAME_CHANGES,
                    severity=severity,
                    consecutive_count=consecutive_no_changes,
                    context_data={
                        'frame_change_rate': sum(frame_change_history) / len(frame_change_history),
                        'recent_frame_changes': frame_change_history[-10:],
                        'total_actions': len(frame_change_history)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking frame changes: {e}")
            return None
    
    async def _check_coordinate_stuck(self, game_id: str, current_state: Dict[str, Any]) -> Optional[StagnationEvent]:
        """Check for coordinate-based stagnation."""
        try:
            # Load recent coordinate attempts for this game
            recent_coordinates = await self._get_recent_coordinates(game_id)
            if len(recent_coordinates) < 3:
                return None
            
            # Check for repeated coordinate attempts
            coordinate_counts = {}
            for coord in recent_coordinates[-10:]:  # Last 10 coordinates
                coord_key = (coord['x'], coord['y'])
                coordinate_counts[coord_key] = coordinate_counts.get(coord_key, 0) + 1
            
            # Find most repeated coordinate
            max_repetitions = max(coordinate_counts.values()) if coordinate_counts else 0
            
            if max_repetitions >= 3:
                severity = min(1.0, max_repetitions / 5.0)
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.COORDINATE_STUCK,
                    severity=severity,
                    consecutive_count=max_repetitions,
                    context_data={
                        'coordinate_counts': coordinate_counts,
                        'most_repeated_coord': max(coordinate_counts.items(), key=lambda x: x[1])[0],
                        'coordinate_diversity': len(coordinate_counts)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking coordinate stuck: {e}")
            return None
    
    async def _check_general_stagnation(self, 
                                      performance_history: List[Dict[str, Any]], 
                                      action_history: List[int]) -> Optional[StagnationEvent]:
        """Check for general stagnation patterns."""
        try:
            if len(performance_history) < 8 or len(action_history) < 8:
                return None
            
            # Check for lack of progress indicators
            recent_scores = [p.get('score', 0) for p in performance_history[-8:]]
            score_variance = np.var(recent_scores)
            score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            # Check for low action diversity
            recent_actions = []
            for action in action_history[-8:]:
                if isinstance(action, dict):
                    recent_actions.append(action.get('action', action.get('id', 0)))
                else:
                    recent_actions.append(action)
            
            unique_actions = len(set(recent_actions))
            action_diversity_ratio = unique_actions / len(recent_actions)
            
            # Calculate stagnation score
            stagnation_score = 0.0
            
            # Low score variance indicates stagnation
            if score_variance < 1.0:
                stagnation_score += 0.3
            
            # Flat or declining trend
            if score_trend <= 0:
                stagnation_score += 0.4
            
            # Low action diversity
            if action_diversity_ratio < 0.4:
                stagnation_score += 0.3
            
            if stagnation_score >= 0.6:  # Threshold for general stagnation
                severity = min(1.0, stagnation_score)
                
                return StagnationEvent(
                    game_id="",  # Will be set by caller
                    session_id="",  # Will be set by caller
                    stagnation_type=StagnationType.GENERAL_STAGNATION,
                    severity=severity,
                    consecutive_count=8,  # Based on analysis window
                    context_data={
                        'score_variance': score_variance,
                        'score_trend': score_trend,
                        'action_diversity_ratio': action_diversity_ratio,
                        'stagnation_score': stagnation_score,
                        'recent_scores': recent_scores,
                        'recent_actions': action_history[-8:]
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking general stagnation: {e}")
            return None
    
    def _count_consecutive_decreases(self, scores: List[float]) -> int:
        """Count consecutive score decreases."""
        if len(scores) < 2:
            return 0
        
        consecutive_decreases = 0
        for i in range(len(scores) - 1, 0, -1):
            if scores[i] < scores[i-1]:
                consecutive_decreases += 1
            else:
                break
        
        return consecutive_decreases
    
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
    
    async def _store_stagnation_event(self, event: StagnationEvent):
        """Store stagnation event in database."""
        try:
            await self.integration.db.execute("""
                INSERT INTO stagnation_events
                (game_id, session_id, stagnation_type, severity, consecutive_count,
                 stagnation_context, detection_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.game_id, event.session_id, event.stagnation_type.value,
                event.severity, event.consecutive_count, 
                json.dumps(event.context_data, default=str, ensure_ascii=False),
                event.detection_timestamp
            ))
            
        except Exception as e:
            logger.error(f"Error storing stagnation event: {e}")
    
    async def get_recovery_strategy(self, stagnation_event: StagnationEvent) -> List[str]:
        """Get recovery strategies for a stagnation event."""
        try:
            strategies = self.recovery_strategies.get(stagnation_event.stagnation_type, [])
            
            # Add severity-based strategies
            if stagnation_event.severity > 0.8:
                strategies = ['emergency_override'] + strategies
            elif stagnation_event.severity > 0.6:
                strategies = ['try_action5'] + strategies
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error getting recovery strategy: {e}")
            return ['try_action5']  # Default fallback
    
    async def record_recovery_attempt(self, 
                                    stagnation_event: StagnationEvent,
                                    recovery_action: str,
                                    successful: bool):
        """Record the result of a recovery attempt."""
        try:
            # Update the stagnation event
            stagnation_event.recovery_action = recovery_action
            stagnation_event.recovery_successful = successful
            
            # Update in database
            await self.integration.db.execute("""
                UPDATE stagnation_events
                SET recovery_action = ?, recovery_successful = ?
                WHERE game_id = ? AND detection_timestamp = ?
            """, (recovery_action, successful, stagnation_event.game_id, stagnation_event.detection_timestamp))
            
            # Log recovery attempt
            if successful:
                logger.info(f"Recovery successful: {recovery_action} for {stagnation_event.stagnation_type.value}")
            else:
                logger.warning(f"Recovery failed: {recovery_action} for {stagnation_event.stagnation_type.value}")
            
        except Exception as e:
            logger.error(f"Error recording recovery attempt: {e}")
    
    async def get_stagnation_statistics(self, game_id: str) -> Dict[str, Any]:
        """Get stagnation statistics for a game."""
        try:
            if game_id not in self.stagnation_history:
                return {
                    'total_stagnation_events': 0,
                    'stagnation_types': {},
                    'recovery_success_rate': 0.0,
                    'average_severity': 0.0
                }
            
            events = self.stagnation_history[game_id]
            
            # Count by type
            type_counts = {}
            recovery_attempts = 0
            successful_recoveries = 0
            total_severity = 0.0
            
            for event in events:
                type_counts[event.stagnation_type.value] = type_counts.get(event.stagnation_type.value, 0) + 1
                total_severity += event.severity
                
                if event.recovery_action:
                    recovery_attempts += 1
                    if event.recovery_successful:
                        successful_recoveries += 1
            
            recovery_success_rate = (successful_recoveries / recovery_attempts) if recovery_attempts > 0 else 0.0
            average_severity = total_severity / len(events) if events else 0.0
            
            return {
                'total_stagnation_events': len(events),
                'stagnation_types': type_counts,
                'recovery_success_rate': recovery_success_rate,
                'average_severity': average_severity,
                'recent_events': [
                    {
                        'type': event.stagnation_type.value,
                        'severity': event.severity,
                        'count': event.consecutive_count,
                        'recovery_action': event.recovery_action,
                        'successful': event.recovery_successful
                    } for event in events[-5:]  # Last 5 events
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting stagnation statistics: {e}")
            return {}
    
    async def should_trigger_emergency_override(self, game_id: str) -> bool:
        """Check if emergency override should be triggered."""
        try:
            if game_id not in self.stagnation_history:
                return False
            
            events = self.stagnation_history[game_id]
            if not events:
                return False
            
            # Check recent events
            recent_events = events[-3:]  # Last 3 events
            
            # Trigger if multiple recent stagnation events with high severity
            high_severity_events = [e for e in recent_events if e.severity > 0.7]
            if len(high_severity_events) >= 2:
                return True
            
            # Trigger if recent recovery attempts failed
            failed_recoveries = [e for e in recent_events if e.recovery_action and not e.recovery_successful]
            if len(failed_recoveries) >= 2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency override trigger: {e}")
            return False
