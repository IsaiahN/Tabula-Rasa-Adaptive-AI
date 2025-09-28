"""
Training Session Manager

Manages training sessions, including session lifecycle, state tracking,
and coordination between different session types.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .session_tracker import SessionTracker, SessionInfo
from .position_tracker import PositionTracker

logger = logging.getLogger(__name__)

@dataclass
class TrainingSessionConfig:
    """Configuration for a training session."""
    max_actions: int = 1000
    timeout_seconds: int = 300
    enable_position_tracking: bool = True
    enable_memory_tracking: bool = True
    enable_performance_tracking: bool = True

class TrainingSessionManager:
    """Manages training sessions and their lifecycle."""
    
    def __init__(self, config: Optional[TrainingSessionConfig] = None):
        self.config = config or TrainingSessionConfig()
        self.session_tracker = SessionTracker()
        self.position_trackers: Dict[str, PositionTracker] = {}
        self.session_states: Dict[str, Dict[str, Any]] = {}
        self.session_results: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, game_id: str, session_id: Optional[str] = None) -> str:
        """Create a new training session."""
        session_id = self.session_tracker.start_session(game_id, session_id)
        
        # Initialize position tracker if enabled
        if self.config.enable_position_tracking:
            self.position_trackers[session_id] = PositionTracker()
        
        # Initialize session state
        self.session_states[session_id] = {
            'game_id': game_id,
            'actions_taken': 0,
            'current_position': None,
            'session_start_time': datetime.now(),
            'last_action_time': None,
            'memory_usage': 0,
            'performance_metrics': {}
        }
        
        logger.info(f"Created training session {session_id} for game {game_id}")
        return session_id
    
    def end_session(self, session_id: str, status: str = "completed", 
                   score: float = 0.0, win: bool = False, 
                   error_message: Optional[str] = None) -> None:
        """End a training session."""
        # Update session tracker
        self.session_tracker.end_session(session_id, status, score, win, error_message)
        
        # Store session results
        session_info = self.session_tracker.get_session(session_id)
        if session_info:
            self.session_results[session_id] = {
                'session_info': session_info,
                'position_stats': self.get_position_stats(session_id),
                'session_state': self.session_states.get(session_id, {}),
                'end_time': datetime.now()
            }
        
        # Clean up position tracker
        if session_id in self.position_trackers:
            del self.position_trackers[session_id]
        
        # Clean up session state
        if session_id in self.session_states:
            del self.session_states[session_id]
        
        logger.info(f"Ended training session {session_id} with status {status}")
    
    def update_session_action(self, session_id: str, action: Dict[str, Any], 
                            position: Optional[Tuple[int, int]] = None) -> None:
        """Update session with a new action."""
        if session_id not in self.session_states:
            logger.warning(f"Session {session_id} not found")
            return
        
        # Update action count
        self.session_states[session_id]['actions_taken'] += 1
        self.session_states[session_id]['last_action_time'] = datetime.now()
        
        # Update session tracker
        self.session_tracker.update_session_actions(
            session_id, 
            self.session_states[session_id]['actions_taken']
        )
        
        # Update position if provided
        if position is not None:
            self.update_position(session_id, position)
    
    def update_position(self, session_id: str, position: Tuple[int, int]) -> None:
        """Update position for a session."""
        if session_id in self.position_trackers:
            self.position_trackers[session_id].update_position(position)
            self.session_states[session_id]['current_position'] = position
    
    def get_position_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get position statistics for a session."""
        if session_id in self.position_trackers:
            return self.position_trackers[session_id].get_position_stats()
        return None
    
    def is_session_stuck(self, session_id: str) -> bool:
        """Check if a session is stuck."""
        if session_id in self.position_trackers:
            return self.position_trackers[session_id].is_stuck()
        return False
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state."""
        return self.session_states.get(session_id)
    
    def get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session result."""
        return self.session_results.get(session_id)
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """Get all active sessions."""
        return self.session_tracker.get_active_sessions()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        return self.session_tracker.get_session_stats()
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions."""
        cleaned_count = self.session_tracker.cleanup_old_sessions(max_age_hours)
        
        # Clean up position trackers for removed sessions
        active_session_ids = {s.session_id for s in self.get_active_sessions()}
        for session_id in list(self.position_trackers.keys()):
            if session_id not in active_session_ids:
                del self.position_trackers[session_id]
        
        # Clean up session states
        for session_id in list(self.session_states.keys()):
            if session_id not in active_session_ids:
                del self.session_states[session_id]
        
        return cleaned_count
    
    def reset(self) -> None:
        """Reset all session management."""
        self.session_tracker.reset()
        self.position_trackers.clear()
        self.session_states.clear()
        self.session_results.clear()
        logger.info("Training session manager reset")
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a comprehensive summary of a session."""
        session_info = self.session_tracker.get_session(session_id)
        if not session_info:
            return None
        
        position_stats = self.get_position_stats(session_id)
        session_state = self.session_states.get(session_id, {})
        
        return {
            'session_info': session_info,
            'position_stats': position_stats,
            'session_state': session_state,
            'is_stuck': self.is_session_stuck(session_id),
            'duration_seconds': (
                (session_info.end_time or datetime.now()) - session_info.start_time
            ).total_seconds() if session_info.end_time else None
        }

    def should_reflect(self, action_count: int) -> bool:
        """Determine if the system should perform self-reflection based on action count."""
        # Default reflection frequency - can be made configurable
        reflection_frequency = getattr(self.config, 'reflection_frequency', 10) if self.config else 10
        return action_count > 0 and action_count % reflection_frequency == 0
