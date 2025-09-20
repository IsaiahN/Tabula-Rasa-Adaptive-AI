"""
Session Tracker

Tracks training sessions, their status, and performance metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SessionInfo:
    """Information about a training session."""
    session_id: str
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"  # active, completed, failed, cancelled
    total_actions: int = 0
    score: float = 0.0
    win: bool = False
    error_message: Optional[str] = None

class SessionTracker:
    """Tracks training sessions and their performance."""
    
    def __init__(self, max_sessions: int = 1000):
        self.sessions: Dict[str, SessionInfo] = {}
        self.active_sessions: List[str] = []
        self.completed_sessions: List[str] = []
        self.failed_sessions: List[str] = []
        self.max_sessions = max_sessions
        self.session_counter = 0
    
    def start_session(self, game_id: str, session_id: Optional[str] = None) -> str:
        """Start a new training session."""
        if session_id is None:
            session_id = f"session_{self.session_counter}_{int(datetime.now().timestamp())}"
            self.session_counter += 1
        
        session_info = SessionInfo(
            session_id=session_id,
            game_id=game_id,
            start_time=datetime.now()
        )
        
        self.sessions[session_id] = session_info
        self.active_sessions.append(session_id)
        
        logger.info(f"Started session {session_id} for game {game_id}")
        return session_id
    
    def end_session(self, session_id: str, status: str = "completed", 
                   score: float = 0.0, win: bool = False, 
                   error_message: Optional[str] = None) -> None:
        """End a training session."""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        session_info = self.sessions[session_id]
        session_info.end_time = datetime.now()
        session_info.status = status
        session_info.score = score
        session_info.win = win
        session_info.error_message = error_message
        
        # Move from active to appropriate list
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
        
        if status == "completed":
            self.completed_sessions.append(session_id)
        elif status == "failed":
            self.failed_sessions.append(session_id)
        
        logger.info(f"Ended session {session_id} with status {status}")
    
    def update_session_actions(self, session_id: str, action_count: int) -> None:
        """Update the action count for a session."""
        if session_id in self.sessions:
            self.sessions[session_id].total_actions = action_count
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """Get all active sessions."""
        return [self.sessions[sid] for sid in self.active_sessions if sid in self.sessions]
    
    def get_completed_sessions(self) -> List[SessionInfo]:
        """Get all completed sessions."""
        return [self.sessions[sid] for sid in self.completed_sessions if sid in self.sessions]
    
    def get_failed_sessions(self) -> List[SessionInfo]:
        """Get all failed sessions."""
        return [self.sessions[sid] for sid in self.failed_sessions if sid in self.sessions]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        total_sessions = len(self.sessions)
        active_count = len(self.active_sessions)
        completed_count = len(self.completed_sessions)
        failed_count = len(self.failed_sessions)
        
        # Calculate success rate
        success_rate = 0.0
        if completed_count + failed_count > 0:
            success_rate = completed_count / (completed_count + failed_count)
        
        # Calculate average score
        avg_score = 0.0
        if completed_count > 0:
            scores = [s.score for s in self.get_completed_sessions()]
            avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate average actions
        avg_actions = 0.0
        if total_sessions > 0:
            action_counts = [s.total_actions for s in self.sessions.values()]
            avg_actions = sum(action_counts) / len(action_counts) if action_counts else 0.0
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_count,
            'completed_sessions': completed_count,
            'failed_sessions': failed_count,
            'success_rate': success_rate,
            'average_score': avg_score,
            'average_actions': avg_actions
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions to prevent memory bloat."""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        sessions_to_remove = []
        for session_id, session_info in self.sessions.items():
            if session_info.start_time < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self._remove_session(session_id)
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        return len(sessions_to_remove)
    
    def _remove_session(self, session_id: str) -> None:
        """Remove a session from all tracking lists."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        for session_list in [self.active_sessions, self.completed_sessions, self.failed_sessions]:
            if session_id in session_list:
                session_list.remove(session_id)
    
    def reset(self) -> None:
        """Reset all session tracking."""
        self.sessions.clear()
        self.active_sessions.clear()
        self.completed_sessions.clear()
        self.failed_sessions.clear()
        self.session_counter = 0
        logger.info("Session tracker reset")
