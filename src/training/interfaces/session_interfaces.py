"""
Session Management Interfaces

Defines interfaces for session management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_interfaces import SessionInterface, ComponentInterface


class SessionConfigInterface(ABC):
    """Interface for session configuration."""
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        pass


class SessionManagerInterface(SessionInterface, ComponentInterface):
    """Interface for session managers."""
    
    @abstractmethod
    def create_session(self, session_type: str, config: Dict[str, Any]) -> str:
        """Create a new session and return session ID."""
        pass
    
    @abstractmethod
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information."""
        pass
    
    @abstractmethod
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session data."""
        pass
    
    @abstractmethod
    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by status."""
        pass
    
    @abstractmethod
    def cleanup_old_sessions(self, older_than: datetime) -> int:
        """Clean up sessions older than specified time."""
        pass


class SessionTrackerInterface(ComponentInterface):
    """Interface for session tracking components."""
    
    @abstractmethod
    def track_event(self, session_id: str, event_type: str, 
                   data: Dict[str, Any]) -> None:
        """Track an event in a session."""
        pass
    
    @abstractmethod
    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all events for a session."""
        pass
    
    @abstractmethod
    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for a session."""
        pass
    
    @abstractmethod
    def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export complete session data."""
        pass
