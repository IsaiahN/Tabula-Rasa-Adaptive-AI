"""
Base Interface Definitions

Defines the fundamental interfaces that all training system components must implement.
These interfaces ensure consistent APIs and enable proper dependency injection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ComponentState:
    """Standard state representation for all components.
    
    Attributes:
        name: Component name identifier
        status: Current component status ('initialized', 'running', 'paused', 'stopped', 'error')
        last_updated: Timestamp of last state update
        metadata: Additional component-specific metadata
    """
    name: str
    status: str  # 'initialized', 'running', 'paused', 'stopped', 'error'
    last_updated: datetime
    metadata: Dict[str, Any]


class ComponentInterface(ABC):
    """Base interface for all training system components.
    
    This interface defines the minimum contract that all components
    in the training system must implement. It ensures consistent
    lifecycle management and health monitoring across all components.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the component.
        
        This method should perform any necessary setup operations
        required before the component can be used. It should be
        idempotent and safe to call multiple times.
        
        Raises:
            InitializationError: If component initialization fails
        """
        pass
    
    @abstractmethod
    def get_state(self) -> ComponentState:
        """Get current component state.
        
        Returns:
            ComponentState object containing current component status
            and metadata
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up component resources.
        
        This method should release any resources held by the component
        and prepare it for shutdown. It should be safe to call multiple
        times and should not raise exceptions.
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if component is healthy.
        
        Returns:
            True if component is functioning normally, False otherwise
        """
        pass


class MemoryInterface(ABC):
    """Interface for memory management components."""
    
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a value in memory."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all keys, optionally filtered by pattern."""
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        pass


class SessionInterface(ABC):
    """Interface for session management components."""
    
    @abstractmethod
    def start_session(self, session_id: str, config: Dict[str, Any]) -> None:
        """Start a new session."""
        pass
    
    @abstractmethod
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session and return results."""
        pass
    
    @abstractmethod
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        pass
    
    @abstractmethod
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific session."""
        pass


class APIManagerInterface(ABC):
    """Interface for API management components."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize API connections."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close API connections."""
        pass
    
    @abstractmethod
    async def make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API request."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if API is connected."""
        pass


class PerformanceInterface(ABC):
    """Interface for performance monitoring components."""
    
    @abstractmethod
    def start_monitoring(self, operation_id: str) -> None:
        """Start monitoring an operation."""
        pass
    
    @abstractmethod
    def stop_monitoring(self, operation_id: str) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        pass
    
    @abstractmethod
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        pass
    
    @abstractmethod
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric."""
        pass


class GovernorInterface(ABC):
    """Interface for governor/meta-cognitive components."""
    
    @abstractmethod
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make a meta-cognitive decision."""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        pass
    
    @abstractmethod
    def update_goals(self, goals: List[str]) -> None:
        """Update system goals."""
        pass


class LearningInterface(ABC):
    """Interface for learning components."""
    
    @abstractmethod
    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Learn from an experience."""
        pass
    
    @abstractmethod
    def apply_learning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned knowledge to a context."""
        pass
    
    @abstractmethod
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        pass
