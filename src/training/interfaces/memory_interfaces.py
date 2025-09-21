"""
Memory System Interfaces

Defines interfaces for memory management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from .base_interfaces import MemoryInterface, ComponentInterface


class MemoryManagerInterface(MemoryInterface, ComponentInterface):
    """Interface for memory managers."""
    
    @abstractmethod
    def create_memory_pool(self, name: str, size: int) -> str:
        """Create a new memory pool."""
        pass
    
    @abstractmethod
    def delete_memory_pool(self, name: str) -> bool:
        """Delete a memory pool."""
        pass
    
    @abstractmethod
    def get_memory_pools(self) -> List[Dict[str, Any]]:
        """Get all memory pools."""
        pass
    
    @abstractmethod
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        pass


class ActionMemoryInterface(MemoryInterface):
    """Interface for action-specific memory management."""
    
    @abstractmethod
    def store_action(self, action_id: str, action_data: Dict[str, Any], 
                    context: Optional[Dict[str, Any]] = None) -> None:
        """Store action data with context."""
        pass
    
    @abstractmethod
    def retrieve_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve action data."""
        pass
    
    @abstractmethod
    def find_similar_actions(self, action_data: Dict[str, Any], 
                           threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar actions based on data similarity."""
        pass
    
    @abstractmethod
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get action memory statistics."""
        pass


class PatternMemoryInterface(MemoryInterface):
    """Interface for pattern-specific memory management."""
    
    @abstractmethod
    def store_pattern(self, pattern_id: str, pattern_data: Dict[str, Any],
                     confidence: float = 1.0) -> None:
        """Store pattern data with confidence score."""
        pass
    
    @abstractmethod
    def retrieve_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve pattern data."""
        pass
    
    @abstractmethod
    def match_patterns(self, input_data: Dict[str, Any], 
                      min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Find matching patterns for input data."""
        pass
    
    @abstractmethod
    def update_pattern_confidence(self, pattern_id: str, confidence: float) -> None:
        """Update pattern confidence score."""
        pass
    
    @abstractmethod
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern memory statistics."""
        pass
