"""
Learning System Interfaces

Defines interfaces for learning and knowledge transfer components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .base_interfaces import LearningInterface, ComponentInterface


class LearningEngineInterface(LearningInterface, ComponentInterface):
    """Interface for learning engines."""
    
    @abstractmethod
    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process a learning experience."""
        pass
    
    @abstractmethod
    def extract_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from data."""
        pass
    
    @abstractmethod
    def apply_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply learned patterns to context."""
        pass
    
    @abstractmethod
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get learning progress metrics."""
        pass
    
    @abstractmethod
    def reset_learning(self) -> None:
        """Reset learning state."""
        pass


class PatternLearnerInterface(ComponentInterface):
    """Interface for pattern learning components."""
    
    @abstractmethod
    def learn_pattern(self, pattern_data: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> str:
        """Learn a new pattern and return pattern ID."""
        pass
    
    @abstractmethod
    def recognize_pattern(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize patterns in input data."""
        pass
    
    @abstractmethod
    def update_pattern(self, pattern_id: str, updates: Dict[str, Any]) -> None:
        """Update an existing pattern."""
        pass
    
    @abstractmethod
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern."""
        pass
    
    @abstractmethod
    def get_pattern_catalog(self) -> List[Dict[str, Any]]:
        """Get catalog of all patterns."""
        pass


class KnowledgeTransferInterface(ComponentInterface):
    """Interface for knowledge transfer components."""
    
    @abstractmethod
    def transfer_knowledge(self, source_context: Dict[str, Any], 
                          target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge between contexts."""
        pass
    
    @abstractmethod
    def identify_transferable_knowledge(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify knowledge that can be transferred."""
        pass
    
    @abstractmethod
    def apply_transferred_knowledge(self, knowledge: Dict[str, Any], 
                                   context: Dict[str, Any]) -> bool:
        """Apply transferred knowledge to context."""
        pass
    
    @abstractmethod
    def get_transfer_history(self) -> List[Dict[str, Any]]:
        """Get history of knowledge transfers."""
        pass
    
    @abstractmethod
    def evaluate_transfer_effectiveness(self) -> Dict[str, Any]:
        """Evaluate effectiveness of knowledge transfers."""
        pass
