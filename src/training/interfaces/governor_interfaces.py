"""
Governor System Interfaces

Defines interfaces for governor and meta-cognitive components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .base_interfaces import GovernorInterface, ComponentInterface


class MetaCognitiveInterface(ABC):
    """Interface for meta-cognitive controllers."""
    
    @abstractmethod
    def analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current system state."""
        pass
    
    @abstractmethod
    def generate_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on context."""
        pass
    
    @abstractmethod
    def update_cognitive_model(self, data: Dict[str, Any]) -> None:
        """Update the cognitive model."""
        pass
    
    @abstractmethod
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        pass


class GovernorInterface(GovernorInterface, ComponentInterface):
    """Enhanced interface for governor systems."""
    
    @abstractmethod
    def set_goals(self, goals: List[Dict[str, Any]]) -> None:
        """Set system goals."""
        pass
    
    @abstractmethod
    def get_goals(self) -> List[Dict[str, Any]]:
        """Get current goals."""
        pass
    
    @abstractmethod
    def evaluate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a decision against current goals."""
        pass
    
    @abstractmethod
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of decisions made."""
        pass
    
    @abstractmethod
    def update_governance_rules(self, rules: Dict[str, Any]) -> None:
        """Update governance rules."""
        pass
    
    @abstractmethod
    def get_governance_rules(self) -> Dict[str, Any]:
        """Get current governance rules."""
        pass
