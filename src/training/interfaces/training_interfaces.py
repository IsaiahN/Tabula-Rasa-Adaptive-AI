"""
Training System Specific Interfaces

Defines interfaces specific to the training orchestrators and main components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .base_interfaces import ComponentInterface


class TrainingOrchestratorInterface(ComponentInterface):
    """Base interface for training orchestrators."""
    
    @abstractmethod
    async def run_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run training with given configuration."""
        pass
    
    @abstractmethod
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        pass
    
    @abstractmethod
    def pause_training(self) -> None:
        """Pause training."""
        pass
    
    @abstractmethod
    def resume_training(self) -> None:
        """Resume training."""
        pass
    
    @abstractmethod
    def stop_training(self) -> Dict[str, Any]:
        """Stop training and return results."""
        pass


class ContinuousLearningInterface(TrainingOrchestratorInterface):
    """Interface for continuous learning systems."""
    
    @abstractmethod
    async def run_continuous_learning(self, max_games: int = 100) -> Dict[str, Any]:
        """Run continuous learning for specified number of games."""
        pass
    
    @abstractmethod
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get current learning progress."""
        pass
    
    @abstractmethod
    def add_learning_goal(self, goal: str, priority: int = 1) -> None:
        """Add a learning goal."""
        pass
    
    @abstractmethod
    def get_learning_goals(self) -> List[Dict[str, Any]]:
        """Get all learning goals."""
        pass


class MasterTrainerInterface(TrainingOrchestratorInterface):
    """Interface for master trainer systems."""
    
    @abstractmethod
    def set_training_mode(self, mode: str) -> None:
        """Set the training mode."""
        pass
    
    @abstractmethod
    def get_available_modes(self) -> List[str]:
        """Get available training modes."""
        pass
    
    @abstractmethod
    def configure_training(self, config: Dict[str, Any]) -> None:
        """Configure training parameters."""
        pass
    
    @abstractmethod
    def get_training_config(self) -> Dict[str, Any]:
        """Get current training configuration."""
        pass
    
    @abstractmethod
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any errors."""
        pass
