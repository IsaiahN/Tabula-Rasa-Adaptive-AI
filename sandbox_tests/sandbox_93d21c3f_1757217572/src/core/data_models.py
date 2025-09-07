"""
Core data structures for the adaptive learning agent.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque
import torch
from torch import Tensor


@dataclass
class SensoryInput:
    """Represents multi-modal sensory input to the agent."""
    visual: Tensor  # Shape: [channels, height, width]
    proprioception: Tensor  # Shape: [joint_angles, velocities]
    energy_level: float
    timestamp: int


@dataclass
class Prediction:
    """Prediction output from the predictive core."""
    next_sensory: SensoryInput
    confidence: Tensor
    prediction_error: float


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: SensoryInput
    action: Tensor
    next_state: SensoryInput
    learning_progress: float
    energy_change: float
    timestamp: int


@dataclass
class Goal:
    """Represents a goal in the agent's goal system."""
    target_state_cluster: Tensor  # Centroid in latent space
    achievement_radius: float
    success_rate: float
    learning_progress_history: List[float]
    creation_timestamp: int
    goal_id: str
    goal_type: str  # "survival", "template", "emergent"


@dataclass
class AgentState:
    """Complete state of the agent."""
    position: Tensor  # 3D coordinates
    orientation: Tensor  # Quaternion
    energy: float
    hidden_state: Tensor  # Recurrent model state
    active_goals: List[Goal]
    memory_state: Optional[Tensor] = None  # DNC memory state
    timestamp: int = 0


class ReplayBuffer:
    """Experience replay buffer with prioritized sampling."""
    
    def __init__(self, capacity: int = 100000):
        self.experiences: deque = deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.experiences.append(experience)
        
    def sample_high_error(self, batch_size: int) -> List[Experience]:
        """Sample experiences with high prediction error."""
        if len(self.experiences) < batch_size:
            return list(self.experiences)
            
        # Sort by prediction error and sample from top experiences
        sorted_exp = sorted(self.experiences, 
                          key=lambda x: abs(x.learning_progress), 
                          reverse=True)
        return sorted_exp[:batch_size]
        
    def sample_random(self, batch_size: int) -> List[Experience]:
        """Sample random experiences."""
        import random
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        return random.sample(list(self.experiences), batch_size)
        
    def __len__(self):
        return len(self.experiences)


class GoalMemory:
    """Memory system for managing goals."""
    
    def __init__(self):
        self.active_goals: List[Goal] = []
        self.retired_goals: List[Goal] = []
        
    def add_goal(self, goal: Goal):
        """Add new goal to active goals."""
        self.active_goals.append(goal)
        
    def retire_goal(self, goal: Goal):
        """Move goal from active to retired."""
        if goal in self.active_goals:
            self.active_goals.remove(goal)
            self.retired_goals.append(goal)
            
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return self.active_goals.copy()
        
    def update_goal_success_rate(self, goal_id: str, success: bool):
        """Update success rate for a specific goal."""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                # Simple running average update
                current_rate = goal.success_rate
                goal.success_rate = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
                break


@dataclass
class MetricsSnapshot:
    """Snapshot of agent metrics at a point in time."""
    timestamp: int
    energy: float
    learning_progress: float
    prediction_error: float
    memory_usage: float
    active_goals_count: int
    position: Tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'energy': self.energy,
            'learning_progress': self.learning_progress,
            'prediction_error': self.prediction_error,
            'memory_usage': self.memory_usage,
            'active_goals_count': self.active_goals_count,
            'position': self.position.tolist() if self.position is not None else None
        }