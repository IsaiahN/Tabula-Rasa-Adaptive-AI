"""
Energy and Death System - Survival pressure through limited resources.

This module implements the energy consumption, death mechanics, and 
selective memory preservation system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from src.core.data_models import AgentState

logger = logging.getLogger(__name__)


class EnergySystem:
    """
    Manages agent energy consumption, death detection, and respawn mechanics.
    """
    
    def __init__(
        self,
        max_energy: float = 100.0,
        base_consumption: float = 0.01,
        action_multiplier: float = 0.5,
        computation_multiplier: float = 0.001,
        food_energy_value: float = 10.0
    ):
        self.max_energy = max_energy
        self.base_consumption = base_consumption
        self.action_multiplier = action_multiplier
        self.computation_multiplier = computation_multiplier
        self.food_energy_value = food_energy_value
        
        # Current energy level
        self.current_energy = max_energy
        
        # Energy tracking
        self.energy_history = []
        self.consumption_history = []
        
    def consume_energy(
        self, 
        action_cost: float = 0.0, 
        computation_cost: float = 0.0
    ) -> float:
        """
        Consume energy based on actions and computation.
        
        Args:
            action_cost: Cost of current action (0-1 range)
            computation_cost: Computational cost (number of forward passes)
            
        Returns:
            Remaining energy after consumption
        """
        # Calculate total consumption
        total_consumption = (
            self.base_consumption +
            action_cost * self.action_multiplier +
            computation_cost * self.computation_multiplier
        )
        
        # Update energy
        self.current_energy = max(0.0, self.current_energy - total_consumption)
        
        # Track consumption
        self.consumption_history.append(total_consumption)
        self.energy_history.append(self.current_energy)
        
        # Keep history bounded
        if len(self.energy_history) > 10000:
            self.energy_history = self.energy_history[-5000:]
            self.consumption_history = self.consumption_history[-5000:]
            
        return self.current_energy
        
    def add_energy(self, amount: float) -> float:
        """
        Add energy (e.g., from food sources).
        
        Args:
            amount: Energy to add
            
        Returns:
            New energy level (capped at max)
        """
        self.current_energy = min(self.max_energy, self.current_energy + amount)
        self.energy_history.append(self.current_energy)
        return self.current_energy
        
    def is_dead(self) -> bool:
        """Check if agent has died (energy <= 0)."""
        return self.current_energy <= 0.0
        
    def get_energy_level(self) -> float:
        """Get current energy level."""
        return self.current_energy
        
    def get_energy_ratio(self) -> float:
        """Get energy as ratio of maximum (0-1)."""
        return self.current_energy / self.max_energy
        
    def should_sleep(self) -> bool:
        """Check if agent should enter sleep mode (low energy)."""
        return self.current_energy < 0.2 * self.max_energy
        
    def reset_energy(self):
        """Reset energy to maximum (for respawn)."""
        self.current_energy = self.max_energy
        
    def get_energy_metrics(self) -> Dict[str, float]:
        """Get energy system metrics for monitoring."""
        if not self.energy_history:
            return {
                'current_energy': self.current_energy,
                'energy_ratio': self.get_energy_ratio(),
                'average_consumption': 0.0,
                'energy_trend': 0.0
            }
            
        recent_consumption = self.consumption_history[-100:] if len(self.consumption_history) >= 100 else self.consumption_history
        recent_energy = self.energy_history[-100:] if len(self.energy_history) >= 100 else self.energy_history
        
        # Calculate trend
        if len(recent_energy) > 10:
            x = np.arange(len(recent_energy))
            trend = np.polyfit(x, recent_energy, 1)[0]
        else:
            trend = 0.0
            
        return {
            'current_energy': self.current_energy,
            'energy_ratio': self.get_energy_ratio(),
            'average_consumption': np.mean(recent_consumption) if recent_consumption else 0.0,
            'energy_trend': trend
        }


class ImportanceNetwork(nn.Module):
    """
    Neural network that learns which memories are important to preserve across deaths.
    """
    
    def __init__(self, memory_size: int = 512, word_size: int = 64):
        super().__init__()
        
        self.memory_size = memory_size
        self.word_size = word_size
        
        # Network to score memory importance
        self.importance_net = nn.Sequential(
            nn.Linear(word_size + 1, 128),  # +1 for usage frequency
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Importance score 0-1
        )
        
    def forward(
        self, 
        memory_matrix: torch.Tensor, 
        usage_history: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute importance scores for memory locations.
        
        Args:
            memory_matrix: [memory_size, word_size]
            usage_history: [memory_size] - usage frequency
            
        Returns:
            importance_scores: [memory_size] - importance scores
        """
        # Combine memory content with usage frequency
        memory_with_usage = torch.cat([
            memory_matrix, 
            usage_history.unsqueeze(-1)
        ], dim=-1)
        
        # Compute importance scores
        importance_scores = self.importance_net(memory_with_usage).squeeze(-1)
        
        return importance_scores


class HeuristicImportanceScorer:
    """
    Rule-based importance scoring for Phase 1 (before learned network).
    """
    
    def __init__(self):
        self.usage_weight = 0.6
        self.recency_weight = 0.3
        self.diversity_weight = 0.1
        
    def score(
        self, 
        memory_matrix: torch.Tensor, 
        usage_history: torch.Tensor,
        access_recency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute heuristic importance scores.
        
        Args:
            memory_matrix: [memory_size, word_size]
            usage_history: [memory_size] - usage frequency
            access_recency: [memory_size] - steps since last access
            
        Returns:
            importance_scores: [memory_size] - importance scores
        """
        memory_size = memory_matrix.size(0)
        
        # Usage-based score (normalized)
        usage_scores = usage_history / (usage_history.max() + 1e-8)
        
        # Recency score (more recent = higher score)
        if access_recency is not None:
            max_recency = access_recency.max() + 1e-8
            recency_scores = 1.0 - (access_recency / max_recency)
        else:
            recency_scores = torch.ones_like(usage_scores)
            
        # Diversity score (unique memories are more important)
        memory_norms = torch.norm(memory_matrix, dim=-1)
        diversity_scores = memory_norms / (memory_norms.max() + 1e-8)
        
        # Combined score
        importance_scores = (
            self.usage_weight * usage_scores +
            self.recency_weight * recency_scores +
            self.diversity_weight * diversity_scores
        )
        
        return importance_scores


class DeathManager:
    """
    Manages agent death, memory preservation, and respawn mechanics.
    """
    
    def __init__(
        self, 
        memory_size: int = 512,
        word_size: int = 64,
        use_learned_importance: bool = False,
        preservation_ratio: float = 0.2
    ):
        self.memory_size = memory_size
        self.word_size = word_size
        self.use_learned_importance = use_learned_importance
        self.preservation_ratio = preservation_ratio
        
        # Importance scoring systems
        self.heuristic_scorer = HeuristicImportanceScorer()
        self.learned_network = ImportanceNetwork(memory_size, word_size) if use_learned_importance else None
        
        # Death/rebirth tracking
        self.death_count = 0
        self.recovery_times = []
        
    def compute_memory_importance(
        self, 
        memory_matrix: torch.Tensor, 
        usage_history: torch.Tensor,
        access_recency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute importance scores for memory preservation.
        
        Args:
            memory_matrix: [memory_size, word_size]
            usage_history: [memory_size] - usage frequency
            access_recency: [memory_size] - steps since last access
            
        Returns:
            importance_scores: [memory_size] - importance scores
        """
        if not self.use_learned_importance or self.learned_network is None:
            # Use heuristic scoring
            return self.heuristic_scorer.score(memory_matrix, usage_history, access_recency)
        else:
            # Use learned network
            return self.learned_network(memory_matrix, usage_history)
            
    def selective_reset(self, agent_state: AgentState) -> AgentState:
        """
        Perform selective reset preserving important memories.
        
        Args:
            agent_state: Current agent state
            
        Returns:
            new_state: Reset agent state with preserved memories
        """
        self.death_count += 1
        
        # Compute memory importance if memory exists
        if agent_state.memory_state is not None and hasattr(agent_state, 'memory_usage'):
            importance_scores = self.compute_memory_importance(
                agent_state.memory_state, 
                getattr(agent_state, 'memory_usage', torch.ones(self.memory_size))
            )
            
            # Preserve top memories by importance
            num_preserve = int(self.memory_size * self.preservation_ratio)
            _, top_indices = torch.topk(importance_scores, num_preserve)
            
            # Create preservation mask
            preserve_mask = torch.zeros(self.memory_size, dtype=torch.bool)
            preserve_mask[top_indices] = True
            
            # Apply selective preservation
            preserved_memory = agent_state.memory_state.clone()
            preserved_memory[~preserve_mask] = 0.0
        else:
            preserved_memory = None
            
        # Create new agent state
        new_state = AgentState(
            position=self._random_spawn_position(),
            orientation=self._random_spawn_orientation(),
            energy=100.0,  # Full energy on respawn
            hidden_state=None,  # Reset recurrent state (will be reinitialized)
            active_goals=[],  # Clear goals
            memory_state=preserved_memory,
            timestamp=0
        )
        
        logger.info(f"Agent death #{self.death_count}. Memory preservation: {self.preservation_ratio:.1%}")
        
        return new_state
        
    def _random_spawn_position(self) -> torch.Tensor:
        """Generate random spawn position."""
        # Spawn in safe area (center of environment)
        return torch.tensor([0.0, 0.0, 1.0])  # x, y, z coordinates
        
    def _random_spawn_orientation(self) -> torch.Tensor:
        """Generate random spawn orientation."""
        # Random quaternion
        angles = torch.rand(3) * 2 * np.pi
        return torch.tensor([0.0, 0.0, 0.0, 1.0])  # Identity quaternion for simplicity
        
    def update_importance_network(
        self, 
        death_experiences: list, 
        recovery_performances: list
    ):
        """
        Update learned importance network based on death/recovery data.
        
        Args:
            death_experiences: List of (memory, usage, recovery_time) tuples
            recovery_performances: List of performance metrics after recovery
        """
        if not self.use_learned_importance or self.learned_network is None:
            return
            
        # TODO: Implement REINFORCE-style training
        # This would train the importance network to maximize recovery speed
        pass
        
    def get_death_metrics(self) -> Dict[str, float]:
        """Get death and recovery metrics."""
        avg_recovery = np.mean(self.recovery_times) if self.recovery_times else 0.0
        
        return {
            'death_count': self.death_count,
            'average_recovery_time': avg_recovery,
            'preservation_ratio': self.preservation_ratio
        }