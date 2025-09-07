"""
Action Selection System - Phase 2 implementation.

This module implements intelligent action selection for the adaptive learning agent,
including movement, food collection, and exploration strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

from .data_models import SensoryInput, AgentState, Goal

logger = logging.getLogger(__name__)


class ActionSelectionNetwork(nn.Module):
    """
    Neural network for action selection based on current state and goals.
    
    This network learns to select optimal actions based on:
    - Current sensory input
    - Agent state (position, energy, etc.)
    - Active goals
    - Memory state
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        action_size: int = 8,
        num_goals: int = 5
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.num_goals = num_goals
        
        # Main action selection network
        self.action_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Goal-specific action networks
        self.goal_networks = nn.ModuleDict({
            'survival': nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            ),
            'exploration': nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            ),
            'food_collection': nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size)
            )
        })
        
        # Action value estimation
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(
        self, 
        state_representation: torch.Tensor,
        active_goals: List[Goal],
        energy_level: float
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for action selection.
        
        Args:
            state_representation: Encoded state from predictive core
            active_goals: List of currently active goals
            energy_level: Current energy level (0-100)
            
        Returns:
            action_logits: Raw action logits
            action_values: Estimated action values
            debug_info: Additional information for debugging
        """
        batch_size = state_representation.size(0)
        
        # Main action selection
        action_logits = self.action_network(state_representation)
        
        # Goal-specific action selection
        goal_actions = {}
        for goal in active_goals:
            if goal.goal_type in self.goal_networks:
                goal_net = self.goal_networks[goal.goal_type]
                goal_actions[goal.goal_type] = goal_net(state_representation)
        
        # Value estimation
        action_values = self.value_network(state_representation)
        
        # Combine goal-specific actions with main actions
        if goal_actions:
            # Weight by goal priority and energy level
            energy_weight = min(energy_level / 100.0, 1.0)
            for goal_type, goal_action in goal_actions.items():
                if goal_type == 'survival' and energy_level < 30:
                    # Prioritize survival when energy is low
                    action_logits = action_logits + goal_action * 2.0 * energy_weight
                elif goal_type == 'food_collection' and energy_level < 50:
                    # Prioritize food collection when energy is moderate
                    action_logits = action_logits + goal_action * 1.5 * energy_weight
                else:
                    # Normal goal weighting
                    action_logits = action_logits + goal_action * energy_weight
        
        debug_info = {
            'goal_actions': goal_actions,
            'energy_weight': energy_weight,
            'action_logits_norm': torch.norm(action_logits, dim=-1).mean().item()
        }
        
        return action_logits, action_values, debug_info


class ActionExecutor:
    """
    Executes selected actions in the environment.
    
    Handles:
    - Movement in 3D space
    - Food collection
    - Energy management
    - Goal progress tracking
    """
    
    def __init__(self, max_velocity: float = 2.0, action_noise: float = 0.1):
        self.max_velocity = max_velocity
        self.action_noise = action_noise
        
    def execute_action(
        self,
        action: torch.Tensor,
        agent_state: AgentState,
        environment_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the selected action in the environment.
        
        Args:
            action: Selected action tensor
            agent_state: Current agent state
            environment_state: Current environment state
            
        Returns:
            results: Dictionary with action results
        """
        # Parse action components
        movement = action[:3]  # x, y, z movement
        interaction = action[3:6]  # interaction actions (collect, use, etc.)
        exploration = action[6:8]  # exploration actions
        
        # Apply action noise for exploration
        if self.action_noise > 0:
            noise = torch.randn_like(movement) * self.action_noise
            movement = movement + noise
        
        # Clamp movement to max velocity
        movement_norm = torch.norm(movement)
        if movement_norm > self.max_velocity:
            movement = movement * (self.max_velocity / movement_norm)
        
        # Update position
        new_position = agent_state.position + movement
        
        # Check for food collection
        food_collected = self._check_food_collection(new_position, environment_state)
        
        # Check for obstacles
        collision = self._check_collision(new_position, environment_state)
        if collision:
            new_position = agent_state.position  # Don't move if collision
        
        # Calculate action cost
        action_cost = torch.norm(movement).item()
        
        # Update agent state
        agent_state.position = new_position
        agent_state.previous_position = agent_state.position.clone()
        
        return {
            'new_position': new_position,
            'food_collected': food_collected,
            'collision': collision,
            'action_cost': action_cost,
            'movement_magnitude': movement_norm.item()
        }
    
    def _check_food_collection(
        self, 
        position: torch.Tensor, 
        environment_state: Dict[str, Any]
    ) -> bool:
        """Check if agent can collect food at current position."""
        # This would integrate with the environment's food collection system
        # For now, return False
        return False
    
    def _check_collision(
        self, 
        position: torch.Tensor, 
        environment_state: Dict[str, Any]
    ) -> bool:
        """Check if agent would collide with obstacles at new position."""
        # This would integrate with the environment's collision detection
        # For now, return False
        return False


class ExplorationStrategy:
    """
    Implements exploration strategies for the agent.
    
    Strategies:
    - Random exploration
    - Goal-directed exploration
    - Curiosity-driven exploration
    - Memory-based exploration
    """
    
    def __init__(self, exploration_rate: float = 0.1, curiosity_weight: float = 0.3):
        self.exploration_rate = exploration_rate
        self.curiosity_weight = curiosity_weight
        self.visited_positions = []
        self.exploration_bonus = 0.0
        
    def get_exploration_bonus(
        self, 
        position: torch.Tensor, 
        learning_progress: float
    ) -> float:
        """
        Calculate exploration bonus for a position.
        
        Args:
            position: Position to evaluate
            learning_progress: Current learning progress signal
            
        Returns:
            bonus: Exploration bonus value
        """
        # Distance-based exploration bonus
        if not self.visited_positions:
            return 1.0
        
        # Calculate distance to nearest visited position
        distances = [torch.norm(position - torch.tensor(p)) for p in self.visited_positions]
        min_distance = min(distances)
        
        # Normalize distance (assuming world size is roughly 20x20x5)
        normalized_distance = min_distance / 25.0  # sqrt(20^2 + 20^2 + 5^2)
        
        # Combine with learning progress
        exploration_bonus = normalized_distance * (1.0 + learning_progress * self.curiosity_weight)
        
        return exploration_bonus
    
    def update_visited_positions(self, position: torch.Tensor):
        """Update list of visited positions."""
        pos_list = position.tolist()
        if pos_list not in self.visited_positions:
            self.visited_positions.append(pos_list)
            
        # Keep only recent positions to avoid memory bloat
        if len(self.visited_positions) > 1000:
            self.visited_positions = self.visited_positions[-500:]
    
    def should_explore(self, learning_progress: float, energy_level: float) -> bool:
        """
        Determine if agent should explore or exploit.
        
        Args:
            learning_progress: Current learning progress
            energy_level: Current energy level
            
        Returns:
            should_explore: True if agent should explore
        """
        # High learning progress encourages exploration
        lp_factor = learning_progress * 2.0
        
        # Low energy encourages exploitation (survival)
        energy_factor = max(0, (100 - energy_level) / 100.0)
        
        # Combine factors
        exploration_probability = self.exploration_rate + lp_factor - energy_factor
        
        return np.random.random() < exploration_probability 