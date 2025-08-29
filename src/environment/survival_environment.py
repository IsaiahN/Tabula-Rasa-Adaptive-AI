"""
Survival Environment - Main environment for Phase 1 agent testing.

This module implements a simple 3D environment with basic survival mechanics
including food sources, energy consumption, and physics simulation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import time

from core.data_models import SensoryInput, AgentState

logger = logging.getLogger(__name__)


@dataclass
class FoodSource:
    """Represents a food source in the environment."""
    position: torch.Tensor  # 3D coordinates
    energy_value: float
    respawn_time: float
    last_consumed: float
    is_active: bool = True


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""
    position: torch.Tensor  # 3D coordinates
    size: torch.Tensor  # 3D dimensions
    type: str  # "wall", "rock", etc.


class SurvivalEnvironment:
    """
    Simple 3D survival environment for Phase 1 testing.
    
    Features:
    - 3D grid-based world
    - Food sources that respawn
    - Basic physics and collision detection
    - Energy consumption mechanics
    - Configurable complexity
    """
    
    def __init__(
        self,
        world_size: Tuple[int, int, int] = (20, 20, 5),
        num_food_sources: int = 5,
        food_respawn_time: float = 30.0,
        food_energy_value: float = 10.0,
        complexity_level: int = 1,
        physics_enabled: bool = True
    ):
        self.world_size = world_size
        self.num_food_sources = num_food_sources
        self.food_respawn_time = food_respawn_time
        self.food_energy_value = food_energy_value
        self.complexity_level = complexity_level
        self.physics_enabled = physics_enabled
        
        # Environment state
        self.food_sources: List[FoodSource] = []
        self.obstacles: List[Obstacle] = []
        self.time_step = 0
        
        # Initialize environment
        self._generate_food_sources()
        self._generate_obstacles()
        
        # Physics parameters
        self.gravity = -9.81
        self.friction = 0.8
        self.max_velocity = 5.0
        
        logger.info(f"Survival environment initialized: {world_size}, complexity {complexity_level}")
        
    def _generate_food_sources(self):
        """Generate initial food sources in the environment."""
        for i in range(self.num_food_sources):
            # Random position within world bounds
            x = np.random.uniform(-self.world_size[0]//2, self.world_size[0]//2)
            y = np.random.uniform(-self.world_size[1]//2, self.world_size[1]//2)
            z = 1.0  # Ground level
            
            food_source = FoodSource(
                position=torch.tensor([x, y, z]),
                energy_value=self.food_energy_value,
                respawn_time=self.food_respawn_time,
                last_consumed=0.0
            )
            
            self.food_sources.append(food_source)
            
        logger.info(f"Generated {self.num_food_sources} food sources")
        
    def _generate_obstacles(self):
        """Generate obstacles based on complexity level."""
        if self.complexity_level < 2:
            return  # No obstacles in basic complexity
            
        num_obstacles = min(self.complexity_level * 2, 10)
        
        for i in range(num_obstacles):
            # Random position
            x = np.random.uniform(-self.world_size[0]//2 + 2, self.world_size[0]//2 - 2)
            y = np.random.uniform(-self.world_size[1]//2 + 2, self.world_size[1]//2 - 2)
            z = 0.5  # Half height
            
            # Random size
            size_x = np.random.uniform(1.0, 3.0)
            size_y = np.random.uniform(1.0, 3.0)
            size_z = np.random.uniform(1.0, 2.0)
            
            obstacle = Obstacle(
                position=torch.tensor([x, y, z]),
                size=torch.tensor([size_x, size_y, size_z]),
                type="rock"
            )
            
            self.obstacles.append(obstacle)
            
        logger.info(f"Generated {num_obstacles} obstacles")
        
    def step(
        self, 
        agent,  # Remove type hint to avoid circular import
        action: torch.Tensor
    ) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """
        Execute one environment step.
        
        Args:
            agent: The learning agent
            action: Action to take
            
        Returns:
            sensory_input: Sensory input for the agent
            action_result: Results of the action
            done: Whether episode is complete
        """
        self.time_step += 1
        
        # Get current agent state
        agent_state = agent.get_agent_state()
        
        # Apply action and update physics
        new_position, action_result = self._apply_action(agent_state, action)
        
        # Check for food consumption
        food_consumed, food_source = self._check_food_consumption(new_position)
        if food_consumed:
            action_result['food_collected'] = True
            action_result['energy_gained'] = self.food_energy_value
            action_result['food_position'] = food_source.position.tolist()
            # Add energy to agent
            agent.energy_system.add_energy(self.food_energy_value)
            # Mark food source as consumed
            food_source.is_active = False
            food_source.last_consumed = self.time_step
            logger.info(f"Agent collected food at {food_source.position}, gained {self.food_energy_value} energy")
        else:
            action_result['food_collected'] = False
            action_result['energy_gained'] = 0.0
            
        # Check for collisions
        collision = self._check_collisions(new_position)
        if collision:
            action_result['collision'] = True
            action_result['damage'] = 5.0
            # Reduce energy due to collision
            agent.energy_system.consume_energy(action_cost=0.0, computation_cost=5.0)
        else:
            action_result['collision'] = False
            action_result['damage'] = 0.0
            
        # Update food sources (respawn logic)
        self._update_food_sources()
        
        # Generate sensory input
        sensory_input = self._generate_sensory_input(agent_state, new_position)
        
        # Check if episode is done
        done = self._check_episode_done(agent_state)
        
        return sensory_input, action_result, done
        
    def _apply_action(
        self, 
        agent_state: AgentState, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply action and update agent position.
        
        Args:
            agent_state: Current agent state
            action: Action vector [dx, dy, dz]
            
        Returns:
            new_position: Updated position
            action_result: Results of the action
        """
        current_pos = agent_state.position.clone()
        
        # Parse action (assuming [dx, dy, dz] movement)
        if action is None or action.numel() == 0:
            movement = torch.zeros(3)
        else:
            # Normalize action to reasonable movement range
            movement = torch.clamp(action[:3], -1.0, 1.0) * 0.5
            
        # Apply movement
        new_position = current_pos + movement
        
        # Apply physics constraints
        if self.physics_enabled:
            new_position = self._apply_physics_constraints(new_position)
            
        # Keep within world bounds
        new_position = self._clamp_to_world_bounds(new_position)
        
        action_result = {
            'movement_applied': movement.tolist(),
            'position_change': (new_position - current_pos).tolist(),
            'action_magnitude': torch.norm(movement).item()
        }
        
        return new_position, action_result
        
    def _apply_physics_constraints(self, position: torch.Tensor) -> torch.Tensor:
        """Apply basic physics constraints."""
        # Ground constraint
        if position[2] < 1.0:
            position[2] = 1.0
            
        # Ceiling constraint
        if position[2] > self.world_size[2] - 1:
            position[2] = self.world_size[2] - 1
            
        return position
        
    def _clamp_to_world_bounds(self, position: torch.Tensor) -> torch.Tensor:
        """Clamp position to world boundaries."""
        half_size = torch.tensor(self.world_size) / 2
        
        position[0] = torch.clamp(position[0], -half_size[0] + 1, half_size[0] - 1)
        position[1] = torch.clamp(position[1], -half_size[1] + 1, half_size[1] - 1)
        position[2] = torch.clamp(position[2], 1.0, half_size[2] - 1)
        
        return position
        
    def _check_food_consumption(self, position: torch.Tensor) -> Tuple[bool, Optional[FoodSource]]:
        """Check if agent is consuming food at current position."""
        for food_source in self.food_sources:
            if not food_source.is_active:
                continue
                
            # Check distance to food source
            distance = torch.norm(position - food_source.position)
            if distance < 1.5:  # Consumption radius
                return True, food_source
                
        return False, None
        
    def _check_collisions(self, position: torch.Tensor) -> bool:
        """Check for collisions with obstacles."""
        for obstacle in self.obstacles:
            # Simple bounding box collision detection
            half_size = obstacle.size / 2
            min_bound = obstacle.position - half_size
            max_bound = obstacle.position + half_size
            
            if (position >= min_bound).all() and (position <= max_bound).all():
                return True
                
        return False
        
    def _update_food_sources(self):
        """Update food source respawn logic."""
        current_time = self.time_step
        
        for food_source in self.food_sources:
            if not food_source.is_active:
                # Check if enough time has passed for respawn
                if current_time - food_source.last_consumed >= self.food_respawn_time:
                    food_source.is_active = True
                    logger.debug(f"Food respawned at {food_source.position}")
                    
    def _generate_sensory_input(
        self, 
        agent_state: AgentState, 
        position: torch.Tensor
    ) -> SensoryInput:
        """
        Generate sensory input for the agent.
        
        Args:
            agent_state: Current agent state
            position: Current position
            
        Returns:
            sensory_input: Multi-modal sensory data
        """
        # Visual input (simplified - would be actual rendering in real implementation)
        visual_input = self._generate_visual_input(position)
        
        # Proprioception (position, orientation, velocity)
        proprioception = self._generate_proprioception(agent_state, position)
        
        # Energy level
        energy_level = agent_state.energy
        
        # Create sensory input
        sensory_input = SensoryInput(
            visual=visual_input,
            proprioception=proprioception,
            energy_level=energy_level,
            timestamp=self.time_step
        )
        
        return sensory_input
        
    def _generate_visual_input(self, position: torch.Tensor) -> torch.Tensor:
        """Generate simplified visual input."""
        # For Phase 1, generate a simple visual representation
        # This would be replaced with actual rendering in later phases
        
        # Create a 64x64 visual field centered on the agent
        # visual_size should be (channels, height, width)
        visual_input = torch.zeros(3, 64, 64)  # RGB channels, height, width
        
        # Add food source indicators
        for food_source in self.food_sources:
            if food_source.is_active:
                # Convert world position to visual coordinates
                rel_pos = food_source.position - position
                visual_x = int(32 + rel_pos[0] * 10)  # Scale factor
                visual_y = int(32 + rel_pos[1] * 10)
                
                # Clamp to visual bounds
                visual_x = max(0, min(63, visual_x))
                visual_y = max(0, min(63, visual_y))
                
                # Mark food source in visual field (green channel)
                visual_input[1, visual_y, visual_x] = 1.0
                
        # Add obstacle indicators (red channel)
        for obstacle in self.obstacles:
            rel_pos = obstacle.position - position
            visual_x = int(32 + rel_pos[0] * 10)
            visual_y = int(32 + rel_pos[1] * 10)
            
            visual_x = max(0, min(63, visual_x))
            visual_y = max(0, min(63, visual_y))
            
            visual_input[0, visual_y, visual_x] = 1.0
            
        return visual_input.unsqueeze(0)  # Add batch dimension (1, 3, 64, 64)
        
    def _generate_proprioception(
        self, 
        agent_state: AgentState, 
        position: torch.Tensor
    ) -> torch.Tensor:
        """Generate proprioceptive input."""
        # Position (x, y, z)
        pos_input = position
        
        # Orientation (quaternion)
        orientation = agent_state.orientation
        
        # Velocity (simplified - difference from previous position)
        if hasattr(agent_state, 'previous_position'):
            velocity = position - agent_state.previous_position
        else:
            velocity = torch.zeros(3)
            
        # Energy level (normalized)
        energy_norm = torch.tensor([agent_state.energy / 100.0])
        
        # Combine all proprioceptive inputs
        proprioception = torch.cat([
            pos_input,      # 3 values
            orientation,     # 4 values
            velocity,        # 3 values
            energy_norm      # 1 value
        ])
        
        return proprioception.unsqueeze(0)  # Add batch dimension (1, 11)
        
    def _check_episode_done(self, agent_state: AgentState) -> bool:
        """Check if episode should end."""
        # Episode ends if agent dies
        if agent_state.energy <= 0:
            return True
            
        # Episode ends after maximum steps
        if self.time_step >= 10000:  # 10k steps max
            return True
            
        return False
        
    def increase_complexity(self):
        """Increase environment complexity."""
        self.complexity_level += 1
        
        # Add more obstacles
        self._generate_obstacles()
        
        # Add more food sources
        additional_food = min(2, self.complexity_level)
        for i in range(additional_food):
            x = np.random.uniform(-self.world_size[0]//2, self.world_size[0]//2)
            y = np.random.uniform(-self.world_size[1]//2, self.world_size[1]//2)
            z = 1.0
            
            food_source = FoodSource(
                position=torch.tensor([x, y, z]),
                energy_value=self.food_energy_value,
                respawn_time=self.food_respawn_time,
                last_consumed=0.0
            )
            
            self.food_sources.append(food_source)
            
        logger.info(f"Environment complexity increased to level {self.complexity_level}")
        
    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state for monitoring."""
        active_food = sum(1 for f in self.food_sources if f.is_active)
        
        return {
            'time_step': self.time_step,
            'complexity_level': self.complexity_level,
            'active_food_sources': active_food,
            'total_food_sources': len(self.food_sources),
            'obstacles': len(self.obstacles),
            'world_size': self.world_size
        }
        
    def reset(self):
        """Reset environment for new episode."""
        self.time_step = 0
        
        # Reactivate all food sources
        for food_source in self.food_sources:
            food_source.is_active = True
            food_source.last_consumed = 0.0
            
        logger.info("Environment reset for new episode") 