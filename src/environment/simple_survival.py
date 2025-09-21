"""
Simple survival environment for Phase 0 energy system testing.

This provides a minimal 2D grid world with energy sources and basic physics
for validating the energy and death mechanics.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
import random

from ..core.data_models import SensoryInput, AgentState
from ..core.unified_energy_system import UnifiedEnergySystem


class SimpleSurvivalEnvironment:
    """
    Simple 2D grid world for survival testing.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),
        num_food_sources: int = 5,
        food_respawn_rate: float = 0.01,
        visual_range: int = 5,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.num_food_sources = num_food_sources
        self.food_respawn_rate = food_respawn_rate
        self.visual_range = visual_range
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        # Environment state
        self.grid = np.zeros(grid_size, dtype=np.float32)  # 0=empty, 1=food, -1=obstacle
        self.agent_position = np.array([grid_size[0]//2, grid_size[1]//2])
        self.step_count = 0
        
        # Initialize food sources
        self._spawn_food_sources()
        
    def _spawn_food_sources(self):
        """Spawn food sources randomly in the environment."""
        # Clear existing food
        self.grid[self.grid == 1.0] = 0.0
        
        # Spawn new food sources
        for _ in range(self.num_food_sources):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                
                # Don't spawn on agent or existing food
                if (x, y) != tuple(self.agent_position) and self.grid[x, y] == 0:
                    self.grid[x, y] = 1.0
                    break
                    
    def _get_visual_observation(self) -> torch.Tensor:
        """Get visual observation around agent."""
        # Create visual field centered on agent
        visual_size = 2 * self.visual_range + 1
        visual_field = np.zeros((3, visual_size, visual_size), dtype=np.float32)
        
        agent_x, agent_y = self.agent_position
        
        for dx in range(-self.visual_range, self.visual_range + 1):
            for dy in range(-self.visual_range, self.visual_range + 1):
                world_x = agent_x + dx
                world_y = agent_y + dy
                
                vis_x = dx + self.visual_range
                vis_y = dy + self.visual_range
                
                # Check bounds
                if 0 <= world_x < self.grid_size[0] and 0 <= world_y < self.grid_size[1]:
                    cell_value = self.grid[world_x, world_y]
                    
                    if cell_value == 1.0:  # Food
                        visual_field[0, vis_x, vis_y] = 1.0  # Red channel for food
                    elif cell_value == -1.0:  # Obstacle
                        visual_field[2, vis_x, vis_y] = 1.0  # Blue channel for obstacles
                else:
                    # Out of bounds (walls)
                    visual_field[2, vis_x, vis_y] = 0.5
                    
        # Mark agent position
        center = self.visual_range
        visual_field[1, center, center] = 1.0  # Green channel for agent
        
        return torch.from_numpy(visual_field)
        
    def _get_proprioception(self) -> torch.Tensor:
        """Get proprioceptive information."""
        # Simple proprioception: position, velocity, etc.
        proprio = torch.zeros(4)
        
        # Normalized position
        proprio[0] = self.agent_position[0] / self.grid_size[0]
        proprio[1] = self.agent_position[1] / self.grid_size[1]
        
        # Distance to nearest food
        food_positions = np.argwhere(self.grid == 1.0)
        if len(food_positions) > 0:
            distances = np.linalg.norm(food_positions - self.agent_position, axis=1)
            min_distance = np.min(distances)
            proprio[2] = min_distance / np.sqrt(sum(s**2 for s in self.grid_size))
        else:
            proprio[2] = 1.0  # Max distance if no food
            
        # Step count (normalized)
        proprio[3] = min(self.step_count / 1000.0, 1.0)
        
        return proprio
        
    def get_observation(self, energy_level: float) -> SensoryInput:
        """Get current sensory observation."""
        visual = self._get_visual_observation()
        proprioception = self._get_proprioception()
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=energy_level,
            timestamp=self.step_count
        )
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action tensor [dx, dy] for movement
            
        Returns:
            observation: New sensory input
            reward: Environment reward (food collection)
            done: Episode termination flag
            info: Additional information
        """
        self.step_count += 1
        
        # Parse action (movement)
        if len(action) >= 2:
            dx = torch.clamp(action[0], -1, 1).item()
            dy = torch.clamp(action[1], -1, 1).item()
        else:
            dx, dy = 0, 0
            
        # Move agent
        new_x = np.clip(self.agent_position[0] + int(np.round(dx)), 0, self.grid_size[0] - 1)
        new_y = np.clip(self.agent_position[1] + int(np.round(dy)), 0, self.grid_size[1] - 1)
        
        # Check for obstacles
        if self.grid[new_x, new_y] != -1.0:  # Not an obstacle
            self.agent_position = np.array([new_x, new_y])
            
        # Check for food collection
        reward = 0.0
        if self.grid[self.agent_position[0], self.agent_position[1]] == 1.0:
            # Collect food
            self.grid[self.agent_position[0], self.agent_position[1]] = 0.0
            reward = 1.0
            
        # Randomly respawn food
        if random.random() < self.food_respawn_rate:
            self._spawn_single_food()
            
        # Get new observation (energy will be filled by caller)
        observation = self.get_observation(0.0)  # Placeholder energy
        
        # Episode termination (for now, never ends)
        done = False
        
        info = {
            'food_collected': reward > 0,
            'agent_position': self.agent_position.copy(),
            'food_count': np.sum(self.grid == 1.0)
        }
        
        return observation, reward, done, info
        
    def _spawn_single_food(self):
        """Spawn a single food source randomly."""
        attempts = 0
        while attempts < 50:  # Prevent infinite loop
            x = random.randint(0, self.grid_size[0] - 1)
            y = random.randint(0, self.grid_size[1] - 1)
            
            # Don't spawn on agent or existing food/obstacles
            if (x, y) != tuple(self.agent_position) and self.grid[x, y] == 0:
                self.grid[x, y] = 1.0
                break
                
            attempts += 1
            
    def reset(self) -> SensoryInput:
        """Reset environment to initial state."""
        self.grid.fill(0.0)
        self.agent_position = np.array([self.grid_size[0]//2, self.grid_size[1]//2])
        self.step_count = 0
        
        self._spawn_food_sources()
        
        return self.get_observation(0.0)  # Placeholder energy
        
    def render(self) -> str:
        """Render environment as ASCII string."""
        display = np.full(self.grid_size, '.', dtype=str)
        
        # Add food
        food_positions = np.argwhere(self.grid == 1.0)
        for pos in food_positions:
            display[pos[0], pos[1]] = 'F'
            
        # Add obstacles
        obstacle_positions = np.argwhere(self.grid == -1.0)
        for pos in obstacle_positions:
            display[pos[0], pos[1]] = '#'
            
        # Add agent
        display[self.agent_position[0], self.agent_position[1]] = 'A'
        
        # Convert to string
        lines = []
        for row in display:
            lines.append(''.join(row))
            
        return '\n'.join(lines)
        
    def get_state_dict(self) -> Dict:
        """Get environment state for saving/loading."""
        return {
            'grid': self.grid.copy(),
            'agent_position': self.agent_position.copy(),
            'step_count': self.step_count
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load environment state."""
        self.grid = state_dict['grid'].copy()
        self.agent_position = state_dict['agent_position'].copy()
        self.step_count = state_dict['step_count']


class RandomAgent:
    """Simple random agent for testing environment."""
    
    def __init__(self, action_space_size: int = 2):
        self.action_space_size = action_space_size
        
    def act(self, observation: SensoryInput) -> torch.Tensor:
        """Generate random action."""
        return torch.randn(self.action_space_size) * 0.5  # Small random movements
        
    def get_action_cost(self, action: torch.Tensor) -> float:
        """Compute action cost for energy system."""
        return torch.norm(action).item() * 0.1  # Cost proportional to action magnitude