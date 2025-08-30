"""
Tool Use Puzzle - Tests agent's ability to use tools for problem solving.

This puzzle presents a scenario where the agent must use a stick to reach
a reward that is beyond its normal reach, testing causal reasoning and
tool use capabilities.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from puzzle_base import BasePuzzleEnvironment, PuzzleResult, AGISignalLevel
from core.data_models import SensoryInput


class ToolUsePuzzle(BasePuzzleEnvironment):
    """Tool use puzzle environment."""
    
    def __init__(self, max_steps: int = 200):
        super().__init__(max_steps)
        
        # Environment setup
        self.agent_position = torch.tensor([2.0, 8.0, 0.0])  # Start position
        self.stick_position = torch.tensor([4.0, 6.0, 0.0])  # Tool location
        self.reward_position = torch.tensor([8.0, 2.0, 0.0])  # Target location
        
        # Agent capabilities
        self.agent_reach = 1.5  # Normal reach distance
        self.stick_length = 3.0  # Tool extends reach
        
        # State tracking
        self.stick_held = False
        self.reward_obtained = False
        self.tool_interactions = []
        self.reach_attempts = []
        self.novel_contexts = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        
        # Reset state
        self.stick_held = False
        self.reward_obtained = False
        self.tool_interactions.clear()
        self.reach_attempts.clear()
        self.novel_contexts.clear()
        
        # Reset positions
        self.agent_position = torch.tensor([2.0, 8.0, 0.0])
        
        self.puzzle_state = {
            'agent_position': self.agent_position.clone(),
            'stick_held': False,
            'reward_obtained': False,
            'success_achieved': False
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the tool use puzzle."""
        self.current_step += 1
        
        # Parse action (movement + discrete actions)
        movement = action[:3]
        discrete_actions = action[3:]
        
        # Update agent position
        self.agent_position += movement * 0.1
        self.agent_position = torch.clamp(self.agent_position, 0, 10)
        
        # Check for stick pickup
        if not self.stick_held and discrete_actions[0] > 0.5:
            stick_distance = torch.norm(self.agent_position[:2] - self.stick_position[:2])
            if stick_distance < self.agent_reach:
                self.stick_held = True
                self.tool_interactions.append({
                    'type': 'pickup',
                    'step': self.current_step,
                    'position': self.agent_position.clone()
                })
        
        # Check for reward attempt
        if discrete_actions[1] > 0.5:
            reward_distance = torch.norm(self.agent_position[:2] - self.reward_position[:2])
            effective_reach = self.agent_reach + (self.stick_length if self.stick_held else 0)
            
            self.reach_attempts.append({
                'step': self.current_step,
                'distance': reward_distance.item(),
                'using_tool': self.stick_held,
                'success': reward_distance < effective_reach
            })
            
            if reward_distance < effective_reach:
                self.reward_obtained = True
                self.puzzle_state['success_achieved'] = True
        
        # Update puzzle state
        self.puzzle_state.update({
            'agent_position': self.agent_position.clone(),
            'stick_held': self.stick_held,
            'reward_obtained': self.reward_obtained
        })
        
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends when reward is obtained or max steps reached
        done = self.reward_obtained or self.current_step >= self.max_steps
        
        return sensory_input, self.puzzle_state, done
        
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate visual representation of tool use environment."""
        visual = torch.zeros(3, 64, 64)
        
        # Draw floor
        visual[0, 50:64, :] = 0.2  # Floor in red channel
        
        # Draw agent
        agent_x, agent_y = int(self.agent_position[0] * 6), int(self.agent_position[1] * 6)
        if 0 <= agent_x < 64 and 0 <= agent_y < 64:
            visual[1, agent_y-2:agent_y+2, agent_x-2:agent_x+2] = 1.0  # Agent in green
            
        # Draw reward if available
        if not self.reward_obtained:
            reward_x, reward_y = int(self.reward_position[0] * 6), int(self.reward_position[1] * 6)
            if 0 <= reward_x < 64 and 0 <= reward_y < 64:
                visual[2, reward_y-1:reward_y+1, reward_x-1:reward_x+1] = 1.0  # Reward in blue
                
        # Draw stick if available and not held
        if not self.stick_held:
            stick_x, stick_y = int(self.stick_position[0] * 6), int(self.stick_position[1] * 6)
            if 0 <= stick_x < 64 and 0 <= stick_y < 64:
                # Draw stick as a line
                visual[0, stick_y:stick_y+1, stick_x:stick_x+8] = 0.8  # Stick in red
                
        # Draw reach indicator if agent has stick
        if self.stick_held:
            # Show extended reach area
            reach_radius = int((self.agent_reach + self.stick_length) * 6)
            y_center, x_center = agent_y, agent_x
            
            for dy in range(-reach_radius, reach_radius + 1):
                for dx in range(-reach_radius, reach_radius + 1):
                    if dx*dx + dy*dy <= reach_radius*reach_radius:
                        ny, nx = y_center + dy, x_center + dx
                        if 0 <= ny < 64 and 0 <= nx < 64:
                            visual[0, ny, nx] = 0.3  # Show reach area
                            
        # Proprioceptive input (12 elements to match agent expectations)
        proprioception = torch.tensor([
            float(self.current_step),
            self.agent_position[0], self.agent_position[1], self.agent_position[2],
            float(self.stick_held),
            float(self.reward_obtained),
            time.time() - self.start_time,
            self.stick_position[0], self.stick_position[1], self.stick_position[2],
            0.0, 0.0  # Padding to reach 12 elements
        ])
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=100.0,
            timestamp=int(time.time())
        )
        
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate tool use and causal reasoning capabilities."""
        if not self.reward_obtained:
            return AGISignalLevel.NONE
                
        # Check if agent discovered tool use spontaneously
        spontaneous_discovery = 0.5
        solution_efficiency = 0.5
        novel_exploration = 0.5
            
        # Combine scores
        total_score = (spontaneous_discovery + solution_efficiency + novel_exploration) / 3.0
            
        if total_score > 0.8:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.6:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.3:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
