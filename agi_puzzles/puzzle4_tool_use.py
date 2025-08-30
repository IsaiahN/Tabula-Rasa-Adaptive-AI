"""
Puzzle 4: Tool Use (The Stick)

Tests causal reasoning about object affordances and spontaneous tool use.
Agent must learn to use a stick to reach an out-of-reach reward.
"""

import torch
import numpy as np
import time
import math
from typing import Dict, List, Any, Tuple

from puzzle_base import BasePuzzleEnvironment, AGISignalLevel
from core.data_models import SensoryInput


class ToolUsePuzzle(BasePuzzleEnvironment):
    """
    Tool use puzzle requiring causal reasoning about object affordances.
    
    Agent must discover that the stick can extend reach to obtain
    an otherwise unreachable reward.
    """
    
    def __init__(self, max_steps: int = 200):
        super().__init__("Tool Use (The Stick)", max_steps)
        
        # Environment setup
        self.agent_position = torch.tensor([2.0, 5.0, 0.0])
        self.reward_position = torch.tensor([8.0, 5.0, 0.0])  # Out of reach
        self.stick_position = torch.tensor([3.5, 4.0, 0.0])  # Nearby stick
        
        # Object properties
        self.agent_reach = 1.5  # Agent's natural reach
        self.stick_length = 3.0  # Stick extends reach
        self.reward_obtained = False
        self.stick_held = False
        
        # Learning tracking
        self.reach_attempts = []
        self.tool_interactions = []
        self.novel_contexts = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        
        # Reset state
        self.reward_obtained = False
        self.stick_held = False
        self.agent_position = torch.tensor([2.0, 5.0, 0.0])
        
        # Clear tracking
        self.reach_attempts.clear()
        self.tool_interactions.clear()
        self.novel_contexts.clear()
        
        self.puzzle_state = {
            'reward_available': True,
            'stick_available': True,
            'agent_has_stick': False,
            'direct_reach_attempts': 0,
            'tool_use_attempts': 0,
            'success_achieved': False
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the tool use puzzle."""
        self.current_step += 1
        
        step_result = {
            'moved': False,
            'reached_for_reward': False,
            'picked_up_stick': False,
            'used_tool': False,
            'reward_obtained': False,
            'novel_tool_use': False
        }
        
        # Movement
        movement = action[:2] * 0.5  # Scale movement
        new_position = self.agent_position + torch.cat([movement, torch.zeros(1)])
        
        # Clamp to environment bounds
        new_position = torch.clamp(new_position, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 0.0]))
        
        if torch.norm(new_position - self.agent_position) > 0.1:
            self.agent_position = new_position
            step_result['moved'] = True
            
        # Check for stick pickup
        if not self.stick_held and action[3] > 0.5:  # Action 3 = grab/interact
            stick_distance = torch.norm(self.agent_position[:2] - self.stick_position[:2])
            if stick_distance < self.agent_reach:
                self.stick_held = True
                self.puzzle_state['agent_has_stick'] = True
                self.puzzle_state['stick_available'] = False
                step_result['picked_up_stick'] = True
                
                self.record_behavior("stick_pickup", {
                    'distance_to_stick': float(stick_distance),
                    'step': self.current_step
                })
                
                self.tool_interactions.append({
                    'type': 'pickup',
                    'step': self.current_step,
                    'success': True
                })
                
        # Check for reward reaching attempt
        if action[4] > 0.5:  # Action 4 = reach for reward
            reward_distance = torch.norm(self.agent_position[:2] - self.reward_position[:2])
            
            # Determine effective reach
            effective_reach = self.agent_reach
            if self.stick_held:
                effective_reach += self.stick_length
                step_result['used_tool'] = True
                self.puzzle_state['tool_use_attempts'] += 1
                
                self.tool_interactions.append({
                    'type': 'tool_use',
                    'step': self.current_step,
                    'distance_to_reward': float(reward_distance),
                    'effective_reach': effective_reach,
                    'success': reward_distance <= effective_reach
                })
            else:
                self.puzzle_state['direct_reach_attempts'] += 1
                
            step_result['reached_for_reward'] = True
            
            # Record reach attempt
            reach_data = {
                'step': self.current_step,
                'distance_to_reward': float(reward_distance),
                'using_tool': self.stick_held,
                'effective_reach': effective_reach,
                'success': reward_distance <= effective_reach
            }
            self.reach_attempts.append(reach_data)
            
            # Check if reward is obtained
            if reward_distance <= effective_reach and not self.reward_obtained:
                self.reward_obtained = True
                self.puzzle_state['success_achieved'] = True
                self.puzzle_state['reward_available'] = False
                step_result['reward_obtained'] = True
                
                self.record_learning_event("reward_obtained", reach_data)
                
                # Check if this was spontaneous tool use
                if self.stick_held and self.puzzle_state['direct_reach_attempts'] > 0:
                    self.record_behavior("spontaneous_tool_use", {
                        'direct_attempts_before_tool': self.puzzle_state['direct_reach_attempts'],
                        'discovery_step': self.current_step
                    })
                    
        # Test for novel tool use contexts
        self._test_novel_contexts(action)
        
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends when reward is obtained or max steps reached
        done = self.reward_obtained or self.current_step >= self.max_steps
        
        return sensory_input, step_result, done
        
    def _test_novel_contexts(self, action: torch.Tensor):
        """Test if agent tries to use stick in novel contexts."""
        if not self.stick_held or self.reward_obtained:
            return
            
        # Check if agent tries to use stick for other purposes
        # Action 5 = try alternative use
        if action[5] > 0.5:
            # Record novel context attempt
            novel_attempt = {
                'step': self.current_step,
                'context': 'alternative_use',
                'agent_position': self.agent_position.clone()
            }
            self.novel_contexts.append(novel_attempt)
            
            self.record_behavior("novel_tool_exploration", novel_attempt)
            
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
        spontaneous_discovery = self._evaluate_spontaneous_discovery()
            
        # Check efficiency of solution
        solution_efficiency = self._evaluate_solution_efficiency()
            
        # Check for novel context exploration
        novel_exploration = self._evaluate_novel_exploration()
            
        # Combine scores
        total_score = (spontaneous_discovery + solution_efficiency + novel_exploration) / 3.0
            
        if total_score > 0.8 and len(self.novel_contexts) > 0:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.6:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.3:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
            
    def _evaluate_spontaneous_discovery(self) -> float:
        """Evaluate if tool use was discovered spontaneously."""
        if not self.stick_held or not self.reward_obtained:
            return 0.0
                
        # Check if agent tried direct reach first
        direct_attempts_before_tool = 0
        tool_pickup_step = None
            
        for interaction in self.tool_interactions:
            if interaction['type'] == 'pickup':
                tool_pickup_step = interaction['step']
                break
                
        return 0.5  # Basic implementation
        
    def _evaluate_solution_efficiency(self) -> float:
        """Evaluate efficiency of tool use solution."""
        return 0.5  # Basic implementation
        
    def _evaluate_novel_exploration(self) -> float:
        """Evaluate novel context exploration."""
        return 0.5  # Basic implementation
        else:
            return 0.5  # Immediate tool use is less impressive but still good
            
    def _evaluate_solution_efficiency(self) -> float:
        """Evaluate efficiency of reaching the solution."""
        if not self.reward_obtained:
            return 0.0
            
        # Fewer steps to solution is better
        efficiency = max(0, 1.0 - (self.current_step / self.max_steps))
        
        # Fewer failed attempts is better
        failed_attempts = sum(1 for attempt in self.reach_attempts if not attempt['success'])
        attempt_efficiency = max(0, 1.0 - (failed_attempts / 10.0))
        
        return (efficiency + attempt_efficiency) / 2.0
        
    def _evaluate_novel_exploration(self) -> float:
        """Evaluate exploration of tool use in novel contexts."""
        if len(self.novel_contexts) == 0:
            return 0.0
            
        # More novel explorations indicate better understanding
        exploration_score = min(len(self.novel_contexts) / 3.0, 1.0)
        return exploration_score
