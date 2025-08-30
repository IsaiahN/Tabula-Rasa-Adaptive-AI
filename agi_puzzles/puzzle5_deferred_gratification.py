"""
Puzzle 5: Deferred Gratification (Marshmallow Test)

Tests self-control and temporal planning by requiring agents to wait
for a larger reward instead of taking an immediate smaller one.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Tuple

from puzzle_base import BasePuzzleEnvironment, AGISignalLevel
from core.data_models import SensoryInput


class DeferredGratificationPuzzle(BasePuzzleEnvironment):
    """
    Deferred gratification test requiring self-control and planning.
    
    Agent can take small immediate reward or wait for larger delayed reward.
    Tests ability to resist temptation and plan over time horizons.
    """
    
    def __init__(self, max_steps: int = 600):
        super().__init__("Deferred Gratification (Marshmallow)", max_steps)
        
        # Reward parameters
        self.small_reward_value = 5.0
        self.large_reward_value = 15.0
        self.wait_time_required = 20.0  # seconds
        
        # Environment state
        self.small_reward_position = torch.tensor([4.0, 5.0, 0.0])
        self.large_reward_spawn_position = torch.tensor([8.0, 5.0, 0.0])
        self.agent_position = torch.tensor([2.0, 5.0, 0.0])
        
        # Trial state
        self.trial_start_time = None
        self.small_reward_taken = False
        self.large_reward_available = False
        self.waiting_period_active = False
        
        # Learning tracking
        self.choice_history = []
        self.waiting_behaviors = []
        self.temptation_resistance = []
        self.boredom_responses = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        self.trial_start_time = self.start_time
        
        # Reset trial state
        self.small_reward_taken = False
        self.large_reward_available = False
        self.waiting_period_active = True
        
        # Clear tracking
        self.choice_history.clear()
        self.waiting_behaviors.clear()
        self.temptation_resistance.clear()
        self.boredom_responses.clear()
        
        self.puzzle_state = {
            'trial_number': 0,
            'total_energy_gained': 0.0,
            'successful_waits': 0,
            'immediate_takes': 0,
            'current_wait_time': 0.0,
            'max_wait_achieved': 0.0
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the deferred gratification puzzle."""
        self.current_step += 1
        current_time = time.time()
        
        step_result = {
            'moved': False,
            'took_small_reward': False,
            'took_large_reward': False,
            'large_reward_spawned': False,
            'trial_completed': False,
            'waiting_behavior': None,
            'temptation_resisted': False
        }
        
        # Update wait time
        if self.waiting_period_active:
            wait_time = current_time - self.trial_start_time
            self.puzzle_state['current_wait_time'] = wait_time
            self.puzzle_state['max_wait_achieved'] = max(
                self.puzzle_state['max_wait_achieved'], wait_time
            )
            
        # Movement
        movement = action[:2] * 0.3  # Slower movement to encourage waiting
        new_position = self.agent_position + torch.cat([movement, torch.zeros(1)])
        new_position = torch.clamp(new_position, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 0.0]))
        
        if torch.norm(new_position - self.agent_position) > 0.05:
            self.agent_position = new_position
            step_result['moved'] = True
            
            # Record movement during waiting as potential boredom/exploration
            if self.waiting_period_active:
                self._record_waiting_behavior("movement", {
                    'position': new_position.clone(),
                    'wait_time': self.puzzle_state['current_wait_time']
                })
                
        # Check for small reward interaction
        if action[3] > 0.5 and not self.small_reward_taken:  # Action 3 = take small reward
            small_reward_distance = torch.norm(self.agent_position[:2] - self.small_reward_position[:2])
            
            if small_reward_distance < 1.0:
                # Agent takes immediate small reward
                self.small_reward_taken = True
                self.waiting_period_active = False
                self.puzzle_state['total_energy_gained'] += self.small_reward_value
                self.puzzle_state['immediate_takes'] += 1
                
                step_result['took_small_reward'] = True
                step_result['trial_completed'] = True
                
                # Record choice
                choice_data = {
                    'trial': self.puzzle_state['trial_number'],
                    'choice': 'immediate',
                    'reward_value': self.small_reward_value,
                    'wait_time': self.puzzle_state['current_wait_time'],
                    'step': self.current_step
                }
                self.choice_history.append(choice_data)
                
                self.record_learning_event("immediate_gratification", choice_data)
                
            else:
                # Agent tried to take reward but too far away
                step_result['temptation_resisted'] = True
                self._record_temptation_resistance(small_reward_distance)
                
        # Check if large reward should spawn
        if (self.waiting_period_active and 
            not self.large_reward_available and 
            self.puzzle_state['current_wait_time'] >= self.wait_time_required):
            
            self.large_reward_available = True
            step_result['large_reward_spawned'] = True
            
            self.record_behavior("large_reward_spawned", {
                'wait_time_achieved': self.puzzle_state['current_wait_time']
            })
            
        # Check for large reward interaction
        if action[4] > 0.5 and self.large_reward_available:  # Action 4 = take large reward
            large_reward_distance = torch.norm(self.agent_position[:2] - self.large_reward_spawn_position[:2])
            
            if large_reward_distance < 1.0:
                # Agent successfully waited and takes large reward
                self.waiting_period_active = False
                self.puzzle_state['total_energy_gained'] += self.large_reward_value
                self.puzzle_state['successful_waits'] += 1
                
                step_result['took_large_reward'] = True
                step_result['trial_completed'] = True
                
                # Record choice
                choice_data = {
                    'trial': self.puzzle_state['trial_number'],
                    'choice': 'delayed',
                    'reward_value': self.large_reward_value,
                    'wait_time': self.puzzle_state['current_wait_time'],
                    'step': self.current_step
                }
                self.choice_history.append(choice_data)
                
                self.record_learning_event("deferred_gratification_success", choice_data)
                
        # Detect boredom and exploration behaviors
        if self.waiting_period_active:
            self._detect_boredom_behaviors(action)
            
        # Start new trial if current one completed
        if step_result['trial_completed']:
            self._start_new_trial()
            
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends after several trials or max steps
        done = (self.puzzle_state['trial_number'] >= 8 or 
                self.current_step >= self.max_steps)
        
        return sensory_input, step_result, done
        
    def _record_waiting_behavior(self, behavior_type: str, details: Dict):
        """Record behavior during waiting period."""
        waiting_behavior = {
            'type': behavior_type,
            'details': details,
            'step': self.current_step,
            'wait_time': self.puzzle_state['current_wait_time']
        }
        self.waiting_behaviors.append(waiting_behavior)
        
    def _record_temptation_resistance(self, distance_to_temptation: float):
        """Record instance of resisting temptation."""
        resistance_data = {
            'step': self.current_step,
            'wait_time': self.puzzle_state['current_wait_time'],
            'distance_to_temptation': float(distance_to_temptation),
            'resistance_strength': min(distance_to_temptation / 2.0, 1.0)
        }
        self.temptation_resistance.append(resistance_data)
        
        self.record_behavior("temptation_resistance", resistance_data)
        
    def _detect_boredom_behaviors(self, action: torch.Tensor):
        """Detect behaviors indicating boredom during waiting."""
        # High activity in exploration actions might indicate boredom
        exploration_activity = float(torch.mean(action[5:]))  # Last action dimensions
        
        if exploration_activity > 0.6:
            boredom_data = {
                'step': self.current_step,
                'wait_time': self.puzzle_state['current_wait_time'],
                'exploration_level': exploration_activity,
                'position': self.agent_position.clone()
            }
            self.boredom_responses.append(boredom_data)
            
            # Record as waiting behavior
            self._record_waiting_behavior("boredom_exploration", boredom_data)
            
    def _start_new_trial(self):
        """Start a new trial."""
        self.trial_start_time = time.time()
        self.small_reward_taken = False
        self.large_reward_available = False
        self.waiting_period_active = True
        self.puzzle_state['trial_number'] += 1
        self.puzzle_state['current_wait_time'] = 0.0
        
        # Reset agent position
        self.agent_position = torch.tensor([2.0, 5.0, 0.0])
        
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate visual representation of gratification test."""
        visual = torch.zeros(3, 64, 64)
        
        # Draw floor
        visual[0, 50:64, :] = 0.2
        
        # Draw agent
        agent_x, agent_y = int(self.agent_position[0] * 6), int(self.agent_position[1] * 6)
        if 0 <= agent_x < 64 and 0 <= agent_y < 64:
            visual[1, agent_y-2:agent_y+2, agent_x-2:agent_x+2] = 1.0
            
        # Draw small reward if available
        if not self.small_reward_taken:
            small_x, small_y = int(self.small_reward_position[0] * 6), int(self.small_reward_position[1] * 6)
            if 0 <= small_x < 64 and 0 <= small_y < 64:
                visual[2, small_y-1:small_y+1, small_x-1:small_x+1] = 0.6  # Small reward
                
        # Draw large reward if available
        if self.large_reward_available:
            large_x, large_y = int(self.large_reward_spawn_position[0] * 6), int(self.large_reward_spawn_position[1] * 6)
            if 0 <= large_x < 64 and 0 <= large_y < 64:
                visual[2, large_y-2:large_y+2, large_x-2:large_x+2] = 1.0  # Large reward
                
        # Draw wait time indicator
        if self.waiting_period_active:
            wait_progress = min(self.puzzle_state['current_wait_time'] / self.wait_time_required, 1.0)
            progress_width = int(wait_progress * 20)
            visual[0, 5:10, 5:5+progress_width] = 0.8  # Progress bar
            
        # Proprioceptive input (12 elements to match agent expectations)
        proprioception = torch.tensor([
            float(self.puzzle_state['trial_number']),
            float(self.puzzle_state['wait_time']),
            float(self.puzzle_state['max_wait_achieved']),
            float(self.current_step),
            time.time() - self.start_time,
            float(self.puzzle_state['reward_available']),
            float(self.puzzle_state['waiting']),
            self.agent_position[0], self.agent_position[1], self.agent_position[2],
            0.0, 0.0  # Padding to reach 12 elements
        ])
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=self.puzzle_state['total_energy_gained'],
            timestamp=int(time.time())
        )
        
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate self-control and temporal planning capabilities."""
        if len(self.choice_history) < 3:
            return AGISignalLevel.NONE
            
        # Calculate success rate for delayed gratification
        delayed_choices = sum(1 for choice in self.choice_history if choice['choice'] == 'delayed')
        success_rate = delayed_choices / len(self.choice_history)
        
        # Evaluate learning progression
        learning_progression = self._evaluate_learning_progression()
        
        # Evaluate waiting strategies
        waiting_strategy_score = self._evaluate_waiting_strategies()
        
        # Evaluate temptation resistance
        resistance_score = self._evaluate_temptation_resistance()
        
        # Combine scores
        total_score = (success_rate + learning_progression + waiting_strategy_score + resistance_score) / 4.0
        
        if total_score > 0.75 and success_rate > 0.6:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.5 and success_rate > 0.3:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.25:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
            
    def _evaluate_learning_progression(self) -> float:
        """Evaluate if agent learns to wait over time."""
        if len(self.choice_history) < 4:
            return 0.0
            
        # Compare early vs late choices
        early_choices = self.choice_history[:len(self.choice_history)//2]
        late_choices = self.choice_history[len(self.choice_history)//2:]
        
        early_delay_rate = sum(1 for c in early_choices if c['choice'] == 'delayed') / len(early_choices)
        late_delay_rate = sum(1 for c in late_choices if c['choice'] == 'delayed') / len(late_choices)
        
        improvement = max(0, late_delay_rate - early_delay_rate)
        return min(improvement * 2.0, 1.0)
        
    def _evaluate_waiting_strategies(self) -> float:
        """Evaluate sophistication of waiting strategies."""
        if len(self.waiting_behaviors) == 0:
            return 0.0
            
        # Look for diverse waiting behaviors (indicates active coping)
        behavior_types = set(b['type'] for b in self.waiting_behaviors)
        strategy_diversity = len(behavior_types) / 5.0  # Normalize by max expected types
        
        # Look for patterns in waiting behavior
        movement_behaviors = [b for b in self.waiting_behaviors if b['type'] == 'movement']
        if len(movement_behaviors) > 0:
            # Consistent movement patterns suggest deliberate distraction strategy
            avg_movement_time = np.mean([b['wait_time'] for b in movement_behaviors])
            if 5.0 < avg_movement_time < 15.0:  # Movement during middle of wait period
                strategy_diversity += 0.3
                
        return min(strategy_diversity, 1.0)
        
    def _evaluate_temptation_resistance(self) -> float:
        """Evaluate ability to resist temptation."""
        if len(self.temptation_resistance) == 0:
            return 0.5  # Neutral if no temptation events
            
        # Higher resistance strength is better
        avg_resistance = np.mean([r['resistance_strength'] for r in self.temptation_resistance])
        
        # More resistance events during longer waits is better
        long_wait_resistance = sum(
            1 for r in self.temptation_resistance 
            if r['wait_time'] > 10.0
        )
        
        resistance_score = (avg_resistance + min(long_wait_resistance / 3.0, 1.0)) / 2.0
        return resistance_score
