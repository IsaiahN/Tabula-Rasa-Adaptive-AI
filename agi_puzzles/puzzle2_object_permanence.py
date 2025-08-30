"""
Puzzle 2: Object Permanence (Peekaboo)

Tests the agent's understanding that objects continue to exist when not visible.
A block is sometimes hidden by a curtain vs actually removed.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Tuple

from puzzle_base import BasePuzzleEnvironment, AGISignalLevel
from core.data_models import SensoryInput


class ObjectPermanencePuzzle(BasePuzzleEnvironment):
    """
    Object permanence test with occlusion vs removal.
    
    Agent must learn to distinguish between temporary occlusion
    and actual object removal to predict object existence.
    """
    
    def __init__(self, max_steps: int = 300):
        super().__init__("Object Permanence (Peekaboo)", max_steps)
        
        # Environment setup
        self.block_position = torch.tensor([5.0, 5.0, 0.0])
        self.curtain_position = torch.tensor([5.0, 5.0, 1.0])
        
        # Puzzle state
        self.block_exists = True
        self.curtain_down = False
        self.curtain_action_type = "hide"  # "hide" or "remove"
        
        # Timing
        self.curtain_down_time = 0
        self.curtain_cycle_duration = 8.0  # seconds
        self.last_curtain_action = 0
        
        # Learning tracking
        self.prediction_history = []
        self.surprise_responses = []
        self.occlusion_vs_removal_trials = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        
        # Reset state
        self.block_exists = True
        self.curtain_down = False
        self.curtain_action_type = "hide"
        self.curtain_down_time = 0
        self.last_curtain_action = 0
        
        # Clear tracking
        self.prediction_history.clear()
        self.surprise_responses.clear()
        self.occlusion_vs_removal_trials.clear()
        
        self.puzzle_state = {
            'block_visible': True,
            'block_exists': True,
            'curtain_down': False,
            'trials_completed': 0,
            'correct_predictions': 0
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the puzzle."""
        self.current_step += 1
        current_time = time.time()
        
        step_result = {
            'curtain_action': False,
            'block_revealed': False,
            'prediction_made': False,
            'prediction_correct': False,
            'surprise_detected': False,
            'trial_completed': False
        }
        
        # Check if it's time for curtain action
        if not self.curtain_down and (current_time - self.last_curtain_action) > self.curtain_cycle_duration:
            # Lower curtain and decide action type
            self.curtain_down = True
            self.curtain_down_time = current_time
            self.curtain_action_type = np.random.choice(["hide", "remove"], p=[0.7, 0.3])
            
            if self.curtain_action_type == "remove":
                self.block_exists = False
                
            self.puzzle_state['curtain_down'] = True
            self.puzzle_state['block_visible'] = False
            
            step_result['curtain_action'] = True
            
            self.record_behavior("curtain_lowered", {
                'action_type': self.curtain_action_type,
                'block_exists': self.block_exists
            })
            
        # Check if curtain should be raised
        elif self.curtain_down and (current_time - self.curtain_down_time) > 3.0:
            # Raise curtain and check agent prediction
            agent_prediction = action[3] > 0.5  # Action dimension 3 = predict block exists
            
            # Record prediction
            prediction_data = {
                'predicted_exists': agent_prediction,
                'actually_exists': self.block_exists,
                'action_type': self.curtain_action_type,
                'time': current_time - self.start_time
            }
            self.prediction_history.append(prediction_data)
            
            # Check if prediction is correct
            prediction_correct = agent_prediction == self.block_exists
            
            if prediction_correct:
                self.puzzle_state['correct_predictions'] += 1
                
            # Measure surprise response
            surprise_level = self._measure_surprise_response(action, prediction_data)
            
            # Record trial completion
            trial_data = {
                'trial_number': self.puzzle_state['trials_completed'],
                'curtain_action_type': self.curtain_action_type,
                'prediction_correct': prediction_correct,
                'surprise_level': surprise_level,
                'block_exists': self.block_exists
            }
            self.occlusion_vs_removal_trials.append(trial_data)
            
            # Reset for next trial
            self.curtain_down = False
            self.puzzle_state['curtain_down'] = False
            self.puzzle_state['block_visible'] = self.block_exists
            self.puzzle_state['trials_completed'] += 1
            self.last_curtain_action = current_time
            
            # Reset block for next trial
            if not self.block_exists:
                self.block_exists = True
                self.puzzle_state['block_exists'] = True
                
            step_result.update({
                'block_revealed': True,
                'prediction_made': True,
                'prediction_correct': prediction_correct,
                'surprise_detected': surprise_level > 0.5,
                'trial_completed': True
            })
            
            # Record learning events
            if prediction_correct:
                self.record_learning_event("correct_prediction", prediction_data)
            else:
                self.record_learning_event("incorrect_prediction", prediction_data)
                
            # Record surprise if appropriate
            if surprise_level > 0.3:
                self.record_surprise(
                    expected=agent_prediction,
                    observed=self.block_exists,
                    surprise_level=surprise_level
                )
                
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends after sufficient trials
        done = self.puzzle_state['trials_completed'] >= 15 or self.current_step >= self.max_steps
        
        return sensory_input, step_result, done
        
    def _measure_surprise_response(self, action: torch.Tensor, prediction_data: Dict) -> float:
        """Measure agent's surprise response to block revelation."""
        # Analyze action vector for surprise indicators
        # Higher values in action dimensions 4-5 could indicate surprise/attention
        attention_response = float(torch.mean(action[4:6]))
        
        # Calculate expected vs observed mismatch
        prediction_error = abs(float(action[3]) - (1.0 if self.block_exists else 0.0))
        
        # AGI signal: Should be more surprised by removal than occlusion
        expected_surprise = 0.8 if self.curtain_action_type == "remove" else 0.2
        
        # Record surprise response
        surprise_data = {
            'attention_response': attention_response,
            'prediction_error': prediction_error,
            'action_type': self.curtain_action_type,
            'expected_surprise': expected_surprise
        }
        self.surprise_responses.append(surprise_data)
        
        # Return combined surprise level
        return (attention_response + prediction_error) / 2.0
        
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate visual representation of current puzzle state."""
        visual = torch.zeros(3, 64, 64)
        
        # Draw floor
        visual[0, 50:64, :] = 0.3  # Floor in red channel
        
        # Draw block if visible
        if self.puzzle_state['block_visible'] and self.block_exists:
            block_x, block_y = int(self.block_position[0] * 6), int(self.block_position[1] * 6)
            if 0 <= block_x < 64 and 0 <= block_y < 64:
                visual[1, block_y-3:block_y+3, block_x-3:block_x+3] = 1.0  # Block in green
                
        # Draw curtain if down
        if self.curtain_down:
            visual[2, 20:50, 25:35] = 0.8  # Curtain in blue channel
            
        # Proprioceptive input
        proprioception = torch.tensor([
            float(self.puzzle_state['trials_completed']),
            float(self.puzzle_state['correct_predictions']),
            float(self.curtain_down),
            time.time() - self.start_time,
            float(len(self.prediction_history))
        ])
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=100.0,
            timestamp=int(time.time())
        )
        
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate object permanence understanding."""
        if len(self.occlusion_vs_removal_trials) < 5:
            return AGISignalLevel.NONE
            
        # Analyze prediction accuracy for different trial types
        hide_trials = [t for t in self.occlusion_vs_removal_trials if t['curtain_action_type'] == 'hide']
        remove_trials = [t for t in self.occlusion_vs_removal_trials if t['curtain_action_type'] == 'remove']
        
        if len(hide_trials) < 2 or len(remove_trials) < 2:
            return AGISignalLevel.BASIC
            
        # Calculate accuracy for each trial type
        hide_accuracy = sum(1 for t in hide_trials if t['prediction_correct']) / len(hide_trials)
        remove_accuracy = sum(1 for t in remove_trials if t['prediction_correct']) / len(remove_trials)
        
        # AGI signal: Should predict block exists for hide, doesn't exist for remove
        discrimination_score = (hide_accuracy + remove_accuracy) / 2.0
        
        # Check surprise patterns
        surprise_discrimination = self._evaluate_surprise_discrimination()
        
        # Overall score
        total_score = (discrimination_score + surprise_discrimination) / 2.0
        
        if total_score > 0.8:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.6:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.4:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
            
    def _evaluate_surprise_discrimination(self) -> float:
        """Evaluate if agent shows appropriate surprise patterns."""
        if len(self.surprise_responses) < 5:
            return 0.0
            
        hide_surprises = [s for s in self.surprise_responses if s['action_type'] == 'hide']
        remove_surprises = [s for s in self.surprise_responses if s['action_type'] == 'remove']
        
        if len(hide_surprises) == 0 or len(remove_surprises) == 0:
            return 0.0
            
        avg_hide_surprise = np.mean([s['attention_response'] for s in hide_surprises])
        avg_remove_surprise = np.mean([s['attention_response'] for s in remove_surprises])
        
        # Should be more surprised by removal than hiding
        surprise_discrimination = max(0, avg_remove_surprise - avg_hide_surprise)
        return min(surprise_discrimination * 2.0, 1.0)
