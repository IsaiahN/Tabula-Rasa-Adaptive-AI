"""
Puzzle 1: Hidden Cause (Baby Physics)

Tests the agent's ability to learn causality from invisible variables.
The ramp surface randomly toggles between slippery/sticky every 10 seconds,
affecting whether the ball lands in the target box.
"""

import torch
import numpy as np
import time
import math
from typing import Dict, List, Any, Tuple

from puzzle_base import BasePuzzleEnvironment, AGISignalLevel
from core.data_models import SensoryInput


class HiddenCausePuzzle(BasePuzzleEnvironment):
    """
    Ramp physics puzzle with hidden periodic variable.
    
    The agent must learn to predict ball success based on timing,
    discovering the hidden 10-second cycle of surface properties.
    """
    
    def __init__(self, max_steps: int = 500):
        super().__init__("Hidden Cause (Baby Physics)", max_steps)
        
        # Puzzle parameters
        self.ramp_angle = 30.0  # degrees
        self.ball_mass = 1.0
        self.surface_cycle_time = 10.0  # seconds
        self.sticky_friction = 0.8  # High friction
        self.slippery_friction = 0.1  # Low friction
        
        # Environment state
        self.ramp_position = torch.tensor([5.0, 5.0, 2.0])
        self.box_position = torch.tensor([8.0, 5.0, 0.0])
        self.ball_start_position = torch.tensor([5.0, 5.0, 3.0])
        
        # Hidden state tracking
        self.cycle_start_time = None
        self.current_surface_type = "sticky"  # Start with sticky
        self.surface_switches = []
        
        # Agent learning tracking
        self.attempt_history = []
        self.timing_patterns = []
        self.prediction_accuracy = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        self.cycle_start_time = self.start_time
        
        # Reset tracking
        self.attempt_history.clear()
        self.timing_patterns.clear()
        self.prediction_accuracy.clear()
        self.surface_switches.clear()
        
        # Initial state
        self.puzzle_state = {
            'ball_position': self.ball_start_position.clone(),
            'ball_velocity': torch.zeros(3),
            'surface_type': self.current_surface_type,
            'last_attempt_time': 0,
            'attempts_made': 0
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the puzzle."""
        self.current_step += 1
        current_time = time.time()
        
        # Update hidden surface state based on cycle
        self._update_surface_state(current_time)
        
        # Check if agent is attempting to release the ball
        release_ball = action[3] > 0.5  # Action dimension 3 = release ball
        
        step_result = {
            'ball_released': False,
            'ball_landed_in_box': False,
            'surface_type_revealed': False,
            'success': False,
            'prediction_made': False,
            'timing_behavior': None
        }
        
        if release_ball:
            # Agent is making an attempt
            attempt_time = current_time - self.start_time
            
            # Record timing behavior
            timing_behavior = self._analyze_timing_behavior(attempt_time)
            step_result['timing_behavior'] = timing_behavior
            
            # Simulate ball physics
            success = self._simulate_ball_physics()
            
            # Record attempt
            attempt_data = {
                'time': attempt_time,
                'surface_type': self.current_surface_type,
                'success': success,
                'predicted_success': action[4] > 0.5,  # Action dimension 4 = prediction
                'timing_pattern': timing_behavior
            }
            self.attempt_history.append(attempt_data)
            
            # Update puzzle state
            self.puzzle_state['attempts_made'] += 1
            self.puzzle_state['last_attempt_time'] = attempt_time
            
            step_result.update({
                'ball_released': True,
                'ball_landed_in_box': success,
                'success': success,
                'prediction_made': True,
                'surface_type_revealed': True  # Agent can observe result
            })
            
            # Record learning events
            if success:
                self.record_learning_event("successful_attempt", attempt_data)
            else:
                self.record_learning_event("failed_attempt", attempt_data)
                
            # Check prediction accuracy
            prediction_correct = (action[4] > 0.5) == success
            self.prediction_accuracy.append(prediction_correct)
            
            if prediction_correct:
                self.record_behavior("correct_prediction", attempt_data)
            else:
                # Record surprise if prediction was wrong
                self.record_surprise(
                    expected=action[4] > 0.5,
                    observed=success,
                    surprise_level=abs(float(action[4]) - (1.0 if success else 0.0))
                )
                
            # Reset ball for next attempt
            self.puzzle_state['ball_position'] = self.ball_start_position.clone()
            self.puzzle_state['ball_velocity'] = torch.zeros(3)
            
        # Check for hypothesis testing behavior
        self._detect_hypothesis_testing()
        
        # Generate new sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends after max steps or sufficient attempts
        done = (self.current_step >= self.max_steps or 
                self.puzzle_state['attempts_made'] >= 20)
        
        return sensory_input, step_result, done
        
    def _update_surface_state(self, current_time: float):
        """Update the hidden surface state based on time cycle."""
        time_in_cycle = (current_time - self.cycle_start_time) % self.surface_cycle_time
        
        # Switch every 10 seconds
        new_surface_type = "slippery" if time_in_cycle < 5.0 else "sticky"
        
        if new_surface_type != self.current_surface_type:
            self.surface_switches.append({
                'time': current_time - self.start_time,
                'from': self.current_surface_type,
                'to': new_surface_type
            })
            self.current_surface_type = new_surface_type
            self.puzzle_state['surface_type'] = new_surface_type
            
    def _simulate_ball_physics(self) -> bool:
        """Simulate ball rolling down ramp and check if it lands in box."""
        # Get current friction coefficient
        friction = self.sticky_friction if self.current_surface_type == "sticky" else self.slippery_friction
        
        # Simple physics simulation
        gravity = 9.81
        ramp_angle_rad = math.radians(self.ramp_angle)
        
        # Calculate acceleration down ramp
        acceleration = gravity * (math.sin(ramp_angle_rad) - friction * math.cos(ramp_angle_rad))
        
        # Calculate final velocity and distance
        ramp_length = 3.0
        time_on_ramp = math.sqrt(2 * ramp_length / max(acceleration, 0.1))
        final_velocity = acceleration * time_on_ramp
        
        # Calculate projectile motion
        horizontal_distance = final_velocity * 0.5  # Simplified
        
        # Add some randomness
        horizontal_distance += np.random.normal(0, 0.2)
        
        # Check if ball lands in box (box is 3 units away horizontally)
        target_distance = 3.0
        success = abs(horizontal_distance - target_distance) < 0.5
        
        return success
        
    def _analyze_timing_behavior(self, attempt_time: float) -> str:
        """Analyze the timing pattern of the agent's attempt."""
        if len(self.attempt_history) < 2:
            return "initial_attempt"
            
        # Check if agent is timing attempts to the cycle
        time_since_last_switch = attempt_time % self.surface_cycle_time
        
        if time_since_last_switch < 1.0:
            return "just_after_switch"
        elif time_since_last_switch > 4.0 and time_since_last_switch < 6.0:
            return "mid_cycle"
        elif time_since_last_switch > 9.0:
            return "just_before_switch"
        else:
            return "random_timing"
            
    def _detect_hypothesis_testing(self):
        """Detect if agent is systematically testing hypotheses about timing."""
        if len(self.attempt_history) < 5:
            return
            
        recent_attempts = self.attempt_history[-5:]
        
        # Check for systematic timing patterns
        timing_intervals = []
        for i in range(1, len(recent_attempts)):
            interval = recent_attempts[i]['time'] - recent_attempts[i-1]['time']
            timing_intervals.append(interval)
            
        # Look for regular intervals (hypothesis: agent testing periodic timing)
        if len(timing_intervals) >= 3:
            avg_interval = np.mean(timing_intervals)
            interval_std = np.std(timing_intervals)
            
            # Regular timing suggests hypothesis testing
            if interval_std < 2.0 and 8.0 < avg_interval < 12.0:
                self.record_hypothesis_test(
                    hypothesis="surface_changes_periodically",
                    test_action="regular_timing_attempts",
                    result=True
                )
                self.record_behavior("systematic_timing", {
                    'average_interval': avg_interval,
                    'consistency': 1.0 - (interval_std / avg_interval)
                })
                
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate sensory input showing current puzzle state."""
        # Create visual representation
        visual = torch.zeros(3, 64, 64)  # RGB image
        
        # Draw ramp (simplified representation)
        visual[0, 20:30, 20:40] = 0.8  # Ramp in red channel
        
        # Draw ball
        ball_pos = self.puzzle_state['ball_position']
        ball_x, ball_y = int(ball_pos[0] * 3), int(ball_pos[1] * 3)
        if 0 <= ball_x < 64 and 0 <= ball_y < 64:
            visual[1, ball_y-2:ball_y+2, ball_x-2:ball_x+2] = 1.0  # Ball in green
            
        # Draw target box
        box_x, box_y = int(self.box_position[0] * 3), int(self.box_position[1] * 3)
        if 0 <= box_x < 64 and 0 <= box_y < 64:
            visual[2, box_y-3:box_y+3, box_x-3:box_x+3] = 1.0  # Box in blue
            
        # Proprioceptive input (12 elements to match agent expectations)
        proprioception = torch.tensor([
            float(self.puzzle_state['attempts_made']),
            time.time() - self.start_time,  # Time elapsed
            float(len(self.attempt_history)),
            float(sum(1 for a in self.attempt_history if a['success'])),  # Success count
            float(self.current_step),
            float(self.current_surface_type),  # Surface type indicator
            ball_pos[0], ball_pos[1], ball_pos[2],  # Ball position
            0.0, 0.0, 0.0  # Padding to reach 12 elements
        ])
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=100.0,  # Not relevant for this puzzle
            timestamp=int(time.time())
        )
        
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate AGI capability level based on agent behavior."""
        if len(self.attempt_history) < 5:
            return AGISignalLevel.NONE
            
        # Check for timing awareness
        timing_awareness_score = self._evaluate_timing_awareness()
        
        # Check for prediction improvement
        prediction_improvement_score = self._evaluate_prediction_improvement()
        
        # Check for hypothesis testing
        hypothesis_testing_score = self._evaluate_hypothesis_testing()
        
        # Combine scores
        total_score = (timing_awareness_score + prediction_improvement_score + hypothesis_testing_score) / 3.0
        
        if total_score > 0.8:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.6:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.3:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
            
    def _evaluate_timing_awareness(self) -> float:
        """Evaluate if agent shows awareness of timing patterns."""
        if len(self.attempt_history) < 8:
            return 0.0
            
        # Check if agent attempts correlate with surface state
        sticky_attempts = [a for a in self.attempt_history if a['surface_type'] == 'sticky']
        slippery_attempts = [a for a in self.attempt_history if a['surface_type'] == 'slippery']
        
        if len(sticky_attempts) == 0 or len(slippery_attempts) == 0:
            return 0.0
            
        # Success rates on different surfaces
        sticky_success_rate = sum(1 for a in sticky_attempts if a['success']) / len(sticky_attempts)
        slippery_success_rate = sum(1 for a in slippery_attempts if a['success']) / len(slippery_attempts)
        
        # Agent should learn sticky surface is better
        surface_discrimination = max(0, sticky_success_rate - slippery_success_rate)
        
        # Check for timing patterns in recent attempts
        recent_attempts = self.attempt_history[-8:]
        timing_regularity = self._calculate_timing_regularity(recent_attempts)
        
        return (surface_discrimination + timing_regularity) / 2.0
        
    def _evaluate_prediction_improvement(self) -> float:
        """Evaluate if agent's predictions improve over time."""
        if len(self.prediction_accuracy) < 10:
            return 0.0
            
        # Compare early vs late prediction accuracy
        early_accuracy = np.mean(self.prediction_accuracy[:5])
        late_accuracy = np.mean(self.prediction_accuracy[-5:])
        
        improvement = max(0, late_accuracy - early_accuracy)
        return min(improvement * 2.0, 1.0)  # Scale to 0-1
        
    def _evaluate_hypothesis_testing(self) -> float:
        """Evaluate systematic hypothesis testing behavior."""
        # Look for evidence of systematic testing
        hypothesis_tests = len(self.hypothesis_tests)
        systematic_behaviors = len([b for b in self.behavior_history if b['type'] == 'systematic_timing'])
        
        if hypothesis_tests > 0 or systematic_behaviors > 0:
            return min((hypothesis_tests + systematic_behaviors) / 3.0, 1.0)
        else:
            return 0.0
            
    def _calculate_timing_regularity(self, attempts: List[Dict]) -> float:
        """Calculate how regular the timing intervals are."""
        if len(attempts) < 3:
            return 0.0
            
        intervals = []
        for i in range(1, len(attempts)):
            interval = attempts[i]['time'] - attempts[i-1]['time']
            intervals.append(interval)
            
        if len(intervals) < 2:
            return 0.0
            
        # Regular intervals suggest timing awareness
        avg_interval = np.mean(intervals)
        interval_std = np.std(intervals)
        
        if avg_interval == 0:
            return 0.0
            
        regularity = 1.0 - min(interval_std / avg_interval, 1.0)
        
        # Bonus for intervals near the cycle time
        cycle_alignment = 1.0 - min(abs(avg_interval - self.surface_cycle_time) / self.surface_cycle_time, 1.0)
        
        return (regularity + cycle_alignment) / 2.0
