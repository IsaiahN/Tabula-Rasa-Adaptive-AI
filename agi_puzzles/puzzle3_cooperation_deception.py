"""
Puzzle 3: Cooperation & Deception (Food Game)

Tests theory of mind and trust mechanisms through multi-agent food sharing.
One agent can signal intentions, the other must learn to predict reliability.
"""

import torch
import numpy as np
import time
from typing import Dict, List, Any, Tuple

from puzzle_base import BasePuzzleEnvironment, AGISignalLevel
from core.data_models import SensoryInput


class CooperationDeceptionPuzzle(BasePuzzleEnvironment):
    """
    Two-agent food sharing puzzle with signaling and deception.
    
    Tests theory of mind by requiring agents to predict partner
    reliability based on signals and past behavior.
    """
    
    def __init__(self, max_steps: int = 400):
        super().__init__("Cooperation & Deception (Food Game)", max_steps)
        
        # Environment setup
        self.apple_position = torch.tensor([7.5, 5.0, 0.0])
        self.agent1_position = torch.tensor([3.0, 5.0, 0.0])  # Signaler
        self.agent2_position = torch.tensor([12.0, 5.0, 0.0])  # Observer
        
        # Game state
        self.current_signaler = "agent1"  # Who controls the apple
        self.signal_sent = None  # "share" or "keep"
        self.actual_intention = None  # True intention
        self.apple_available = True
        
        # Partner behavior modeling
        self.signaler_reliability = 0.7  # How often signaler is honest
        self.interaction_history = []
        self.trust_updates = []
        
        # Learning tracking
        self.prediction_accuracy = []
        self.trust_calibration = []
        self.strategy_adaptations = []
        
    def reset(self) -> SensoryInput:
        """Reset puzzle to initial state."""
        self.current_step = 0
        self.start_time = time.time()
        
        # Reset game state
        self.apple_available = True
        self.signal_sent = None
        self.actual_intention = None
        self.current_signaler = "agent1"
        
        # Clear tracking
        self.interaction_history.clear()
        self.trust_updates.clear()
        self.prediction_accuracy.clear()
        self.trust_calibration.clear()
        self.strategy_adaptations.clear()
        
        self.puzzle_state = {
            'round_number': 0,
            'agent1_energy': 100.0,
            'agent2_energy': 100.0,
            'total_apples_shared': 0,
            'total_apples_kept': 0,
            'current_phase': 'signaling'
        }
        
        return self._generate_sensory_input()
        
    def step(self, action: torch.Tensor) -> Tuple[SensoryInput, Dict[str, Any], bool]:
        """Execute one step in the cooperation/deception game."""
        self.current_step += 1
        
        step_result = {
            'signal_sent': False,
            'prediction_made': False,
            'apple_shared': False,
            'trust_updated': False,
            'strategy_changed': False,
            'round_completed': False
        }
        
        if self.puzzle_state['current_phase'] == 'signaling':
            # Signaler sends signal about intention
            signal_action = action[3] > 0.5  # Action 3 = signal "share"
            self.signal_sent = "share" if signal_action else "keep"
            
            # Determine actual intention (may differ from signal)
            honesty_roll = np.random.random()
            if honesty_roll < self.signaler_reliability:
                self.actual_intention = self.signal_sent  # Honest
            else:
                self.actual_intention = "keep" if self.signal_sent == "share" else "share"  # Deceptive
                
            step_result['signal_sent'] = True
            self.puzzle_state['current_phase'] = 'prediction'
            
            self.record_behavior("signal_sent", {
                'signal': self.signal_sent,
                'actual_intention': self.actual_intention,
                'is_honest': self.signal_sent == self.actual_intention
            })
            
        elif self.puzzle_state['current_phase'] == 'prediction':
            # Observer predicts signaler's actual behavior
            predicted_sharing = action[4] > 0.5  # Action 4 = predict sharing
            
            # Execute actual behavior
            will_actually_share = self.actual_intention == "share"
            
            # Update energies based on outcome
            if will_actually_share:
                # Apple is shared
                energy_gain = 10.0
                self.puzzle_state['agent1_energy'] += energy_gain / 2
                self.puzzle_state['agent2_energy'] += energy_gain / 2
                self.puzzle_state['total_apples_shared'] += 1
            else:
                # Apple is kept by signaler
                self.puzzle_state['agent1_energy'] += 10.0
                self.puzzle_state['total_apples_kept'] += 1
                
            # Record interaction
            interaction_data = {
                'round': self.puzzle_state['round_number'],
                'signal_sent': self.signal_sent,
                'actual_behavior': self.actual_intention,
                'observer_prediction': predicted_sharing,
                'prediction_correct': predicted_sharing == will_actually_share,
                'signaler_honest': self.signal_sent == self.actual_intention
            }
            self.interaction_history.append(interaction_data)
            
            # Update prediction accuracy
            prediction_correct = predicted_sharing == will_actually_share
            self.prediction_accuracy.append(prediction_correct)
            
            # Detect trust updates and strategy changes
            trust_change = self._detect_trust_update(interaction_data)
            strategy_change = self._detect_strategy_adaptation()
            
            step_result.update({
                'prediction_made': True,
                'apple_shared': will_actually_share,
                'trust_updated': trust_change,
                'strategy_changed': strategy_change,
                'round_completed': True
            })
            
            # Record learning events
            if prediction_correct:
                self.record_learning_event("correct_trust_prediction", interaction_data)
            else:
                self.record_learning_event("trust_prediction_error", interaction_data)
                
            # Prepare for next round
            self.puzzle_state['round_number'] += 1
            self.puzzle_state['current_phase'] = 'signaling'
            
            # Occasionally switch roles
            if self.puzzle_state['round_number'] % 5 == 0:
                self.current_signaler = "agent2" if self.current_signaler == "agent1" else "agent1"
                
        # Generate sensory input
        sensory_input = self._generate_sensory_input()
        
        # Episode ends after sufficient rounds
        done = (self.puzzle_state['round_number'] >= 20 or 
                self.current_step >= self.max_steps)
        
        return sensory_input, step_result, done
        
    def _detect_trust_update(self, interaction_data: Dict) -> bool:
        """Detect if agent is updating trust model of partner."""
        if len(self.interaction_history) < 3:
            return False
            
        # Look for patterns in recent interactions
        recent_interactions = self.interaction_history[-3:]
        
        # Check if agent's predictions are adapting to partner's reliability
        honest_signals = sum(1 for i in recent_interactions if i['signaler_honest'])
        dishonest_signals = len(recent_interactions) - honest_signals
        
        # If agent's recent predictions align with partner's honesty pattern
        if honest_signals > dishonest_signals:
            # Partner is mostly honest - agent should trust signals
            trust_aligned_predictions = sum(
                1 for i in recent_interactions 
                if i['observer_prediction'] == (i['signal_sent'] == "share")
            )
        else:
            # Partner is mostly dishonest - agent should distrust signals
            trust_aligned_predictions = sum(
                1 for i in recent_interactions 
                if i['observer_prediction'] != (i['signal_sent'] == "share")
            )
            
        trust_alignment = trust_aligned_predictions / len(recent_interactions)
        
        if trust_alignment > 0.7:
            self.trust_updates.append({
                'round': self.puzzle_state['round_number'],
                'trust_level': trust_alignment,
                'partner_honesty': honest_signals / len(recent_interactions)
            })
            return True
            
        return False
        
    def _detect_strategy_adaptation(self) -> bool:
        """Detect strategic adaptation to partner behavior."""
        if len(self.interaction_history) < 6:
            return False
            
        # Compare early vs recent prediction strategies
        early_interactions = self.interaction_history[:3]
        recent_interactions = self.interaction_history[-3:]
        
        early_trust_rate = sum(
            1 for i in early_interactions 
            if i['observer_prediction'] == (i['signal_sent'] == "share")
        ) / len(early_interactions)
        
        recent_trust_rate = sum(
            1 for i in recent_interactions 
            if i['observer_prediction'] == (i['signal_sent'] == "share")
        ) / len(recent_interactions)
        
        # Significant change in trust behavior indicates adaptation
        strategy_change = abs(recent_trust_rate - early_trust_rate) > 0.4
        
        if strategy_change:
            self.strategy_adaptations.append({
                'round': self.puzzle_state['round_number'],
                'early_trust_rate': early_trust_rate,
                'recent_trust_rate': recent_trust_rate,
                'adaptation_magnitude': abs(recent_trust_rate - early_trust_rate)
            })
            
            self.record_behavior("strategy_adaptation", {
                'trust_change': recent_trust_rate - early_trust_rate,
                'adaptation_round': self.puzzle_state['round_number']
            })
            
        return strategy_change
        
    def _generate_sensory_input(self) -> SensoryInput:
        """Generate visual representation of cooperation game."""
        visual = torch.zeros(3, 64, 64)
        
        # Draw agents
        agent1_x, agent1_y = int(self.agent1_position[0] * 4), int(self.agent1_position[1] * 4)
        agent2_x, agent2_y = int(self.agent2_position[0] * 4), int(self.agent2_position[1] * 4)
        
        if 0 <= agent1_x < 64 and 0 <= agent1_y < 64:
            visual[0, agent1_y-2:agent1_y+2, agent1_x-2:agent1_x+2] = 1.0  # Agent 1 in red
            
        if 0 <= agent2_x < 64 and 0 <= agent2_y < 64:
            visual[1, agent2_y-2:agent2_y+2, agent2_x-2:agent2_x+2] = 1.0  # Agent 2 in green
            
        # Draw apple if available
        if self.apple_available:
            apple_x, apple_y = int(self.apple_position[0] * 4), int(self.apple_position[1] * 4)
            if 0 <= apple_x < 64 and 0 <= apple_y < 64:
                visual[2, apple_y-1:apple_y+1, apple_x-1:apple_x+1] = 1.0  # Apple in blue
                
        # Draw signal indicator if signal was sent
        if self.signal_sent:
            signal_value = 0.8 if self.signal_sent == "share" else 0.3
            visual[0, 5:15, 5:15] = signal_value  # Signal indicator
            
        # Proprioceptive input
        proprioception = torch.tensor([
            float(self.puzzle_state['round_number']),
            self.puzzle_state['agent1_energy'] / 100.0,
            self.puzzle_state['agent2_energy'] / 100.0,
            float(self.signal_sent == "share") if self.signal_sent else 0.5,
            float(len(self.interaction_history)),
            float(sum(1 for i in self.interaction_history if i['prediction_correct'])) / max(len(self.interaction_history), 1)
        ])
        
        return SensoryInput(
            visual=visual,
            proprioception=proprioception,
            energy_level=self.puzzle_state['agent2_energy'],
            timestamp=int(time.time())
        )
        
    def evaluate_agi_signals(self) -> AGISignalLevel:
        """Evaluate theory of mind and trust learning capabilities."""
        if len(self.interaction_history) < 8:
            return AGISignalLevel.NONE
            
        # Evaluate prediction accuracy improvement
        early_accuracy = np.mean(self.prediction_accuracy[:4]) if len(self.prediction_accuracy) >= 4 else 0
        late_accuracy = np.mean(self.prediction_accuracy[-4:]) if len(self.prediction_accuracy) >= 4 else 0
        learning_improvement = max(0, late_accuracy - early_accuracy)
        
        # Evaluate trust calibration
        trust_calibration_score = self._evaluate_trust_calibration()
        
        # Evaluate strategy adaptation
        adaptation_score = len(self.strategy_adaptations) / max(self.puzzle_state['round_number'], 1)
        adaptation_score = min(adaptation_score, 1.0)
        
        # Overall AGI score
        total_score = (learning_improvement + trust_calibration_score + adaptation_score) / 3.0
        
        if total_score > 0.7 and len(self.strategy_adaptations) > 0:
            return AGISignalLevel.ADVANCED
        elif total_score > 0.5:
            return AGISignalLevel.INTERMEDIATE
        elif total_score > 0.2:
            return AGISignalLevel.BASIC
        else:
            return AGISignalLevel.NONE
            
    def _evaluate_trust_calibration(self) -> float:
        """Evaluate how well agent calibrates trust to partner reliability."""
        if len(self.interaction_history) < 6:
            return 0.0
            
        # Calculate actual partner honesty rate
        honest_interactions = sum(1 for i in self.interaction_history if i['signaler_honest'])
        actual_honesty_rate = honest_interactions / len(self.interaction_history)
        
        # Calculate agent's implicit trust rate (how often they trust signals)
        trust_rate = sum(
            1 for i in self.interaction_history 
            if i['observer_prediction'] == (i['signal_sent'] == "share")
        ) / len(self.interaction_history)
        
        # Good calibration means trust rate matches actual honesty rate
        calibration_error = abs(trust_rate - actual_honesty_rate)
        calibration_score = max(0, 1.0 - calibration_error * 2.0)
        
        return calibration_score
