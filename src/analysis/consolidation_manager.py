"""Manages consolidation of game experience during rest periods."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
import json

from src.analysis.game_state_replay import GameStateReplay, GameReplayInsight
from src.analysis.frame_analysis import analyze_frame_sequence

logger = logging.getLogger(__name__)

@dataclass
class ConsolidationResult:
    """Results from a consolidation period."""
    new_patterns_discovered: List[Dict[str, Any]]
    confirmed_hypotheses: List[Dict[str, Any]]
    rejected_hypotheses: List[Dict[str, Any]]
    updated_action_values: Dict[str, float]
    movement_optimizations: List[str]
    next_state_predictions: List[Tuple[np.ndarray, float]]

class ConsolidationManager:
    """Manages memory consolidation during rest periods."""
    
    def __init__(self, 
                 game_state_replay: GameStateReplay,
                 db_path: str):
        """Initialize consolidation manager.
        
        Args:
            game_state_replay: GameStateReplay instance for analyzing moves
            db_path: Path to the SQLite database
        """
        self.replay = game_state_replay
        self.db_path = db_path
        self.pending_hypotheses: List[Dict[str, Any]] = []
        self.action_values: Dict[str, float] = {}
        
    async def mid_game_rest(self,
                           game_states: List[np.ndarray],
                           actions: List[str],
                           scores: List[float],
                           game_type: str) -> ConsolidationResult:
        """Perform mid-game analysis and consolidation.
        
        Args:
            game_states: List of game states so far
            actions: List of actions taken
            scores: List of scores achieved
            game_type: Type of game being played
        """
        # Analyze recent moves
        recent_moves = self.replay.analyze_move_sequence(
            game_states[-10:],  # Look at last 10 moves
            actions[-10:],
            scores[-10:],
            game_type
        )
        
        # Predict remaining moves
        moves_remaining = self.replay.predict_remaining_moves(
            game_states[-1],  # Current state
            game_type,
            len(actions)
        )
        
        # Generate next state predictions
        next_predictions = self.replay.predict_next_states(
            game_states[-1],
            self._get_possible_actions(game_type),
            game_type
        )
        
        # Analyze patterns and formulate hypotheses
        new_patterns = await self._analyze_patterns(
            game_states, actions, scores
        )
        
        # Test existing hypotheses
        confirmed, rejected = await self._test_hypotheses(
            game_states, actions, scores
        )
        
        # Update action values based on recent performance
        updated_values = await self._update_action_values(
            recent_moves, game_type
        )
        
        # Generate movement optimizations
        optimizations = self._optimize_movements(recent_moves)
        
        return ConsolidationResult(
            new_patterns_discovered=new_patterns,
            confirmed_hypotheses=confirmed,
            rejected_hypotheses=rejected,
            updated_action_values=updated_values,
            movement_optimizations=optimizations,
            next_state_predictions=next_predictions
        )
        
    async def post_game_consolidation(self,
                                    game_states: List[np.ndarray],
                                    actions: List[str],
                                    scores: List[float],
                                    final_score: float,
                                    game_type: str) -> ConsolidationResult:
        """Perform post-game analysis and consolidation.
        
        Args:
            game_states: Complete list of game states
            actions: Complete list of actions taken
            scores: Complete list of scores
            final_score: Final game score
            game_type: Type of game played
        """
        # First, get full game analysis
        move_analyses = self.replay.analyze_move_sequence(
            game_states,
            actions,
            scores,
            game_type
        )
        
        # Identify key moments and transitions
        critical_states = self._identify_critical_states(
            game_states, scores
        )
        
        # Deep pattern analysis
        patterns = await self._deep_pattern_analysis(
            game_states, actions, scores
        )
        
        # Analyze action effectiveness
        effective_actions = []
        for move in move_analyses:
            if move.score_change > 0 or move.state_change_magnitude > 0.2:
                effective_actions.append({
                    'action': move.action,
                    'score_change': move.score_change,
                    'state_impact': move.state_change_magnitude
                })
        
        # Generate new hypotheses based on game outcome
        new_hypotheses = []
        
        # Look for successful action sequences
        window_size = 3
        for i in range(len(move_analyses) - window_size):
            sequence = move_analyses[i:i + window_size]
            total_score_change = sum(m.score_change for m in sequence)
            
            if total_score_change > 0:
                new_hypotheses.append({
                    'type': 'action_sequence',
                    'sequence': [m.action for m in sequence],
                    'min_score_gain': total_score_change * 0.8,  # 80% threshold
                    'confidence': 0.1  # Initial confidence
                })
        
        # Look for effective state patterns
        for move in move_analyses:
            if move.score_change > final_score * 0.2:  # Significant score gain
                new_hypotheses.append({
                    'type': 'state_pattern',
                    'pattern': move.before_state,
                    'expected_gain': move.score_change,
                    'confidence': 0.1
                })
                
        self.pending_hypotheses.extend(new_hypotheses)
        
        # Test and update hypotheses
        confirmed, rejected = await self._test_hypotheses(
            game_states, actions, scores
        )
        
        # Update action values based on full game
        updated_values = await self._update_action_values(
            move_analyses, game_type
        )
        
        # Generate movement optimizations
        optimizations = self._optimize_movements(move_analyses)
        
        # Predict optimal states for similar future games
        next_game_predictions = self._predict_optimal_trajectory(
            game_states, scores, game_type
        )
        
        return ConsolidationResult(
            new_patterns_discovered=patterns,
            confirmed_hypotheses=confirmed,
            rejected_hypotheses=rejected,
            updated_action_values=updated_values,
            movement_optimizations=optimizations,
            next_state_predictions=next_game_predictions
        )
    
    async def _analyze_patterns(self,
                             states: List[np.ndarray],
                             actions: List[str],
                             scores: List[float]) -> List[Dict[str, Any]]:
        """Analyze game states for patterns."""
        patterns = []
        
        # Look for repeating sequences
        sequence_length = 3
        for i in range(len(states) - sequence_length):
            sequence = states[i:i + sequence_length]
            
            # Check if sequence led to score increase
            if scores[i + sequence_length] > scores[i]:
                pattern = {
                    'type': 'sequence',
                    'states': sequence,
                    'actions': actions[i:i + sequence_length],
                    'score_gain': scores[i + sequence_length] - scores[i]
                }
                patterns.append(pattern)
        
        return patterns
    
    async def _test_hypotheses(self,
                            states: List[np.ndarray],
                            actions: List[str],
                            scores: List[float]) -> Tuple[List[Dict[str, Any]], 
                                                        List[Dict[str, Any]]]:
        """Test pending hypotheses against new data."""
        confirmed = []
        rejected = []
        
        for hypothesis in self.pending_hypotheses:
            if self._hypothesis_matches(hypothesis, states, actions, scores):
                confirmed.append(hypothesis)
            else:
                rejected.append(hypothesis)
        
        # Remove tested hypotheses
        self.pending_hypotheses = [h for h in self.pending_hypotheses 
                                 if h not in confirmed and h not in rejected]
        
        return confirmed, rejected
    
    async def _update_action_values(self,
                                 moves: List[Any],
                                 game_type: str) -> Dict[str, float]:
        """Update action values based on recent performance."""
        updates = {}
        
        for move in moves:
            action = move.action
            current_value = self.action_values.get(action, 0.0)
            
            # Calculate new value based on immediate and delayed rewards
            immediate_reward = move.score_change
            state_change_value = move.state_change_magnitude * 0.1
            success_prob = move.success_probability
            
            new_value = current_value * 0.9 + (
                immediate_reward + state_change_value
            ) * success_prob
            
            updates[action] = new_value
            self.action_values[action] = new_value
            
        return updates
    
    def _optimize_movements(self, moves: List[Any]) -> List[str]:
        """Generate movement optimization suggestions."""
        optimizations = []
        
        for i in range(len(moves) - 1):
            current = moves[i]
            next_move = moves[i + 1]
            
            # Check for redundant moves
            if (current.state_change_magnitude < 0.1 and 
                current.score_change <= 0):
                optimizations.append(
                    f"Avoid {current.action} when no state change results"
                )
            
            # Check for cancelling moves
            if np.array_equal(current.before_state, next_move.after_state):
                optimizations.append(
                    f"Moves {current.action} and {next_move.action} cancel out"
                )
        
        return optimizations
    
    def _identify_critical_states(self,
                               states: List[np.ndarray],
                               scores: List[float]) -> List[np.ndarray]:
        """Identify states where significant changes occurred."""
        critical = []
        
        for i in range(1, len(states)):
            # Major score changes
            if abs(scores[i] - scores[i-1]) > 5:
                critical.append(states[i])
                
            # Significant state changes
            if np.sum(states[i] != states[i-1]) > states[i].size * 0.3:
                critical.append(states[i])
        
        return critical
    
    async def _deep_pattern_analysis(self,
                                  states: List[np.ndarray],
                                  actions: List[str],
                                  scores: List[float]) -> List[Dict[str, Any]]:
        """Perform deep analysis of game patterns."""
        patterns = []
        
        # Analyze different window sizes
        for window in [2, 3, 4, 5]:
            for i in range(len(states) - window):
                sequence = states[i:i + window]
                sequence_actions = actions[i:i + window]
                sequence_scores = scores[i:i + window]
                
                # Calculate sequence properties
                score_gain = sequence_scores[-1] - sequence_scores[0]
                state_changes = [
                    np.sum(b != a) 
                    for a, b in zip(sequence[:-1], sequence[1:])
                ]
                
                if score_gain > 0 and all(c > 0 for c in state_changes):
                    patterns.append({
                        'type': 'progressive_sequence',
                        'length': window,
                        'actions': sequence_actions,
                        'score_gain': score_gain,
                        'reliability': self._calculate_sequence_reliability(
                            sequence_actions, states, scores
                        )
                    })
        
        return patterns
    
    def _calculate_sequence_reliability(self,
                                    action_sequence: List[str],
                                    all_states: List[np.ndarray],
                                    all_scores: List[float]) -> float:
        """Calculate how reliably a sequence produces positive results."""
        sequence_length = len(action_sequence)
        occurrences = 0
        successful = 0
        
        for i in range(len(all_states) - sequence_length):
            current_actions = [
                self._get_action_type(a) 
                for a in action_sequence
            ]
            
            if current_actions == action_sequence:
                occurrences += 1
                if all_scores[i + sequence_length] > all_scores[i]:
                    successful += 1
                    
        return successful / max(1, occurrences)
    
    def _get_action_type(self, action: str) -> str:
        """Get the general type of an action."""
        # This would classify actions into general types
        # like "move_right", "click", etc.
        return action
    
    def _get_possible_actions(self, game_type: str) -> List[str]:
        """Get list of possible actions for a game type."""
        # This would normally query from game configuration
        # Placeholder implementation
        return ["click", "move_left", "move_right", "move_up", "move_down"]
    
    def _predict_optimal_trajectory(self,
                                game_states: List[np.ndarray],
                                scores: List[float],
                                game_type: str) -> List[Tuple[np.ndarray, float]]:
        """Predict optimal state trajectory for future games."""
        # Find high-scoring state transitions
        good_transitions = []
        
        for i in range(len(game_states) - 1):
            if scores[i + 1] > scores[i]:
                good_transitions.append({
                    'from_state': game_states[i],
                    'to_state': game_states[i + 1],
                    'score_gain': scores[i + 1] - scores[i]
                })
        
        # Sort by score gain
        good_transitions.sort(key=lambda x: x['score_gain'], reverse=True)
        
        # Return top transitions as predictions
        return [(t['to_state'], t['score_gain']) for t in good_transitions[:5]]
    
    def _hypothesis_matches(self,
                         hypothesis: Dict[str, Any],
                         states: List[np.ndarray],
                         actions: List[str],
                         scores: List[float]) -> bool:
        """Check if a hypothesis matches observed data."""
        if hypothesis['type'] == 'action_sequence':
            # Check if action sequence appears with expected outcome
            seq = hypothesis['sequence']
            for i in range(len(actions) - len(seq)):
                if actions[i:i+len(seq)] == seq:
                    actual_score_change = scores[i+len(seq)] - scores[i]
                    if actual_score_change >= hypothesis['min_score_gain']:
                        return True
                        
        elif hypothesis['type'] == 'state_pattern':
            # Check if state pattern appears with expected transformation
            pattern = hypothesis['pattern']
            for state in states:
                if np.array_equal(state, pattern):
                    return True
                    
        return False