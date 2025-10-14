"""Game state replay analysis for understanding patterns and predicting future states."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MoveAnalysis:
    """Analysis of a single move's impact."""
    action: str
    before_state: np.ndarray
    after_state: np.ndarray
    score_change: float
    state_change_magnitude: float
    regions_affected: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    pattern_matched: Optional[str] = None
    success_probability: float = 0.0

@dataclass
class GameReplayInsight:
    """Insights gathered from replaying and analyzing game states."""
    effective_moves: List[MoveAnalysis]
    ineffective_moves: List[MoveAnalysis]
    critical_states: List[np.ndarray]  # States where significant changes occurred
    discovered_patterns: List[Dict[str, Any]]
    estimated_moves_remaining: int
    high_value_regions: List[Tuple[int, int, int, int]]  # Regions that led to score increases
    predicted_next_states: List[np.ndarray]
    success_probability: float

class GameStateReplay:
    """Analyzes game states through replay to understand patterns and predict future states."""
    
    def __init__(self, game_history_db: str):
        """Initialize the replay analyzer.
        
        Args:
            game_history_db: Path to database with game history
        """
        self.db_path = game_history_db
        self.state_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.typical_game_lengths: Dict[str, int] = {}
        
    def analyze_move_sequence(self,
                            states: List[np.ndarray],
                            actions: List[str],
                            scores: List[float],
                            game_type: str) -> List[MoveAnalysis]:
        """Analyze a sequence of moves and their impacts.
        
        Args:
            states: List of game states
            actions: List of actions taken
            scores: List of scores after each action
            game_type: Type of game being analyzed
            
        Returns:
            List of move analysis results
        """
        analyses = []
        
        for i in range(len(actions)):
            before_state = states[i]
            after_state = states[i + 1]
            score_change = scores[i + 1] - scores[i]
            
            # Calculate how much the state changed
            state_change = after_state - before_state
            change_magnitude = np.sum(np.abs(state_change))
            
            # Find regions that were affected
            changes = np.where(state_change != 0)
            if len(changes[0]) > 0:
                min_y, max_y = changes[0].min(), changes[0].max()
                min_x, max_x = changes[1].min(), changes[1].max()
                affected_region = (min_x, min_y, max_x + 1, max_y + 1)
            else:
                affected_region = (0, 0, 0, 0)
            
            # Check if this matches any known patterns
            pattern_match = self._match_state_pattern(
                before_state, after_state, game_type
            )
            
            # Calculate success probability based on historical data
            success_prob = self._calculate_move_success_probability(
                before_state, actions[i], game_type
            )
            
            analysis = MoveAnalysis(
                action=actions[i],
                before_state=before_state,
                after_state=after_state,
                score_change=score_change,
                state_change_magnitude=change_magnitude,
                regions_affected=[affected_region],
                pattern_matched=pattern_match,
                success_probability=success_prob
            )
            analyses.append(analysis)
            
        return analyses
    
    def predict_remaining_moves(self,
                              current_state: np.ndarray,
                              game_type: str,
                              moves_made: int) -> int:
        """Estimate how many moves remain before game over.
        
        Args:
            current_state: Current game state
            game_type: Type of game being analyzed
            moves_made: Number of moves already made
            
        Returns:
            Estimated number of moves remaining
        """
        # Get typical game length for this type
        typical_length = self.typical_game_lengths.get(game_type, 50)
        
        # Adjust based on current state complexity
        state_complexity = np.sum(current_state != 0) / current_state.size
        remaining = max(0, typical_length - moves_made)
        
        # Adjust remaining moves based on state complexity
        if state_complexity > 0.8:  # Very complex state
            remaining = int(remaining * 0.7)  # Less moves likely remaining
        elif state_complexity < 0.2:  # Simple state
            remaining = int(remaining * 1.3)  # More moves likely possible
            
        return remaining
    
    def predict_next_states(self,
                          current_state: np.ndarray,
                          possible_actions: List[str],
                          game_type: str) -> List[Tuple[np.ndarray, float]]:
        """Predict possible next states for each action.
        
        Args:
            current_state: Current game state
            possible_actions: List of possible actions
            game_type: Type of game being analyzed
            
        Returns:
            List of (predicted_state, probability) tuples
        """
        predictions = []
        
        for action in possible_actions:
            # Find similar historical states and their outcomes
            similar_states = self._find_similar_states(
                current_state, game_type
            )
            
            if similar_states:
                # Aggregate the next states that followed this action
                next_states = [s['next_state'] for s in similar_states 
                             if s['action'] == action]
                
                if next_states:
                    # Create probability-weighted prediction
                    predicted = np.mean(next_states, axis=0)
                    confidence = len(next_states) / len(similar_states)
                    predictions.append((predicted, confidence))
                
        return predictions
    
    def _match_state_pattern(self,
                           before: np.ndarray,
                           after: np.ndarray,
                           game_type: str) -> Optional[str]:
        """Match state transition against known patterns."""
        for pattern in self.state_patterns.get(game_type, []):
            if self._states_match_pattern(before, after, pattern):
                return pattern['name']
        return None
    
    def _states_match_pattern(self,
                            before: np.ndarray,
                            after: np.ndarray,
                            pattern: Dict[str, Any]) -> bool:
        """Check if state transition matches a pattern."""
        # Simplified pattern matching
        if 'key_changes' in pattern:
            for change in pattern['key_changes']:
                if not self._verify_state_change(before, after, change):
                    return False
            return True
        return False
    
    def _verify_state_change(self,
                           before: np.ndarray,
                           after: np.ndarray,
                           change: Dict[str, Any]) -> bool:
        """Verify a specific change pattern occurred."""
        if 'region' in change:
            x1, y1, x2, y2 = change['region']
            region_before = before[y1:y2, x1:x2]
            region_after = after[y1:y2, x1:x2]
            
            if 'transformation' in change:
                if change['transformation'] == 'clear':
                    return np.all(region_after == 0)
                elif change['transformation'] == 'fill':
                    return np.all(region_after != 0)
                    
        return False
    
    def _calculate_move_success_probability(self,
                                         state: np.ndarray,
                                         action: str,
                                         game_type: str) -> float:
        """Calculate probability of a move being successful."""
        similar_states = self._find_similar_states(state, game_type)
        if not similar_states:
            return 0.0
            
        # Count successful similar moves
        successful = sum(1 for s in similar_states 
                        if s['action'] == action and s['score_change'] > 0)
        return successful / len(similar_states)
    
    def _find_similar_states(self,
                           state: np.ndarray,
                           game_type: str,
                           threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar states from game history."""
        # This would normally query the database
        # For now return empty list as placeholder
        return []