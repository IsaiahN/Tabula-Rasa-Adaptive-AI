"""Game analysis functionality for mid-game and post-game analysis."""

import logging
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

from src.analysis.consolidation_manager import ConsolidationManager
from src.analysis.game_state_replay import GameStateReplay

logger = logging.getLogger(__name__)

class GameAnalyzer:
    """Handles mid-game and post-game analysis."""

    def __init__(self, consolidation_manager: ConsolidationManager):
        """Initialize game analyzer.
        
        Args:
            consolidation_manager: ConsolidationManager instance for analysis
        """
        self.consolidation_manager = consolidation_manager
        self.current_game_states: List[np.ndarray] = []
        self.current_actions: List[str] = []
        self.current_scores: List[float] = []

    def record_game_state(self,
                         state: np.ndarray,
                         action: str,
                         score: float) -> None:
        """Record a game state for analysis.
        
        Args:
            state: Current game state
            action: Action taken
            score: Score achieved
        """
        self.current_game_states.append(state)
        self.current_actions.append(action)
        self.current_scores.append(score)

    async def do_mid_game_analysis(self,
                                game_type: str,
                                moves_made: int) -> Dict[str, Any]:
        """Perform mid-game analysis and rest period.
        
        Args:
            game_type: Type of game being played
            moves_made: Number of moves made so far
            
        Returns:
            Dict containing mid-game analysis results
        """
        try:
            # Check if we should do mid-game analysis
            if moves_made % 10 != 0:  # Every 10 moves
                return {'message': 'Analysis not needed yet'}
                
            consolidation = await self.consolidation_manager.mid_game_rest(
                self.current_game_states,
                self.current_actions,
                self.current_scores,
                game_type
            )
            
            results = {
                'new_patterns': [],
                'confirmed_hypotheses': [],
                'movement_optimizations': [],
                'top_actions': [],
                'next_state_predictions': 0
            }
            
            # Gather insights
            if consolidation.new_patterns_discovered:
                results['new_patterns'] = [
                    {
                        'type': pattern['type'],
                        'score_gain': pattern.get('score_gain', 0)
                    }
                    for pattern in consolidation.new_patterns_discovered[:3]
                ]
                
            if consolidation.confirmed_hypotheses:
                results['confirmed_hypotheses'] = [
                    {'type': hypothesis['type']}
                    for hypothesis in consolidation.confirmed_hypotheses[:3]
                ]
                
            # Movement optimizations
            if consolidation.movement_optimizations:
                results['movement_optimizations'] = consolidation.movement_optimizations[:3]
                
            # Action values
            if consolidation.updated_action_values:
                results['top_actions'] = [
                    {'action': action, 'value': value}
                    for action, value in sorted(
                        consolidation.updated_action_values.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                ]
            
            # Predictions
            if consolidation.next_state_predictions:
                results['next_state_predictions'] = len(consolidation.next_state_predictions)
                
            return results
                
        except Exception as e:
            logger.error(f"Error during mid-game analysis: {e}")
            return {'error': str(e)}
            
    async def do_post_game_analysis(self,
                                 game_type: str,
                                 final_score: float,
                                 win: bool) -> Dict[str, Any]:
        """Perform post-game analysis and consolidation.
        
        Args:
            game_type: Type of game played
            final_score: Final game score
            win: Whether the game was won
            
        Returns:
            Dict containing post-game analysis results
        """
        try:
            consolidation = await self.consolidation_manager.post_game_consolidation(
                self.current_game_states,
                self.current_actions,
                self.current_scores,
                final_score,
                game_type
            )
            
            results = {
                'patterns': [],
                'confirmed_strategies': [],
                'rejected_strategies': [],
                'action_rankings': [],
                'optimization_tips': [],
                'final_score': final_score,
                'win': win
            }
            
            if consolidation.new_patterns_discovered:
                for pattern in consolidation.new_patterns_discovered[:5]:
                    if pattern['type'] == 'progressive_sequence':
                        results['patterns'].append({
                            'type': 'sequence',
                            'length': pattern['length'],
                            'score_gain': pattern['score_gain'],
                            'reliability': pattern['reliability']
                        })
                    elif pattern['type'] == 'state_pattern':
                        results['patterns'].append({
                            'type': 'state',
                            'expected_gain': pattern['expected_gain']
                        })
                        
            if consolidation.confirmed_hypotheses:
                for hypothesis in consolidation.confirmed_hypotheses[:5]:
                    if hypothesis['type'] == 'action_sequence':
                        results['confirmed_strategies'].append({
                            'sequence': hypothesis['sequence'],
                            'min_score_gain': hypothesis['min_score_gain']
                        })
                        
            if consolidation.rejected_hypotheses:
                results['rejected_strategies'] = [
                    {'type': hypothesis['type']}
                    for hypothesis in consolidation.rejected_hypotheses[:5]
                ]
                    
            if consolidation.updated_action_values:
                results['action_rankings'] = [
                    {'action': action, 'value': value}
                    for action, value in sorted(
                        consolidation.updated_action_values.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
                    
            if consolidation.movement_optimizations:
                results['optimization_tips'] = consolidation.movement_optimizations[:5]
                    
            # Reset game state tracking
            self.current_game_states = []
            self.current_actions = []
            self.current_scores = []
            
            return results
            
        except Exception as e:
            logger.error(f"Error during post-game analysis: {e}")
            return {'error': str(e)}

    def reset(self) -> None:
        """Reset game state tracking."""
        self.current_game_states = []
        self.current_actions = []
        self.current_scores = []